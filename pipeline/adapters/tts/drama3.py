"""Fish Audio Drama 3 served by vLLM-Omni (POST /v1/audio/speech).

Voices are clones of a frozen reference recording. A voice spec is
``ref:<sha16>``: the reference is ``<refs_dir>/<sha16>.wav`` and its
transcript ``<sha16>.txt`` (the cast stage writes both). The first time a
reference is used it is uploaded to the server as voice ``ab-<sha16>``
(POST /v1/audio/voices, kept by the server across restarts), so a request
carries only the voice name, not 1 MB of audio. ``zero:`` (no reference)
lets the model pick a voice from the seed; the cast stage uses it to design
new voices.

Model-specific limits stay here:
- The API takes one reference per request, so a request is one speaker
  (max_speakers = 1) and block_mode stays per utterance.
- The server stops at max_new_tokens codec frames (~21.5 per second; the
  deploy default of 500 is 23 s, too short for long sentences). Each request
  asks for room for a slow reading of its text; output that fills the budget
  is reported as truncated (a runaway generation) instead of being cut off.
- Requests run concurrently (synth: tts.concurrency); the server batches
  them. Measured 2026-11-13: 16 streams ~ 9.9x real time.
"""
from __future__ import annotations

import io
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from .base import SynthRequest, SynthResult

FRAME_SEC = 1 / 21.5            # Drama 3 codec frame (DAC, ~46.4 ms)


class TTSError(RuntimeError):
    pass


class TTSUnavailable(TTSError):
    """Connection / timeout / 5xx after retries: synth stops the run (`service_down`)."""
    service_down = True


class TTSTruncated(TTSError):
    """The output filled max_new_tokens: a runaway generation, or a text far longer than expected.
    synth retries with another seed (`reseed`); `audio` is what came back, for tts.drama3.on_runaway: cut."""
    reseed = True

    def __init__(self, msg: str, audio: np.ndarray | None = None):
        super().__init__(msg)
        self.audio = audio


def spoken_text(text: str) -> str:
    """Drama 3 reads [..] as an inline tag (like [sighs]); bracketed or parenthesised text ran away on every
    seed, e.g. a stage direction "(TO SOMEONE:)" or a bracketed caption (seen in two long books, 2026-11-17). An audiobook
    does not say the brackets, so they are dropped and the words kept. The take key still uses tts_text."""
    out = re.sub(r"[\[\]()]", "", text)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out or text


def ref_sha(spec: str) -> str | None:
    return spec[4:] if spec.startswith("ref:") else None


class Drama3TTS:
    name = "drama3"
    sample_rate = 44100
    max_speakers = 1
    gives_boundaries = False

    def __init__(self, cfg: dict, refs_dir: Path | None = None):
        self.endpoint = cfg.get("endpoint", "http://localhost:8091").rstrip("/")
        self.model = cfg.get("model", "fishaudio/drama-3")
        self.refs_dir = Path(refs_dir) if refs_dir else None
        # A speech request answers only when the whole take is done. Under load (16 streams) a stream runs at
        # about half real time, so a take that fills its budget takes ~2x the budget: the timeout follows the
        # budget (a fixed 300 s timed out on a long take and stopped a whole-book run, 2026-11-16).
        # A stream slows as the server runs more at once (one 128 GB GPU host, real utterances: 16 streams 0.42x real time each,
        # 32 0.32x, 64 0.20x), so a take that fills its budget takes up to ~5x the budget at 64 streams; 3x per
        # frame timed out there and stopped a whole-book run (2026-11-18). The default grows with tts.concurrency.
        self.timeout = float(cfg.get("timeout", 300))            # floor, and for non-speech calls
        streams = max(1, int(cfg.get("concurrency", 1)))
        self.timeout_per_frame = float(cfg.get("timeout_per_frame", max(3.0, streams / 6) * FRAME_SEC))
        self.retries = int(cfg.get("retries", 3))
        self.backoff = float(cfg.get("backoff_seconds", 2.0))
        self.min_cps = float(cfg.get("min_chars_per_sec", 5.0))   # budget: room for reading this slowly
        self.min_frames = int(cfg.get("min_frames", 120))           # ... but never under ~5.6 s
        self.max_frames = int(cfg.get("max_frames", 4000))          # ... nor over ~186 s
        self.format = cfg.get("response_format", "flac")            # lossless, half the bytes of wav
        self._lock = threading.Lock()
        self._voices: set[str] | None = None
        models = self._get("/v1/models")
        ids = [m["id"] for m in models.get("data", [])]
        if self.model not in ids:
            raise TTSError(f"Drama 3 service at {self.endpoint} serves {ids}, not {self.model}")
        self.model_version = cfg.get("model_version") or self.model

    # -- HTTP -----------------------------------------------------------------
    def _request(self, method: str, path: str, body: bytes | None = None, ctype: str | None = None,
                 timeout: float | None = None) -> bytes:
        last, timeouts = None, 0
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.endpoint + path, data=body, method=method,
                                         headers={"Content-Type": ctype} if ctype else {})
            try:
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode("utf-8", "replace")
                if e.code < 500:
                    raise TTSError(f"Drama 3 {method} {path}: HTTP {e.code}: {detail}") from e
                last = f"HTTP {e.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
                    timeouts += 1           # a hung service or tunnel: one more try, not three
                    if timeouts > 1:
                        break
            if attempt < self.retries:
                time.sleep(self.backoff * 2 ** attempt)
        raise TTSUnavailable(f"Drama 3 service at {self.endpoint} not answering ({last}). Is it up, and the "
                             f"SSH tunnel to it open if it runs on another machine? (docs/MODELS.md)")

    def _get(self, path: str) -> dict:
        return json.loads(self._request("GET", path, timeout=30))

    # -- voices ---------------------------------------------------------------
    def _server_voices(self) -> set[str]:
        data = self._get("/v1/audio/voices")
        names = set(data.get("voices", []))
        for v in data.get("uploaded_voices", []):
            if isinstance(v, dict):
                names |= {str(v.get("id", "")), str(v.get("name", ""))}
        return {n.lower() for n in names}

    def ensure_voice(self, sha: str) -> str:
        """Upload reference <sha> once; returns the server's voice name."""
        name = f"ab-{sha}"
        with self._lock:
            if self._voices is None:
                self._voices = self._server_voices()
            if name in self._voices:
                return name
            if self.refs_dir is None:
                raise TTSError(f"voice ref:{sha} needs the book's voices/refs directory")
            wav, txt = self.refs_dir / f"{sha}.wav", self.refs_dir / f"{sha}.txt"
            if not wav.exists() or not txt.exists():
                raise TTSError(f"reference {wav} (or its .txt transcript) is missing; rerun the cast stage")
            fields = {"name": name, "consent": "synthetic-original", "ref_text": txt.read_text().strip()}
            body, ctype = _multipart(fields, "audio_sample", wav.name, wav.read_bytes())
            self._request("POST", "/v1/audio/voices", body, ctype, timeout=120)
            self._voices.add(name)
            return name

    # -- synthesis ------------------------------------------------------------
    def frame_budget(self, text: str) -> int:
        n = math.ceil(len(text) / self.min_cps / FRAME_SEC) + 40
        return max(self.min_frames, min(self.max_frames, n))

    def synthesize(self, req: SynthRequest) -> SynthResult:
        if len(req.utterances) != 1:
            raise ValueError("Drama 3 adapter takes one utterance per request (one reference voice)")
        u = req.utterances[0]
        text = spoken_text(u.tts_text)
        budget = self.frame_budget(text)
        body = {"model": self.model, "input": text, "response_format": self.format,
                "seed": int(req.seed), "max_new_tokens": budget, **(req.params or {})}
        sha = ref_sha(u.voice_ref)
        if sha:
            body["voice"] = self.ensure_voice(sha)
        elif not u.voice_ref.startswith("zero"):
            raise TTSError(f"Drama 3 voice must be ref:<sha> or zero, got {u.voice_ref!r} "
                           f"(a Kokoro voice? run the cast stage with the Drama 3 designer)")
        raw = self._request("POST", "/v1/audio/speech", json.dumps(body).encode(), "application/json",
                            timeout=max(self.timeout, budget * self.timeout_per_frame))
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        audio = np.asarray(audio, np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            raise TTSError(f"Drama 3 returned {sr} Hz, expected {self.sample_rate}")
        if audio.size / sr >= (budget - 3) * FRAME_SEC:
            raise TTSTruncated(f"output filled the {budget}-frame budget ({audio.size / sr:.1f} s) "
                               f"for {len(u.tts_text)} chars: {u.tts_text[:60]!r}", audio=audio)
        return SynthResult(audio)


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.write(f'--{b}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
              f"Content-Type: audio/wav\r\n\r\n".encode())
    out.write(data)
    out.write(f"\r\n--{b}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={b}"
