"""Kokoro via the local HTTP service in services/kokoro/server.py."""
from __future__ import annotations

import io
import json
import urllib.request

import numpy as np
import soundfile as sf

from .base import SynthRequest, SynthResult


class KokoroTTS:
    name = "kokoro"
    sample_rate = 24000
    max_speakers = 1
    gives_boundaries = False

    def __init__(self, cfg: dict):
        self.endpoint = cfg.get("endpoint", "http://localhost:9001").rstrip("/")
        info = self._get("/info")
        self.model_version = info["model_version"]
        self.sample_rate = info["sample_rate"]

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.endpoint + path, timeout=10) as r:
                return json.load(r)
        except OSError as e:
            raise RuntimeError(f"Kokoro service not reachable at {self.endpoint} "
                               f"(start it: .venv/bin/python services/kokoro/server.py): {e}") from e

    def synthesize(self, req: SynthRequest) -> SynthResult:
        if len(req.utterances) != 1:
            raise ValueError("Kokoro adapter takes one utterance per request")
        u = req.utterances[0]
        body = json.dumps({"text": u.tts_text, "voice": u.voice_ref,
                           "speed": req.params.get("speed", 1.0)}).encode()
        r = urllib.request.Request(self.endpoint + "/synthesize", data=body,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=300) as resp:
            audio, sr = sf.read(io.BytesIO(resp.read()), dtype="float32")
        if sr != self.sample_rate:
            raise RuntimeError(f"Kokoro returned {sr} Hz, expected {self.sample_rate}")
        return SynthResult(np.asarray(audio, np.float32).reshape(-1))
