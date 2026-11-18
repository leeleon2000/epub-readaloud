#!/usr/bin/env python3
"""Minimal Kokoro TTS service (stdlib HTTP, one request at a time).

    .venv/bin/python services/kokoro/server.py [--port 9001] [--device cpu|mps|cuda]

GET  /info        -> {"model_version", "sample_rate"}
POST /synthesize  {"text", "voice", "speed"} -> audio/wav (16-bit mono)

voice is a Kokoro voice name ("af_heart") or a weighted mix with an optional
G2P language prefix: "b|bf_emma:0.7+bf_lily:0.3" (a = American, b = British;
default: the first letter of the first voice).
"""
from __future__ import annotations

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

SR = 24000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import kokoro
    from kokoro import KModel, KPipeline
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
    if args.device:
        model = model.to(args.device)
    pipes = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model) for lang in ("a", "b")}
    # Model version is independent of the G2P language; the language is part of each voice spec.
    version = f"kokoro-{getattr(kokoro, '__version__', '?')}-82M"
    lock = threading.Lock()

    def resolve_voice(spec: str):
        """-> (pipeline, voice key registered in that pipeline)."""
        lang, _, mix = spec.partition("|") if "|" in spec else ("", "", spec)
        parts = []
        for item in mix.split("+"):
            name, _, w = item.partition(":")
            parts.append((name.strip(), float(w) if w else 1.0))
        lang = lang or parts[0][0][0]
        pipe = pipes["b" if lang == "b" else "a"]
        if len(parts) == 1:
            return pipe, parts[0][0]
        if spec not in pipe.voices:
            total = sum(w for _, w in parts)
            pipe.voices[spec] = sum(pipe.load_single_voice(n) * (w / total) for n, w in parts)
        return pipe, spec

    class H(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/info":
                self._json(200, {"model_version": version, "sample_rate": SR})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/synthesize":
                return self._json(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                with lock:
                    pipe, voice = resolve_voice(req.get("voice", "af_heart"))
                    parts = []
                    for r in pipe(req["text"], voice=voice, speed=float(req.get("speed", 1.0))):
                        a = r.audio
                        if a is None:
                            continue
                        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
                        parts.append(a.astype(np.float32).reshape(-1))
                if not parts:
                    return self._json(422, {"error": "no audio produced"})
                buf = io.BytesIO()
                sf.write(buf, np.concatenate(parts), SR, format="WAV", subtype="PCM_16")
                data = buf.getvalue()
            except Exception as e:  # report to client instead of dropping the connection
                return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *a):
            pass

    print(f"Kokoro service {version} on http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
