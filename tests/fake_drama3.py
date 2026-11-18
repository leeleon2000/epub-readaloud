"""A stand-in for the vLLM-Omni Drama 3 speech API, for tests (no GPU, no network).

Speech is a buzz (a pitch plus harmonics) of 60 ms per character, so the
pitch-based designer can be tested: with no reference the pitch comes from the
seed (seed % 3: 0 -> male 105-135 Hz, 1 -> female 195-245 Hz, 2 -> neutral 150-175 Hz); with an
uploaded voice it is that voice's pitch. "RUNAWAY" in the text fills the
whole frame budget, and so does the first request for a text in `runaway_once`.
`fail_next` makes the next requests answer 503.
"""
from __future__ import annotations

import io
import json
import threading
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

from pipeline import audio

SR = 44100
FRAME = 1 / 21.5


class FakeDrama3:
    def __init__(self):
        self.voices: dict[str, dict] = {}       # name -> {"ref_text", "f0"}
        self.requests: list[dict] = []
        self.uploads = 0
        self.fail_next = 0
        self.runaway_once: set[str] = set()     # texts whose first request runs away
        self.lock = threading.Lock()
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._send(200, json.dumps({"data": [{"id": "fishaudio/drama-3"}]}).encode())
                if self.path == "/v1/audio/voices":
                    up = [{"id": n, "name": n, "ref_text": v["ref_text"]} for n, v in fake.voices.items()]
                    return self._send(200, json.dumps({"voices": sorted(fake.voices), "uploaded_voices": up}).encode())
                self._send(404, b"{}")

            def do_POST(self):
                with fake.lock:
                    if fake.fail_next:
                        fake.fail_next -= 1
                        return self._send(503, b'{"error": "busy"}')
                n = int(self.headers["Content-Length"])
                raw = self.rfile.read(n)
                if self.path == "/v1/audio/voices":
                    msg = BytesParser(policy=email_policy).parsebytes(
                        b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + raw)
                    form = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                            for part in msg.iter_parts()}
                    a, sr = sf.read(io.BytesIO(form["audio_sample"]), dtype="float32")
                    name = form["name"].decode().lower()
                    with fake.lock:
                        fake.voices[name] = {"ref_text": form["ref_text"].decode(), "f0": audio.median_f0(a, sr)}
                        fake.uploads += 1
                    return self._send(200, json.dumps({"success": True, "voice": {"id": name}}).encode())
                if self.path == "/v1/audio/speech":
                    body = json.loads(raw)
                    with fake.lock:
                        fake.requests.append(body)
                    v = body.get("voice")
                    if v and v.lower() not in fake.voices:
                        return self._send(400, b'{"error": "Unknown voice"}')
                    seed = int(body.get("seed") or 0)
                    if v:
                        f0 = fake.voices[v.lower()]["f0"]
                    else:
                        f0 = {0: 105 + seed % 31, 1: 195 + seed % 51, 2: 150 + seed % 26}[seed % 3]
                    text = body["input"]
                    dur = 0.06 * len(text)
                    cap = body.get("max_new_tokens", 500) * FRAME
                    with fake.lock:
                        once = text in fake.runaway_once
                        fake.runaway_once.discard(text)
                    if "RUNAWAY" in text or once or dur > cap:
                        dur = cap
                    t = np.arange(int(dur * SR)) / SR
                    x = sum(0.2 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6)).astype(np.float32)
                    buf = io.BytesIO()
                    sf.write(buf, x, SR, format=body.get("response_format", "wav").upper())
                    return self._send(200, buf.getvalue(), "audio/" + body.get("response_format", "wav"))
                self._send(404, b"{}")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
