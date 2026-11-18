"""Drama 3 throughput on real book utterances, at several concurrencies (2026-11-17).

drama3_bench.py sends one long passage in the default voice; the pipeline sends many short utterances in a
cloned voice, where the reference prompt is a large part of each request. This sends the same N utterances of
a book (reading order, story text) at each concurrency, in an uploaded voice, as the adapter would, and reports
audio seconds produced per wall second.

Run on the GPU host from the repository root while Drama 3 is up and nothing else uses it:
  PYTHONPATH=$PWD .venv/bin/python scripts/drama3_pipeline_bench.py <book_id> 16 32 [--n 160]
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from pipeline.adapters.tts.drama3 import Drama3TTS

URL = "http://localhost:8091"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("conc", nargs="+", type=int)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--skip", type=int, default=500, help="utterances to skip (front matter)")
    a = ap.parse_args()
    con = sqlite3.connect(Path("books") / a.book / "project.db")
    texts = [r[0] for r in con.execute(
        "SELECT u.tts_text FROM utterance u JOIN sentence s ON s.id=u.sentence_id WHERE s.speak=1 "
        "ORDER BY s.doc_id, s.seq, u.seq LIMIT ? OFFSET ?", (a.n, a.skip))]
    voices = json.load(urllib.request.urlopen(URL + "/v1/audio/voices"))
    voice = sorted(v["id"] for v in voices.get("uploaded_voices", []) if v["id"].startswith("ab-"))[0]
    tts = Drama3TTS({"endpoint": URL})
    print(f"{len(texts)} utterances of {a.book} ({sum(map(len, texts))} chars), voice {voice}", flush=True)

    def one(text: str) -> float:
        body = {"model": tts.model, "input": text, "response_format": "flac", "seed": 1234,
                "max_new_tokens": tts.frame_budget(text), "voice": voice}
        req = urllib.request.Request(URL + "/v1/audio/speech", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return sf.info(io.BytesIO(r.read())).duration

    one(texts[0])                                                   # warm-up
    for c in a.conc:
        t0 = time.time()
        with ThreadPoolExecutor(c) as ex:
            audio = sum(ex.map(one, texts))
        wall = time.time() - t0
        print(f"concurrency {c:3d}: {audio:7.1f} s audio in {wall:6.1f} s = {audio / wall:5.2f}x real time", flush=True)


if __name__ == "__main__":
    main()
