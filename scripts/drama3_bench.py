"""Concurrency benchmark for Drama 3 on vLLM-Omni.

Run on the GPU host:  python scripts/drama3_bench.py [concurrency ...]
Reports per-request RTF (gen time / audio length) and aggregate throughput
(audio seconds produced per wall-clock second).
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import soundfile as sf

URL = "http://localhost:8091/v1/audio/speech"
TEXT = (
    "The rain had not stopped for three days. Margaret stood at the window, "
    "watching the water run down the glass, and wondered whether he would "
    "come back at all, or whether the letter had been the last she would hear of him."
)


def one(_):
    t0 = time.time()
    r = requests.post(URL, json={"model": "fishaudio/drama-3", "input": TEXT,
                                 "response_format": "wav"}, timeout=600)
    r.raise_for_status()
    dt = time.time() - t0
    info = sf.info(io.BytesIO(r.content))
    return dt, info.duration


def run(n):
    t0 = time.time()
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(one, range(n)))
    wall = time.time() - t0
    audio = sum(d for _, d in res)
    rtfs = [g / d for g, d in res]
    print(f"conc={n:3d}  wall={wall:6.1f}s  audio={audio:7.1f}s  "
          f"throughput={audio / wall:5.2f}x realtime  "
          f"per-req RTF avg={sum(rtfs) / n:.2f} max={max(rtfs):.2f}", flush=True)


if __name__ == "__main__":
    run(1)  # warm-up
    for n in map(int, sys.argv[1:] or ["1", "4", "8", "16"]):
        run(n)
