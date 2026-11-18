"""Deterministic fake TTS: one short tone per word. For tests and Mac dev."""
from __future__ import annotations

import hashlib

import numpy as np

from .base import SynthRequest, SynthResult


class MockTTS:
    name = "mock"
    model_version = "mock-1"
    sample_rate = 24000
    max_speakers = 1
    gives_boundaries = False

    def __init__(self, cfg: dict | None = None):
        pass

    def synthesize(self, req: SynthRequest) -> SynthResult:
        sr = self.sample_rate
        parts = [np.zeros(int(0.05 * sr), np.float32)]   # leading silence to exercise trimming
        for u in req.utterances:
            for w in u.tts_text.split():
                h = int(hashlib.md5(f"{w}{req.seed}{u.voice_ref}".encode()).hexdigest()[:4], 16)
                n = int((0.12 + 0.02 * min(len(w), 10)) * sr)
                t = np.arange(n) / sr
                env = np.minimum(1, np.minimum(t, t[::-1]) / 0.005)
                parts.append((0.3 * np.sin(2 * np.pi * (300 + h % 400) * t) * env).astype(np.float32))
                parts.append(np.zeros(int(0.06 * sr), np.float32))
        parts.append(np.zeros(int(0.1 * sr), np.float32))
        return SynthResult(np.concatenate(parts))
