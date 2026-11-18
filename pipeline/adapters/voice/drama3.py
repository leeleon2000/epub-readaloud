"""Drama 3 voice designer (M5): original voices from the model itself.

Drama 3 has no voice-design-from-description mode, but with no reference it
reads in a voice of its own choosing (random: the request seed does not pin
it down when requests are batched, 2026-11-15). A design reads the cast's
fixed sample passage that way for seed, seed + 7919, ... (a few at a time)
and keeps the first candidate, in seed order, that

- sounds like the wanted gender (median pitch: male at most
  cast.male_max_pitch, 115 Hz; female over 170 Hz), or for "neutral" (the default narrator, 2026-11-15: a
  female narrator at 199 Hz sounded too much like the 222 Hz female voice)
  falls in the band between (cast.neutral_pitch, 150-172 Hz),
- reads at a plausible speed (no runaway, no skipped text), and
- is at least cast.min_pitch_gap (15 %) away in pitch from every voice
  already cast.

The candidate is then frozen as the reference: ``voices/refs/<sha16>.wav``
plus its transcript ``.txt`` and a ``.json`` note (seed, pitch). The voice
spec is ``ref:<sha16>``; every utterance of that voice clones the same file
(rule: never use generated book audio as a new reference). No real person's
voice is copied (F5).
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ... import audio
from ..tts import SynthRequest, Utterance

MALE_MAX, FEMALE_MIN = 150.0, 170.0      # Hz, median pitch of the voiced frames
CPS = (8.0, 25.0)                        # plausible characters per second for the sample passage


def pitch_gender(f0: float) -> str:
    if f0 <= 0:
        return ""
    return "male" if f0 < MALE_MAX else "female" if f0 > FEMALE_MIN else "ambiguous"


class Drama3VoiceDesigner:
    name = "drama3"

    def __init__(self, cfg: dict, tts, refs_dir: Path, sample_text: str):
        self.tts = tts
        self.refs_dir = Path(refs_dir)
        self.text = sample_text
        self.max_tries = int(cfg.get("max_tries", 24))
        self.neutral = tuple(float(x) for x in cfg.get("neutral_pitch", (150.0, 172.0)))
        self.min_gap = float(cfg.get("min_pitch_gap", 0.15))   # log pitch ratio; voices closer count as alike
        # A designed male voice must be clearly low: a 128 Hz one sounded like a woman (2026-11-17).
        # pitch_gender() still splits male / female at 150 Hz.
        self.male_max = float(cfg.get("male_max_pitch", 115.0))
        self.parallel = max(1, int(cfg.get("design_parallel", 4)))

    def _zero_shot(self, seed: int) -> np.ndarray:
        return self.tts.synthesize(SynthRequest([Utterance(self.text, "zero")], seed=seed)).audio

    def primaries(self, ref: str) -> list[str]:
        return [ref] if ref and ref.startswith("ref:") else []

    def info(self, ref: str) -> dict:
        p = self.refs_dir / f"{ref[4:]}.json"
        return json.loads(p.read_text()) if ref.startswith("ref:") and p.exists() else {}

    def _miss(self, want: str, f0: float) -> float:
        """How far (Hz) the pitch is from what the wanted voice needs; 0 = fits."""
        if f0 <= 0:
            return 1e9
        if want == "neutral":
            lo, hi = self.neutral
        elif want == "male":
            lo, hi = 0.0, self.male_max
        elif want == "female":
            lo, hi = FEMALE_MIN, 1e9
        else:                                   # either gender, but not the ambiguous band
            return 0.0 if pitch_gender(f0) in ("male", "female") else min(f0 - MALE_MAX, FEMALE_MIN - f0)
        return max(lo - f0, f0 - hi, 0.0)

    def design(self, gender: str, age: str, accent: str, seed: int, avoid: set[str]) -> tuple[str, str]:
        """-> (voice spec, design note). gender: male | female | neutral | "" (either).
        `avoid` = specs already cast."""
        want = gender if gender in ("male", "female", "neutral") else ""
        taken = [t for t in (self.info(r) for r in avoid) if t.get("f0")]
        sr = self.tts.sample_rate
        seeds = [seed + k * 7919 for k in range(self.max_tries)]
        made: dict[int, np.ndarray] = {}
        best = None
        with ThreadPoolExecutor(self.parallel) as ex:
            for k, s in enumerate(seeds):
                if s not in made:            # candidates are made `parallel` at a time, judged in seed order
                    batch = seeds[k:k + self.parallel]
                    made.update(zip(batch, ex.map(self._zero_shot, batch)))
                a = audio.trim(made[s], sr)
                dur = a.size / sr
                f0 = audio.median_f0(a, sr)
                miss = self._miss(want, f0)
                g = "neutral" if want == "neutral" and miss == 0 else pitch_gender(f0)
                cps = len(self.text) / max(dur, 1e-6)
                gap = min((abs(np.log(f0 / t["f0"])) for t in taken), default=9.0) if f0 > 0 else 0.0
                ok_speed = CPS[0] <= cps <= CPS[1]
                if miss == 0 and ok_speed and gap >= self.min_gap:
                    return self._freeze(a, sr, s, f0, g, dur, tries=k + 1)
                score = (ok_speed, -miss, min(gap, self.min_gap))     # the least bad one, in case none passes
                if best is None or score > best[0]:
                    best = (score, a, s, f0, g, dur)
        _, a, s, f0, g, dur = best
        ref, note = self._freeze(a, sr, s, f0, g, dur, tries=self.max_tries)
        return ref, note + f"; NO CANDIDATE PASSED (wanted {want or 'any'}), listen and --reseed"

    def _freeze(self, a: np.ndarray, sr: int, seed: int, f0: float, g: str, dur: float, tries: int):
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.refs_dir / "_candidate.wav"
        audio.write(tmp, a, sr)
        sha = hashlib.sha256(tmp.read_bytes()).hexdigest()[:16]
        tmp.replace(self.refs_dir / f"{sha}.wav")
        (self.refs_dir / f"{sha}.txt").write_text(self.text + "\n")
        meta = {"seed": seed, "f0": round(f0, 1), "gender": g, "duration": round(dur, 2),
                "model": self.tts.model_version, "tries": tries}
        (self.refs_dir / f"{sha}.json").write_text(json.dumps(meta) + "\n")
        return f"ref:{sha}", f"{self.tts.model_version} zero-shot seed {seed}: {g}, pitch {f0:.0f} Hz ({tries} tried)"
