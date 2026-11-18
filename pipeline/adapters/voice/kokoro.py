"""Kokoro voice designer (M4 stand-in for a real voice-design model).

Kokoro cannot clone from reference audio. English has 28 fixed voices, which
can be mixed by weight. A design picks two voices of the character's gender,
preferring their accent, and a mix ratio, all from the seed. The voice_ref is
the Kokoro voice spec, e.g. "b|bf_emma:0.70+bf_lily:0.30".
"""
from __future__ import annotations

import random

VOICES = {
    ("female", "us"): ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova", "af_alloy",
                       "af_aoede", "af_jessica", "af_kore", "af_river"],
    ("male", "us"): ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
                     "am_puck", "am_santa"],
    ("female", "gb"): ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    ("male", "gb"): ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}
# Lighter / younger-sounding voices first for child and teen characters.
YOUNG = {"af_sky", "af_nova", "af_jessica", "bf_lily", "bf_alice", "am_puck", "am_echo", "bm_lewis"}
OLD = {"am_santa", "am_onyx", "bm_george", "bm_daniel", "af_river", "bf_isabella"}


def _accent(card_accent: str, default: str) -> str:
    a = (card_accent or default or "").lower()
    return "gb" if any(k in a for k in ("brit", "english", "england", "scot", "irish", "gb", "uk")) else "us"


class KokoroVoiceDesigner:
    name = "kokoro-mix"

    def __init__(self, cfg: dict):
        self.default_accent = cfg.get("default_accent", "")

    def primaries(self, ref: str) -> list[str]:
        mix = ref.split("|", 1)[-1]
        return [p.split(":")[0] for p in mix.split("+")]

    def design(self, gender: str, age: str, accent: str, seed: int, avoid: set[str]) -> tuple[str, str]:
        """-> (voice_ref, human-readable design note). `avoid` = primary voices already taken."""
        rng = random.Random(seed)
        g = gender if gender in ("female", "male") else rng.choice(["female", "male"])
        acc = _accent(accent, self.default_accent)
        pool = list(VOICES[(g, acc)])
        other = VOICES[(g, "us" if acc == "gb" else "gb")]
        age = (age or "").lower()
        pref = YOUNG if age in ("child", "teen") else OLD if age in ("elderly", "middle-aged") else set()
        free = [v for v in pool if v not in avoid] or [v for v in other if v not in avoid] or pool
        ranked = sorted(free, key=lambda v: (v not in pref, rng.random()))
        primary = ranked[0]
        second_pool = [v for v in pool + other if v != primary]
        second = sorted(second_pool, key=lambda v: (v not in pref, rng.random()))[0]
        w = round(rng.uniform(0.6, 0.8), 2)
        lang = "b" if acc == "gb" else "a"
        ref = f"{lang}|{primary}:{w:.2f}+{second}:{1 - w:.2f}"
        return ref, f"{g}, {age or 'unknown age'}, {'British' if acc == 'gb' else 'American'}: {primary} {w:.0%} + {second}"
