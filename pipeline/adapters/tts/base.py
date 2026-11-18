from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Utterance:
    tts_text: str
    voice_ref: str            # cast-table voice name
    style: str | None = None  # tone/emotion tag


@dataclass
class SynthRequest:
    utterances: list[Utterance]
    seed: int
    params: dict = field(default_factory=dict)


@dataclass
class SynthResult:
    audio: np.ndarray                                   # float32 mono, -1..1
    boundaries: list[tuple[float, float]] | None = None  # per utterance; None -> aligner


class TTSAdapter(Protocol):
    name: str
    model_version: str
    sample_rate: int
    max_speakers: int
    gives_boundaries: bool

    def synthesize(self, req: SynthRequest) -> SynthResult: ...
