from __future__ import annotations

from pathlib import Path

from .base import SynthRequest, SynthResult, TTSAdapter, Utterance  # noqa: F401


def settings(t: dict) -> dict:
    """The tts section for the selected adapter: keys under tts.<adapter> (e.g. tts.drama3)
    override the shared ones, so `--tts drama3` switches endpoint, concurrency and params too."""
    sub = t.get(t["adapter"])
    return {**t, **sub} if isinstance(sub, dict) else dict(t)


def get_tts(cfg: dict, refs_dir: Path | None = None) -> "TTSAdapter":
    """cfg = the config's tts section. refs_dir = the book's voices/refs (reference voices)."""
    cfg = settings(cfg)
    name = cfg["adapter"]
    if name == "mock":
        from .mock import MockTTS
        return MockTTS(cfg)
    if name == "kokoro":
        from .kokoro import KokoroTTS
        return KokoroTTS(cfg)
    if name == "drama3":
        from .drama3 import Drama3TTS
        return Drama3TTS(cfg, refs_dir)
    raise ValueError(f"unknown TTS adapter: {name}")
