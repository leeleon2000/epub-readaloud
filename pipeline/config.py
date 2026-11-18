from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "config.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load(book_dir: Path | None = None, path: Path | None = None) -> dict:
    cfg = yaml.safe_load((path or DEFAULT_CONFIG).read_text()) or {}
    if book_dir is not None and (book_dir / "config.yaml").exists():
        cfg = _merge(cfg, yaml.safe_load((book_dir / "config.yaml").read_text()) or {})
    if cfg.get("block_mode", "sentence") != "sentence":
        raise ValueError("block_mode other than 'sentence' arrives in M4")
    return cfg
