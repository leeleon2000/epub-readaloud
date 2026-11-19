"""Hearing the result, and making parts of it again: the three voices' samples, a chapter sentence by sentence,
and the lines whose audio came out wrong.

Nothing here generates audio itself. A voice is redesigned by `readaloud cast --reseed`; a line is made again by
`readaloud regen` (which drops its take) followed by a run of synth, assemble and package for that chapter.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .. import config
from ..book import Book
from ..stages import cast as cast_stage
from . import api, edit
from .api import ApiError

UTTERANCE_ID = re.compile(r"^ab\d{3}s\d{5}u\d+$")
LABEL = {"narrator": "Narrator", "_generic_male": "Male voice", "_generic_female": "Female voice"}


def _book(d: Path) -> Book:
    return Book(d.name, d)


def cfg_for(d: Path, jobs) -> dict:
    """The book's settings with the speech adapter of its last job: a book run with `--tts drama3` keeps its Drama 3
    cast file and samples in view even when config.yaml still says kokoro."""
    cfg = config.load(d)
    args = edit.last_adapters(jobs, d.name)
    if "--tts" in args:
        cfg["tts"]["adapter"] = args[args.index("--tts") + 1]
    return cfg


# ---- the three voices -------------------------------------------------------------------------------------------------

def cast(d: Path, jobs) -> dict:
    book, cfg = _book(d), cfg_for(d, jobs)
    p = cast_stage.cast_path(book, cfg)
    data = cast_stage.load_cast(book, cfg) if p.exists() else {"narrator": {}, "characters": {}}
    sdir = cast_stage.samples_dir(book, cfg)
    entries = ([("narrator", data.get("narrator") or {})] if data.get("narrator") else []) + list((data.get("characters") or {}).items())
    designer = cast_stage.designer_name(cfg)
    voices = []
    for vid, e in entries:
        voices.append({"id": vid, "label": LABEL.get(vid, e.get("name") or vid), "voice": str(e.get("voice", "")),
                       "design": e.get("design", ""), "frozen": bool(e.get("frozen")),
                       "sample": (sdir / f"{vid}.wav").exists(),
                       "can_reseed": not (vid == "narrator" and designer == "kokoro-mix")})
    return {"file": p.name, "exists": p.exists(), "designer": designer, "adapter": cfg["tts"]["adapter"], "voices": voices,
            "sample_text": cast_stage.SAMPLE_TEXT}


def _voice_id(d: Path, jobs, vid: str) -> str:
    if vid not in {v["id"] for v in cast(d, jobs)["voices"]}:
        raise ApiError("no such voice", 404)
    return vid


def cast_sample(d: Path, jobs, vid: str) -> Path:
    p = cast_stage.samples_dir(_book(d), cfg_for(d, jobs)) / f"{_voice_id(d, jobs, vid)}.wav"
    if not p.is_file():
        raise ApiError("this voice has no sample yet: run the cast stage", 404)
    return p


def cast_action(root: Path, d: Path, jobs, vid: str, action: str) -> dict:
    if action not in ("reseed", "freeze", "unfreeze"):
        raise ApiError("action must be reseed, freeze or unfreeze")
    _voice_id(d, jobs, vid)
    edit._idle(d, jobs)
    rc, out = api.cli(root, "cast", d.name, f"--{action}", vid, "--tts", cfg_for(d, jobs)["tts"]["adapter"])
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "could not change the voice")
    return {**cast(d, jobs), "message": out.splitlines()[-1] if action == "reseed" and out else ""}


# ---- a chapter, sentence by sentence -------------------------------------------------------------------------------------

def _flagged(d: Path) -> dict[str, str]:
    rep = d / "reports" / "synth_flagged.tsv"
    out = {}
    if rep.exists():
        for line in rep.read_text().splitlines()[1:]:
            parts = line.split("\t")
            if parts and parts[0]:
                out[parts[0]] = parts[2] if len(parts) > 2 else ""
    return out


def chapter_file(d: Path, n: int, suffix: str) -> Path:
    return _book(d).chapters / f"{int(n):03d}{suffix}"


def chapter(d: Path, n: int) -> dict:
    con = edit._need_db(d)
    try:
        if con.execute("SELECT 1 FROM doc WHERE id=? AND has_speech=1", (n,)).fetchone() is None:
            raise ApiError("no such chapter", 404)
        tl_path, wav = chapter_file(d, n, ".json"), chapter_file(d, n, ".wav")
        times, ms = {}, 0
        if tl_path.exists() and wav.exists():
            tl = json.loads(tl_path.read_text())
            times = {sid: (int(a), int(b)) for sid, a, b in tl["clips"]}
            ms = int(tl.get("file_ms") or 0)
        flagged = _flagged(d)
        pieces: dict[str, list] = {}
        try:
            rows = con.execute(
                "SELECT u.id, u.sentence_id, u.tts_text, u.is_quote, u.speaker_id, t.seed, t.duration, "
                "sel.take_id IS NOT NULL made FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                "LEFT JOIN selection sel ON sel.block_id=u.id LEFT JOIN take t ON t.id=sel.take_id "
                "WHERE s.doc_id=? ORDER BY s.seq, u.seq", (n,)).fetchall()
        except sqlite3.Error:
            rows = []
        for u in rows:
            pieces.setdefault(u["sentence_id"], []).append({
                "id": u["id"], "text": u["tts_text"], "quote": bool(u["is_quote"]),
                "voice": edit.SPEAKER_VOICE.get(u["speaker_id"], "narrator"), "made": bool(u["made"]),
                "flagged": flagged.get(u["id"])})
        sentences = []
        for s in con.execute("SELECT id, kind, text, speak FROM sentence WHERE doc_id=? ORDER BY seq", (n,)):
            if not s["speak"]:
                continue
            a, b = times.get(s["id"], (None, None))
            sentences.append({"id": s["id"], "kind": s["kind"], "text": s["text"], "start": a, "end": b,
                              "pieces": pieces.get(s["id"], [])})
        return {"chapter": n, "audio": bool(times), "ms": ms, "sentences": sentences,
                "flagged": sum(1 for s in sentences for p in s["pieces"] if p["flagged"] is not None)}
    finally:
        con.close()


def chapter_audio(d: Path, n: int) -> Path:
    p = chapter_file(d, n, ".wav")
    if not p.is_file():
        raise ApiError("this chapter has no audio yet", 404)
    return p


def flagged(d: Path) -> dict:
    """Lines whose audio was cut short because the speech model ran away on every seed (reports/synth_flagged.tsv)."""
    fl = _flagged(d)
    rows = []
    if fl:
        con = edit._need_db(d)
        try:
            for uid, why in fl.items():
                r = con.execute("SELECT u.tts_text, s.doc_id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                                "WHERE u.id=?", (uid,)).fetchone()
                rows.append({"id": uid, "why": why, "text": r["tts_text"] if r else "", "chapter": r["doc_id"] if r else None})
        finally:
            con.close()
    return {"flagged": rows}


def redo(root: Path, d: Path, jobs, utterances: list, next_seed: bool) -> dict:
    """`readaloud regen`: drop the takes of these pieces, so that the next synth makes them again. Returns the chapters
    they are in, for the job that makes them."""
    ids = [str(u) for u in utterances or []]
    if not ids or any(not UTTERANCE_ID.match(u) for u in ids):
        raise ApiError("give the ids of the pieces to make again")
    con = edit._need_db(d)
    try:
        q = ",".join("?" * len(ids))
        found = {r["id"]: r["doc_id"] for r in con.execute(
            f"SELECT u.id, s.doc_id FROM utterance u JOIN sentence s ON s.id=u.sentence_id WHERE u.id IN ({q})", ids)}
    finally:
        con.close()
    missing = [u for u in ids if u not in found]
    if missing:
        raise ApiError(f"no such piece: {missing[0]}", 404)
    edit._idle(d, jobs)
    rc, out = api.cli(root, "regen", d.name, *ids, *(["--next-seed"] if next_seed else []))
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "could not drop the takes")
    return {"message": out.splitlines()[0] if out else "", "chapters": sorted(set(found.values()))}


def redo_args(book_id: str, jobs, chapters: list[int]) -> list[str]:
    return ["run", api.check_id(book_id), *edit.last_adapters(jobs, book_id), "--stages", "synth,assemble,package",
            "--docs", ",".join(str(int(c)) for c in chapters)]
