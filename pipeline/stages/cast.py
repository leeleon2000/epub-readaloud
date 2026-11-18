"""Stage: cast (F5). One voice per character, a sample of each, kept in cast.yaml.

cast.yaml is the human-edited source of truth: existing entries are never
changed by this stage (edit them, or use `readaloud cast --reseed/--freeze`); only
characters without an entry get a new design. Every character reads the
same fixed passage so samples can be compared side by side.

Each voice designer has its own cast file and samples, so switching the TTS
model never mixes voice specs (M5, 2026-11-15):
- kokoro-mix (tts kokoro / mock): cast.yaml, voices/<id>.wav.
- drama3 (tts drama3): cast.drama3.yaml, voices/drama3/<id>.wav; the
  references the voices clone are in voices/refs/. The narrator is designed
  too (cast.narrator_gender: neutral by default, a pitch between the male
  and female voices), and with multi_voice off the stage designs only the
  narrator. Changing narrator_gender redesigns a narrator nobody froze or edited.
With attribution v3 only the narrator and the two generic voices are cast
(no voices for leftover v2 character cards).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from .. import audio
from ..adapters.tts import SynthRequest, Utterance, get_tts, settings
from ..adapters.voice.kokoro import KokoroVoiceDesigner
from ..book import Book
from ..memory.db import CharacterDB

SAMPLE_TEXT = ("The lighthouse stood at the very edge of the world, and every evening the keeper climbed "
               "its hundred steps to light the lamp. Nobody had come that way in years, and yet, tonight, "
               "someone was knocking at the door.")
GENERIC = {"unknown": ("unknown", "adult"), "_generic_female": ("female", "adult"),
           "_generic_male": ("male", "adult")}


def designer_name(cfg: dict | None) -> str:
    """kokoro-mix | drama3. cast.designer: auto follows tts.adapter."""
    if cfg is None:
        return "kokoro-mix"
    d = (cfg.get("cast") or {}).get("designer", "auto")
    if d == "auto":
        return "drama3" if cfg["tts"]["adapter"] == "drama3" else "kokoro-mix"
    return d


def needs_reference(cfg: dict) -> bool:
    """The designer clones reference recordings: the narrator must be cast even with multi_voice off."""
    return designer_name(cfg) == "drama3"


def cast_path(book: Book, cfg: dict | None = None) -> Path:
    d = designer_name(cfg)
    return book.dir / ("cast.yaml" if d == "kokoro-mix" else f"cast.{d}.yaml")


def samples_dir(book: Book, cfg: dict | None = None) -> Path:
    d = designer_name(cfg)
    return book.dir / "voices" if d == "kokoro-mix" else book.dir / "voices" / d


def refs_dir(book: Book) -> Path:
    return book.dir / "voices" / "refs"


def load_cast(book: Book, cfg: dict | None = None) -> dict:
    p = cast_path(book, cfg)
    data = yaml.safe_load(p.read_text()) if p.exists() else None
    return data or {"narrator": {}, "characters": {}}


SPEC_HELP = {
    "kokoro-mix": "# Voice spec (Kokoro): [a|b]|voice:weight+voice:weight  (a = American, b = British G2P)\n",
    "drama3": "# Voice spec (Drama 3): ref:<sha16> = clone of voices/refs/<sha16>.wav (designed by the model, frozen).\n"
              "# --reseed designs a new one; frozen: true keeps it. Samples: voices/drama3/<id>.wav\n",
}


def save_cast(book: Book, data: dict, cfg: dict | None = None) -> None:
    cast_path(book, cfg).write_text(
        "# Character -> voice. Edit `voice` to recast; set frozen: true to lock.\n"
        + SPEC_HELP.get(designer_name(cfg), "")
        + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110))


def _seed(base: int, cid: str, bump: int = 0) -> int:
    return base + int(hashlib.md5(cid.encode()).hexdigest()[:6], 16) % 100000 + bump * 7919


def designer_for(cfg: dict, book: Book | None = None, tts=None):
    if designer_name(cfg) == "drama3":
        from ..adapters.voice.drama3 import Drama3VoiceDesigner
        tts = tts or get_tts(cfg["tts"], refs_dir(book))
        return Drama3VoiceDesigner(cfg.get("cast", {}), tts, refs_dir(book), SAMPLE_TEXT)
    return KokoroVoiceDesigner(cfg.get("cast", {}))


def run(book: Book, con, cfg: dict, log=print, doc_ids=None, tts=None) -> None:
    data = load_cast(book, cfg)
    narrator = data.setdefault("narrator", {}) or {}
    data["narrator"] = narrator
    chars = data.setdefault("characters", {}) or {}
    data["characters"] = chars
    designer = designer_for(cfg, book, tts)
    tts = getattr(designer, "tts", tts)
    if designer.name == "kokoro-mix":
        narrator.setdefault("voice", cfg["tts"]["voice"])
    elif not narrator.get("voice") or (not narrator.get("frozen") and narrator.get("basis") not in
                                       (None, _narrator_basis(cfg)) and narrator.get("voice") == narrator.get("auto_voice")):
        g = (cfg.get("cast") or {}).get("narrator_gender", "")
        seed = _seed(cfg["tts"]["seed"], "narrator")
        others = {p for e in chars.values() if e.get("voice") for p in designer.primaries(e["voice"])}
        ref, note = designer.design(g, "adult", "", seed, avoid=others)
        narrator.clear()
        narrator.update({"voice": ref, "seed": seed, "design": note, "frozen": False,
                         "basis": _narrator_basis(cfg), "auto_voice": ref})
        log(f"cast: narrator designed: {ref} ({note})")
    if not cfg.get("multi_voice"):
        save_cast(book, data, cfg)
        _samples(book, con, cfg, data, log, tts, only=["narrator"])
        return
    db = CharacterDB(con)
    known = {r["id"] for r in con.execute("SELECT id FROM character")}
    stale = [k for k, e in chars.items() if k not in known and k not in GENERIC and not e.get("frozen")]
    for k in stale:
        del chars[k]
    if stale:
        log(f"cast: dropped {len(stale)} voices of characters that no longer exist: {', '.join(stale)}")
    taken = {p for e in [narrator, *chars.values()] if e.get("voice") for p in designer.primaries(e["voice"])}

    new = redesigned = 0
    if str((cfg.get("attribute") or {}).get("version", "v2")) == "v3":     # speakers: narrator + two generic voices
        todo = [(g, None) for g in ("_generic_female", "_generic_male")]
    else:
        todo = [(cid, db.card(cid)) for cid in db.active_ids()] + [(g, None) for g in GENERIC]
    for cid, card in todo:
        gender, age = (card.gender, card.age) if card else GENERIC[cid]
        accent = card.accent if card else ""
        basis = f"{gender}/{age}/{accent}"
        e = chars.get(cid)
        if e is not None:
            # Auto-redesign only if the card changed and nobody touched the voice
            # (not frozen, still the voice this stage designed last time).
            if e.get("frozen") or e.get("basis") in (None, basis) or e.get("voice") != e.get("auto_voice"):
                continue
            taken.difference_update(designer.primaries(e["voice"]))
        seed = _seed(cfg["tts"]["seed"], cid)
        avoid = taken if card or designer.name != "kokoro-mix" else set()
        ref, note = designer.design(gender, age, accent, seed, avoid=avoid)
        taken.update(designer.primaries(ref))
        chars[cid] = {"name": card.name if card else cid, "voice": ref, "seed": seed, "design": note,
                      "voice_description": (card.voice_description if card else ""), "frozen": False,
                      "basis": basis, "auto_voice": ref}
        if e is None:
            new += 1
        else:
            redesigned += 1
            log(f"cast: {cid} redesigned ({e.get('basis')} -> {basis}): {ref}")
    save_cast(book, data, cfg)
    _samples(book, con, cfg, data, log, tts)
    log(f"cast: {len(chars)} voices ({new} new, {redesigned} redesigned); listen with "
        f"`readaloud cast {book.id} --play <id>`, edit {cast_path(book, cfg).name} or `readaloud cast {book.id} --freeze <id>`")


def _narrator_basis(cfg: dict) -> str:
    return f"narrator/{(cfg.get('cast') or {}).get('narrator_gender', '')}"


def _samples(book: Book, con, cfg, data, log, tts, only: list[str] | None = None) -> None:
    """Each voice reads SAMPLE_TEXT (Drama 3: the clone of its reference, i.e. what the book will sound like)."""
    entries = [("narrator", data["narrator"])] + list(data["characters"].items())
    if only is not None:
        entries = [(cid, e) for cid, e in entries if cid in only]
    sdir = samples_dir(book, cfg)
    need = [(cid, e) for cid, e in entries
            if e.get("sample_voice") != e["voice"] or not (sdir / f"{cid}.wav").exists()]
    if not need:
        return
    t = settings(cfg["tts"])
    tts = tts or get_tts(cfg["tts"], refs_dir(book))
    for cid, e in need:
        res = tts.synthesize(SynthRequest([Utterance(SAMPLE_TEXT, e["voice"])], seed=t["seed"],
                                          params=t.get("params", {})))
        path = sdir / f"{cid}.wav"
        audio.write(path, res.audio, tts.sample_rate)
        e["sample_voice"] = e["voice"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        con.execute("INSERT OR REPLACE INTO voice(ref, character_id, path, sha, model, design, seed, frozen) "
                    "VALUES (?,?,?,?,?,?,?,?)", (e["voice"], cid, str(path.relative_to(book.dir)), sha,
                                                 tts.model_version, e.get("design", ""), e.get("seed"),
                                                 int(bool(e.get("frozen")))))
    con.commit()
    save_cast(book, data, cfg)
    log(f"cast: {len(need)} samples written to {sdir.relative_to(book.dir)}/")


def voice_for(book_cast: dict, db: CharacterDB, speaker: str) -> str:
    """Voice spec for an utterance speaker id."""
    chars = book_cast.get("characters", {})
    if speaker == "narrator":
        return book_cast["narrator"]["voice"]
    sid = db.resolve(speaker) if speaker != "unknown" else speaker
    if sid in chars:
        return chars[sid]["voice"]
    card = db.card(sid) if sid != "unknown" else None
    g = card.gender if card else ""
    key = "_generic_female" if g == "female" else "_generic_male" if g == "male" else "unknown"
    return chars.get(key, {}).get("voice") or book_cast["narrator"]["voice"]
