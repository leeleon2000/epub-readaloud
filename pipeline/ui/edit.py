"""Looking at what the pipeline decided, and changing it: warnings, section categories, lines that are not read,
the voice of each quotation, the gender table, and a book's settings.

The files a person may edit stay the source of truth (sections.yaml, attribution_overrides.csv, genders.yaml,
config.yaml): the UI writes those files, and the pipeline reads them on its next run exactly as when they are edited
by hand. Changes that the command line makes through the database (a section's category, a warning's state) go
through the command line.
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path

import yaml

from .. import config
from ..book import Book
from ..stages import classify
from ..v3 import genders as G
from ..v3.tags import norm_phrase
from . import api
from .api import ApiError

VOICES = ("male", "female", "narrator")
SPEAKER_VOICE = {"_generic_male": "male", "_generic_female": "female"}
CSV_FIELDS = ["quote_id", "speaker", "style", "note"]
SECTION_ID = re.compile(r"^d\d+s\d+$")
WARNING_HELP = {
    "SECTION_UNCERTAIN": ("The pipeline could not tell whether a section is story.",
                          "Give it a category on the Sections tab, then run again."),
    "LLM_FAILED": ("One request to the language model failed or was cut off. It is retried at the end of the run.",
                   "Run again; nothing else is lost."),
    "LLM_DOWN": ("Several requests in a row failed.",
                 "Check that the language model server answers (Models page), then run again."),
    "TRUNCATED": ("A model answer hit the length limit.",
                  "Usually handled by splitting the request. If it repeats, raise llm.max_tokens in Settings."),
    "REPEATED": ("The same warning keeps coming back.", "Something systematic: read the earlier warnings."),
}
CATEGORY_HELP = {
    "story": "The book itself. Searched for dialogue and read aloud.",
    "front_matter": "Title page, dedication, preface. Read by the narrator.",
    "notes": "Notes. Read by the narrator.",
    "back_matter": "Index, appendix, afterword. Read by the narrator.",
    "toc": "Contents. Not read.",
    "captions": "Captions and lists of illustrations. Not read.",
    "boilerplate": "Licence and copyright text. Not read.",
}


def _book(d: Path) -> Book:
    return Book(d.name, d)


def _need_db(d: Path) -> sqlite3.Connection:
    con = api.db(d)
    if con is None:
        raise ApiError("run the book first", 409)
    return con


def _idle(d: Path, jobs) -> None:
    if jobs.running(d.name):
        raise ApiError("a job is running for this book; make this change when it has finished", 409)


# ---- warnings ----------------------------------------------------------------------------------------------------------

def warnings(d: Path, include_handled: bool = False) -> dict:
    con = _need_db(d)
    try:
        where = "" if include_handled else "WHERE status='open'"
        rows = []
        for w in con.execute(f"SELECT * FROM warning {where} ORDER BY status='open' DESC, id DESC"):
            det = json.loads(w["details"] or "{}")
            what, todo = WARNING_HELP.get(w["code"], ("", ""))
            rows.append({"id": w["id"], "code": w["code"], "severity": w["severity"], "status": w["status"],
                         "where": w["chunk_id"] or "", "count": w["count"] or 1, "message": det.get("message", ""),
                         "opening": det.get("opening", ""), "created": w["created"], "note": w["note"] or "",
                         "handled_by": w["handled_by"] or "", "what": what, "todo": todo,
                         "section": w["chunk_id"] if w["chunk_id"] and SECTION_ID.match(w["chunk_id"]) else None})
        n_open = con.execute("SELECT COUNT(*) FROM warning WHERE status='open'").fetchone()[0]
        return {"warnings": rows, "open": n_open}
    finally:
        con.close()


def warning_action(root: Path, d: Path, jobs, wid: int, action: str, note: str) -> dict:
    if action not in ("ack", "resolve"):
        raise ApiError("action must be ack or resolve")
    _idle(d, jobs)
    rc, out = api.cli(root, "warnings", d.name, f"--{action}", str(int(wid)), "--note", note or "", "--by", "ui")
    if rc or f"no warning {int(wid)}" in out:
        raise ApiError(out.splitlines()[0] if out else "could not change the warning", 404 if "no warning" in out else 400)
    return warnings(d)


# ---- sections and lines that are not read -------------------------------------------------------------------------------

def sections(d: Path) -> dict:
    con = _need_db(d)
    try:
        rows = []
        for r in con.execute(
                "SELECT s.*, (SELECT text FROM sentence x WHERE x.doc_id=s.doc_id AND x.seq=s.seq_from) first, "
                "(SELECT COUNT(*) FROM sentence x WHERE x.section_id=s.id AND x.speak=1) spoken "
                "FROM section s ORDER BY doc_id, seq_from"):
            rows.append({"id": r["id"], "chapter": r["doc_id"], "title": r["title"] or "", "category": r["category"],
                         "source": r["source"], "confidence": r["confidence"], "reason": r["reason"] or "",
                         "first": (r["first"] or "")[:140], "sentences": r["seq_to"] - r["seq_from"] + 1,
                         "spoken": r["spoken"]})
        cats = [c for c in CATEGORY_HELP if c in classify.CATEGORIES]
        return {"sections": rows, "categories": [{"id": c, "help": CATEGORY_HELP[c], "read": c in classify.SPOKEN}
                                                 for c in cats], "skip": classify.load_skip(_book(d))}
    finally:
        con.close()


def set_sections(root: Path, d: Path, jobs, assign: dict) -> dict:
    if not assign or not isinstance(assign, dict):
        raise ApiError("nothing to set")
    for sid, cat in assign.items():
        if not SECTION_ID.match(str(sid)):
            raise ApiError(f"not a section id: {sid}")
        if cat not in CATEGORY_HELP:
            raise ApiError(f"unknown category: {cat}")
    _idle(d, jobs)
    rc, out = api.cli(root, "sections", d.name, "--set", *[f"{k}={v}" for k, v in assign.items()])
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "could not set the sections")
    return sections(d)


def section_sentences(d: Path, sid: str) -> dict:
    if not SECTION_ID.match(sid):
        raise ApiError("not a section id")
    con = _need_db(d)
    try:
        s = con.execute("SELECT * FROM section WHERE id=?", (sid,)).fetchone()
        if s is None:
            raise ApiError("no such section", 404)
        skip = set(classify.load_skip(_book(d)))
        rows = [{"id": r["id"], "kind": r["kind"], "text": r["text"], "read": bool(r["speak"]), "skipped": r["id"] in skip}
                for r in con.execute("SELECT id, kind, text, speak FROM sentence WHERE doc_id=? AND seq BETWEEN ? AND ? "
                                     "ORDER BY seq", (s["doc_id"], s["seq_from"], s["seq_to"]))]
        return {"section": sid, "category": s["category"], "sentences": rows}
    finally:
        con.close()


def set_skip(root: Path, d: Path, jobs, add: list, remove: list) -> dict:
    """`skip:` in sections.yaml, then the classify stage again so the change shows (no model call for decided sections)."""
    add, remove = [str(x) for x in add or []], [str(x) for x in remove or []]
    if not add and not remove:
        raise ApiError("nothing to change")
    con = _need_db(d)
    try:
        for sid in add:
            if con.execute("SELECT 1 FROM sentence WHERE id=?", (sid,)).fetchone() is None:
                raise ApiError(f"no sentence {sid}", 404)
    finally:
        con.close()
    _idle(d, jobs)
    book = _book(d)
    skip = [x for x in classify.load_skip(book) if x not in remove] + add
    classify.save_skip(book, skip)
    rc, out = api.cli(root, "run", d.name, "--stages", "classify")
    if rc not in (0, api_paused()):
        raise ApiError(out.splitlines()[-1] if out else "the classify stage failed")
    return {"skip": classify.load_skip(book)}


def api_paused() -> int:
    from ..cli import EXIT_PAUSED
    return EXIT_PAUSED


# ---- which voice reads each quotation ------------------------------------------------------------------------------------

def overrides_path(d: Path) -> Path:
    return d / "attribution_overrides.csv"


def read_overrides(d: Path) -> list[dict]:
    p = overrides_path(d)
    if not p.exists():
        return []
    with p.open(newline="") as f:
        return [{k: (r.get(k) or "") for k in CSV_FIELDS} for r in csv.DictReader(f) if (r.get("quote_id") or "").strip()]


def _voice(speaker: str) -> str:
    return SPEAKER_VOICE.get(speaker, "narrator")


def quotes(d: Path, chapter: int | None = None, voice: str = "", search: str = "", review: bool = False,
           changed: bool = False, offset: int = 0, limit: int = 100) -> dict:
    con = _need_db(d)
    try:
        asked = {r["quote_id"]: r["speaker"].strip() for r in read_overrides(d)
                 if not r["quote_id"].startswith("#") and r["speaker"].strip()}
        rows = con.execute(
            "SELECT u.quote_id q, MIN(s.doc_id) doc, MIN(s.seq) seq, GROUP_CONCAT(u.tts_text, ' ') text, "
            "MIN(u.speaker_id) speaker, MIN(u.confidence) conf, MAX(u.source='manual') manual "
            "FROM utterance u JOIN sentence s ON s.id=u.sentence_id WHERE u.is_quote=1 AND u.quote_id IS NOT NULL "
            "GROUP BY u.quote_id ORDER BY doc, seq").fetchall()
        try:
            extra = {r["quote_id"]: r for r in con.execute("SELECT quote_id, tag_text, rule, who, basis FROM v3_quote")}
        except sqlite3.Error:
            extra = {}
        out, counts = [], {"male": 0, "female": 0, "narrator": 0}
        pending = 0
        needle = search.strip().lower()
        for r in rows:
            v = _voice(r["speaker"])
            counts[v] += 1
            x = extra.get(r["q"])
            want = asked.get(r["q"])
            is_pending = bool(want) and want != v
            pending += is_pending
            unsure = v == "narrator" and not r["manual"] and x is not None and not x["rule"]
            if chapter is not None and r["doc"] != chapter:
                continue
            if voice and v != voice:
                continue
            if needle and needle not in (r["text"] or "").lower():
                continue
            if review and not unsure:
                continue
            if changed and not want:
                continue
            out.append({"id": r["q"], "chapter": r["doc"], "text": r["text"], "voice": v,
                        "confidence": r["conf"], "manual": bool(r["manual"]), "unsure": unsure,
                        "tag": (x["tag_text"] if x else "") or "", "who": (x["who"] if x else "") or "",
                        "why": (x["rule"] if x else "") or (x["basis"] if x else "") or "",
                        "asked": want or "", "pending": is_pending})
        return {"quotes": out[offset:offset + limit], "matching": len(out), "total": len(rows), "counts": counts,
                "pending": pending, "offset": offset, "limit": limit,
                "chapters": sorted({r["doc"] for r in rows})}
    finally:
        con.close()


def set_quotes(d: Path, changes: dict) -> dict:
    """Write the wanted voices into attribution_overrides.csv (null removes a row). The voices stage applies the file."""
    if not changes or not isinstance(changes, dict):
        raise ApiError("nothing to change")
    for q, v in changes.items():
        if v is not None and v not in VOICES:
            raise ApiError(f"a voice is one of {', '.join(VOICES)}")
    con = _need_db(d)
    try:
        for q in changes:
            if con.execute("SELECT 1 FROM utterance WHERE quote_id=? LIMIT 1", (q,)).fetchone() is None:
                raise ApiError(f"no quotation {q}", 404)
    finally:
        con.close()
    rows = [r for r in read_overrides(d) if r["quote_id"] not in changes]
    rows += [{"quote_id": q, "speaker": v, "style": "", "note": "set in the UI"} for q, v in changes.items() if v]
    p = overrides_path(d)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return {"rows": len(rows)}


def last_adapters(jobs, book_id: str) -> list[str]:
    """The --tts / --llm of the book's last job, so that applying a change does not switch models by surprise."""
    st = jobs.state(book_id) or {}
    args, out = st.get("args") or [], []
    for flag in ("--tts", "--llm"):
        if flag in args and args.index(flag) + 1 < len(args):
            out += [flag, args[args.index(flag) + 1]]
    return out


# ---- the gender table ------------------------------------------------------------------------------------------------------

def genders(d: Path) -> dict:
    manual = G.load_manual(d)
    auto = []
    p = G.path(d)
    if p.exists():
        in_auto = False
        for line in p.read_text().splitlines():
            if line.startswith("auto:"):
                in_auto = True
                continue
            if not in_auto:
                continue
            m = re.match(r'^\s+("(?:[^"\\]|\\.)*"|[^:]+):\s*(\w+)\s*(?:#\s*(.*?);\s*used\s+(\d+)x)?\s*$', line)
            if m:
                name = json.loads(m.group(1)) if m.group(1).startswith('"') else m.group(1).strip()
                auto.append({"name": name, "gender": m.group(2), "source": m.group(3) or "", "used": int(m.group(4) or 0)})
    return {"manual": [{"name": k, "gender": v} for k, v in sorted(manual.items())], "auto": auto,
            "exists": p.exists()}


def set_genders(d: Path, changes: dict) -> dict:
    """Edit `manual:` in genders.yaml (null removes a name); everything under `auto:` is left as the pipeline wrote it."""
    if not changes or not isinstance(changes, dict):
        raise ApiError("nothing to change")
    manual = G.load_manual(d)
    for name, g in changes.items():
        key = norm_phrase(str(name))
        if not key:
            raise ApiError("a name is needed")
        if g is None:
            manual.pop(key, None)
        elif g in ("male", "female", "unknown", "narrator"):
            manual[key] = g
        else:
            raise ApiError("a gender is male, female, unknown or narrator")
    p = G.path(d)
    head = ["# Name -> gender for v3 attribution (docs/DESIGN.md).",
            "# Edit `manual:` (name: male | female | unknown | narrator); it always wins. `auto:` is rewritten",
            "# on every run; the comment says where each answer came from and how often the name is used."]
    block = (["manual:"] + [f"  {json.dumps(k, ensure_ascii=False)}: {v}" for k, v in sorted(manual.items())]) \
        if manual else ["manual: {}"]
    tail = ["auto: {}"]
    if p.exists():
        lines = p.read_text().splitlines()
        i = next((k for k, ln in enumerate(lines) if ln.startswith("auto:")), None)
        if i is not None:
            tail = lines[i:]
    p.write_text("\n".join(head + block + tail) + "\n")
    yaml.safe_load(p.read_text())          # a file the pipeline cannot read must not be left behind
    return genders(d)


# ---- settings ---------------------------------------------------------------------------------------------------------------

FIELDS = [
    ("multi_voice", "Three voices", "bool", None,
     "Off: one narrator reads everything, and no language model is needed."),
    ("tts.adapter", "Speech", "select", ["kokoro", "drama3", "mock"],
     "kokoro runs on a laptop; drama3 needs a GPU machine; mock is a placeholder tone."),
    ("tts.endpoint", "Kokoro address", "text", None, "Where the Kokoro service listens."),
    ("tts.voice", "Kokoro narrator voice", "text", None, "A Kokoro voice name, or a mix such as b|bf_emma:0.70+bf_lily:0.30."),
    ("tts.drama3.endpoint", "Drama 3 address", "text", None, "Where the Drama 3 server listens."),
    ("tts.drama3.concurrency", "Drama 3 streams", "number", None,
     "Requests sent at once. Use the number of streams the server was started with."),
    ("tts.drama3.on_runaway", "A line Drama 3 cannot read", "select", ["fail", "cut"],
     "fail stops before assembly; cut keeps the start of the line, lists it, and finishes the book."),
    ("llm.adapter", "Language model", "select", ["openai_compat", "mock"], "Any OpenAI-compatible server, or none (mock)."),
    ("llm.endpoint", "Language model address", "text", None, "The server's /v1 address."),
    ("llm.model", "Language model name", "text", None, "The model name the server expects."),
    ("llm.max_tokens", "Longest model answer", "number", None, "Raise it if TRUNCATED warnings repeat."),
    ("voices.threshold", "How sure before a voice is used", "number", None,
     "A quotation gets a male or female voice only at this confidence; below it the narrator reads it."),
    ("output.format", "Audio format", "select", ["aac", "mp3cbr"], "AAC stayed in sync in every reader tried."),
    ("output.bitrate", "Audio bitrate", "text", None, "48k for AAC, 64k for MP3."),
    ("monitor.no_pause", "Never pause on warnings", "bool", None,
     "For an unattended run: warnings are recorded and the run carries on."),
]


def _get(cfg: dict, path: str):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set(cfg: dict, path: str, value) -> None:
    keys = path.split(".")
    for k in keys[:-1]:
        nxt = cfg.get(k)
        if not isinstance(nxt, dict):
            nxt = cfg[k] = {}
        cfg = nxt
    cfg[keys[-1]] = value


def _unset(cfg: dict, path: str) -> None:
    keys = path.split(".")
    trail = []
    cur = cfg
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            return
        trail.append((cur, k))
        cur = cur[k]
    cur.pop(keys[-1], None)
    for parent, k in reversed(trail):          # no empty sections left behind
        if parent[k] == {}:
            parent.pop(k)


def settings(d: Path) -> dict:
    p = d / "config.yaml"
    own = (yaml.safe_load(p.read_text()) if p.exists() else None) or {}
    base, eff = config.load(), config.load(d)
    fields = [{"key": k, "label": label, "type": typ, "options": opts, "help": hlp, "value": _get(eff, k),
               "default": _get(base, k), "own": _get(own, k) is not None} for k, label, typ, opts, hlp in FIELDS]
    return {"fields": fields, "text": p.read_text() if p.exists() else "", "file": str(p)}


def save_settings(d: Path, jobs, body: dict) -> dict:
    """`values`: {key: value, or null for "as the defaults say"} for the form; or `text`: the book's whole config.yaml."""
    _idle(d, jobs)
    p = d / "config.yaml"
    old = p.read_text() if p.exists() else None
    if "text" in body:
        text = str(body["text"])
        try:
            data = yaml.safe_load(text) if text.strip() else {}
        except yaml.YAMLError as e:
            raise ApiError(f"not valid YAML: {str(e).splitlines()[0]}") from e
        if data is not None and not isinstance(data, dict):
            raise ApiError("a config file is a list of key: value lines")
        new = text if text.strip() else None
    else:
        known = {k: (typ, opts) for k, _, typ, opts, _ in FIELDS}
        own = (yaml.safe_load(old) if old else None) or {}
        base = config.load()
        for key, val in (body.get("values") or {}).items():
            if key not in known:
                raise ApiError(f"unknown setting: {key}")
            typ, opts = known[key]
            if val is None or val == "":
                _unset(own, key)
                continue
            if typ == "bool":
                val = bool(val)
            elif typ == "number":
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    raise ApiError(f"{key} must be a number") from None
                val = int(val) if val == int(val) and isinstance(_get(base, key), int) else val
            elif typ == "select" and val not in opts:
                raise ApiError(f"{key} must be one of {', '.join(opts)}")
            else:
                val = str(val)
            if val == _get(base, key):
                _unset(own, key)             # the same as the default: no need for a line of its own
            else:
                _set(own, key, val)
        new = yaml.safe_dump(own, sort_keys=False, allow_unicode=True) if own else None
    if new is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(new if new.endswith("\n") else new + "\n")
    try:
        config.load(d)
    except Exception as e:  # noqa: BLE001 - put the old file back; a book must stay runnable
        if old is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(old)
        raise ApiError(f"those settings do not load: {e}") from e
    return settings(d)
