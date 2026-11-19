"""What the UI's pages ask for, as plain data.

Reading goes straight to a book's database through a read-only connection, so a page never waits on (or blocks) a
running job. Changing anything goes through the command line itself (`cli()` below), so the UI cannot drift from it.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from .. import config
from ..config import REPO
from . import md

BOOK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
SOURCES = (".epub", ".azw3", ".mobi")
DOCS = [("README", "README.md", "Overview"), ("USAGE", "docs/USAGE.md", "Using it, step by step"),
        ("MODELS", "docs/MODELS.md", "Serving the models"), ("DESIGN", "docs/DESIGN.md", "How it works")]
STAGE_HELP = {
    "ingest": "Unpacks the book. EPUB 2 is upgraded to EPUB 3.",
    "segment": "Wraps every sentence in a span and checks that the displayed text did not change.",
    "classify": "Decides which sections are story and which are not read (contents, captions, licence text).",
    "cast": "Makes the narrator, male and female voices. Needs the speech service.",
    "voices": "Gives every quotation one of the three voices. Needs the language model.",
    "synth": "Generates the audio, one take per piece of text. Needs the speech service.",
    "assemble": "Joins the takes into one audio file per chapter, with the timing of every sentence.",
    "package": "Writes the EPUB 3 with Media Overlays and runs epubcheck.",
}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def check_id(book_id: str) -> str:
    if not BOOK_ID.match(book_id or ""):
        raise ApiError("a book id is lower-case letters, digits and hyphens", 400)
    return book_id


def book_dir(root: Path, book_id: str) -> Path:
    d = Path(root) / check_id(book_id)
    if not (d / "source.epub").exists():
        raise ApiError(f"no such book: {book_id}", 404)
    return d


def db(d: Path) -> sqlite3.Connection | None:
    """A read-only connection to a book's database, or None before the first stage has made one."""
    p = d / "project.db"
    if not p.exists():
        return None
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=5)
    con.row_factory = sqlite3.Row
    return con


def cli(root: Path, *args: str, timeout: float = 600) -> tuple[int, str]:
    """Run `readaloud --books <root> <args...>` and wait: (exit code, what it printed)."""
    p = subprocess.run([sys.executable, "-m", "pipeline.cli", "--books", str(root), *args], cwd=REPO,
                       capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    return p.returncode, (p.stdout + p.stderr).strip()


def _one(con, sql: str, *params, default=0):
    try:
        r = con.execute(sql, params).fetchone()
    except sqlite3.Error:
        return default
    return r[0] if r and r[0] is not None else default


def _cfg(d: Path) -> dict:
    return config.load(d)


def stages(d: Path, con, cfg: dict) -> list[dict]:
    """Where each stage of a full run stands: done, partial or todo, with one line of detail."""
    from ..cli import planned_stages
    from ..stages import cast as cast_stage
    from ..book import Book
    book = Book(d.name, d)
    out = []
    n_docs = _one(con, "SELECT COUNT(*) FROM doc") if con else 0
    n_speech = _one(con, "SELECT COUNT(*) FROM doc WHERE has_speech=1") if con else 0
    n_sent = _one(con, "SELECT COUNT(*) FROM sentence") if con else 0
    n_sec = _one(con, "SELECT COUNT(*) FROM section") if con else 0
    n_utt = _one(con, "SELECT COUNT(*) FROM utterance u JOIN sentence s ON s.id=u.sentence_id WHERE s.speak=1") if con else 0
    n_sel = _one(con, "SELECT COUNT(*) FROM selection") if con else 0
    quotes = {}
    if con:
        try:
            quotes = {r[0]: r[1] for r in con.execute(
                "SELECT speaker_id, COUNT(DISTINCT quote_id) FROM utterance WHERE is_quote=1 GROUP BY 1")}
        except sqlite3.Error:
            quotes = {}
    n_wav = len(list(book.chapters.glob("*.wav"))) if book.chapters.exists() else 0
    outs = sorted(book.out.glob("*.epub")) if book.out.exists() else []
    for st in planned_stages(cfg):
        state, detail = "todo", ""
        if st == "ingest" and n_docs:
            state, detail = "done", f"{n_docs} files, {n_speech} with text to read"
        elif st == "segment" and n_sent:
            state, detail = "done", f"{n_sent:,} sentences"
        elif st == "classify" and n_sec:
            story = _one(con, "SELECT COUNT(*) FROM section WHERE category='story'")
            state, detail = "done", f"{n_sec} sections, {story} of them story"
        elif st == "cast":
            p = cast_stage.cast_path(book, cfg)
            if p.exists():
                state, detail = "done", p.name
        elif st == "voices" and quotes:
            m, f = quotes.get("_generic_male", 0), quotes.get("_generic_female", 0)
            total = sum(quotes.values())
            state, detail = "done", f"{total:,} quotations: {m:,} male, {f:,} female, {total - m - f:,} narrator"
        elif st == "synth" and n_utt and n_sel:
            state = "done" if n_sel >= n_utt else "partial"
            detail = f"{n_sel:,} of {n_utt:,} pieces"
        elif st == "assemble" and n_wav:
            state = "done" if n_speech and n_wav >= n_speech else "partial"
            detail = f"{n_wav} of {n_speech} chapters"
        elif st == "package" and outs:
            state, detail = "done", outs[-1].name
        out.append({"stage": st, "state": state, "detail": detail, "help": STAGE_HELP.get(st, "")})
    return out


def outputs(d: Path) -> list[dict]:
    out = d / "out"
    files = sorted(out.glob("*.epub"), key=lambda p: p.stat().st_mtime, reverse=True) if out.exists() else []
    return [{"name": p.name, "bytes": p.stat().st_size, "modified": p.stat().st_mtime} for p in files]


def book_summary(root: Path, d: Path, jobs) -> dict:
    con = db(d)
    try:
        cfg = _cfg(d)
        st = stages(d, con, cfg)
        title = _one(con, "SELECT value FROM meta WHERE key='title'", default="") if con else ""
        warn = _one(con, "SELECT COUNT(*) FROM warning WHERE status='open'") if con else 0
        return {"id": d.name, "title": title or d.name, "stages": st,
                "done": sum(1 for s in st if s["state"] == "done"), "planned": len(st),
                "warnings": warn, "outputs": outputs(d), "job": jobs.state(d.name)}
    finally:
        if con:
            con.close()


def books(root: Path, jobs) -> list[dict]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and BOOK_ID.match(d.name) and (d / "source.epub").exists():
            try:
                out.append(book_summary(root, d, jobs))
            except Exception as e:  # one unreadable book must not hide the others
                out.append({"id": d.name, "title": d.name, "stages": [], "done": 0, "planned": 0, "warnings": 0,
                            "outputs": [], "job": None, "error": str(e)})
    return out


def book(root: Path, book_id: str, jobs) -> dict:
    d = book_dir(root, book_id)
    out = book_summary(root, d, jobs)
    cfg = _cfg(d)
    con = db(d)
    chapters = []
    try:
        if con:
            from ..book import Book
            b = Book(d.name, d)
            for r in con.execute(
                    "SELECT d.id, d.href, COUNT(DISTINCT s.id) n, COUNT(u.id) nu, COUNT(sel.take_id) synth, "
                    "(SELECT text FROM sentence h WHERE h.doc_id=d.id AND h.kind='heading' ORDER BY h.seq LIMIT 1) head "
                    "FROM doc d LEFT JOIN sentence s ON s.doc_id=d.id AND s.speak=1 "
                    "LEFT JOIN utterance u ON u.sentence_id=s.id LEFT JOIN selection sel ON sel.block_id=u.id "
                    "WHERE d.has_speech=1 GROUP BY d.id ORDER BY d.id"):
                chapters.append({"id": r["id"], "file": r["href"].rsplit("/", 1)[-1], "heading": (r["head"] or "")[:80],
                                 "sentences": r["n"], "pieces": r["nu"], "synthesised": r["synth"],
                                 "assembled": (b.chapters / f"{r['id']:03d}.wav").exists()})
            out["source_version"] = _one(con, "SELECT value FROM meta WHERE key='epub_version_in'", default="")
    finally:
        if con:
            con.close()
    out["chapters"] = chapters
    out["config"] = {"tts": cfg["tts"]["adapter"], "llm": cfg["llm"]["adapter"],
                     "multi_voice": bool(cfg.get("multi_voice")),
                     "own_config": (d / "config.yaml").exists()}
    return out


def sample_book() -> Path | None:
    p = REPO / "tests" / "fixtures" / "alice_epub2.epub"
    return p if p.exists() else None


def add_book(root: Path, source: Path, book_id: str | None) -> dict:
    """`readaloud init`: a new book from an EPUB (or AZW3 / MOBI, converted with Calibre)."""
    args = ["init", str(source)] + (["--id", check_id(book_id)] if book_id else [])
    rc, out = cli(root, *args)
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "could not add the book", 400)
    m = re.search(r"initialized .*?([^/\\\s]+)\s*$", out, re.M)
    new_id = book_id or (m.group(1) if m else source.stem)
    return {"id": new_id, "message": out}


def save_upload(root: Path, name: str, data: bytes) -> Path:
    """An uploaded book, written where `init` can read it; the caller removes it afterwards."""
    safe = re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).name) or "book.epub"
    if Path(safe).suffix.lower() not in SOURCES:
        raise ApiError("the file must be an .epub, .azw3 or .mobi", 400)
    up = Path(root) / ".uploads"
    up.mkdir(parents=True, exist_ok=True)
    p = up / safe
    p.write_bytes(data)
    return p


def remove_upload(p: Path) -> None:
    p.unlink(missing_ok=True)
    try:
        p.parent.rmdir()
    except OSError:
        pass


AUDIO_STAGES = ["synth", "assemble", "package"]


def run_args(book_id: str, body: dict, cfg: dict | None = None) -> tuple[list[list[str]], str]:
    """The `readaloud run` commands for a request from the Run panel, and a short label for the job.

    Chapters limit the audio only. Which voice reads a line is decided from the whole book's text (the table of names
    and genders is learnt across it; with one chapter alone, a third of that chapter's quotations lost their voice), so
    a request for some chapters is two commands: the text stages for the book, then the audio stages for the chapters.
    Explicit `stages` are run as given."""
    from ..cli import STAGES, planned_stages
    args = ["run", check_id(book_id)]
    label = "whole book"
    tts, llm = body.get("tts"), body.get("llm")
    if tts:
        if tts not in ("mock", "kokoro", "drama3"):
            raise ApiError(f"unknown speech adapter: {tts}")
        args += ["--tts", tts]
    if llm:
        if llm not in ("mock", "openai_compat"):
            raise ApiError(f"unknown language model adapter: {llm}")
        args += ["--llm", llm]
    docs = body.get("docs")
    ids = [str(int(x)) for x in (docs if isinstance(docs, list) else str(docs).split(",")) if str(x).strip()] if docs else []
    only, until = body.get("stages"), body.get("until")
    if only:
        only = [s for s in (only if isinstance(only, list) else str(only).split(",")) if s]
        bad = [s for s in only if s not in STAGES]
        if bad:
            raise ApiError(f"unknown stage: {bad[0]}")
        args += ["--stages", ",".join(only)] + (["--docs", ",".join(ids)] if ids else [])
        return [args], "stages " + ", ".join(only) + (f", chapters {', '.join(ids)}" if ids else "")
    if until and until not in STAGES:
        raise ApiError(f"unknown stage: {until}")
    until = until or "package"
    if ids and until in AUDIO_STAGES:
        plan = planned_stages(cfg or config.load())
        text = [s for s in plan if s not in AUDIO_STAGES]
        audio = AUDIO_STAGES[: AUDIO_STAGES.index(until) + 1]
        return ([args + ["--until", text[-1]], args + ["--stages", ",".join(audio), "--docs", ",".join(ids)]],
                f"chapters {', '.join(ids)}" + ("" if until == "package" else f", up to {until}"))
    if until != "package":
        args += ["--until", until]
        label = f"up to {until}"
    return [args], label


def output_file(root: Path, book_id: str, name: str) -> Path:
    d = book_dir(root, book_id)
    p = (d / "out" / Path(name).name)
    if p.suffix != ".epub" or not p.is_file():
        raise ApiError("no such file", 404)
    return p


def delete_book(root: Path, book_id: str, jobs) -> None:
    d = book_dir(root, book_id)
    if jobs.running(book_id):
        raise ApiError("stop the running job first", 409)
    shutil.rmtree(d)


def doc_list() -> list[dict]:
    return [{"id": i, "title": t} for i, path, t in DOCS if (REPO / path).exists()]


def doc(doc_id: str) -> dict:
    for i, path, title in DOCS:
        if i == doc_id and (REPO / path).exists():
            body, heads = md.render((REPO / path).read_text(encoding="utf-8"), here=i, base=str(Path(path).parent.as_posix()).strip("."))
            return {"id": i, "title": title, "html": body, "headings": [h for h in heads if h["level"] <= 3]}
    raise ApiError("no such document", 404)


def repo_file(rel: str) -> Path:
    """A picture a document shows (docs/img/...). Nothing else in the repository is served."""
    p = (REPO / rel).resolve()
    img = (REPO / "docs" / "img").resolve()
    if img not in p.parents or not p.is_file() or p.suffix.lower() not in (".gif", ".png", ".jpg", ".jpeg", ".svg"):
        raise ApiError("not found", 404)
    return p


def dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
