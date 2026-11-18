"""Stage: classify (content classification, docs/DESIGN.md). What kind of text is each section?

Sections are runs of sentences inside one document, cut at headings and at
the Gutenberg START/END markers. Consecutive headings stay in one section
(chapter number + subtitle), except that a chapter-like or keyword heading
(Contents, Preface, ...) always starts a new one, so an illustration caption
set as a heading stays separate from the chapter after it. Rules decide most sections; the LLM sees
only "heading + first ~100 words + position" for the rest (fixed input size).
books/<id>/sections.yaml overrides everything (`readaloud sections --set`). Its `skip:` list names single
sentences that are never read.

Effects:
  story                                   -> characters + attribute, narrated
  front_matter / notes / back_matter      -> narrated by the narrator only
  uncertain (LLM confidence too low)      -> narrated, kept out of characters until resolved
  toc / captions / boilerplate            -> not narrated, not in SMIL (sentence.speak = 0)
Page-number sentences (digits, upper-case roman numerals, "PAGE") are never narrated.
"""
from __future__ import annotations

import json
import re

import yaml

from ..adapters.llm import LLMError, get_llm
from ..book import Book
from ..memory import prompts

SPOKEN = {"story", "front_matter", "notes", "back_matter", "uncertain"}
# Only these carry over to a next document that starts without a heading. A front-matter or copyright
# document says nothing about the next one: one omnibus edition has no heading tags at all,
# and "continues" spread its title page (front_matter) and copyright pages (boilerplate) over the whole
# book, so not one sentence was story (2026-11-16). The model decides those documents instead.
CONTINUES = {"story", "notes"}
CATEGORIES = set(prompts.CATEGORIES) | {"uncertain"}

PG_START = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG", re.I)
# A page number may sit in brackets or end in a full stop, nothing else: “I?” and “I!” are someone speaking
# (found on The Mysterious Affair at Styles and Oliver Twist, where those lines were silently not read).
PAGE = re.compile(r"^[\s\[\(]*(?:\d{1,4}|[IVXLCDM]{1,7}|PAGE)[\s\]\).]*$")
CHAPTERISH = re.compile(r"^\W*(?:chapter|chap\b|book\b|part\b|volume\b|vol\.|letter\b|canto\b|prologue|epilogue|"
                        r"stave\b|[ivxlcdm]+\W*$|\d+\W*$)", re.I)
# A copyright page (publisher, ISBN, Library of Congress data) is boilerplate, not read: Drama 3 runs away on
# ISBNs and call numbers (2026-11-15: three lines failed every seed and blocked packaging).
COPYRIGHT = re.compile(r"\bISBN\b|copyright\s*©|©\s*(?:19|20)\d\d|\ball rights reserved\b|library of congress|"
                       r"\bprinted in\b.*\b(?:united states|u\.s\.a|great britain)\b", re.I)
TITLE_RULES = [
    (re.compile(r"^\W*(?:table of )?contents\b", re.I), "toc"),
    (re.compile(r"^\W*(?:list of )?illustrations\b", re.I), "captions"),
    (re.compile(r"^\W*(?:preface|introduction|foreword|prefatory|dedication|editor'?s note|"
                r"translator'?s note)\b", re.I), "front_matter"),
    (re.compile(r"^\W*(?:notes|endnotes|footnotes)\b", re.I), "notes"),
    (re.compile(r"^\W*(?:index|appendix|glossary|bibliography|afterword|advertisements?)\b", re.I), "back_matter"),
]


def sections_path(book: Book):
    return book.dir / "sections.yaml"


def _sections_file(book: Book) -> dict:
    p = sections_path(book)
    return (yaml.safe_load(p.read_text()) if p.exists() else None) or {}


def load_overrides(book: Book) -> dict[str, str]:
    return {str(k): v for k, v in _sections_file(book).items() if k != "skip"}


def load_skip(book: Book) -> list[str]:
    """Sentence ids that are never read, whatever their section (`skip:` in sections.yaml): a printer's imprint
    after the last line of a novel, a stray credit. A section can only be set as a whole; this is for single lines."""
    return [str(x) for x in (_sections_file(book).get("skip") or [])]


def save_overrides(book: Book, data: dict[str, str]) -> None:
    skip = load_skip(book)
    sections_path(book).write_text(
        "# Manual section categories (win over rules and the model). Categories:\n"
        f"# {', '.join(sorted(CATEGORIES - {'uncertain'}))}\n"
        "# skip: [sentence ids] = single sentences that are never read (ids as in the page spans, e.g. ab016s00171)\n"
        + yaml.safe_dump(data, sort_keys=True) + (yaml.safe_dump({"skip": skip}) if skip else ""))


def build_sections(con) -> list[dict]:
    """Cut every document into sections; returns them in book order (not yet classified)."""
    out = []
    for d in con.execute("SELECT id FROM doc ORDER BY id").fetchall():
        rows = con.execute("SELECT seq, kind, text FROM sentence WHERE doc_id=? ORDER BY seq", (d["id"],)).fetchall()
        cur, prev_kind = None, None
        for r in rows:
            heading_starts = r["kind"] == "heading" and (prev_kind != "heading" or CHAPTERISH.search(r["text"])
                                                         or any(rx.search(r["text"]) for rx, _ in TITLE_RULES))
            start_new = cur is None or heading_starts or PG_END.search(r["text"]) or cur.get("close")
            prev_kind = r["kind"]
            if start_new:
                cur = {"doc_id": d["id"], "seq_from": r["seq"], "seq_to": r["seq"],
                       "title": r["text"] if r["kind"] == "heading" else "", "texts": []}
                out.append(cur)
            cur["seq_to"] = r["seq"]
            cur["texts"].append(r["text"])
            if PG_START.search(r["text"]):
                cur["close"] = True          # the START marker ends the header section
    for s in out:
        s["id"] = f"d{s['doc_id']:03d}s{s['seq_from']:05d}"
        s.pop("close", None)
    return out


def _rules(secs: list[dict]) -> None:
    """Fill category/confidence/reason for sections the rules can decide."""
    start = next((i for i, s in enumerate(secs) if any(PG_START.search(t) for t in s["texts"])), None)
    end = next((i for i, s in enumerate(secs) if any(PG_END.search(t) for t in s["texts"])), None)
    for i, s in enumerate(secs):
        if start is not None and i <= start:
            s.update(category="boilerplate", confidence=1.0, reason="before the Gutenberg START marker")
        elif end is not None and i >= end:
            s.update(category="boilerplate", confidence=1.0, reason="after the Gutenberg END marker")
        elif len(s["texts"]) <= 60 and len({m.group(0).lower()[:4] for t in s["texts"]
                                             for m in COPYRIGHT.finditer(t)}) >= 2:
            s.update(category="boilerplate", confidence=0.95, reason="copyright page (ISBN, copyright notice)")
        elif s["title"] and CHAPTERISH.search(s["title"]):
            s.update(category="story", confidence=0.95, reason="chapter-like heading")
        else:
            for rx, cat in TITLE_RULES:
                if s["title"] and rx.search(s["title"]):
                    s.update(category=cat, confidence=0.95, reason=f"heading '{s['title'][:40]}'")
                    break
    # A heading-only section right before a chapter heading is an illustration caption.
    for i, s in enumerate(secs):
        nxt = secs[i + 1] if i + 1 < len(secs) else None
        if ("category" not in s and len(s["texts"]) == 1 and s["title"] and nxt and nxt["doc_id"] == s["doc_id"]
                and nxt["title"] and CHAPTERISH.search(nxt["title"])):
            s.update(category="captions", confidence=0.9, reason="heading-only section before a chapter heading")
    # A document that starts without a heading continues the previous document's last section.
    for i, s in enumerate(secs):
        if ("category" not in s and i > 0 and not s["title"] and s["seq_from"] == 0
                and secs[i - 1].get("category") in CONTINUES):
            p = secs[i - 1]
            s.update(category=p["category"], confidence=p["confidence"], reason=f"continues {p['id']}")


def run(book: Book, con, cfg: dict, log=print, obs=None, llm=None, **_) -> None:
    skip = load_skip(book)              # checked before anything is written: an error must not leave a transaction open
    if skip:
        known = {r[0] for r in con.execute(f"SELECT id FROM sentence WHERE id IN ({','.join('?' * len(skip))})", skip)}
        missing = [x for x in skip if x not in known]
        if missing:
            raise SystemExit(f"classify: sections.yaml skip: no sentence {missing[0]!r}")
    from ..observe import Observer
    obs = obs or Observer.null(log)
    ccfg, mon = cfg.get("classify", {}), cfg.get("monitor", {})
    secs = build_sections(con)
    _rules(secs)
    prev = {r["id"]: r for r in con.execute("SELECT * FROM section")}
    overrides = load_overrides(book)
    use_llm = cfg.get("multi_voice") and ccfg.get("use_llm", True)
    n_docs = con.execute("SELECT COUNT(*) FROM doc").fetchone()[0]
    asked = 0
    # Two passes so that "continues the previous document" can follow a model decision.
    for _pass in range(2):
        for i, s in enumerate(secs):
            if "category" in s:
                continue
            if i > 0 and not s["title"] and s["seq_from"] == 0 and secs[i - 1].get("category") in CONTINUES:
                p = secs[i - 1]
                s.update(category=p["category"], confidence=p["confidence"], reason=f"continues {p['id']}")
                continue
            if _pass == 1:
                continue
            old = prev.get(s["id"])
            if (old and old["source"] == "model" and old["seq_to"] == s["seq_to"]
                    and not (old["reason"] or "").startswith("LLM failed")):
                s.update(category=old["category"], confidence=old["confidence"], reason=old["reason"], source="model")
                continue
            if not use_llm:
                s.update(category="story", confidence=0.5, reason="no rule matched (LLM disabled)")
                continue
            llm = llm or get_llm(cfg["llm"], mon)
            llm.obs = obs
            obs.chunk_id = s["id"]
            opening = " ".join(" ".join(s["texts"]).split()[:100])
            position = f"document {s['doc_id']} of {n_docs - 1}, section {i + 1} of {len(secs)}"
            msgs, schema, payload = prompts.classify(s["title"], opening, position)
            try:
                out = llm.complete_json(msgs, schema, "classify_section", payload, max_tokens=300)
            except LLMError as e:
                obs.warn("LLM_FAILED", "warn", s["id"], f"classify: {e}")
                # source 'rule' (not 'model') so the next run asks again instead of reusing a failure
                s.update(category="uncertain", confidence=0.0, reason=f"LLM failed: {e}"[:200], source="rule")
                continue
            asked += 1
            cat, conf = out["category"], float(out["confidence"])
            if conf < mon.get("section_min_confidence", 0.7):
                s.update(category="uncertain", confidence=conf, reason=f"model said {cat}: {out['reason']}"[:200])
            else:
                s.update(category=cat, confidence=conf, reason=out["reason"][:200])
            s["source"] = "model"
    obs.chunk_id = None
    for s in secs:
        s.setdefault("source", "rule")
        if s["id"] in overrides:
            s.update(category=overrides[s["id"]], confidence=1.0, source="manual", reason="sections.yaml")
        if s["category"] not in CATEGORIES:
            raise SystemExit(f"classify: {s['id']}: unknown category {s['category']!r}")

    con.execute("DELETE FROM section")
    con.executemany("INSERT INTO section(id, doc_id, seq_from, seq_to, title, category, confidence, source, reason) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    [(s["id"], s["doc_id"], s["seq_from"], s["seq_to"], s["title"][:200], s["category"],
                      s["confidence"], s["source"], s["reason"]) for s in secs])
    for s in secs:
        con.execute("UPDATE sentence SET section_id=?, speak=? WHERE doc_id=? AND seq BETWEEN ? AND ?",
                    (s["id"], int(s["category"] in SPOKEN), s["doc_id"], s["seq_from"], s["seq_to"]))
    con.execute("UPDATE sentence SET speak=0 WHERE kind='caption'")      # illustration captions (segment_core.in_caption)
    if skip:
        con.execute(f"UPDATE sentence SET speak=0 WHERE id IN ({','.join('?' * len(skip))})", skip)
    for r in con.execute("SELECT id, text FROM sentence WHERE speak=1 AND kind='body' AND length(text) <= 12").fetchall():
        if PAGE.match(r["text"]):
            con.execute("UPDATE sentence SET speak=0 WHERE id=?", (r["id"],))
    con.execute("UPDATE doc SET has_speech = EXISTS (SELECT 1 FROM sentence s WHERE s.doc_id=doc.id AND s.speak=1)")
    con.commit()

    # Warnings about sections that are now decided are resolved automatically.
    decided = [s["id"] for s in secs if s["category"] != "uncertain"]
    for sid in decided:
        con.execute("UPDATE warning SET status='resolved', handled_by='classify', note='section now classified' "
                    "WHERE code IN ('SECTION_UNCERTAIN','LLM_FAILED') AND chunk_id=? AND status='open'", (sid,))
    con.commit()

    # Uncertain sections: pause before the character pass has started, else just warn.
    started = con.execute("SELECT 1 FROM chunk WHERE pass1='done' LIMIT 1").fetchone()
    acked = {r["chunk_id"] for r in con.execute(
        "SELECT chunk_id FROM warning WHERE code='SECTION_UNCERTAIN' AND status!='open'")}
    unsure = [s for s in secs if s["category"] == "uncertain" and s["id"] not in acked]
    counts = {}
    for s in secs:
        counts[s["category"]] = counts.get(s["category"], 0) + (s["seq_to"] - s["seq_from"] + 1)
    log(f"classify: {len(secs)} sections ({asked} asked the LLM); sentences by category: "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    for s in unsure:
        obs.warn("SECTION_UNCERTAIN", "warn" if started else "pause", s["id"],
                 f"'{s['title'][:40]}': {s['reason']}", opening=" ".join(" ".join(s["texts"]).split()[:40]))
    obs.checkpoint()
    if unsure and not started:
        from ..observe import PipelinePaused
        raise PipelinePaused("SECTION_UNCERTAIN", unsure[0]["id"],
                             f"{len(unsure)} sections need a category: readaloud sections {book.id}")


def story_filter_sql(alias: str = "s") -> str:
    """SQL condition: sentence belongs to a story section and is narrated."""
    return (f"{alias}.speak=1 AND EXISTS (SELECT 1 FROM section x WHERE x.id={alias}.section_id "
            f"AND x.category='story')")


def set_categories(book: Book, con, assignments: dict[str, str]) -> list[str]:
    """Manual override; returns chunk ids whose character pass is now out of date."""
    bad = [v for v in assignments.values() if v not in CATEGORIES - {"uncertain"}]
    if bad:
        raise SystemExit(f"unknown category: {bad[0]} (use one of {sorted(CATEGORIES - {'uncertain'})})")
    known = {r["id"] for r in con.execute("SELECT id FROM section")}
    missing = [k for k in assignments if k not in known]
    if missing:
        raise SystemExit(f"unknown section: {missing[0]} (see `readaloud sections {book.id}`)")
    data = load_overrides(book)
    data.update(assignments)
    save_overrides(book, data)
    affected = []
    for sid, cat in assignments.items():
        s = con.execute("SELECT * FROM section WHERE id=?", (sid,)).fetchone()
        for c in con.execute("SELECT id FROM chunk WHERE doc_id=? AND seq_to >= ? AND seq_from <= ? AND pass1='done'",
                             (s["doc_id"], s["seq_from"], s["seq_to"])):
            if c["id"] not in affected:
                affected.append(c["id"])
        con.execute("UPDATE warning SET status='resolved', handled_by='readaloud sections', note=? "
                    "WHERE code='SECTION_UNCERTAIN' AND chunk_id=? AND status='open'", (f"set to {cat}", sid))
    con.commit()
    return affected
