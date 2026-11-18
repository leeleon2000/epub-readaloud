"""Stage 2: wrap sentences in <span id>, fill the sentence table (F1).

Always recomputed from work/base (cheap and deterministic, so sentence ids
are stable across runs). Takes are keyed by text, not by sentence id, so
re-segmenting never forces re-synthesis of unchanged sentences.
"""
from __future__ import annotations

import copy

from .. import epub, xhtml
from ..book import Book
from ..displaycheck import check
from ..segment_core import segment_document


def run(book: Book, con, cfg: dict, log=print, **_) -> None:
    pkg = epub.open_package(book.base)
    docs = con.execute("SELECT id, href FROM doc ORDER BY id").fetchall()
    rows, has_speech = [], {}
    for d in docs:
        tree = xhtml.parse(pkg.file(d["href"]))
        orig = copy.deepcopy(xhtml.body(tree))
        res = segment_document(tree, f"ab{d['id']:03d}s")
        check(orig, xhtml.body(tree), {s.sid for s in res.sentences}, f"segment {d['href']}")
        xhtml.write(tree, book.seg / pkg.opf_dir / d["href"])
        rows += [(s.sid, d["id"], k, s.para, s.kind, s.text, s.tts_text) for k, s in enumerate(res.sentences)]
        has_speech[d["id"]] = 1 if res.sentences else 0

    old = [tuple(r) for r in con.execute(
        "SELECT id, doc_id, seq, para, kind, text, tts_text FROM sentence ORDER BY doc_id, seq")]
    n_docs = sum(has_speech.values())
    if old == rows and con.execute("SELECT COUNT(*) FROM utterance").fetchone()[0]:
        log(f"segment: {len(rows)} sentences in {n_docs}/{len(docs)} documents (unchanged); display check passed")
        return
    if old and old != rows:
        # Text changed (new source or new segmenter): everything derived from sentences is stale.
        log("segment: sentences changed; resetting utterances, chunks and attributions")
    if old or con.execute("SELECT COUNT(*) FROM utterance").fetchone()[0]:
        for t in ("sentence", "utterance", "chunk", "block", "selection", "section"):
            con.execute(f"DELETE FROM {t}")
    con.executemany("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text, speaker) "
                    "VALUES (?,?,?,?,?,?,?, 'narrator')", rows)
    con.executemany("INSERT INTO utterance(id, sentence_id, seq, start, end, tts_text) VALUES (?,?,0,0,?,?)",
                    [(f"{r[0]}u0", r[0], len(r[6]), r[6]) for r in rows])
    for doc_id, hs in has_speech.items():
        con.execute("UPDATE doc SET has_speech=? WHERE id=?", (hs, doc_id))
    con.commit()
    log(f"segment: {len(rows)} sentences in {n_docs}/{len(docs)} documents; display check passed")
