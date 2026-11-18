"""Fixed-size text chunks for the LLM passes (never crossing a document).

Chunks are cut at paragraph boundaries once `chunk_words` is reached, so an
LLM call's input size does not depend on book length (a design rule, docs/DESIGN.md).

`chunk_mode: scene` (docs/DESIGN.md, the simple version): once `chunk_words` is reached,
the cut waits for the first paragraph boundary that is not between two speech paragraphs (a paragraph
with a quotation, by pipeline/quotes.py), so a conversation is not split. No upper limit.
"""
from __future__ import annotations

from collections import defaultdict


MODES = ("words", "scene")


def speech_paragraphs(con, paras: dict[int, list], style: str, doc_id: int) -> set[int]:
    """Paragraphs (of one doc) with a quotation in their story sentences (classify; a preface quoting the
    novel is not a conversation). All sentences count if the book has not been classified."""
    from .quotes import split_paragraph
    story = None
    if con.execute("SELECT 1 FROM section LIMIT 1").fetchone():
        from .stages.classify import story_filter_sql
        story = {r[0] for r in con.execute(f"SELECT s.seq FROM sentence s WHERE s.doc_id=? AND {story_filter_sql('s')}",
                                           (doc_id,))}
    out = set()
    for p, rows in paras.items():
        rows = [r for r in rows if story is None or r["seq"] in story]
        if not rows:
            continue
        pcs = split_paragraph([r["tts_text"] for r in rows], style, f"{doc_id:03d}p{p:04d}")
        if any(x.is_quote for sent in pcs for x in sent):
            out.add(p)
    return out


def ensure_chunks(con, doc_ids: list[int], chunk_words: int, mode: str = "words", style: str | None = None) -> None:
    """Chunks for docs that have none yet (a doc chunked once keeps its chunks, whatever the mode now)."""
    if mode not in MODES:
        raise ValueError(f"character_memory.chunk_mode must be one of {MODES}, not {mode!r}")
    if mode == "scene" and not style:
        raise ValueError("chunk_mode scene needs the quote style")
    for doc_id in doc_ids:
        if con.execute("SELECT 1 FROM chunk WHERE doc_id=?", (doc_id,)).fetchone():
            continue
        rows = con.execute("SELECT seq, para, tts_text FROM sentence WHERE doc_id=? ORDER BY seq", (doc_id,)).fetchall()
        paras = defaultdict(list)
        for r in rows:
            paras[r["para"]].append(r)
        order = sorted(paras)
        speech = speech_paragraphs(con, paras, style, doc_id) if mode == "scene" else set()
        k, cur, words = 0, [], 0
        for i, p in enumerate(order):
            cur.extend(paras[p])
            words += sum(len(r["tts_text"].split()) for r in paras[p])
            nxt = order[i + 1] if i + 1 < len(order) else None
            if mode == "scene" and nxt is not None and p in speech and nxt in speech:
                continue                    # inside a conversation: cut later
            if words >= chunk_words:
                con.execute("INSERT INTO chunk(id, doc_id, seq_from, seq_to) VALUES (?,?,?,?)",
                            (f"c{doc_id:03d}-{k:03d}", doc_id, cur[0]["seq"], cur[-1]["seq"]))
                k, cur, words = k + 1, [], 0
        if cur:
            con.execute("INSERT INTO chunk(id, doc_id, seq_from, seq_to) VALUES (?,?,?,?)",
                        (f"c{doc_id:03d}-{k:03d}", doc_id, cur[0]["seq"], cur[-1]["seq"]))
    con.commit()


def chunk_rows(con, doc_ids: list[int]):
    q = ",".join("?" * len(doc_ids))
    return con.execute(f"SELECT * FROM chunk WHERE doc_id IN ({q}) ORDER BY doc_id, seq_from", doc_ids).fetchall()


def chunk_sentences(con, chunk) -> list:
    return con.execute("SELECT * FROM sentence WHERE doc_id=? AND seq BETWEEN ? AND ? ORDER BY seq",
                       (chunk["doc_id"], chunk["seq_from"], chunk["seq_to"])).fetchall()


def story_sentences(con, chunk) -> list:
    """Sentences of the chunk that belong to story sections (see classify); all of them if the
    book has not been classified."""
    if not con.execute("SELECT 1 FROM section LIMIT 1").fetchone():
        return chunk_sentences(con, chunk)
    from .stages.classify import story_filter_sql
    return con.execute(f"SELECT * FROM sentence s WHERE s.doc_id=? AND s.seq BETWEEN ? AND ? AND "
                       f"{story_filter_sql('s')} ORDER BY s.seq",
                       (chunk["doc_id"], chunk["seq_from"], chunk["seq_to"])).fetchall()


def chunk_text(sentences) -> str:
    """Sentences joined; blank line between paragraphs."""
    out, prev = [], None
    for s in sentences:
        if prev is not None and s["para"] != prev:
            out.append("\n\n")
        elif prev is not None:
            out.append(" ")
        out.append(s["tts_text"])
        prev = s["para"]
    return "".join(out)


def scope_docs(con, doc_ids: list[int] | None) -> list[int]:
    ids = [r["id"] for r in con.execute("SELECT id FROM doc WHERE has_speech=1 ORDER BY id")]
    return [i for i in ids if doc_ids is None or i in doc_ids]
