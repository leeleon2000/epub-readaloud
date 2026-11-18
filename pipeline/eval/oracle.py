"""Upper-bound experiment (docs/DESIGN.md): attribute with PDNC's character list.

    make_oracle_book(src_dir, out_dir, pdnc_dir, overrides)
    -> out_dir is a copy of the book whose character memory is replaced by PDNC's characters
       (manual events, clean ids like `mrs_bennet`), every chunk marked as having passed the
       character stage, and attribute reset. Then: `readaloud run <out> --stages attribute --docs ...`.

Per-chunk character lists (the attribute candidates) are what a perfect character stage would
produce: PDNC characters whose (unambiguous) alias occurs in the chunk text, plus the gold speakers
of the PDNC quotes located in the chunk.
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections import defaultdict
from pathlib import Path

from .. import store
from ..chunks import chunk_sentences
from ..memory.db import CharacterDB, norm, slug
from .pdnc import Stream, _story_items, align, load_gold

GENDER = {"M": "male", "F": "female"}


def _copy_book(src: Path, out: Path) -> None:
    out.mkdir(parents=True)
    for f in ("source.epub", "config.yaml", "cast.yaml", "sections.yaml", "lexicon.yaml"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    for d in ("src", "base", "seg"):
        if (src / "work" / d).exists():
            shutil.copytree(src / "work" / d, out / "work" / d)
    a = sqlite3.connect(src / "project.db")
    b = sqlite3.connect(out / "project.db")
    a.backup(b)
    a.close()
    b.close()


def make_oracle_book(src: Path, out: Path, pdnc_dir: Path, overrides: Path | None = None, log=print) -> dict:
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first")
    _copy_book(src, out)
    con = store.connect(out)
    gold, quotes, applied, ambiguous = load_gold(pdnc_dir, overrides)

    for t in ("character", "alias", "alias_fts", "character_event", "rollback", "warning", "llm_call",
              "chunk_metrics", "run"):
        con.execute(f"DELETE FROM {t}")
    con.execute("UPDATE chunk SET pass1='done', pass2='pending', characters=NULL, scene=NULL")
    con.execute("DELETE FROM utterance WHERE source != 'default'")

    db = CharacterDB(con, run_id="oracle")
    ids = {}
    for g in gold.values():
        cid = slug(g.name)
        while cid in ids.values():
            cid += "_"
        ids[g.name] = cid
        db.apply("create", {"id": cid, "name": g.name, "gender": GENDER.get(g.gender),
                            "profile": f"PDNC {g.category} character"}, source="manual")
        for a in sorted(g.aliases):
            if a != g.name and norm(a) not in ambiguous:
                db.apply("alias", {"id": cid, "alias": a, "strength": "strong"}, source="manual")

    # where each PDNC quote sits (sentence id -> chunk)
    st = Stream(_story_items(con))
    align(st, quotes)
    chunks = con.execute("SELECT * FROM chunk ORDER BY doc_id, seq_from").fetchall()
    chunk_of = {}
    for c in chunks:
        for s in chunk_sentences(con, c):
            chunk_of[s["id"]] = c["id"]
    speakers = defaultdict(set)
    for q in quotes:
        if q["sentence"] in chunk_of and q["speaker"] in ids:
            speakers[chunk_of[q["sentence"]]].add(ids[q["speaker"]])
    pats = [(ids[g.name], re.compile(r"(?<![A-Za-z])" + re.escape(a) + r"(?![A-Za-z])"))
            for g in gold.values() for a in g.aliases if norm(a) not in ambiguous and a[:1].isupper()]
    n_cands = []
    for c in chunks:
        text = " ".join(s["tts_text"] for s in chunk_sentences(con, c))
        present = {cid for cid, p in pats if p.search(text)} | speakers[c["id"]]
        order = sorted(present)
        con.execute("UPDATE chunk SET characters=? WHERE id=?", (json.dumps(order), c["id"]))
        n_cands.append(len(order))
    store.set_meta(con, "oracle_characters", json.dumps({"pdnc": str(pdnc_dir), "source_book": str(src),
                                                         "overrides": len(applied)}))
    con.commit()
    info = {"characters": len(ids), "chunks": len(chunks),
            "candidates_per_chunk": round(sum(n_cands) / max(len(n_cands), 1), 1),
            "speakers_added": sum(len(v) for v in speakers.values())}
    log(f"oracle book {out.name}: {info['characters']} PDNC characters, {info['chunks']} chunks marked done, "
        f"{info['candidates_per_chunk']} candidates per chunk on average")
    return info
