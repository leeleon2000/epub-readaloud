"""Recoverable rollback of character-pass results (error handling, docs/DESIGN.md).

Nothing is deleted: model events of the rolled-back chunks get
`superseded_by = <rollback id>`, the character tables are rebuilt by replaying
the events still in effect, and the chunks go back to pending. Manual events
are never superseded. `restore()` undoes a rollback (and voids whatever the
re-run produced for those chunks in the meantime).
"""
from __future__ import annotations

import datetime as dt
import json
import random

from .db import CharacterDB


def _rid(prefix: str = "rb") -> str:
    return f"{prefix}-{dt.datetime.now():%Y%m%d-%H%M%S}-{random.randrange(16**3):03x}"


def chunk_range(con, chunk_from: str | None, chunk_to: str | None = None) -> list[str]:
    ids = [r["id"] for r in con.execute("SELECT id FROM chunk ORDER BY doc_id, seq_from")]
    if chunk_from is None:
        return ids
    if chunk_from not in ids or (chunk_to and chunk_to not in ids):
        raise SystemExit(f"unknown chunk id: {chunk_from if chunk_from not in ids else chunk_to}")
    a = ids.index(chunk_from)
    b = ids.index(chunk_to) if chunk_to else len(ids) - 1
    return ids[a:b + 1]


def rollback(con, chunk_ids: list[str], run_id: str | None = None, note: str = "",
             include_chunkless: bool = True) -> tuple[str, list[dict]]:
    """Void model events of these chunks (+ later chunk-less model events such as merge-scan
    results). Returns (rollback id, orphaned manual events)."""
    rid = _rid()
    mark = con.execute("SELECT IFNULL(MAX(id), 0) FROM character_event").fetchone()[0]
    q = ",".join("?" * len(chunk_ids))
    first = con.execute(f"SELECT MIN(id) FROM character_event WHERE source='model' AND superseded_by IS NULL "
                        f"AND chunk_id IN ({q})", chunk_ids).fetchone()[0]
    con.execute(f"UPDATE character_event SET superseded_by=? WHERE source='model' AND superseded_by IS NULL "
                f"AND chunk_id IN ({q})", (rid, *chunk_ids))
    if include_chunkless and first is not None:
        con.execute("UPDATE character_event SET superseded_by=? WHERE source='model' AND superseded_by IS NULL "
                    "AND chunk_id IS NULL AND id > ?", (rid, first))
    con.execute("INSERT INTO rollback(id, run_id, chunks, event_mark, note) VALUES (?,?,?,?,?)",
                (rid, run_id, json.dumps(chunk_ids), mark, note))
    con.execute(f"UPDATE chunk SET pass1='pending', characters=NULL WHERE id IN ({q})", chunk_ids)
    # Warnings about results that were just voided are obsolete.
    con.execute(f"UPDATE warning SET status='resolved', handled_by='rollback', note=? "
                f"WHERE status='open' AND chunk_id IN ({q})", (f"chunk rolled back by {rid}", *chunk_ids))
    orphans = CharacterDB(con).replay()
    mark_stale(con, chunk_ids=chunk_ids)
    con.commit()
    return rid, orphans


def restore(con, rid: str, run_id: str | None = None) -> tuple[str, list[dict]]:
    """Undo rollback `rid`: void model events produced for its chunks after it, revive its events."""
    rb = con.execute("SELECT * FROM rollback WHERE id=?", (rid,)).fetchone()
    if not rb:
        raise SystemExit(f"no rollback {rid}")
    if rb["restored_by"]:
        raise SystemExit(f"{rid} was already restored by {rb['restored_by']}")
    chunk_ids = json.loads(rb["chunks"])
    q = ",".join("?" * len(chunk_ids))
    rrid = _rid("rs")
    con.execute(f"UPDATE character_event SET superseded_by=? WHERE source='model' AND superseded_by IS NULL "
                f"AND id > ? AND (chunk_id IN ({q}) OR chunk_id IS NULL)", (rrid, rb["event_mark"], *chunk_ids))
    con.execute("UPDATE character_event SET superseded_by=NULL WHERE superseded_by=?", (rid,))
    con.execute("INSERT INTO rollback(id, run_id, chunks, event_mark, note) VALUES (?,?,?,?,?)",
                (rrid, run_id, rb["chunks"], con.execute("SELECT MAX(id) FROM character_event").fetchone()[0],
                 f"restore of {rid}"))
    con.execute("UPDATE rollback SET restored_by=? WHERE id=?", (rrid, rid))
    # Chunk character lists come back from the revived events.
    for cid in chunk_ids:
        seen = []
        for r in con.execute("SELECT payload, type FROM character_event WHERE chunk_id=? AND superseded_by IS NULL "
                             "AND type IN ('create','mention') ORDER BY id", (cid,)):
            x = json.loads(r["payload"])["id"]
            if x not in seen:
                seen.append(x)
        done = con.execute("SELECT 1 FROM character_event WHERE chunk_id=? AND superseded_by IS NULL", (cid,)).fetchone()
        con.execute("UPDATE chunk SET pass1=?, characters=? WHERE id=?",
                    ("done" if done or seen == [] else "pending", json.dumps(seen), cid))
    orphans = CharacterDB(con).replay()
    mark_stale(con, chunk_ids=chunk_ids)
    con.commit()
    return rrid, orphans


def mark_stale(con, chunk_ids: list[str] | None = None, char_ids: set[str] | None = None) -> int:
    """Downstream invalidation: attribution of these chunks, of chunks that involve these characters,
    and of chunks whose utterances name a character that no longer exists, becomes 'stale'."""
    db = CharacterDB(con)
    n = 0
    for c in con.execute("SELECT * FROM chunk WHERE pass2='done'").fetchall():
        hit = bool(chunk_ids) and c["id"] in chunk_ids
        speakers = {r["speaker_id"] for r in con.execute(
            "SELECT DISTINCT u.speaker_id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
            "WHERE s.doc_id=? AND s.seq BETWEEN ? AND ?", (c["doc_id"], c["seq_from"], c["seq_to"]))}
        named = speakers - {"narrator", "unknown"}
        if not hit and char_ids:
            hit = bool(named & char_ids) or bool(set(json.loads(c["characters"] or "[]")) & char_ids)
        if not hit:
            hit = any(not db.is_real(db.resolve(x)) for x in named)
        if hit:
            con.execute("UPDATE chunk SET pass2='stale' WHERE id=?", (c["id"],))
            n += 1
    return n
