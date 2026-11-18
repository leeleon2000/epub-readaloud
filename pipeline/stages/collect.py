"""Stage: collect (v2 P1, docs/DESIGN.md). Who is in each chunk, and how they are referred to.

No character cards are given to the model, so chunks are independent and run in parallel
(`character_memory.p1_concurrency`). Output per chunk goes to p1_person / p1_relation; P2 (roster)
reads it. Grounding is checked by the program, not trusted:
  - every reference must occur verbatim in the chunk (else dropped),
  - `related.of_ref` must occur verbatim,
  - a relation's evidence must be a verbatim passage of the chunk that contains both people.
If the model's output is truncated, the chunk is split in half at a paragraph boundary and each half
is collected separately (up to two levels).

Runs only when `character_memory.roster: v2`, or when asked for explicitly (`readaloud run --stages collect`).
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..adapters.llm import LLMError, LLMTruncated, get_llm
from ..book import Book
from ..chunks import chunk_rows, chunk_text, ensure_chunks, scope_docs, story_sentences
from ..memory import names, prompts
from ..memory.db import norm

METRIC_PASS = 10            # chunk_metrics.pass for P1
PRONOUNS = {"i", "me", "my", "myself", "you", "your", "yourself", "he", "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "we", "us", "our", "they", "them", "their", "themselves", "one",
            "someone", "somebody", "anyone", "anybody", "nobody", "everyone", "everybody"}
MAX_SPLIT_DEPTH = 2


def _flat(s: str) -> str:
    """For verbatim checks: curly quotes and dashes unified, whitespace collapsed."""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("—", "--").replace("–", "-")
    return " ".join(s.split())


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None, llm=None, obs=None) -> None:
    from ..observe import Observer
    obs = obs or Observer.null(log)
    mcfg, mon = cfg["character_memory"], cfg.get("monitor", {})
    docs = scope_docs(con, doc_ids)
    ensure_chunks(con, docs, mcfg["chunk_words"])
    chunks = chunk_rows(con, docs)
    pending = [c for c in chunks if c["p1"] != "done"]
    workers = int(mcfg.get("p1_concurrency", 8))
    log(f"collect: {len(chunks)} chunks in scope, {len(pending)} pending ({workers} in parallel)")
    if not pending:
        return
    llm = llm or get_llm(cfg["llm"], mon)
    llm.obs = obs
    max_tokens = int(mcfg.get("p1_max_tokens", 4000))
    work = {c["id"]: (c, story_sentences(con, c)) for c in pending}      # DB reads stay on this thread
    t0 = time.time()
    failed = _pass(con, llm, obs, work, list(work), workers, max_tokens, mon)
    if failed:                                                          # one more try once the pass is through
        _pass(con, llm, obs, work, failed, 1, max_tokens, mon, retry=True)
    obs.chunk_id = None
    n = con.execute(f"SELECT COUNT(*) FROM p1_person p JOIN chunk c ON c.id=p.chunk_id "
                    f"WHERE c.doc_id IN ({','.join('?' * len(docs))})", docs).fetchone()[0]
    left = con.execute(f"SELECT COUNT(*) FROM chunk WHERE p1='failed' AND doc_id IN ({','.join('?' * len(docs))})",
                       docs).fetchone()[0]
    log(f"collect: {n} people entries in {len(chunks)} chunks ({time.time() - t0:.0f}s)"
        + (f", {left} chunks FAILED (rerun to retry)" if left else ""))


def _pass(con, llm, obs, work, ids, workers, max_tokens, mon, retry=False) -> list[str]:
    failed, streak = [], 0
    ex = ThreadPoolExecutor(max_workers=max(1, workers))
    try:
        futs = {ex.submit(_call, llm, obs, cid, work[cid][1], max_tokens): cid for cid in ids}
        for f in as_completed(futs):
            cid = futs[f]
            c, sents = work[cid]
            obs.chunk_id = cid
            try:
                out, splits, secs = f.result()
            except LLMError as e:
                con.execute("UPDATE chunk SET p1='failed' WHERE id=?", (cid,))
                con.commit()
                obs.warn("LLM_FAILED", "warn", cid, f"collect: {type(e).__name__}: {e}"[:300])
                obs.flush()
                failed.append(cid)
                streak += 1
                if not retry and streak >= mon.get("max_consecutive_failures", 3):
                    obs.pause("LLM_DOWN", cid, f"{streak} chunks in a row failed; check the LLM service")
                obs.checkpoint()
                continue
            streak = 0
            _store(con, obs, c, sents, out, splits, secs)
            obs.checkpoint()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return failed


def _call(llm, obs, cid, sents, max_tokens):
    t0 = time.time()
    with obs.in_chunk(cid):
        out, splits = _collect(llm, sents, max_tokens, 0)
    return out, splits, time.time() - t0


def _collect(llm, sents, max_tokens, depth) -> tuple[dict, int]:
    text = chunk_text(sents)
    if not text.strip():
        return {"people": [], "relations": []}, 0
    msgs, schema, payload = prompts.collect(text)
    try:
        return llm.complete_json(msgs, schema, "collect", payload, max_tokens=max_tokens, truncation_retries=0), 0
    except LLMTruncated:
        paras = sorted({s["para"] for s in sents})
        if depth >= MAX_SPLIT_DEPTH or len(paras) < 2:
            raise
        cut = paras[len(paras) // 2]
        a, sa = _collect(llm, [s for s in sents if s["para"] < cut], max_tokens, depth + 1)
        b, sb = _collect(llm, [s for s in sents if s["para"] >= cut], max_tokens, depth + 1)
        return {"people": a["people"] + b["people"], "relations": a["relations"] + b["relations"]}, 1 + sa + sb


def ground(text: str, out: dict) -> tuple[list[dict], list[dict], dict]:
    """Keep only what is verbatim in `text`. Returns (people, relations, counts)."""
    flat = _flat(text)
    cnt = {"refs": 0, "refs_dropped": 0, "related_dropped": 0, "relations": 0, "relations_dropped": 0}
    people = []
    for p in out.get("people", []):
        refs, seen = [], set()
        for r in p.get("refs", []):
            t = r.get("text", "").strip().strip(",.;:!?\"'“”‘’")
            if not t or t.lower() in PRONOUNS or norm(t) in seen:
                continue
            if _flat(t) not in flat:
                cnt["refs_dropped"] += 1
                continue
            seen.add(norm(t))
            refs.append({"text": t, "kind": r.get("kind", "name"), "rule_kind": names.parse(t).kind})
        if not refs:
            continue
        related = []
        for r in p.get("related", []):
            if r.get("of_ref") and _flat(r["of_ref"]) in flat and r.get("relation"):
                related.append({"relation": r["relation"], "of_ref": r["of_ref"]})
            else:
                cnt["related_dropped"] += 1
        cnt["refs"] += len(refs)
        people.append({"refs": refs, "gender_cue": p.get("gender_cue", "unknown"), "related": related,
                       "traits": (p.get("traits") or "").strip()})
    rels = []
    for r in out.get("relations", []):
        ev = _flat(r.get("evidence", "").strip())
        if ev and ev in flat and _flat(r["a"]) in flat and _flat(r["b"]) in flat and r["a"] != r["b"] \
                and (_flat(r["a"]) in ev or _flat(r["b"]) in ev):
            rels.append({k: r[k] for k in ("a", "type", "b", "evidence")})
        else:
            cnt["relations_dropped"] += 1
    cnt["relations"] = len(rels)
    return people, rels, cnt


def _store(con, obs, c, sents, out, splits, secs) -> None:
    people, rels, cnt = ground(chunk_text(sents), out)
    con.execute("DELETE FROM p1_person WHERE chunk_id=?", (c["id"],))
    con.execute("DELETE FROM p1_relation WHERE chunk_id=?", (c["id"],))
    con.executemany("INSERT INTO p1_person(chunk_id, idx, refs, gender_cue, related, traits) VALUES (?,?,?,?,?,?)",
                    [(c["id"], i, json.dumps(p["refs"], ensure_ascii=False), p["gender_cue"],
                      json.dumps(p["related"], ensure_ascii=False), p["traits"]) for i, p in enumerate(people)])
    con.executemany("INSERT INTO p1_relation(chunk_id, a, type, b, evidence) VALUES (?,?,?,?,?)",
                    [(c["id"], r["a"], r["type"], r["b"], r["evidence"]) for r in rels])
    con.execute("UPDATE chunk SET p1='done' WHERE id=?", (c["id"],))
    con.commit()
    m = {"people": len(people), **cnt, "splits": splits, "seconds": round(secs, 1), "story_sentences": len(sents)}
    obs.metric(c["id"], METRIC_PASS, m)
    if cnt["refs_dropped"] or cnt["relations_dropped"]:
        obs.warn("GROUNDING_REJECT", "info", c["id"],
                 f"collect: {cnt['refs_dropped']} references, {cnt['relations_dropped']} relations not verbatim",
                 refs=cnt["refs_dropped"], relations=cnt["relations_dropped"], mention=f"collect:{c['id']}")
    obs.log(f"collect: {c['id']} {len(people)} people, {cnt['refs']} refs"
            + (f" ({cnt['refs_dropped']} dropped)" if cnt["refs_dropped"] else "")
            + f", {cnt['relations']} relations" + (f" ({cnt['relations_dropped']} dropped)" if cnt["relations_dropped"] else "")
            + (f", split {splits}x" if splits else "") + f"  ({secs:.0f}s)", chunk_id=c["id"])
