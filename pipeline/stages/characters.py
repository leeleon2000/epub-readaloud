"""Stage: characters (pass 1, F4). Build the character list chunk by chunk.

Per chunk (story sentences only, see classify): (1) the LLM extracts mentions,
(2) retrieval finds candidate cards, (3) the LLM resolves mentions in batches;
every proposal passes the validation layer before it is applied. Each chunk
commits on its own (checkpoint). Failures are isolated per chunk (LLM_FAILED,
retried once at the end; LLM_DOWN pauses after several in a row) and
anomalies raise warnings; CHAR_SPIKE pauses the run after the chunk commits.
After all chunks: a merge scan, characters.yaml and reports/characters_report.md.
"""
from __future__ import annotations

import json
import time
from itertools import combinations

import yaml
from rapidfuzz import fuzz

from ..adapters.llm import LLMError, LLMTruncated, get_llm
from ..book import Book
from ..chunks import chunk_rows, chunk_text, ensure_chunks, scope_docs, story_sentences
from ..memory import prompts
from ..memory.db import CharacterDB, norm, norm_core
from ..memory.retrieve import candidates
from ..memory.validate import Validator, is_proper_name

STRENGTH = {"name": "strong", "title": "medium", "description": "weak", "relational": "weak"}
PRONOUNS = {"i", "me", "my", "myself", "you", "your", "yourself", "he", "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "itself", "we", "us", "our", "they", "them", "their", "themselves",
            "one", "someone", "somebody", "anyone", "anybody", "nobody", "everyone", "everybody"}


def _approx_tokens(lines: list[str]) -> int:
    return sum(len(x) for x in lines) // 4


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None, llm=None, obs=None) -> None:
    from ..observe import Observer
    obs = obs or Observer.null(log)
    mcfg, mon = cfg["character_memory"], cfg.get("monitor", {})
    docs = scope_docs(con, doc_ids)
    ensure_chunks(con, docs, mcfg["chunk_words"])
    chunks = chunk_rows(con, docs)
    pending = [c for c in chunks if c["pass1"] != "done"]
    log(f"characters: {len(chunks)} chunks in scope, {len(pending)} pending")
    db = CharacterDB(con, profile_max_chars=mcfg["profile_max_tokens"] * 4, run_id=obs.run_id)
    if pending:
        llm = llm or get_llm(cfg["llm"], mon)
        llm.obs = obs
        failed, streak = [], 0
        for c in pending:
            if _try_chunk(db, llm, con, c, chunks, mcfg, mon, obs):
                streak = 0
            else:
                failed.append(c)
                streak += 1
                if streak >= mon.get("max_consecutive_failures", 3):
                    obs.pause("LLM_DOWN", c["id"], f"{streak} chunks in a row failed; check the LLM service")
                obs.checkpoint()
        for c in failed:            # one more try once the pass is through
            row = con.execute("SELECT * FROM chunk WHERE id=?", (c["id"],)).fetchone()
            _try_chunk(db, llm, con, row, chunks, mcfg, mon, obs)
        obs.chunk_id = None
        _merge_scan(db, llm, con, mcfg, mon, obs, log)
    export_yaml(book, db)
    write_report(book, con, db)
    n = len(db.active_ids())
    left = con.execute(f"SELECT COUNT(*) FROM chunk WHERE pass1='failed' AND doc_id IN ({','.join('?' * len(docs))})",
                       docs).fetchone()[0]
    log(f"characters: {n} characters" + (f", {left} chunks FAILED (rerun to retry)" if left else "")
        + f"; review {book.dir.name}/characters.yaml (or `readaloud characters {book.id}`)")


def _try_chunk(db, llm, con, c, chunks, mcfg, mon, obs) -> bool:
    """Process one chunk; on an LLM failure roll the chunk back, mark it failed and go on."""
    obs.chunk_id = c["id"]
    try:
        new_chars = _process_chunk(db, llm, con, c, chunks, mcfg, mon, obs)
    except LLMError as e:
        con.rollback()
        con.execute("UPDATE chunk SET pass1='failed' WHERE id=?", (c["id"],))
        con.commit()
        obs.flush()
        obs.warn("LLM_FAILED", "warn", c["id"], f"characters: {type(e).__name__}: {e}"[:300])
        obs.flush()          # the caller checks LLM_DOWN first, then obs.checkpoint()
        return False
    obs.checkpoint()
    if new_chars > mon.get("new_chars_per_chunk_max", 8):
        sample = [s["tts_text"][:120] for s in story_sentences(con, c)[:3]]
        obs.pause("CHAR_SPIKE", c["id"], f"{new_chars} new characters in one chunk", new_characters=new_chars,
                  opening=sample)
    _growth_check(con, c, mon, obs)
    return True


def _recent(con, chunks, c, k: int) -> list[str]:
    idx = [x["id"] for x in chunks].index(c["id"])
    out: list[str] = []
    for prev in reversed(chunks[max(0, idx - k):idx]):
        r = con.execute("SELECT characters FROM chunk WHERE id=?", (prev["id"],)).fetchone()
        for cid in json.loads(r["characters"] or "[]"):
            if cid not in out:
                out.append(cid)
    return out


def _process_chunk(db: CharacterDB, llm, con, c, chunks, mcfg, mon, obs) -> int:
    """Returns the number of characters created in this chunk."""
    t0 = time.time()
    sents = story_sentences(con, c)
    text = chunk_text(sents)
    m = {"mentions": 0, "existing": 0, "new": 0, "provisional": 0, "anonymous": 0, "rejected": {},
         "merges": 0, "truncations": 0, "new_chars": 0, "story_sentences": len(sents)}
    seen: list[str] = []
    mentions = []
    val = Validator(db, obs, c["id"], text, mon)
    if text.strip():
        recent = [db.resolve(x) for x in _recent(con, chunks, c, mcfg["recent_chunks"])]
        msgs, schema, payload = prompts.extract(text)
        keys = set()
        for x in llm.complete_json(msgs, schema, "extract_mentions", payload, max_tokens=3000)["mentions"]:
            t = x["text"].strip()
            if not t or t.lower() in PRONOUNS or norm(t) in keys:
                continue
            keys.add(norm(t))
            mentions.append({**x, "text": t})
        m["mentions"] = len(mentions)
        bs = int(mcfg.get("resolve_batch", 10))
        for k in range(0, len(mentions), bs):
            _resolve_batch(db, llm, c, text, mentions[k:k + bs], recent, mcfg, seen, val, m, obs)
        val.finish_chunk()
    m["rejected"] = dict(val.rejects)
    m["seconds"] = round(time.time() - t0, 1)
    con.execute("UPDATE chunk SET pass1='done', characters=? WHERE id=?",
                (json.dumps([db.resolve(x) for x in seen]), c["id"]))
    con.commit()     # a chunk is atomic: an error before this rolls the whole chunk back
    obs.metric(c["id"], 1, m)
    rej = sum(m["rejected"].values())
    obs.log(f"characters: {c['id']} {m['mentions']} mentions -> {m['existing']} known, {m['new']} new, "
            f"{m['provisional']} provisional, {rej} rejected  ({m['seconds']:.0f}s)"
            + (f"  new: {', '.join(m['created'])}" if m.get("created") else ""), chunk_id=c["id"])
    return m["new_chars"]


def _resolve_batch(db, llm, c, text, mentions, recent, mcfg, seen, val, m, obs) -> None:
    """Resolve a batch of mentions; on output truncation split the batch in half."""
    roster = [cd.compact() for cd in db.roster()]
    if _approx_tokens(roster) > mcfg["roster_token_budget"]:
        roster = []          # too big to send whole; rely on per-mention candidates
    cand_ids: list[str] = []
    for x in mentions:
        for cid, _ in candidates(db, x["text"], c["doc_id"], recent, mcfg["max_candidates"]):
            if cid not in cand_ids and db.is_real(cid):
                cand_ids.append(cid)
    cards = [db.card(cid).full() for cid in cand_ids[: mcfg["max_candidates"]]]
    msgs, schema, payload = prompts.resolve(text, mentions, roster, cards, recent, mcfg["profile_max_tokens"] * 4)
    try:
        out = llm.complete_json(msgs, schema, "resolve_mentions", payload, max_tokens=4000)
    except LLMTruncated:
        if len(mentions) == 1:
            raise
        m["truncations"] += 1
        obs.warn("TRUNCATED", "info", c["id"], f"resolve output truncated; splitting {len(mentions)} mentions")
        half = len(mentions) // 2
        _resolve_batch(db, llm, c, text, mentions[:half], recent, mcfg, seen, val, m, obs)
        _resolve_batch(db, llm, c, text, mentions[half:], recent, mcfg, seen, val, m, obs)
        return
    resolved: dict[int, str] = {}
    valid = [r for r in out["resolutions"] if 0 <= r["mention"] < len(mentions)]
    # Independent resolutions first, then mentions that point at another mention (same_as).
    for res in sorted(valid, key=lambda r: (r.get("same_as", -1) >= 0, r["mention"])):
        i, j = res["mention"], res.get("same_as", -1)
        if j >= 0 and j != i and j in resolved:
            res = {**res, "action": "existing", "character_id": resolved[j]}
        anonymous = res["action"] == "anonymous" or mentions[i].get("kind") == "generic"
        res = val.check_resolution(res, mentions[i])
        if res is None:          # anonymous, or refused by validation (counted in val.rejects)
            m["anonymous"] += int(anonymous)
            continue
        cid = _apply_resolution(db, res, mentions[i], c, val, m)
        if cid:
            resolved[i] = cid
            if cid not in seen:
                seen.append(cid)
    for mg in out.get("merges", []):
        a, b = db.resolve(mg["from_id"]), db.resolve(mg["into_id"])
        if db.is_real(a) and db.is_real(b) and a != b and val.merge_ok(a, b, None, mg["reason"]):
            db.apply("merge", {"from": a, "into": b, "reason": mg["reason"]}, chunk_id=c["id"])
            m["merges"] += 1
            obs.log(f"characters: merged {a} -> {b} ({mg['reason']})", chunk_id=c["id"])


def _strength(x: dict) -> str:
    """'Mr. Bingley' is a real name even when the model calls it a title; 'the Queen' is not."""
    if x["kind"] in ("name", "title") and is_proper_name(x["text"]):
        return "strong"
    return STRENGTH.get(x["kind"], "weak")


def _apply_resolution(db: CharacterDB, res: dict, x: dict, c, val: Validator, m: dict) -> str | None:
    action = res["action"]
    strength = _strength(x)
    scope = {"scope_from": c["doc_id"], "scope_to": c["doc_id"]} if strength == "weak" else {}
    if action == "existing":
        cid = res["character_id"]
        if val.alias_ok(cid, x["text"], strength):
            db.apply("alias", {"id": cid, "alias": x["text"], "strength": strength, **scope}, chunk_id=c["id"])
        db.apply("mention", {"id": cid}, chunk_id=c["id"])
        upd = {k: res[k] for k in ("profile", "voice_description") if res.get(k)}
        upd.update(val.attrs(cid, res))
        # A provisional card that is now called by a real name in the text becomes a full character.
        card = db.card(cid)
        if card.status == "provisional" and strength == "strong" and val.grounded(x["text"]):
            upd["status"] = "active"
            if not is_proper_name(card.name):
                upd["name"] = x["text"]
        if upd:
            db.apply("profile", {"id": cid, **upd}, chunk_id=c["id"])
        m["existing"] += 1
        return cid
    name = (res.get("name") or x["text"]).strip()
    cid = db.new_id(name, provisional_tag=c["id"].replace("-", "") if action == "provisional" else None)
    if not val.alias_ok(cid, name, "strong" if action == "new" and is_proper_name(name) else "weak"):
        val.reject("alias_conflict", mention=x["text"], name=name)
        return None
    db.apply("create", {"id": cid, "name": name, "gender": res.get("gender"), "age": res.get("age"),
                        "accent": res.get("accent"), "profile": res.get("profile"),
                        "voice_description": res.get("voice_description"), "first_seen": c["id"],
                        "status": "provisional" if action == "provisional" else "active"}, chunk_id=c["id"])
    if norm_core(x["text"]) != norm_core(name) and val.alias_ok(cid, x["text"], strength):
        db.apply("alias", {"id": cid, "alias": x["text"], "strength": strength, **scope}, chunk_id=c["id"])
    db.apply("mention", {"id": cid}, chunk_id=c["id"])
    m["provisional" if action == "provisional" else "new"] += 1
    m["new_chars"] += 1
    m.setdefault("created", []).append(cid)
    return cid


def _growth_check(con, c, mon, obs) -> None:
    """CHAR_GROWTH: new characters per chunk should fall as the book goes on."""
    k = int(mon.get("growth_window_chunks", 10))
    rows = con.execute("SELECT m.metrics FROM chunk ch JOIN chunk_metrics m ON m.chunk_id=ch.id AND m.pass=1 "
                       "AND m.run_id=(SELECT MAX(run_id) FROM chunk_metrics x WHERE x.chunk_id=ch.id AND x.pass=1) "
                       "WHERE ch.pass1='done' ORDER BY ch.doc_id, ch.seq_from").fetchall()
    series = [json.loads(r["metrics"]).get("new_chars", 0) for r in rows]
    if len(series) < 2 * k or len(series) % k:
        return
    last, prev = sum(series[-k:]), sum(series[-2 * k:-k])
    if last >= prev and last >= k // 2:
        obs.warn("CHAR_GROWTH", "warn", c["id"], f"new characters in the last {k} chunks: {last} (previous {k}: {prev})",
                 last=last, previous=prev)


def _merge_scan(db: CharacterDB, llm, con, mcfg, mon, obs, log) -> None:
    """Ask about likely duplicates: shared name token, overlapping aliases, or similar names."""
    cards = db.roster()
    asked = {tuple(sorted(json.loads(r["payload"])["pair"])) for r in con.execute(
        "SELECT payload FROM character_event WHERE type='edit' AND superseded_by IS NULL "
        "AND payload LIKE '%\"pair\"%'")}
    val = Validator(db, obs, None, "", mon)
    merged = 0
    for a, b in combinations(cards, 2):
        pair = tuple(sorted((a.id, b.id)))
        if pair in asked or not (db.is_real(a.id) and db.is_real(b.id)):
            continue
        ta = {w for al in a.aliases for w in norm_core(al).split() if len(w) > 2}
        tb = {w for al in b.aliases for w in norm_core(al).split() if len(w) > 2}
        if not (ta & tb or fuzz.WRatio(norm_core(a.name), norm_core(b.name)) >= 88):
            continue
        msgs, schema, payload = prompts.same_person(a.full(), b.full(), _evidence(con, a), _evidence(con, b))
        try:
            out = llm.complete_json(msgs, schema, "same_person", payload, max_tokens=600)
        except LLMError as e:
            obs.warn("LLM_FAILED", "warn", None, f"merge scan {pair}: {e}"[:300])
            continue
        db.apply("edit", {"pair": list(pair), "same": out["same"], "confidence": out["confidence"],
                          "reason": out["reason"]})
        if out["same"]:
            keep, drop = (a, b) if a.mentions >= b.mentions else (b, a)
            if val.merge_ok(drop.id, keep.id, out["confidence"], out["reason"]):
                db.apply("merge", {"from": drop.id, "into": keep.id, "reason": out["reason"]})
                merged += 1
                log(f"characters: merge scan merged {drop.id} -> {keep.id} ({out['reason']})")
        con.commit()
        obs.checkpoint()
    if merged:
        from ..memory.rollback import mark_stale
        mark_stale(con)
        con.commit()


def _evidence(con, card, limit: int = 2, max_chars: int = 600) -> str:
    """A few sentences from processed chunks that mention the character (for the merge check)."""
    names = sorted({a for a in card.aliases if len(a) > 2}, key=len, reverse=True)
    from .classify import story_filter_sql
    rows = con.execute("SELECT s.tts_text FROM sentence s JOIN chunk c ON c.doc_id=s.doc_id "
                       "AND s.seq BETWEEN c.seq_from AND c.seq_to WHERE c.pass1='done' AND "
                       + story_filter_sql("s") + " ORDER BY s.doc_id, s.seq")
    out = []
    for r in rows:
        t = r["tts_text"]
        if any(n.lower() in t.lower() for n in names):
            out.append(t[:300])
            if len(out) >= limit:
                break
    return "\n".join(out)[:max_chars]




def reset(book: Book, con, run_id: str | None = None) -> tuple[str, list]:
    """Void ALL model character events (restorable with `--restore`); manual events are kept and replayed."""
    from ..memory.rollback import chunk_range, rollback
    return rollback(con, chunk_range(con, None), run_id=run_id, note="reset")


def export_yaml(book: Book, db: CharacterDB) -> None:
    out = []
    for cd in db.roster():
        out.append({"id": cd.id, "name": cd.name, "aliases": [a for a in cd.aliases if a != cd.name],
                    "gender": cd.gender, "age": cd.age, "accent": cd.accent, "profile": cd.profile,
                    "voice_description": cd.voice_description, "mentions": cd.mentions, "status": cd.status})
    (book.dir / "characters.yaml").write_text(
        "# Generated from project.db. Edit with `readaloud characters` (edits are recorded as events).\n"
        + yaml.safe_dump(out, allow_unicode=True, sort_keys=False, width=100))


def write_report(book: Book, con, db: CharacterDB) -> None:
    """reports/characters_report.md: growth per chunk, top characters, review items, open warnings."""
    rows = con.execute("SELECT ch.id, m.metrics FROM chunk ch LEFT JOIN chunk_metrics m ON m.chunk_id=ch.id "
                       "AND m.pass=1 AND m.run_id=(SELECT MAX(run_id) FROM chunk_metrics x WHERE x.chunk_id=ch.id "
                       "AND x.pass=1) WHERE ch.pass1='done' ORDER BY ch.doc_id, ch.seq_from").fetchall()
    lines = [f"# Characters report: {book.id}", "", "## Growth (new characters per chunk, cumulative)", "",
             "| chunk | mentions | new | provisional | rejected | cumulative |", "|---|---|---|---|---|---|"]
    total = 0
    for r in rows:
        mm = json.loads(r["metrics"]) if r["metrics"] else {}
        total += mm.get("new_chars", 0)
        lines.append(f"| {r['id']} | {mm.get('mentions', '-')} | {mm.get('new', '-')} | {mm.get('provisional', '-')} "
                     f"| {sum((mm.get('rejected') or {}).values()) if mm else '-'} | {total} |")
    lines += ["", "## Top 20 characters", "", "| id | name | mentions | gender | age | status |", "|---|---|---|---|---|---|"]
    for cd in db.roster()[:20]:
        lines.append(f"| {cd.id} | {cd.name} | {cd.mentions} | {cd.gender} | {cd.age} | {cd.status} |")
    warns = con.execute("SELECT * FROM warning WHERE status='open' ORDER BY id").fetchall()
    lines += ["", f"## Open warnings ({len(warns)})", ""]
    for w in warns:
        d = json.loads(w["details"] or "{}")
        lines.append(f"- #{w['id']} `{w['code']}` [{w['severity']}] {w['chunk_id'] or ''} {d.get('message', '')}")
    prov = [cd for cd in db.roster() if cd.status == "provisional"]
    lines += ["", f"## Provisional characters ({len(prov)})", ""] + [f"- {cd.id}: {cd.name}" for cd in prov]
    book.reports.mkdir(parents=True, exist_ok=True)
    (book.reports / "characters_report.md").write_text("\n".join(lines) + "\n")
