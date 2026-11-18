"""Stage: attribute (pass 2, F4/F6). Split sentences into utterances and give
each quotation a speaker, a delivery note and a confidence.

Per chunk (story sentences only): quotes are found by rules (pipeline/quotes.py);
the LLM only picks speakers, from a closed list (characters seen in this and
the previous chunk + recent speakers + narrator + unknown) enforced by the JSON
schema. Confidence is computed from speech tags and turn-taking
(pipeline/attrib_conf.py), not taken from the model. Chunks commit one at a
time; pending, failed and stale chunks are (re)done. Non-story sentences keep a
single narrator utterance. attribution_overrides.csv (quote_id,speaker,style,
note) is applied after every run and always wins.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from ..adapters.llm import LLMError, get_llm
from .. import attrib_turns
from ..attrib_conf import score as conf_score
from ..book import Book
from ..chunks import chunk_rows, chunk_sentences, ensure_chunks, scope_docs, story_sentences
from ..memory import prompts
from ..memory.db import CharacterDB
from ..quotes import detect_style, split_paragraph
from ..store import get_meta, set_meta

LAST_SPEAKERS = 6


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None, llm=None, obs=None) -> None:
    from ..observe import Observer
    obs = obs or Observer.null(log)
    mon = cfg.get("monitor", {})
    docs = scope_docs(con, doc_ids)
    ensure_chunks(con, docs, cfg["character_memory"]["chunk_words"])
    chunks = chunk_rows(con, docs)
    not_ready = [c["id"] for c in chunks if c["pass1"] != "done"]
    if not_ready:
        raise SystemExit(f"attribute: run characters first (pending: {', '.join(not_ready[:5])})")
    style = get_meta(con, "quote_style")
    if not style:
        style = detect_style([r["tts_text"] for r in con.execute("SELECT tts_text FROM sentence")])
        set_meta(con, "quote_style", style)
    pending = [c for c in chunks if c["pass2"] != "done"]
    stale = sum(1 for c in pending if c["pass2"] == "stale")
    log(f"attribute: {len(chunks)} chunks in scope, {len(pending)} to do ({stale} stale) (quote style: {style})")
    db = CharacterDB(con)
    from ..memory import relgraph
    # RELATION_ANOMALY (docs/DESIGN.md): a broken graph is worse than none
    kg = None if get_meta(con, "relations_disabled") == "1" else relgraph.graph_from_db(con, db.resolve)
    card_rel = bool(cfg.get("attribute", {}).get("card_relations", True))
    if kg is None:
        log("attribute: relation graph disabled (RELATION_ANOMALY): no relations on cards, kin tags not resolved")
    if pending:
        llm = llm or get_llm(cfg["llm"], mon)
        llm.obs = obs
        failed, streak = [], 0
        for c in pending:
            if _try_chunk(db, llm, con, c, chunks, style, mon, obs, kg, card_rel):
                streak = 0
            else:
                failed.append(c)
                streak += 1
                if streak >= mon.get("max_consecutive_failures", 3):
                    obs.pause("LLM_DOWN", c["id"], f"{streak} chunks in a row failed; check the LLM service")
                obs.checkpoint()
        for c in failed:
            _try_chunk(db, llm, con, c, chunks, style, mon, obs, kg, card_rel)
        obs.chunk_id = None
    n = apply_overrides(book, con)
    if n:
        log(f"attribute: applied {n} manual overrides from attribution_overrides.csv")
    # Two people talking in turn: the paragraph position decides the untagged quotes (after the manual
    # overrides, which count as reliable speakers; docs/DESIGN.md).
    mode = cfg.get("attribute", {}).get("turns", "fix")
    st = attrib_turns.apply(con, docs, mode)
    if st.get("agree") or st.get("fixed") or st.get("flagged"):
        log(f"attribute: turn-taking rule ({mode}): confirmed {st.get('agree', 0)}, "
            f"changed {st.get('fixed', 0)}, flagged {st.get('flagged', 0)}")
    _summary(con, docs, log)


def _try_chunk(db, llm, con, c, chunks, style, mon, obs, kg=None, card_rel=True) -> bool:
    obs.chunk_id = c["id"]
    try:
        _process_chunk(db, llm, con, c, chunks, style, mon, obs, kg, card_rel)
    except LLMError as e:
        con.rollback()
        con.execute("UPDATE chunk SET pass2='failed' WHERE id=?", (c["id"],))
        con.commit()
        obs.flush()
        obs.warn("LLM_FAILED", "warn", c["id"], f"attribute: {type(e).__name__}: {e}"[:300])
        obs.flush()          # the caller checks LLM_DOWN first, then obs.checkpoint()
        return False
    obs.checkpoint()
    return True


def _pieces_for_chunk(con, c, style):
    """[(sentence row, [Piece])] for the story sentences of the chunk, quote state reset per paragraph."""
    sents = story_sentences(con, c)
    by_para = defaultdict(list)
    for s in sents:
        by_para[s["para"]].append(s)
    out = []
    for p in sorted(by_para):
        ss = by_para[p]
        key = f"{c['doc_id']:03d}p{p:04d}"
        for s, pcs in zip(ss, split_paragraph([s["tts_text"] for s in ss], style, key)):
            out.append((s, pcs))
    return out


REL_WORDS = [  # (relation of the card to z, female word, male word, neutral word)
    ("spouse", "wife", "husband", "spouse"), ("child", "mother", "father", "parent"),
    ("parent", "daughter", "son", "child"), ("sibling", "sister", "brother", "sibling"),
    ("niece_nephew", "aunt", "uncle", "aunt/uncle"), ("aunt_uncle", "niece", "nephew", "niece/nephew"),
    ("in_law", "sister-in-law", "brother-in-law", "in-law"), ("cousin", "cousin", "cousin", "cousin"),
    ("friend", "friend", "friend", "friend")]


def relations_text(kg, cid: str, gender: str | None, others: dict[str, str]) -> str:
    """'wife of C1; mother of C5, C6' for the people on the scene (others: card id -> code).
    Strong relations only (5.6 point 3): the model reads these as facts."""
    if kg is None:
        return ""
    parts = []
    for rel, fw, mw, nw in REL_WORDS:
        who = [others[z] for z in sorted(kg.relation(cid, rel, weak=False)) if z in others]
        if who:
            word = fw if gender == "female" else mw if gender == "male" else nw
            parts.append(f"{word} of {', '.join(who)}")
    return "; ".join(parts)


def _process_chunk(db: CharacterDB, llm, con, c, chunks, style, mon, obs, kg=None, card_rel=True) -> None:
    items = _pieces_for_chunk(con, c, style)
    story_ids = {s["id"] for s, _ in items}
    # Sentences of this chunk outside story sections: one narrator utterance each.
    for s in chunk_sentences(con, c):
        if s["id"] not in story_ids:
            con.execute("DELETE FROM utterance WHERE sentence_id=? AND source != 'default'", (s["id"],))
            con.execute("INSERT OR IGNORE INTO utterance(id, sentence_id, seq, start, end, tts_text) "
                        "VALUES (?,?,0,0,?,?)", (f"{s['id']}u0", s["id"], len(s["tts_text"]), s["tts_text"]))
    qids = []
    for _, pcs in items:
        for p in pcs:
            if p.is_quote and p.quote_id not in qids:
                qids.append(p.quote_id)
    label = {q: f"Q{i + 1}" for i, q in enumerate(qids)}

    idx = [x["id"] for x in chunks].index(c["id"])
    prev_scene = {}
    if idx > 0:   # re-read: the previous chunk may have been processed in this same run
        prev_scene = json.loads(con.execute("SELECT scene FROM chunk WHERE id=?", (chunks[idx - 1]["id"],))
                                .fetchone()["scene"] or "{}")
    last_speakers = [db.resolve(x) for x in prev_scene.get("last_speakers", [])]

    decisions: dict[str, dict] = {}
    kin_stats: dict = {}
    if qids:
        cast_ids: list[str] = []
        for ch in chunks[max(0, idx - 1): idx + 1]:
            row = con.execute("SELECT characters FROM chunk WHERE id=?", (ch["id"],)).fetchone()
            for cid in json.loads(row["characters"] or "[]") + last_speakers:
                cid = db.resolve(cid)
                if db.exists(cid) and cid not in cast_ids:
                    cast_ids.append(cid)
        # Neutral codes C1..Cn: the model never sees our ids (it preferred clean-looking ids such as
        # mrs_hurst over unknown_mrs_bennet_c001003; docs/DESIGN.md).
        code = {cid: f"C{i + 1}" for i, cid in enumerate(cast_ids)}
        uncode = {v: k for k, v in code.items()}
        cards = [db.card(cid).coded(code[cid], relations_text(kg if card_rel else None, cid, db.card(cid).gender,
                                                               {z: code[z] for z in cast_ids if z != cid}))
                 for cid in cast_ids]
        names = {code[cid]: db.card(cid).aliases for cid in cast_ids}
        marked = _marked_text(items, label)
        speakers = list(uncode) + ["narrator", "unknown"]
        msgs, schema, payload = prompts.attribute(marked, list(label.values()), cards, speakers,
                                                  [code[x] for x in last_speakers if x in code], names)
        out = llm.complete_json(msgs, schema, "attribute_quotes", payload, max_tokens=6000)
        inv = {v: k for k, v in label.items()}
        for d in out["quotes"]:
            if d["q"] in inv:
                decisions[inv[d["q"]]] = {**d, "speaker": uncode.get(d["speaker"], d["speaker"])}
        # Computed confidence (speech tags, turn-taking); the model's own number is not used.
        ctx = _quote_context(items, qids)
        conf = conf_score(db, c["doc_id"], [{"speaker": decisions.get(q, {}).get("speaker", "unknown"), **ctx[q]}
                                            for q in qids], kin={"graph": kg, "scene": cast_ids} if kg else None,
                          stats=kin_stats)
        for q, v in zip(qids, conf):
            if q in decisions:
                decisions[q] = {**decisions[q], "confidence": v}

    rows = []
    for s, pcs in items:
        if not pcs:
            pcs_rows = [(f"{s['id']}u0", s["id"], 0, 0, len(s["tts_text"]), s["tts_text"], 0, None,
                         "narrator", None, None, "default", None, None)]
        else:
            pcs_rows = []
            for k, p in enumerate(pcs):
                if p.is_quote:
                    d = decisions.get(p.quote_id)
                    spk = d["speaker"] if d else "unknown"
                    rows_v = (spk, (d or {}).get("style") or None, (d or {}).get("confidence"), "model",
                              spk, (d or {}).get("confidence"))
                else:
                    rows_v = ("narrator", None, None, "rule", None, None)
                pcs_rows.append((f"{s['id']}u{k}", s["id"], k, p.start, p.end, p.text, int(p.is_quote),
                                 p.quote_id, *rows_v))
        rows += pcs_rows
        con.execute("DELETE FROM utterance WHERE sentence_id=?", (s["id"],))
    con.executemany("INSERT INTO utterance(id, sentence_id, seq, start, end, tts_text, is_quote, quote_id, "
                    "speaker_id, style, confidence, source, model_speaker, model_confidence) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    spoken = [decisions[q]["speaker"] for q in qids if q in decisions
              and decisions[q]["speaker"] not in ("narrator", "unknown")]
    scene = {"last_speakers": (last_speakers + spoken)[-LAST_SPEAKERS:]}
    con.execute("UPDATE chunk SET pass2='done', scene=? WHERE id=?", (json.dumps(scene), c["id"]))
    con.commit()
    n_unknown = sum(1 for q in qids if decisions.get(q, {}).get("speaker", "unknown") == "unknown")
    low = sum(1 for q in qids if q in decisions and decisions[q]["confidence"] < 0.6)
    who = sorted({d["speaker"] for d in decisions.values()})
    obs.metric(c["id"], 2, {"quotes": len(qids), "unknown": n_unknown, "low_confidence": low, "speakers": len(who),
                            **kin_stats})
    obs.log(f"attribute: {c['id']} {len(qids)} quotes -> {', '.join(who) or '-'}"
            + (f"  ({n_unknown} unknown, {low} low confidence)" if n_unknown or low else ""), chunk_id=c["id"])
    if len(qids) >= 5 and n_unknown / len(qids) > mon.get("unknown_max", 0.2):
        obs.warn("ATTRIB_UNKNOWN", "warn", c["id"], f"{n_unknown}/{len(qids)} quotes have no speaker",
                 unknown=n_unknown, quotes=len(qids))


def _quote_context(items, qids) -> dict:
    """Per quote id: narration right after its last piece / right before its first piece, and paragraph.
    `between`: narration right before the quote that itself follows another quote in the same paragraph
    (“…,” said Elizabeth, “…”): a speech tag there belongs to both quotes."""
    ctx = {q: {"after": "", "before": "", "between": "", "para": None} for q in qids}
    flat = [(s, p) for s, pcs in items for p in pcs]
    for i, (s, p) in enumerate(flat):
        if not p.is_quote:
            continue
        c = ctx[p.quote_id]
        c["para"] = s["para"]
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        if nxt and not nxt[1].is_quote and nxt[0]["para"] == s["para"] and not c["after"]:
            c["after"] = nxt[1].text
        prv = flat[i - 1] if i > 0 else None
        if prv and not prv[1].is_quote and prv[0]["para"] == s["para"] and not c["before"]:
            c["before"] = prv[1].text
            pp = flat[i - 2] if i > 1 else None
            if pp and pp[1].is_quote and pp[0]["para"] == s["para"]:
                c["between"] = prv[1].text
    return ctx


def _marked_text(items, label) -> str:
    out, prev_para = [], None
    for s, pcs in items:
        if prev_para is not None:
            out.append("\n\n" if s["para"] != prev_para else " ")
        prev_para = s["para"]
        t, buf, cur = s["tts_text"], [], 0
        for p in pcs:
            if p.is_quote:
                buf.append(t[cur:p.start])
                buf.append(f"[{label[p.quote_id]}]{t[p.start:p.end]}[/{label[p.quote_id]}]")
                cur = p.end
        buf.append(t[cur:])
        out.append("".join(buf))
    return "".join(out)


def overrides_path(book: Book):
    return book.dir / "attribution_overrides.csv"


def apply_overrides(book: Book, con) -> int:
    p = overrides_path(book)
    if not p.exists():
        return 0
    n = 0
    with p.open(newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("quote_id") or "").strip()
            if not q or q.startswith("#"):
                continue
            sets, vals = ["source='manual'", "confidence=1.0"], []
            if (row.get("speaker") or "").strip():
                sets.append("speaker_id=?")
                vals.append(row["speaker"].strip())
            if (row.get("style") or "").strip():
                sets.append("style=?")
                vals.append(row["style"].strip())
            n += con.execute(f"UPDATE utterance SET {', '.join(sets)} WHERE quote_id=?", (*vals, q)).rowcount
    con.commit()
    return n


def _summary(con, docs, log) -> None:
    q = ",".join("?" * len(docs))
    rows = con.execute(
        f"SELECT u.speaker_id, COUNT(*) n FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
        f"WHERE s.doc_id IN ({q}) AND u.is_quote=1 GROUP BY u.speaker_id ORDER BY n DESC", docs).fetchall()
    if rows:
        log("attribute: quote pieces by speaker: " + ", ".join(f"{r['speaker_id']} {r['n']}" for r in rows))
