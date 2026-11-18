"""Stage: voices (attribution v3, docs/DESIGN.md). Every quotation gets a male, female or
narrator voice. Replaces collect / roster / characters / attribute when `attribute.version: v3`.

  A. rules over the whole scope: quotes, speech tags, "not dialogue" (pipeline/v3/tags.py), and the
     name -> gender table (pipeline/v3/genders.py, genders.yaml)
  B. the model, per chunk, only for quotes the rules leave open; chunks run in parallel because no state
     passes between them. Answers are kept in v3_quote, so a rerun asks only for what is missing.
  C. decide (pipeline/v3/decide.py) and write utterance rows; speaker_id is _generic_male /
     _generic_female / narrator, so cast and synth need nothing new.

Config (`voices:`): ask none | who | voice | both, second voice | notes, threshold, scene, crowd, vocative,
narrator_guard, concurrency, max_people (config.yaml explains each).

`second: notes` (docs/DESIGN.md): the second model question is not the direct voice question but
`who` again with the scene notes of the chunks around (written first, one call per chunk, kept in chunk.notes)
and a quoted piece of evidence; its answer (v3_quote.who2) is read like `who`.
attribution_overrides.csv applies afterwards (speaker: male / female / narrator).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..adapters.llm import LLMError, LLMTruncated, get_llm
from ..book import Book
from ..chunks import chunk_rows, chunk_sentences, ensure_chunks, scope_docs
from ..quotes import detect_style
from ..store import get_meta, set_meta
from ..v3 import FEMALE, MALE, NARRATOR, decide as D, genders as G, prompts as P
from ..v3.tags import find_tag, norm_phrase, not_dialogue
from .attribute import _marked_text, _pieces_for_chunk, _quote_context, apply_overrides

DEFAULTS = {"ask": "both", "threshold": 0.7, "threshold_first_person": 0.85, "scene": True, "crowd": 4, "vocative": True, "narrator_guard": True,
            "concurrency": 8, "max_people": 30, "first_person_min": 20, "narrator_name_min": 10, "second": "voice"}
SECONDS = ("voice", "notes")
ANSWER_COL = {"who": "who", "voice": "voice_ans", "who2": "who2"}     # ask -> v3_quote column that holds its answer
ANSWER_COLS = "quote_id, who, who_gender, voice_ans, who2, who2_gender, evidence, basis"


def settings(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("voices") or {})}


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None, llm=None, obs=None) -> None:
    from ..observe import Observer
    obs = obs or Observer.null(log)
    vc = settings(cfg)
    if vc["second"] not in SECONDS:
        raise ValueError(f"voices.second must be one of {SECONDS}, not {vc['second']!r}")
    mon = cfg.get("monitor", {})
    t0 = time.time()
    docs = scope_docs(con, doc_ids)
    style = get_meta(con, "quote_style")
    if not style:
        style = detect_style([r["tts_text"] for r in con.execute("SELECT tts_text FROM sentence")])
        set_meta(con, "quote_style", style)
    mcfg = cfg["character_memory"]
    ensure_chunks(con, docs, mcfg["chunk_words"], mcfg.get("chunk_mode", "words"), style)
    chunks = chunk_rows(con, docs)

    # ---- A. rules
    scan = _scan(con, chunks, style)
    quotes, by_chunk = scan["quotes"], scan["by_chunk"]
    n_fp = sum(1 for q in quotes if q["tag_kind"] == "first_person")
    first_person = n_fp >= vc["first_person_min"]
    set_meta(con, "v3_first_person", int(first_person))
    manual = G.load_manual(book.dir)
    need = Counter()
    for k, n in scan["phrases"].items():
        if scan["tag_phrases"].get(k) or n >= 2:
            need[k] = n
    cached = json.loads(get_meta(con, "v3_gender_llm") or "{}")
    llm_g = llm or get_llm(cfg["llm"], mon)          # called only for names no rule decides
    llm_g.obs = obs
    table, cached = G.build(need, scan["paragraphs"], manual, scan["examples"], llm=llm_g, cached_llm=cached,
                            log=log, workers=vc["concurrency"])
    narrator_names, narr_gender = [], None
    if first_person:
        narrator_names = _narrator_names(scan, table, manual, int(vc["narrator_name_min"]))
        narr_gender = next((table.get(k) for k in narrator_names if table.get(k) in ("male", "female")), None)
        for k in narrator_names:
            table.entries[k] = ("narrator", f"narrator's name (addressed {scan['voc_keys'][k]}x, never a speech tag)")
    G.store_table(con, table, cached)
    G.save(book.dir, table, manual, Counter({norm_phrase(k): v for k, v in need.items()}))
    src = Counter(s.split(" ")[0].split(":")[0] for g, s in table.entries.values())
    gen = Counter(g for g, s in table.entries.values())
    log(f"voices: {len(quotes)} quotes in {len(chunks)} chunks; first-person book: {first_person} ({n_fp} 'I' tags); "
        f"name table {len(table.entries)} ({', '.join(f'{k} {v}' for k, v in gen.most_common())}; "
        f"from {', '.join(f'{k} {v}' for k, v in src.most_common())}) -> genders.yaml"
        + (f"; the narrator is called {', '.join(narrator_names)}" if narrator_names else ""))
    for q in quotes:
        _attach(q, table, first_person, vc["second"])
    _store_rules(con, quotes)

    # ---- B. the model, for quotes the rules leave open
    _load_answers(con, quotes, table, first_person, vc["second"])
    second = "who2" if vc["second"] == "notes" else "voice"
    asks = {"none": [], "who": ["who"], "voice": [second], "both": ["who", second]}[vc["ask"]]
    todo_chunks = []
    for c in chunks:
        qs = by_chunk.get(c["id"], [])
        open_q = [q for q in qs if D.tag_class(q)[0] is None]
        missing = [a for a in asks if any(q[ANSWER_COL[a]] is None for q in open_q)]
        if open_q and missing:
            todo_chunks.append((c, missing))
    log(f"voices: {sum(1 for q in quotes if D.tag_class(q)[0] is None)} quotes left to the model; "
        f"{len(todo_chunks)} chunks to ask ({', '.join(asks) or 'rules only'})")
    if todo_chunks:
        llm = llm or llm_g or get_llm(cfg["llm"], mon)
        llm.obs = obs
        if any("who2" in m for _, m in todo_chunks):
            _notes_pass(con, llm, obs, [c for c, m in todo_chunks if "who2" in m], scan, vc, log)
        _model_pass(con, llm, obs, todo_chunks, scan, table, first_person, vc, mon, log)
        _load_answers(con, quotes, table, first_person, vc["second"])

    # ---- C. decide and write
    # first-person book: the two questions agreeing is not enough, a rule must back it (docs/DESIGN.md)
    thr = float(vc["threshold_first_person"] if first_person and vc.get("threshold_first_person") is not None
                else vc["threshold"])
    st = D.decide(quotes, ask=vc["ask"], threshold=thr, scene=bool(vc["scene"]),
                  crowd=int(vc["crowd"] or 0), vocative=bool(vc["vocative"]),
                  narrator_gender=narr_gender if vc["narrator_guard"] else None)
    _write(con, chunks, scan, {q["qid"]: q for q in quotes})
    n = apply_overrides(book, con)
    if n:
        con.execute(f"UPDATE utterance SET speaker_id='{MALE}' WHERE source='manual' AND speaker_id='male'")
        con.execute(f"UPDATE utterance SET speaker_id='{FEMALE}' WHERE source='manual' AND speaker_id='female'")
        con.commit()
        log(f"voices: applied {n} manual overrides from attribution_overrides.csv")
    cls = Counter(q["speaker"] for q in quotes)
    log(f"voices: male {cls[MALE]}, female {cls[FEMALE]}, narrator {cls[NARRATOR]} "
        f"(threshold {thr}{' first-person' if first_person else ''}, ask {vc['ask']}, second {vc['second']}); " + ", ".join(f"{k} {v}" for k, v in sorted(st.items()))
        + f"  [{time.time() - t0:.0f}s]")


# ---------------------------------------------------------------- A

def _scan(con, chunks, style) -> dict:
    quotes: list[dict] = []
    by_chunk: dict[str, list[dict]] = {}
    items_of: dict[str, list] = {}
    phrases, tag_phrases = Counter(), Counter()
    surface: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    window: dict[str, Counter] = {}
    paragraphs: list[list[tuple[str, bool]]] = []
    ord_ = 0

    speakerish = set()           # names that can be offered to the model: tags, vocatives, titled names
    voc_keys, tag_keys = Counter(), Counter()

    def see(phrase, sent_text, chunk_id, tag=False, speaker=False):
        phrase = re.sub(r"[’']s$", "", phrase)
        k = norm_phrase(phrase)
        if not k:
            return
        if tag or speaker or G.title_of(phrase + " "):
            speakerish.add(k)
        phrases[k] += 1
        surface[k][phrase] += 1
        if tag:
            tag_phrases[k] += 1
        if len(examples[k]) < 2 and sent_text not in examples[k]:
            examples[k].append(sent_text)
        window[chunk_id][k] += 1

    for c in chunks:
        items = _pieces_for_chunk(con, c, style)
        items_of[c["id"]] = items
        window[c["id"]] = Counter()
        qids = []
        for _, pcs in items:
            for p in pcs:
                if p.is_quote and p.quote_id not in qids:
                    qids.append(p.quote_id)
        ctx = _quote_context(items, qids)
        para_has_narr = defaultdict(bool)
        for s, pcs in items:
            if not pcs or any(not p.is_quote and any(ch.isalpha() for ch in p.text) for p in pcs):
                para_has_narr[s["para"]] = True
        # pronoun learning input: narration of each sentence, per paragraph
        cur_para, para_list = None, []
        for s, pcs in items:
            if s["para"] != cur_para:
                para_list = []
                paragraphs.append(para_list)
                cur_para = s["para"]
            narr = " … ".join(p.text for p in pcs if not p.is_quote) if pcs else s["tts_text"]
            para_list.append((narr, any(p.is_quote for p in pcs)))
            for p in pcs or []:
                if not p.is_quote:
                    for n in G.narration_names(p.text):
                        see(n, s["tts_text"], c["id"])
            if not pcs:
                for n in G.narration_names(s["tts_text"]):
                    see(n, s["tts_text"], c["id"])
        pieces_of = defaultdict(list)
        for si, (s, pcs) in enumerate(items):
            for k, p in enumerate(pcs):
                if p.is_quote:
                    pieces_of[p.quote_id].append((si, k))
        cq = []
        for qid in qids:
            locs = pieces_of[qid]
            (s0, k0), (s1, k1) = locs[0], locs[-1]
            sent0, pcs0 = items[s0]
            sent1, pcs1 = items[s1]
            text = " ".join(items[si][1][k].text for si, k in locs)
            before = pcs0[k0 - 1].text if k0 > 0 and not pcs0[k0 - 1].is_quote else ""
            after = pcs1[k1 + 1].text if k1 + 1 < len(pcs1) and not pcs1[k1 + 1].is_quote else ""
            x = ctx[qid]
            tag = find_tag(x["after"], x["before"], x["between"])
            rule = None if tag else not_dialogue(text, not para_has_narr[sent0["para"]], before, after)
            vocs = G.vocatives(text)
            for v in vocs:
                see(v, sent0["tts_text"], c["id"], speaker=True)
                voc_keys[norm_phrase(re.sub(r"[’']s$", "", v))] += 1
            if tag and tag.kind in ("name", "np"):
                see(tag.text, sent0["tts_text"], c["id"], tag=True)
                tag_keys[norm_phrase(re.sub(r"[’']s$", "", tag.text))] += 1
            narration = " ".join(p.text for s, pcs in items if s["para"] == sent0["para"] for p in pcs
                                 if not p.is_quote)
            q = {"qid": qid, "chunk": c["id"], "doc": c["doc_id"], "para": sent0["para"], "ord": ord_,
                 "text": text, "words": len(text.split()), "narration": narration,
                 "tag_text": tag.text if tag else None, "tag_kind": tag.kind if tag else None, "rule": rule,
                 "vocatives": [norm_phrase(v) for v in vocs]}
            ord_ += 1
            cq.append(q)
        by_chunk[c["id"]] = cq
        quotes += cq
    best = {k: v.most_common(1)[0][0] for k, v in surface.items()}
    return {"quotes": quotes, "by_chunk": by_chunk, "items": items_of, "paragraphs": paragraphs,
            "phrases": Counter({best[k]: n for k, n in phrases.items()}),
            "tag_phrases": Counter({best[k]: n for k, n in tag_phrases.items()}),
            "examples": {k: v for k, v in examples.items()}, "window": window, "best": best,
            "speakerish": speakerish, "voc_keys": voc_keys, "tag_keys": tag_keys}


def _narrator_names(scan, table: G.Table, manual: dict, min_voc: int) -> list[str]:
    """First-person book: the narrator is the person others address most and who never gets a speech tag
    ("Jake," 101 times, "said Jake" never: he is "I"). Names that share a word with it go along
    (Jake Barnes). genders.yaml `manual:` overrides (set a name to narrator, or to male/female to undo)."""
    cands = [(n, k) for k, n in scan["voc_keys"].most_common()
             if n >= min(5, min_voc) and scan["tag_keys"].get(k, 0) == 0 and table.get(k) in ("male", "female")
             and k not in manual]
    if not cands or cands[0][0] < min_voc:
        return [k for k, v in manual.items() if v == "narrator"]
    top = cands[0][1]
    g = table.get(top)
    # other names of the narrator: also addressed, never tagged, same gender (Barnes, Mr. Barnes; 5+ times)
    accepted = [top] + [k for n, k in cands[1:] if n >= max(5, min_voc // 2) and table.get(k) == g]
    words = {w for k in accepted for w in k.split() if not G.title_of(w.capitalize() + " ")}
    out = list(accepted) + [k for k in table.entries if k not in accepted and k not in manual
                            and scan["tag_keys"].get(k, 0) == 0 and words & set(k.split()) and len(k.split()) > 1]
    return out + [k for k, v in manual.items() if v == "narrator" and k not in out]


def _load_answers(con, quotes, table, first_person, second) -> None:
    have = {r["quote_id"]: r for r in con.execute(f"SELECT {ANSWER_COLS} FROM v3_quote")}
    for q in quotes:
        h = have.get(q["qid"])
        for k in ("who", "who_gender", "voice_ans", "who2", "who2_gender", "evidence", "basis"):
            q[k] = h[k] if h else None
        _attach(q, table, first_person, second)


def _attach(q: dict, table: G.Table, first_person: bool, second: str = "voice") -> None:
    t = q.get("tag_text")
    q["tag_gender"] = None
    if t and q["tag_kind"] == "pronoun":
        q["tag_gender"] = {"he": "male", "she": "female"}[t]
    elif t and q["tag_kind"] in ("name", "np"):
        q["tag_gender"] = table.get(t)
        q["tag_gender_source"] = table.source(t)
    w = q.get("who")
    q["who_table_gender"] = table.get(w) if w and w not in P.FIXED and w != P.NARRATOR_I else None
    w2 = q.get("who2")
    q["who2_table_gender"] = table.get(w2) if w2 and w2 not in P.FIXED and w2 != P.NARRATOR_I else None
    q["first_person"] = first_person
    q["second"] = second


def _store_rules(con, quotes) -> None:
    con.executemany(
        "INSERT INTO v3_quote(quote_id, chunk_id, doc_id, para, ord, text, narration, tag_text, tag_kind, rule) "
        "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(quote_id) DO UPDATE SET chunk_id=excluded.chunk_id, "
        "doc_id=excluded.doc_id, para=excluded.para, ord=excluded.ord, narration=excluded.narration, "
        "tag_text=excluded.tag_text, tag_kind=excluded.tag_kind, rule=excluded.rule, "
        # a changed quote text invalidates the model's answers
        "who=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.who END, "
        "who_gender=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.who_gender END, "
        "voice_ans=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.voice_ans END, "
        "who2=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.who2 END, "
        "who2_gender=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.who2_gender END, "
        "evidence=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.evidence END, "
        "basis=CASE WHEN v3_quote.text=excluded.text THEN v3_quote.basis END, text=excluded.text",
        [(q["qid"], q["chunk"], q["doc"], q["para"], q["ord"], q["text"], q["narration"], q["tag_text"],
          q["tag_kind"], q["rule"]) for q in quotes])
    con.commit()


# ---------------------------------------------------------------- B

def _people(scan, chunks_ids: list[str], table: G.Table, cap: int) -> list[tuple[str, str]]:
    c = Counter()
    for cid in chunks_ids:
        c.update(scan["window"].get(cid, Counter()))
    out = []
    for k, _ in c.most_common():
        if len(out) >= cap:
            break
        if k not in scan["speakerish"] or table.source(k) == G.SCENE_DEPENDENT or table.get(k) == "narrator":
            continue
        name = scan["best"].get(k, k)
        if name in P.FIXED or name == P.NARRATOR_I:
            continue
        g = table.get(name)
        out.append((name, "unknown" if g not in ("male", "female") else g))
    return out


def _known_line(label: str, q: dict) -> str | None:
    c, _ = D.tag_class(q)
    if c is None:
        return None
    if q.get("rule"):
        return f"{label}: {P.NOT_SPEECH}"
    if c == "I":
        return f"{label}: {P.NARRATOR_I}"
    if q["tag_kind"] == "group":
        return f"{label}: several people ({q['tag_text']})"
    return f"{label}: {q['tag_text']} ({c})"


def chunk_order(con) -> list[str]:
    return [r["id"] for r in con.execute("SELECT id FROM chunk ORDER BY doc_id, seq_from")]


def neighbour(order: list[str], cid: str, step: int) -> str | None:
    """The chunk before (step -1) or after (+1) in the same content doc, else None."""
    i = order.index(cid) + step
    if not 0 <= i < len(order):
        return None
    return order[i] if order[i].split("-")[0] == cid.split("-")[0] else None


def chunk_job(scan, c, table, vc, prev: str | None):
    """(label, marked text, known lines, quotes to decide, people) for one chunk's model questions."""
    qs = scan["by_chunk"][c["id"]]
    label = {q["qid"]: f"Q{i + 1}" for i, q in enumerate(qs)}
    marked = _marked_text(scan["items"][c["id"]], label)
    known = [x for q in qs for x in [_known_line(label[q["qid"]], q)] if x]
    todo = [label[q["qid"]] for q in qs if D.tag_class(q)[0] is None]
    people = _people(scan, [x for x in (prev, c["id"]) if x], table, int(vc["max_people"]))
    return label, marked, known, todo, people


def _notes_of(con, cid: str) -> tuple[bool, dict | None]:
    """(written?, the chunk's cleaned scene notes)"""
    r = con.execute("SELECT notes FROM chunk WHERE id=?", (cid,)).fetchone()
    if not r or r["notes"] is None:
        return False, None
    return True, json.loads(r["notes"])


def _notes_pass(con, llm, obs, need: list, scan, vc, log) -> None:
    """Scene notes (15.4) for the chunks whose second `who` is still to ask, and for the chunks before and
    after them in the same doc, so all notes are written before any of them is used. A failed call leaves
    the notes empty; those chunks skip the second question this run and get it on a rerun."""
    order = chunk_order(con)
    want = []
    for c in need:
        for x in (neighbour(order, c["id"], -1), c["id"], neighbour(order, c["id"], 1)):
            if x and x not in want and x in scan["items"] and not _notes_of(con, x)[0]:
                want.append(x)
    if not want:
        return

    def call(cid):
        qs = scan["by_chunk"].get(cid, [])
        text = P.plain(_marked_text(scan["items"][cid], {q["qid"]: f"Q{i + 1}" for i, q in enumerate(qs)}))
        msgs, schema, payload = P.notes(text)
        with obs.in_chunk(cid):
            raw = llm.complete_json(msgs, schema, "v3_notes", payload, max_tokens=2500, truncation_retries=1)
        return P.clean_notes(raw, text)

    t0, failed, dropped = time.time(), [], 0
    ex = ThreadPoolExecutor(max_workers=max(1, int(vc["concurrency"])))
    try:
        futs = {ex.submit(call, cid): cid for cid in want}
        for f in as_completed(futs):
            cid = futs[f]
            try:
                n = f.result()
            except LLMError as e:
                obs.warn("LLM_FAILED", "warn", cid, f"voices notes: {type(e).__name__}: {e}"[:300])
                obs.flush()
                failed.append(cid)
                continue
            dropped += len(n.get("dropped") or [])
            con.execute("UPDATE chunk SET notes=? WHERE id=?", (json.dumps(n, ensure_ascii=False), cid))
            con.commit()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    log(f"voices: scene notes for {len(want) - len(failed)}/{len(want)} chunks in {time.time() - t0:.0f}s"
        f" ({dropped} names not in the text dropped)" + (f", {len(failed)} FAILED (rerun to retry)" if failed else ""))


def _model_pass(con, llm, obs, todo_chunks, scan, table, first_person, vc, mon, log) -> None:
    order = chunk_order(con)
    jobs, no_notes = {}, []
    for c, missing in todo_chunks:
        nt = None
        if "who2" in missing:
            around = [(lab, x) for lab, x in (("PREVIOUS PASSAGE", neighbour(order, c["id"], -1)),
                                              ("THIS PASSAGE", c["id"]), ("NEXT PASSAGE", neighbour(order, c["id"], 1)))
                      if x and x in scan["items"]]
            got = [(lab, *_notes_of(con, x)) for lab, x in around]
            if all(ok for _, ok, _ in got):
                nt = P.notes_text([(lab, n) for lab, _, n in got])
            else:                               # a notes call failed: ask this one on the next run
                missing = [m for m in missing if m != "who2"]
                no_notes.append(c["id"])
        if missing:
            jobs[c["id"]] = (c, missing, *chunk_job(scan, c, table, vc, neighbour(order, c["id"], -1)), nt)
    if no_notes:
        log(f"voices: {len(no_notes)} chunks wait for their scene notes (second question next run)")

    def call(cid):
        c, missing, label, marked, known, todo, people, nt = jobs[cid]
        res = {}
        with obs.in_chunk(cid):
            for ask in missing:
                res[ask] = _ask(llm, ask, marked, todo, known, people, first_person, notes=nt)
        return res

    failed, streak = [], 0
    ex = ThreadPoolExecutor(max_workers=max(1, int(vc["concurrency"])))
    t0 = time.time()
    try:
        futs = {ex.submit(call, cid): cid for cid in jobs}
        for f in as_completed(futs):
            cid = futs[f]
            obs.chunk_id = cid
            c, missing, label, *_ = jobs[cid]
            inv = {v: k for k, v in label.items()}
            try:
                res = f.result()
            except LLMError as e:
                con.execute("UPDATE chunk SET p3='failed' WHERE id=?", (cid,))
                con.commit()
                obs.warn("LLM_FAILED", "warn", cid, f"voices: {type(e).__name__}: {e}"[:300])
                obs.flush()
                failed.append(cid)
                streak += 1
                if streak >= mon.get("max_consecutive_failures", 3) + 2:
                    obs.pause("LLM_DOWN", cid, f"{streak} chunks in a row failed; check the LLM service")
                obs.checkpoint()
                continue
            streak = 0
            for d in res.get("who", []):
                if d["q"] in inv:
                    con.execute("UPDATE v3_quote SET who=?, who_gender=? WHERE quote_id=?",
                                (d["speaker"], d["gender"], inv[d["q"]]))
            for d in res.get("voice", []):
                if d["q"] in inv:
                    con.execute("UPDATE v3_quote SET voice_ans=? WHERE quote_id=?", (d["voice"], inv[d["q"]]))
            for d in res.get("who2", []):
                if d["q"] in inv:
                    con.execute("UPDATE v3_quote SET who2=?, who2_gender=?, evidence=?, basis=? WHERE quote_id=?",
                                (d["speaker"], d["gender"], d.get("evidence"), d.get("basis"), inv[d["q"]]))
            # an answer the model left out counts as unknown, so the chunk is not asked again and again
            for ask in missing:
                col = ANSWER_COL[ask]
                todo_ids = [inv[t] for t in jobs[cid][5]]
                con.execute(f"UPDATE v3_quote SET {col}='unknown' WHERE {col} IS NULL AND quote_id IN "
                            f"({','.join('?' * len(todo_ids))})", todo_ids)
            con.execute("UPDATE chunk SET p3='done' WHERE id=?", (cid,))
            con.commit()
            obs.checkpoint()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    obs.chunk_id = None
    log(f"voices: model pass {len(jobs) - len(failed)}/{len(jobs)} chunks in {time.time() - t0:.0f}s"
        + (f", {len(failed)} FAILED (rerun to retry)" if failed else ""))


def _ask(llm, ask, marked, todo, known, people, first_person, depth=0, notes=None) -> list[dict]:
    per_q = 40
    if ask == "who":
        msgs, schema, payload = P.who(marked, todo, known, people, first_person)
    elif ask == "who2":                         # who again, with the scene notes and evidence (15.18)
        msgs, schema, payload = P.who(marked, todo, known, people, first_person, notes=notes, evidence=True)
        per_q = 130                             # the evidence makes each answer longer
    else:
        msgs, schema, payload = P.voice(marked, todo, known, first_person)
    try:
        out = llm.complete_json(msgs, schema, f"v3_{ask}", payload, max_tokens=per_q * len(todo) + 300,
                                truncation_retries=0)
        return out.get("quotes", [])
    except LLMTruncated:
        if depth >= 2 or len(todo) < 2:
            raise
        h = len(todo) // 2
        return (_ask(llm, ask, marked, todo[:h], known, people, first_person, depth + 1, notes)
                + _ask(llm, ask, marked, todo[h:], known, people, first_person, depth + 1, notes))


# ---------------------------------------------------------------- C

def _write(con, chunks, scan, qmap) -> None:
    for c in chunks:
        items = scan["items"][c["id"]]
        story_ids = {s["id"] for s, _ in items}
        for s in chunk_sentences(con, c):
            if s["id"] not in story_ids:
                con.execute("DELETE FROM utterance WHERE sentence_id=? AND source != 'default'", (s["id"],))
                con.execute("INSERT OR IGNORE INTO utterance(id, sentence_id, seq, start, end, tts_text) "
                            "VALUES (?,?,0,0,?,?)", (f"{s['id']}u0", s["id"], len(s["tts_text"]), s["tts_text"]))
        rows = []
        for s, pcs in items:
            con.execute("DELETE FROM utterance WHERE sentence_id=?", (s["id"],))
            if not pcs:
                rows.append((f"{s['id']}u0", s["id"], 0, 0, len(s["tts_text"]), s["tts_text"], 0, None, NARRATOR,
                             None, None, "default", None, None, None))
                continue
            for k, p in enumerate(pcs):
                if p.is_quote:
                    q = qmap[p.quote_id]
                    who = q.get("who") if q["source"] in ("model", "scene", "paragraph", "vocative") else q.get("tag_text")
                    rows.append((f"{s['id']}u{k}", s["id"], k, p.start, p.end, p.text, 1, p.quote_id, q["speaker"],
                                 None, q["conf"], q["source"], who, q["conf"], q["cls"]))
                else:
                    rows.append((f"{s['id']}u{k}", s["id"], k, p.start, p.end, p.text, 0, None, NARRATOR, None,
                                 None, "rule", None, None, None))
        con.executemany("INSERT INTO utterance(id, sentence_id, seq, start, end, tts_text, is_quote, quote_id, "
                        "speaker_id, style, confidence, source, model_speaker, model_confidence, voice_gender) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
