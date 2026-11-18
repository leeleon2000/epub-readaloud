"""Stage: roster (v2 P2, docs/DESIGN.md). Decide who is who from all of P1's references at once.

1. Name graph: nodes are name-like references from P1 (names.parse(...).is_name), keyed by norm().
   Edges: P1 "same individual in one passage" -> strong, but only between compatible names
   (names.compatible); names.link -> medium (contains / nickname) or weak (same surname).
2. Name groups: connected components (cap `roster_group_max` nodes; bigger ones are cut on strong and
   medium edges only, then by order of appearance).
3. One model call per group with more than one node (parallel; answers cached in p2_group by a hash of
   the group's input, so a rerun with unchanged P1 data makes no calls). Single nodes need no call.
4. Validation by rules: an individual's names must be pairwise compatible (else split, NAME_CONFLICT);
   the name must be one of its references; gender from titles first; relations need verbatim evidence and
   single-value limits (RELATION_CONFLICT); two individuals with one full name -> ALIAS_CONFLICT.
5. Write: previous model events are superseded and the roster is rebuilt from events (manual edits stay),
   then create / alias / mention / relation events; chunk.characters for attribute; characters.yaml and
   reports/roster_report.md.

Runs only with `character_memory.roster: v2` (or `--stages roster`); the v1 characters stage is skipped then.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations

from ..adapters.llm import LLMError, get_llm
from ..book import Book
from ..chunks import chunk_rows, chunk_text, ensure_chunks, scope_docs, story_sentences
from ..memory import names, prompts
from ..memory.db import CharacterDB, norm
from .collect import _flat

RANK = {"strong": 3, "medium": 2, "weak": 1, "hint": 0}     # hint: shown to the model, not used for grouping
ALIAS_STRENGTH = {"name": "strong", "given": "strong", "nickname": "strong", "title_given": "strong",
                  "title_surname": "medium", "surname": "weak", "generation": "medium"}
SINGLE_VALUE = {"spouse_of": 1}         # per individual; parent_of is checked per gender below


@dataclass
class Node:
    key: str
    texts: Counter = field(default_factory=Counter)
    chunks: list = field(default_factory=list)
    genders: Counter = field(default_factory=Counter)
    traits: list = field(default_factory=list)
    count: int = 0

    @property
    def text(self) -> str:
        return self.texts.most_common(1)[0][0]

    @property
    def parsed(self):
        return names.parse(self.text)


def build_graph(persons: list[dict]):
    """persons: [{chunk_id, refs, gender_cue, traits}] in reading order -> (nodes, edges, p1_conflicts)."""
    nodes: dict[str, Node] = {}
    edges: dict[tuple[str, str], dict] = {}
    p1_conflicts = Counter()

    def edge(a, b, strength, why):
        k = tuple(sorted((a, b)))
        e = edges.setdefault(k, {"strength": strength, "why": Counter()})
        if RANK[strength] > RANK[e["strength"]]:
            e["strength"] = strength
        e["why"][why] += 1

    for p in persons:
        keys = []
        for r in p["refs"]:
            n = names.parse(r["text"])
            if not n.is_name:
                continue
            k = norm(r["text"])
            node = nodes.setdefault(k, Node(k))
            node.texts[r["text"]] += 1
            node.count += 1
            if p["chunk_id"] not in node.chunks:
                node.chunks.append(p["chunk_id"])
            node.genders[p.get("gender_cue") or "unknown"] += 1
            if p.get("traits") and len(node.traits) < 30:
                node.traits.append(p["traits"])
            if k not in keys:
                keys.append(k)
        for a, b in combinations(keys, 2):
            if names.compatible(nodes[a].text, nodes[b].text).ok:
                edge(a, b, "hint", f"p1:{p['chunk_id']}")
            else:
                p1_conflicts[tuple(sorted((a, b)))] += 1
    for a, b in combinations(list(nodes), 2):
        lk = names.link(nodes[a].text, nodes[b].text)
        if lk:
            edge(a, b, lk[0], lk[1])
    # A P1 "same individual" claim is strong when a name rule supports it (contains, nickname, same surname)
    # or two passages make it; one passage alone ("Miss Lucas" = "Eliza") stays a hint for the model, so a
    # single misreading cannot join two families into one group.
    for (a, b), e in edges.items():
        p1 = sum(n for w, n in e["why"].items() if w.startswith("p1:"))
        rule = any(not w.startswith("p1:") for w in e["why"])
        if p1 and (rule or p1 >= 2):
            e["strength"] = "strong"
    return nodes, edges, p1_conflicts


def name_groups(nodes: dict[str, Node], edges: dict, cap: int = 40) -> list[list[str]]:
    order = {k: i for i, k in enumerate(nodes)}

    def components(keys, min_rank):
        parent = {k: k for k in keys}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for (a, b), e in edges.items():
            if a in parent and b in parent and RANK[e["strength"]] >= min_rank:
                parent[find(a)] = find(b)
        comp = defaultdict(list)
        for k in keys:
            comp[find(k)].append(k)
        return [sorted(c, key=order.get) for c in comp.values()]

    out = []
    for c in components(list(nodes), RANK["weak"]):
        if len(c) <= cap:
            out.append(c)
            continue
        for c2 in components(c, 2):
            out += [c2[i:i + cap] for i in range(0, len(c2), cap)]
    return sorted(out, key=lambda c: order[c[0]])


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None, llm=None, obs=None) -> None:
    from ..observe import Observer
    obs = obs or Observer.null(log)
    mcfg, mon = cfg["character_memory"], cfg.get("monitor", {})
    docs = scope_docs(con, doc_ids)
    ensure_chunks(con, docs, mcfg["chunk_words"])
    chunks = chunk_rows(con, docs)
    missing = [c["id"] for c in chunks if c["p1"] != "done"]
    if missing:
        raise SystemExit(f"roster: run collect first (pending: {', '.join(missing[:5])})")
    t0 = time.time()
    persons = []
    for c in chunks:
        for r in con.execute("SELECT * FROM p1_person WHERE chunk_id=? ORDER BY idx", (c["id"],)):
            persons.append({"chunk_id": c["id"], "idx": r["idx"], "refs": json.loads(r["refs"]),
                            "gender_cue": r["gender_cue"], "traits": r["traits"], "related": r["related"]})
    nodes, edges, p1_conflicts = build_graph(persons)
    groups = name_groups(nodes, edges, int(mcfg.get("roster_group_max", 40)))
    sents = {c["id"]: story_sentences(con, c) for c in chunks}
    book_flat = _flat(" ".join(chunk_text(s) for s in sents.values()))
    hints = [dict(r) for r in con.execute(
        f"SELECT * FROM p1_relation WHERE chunk_id IN ({','.join('?' * len(chunks))})", [c["id"] for c in chunks])]
    log(f"roster: {len(persons)} P1 entries -> {len(nodes)} names, {len(edges)} links, {len(groups)} name groups "
        f"({sum(len(g) > 1 for g in groups)} need the model)")

    # --- model calls (parallel, cached)
    calls = {}
    for g in groups:
        if len(g) > 1:
            msgs, schema, payload = _prompt(g, nodes, edges, sents, hints)
            h = hashlib.sha256(json.dumps([msgs, schema], sort_keys=True).encode()).hexdigest()[:16]
            calls[tuple(g)] = (h, msgs, schema, payload)
    cached = {r["hash"]: json.loads(r["result"]) for r in con.execute("SELECT hash, result FROM p2_group")
              if r["result"]}
    todo = [g for g, (h, *_) in calls.items() if h not in cached]
    answers = {g: cached[calls[g][0]] for g in calls if calls[g][0] in cached}
    if todo:
        llm = llm or get_llm(cfg["llm"], mon)
        llm.obs = obs
        ex = ThreadPoolExecutor(max_workers=int(mcfg.get("p1_concurrency", 8)))
        try:
            futs = {}
            for i, g in enumerate(todo):
                h, msgs, schema, payload = calls[g]
                futs[ex.submit(_call, llm, obs, f"g-{h[:8]}", msgs, schema, payload)] = g
            for f in as_completed(futs):
                g = futs[f]
                h = calls[g][0]
                try:
                    ans = f.result()
                except LLMError as e:
                    obs.warn("LLM_FAILED", "warn", f"g-{h[:8]}", f"roster: {type(e).__name__}: {e}"[:300],
                             names=[nodes[k].text for k in g])
                    ans = _fallback(g, nodes)
                else:
                    con.execute("INSERT OR REPLACE INTO p2_group(hash, nodes, result) VALUES (?,?,?)",
                                (h, json.dumps(list(g)), json.dumps(ans, ensure_ascii=False)))
                    con.commit()
                answers[g] = ans
                obs.flush()
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    # --- validate and write
    db = CharacterDB(con, profile_max_chars=mcfg["profile_max_tokens"] * 4, run_id=obs.run_id)
    from ..store import get_meta, set_meta
    prev_rel = {(r["a"], r["type"], r["b"]) for r in con.execute("SELECT a, type, b FROM relation")}
    prev_scope = int(get_meta(con, "relations_scope", 0) or 0)
    con.execute("UPDATE character_event SET superseded_by=? WHERE source='model' AND superseded_by IS NULL",
                (f"roster-{obs.run_id}",))
    db.replay()
    report = []
    created: dict[str, list[str]] = {}            # character id -> node keys
    for g in groups:
        ans = answers.get(tuple(g)) or _single(g, nodes)
        created_g, rep = _apply_group(db, g, nodes, ans, book_flat, obs)
        created.update(created_g)
        report.append(rep)
    for rep in report:
        for keys in rep["deferred"]:
            created.update(_shared_or_card(db, keys, nodes, edges, created, rep, obs))
    _reapply_manual(db)
    from ..memory import relgraph
    claims = [c for r in report for c in r["claims"]]
    rcfg = cfg.get("relations", {})
    manual = relgraph.load_manual(book.dir)
    grew = bool(prev_scope) and len(chunks) > prev_scope
    # 5.2: one-passage same-person claims confirmed by a shared strong relation -> merge (listed in the report)
    cands, rcards = relgraph.analyze(db, con, chunks, sents, claims)
    dec = relgraph.decide(cands, rcards, manual, rcfg)
    anomalies = relgraph.anomaly(dec, rcards, prev_rel, grew, rcfg)
    merges = [] if anomalies else _relation_merges(db, edges, nodes, created, cands, dec["accepted"])
    db.con.commit()
    created = _resolve_created(db, created)
    # the graph again with the merged cards; only basic relations are written (5.6)
    cands, rcards = relgraph.analyze(db, con, chunks, sents, claims)
    dec = relgraph.decide(cands, rcards, manual, rcfg)
    anomalies = relgraph.anomaly(dec, rcards, prev_rel, grew, rcfg)
    relrep = relgraph.write(db, cands, dec, obs.run_id, obs)
    married = [] if anomalies else relgraph.married_names(db, dec["accepted"])      # Mrs. Collins -> Charlotte
    created = _resolve_created(db, created)
    _full_name_check(created, nodes, obs)
    set_meta(con, "relations_disabled", "1" if anomalies else "0")
    set_meta(con, "relations_scope", str(len(chunks)))
    relrep.update(merges=merges, married=married, anomalies=anomalies)
    # chunk.characters: who is referred to in each chunk (attribute candidates), plus people referred to
    # only through a relation that the graph resolves to one card ("his lady" -> Mrs. Bennet; 5.3)
    owner = defaultdict(set)
    for cid, keys in created.items():
        for k in keys:
            owner[k].add(cid)
    from ..memory import kinref
    gender = {r["id"]: r["gender"] for r in con.execute("SELECT id, gender FROM character")}
    kg = relgraph.graph_from_db(con, db.resolve, gender)      # weak relations too: an extra candidate is harmless
    con.execute("DELETE FROM mention_link WHERE source='roster'")
    rcards = relgraph.Cards(db)
    n_links = 0
    for c in chunks:
        ids = []
        unnamed = []
        for p in persons:
            if p["chunk_id"] != c["id"]:
                continue
            got = []
            for r in p["refs"]:
                for cid in sorted(owner.get(norm(r["text"]), ())):
                    got.append(cid)
                    if cid not in ids:
                        ids.append(cid)
            if not got:
                unnamed.append(p)
        for p in unnamed:
            hint = None
            for rr in json.loads(p.get("related") or "[]"):
                hint = hint or rcards.of(rr.get("of_ref", ""))
            for r in p["refs"]:
                res = kinref.resolve(kg, r["text"], [hint] if hint else ids, gender, anchor_of=rcards.of)
                if res is None:
                    continue
                targets, anchors, kin = res
                plural = r["text"].rstrip().lower().endswith("s")
                ok = len(targets) == 1 and not plural
                con.execute("INSERT INTO mention_link(chunk_id, mention, anchors, relation, targets, resolved, source, "
                            "run_id) VALUES (?,?,?,?,?,?, 'roster', ?)",
                            (c["id"], r["text"], json.dumps(anchors), kin, json.dumps(sorted(targets)), int(ok),
                             obs.run_id))
                if ok:
                    t = next(iter(targets))
                    n_links += 1
                    if t not in ids:
                        ids.append(t)
                    break
        con.execute("UPDATE chunk SET pass1='done', characters=?, "
                    "pass2=CASE WHEN pass2='done' THEN 'stale' ELSE pass2 END WHERE id=?", (json.dumps(ids), c["id"]))
    relrep["mention_links"] = n_links
    con.commit()
    obs.flush()
    from .characters import export_yaml
    export_yaml(book, db)
    _write_report(book, report, p1_conflicts, nodes, relrep)
    n_split = sum(len(r["splits"]) for r in report)
    n_strong = sum(1 for x in relrep["accepted"] if x[3] == "strong")
    log(f"roster: {len(db.active_ids())} characters from {len(groups)} groups"
        + (f", {n_split} split by name rules" if n_split else "")
        + (f", {len(relrep['merges'])} merged by relations" if relrep.get("merges") else "")
        + (f", {sum(1 for m in married if 'merged' in m)} married names merged" if married else "")
        + f", {len(relrep['accepted'])} relations ({n_strong} strong, {len(relrep['conflicts'])} conflicts), "
        f"{relrep['mention_links']} kin references resolved ({time.time() - t0:.0f}s); "
        f"see {book.dir.name}/reports/roster_report.md")
    if anomalies:       # after everything is written; attribute will not use the graph (5.6 point 6)
        obs.pause("RELATION_ANOMALY", None, "relation graph not used: " + "; ".join(anomalies),
                  reasons=anomalies)


def _examples(key: str, node: Node, sents: dict, k: int = 2) -> list[str]:
    out = []
    t = node.text
    for cid in node.chunks:
        for s in sents.get(cid, []):
            if t in s["tts_text"] and len(out) < k:
                txt = s["tts_text"]
                i = txt.find(t)
                out.append(txt[max(0, i - 90): i + len(t) + 90].strip())
        if len(out) >= k:
            break
    return out


def _prompt(g, nodes, edges, sents, hints):
    code = {k: f"V{i + 1}" for i, k in enumerate(g)}
    variants = [{"code": code[k], "text": nodes[k].text, "count": nodes[k].count,
                 "genders": dict(nodes[k].genders), "examples": _examples(k, nodes[k], sents)} for k in g]
    same = []
    for (a, b), e in edges.items():
        if a in code and b in code:
            n = sum(v for w, v in e["why"].items() if w.startswith("p1:"))
            if n:
                same.append(f"- {code[a]} = {code[b]} ({n} passage{'s' if n > 1 else ''})")
    gk = set(g)
    hl = []
    for h in hints:
        if norm(h["a"]) in gk or norm(h["b"]) in gk:
            hl.append(f'- "{h["a"]}" {h["type"]} "{h["b"]}": “{h["evidence"]}”')
    msgs, schema, payload = prompts.roster(variants, same[:40], hl[:12])
    return msgs, schema, payload


def _call(llm, obs, label, msgs, schema, payload):
    with obs.in_chunk(label):
        return llm.complete_json(msgs, schema, "roster", payload, max_tokens=3000)


def _single(g, nodes):
    return {"individuals": [{"key": "I1", "name": nodes[g[0]].text, "variants": ["V1"], "gender": "unknown",
                             "reason": "single name"}], "shared": [], "relations": [], "unsure": []}


def _fallback(g, nodes):
    """No model answer: rule-based grouping (the mock's rules)."""
    from ..adapters.llm.mock import MockLLM
    return MockLLM()._roster({"variants": [{"code": f"V{i + 1}", "text": nodes[k].text} for i, k in enumerate(g)]})


def _ok(a: str, b: str, spouse_surnames: set) -> bool:
    """Compatible, or a marriage name change: the married form (Mrs./Lady X) carries the spouse's surname."""
    if names.compatible(a, b).ok:
        return True
    if not spouse_surnames or not names.compatible(a, b, married=True).ok:
        return False
    return any(n.title in names.MARRIED and n.surname in spouse_surnames for n in (names.parse(a), names.parse(b)))


def _partition(keys, nodes, spouse_surnames=frozenset()):
    """Split names into groups that are pairwise compatible (greedy, most frequent first)."""
    parts: list[list[str]] = []
    for k in sorted(keys, key=lambda k: (not nodes[k].parsed.lookup, -nodes[k].count)):
        for p in parts:
            if all(_ok(nodes[k].text, nodes[o].text, spouse_surnames) for o in p):
                p.append(k)
                break
        else:
            parts.append([k])
    return parts


def _gender(keys, nodes, model_gender):
    titles = Counter(nodes[k].parsed.gender for k in keys if nodes[k].parsed.gender)
    if titles:
        return titles.most_common(1)[0][0]
    if model_gender in ("male", "female"):
        return model_gender
    cues = Counter()
    for k in keys:
        for g, n in nodes[k].genders.items():
            if g in ("male", "female"):
                cues[g] += n
    return cues.most_common(1)[0][0] if cues else None


def _name(keys, nodes, model_name):
    if model_name and norm(model_name) in keys:
        return nodes[norm(model_name)].text
    def score(k):
        n = nodes[k].parsed
        return (bool(n.given and n.surname), n.lookup, len(nodes[k].text), nodes[k].count)
    return nodes[max(keys, key=score)].text


def _apply_group(db: CharacterDB, g, nodes, ans, book_flat, obs):
    vk = {f"V{i + 1}": k for i, k in enumerate(g)}
    rep = {"names": [nodes[k].text for k in g], "individuals": [], "splits": [], "shared": [], "unsure": [],
           "unplaced": [], "relations": [], "dropped": [], "claims": [], "deferred": []}
    assigned: dict[str, str] = {}
    inds = []
    for d in ans.get("individuals", []):
        keys = [vk[v] for v in d.get("variants", []) if v in vk and vk[v] not in assigned]
        for k in keys:
            assigned[k] = d["key"]
        if keys:
            inds.append({**d, "keys": keys})
    spouse_of = defaultdict(set)              # individual key -> spouse individual keys (model's claim)
    for r in ans.get("relations", []):
        if r.get("type") == "spouse_of":
            spouse_of[r["a"]].add(r["b"])
            spouse_of[r["b"]].add(r["a"])
    ind_keys = {d["key"] for d in inds}
    shared = []
    for sh in ans.get("shared", []):
        k = vk.get(sh.get("variant"))
        who = list(dict.fromkeys(w for w in sh.get("who", []) if w in ind_keys))
        k_ = vk.get(sh.get("variant"))
        if k_:      # a holder's names must be compatible with the shared reference (Miss Bennet is not Mr. Bennet)
            texts_of = {d["key"]: [nodes[x].text for x in d["keys"]] for d in inds}
            who = [w for w in who if all(names.compatible(nodes[k_].text, t).ok for t in texts_of.get(w, []))]
        # a first name / full name / nickname names one person; "shared" needs at least two holders
        if k and k not in assigned and len(who) >= 2 and not nodes[k].parsed.lookup:
            shared.append((k, who))
            assigned[k] = "shared"
    for u in ans.get("unsure", []):
        if u.get("variant") in vk and vk[u["variant"]] not in assigned:
            rep["unsure"].append(nodes[vk[u["variant"]]].text)
    # Not placed (unsure, invalid shared, or omitted): grouped by the name rules instead of dropped, so a
    # model that gives up on a big group cannot lose people (Jane, Sir William in the first v2 run).
    left = [k for k in g if k not in assigned]
    if left:
        fb = _fallback(left, nodes)
        for d in fb["individuals"]:
            keys = [left[int(v[1:]) - 1] for v in d["variants"]]
            if len(keys) == 1 and nodes[keys[0]].parsed.kind == "surname":
                rep["unsure"].append(nodes[keys[0]].text)        # a bare surname alone may be a place
                continue
            if all(nodes[k].parsed.kind == "title_surname" for k in keys):
                rep["deferred"].append(keys)                     # "Miss Bennet": shared or a card, see run()
                continue
            inds.append({"key": f"X{len(inds)}", "name": d["name"], "keys": keys, "gender": "unknown",
                         "reason": "not placed by the model; grouped by name rules"})
            rep["unplaced"].append([nodes[k].text for k in keys])

    created: dict[str, list[str]] = {}
    ikey_to_id: dict[str, str] = {}
    surnames_of = {d["key"]: {nodes[k].parsed.surname for k in d["keys"] if nodes[k].parsed.surname} for d in inds}
    for d in inds:
        sp = set().union(*(surnames_of.get(x, set()) for x in spouse_of.get(d["key"], ())))
        parts = _partition(d["keys"], nodes, sp)
        if len(parts) > 1:
            rep["splits"].append([[nodes[k].text for k in p] for p in parts])
            obs.warn("NAME_CONFLICT", "warn", None,
                     f"roster: split {d.get('name')!r} into " + " | ".join(", ".join(nodes[k].text for k in p)
                                                                         for p in parts),
                     names=[[nodes[k].text for k in p] for p in parts], reason=d.get("reason", ""))
        for i, part in enumerate(parts):
            name = _name(part, nodes, d.get("name") if i == 0 else None)
            cid = db.new_id(name)
            gender = _gender(part, nodes, d.get("gender"))
            traits = []
            for k in part:
                for t in nodes[k].traits:
                    if t not in traits:
                        traits.append(t)
            first = min((nodes[k].chunks[0] for k in part), default=None)
            db.apply("create", {"id": cid, "name": name, "gender": gender, "first_seen": first,
                                "profile": "; ".join(traits)[:400],
                                "status": "provisional" if d.get("provisional") else "active"})
            for k in part:
                for t in nodes[k].texts:
                    if t != name:
                        db.apply("alias", {"id": cid, "alias": t,
                                           "strength": ALIAS_STRENGTH.get(nodes[k].parsed.kind, "medium")})
            db.apply("mention", {"id": cid, "n": sum(nodes[k].count for k in part)})
            created[cid] = list(part)
            if i == 0:
                ikey_to_id[d["key"]] = cid
            rep["individuals"].append({"id": cid, "name": name, "gender": gender,
                                       "names": [nodes[k].text for k in part], "reason": d.get("reason", "")})
    for k, who in shared:
        ids = [ikey_to_id[w] for w in who if w in ikey_to_id]
        for cid in ids:
            db.apply("alias", {"id": cid, "alias": nodes[k].text, "strength": "weak"})
            created[cid].append(k)
        rep["shared"].append({"name": nodes[k].text, "who": ids})
    # relations are claims only: the relation graph (memory/relgraph.py) decides from all of them
    for r in ans.get("relations", []):
        a, b = ikey_to_id.get(r.get("a")), ikey_to_id.get(r.get("b"))
        if a and b and a != b and r.get("evidence"):
            rep["claims"].append((a, b, r["evidence"], None, r.get("type")))
    db.con.commit()
    return created, rep


def _shared_or_card(db: CharacterDB, keys, nodes, edges, created, rep, obs) -> dict:
    """docs/DESIGN.md: a title + surname the model did not place ("Miss Bennet") that two or more
    compatible cards are linked to (P1 same person, or the same surname) becomes their shared reference
    instead of a card of its own."""
    text = nodes[keys[0]].text
    sur = nodes[keys[0]].parsed.surname
    holders = []
    for cid, ks in created.items():
        linked = any(tuple(sorted((k, o))) in edges for k in keys for o in ks)
        same = any(nodes[o].parsed.surname == sur for o in ks)
        if (linked or same) and all(names.compatible(text, nodes[o].text).ok for o in ks) and cid not in holders:
            holders.append(cid)
    if len(holders) >= 2:
        for cid in holders:
            db.apply("alias", {"id": cid, "alias": text, "strength": "weak"})
        rep["shared"].append({"name": text, "who": holders})
        return {cid: created[cid] + list(keys) for cid in holders}
    name = _name(keys, nodes, None)
    cid = db.new_id(name)
    db.apply("create", {"id": cid, "name": name, "gender": _gender(keys, nodes, None),
                        "first_seen": min(nodes[k].chunks[0] for k in keys)})
    for k in keys:
        for t in nodes[k].texts:
            if t != name:
                db.apply("alias", {"id": cid, "alias": t, "strength": "medium"})
    db.apply("mention", {"id": cid, "n": sum(nodes[k].count for k in keys)})
    rep["individuals"].append({"id": cid, "name": name, "gender": None, "names": [nodes[k].text for k in keys],
                               "reason": "not placed by the model; no shared holders"})
    return {cid: list(keys)}


def _resolve_created(db: CharacterDB, created: dict) -> dict:
    out: dict[str, list[str]] = {}
    for cid, keys in created.items():
        r = db.resolve(cid)
        out.setdefault(r, [])
        out[r] += [k for k in keys if k not in out[r]]
    return out


def _relation_merges(db: CharacterDB, edges, nodes, created, support, accepted) -> list[tuple[str, str, str]]:
    """docs/DESIGN.md: two cards joined only by a one-passage P1 same-person claim are merged
    when their names are compatible and, in different passages, each has the same relation to the same
    third person (Charles and Mr. Bingley are both Miss Bingley's brother). Same relation alone is not
    enough (Jane and Lydia are both Elizabeth's sisters): the same-person claim must be there too."""
    owners = defaultdict(set)
    for cid, keys in created.items():
        for k in keys:
            owners[k].add(cid)
    roles: dict[str, dict] = defaultdict(dict)          # card -> (type, other) -> (passages, strong and accepted?)
    for (a, t, b), sup in support.items():
        if t not in ("sibling_of", "spouse_of"):            # parent claims from "her mother" are too noisy
            continue
        est = (a, t, b) in accepted and accepted[(a, t, b)]["strength"] == "strong"
        roles[a][(t, b)] = (sup["chunks"], est)
        roles[b][(t, a)] = (sup["chunks"], est)
    family: dict[str, dict] = defaultdict(lambda: defaultdict(set))   # accepted strong family ties
    for (a, t, b), v in accepted.items():
        if t in ("spouse_of", "sibling_of", "parent_of") and v["strength"] == "strong":
            family[a][b].add(t if t != "parent_of" else "parent")
            family[b][a].add(t if t != "parent_of" else "child")

    def contradict(c1, c2) -> str | None:
        """Merging would make one person e.g. both Elizabeth's mother and her sister."""
        for z in set(family[c1]) & set(family[c2]):
            if family[c1][z] != family[c2][z]:
                return f"{c1} is {'/'.join(sorted(family[c1][z]))} of {z}, {c2} is {'/'.join(sorted(family[c2][z]))}"
        for z in family[c1]:
            if z == c2 and family[c1][z]:
                return f"{c1} and {c2} are already related ({'/'.join(sorted(family[c1][z]))})"
        return None
    texts = {cid: [nodes[k].text for k in keys if len(owners[k]) == 1] for cid, keys in created.items()}
    out = []
    for (ka, kb), e in edges.items():
        p1 = [w[3:] for w in e["why"] if w.startswith("p1:")]
        if not p1 or len(owners[ka]) != 1 or len(owners[kb]) != 1:
            continue
        ca, cb = db.resolve(next(iter(owners[ka]))), db.resolve(next(iter(owners[kb])))
        if ca == cb or names.conflicts(texts.get(ca, []) + texts.get(cb, [])):
            continue
        card_a, card_b = db.card(ca), db.card(cb)
        keep, drop = (ca, cb) if card_a.mentions >= card_b.mentions else (cb, ca)
        if contradict(ca, cb):
            continue
        for role in set(roles[ca]) & set(roles[cb]):
            z = role[1]
            if db.resolve(z) in (ca, cb):
                continue
            (pa, ea), (pb, eb) = roles[ca][role], roles[cb][role]
            if (ea or eb) and any(x != y for x in pa for y in pb):
                why = (f"P1 same person ({', '.join(sorted(set(p1)))}); both {role[0]} {z} "
                       f"in different passages ({sorted(pa)[0]}, {sorted(pb)[0]})")
                out.append((drop, keep, why))
                db.apply("merge", {"from": drop, "into": keep, "reason": why})   # so later pairs see it
                break
    return out


def _reapply_manual(db: CharacterDB) -> None:
    """Manual edits (`readaloud characters --set/--merge/...`) win: after the rebuild, apply them again in order
    to the new cards (ids are stable while names are). Events whose target no longer exists are skipped."""
    for r in db.con.execute("SELECT type, payload FROM character_event WHERE source='manual' AND superseded_by IS NULL "
                            "AND type != 'create' ORDER BY id").fetchall():
        p = json.loads(r["payload"])
        target = p.get("id") or p.get("from")
        if target and not db.exists(target):
            continue
        getattr(db, f"_ev_{r['type']}")(p)
    db.con.commit()


def _full_name_check(created, nodes, obs):
    owners = defaultdict(set)
    for cid, keys in created.items():
        for k in keys:
            fk = names.full_name_key(nodes[k].text)
            if fk:
                owners[fk].add(cid)
    for fk, ids in owners.items():
        if len(ids) > 1:
            obs.warn("ALIAS_CONFLICT", "warn", None, f"roster: full name {' '.join(fk)} on {', '.join(sorted(ids))}",
                     alias=" ".join(fk), owner=sorted(ids), character_id=None)


def _write_report(book: Book, report, p1_conflicts, nodes, relrep=None) -> None:
    L = [f"# Roster report: {book.id}", "", "Name groups and the individuals decided for each (docs/DESIGN.md).", ""]
    for r in report:
        if len(r["names"]) == 1 and not r["splits"]:
            continue
        L += [f"## {', '.join(r['names'][:8])}{' …' if len(r['names']) > 8 else ''}", ""]
        for d in r["individuals"]:
            L.append(f"- **{d['id']}** {d['name']} ({d['gender'] or '?'}): {', '.join(d['names'])}"
                     + (f" — {d['reason']}" if d["reason"] else ""))
        for s in r["shared"]:
            L.append(f"- shared: {s['name']} → {', '.join(s['who'])}")
        for s in r["splits"]:
            L.append(f"- split by name rules: {' | '.join(', '.join(p) for p in s)}")
        for u in r["unplaced"]:
            L.append(f"- not placed by the model, grouped by name rules: {', '.join(u)}")
        if r["unsure"]:
            L.append(f"- unsure (no card): {', '.join(r['unsure'])}")
        L.append("")
    maybe = []
    for r in report:
        for d in r["individuals"]:
            kinds = {names.parse(n).kind for n in d["names"]}
            if kinds <= {"surname", "description", "occupation"}:
                maybe.append(f"{d['id']} ({', '.join(d['names'])})")
    if maybe:
        L += ["## Maybe not characters (places, institutions?) — mark with `readaloud characters --not-character ID`", "",
              *[f"- {m}" for m in maybe], ""]
    if relrep and relrep.get("anomalies"):
        L += ["## RELATION_ANOMALY: the relation graph is not used for attribution (docs/DESIGN.md)", "",
              *[f"- {a}" for a in relrep["anomalies"]], ""]
    if relrep and relrep.get("merges"):
        L += ["## Merged by relations (docs/DESIGN.md; undo with `readaloud characters --split ID`)", ""]
        L += [f"- {frm} → {into}: {why}" for frm, into, why in relrep["merges"]] + [""]
    if relrep and relrep.get("married"):
        L += ["## Married names (5.6 point 8)", ""]
        L += [f"- {m['wife']}: {m['alias']} (spouse_of {m['husband']}, from {m['valid_from']})" if "alias" in m
              else f"- {m['merged']} → {m['into']}: {m['why']}" for m in relrep["married"]] + [""]
    if relrep:
        L += ["## Relations (docs/DESIGN.md)", "",
              "Accepted, strongest first (passages; evidence kinds: named, p1, possessive). Derived relations are "
              "not stored. Every observation is in the relation_claim table; `readaloud relations ID --audit N` shows "
              "evidence.", ""]
        for st in ("strong", "weak"):
            rows = [x for x in relrep["accepted"] if x[3] == st]
            if rows:
                L += [f"{st.capitalize()} ({len(rows)}):", ""]
                L += [f"- {a} {t} {b}  ({n} passage{'s' if n != 1 else ''}; {', '.join(k)})" for (a, t, b), n, k, _ in rows]
                L.append("")
        if relrep["conflicts"]:
            L += ["Conflicts (the whole-graph check after derivation):", ""]
            L += [f"- {' '.join(c['dropped'])} ({c['strength']}): {c['why']}" for c in relrep["conflicts"]] + [""]
        if relrep["weak"]:
            L += ["Weak and far behind the best candidate (not used):", ""]
            L += [f"- {a} {t} {b}  ({n}; {', '.join(m)})" for (a, t, b), n, m in relrep["weak"]] + [""]
    if p1_conflicts:
        L += ["## P1 same-individual claims rejected by name rules", ""]
        L += [f"- {nodes[a].text} ≠ {nodes[b].text} ({n}x)" for (a, b), n in p1_conflicts.most_common()]
    (book.dir / "reports").mkdir(exist_ok=True)
    (book.dir / "reports" / "roster_report.md").write_text("\n".join(L) + "\n")
