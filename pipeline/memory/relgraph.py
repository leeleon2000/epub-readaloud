"""Build the relation graph by rules (docs/DESIGN.md). No model calls.

Claims come from P1 (`p1_person.related`, `p1_relation`) and P2 (the roster model's relations). A claim
only names two people; the model's type label is ignored. The type comes from a kin word in the text, and
every observation is stored in `relation_claim` with its kind of evidence (5.6 points 1-2):

  named       the kin word is tied by grammar to a named person, and the other person is named in the
              same sentence or the observation is the claim's own evidence sentence:
              "his friend Mr. Darcy", "my cousin, Mr. Collins", "Jane and her sister Elizabeth"
  p1          P1 says this reference is whose who: the person is "his wife" of the claimed anchor
  possessive  "her/his <kin word>" attached to the nearest fitting name: "Mrs. Hurst sang with her sister"

A relation is strong with one `named` observation, or with two different kinds from different passages.
Anything else is weak, whatever the number of passages, and is kept only if its passages are not far fewer
than the best candidate for the same person and relation (`relations.share_min`).

Relations enter one by one (manual, strong, weak; family before friend; more passages first) and each is
checked against the whole graph after derivation (kin.Graph.problems): a relation that makes someone both
parent and sibling of a person, a third parent, a generation cycle or a second spouse is a `conflict`.
Only basic relations are written (with strength and valid_from); derived ones are computed when queried.
`relations.yaml` in the book folder holds the human's word: `assert` first, `retract` never (5.6 point 7).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from . import names
from .db import CharacterDB, norm
from .kin import DIRECTED, Graph, kin_words, to_triple

POSS = r"(?:his|her|their|my|your|our)"
SYMMETRIC_RELS = {"spouse", "sibling", "in_law", "cousin", "friend"}
PRON_GENDER = {"his": "male", "her": "female"}
FAMILY = {"spouse_of", "sibling_of", "parent_of"}
# anomaly_max_kin: on pride-full the healthy graph peaks at 11 people per person, the contaminated one at 22
DEFAULTS = {"share_min": 0.3, "anomaly_conflict_ratio": 1.0, "anomaly_max_kin": 16, "anomaly_change": 0.6}


def _flat(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


class Cards:
    """Reference text -> card id, using the roster's aliases (shared/weak aliases do not decide)."""

    def __init__(self, db: CharacterDB):
        self.db = db
        self.by_norm: dict[str, set[str]] = defaultdict(set)
        self.texts: dict[str, set[str]] = defaultdict(set)
        for r in db.con.execute("SELECT a.character_id, a.alias, a.norm, a.strength FROM alias a JOIN character c "
                                "ON c.id=a.character_id WHERE c.status IN ('active', 'provisional')"):
            cid = db.resolve(r["character_id"])
            if r["strength"] != "weak":
                self.by_norm[r["norm"]].add(cid)
            if names.parse(r["alias"]).is_name:
                self.texts[cid].add(r["alias"])
        self.gender = {r["id"]: r["gender"] for r in db.con.execute("SELECT id, gender FROM character")}

    def of(self, text: str) -> str | None:
        ids = self.by_norm.get(norm(text), set())
        return next(iter(ids)) if len(ids) == 1 else None

    def named_in(self, cid: str, sentence: str) -> list[tuple[int, int]]:
        out = []
        for t in self.texts.get(cid, ()):
            for m in re.finditer(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", sentence):
                out.append((m.start(), m.end()))
        return out


def _attach(sentence: str, x: str, y: str, cards: Cards, modes: str = "ADB", paired: bool = False):
    """Kin words in `sentence` that say how x and y are related: [(kin, of_whom, rel, gender, kind)].
    A: "<kin word>[,] <Name>"              Name is the kin of the other one   -> named
       (the other one must be named in the sentence, unless `paired`: the claim's own evidence sentence)
    D: "<Name> ... <kin word> ... <Name>"  symmetric, at most 3 other words   -> named
    B: "<Name> ... his/her <kin word>"     the pronoun fits Name             -> possessive"""
    out = []
    spans = {x: cards.named_in(x, sentence), y: cards.named_in(y, sentence)}
    for s, e, w, (rel, g) in kin_words(sentence):
        after = re.sub(r"^(?:to\s+)?(?:a|an|the)\s+", "", sentence[e:].lstrip(" ,"))
        pm = re.search(r"\b(" + POSS + r")\s+(?:\w+\s+)?$", sentence[:s], re.I)
        poss = pm.group(1).lower() if pm else None
        for kin, other in ((x, y), (y, x)):
            if "A" not in modes or not any(after.startswith(sentence[a:b]) for a, b in spans[kin]):
                continue
            if not paired:
                # "my nephew, Mr. Darcy" (her Ladyship speaking) says nothing about Miss Bennet in the sentence;
                # "his/her <kin> <Name>" belongs to the other person only if named before it with that gender
                if poss in ("my", "your", "our", "their"):
                    continue
                if poss in PRON_GENDER:
                    before = [a for a, b in spans[other] if b <= pm.start()]
                    if not before or (cards.gender.get(other) in ("male", "female")
                                      and cards.gender[other] != PRON_GENDER[poss]):
                        continue
                elif not spans[other]:
                    continue
            out.append((kin, other, rel, g, "named"))
        if "D" in modes and rel in SYMMETRIC_RELS and spans[x] and spans[y]:
            for (a1, b1) in spans[x]:
                for (a2, b2) in spans[y]:
                    lo, hi = (b1, a2) if b1 <= a2 else (b2, a1)
                    gap = sentence[lo:s] + " " + sentence[e:hi]
                    if lo <= s and e <= hi and len(gap.split()) <= 3 and not re.search(r"[.!?;“”\"]", gap):
                        out.append((x, y, rel, g, "named"))
        m = re.search(r"\b(" + POSS + r")\s+(?:\w+\s+)?$", sentence[:s], re.I) if "B" in modes else None
        if m:
            pg = PRON_GENDER.get(m.group(1).lower())
            mentions = sorted(((a, who) for who in (x, y) for a, b in spans[who] if b <= m.start()), reverse=True)
            for _, owner in mentions:                              # nearest fitting mention owns the kin
                if pg and cards.gender.get(owner) in ("male", "female") and cards.gender[owner] != pg:
                    continue
                out.append((y if owner == x else x, owner, rel, g, "possessive"))
                break
    return out


def _sentence_id(evidence: str, index: list[tuple[str, str, str]]) -> tuple[str | None, str | None]:
    """(chunk id, sentence id) of the story sentence where `evidence` starts."""
    head = _flat(evidence)[:60]
    for cid, sid, flat in index:
        if head and head[:40] in flat:
            return cid, sid
    return None, None


def collect_claims(con, cards: Cards, chunks, sents: dict, p2_claims: list[tuple]) -> list[dict]:
    """-> [{x, y, chunk, sentences: [(sid, text)], refs, paired}] pairs said to be related, where to look."""
    claims = []
    rows = {c["id"]: [(s["id"], s["tts_text"]) for s in sents.get(c["id"], [])] for c in chunks}
    index = [(c, sid, _flat(t)) for c in rows for sid, t in rows[c]]
    for c in chunks:
        for r in con.execute("SELECT refs, related FROM p1_person WHERE chunk_id=?", (c["id"],)):
            refs = json.loads(r["refs"])
            ids = {cards.of(x["text"]) for x in refs if names.parse(x["text"]).is_name} - {None}
            rel_refs = [x_["text"] for x_ in refs if x_.get("rule_kind") == "relational"]
            for x in sorted(ids):      # several cards: the roster split this entry (5.2 decides about merging)
                for rr in json.loads(r["related"] or "[]"):
                    y = cards.of(rr.get("of_ref", ""))
                    if y and y != x:
                        claims.append({"x": x, "y": y, "chunk": c["id"], "sentences": rows[c["id"]],
                                       "refs": rel_refs if len(ids) == 1 else [], "paired": False,
                                       "label": rr.get("relation")})
        for r in con.execute("SELECT a, b, type, evidence FROM p1_relation WHERE chunk_id=?", (c["id"],)):
            x, y = cards.of(r["a"]), cards.of(r["b"])
            if x and y and x != y:
                _, sid = _sentence_id(r["evidence"], [i for i in index if i[0] == c["id"]])
                claims.append({"x": x, "y": y, "chunk": c["id"], "sentences": [(sid, r["evidence"])], "refs": [],
                               "paired": True, "label": r["type"]})
    for cl in p2_claims:
        x, y, evidence, chunk = cl[:4]
        if x and y and x != y:
            cid, sid = _sentence_id(evidence, index)
            claims.append({"x": cards.db.resolve(x), "y": cards.db.resolve(y), "chunk": chunk or cid or "p2",
                           "sentences": [(sid, evidence)], "refs": [], "paired": True,
                           "label": cl[4] if len(cl) > 4 else None})
    return claims


# The model's label for a claim never sets the type, but a possessive or P1 observation whose kin word is of
# another kind than the label probably speaks of someone else ("Miss Bingley ... her brother" for a claim
# that Miss Bingley and Mr. Darcy are friends).
LABEL_RELS = {"spouse_of": {"spouse"}, "sibling_of": {"sibling", "in_law"}, "parent_of": {"parent", "child"},
              "friend_of": {"friend"}, "cousin_of": {"cousin"}, "engaged_to": {"spouse"}, "courting": {"spouse"}}


def _label_fits(label, rel: str) -> bool:
    fits = LABEL_RELS.get((label or "").strip().lower())
    return fits is None or rel in fits


def analyze(db: CharacterDB, con, chunks, sents: dict, p2_claims: list[tuple]):
    """Every observation, grouped by candidate relation. No writes.
    -> ({(a, type, b): {"obs": [...], "chunks", "kinds", "strong", "first"}}, cards)"""
    cards = Cards(db)
    order = {}                                         # sentence id -> reading position (valid_from)
    kin_sents = []
    for c in chunks:
        for s in sents.get(c["id"], []):
            order[s["id"]] = len(order)
            if kin_words(s["tts_text"]):
                kin_sents.append((c["id"], s["id"], s["tts_text"]))
    claims = collect_claims(con, cards, chunks, sents, p2_claims)
    pairs = {frozenset((cl["x"], cl["y"])) for cl in claims}
    found = []          # (chunk, sentence id, kin, of, rel, gender, kind, kin word, evidence)
    for cl in claims:
        x, y = cl["x"], cl["y"]
        for ref in cl["refs"]:                                   # p1: x is "his wife" (of y, P1 says)
            for _, _, w, (rel, g) in kin_words(ref):
                if _label_fits(cl.get("label"), rel):
                    found.append((cl["chunk"], None, x, y, rel, g, "p1", w, ref))
        modes = "ADB" if cl["paired"] else "B"                   # possessives only inside the claiming passage
        for sid, snt in cl["sentences"]:
            for kin, of, rel, g, kind in _attach(snt, x, y, cards, modes=modes, paired=cl["paired"]):
                if kind == "possessive" and not _label_fits(cl.get("label"), rel):
                    continue
                w = next((ww for _, _, ww, rr in kin_words(snt) if rr[0] == rel), rel)
                found.append((cl["chunk"], sid, kin, of, rel, g, kind, w, snt))
    for pair in pairs:      # named evidence anywhere in scope, for any pair someone claimed
        x, y = sorted(pair)
        for cid, sid, snt in kin_sents:
            for kin, of, rel, g, kind in _attach(snt, x, y, cards, modes="AD"):
                w = next((ww for _, _, ww, rr in kin_words(snt) if rr[0] == rel), rel)
                found.append((cid, sid, kin, of, rel, g, kind, w, snt))
    cands: dict[tuple, dict] = {}
    for chunk, sid, kin, of, rel, g, kind, w, ev in found:
        if g and kind == "possessive" and cards.gender.get(kin) in ("male", "female") and cards.gender[kin] != g:
            continue                                         # "her sister" must be female
        if g and kind != "possessive" and rel not in SYMMETRIC_RELS and cards.gender.get(kin) in ("male", "female") \
                and cards.gender[kin] != g:
            continue
        t = to_triple(kin, rel, of)
        if not t or not _name_ok(db, t):
            continue
        c = cands.setdefault(t, {"obs": [], "chunks": set(), "kinds": set(), "strong": False, "first": None})
        key = (chunk, sid, kind, ev)
        if key in {(o["chunk"], o["sentence"], o["kind"], o["evidence"]) for o in c["obs"]}:
            continue
        c["obs"].append({"chunk": chunk, "sentence": sid, "kind": kind, "kin": w, "evidence": ev})
        c["chunks"].add(chunk)
        c["kinds"].add(kind)
        if sid and (c["first"] is None or order.get(sid, 1e9) < order.get(c["first"], 1e9)):
            c["first"] = sid
    for c in cands.values():
        by_kind = defaultdict(set)
        for o in c["obs"]:
            by_kind[o["kind"]].add(o["chunk"])
        two_kinds = any(a != b for k1 in by_kind for k2 in by_kind if k1 != k2
                        for a in by_kind[k1] for b in by_kind[k2])
        c["strong"] = "named" in c["kinds"] or two_kinds
    return cands, cards


def load_manual(book_dir: Path | None) -> dict:
    """relations.yaml: {assert: [{a, type, b, note}], retract: [...]} (card ids)."""
    p = Path(book_dir) / "relations.yaml" if book_dir else None
    data = yaml.safe_load(p.read_text()) if p and p.exists() else {}
    data = data or {}

    def key(r):
        t = (r["a"], r["type"], r["b"])
        return t if r["type"] in DIRECTED else (min(t[0], t[2]), t[1], max(t[0], t[2]))
    return {"assert": [key(r) for r in data.get("assert", []) or []],
            "retract": {key(r) for r in data.get("retract", []) or []}}


def decide(cands: dict, cards: Cards, manual: dict | None = None, cfg: dict | None = None) -> dict:
    """Share threshold for weak candidates, then one-by-one entry through the whole-graph check.
    -> {"accepted": {t: {"strength", "source"}}, "conflicts": [...], "weak": [...], "retracted": [...]}"""
    cfg = {**DEFAULTS, **(cfg or {})}
    manual = manual or {"assert": [], "retract": set()}
    # share threshold: a weak candidate must not be far behind the best candidate for the same person + relation
    # Only single-valued roles compete (a person has one spouse, one father, one mother); siblings and
    # children are many and naturally unequal in how often the text names them.
    best = defaultdict(int)
    role = {}
    for t, c in cands.items():
        a, typ, b = t
        if typ == "spouse_of":
            keys = [(a, typ, "s"), (b, typ, "s")]
        elif typ == "parent_of":
            keys = [(b, typ, cards.gender.get(a) or "?")]
        else:
            keys = []
        role[t] = keys
        for k in keys:
            best[k] = max(best[k], len(c["chunks"]))
    weak_out = []
    entries = []
    for t in manual["assert"]:
        entries.append((0, 0, 0, t, "manual"))
    for t, c in cands.items():
        if t in manual["retract"] or t in manual["assert"]:
            continue
        n = len(c["chunks"])
        if not c["strong"] and any(n < cfg["share_min"] * best[k] for k in role[t]):
            weak_out.append(t)
            continue
        entries.append((1 if c["strong"] else 2, 0 if t[1] in FAMILY else 1, -n, t, "strong" if c["strong"] else "weak"))
    accepted: dict[tuple, dict] = {}
    conflicts = []
    known: set[str] = set()                 # problems already there (e.g. from a manual assertion)
    for _, _, _, t, strength in sorted(entries):
        a, typ, b = t
        pair = {a, b}
        clash = next((k for k in accepted if {k[0], k[2]} == pair and k[1] != typ), None)
        if clash and strength != "manual":
            conflicts.append({"dropped": list(t), "why": f"{clash[1]} already holds for this pair",
                              "strength": strength})
            continue
        trial = {**accepted, t: {"strength": "strong" if strength in ("strong", "manual") else "weak"}}
        g = Graph([k for k, v in trial.items() if v["strength"] == "strong"], cards.gender,
                  weak=[k for k, v in trial.items() if v["strength"] == "weak"])
        new = [p for p in g.problems() if p not in known]
        if new and strength != "manual":
            conflicts.append({"dropped": list(t), "why": new[0], "strength": strength})
            continue
        known |= set(new)
        accepted[t] = {"strength": trial[t]["strength"], "source": "manual" if strength == "manual" else "p3"}
    return {"accepted": accepted, "conflicts": conflicts, "weak": weak_out,
            "retracted": [t for t in cands if t in manual["retract"]]}


def anomaly(dec: dict, cards: Cards, previous: set, grew: bool, cfg: dict | None = None) -> list[str]:
    """RELATION_ANOMALY reasons (5.6 point 6): too many conflicts for what was accepted, someone with
    family ties to too much of the roster, or a graph that changed a lot when the scope grew."""
    cfg = {**DEFAULTS, **(cfg or {})}
    acc = dec["accepted"]
    out = []
    n_strong = sum(1 for v in acc.values() if v["strength"] == "strong")
    strong_conf = sum(1 for c in dec["conflicts"] if c.get("strength") == "strong")
    if n_strong and strong_conf > cfg["anomaly_conflict_ratio"] * n_strong:      # weak noise is expected to bounce
        out.append(f"{strong_conf} conflicting strong relations for {n_strong} accepted")
    g = Graph([t for t, v in acc.items() if v["strength"] == "strong"], cards.gender)
    for x in g.people():
        kin = set()
        for rel in ("spouse", "parent", "child", "sibling", "in_law", "aunt_uncle", "niece_nephew", "cousin"):
            kin |= g.relation(x, rel, weak=False)
        if len(kin) > cfg["anomaly_max_kin"]:
            out.append(f"{x} has family ties to {len(kin)} people")
            break
    now = set(acc)
    if grew and previous:
        change = len(now ^ previous) / max(len(now), len(previous))
        if change > cfg["anomaly_change"]:
            out.append(f"the relation graph changed by {change:.0%} after the scope grew")
    return out


def write(db: CharacterDB, cands: dict, dec: dict, run_id: str | None = None, obs=None) -> dict:
    """relation events for the accepted relations (no derived ones) and every observation in relation_claim."""
    con = db.con
    con.execute("DELETE FROM relation_claim")
    acc = dec["accepted"]
    conflict = {tuple(c["dropped"]): c["why"] for c in dec["conflicts"]}
    retracted = set(dec["retracted"])
    for t, c in cands.items():
        status, reason = ("accepted", acc[t]["strength"]) if t in acc else \
            ("conflict", conflict[t]) if t in conflict else ("retracted", "relations.yaml") if t in retracted \
            else ("weak", "far fewer passages than the best candidate for the same person")
        con.executemany("INSERT INTO relation_claim(a, type, b, kin, chunk_id, sentence_id, kind, evidence, status, "
                        "reason, run_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [(t[0], t[1], t[2], o["kin"], o["chunk"], o["sentence"], o["kind"], o["evidence"][:500],
                          status, reason, run_id) for o in c["obs"]])
    for t, v in sorted(acc.items()):
        c = cands.get(t, {"obs": [], "chunks": set(), "kinds": set(), "first": None})
        ev = " | ".join(o["evidence"] for o in c["obs"][:3])[:500]
        db.apply("relation", {"a": t[0], "type": t[1], "b": t[2], "evidence": ev, "source": v["source"],
                              "strength": v["strength"], "valid_from": c["first"], "support": len(c["chunks"]),
                              "kinds": sorted(c["kinds"])})
    for c in dec["conflicts"]:
        if obs is not None:
            obs.warn("RELATION_CONFLICT", "warn", None, f"relations: dropped {c['dropped']} ({c['why']})")
    con.commit()
    return {"accepted": [(t, len(cands.get(t, {}).get("chunks", ())), sorted(cands.get(t, {}).get("kinds", ())),
                          v["strength"]) for t, v in sorted(acc.items())],
            "conflicts": dec["conflicts"],
            "weak": sorted(((t, len(cands[t]["chunks"]), sorted(cands[t]["kinds"])) for t in dec["weak"]),
                           key=lambda x: -x[1])}


def build(db: CharacterDB, con, chunks, sents: dict, p2_claims: list[tuple], obs=None, book_dir=None,
          cfg: dict | None = None) -> dict:
    """analyze + decide + write (tests and simple callers)."""
    cands, cards = analyze(db, con, chunks, sents, p2_claims)
    dec = decide(cands, cards, load_manual(book_dir), cfg)
    return write(db, cands, dec, getattr(obs, "run_id", None), obs)


def married_names(db: CharacterDB, accepted: dict) -> list[dict]:
    """5.6 point 8: a strong spouse_of gives the wife "Mrs. <husband's surname>" from the first evidence on,
    and a card that is only that name (Mrs. Collins) joins her. Two wives with one married name (Mrs. Darcy
    the mother and the bride): alias only, no merge."""
    wives = defaultdict(list)                  # married name -> [(wife, husband, valid_from)]
    for (a, typ, b), v in accepted.items():
        if typ != "spouse_of" or v["strength"] != "strong":
            continue
        ca, cb = db.card(a), db.card(b)
        if not ca or not cb:
            continue
        ga = ca.gender or names.parse(ca.name).gender
        gb = cb.gender or names.parse(cb.name).gender
        if {ga, gb} != {"male", "female"}:
            continue
        hus, wife = (ca, cb) if ga == "male" else (cb, ca)
        sur = next((names.parse(t).surname for t in [hus.name, *hus.aliases] if names.parse(t).surname), None)
        text = next((t for t in [hus.name, *hus.aliases] if names.parse(t).surname == sur), None)
        if not sur or not text:
            continue
        surname_text = re.sub(r"['’]s$", "", text.split()[-1]).strip(".,")
        wives[f"Mrs. {surname_text}"].append((wife.id, hus.id, v.get("valid_from"), sur))
    out = []
    for married, ws in wives.items():
        nm = norm(married)
        for wife, hus, vf, _ in ws:
            if nm not in {norm(a) for a in db.card(wife).aliases}:
                db.apply("alias", {"id": wife, "alias": married, "strength": "medium" if len(ws) == 1 else "weak",
                                   "derived_from": f"{wife} spouse_of {hus}", "valid_from": vf})
                out.append({"wife": wife, "alias": married, "husband": hus, "valid_from": vf})
        if len(ws) != 1:
            continue
        wife, hus, _, sur = ws[0]
        given = {f for a in db.card(wife).aliases if names.parse(a).given for f in names.formal_names(names.parse(a).given)}
        for cid in db.active_ids():
            if cid in (wife, hus):
                continue
            card = db.card(cid)
            if not card:
                continue
            # only her married name: "Mrs. Collins", or her first name with his surname ("Lydia Wickham")
            ps = [names.parse(a) for a in card.aliases]
            if ps and all(p.surname == sur and p.title in (None, "mrs")
                          and (p.given is None or names.formal_names(p.given) & given) for p in ps) \
                    and (any(p.given for p in ps) or {norm(a) for a in card.aliases} == {nm}):
                why = f"married name: {married} is {wife} (spouse_of {hus})"
                db.apply("merge", {"from": cid, "into": wife, "reason": why})
                out.append({"merged": cid, "into": wife, "why": why})
    db.con.commit()
    return out


def _name_ok(db: CharacterDB, t) -> bool:
    """Mr. X and Mrs. X (same surname, opposite title genders) can only be spouses."""
    a, b = db.card(t[0]), db.card(t[2])
    if not a or not b:
        return False
    pa, pb = names.parse(a.name), names.parse(b.name)
    if pa.surname and pa.surname == pb.surname and pa.gender and pb.gender and pa.gender != pb.gender \
            and pa.title in ("mr", "mrs") and pb.title in ("mr", "mrs"):
        return t[1] == "spouse_of"
    return True


def graph_from_db(con, resolve=lambda x: x, gender: dict | None = None) -> Graph:
    """The relation table as a Graph: strong relations derive, weak ones count only as themselves."""
    rows = [(resolve(r["a"]), r["type"], resolve(r["b"]), r["strength"] or "strong")
            for r in con.execute("SELECT a, type, b, strength FROM relation")]
    if gender is None:
        gender = {r["id"]: r["gender"] for r in con.execute("SELECT id, gender FROM character")}
    return Graph([r[:3] for r in rows if r[3] == "strong"], gender, weak=[r[:3] for r in rows if r[3] != "strong"])


def summary_counts(con) -> Counter:
    return Counter(r["status"] for r in con.execute("SELECT status FROM relation_claim"))
