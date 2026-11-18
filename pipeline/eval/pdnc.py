"""Score characters + attribute against PDNC (Project Dialogism Novel Corpus).

See docs/DESIGN.md. PDNC data is used here only: never in prompts or seeds.

    result = evaluate(con, cfg, pdnc_dir, overrides, chapters=(1, 13))
    write_reports(result, out_dir, tag)          # <tag>.json / <tag>.md / <tag>.quotes.csv
    compare([result_a, result_b, ...])           # markdown: metrics side by side + per-quote flips

Matching is by text, not offsets: both sides are reduced to lowercase alphanumerics separated by
single spaces, and each PDNC sub-quotation is searched forward from the previous match.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ..memory.db import norm
from ..quotes import detect_style, split_paragraph
from ..store import get_meta

RESULT_VERSION = 1
WINDOW = 40_000            # max normalized chars between consecutive PDNC matches
HEAD = re.compile(r"^\s*chapter\s+([ivxlcdm]+|\d+)\s*[.:]?\s*$", re.I)
MAIN = ("major", "intermediate")
CONF_BINS = [("≥0.8", 0.8, 9), ("0.6–0.8", 0.6, 0.8), ("<0.6", -1, 0.6)]
NON_SPEAKERS = ("narrator", "unknown")
ATTRIBUTED = ("model", "turns", "manual", "tag", "paragraph", "scene", "vocative", "none")   # utterance.source values that carry a speaker decision (v3: tag ... none)
V3_MALE, V3_FEMALE = "_generic_male", "_generic_female"
CURVE = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def roman(s: str) -> int:
    if s.isdigit():
        return int(s)
    v = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    s = s.lower()
    return sum(-v[c] if i + 1 < len(s) and v[s[i + 1]] > v[c] else v[c] for i, c in enumerate(s))


# In PDNC's novel_text.txt: "CHAPTER IV." alone on a line, "CHAPTER" with the number on the next line, or
# "Chapter IV: Fourth Chapter" (A Room with a View). The titled form must be capitalised as a heading.
PDNC_HEAD = re.compile(r"^[ \t]*(?:(?i:chapter)\s+((?i:[ivxlcdm]+)|\d+)[ \t]*[.:]?[ \t]*$"
                       r"|(?:CHAPTER|Chapter)[ \t]+([IVXLCDM]+|\d+)[ \t]*[.:][ \t]+\S.*$)", re.M)
HEAD_TITLED = re.compile(r"^\s*chapter\s+([ivxlcdm]+|\d+)\s*[.:]\s+\S", re.I)


def chapter_of_heading(text: str, heading: bool = False) -> int | None:
    """Chapter number of "CHAPTER IV." / "Chapter 4"; for a heading sentence also "CHAPTER IV. TITLE ..."."""
    m = HEAD.match(text) or (HEAD_TITLED.match(text) if heading else None)
    return roman(m.group(1)) if m else None


class ChapterCounter:
    """Chapter numbers that keep counting across volumes (Vol. II starts again at "Chapter I")."""

    def __init__(self):
        self.last, self.offset = 0, 0

    def see(self, n: int) -> int:
        if n <= self.last - self.offset:
            self.offset = self.last
        self.last = self.offset + n
        return self.last


# Editions differ on to-night / tonight, to-morrow, to-day. Dropping that hyphen on both sides makes them
# one word; otherwise a short quote ("Nothing to-night,") finds no exact match nearby, matches a later
# occurrence instead, and every quote after it is searched from the wrong place (Oliver Twist, ch. XXVI).
_SOFT_HYPHEN = re.compile(r"\bto(-)(?=(?:night|morrow|day)\b)", re.I)


def soft_hyphens(text: str) -> set[int]:
    """Positions of hyphens that both editions' spellings should ignore."""
    return {m.start(1) for m in _SOFT_HYPHEN.finditer(text)}


def normtext(text: str) -> tuple[str, list[int]]:
    """Lowercase alphanumerics, everything else collapsed to one space; plus orig index per char."""
    out, idx = [], []
    skip = soft_hyphens(text)
    for k, ch in enumerate(text):
        if k in skip:
            continue
        c = ch.lower()
        if c.isalnum():
            out.append(c)
            idx.append(k)
        elif out and out[-1] != " ":
            out.append(" ")
            idx.append(k)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


# ---------------------------------------------------------------- gold side

@dataclass
class GoldChar:
    name: str
    gender: str
    category: str
    aliases: set[str] = field(default_factory=set)


def _lit(s: str):
    s = (s or "").strip()
    return ast.literal_eval(s) if s else []


def load_gold(pdnc_dir: Path, overrides: Path | None = None):
    """(chars, quotes, applied overrides, ambiguous alias norms)."""
    pdnc_dir = Path(pdnc_dir)
    chars: dict[str, GoldChar] = {}
    with open(pdnc_dir / "character_info.csv", newline="") as f:
        for r in csv.DictReader(f):
            chars[r["Main Name"]] = GoldChar(r["Main Name"], r["Gender"], r["Category"],
                                             set(_lit(r["Aliases"])) | {r["Main Name"]})
    text = (pdnc_dir / "novel_text.txt").read_text()
    cc = ChapterCounter()
    heads = [(m.start(), cc.see(roman(m.group(1) or m.group(2))))
             for m in PDNC_HEAD.finditer(text)]

    def chap(pos):
        c = 0
        for o, n in heads:
            if o <= pos:
                c = n
        return c

    quotes = []
    with open(pdnc_dir / "quotation_info.csv", newline="") as f:
        for r in csv.DictReader(f):
            subs, spans = _lit(r["subQuotationList"]), _lit(r["quoteByteSpans"])
            ments = set()
            for group in _lit(r.get("mentionEntitiesList", "")):
                for ents in group:
                    ments.update(ents if isinstance(ents, (list, tuple, set)) else [ents])
            quotes.append({"id": r["quoteID"], "text": r["quoteText"], "subs": subs,
                           "start": spans[0][0] if spans else 0, "speaker": r["speaker"],
                           "addressees": list(_lit(r["addressees"])), "type": r["quoteType"],
                           "mentions": sorted(ments)})
    quotes.sort(key=lambda q: q["start"])
    for q in quotes:
        q["chapter"] = chap(q["start"])

    applied = []
    if overrides and Path(overrides).exists():
        ov = yaml.safe_load(Path(overrides).read_text()) or {}

        def need(e):
            if not e.get("reason"):
                raise SystemExit(f"{overrides}: every override needs a reason: {e}")
            applied.append(e)

        for e in ov.get("split", []):
            need(e)
            chars[e["from"]].aliases -= set(e["aliases"])
            chars[e["name"]] = GoldChar(e["name"], e.get("gender", "X"), e.get("category", "minor"),
                                        set(e["aliases"]) | {e["name"]})
        for e in ov.get("remove_alias", []):
            need(e)
            for a in _as_list(e["alias"]):
                chars[e["character"]].aliases.discard(a)
        for e in ov.get("add_alias", []):
            need(e)
            chars[e["character"]].aliases |= set(_as_list(e["alias"]))
        for e in ov.get("set", []):
            need(e)
            g = chars[e["character"]]
            g.gender = e.get("gender", g.gender)
            g.category = e.get("category", g.category)
        relabel = {}
        for e in ov.get("quote_speaker", []):
            need(e)
            relabel[e["quote"]] = e["speaker"]
        for q in quotes:
            if q["id"] in relabel:
                q["speaker"] = relabel[q["id"]]
    owners = defaultdict(set)
    for g in chars.values():
        for a in g.aliases:
            owners[norm(a)].add(g.name)
    ambiguous = {n: sorted(v) for n, v in owners.items() if len(v) > 1}
    # some PDNC files label quotes with an alias instead of the main name (A Room with a View: "Lucy" 505
    # quotes, main name "Miss Honeychurch"; "Cecil" 96): map a label that is one character's alias to that
    # character, else the quote counted as an unknown speaker (2026-11-16)
    for q in quotes:
        sp = q["speaker"]
        if sp and not sp.startswith("_") and sp not in chars and len(owners.get(norm(sp), ())) == 1:
            q["speaker_label"], q["speaker"] = sp, next(iter(owners[norm(sp)]))
    return chars, quotes, applied, ambiguous


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


# ---------------------------------------------------------------- our side

def _sentence_chapters(con) -> dict[tuple[int, int], int]:
    """(doc, seq) -> chapter number of the last chapter heading at or before it (0 = none yet)."""
    out, cur, cc = {}, 0, ChapterCounter()
    for r in con.execute("SELECT doc_id, seq, text, kind FROM sentence ORDER BY doc_id, seq"):
        n = chapter_of_heading(r["text"], r["kind"] == "heading")
        if n is not None:
            cur = cc.see(n)
        out[(r["doc_id"], r["seq"])] = cur
    return out


def _story_items(con):
    """[(sentence row, [Piece])] for all story sentences, as attribute splits them."""
    if con.execute("SELECT 1 FROM section LIMIT 1").fetchone():
        from ..stages.classify import story_filter_sql
        rows = con.execute(f"SELECT * FROM sentence s WHERE {story_filter_sql('s')} ORDER BY s.doc_id, s.seq")
    else:
        rows = con.execute("SELECT * FROM sentence WHERE speak=1 ORDER BY doc_id, seq")
    rows = rows.fetchall()
    style = get_meta(con, "quote_style") or detect_style(
        [r["tts_text"] for r in con.execute("SELECT tts_text FROM sentence")])
    by_para = defaultdict(list)
    for s in rows:
        by_para[(s["doc_id"], s["para"])].append(s)
    items = []
    for (doc, p), ss in sorted(by_para.items()):
        items += list(zip(ss, split_paragraph([s["tts_text"] for s in ss], style, f"{doc:03d}p{p:04d}")))
    return items


class Stream:
    """Our story text, normalized, with the quote id / sentence index of every char."""

    def __init__(self, items):
        self.items = items
        chars, self.qid, self.sent = [" "], [None], [-1]
        self.quote_first: dict[str, int] = {}          # quote id -> sentence index
        for i, (s, pcs) in enumerate(items):
            t = s["tts_text"]
            owner = [None] * len(t)
            for p in pcs:
                if p.is_quote:
                    self.quote_first.setdefault(p.quote_id, i)
                    for k in range(p.start, p.end):
                        owner[k] = p.quote_id
            skip = soft_hyphens(t)
            for k, ch in enumerate(t):
                if k in skip:
                    continue
                c = ch.lower()
                if c.isalnum():
                    chars.append(c)
                elif chars[-1] != " ":
                    chars.append(" ")
                else:
                    continue
                self.qid.append(owner[k])
                self.sent.append(i)
            if chars[-1] != " ":
                chars.append(" ")
                self.qid.append(None)
                self.sent.append(i)
        self.text = "".join(chars)
        self.words, self.wstart = [], []          # words of self.text and their char offsets
        for m in re.finditer(r"\S+", self.text):
            self.words.append(m.group())
            self.wstart.append(m.start())

    def _word_at(self, pos: int) -> int:
        import bisect
        return bisect.bisect_left(self.wstart, pos)

    def fuzzy(self, sub: str, cur: int, min_share: float = 0.6):
        """Edition differences (chuse/choose, mama/mamma): vote on the offset where sub's words line up."""
        sw = sub.split()
        if len(sw) < 3:
            return None
        w0 = self._word_at(cur)
        win = self.words[w0:w0 + WINDOW // 5]
        where = defaultdict(list)
        for i, w in enumerate(win):
            where[w].append(i)
        votes = Counter()
        for j, w in enumerate(sw):
            for i in where.get(w, ()):
                votes[i - j] += 1
        if not votes:
            return None
        from difflib import SequenceMatcher
        scored = []
        for o, _ in votes.most_common(8):     # re-score the best offsets allowing a few inserted/dropped words
            lo = max(o - 3, 0)
            seg = win[lo:o + len(sw) + 3]
            blocks = [m for m in SequenceMatcher(None, seg, sw, autojunk=False).get_matching_blocks() if m.size]
            if not blocks:
                continue
            n = sum(m.size for m in blocks)
            a_, b_ = lo + blocks[0].a, lo + blocks[-1].a + blocks[-1].size - 1
            scored.append((n, -o, a_, b_))
        if not scored:
            return None
        best = max(x[0] for x in scored)
        if best < min_share * len(sw):
            return None

        def span(a_, b_):
            return self.wstart[w0 + a_], self.wstart[w0 + b_] + len(self.words[w0 + b_])

        cands = sorted((x for x in scored if x[0] == best), key=lambda x: -x[1])
        for _, _, a_, b_ in cands:
            x, y = span(a_, b_)
            if any(self.qid[x:y]):
                return x, y
        return span(cands[0][2], cands[0][3])

    def find(self, sub: str, cur: int):
        """(start, end) of sub (normalized, no padding) at/after cur, on word boundaries.
        Short strings prefer a hit that lies inside one of our quotes."""
        if not sub:
            return None
        # The shorter the string, the nearer it must be: a short one that matches far ahead drags the cursor
        # with it and nothing after it is found. A long quote can still re-sync across a long gap.
        near = WINDOW if len(sub) >= 40 else 8000 if len(sub) >= 15 else 3000
        pat, lim = f" {sub} ", cur + near + len(sub) + 2
        first, i = None, self.text.find(pat, max(cur - 1, 0), lim)
        if len(sub) < 15:
            # A very short quote ("I?", "Yes.") must be a whole quote of ours, not a word inside a longer one:
            # "I" inside “Yes. I understand” moved the cursor past the real “I?” and the rest of the book was
            # searched from the wrong place (The Mysterious Affair at Styles, ch. VI).
            k = i
            while k != -1:
                a, b = k + 1, k + 1 + len(sub)
                q = self.qid[a]
                if q and all(x == q for x in self.qid[a:b]) and self.qid[a - 2] != q and self.qid[min(b + 1, len(self.qid) - 1)] != q:
                    return a, b
                k = self.text.find(pat, k + 1, lim)
            if len(sub) < 5:
                return None
        while i != -1:
            hit = (i + 1, i + 1 + len(sub))
            if first is None:
                first = hit
                if len(sub) >= 40:
                    break
            if any(self.qid[hit[0]:hit[1]]):
                return hit
            i = self.text.find(pat, i + 1, lim)
        if first:
            return first
        if len(sub) >= 30:      # edition differences inside a long quote: anchor on both ends
            a = self.text.find(f" {sub[:20]}", max(cur - 1, 0), lim)
            if a != -1:
                b = self.text.find(f"{sub[-20:]} ", a, a + len(sub) + 200)
                if b != -1:
                    return a + 1, b + 20
        return None


# ---------------------------------------------------------------- evaluation

def align(st: Stream, quotes: list[dict]) -> set[str]:
    """Locate every PDNC quote in our text (in reading order). Sets on each quote: overlap (our quote
    id -> chars), found (sub-quotations found), fuzzy, sentence (our first sentence id), aligned.
    Returns our quote ids touched by any PDNC quote."""
    cur, touched = 0, set()
    for q in quotes:
        q["overlap"], q["found"], q["sentence"] = Counter(), 0, None
        for sub in q["subs"]:
            n, _ = normtext(sub)
            hit = st.find(n, cur)
            if not hit:
                hit = st.fuzzy(n, cur)
                q["fuzzy"] = q.get("fuzzy", 0) + bool(hit)
            if not hit:
                continue
            q["found"] += 1
            cur = hit[1]
            for k in range(*hit):
                if st.qid[k]:
                    q["overlap"][st.qid[k]] += 1
            if q["sentence"] is None:
                q["sentence"] = st.items[st.sent[hit[0]]][0]["id"]
        q["aligned"] = q["found"] == len(q["subs"]) and q["found"] > 0
        touched |= set(q["overlap"])
    return touched


def evaluate(con, pdnc_dir: Path, overrides: Path | None = None, chapters: tuple[int, int] | None = None,
             cfg: dict | None = None, relations: Path | None = None) -> dict:
    lo, hi = chapters or (0, 10 ** 6)       # no --chapters: every quote, also where no chapter heading was found
    in_range = lambda ch: lo <= ch <= hi                                    # noqa: E731
    gold, quotes, applied, ambiguous = load_gold(pdnc_dir, overrides)
    sent_ch = _sentence_chapters(con)
    items = _story_items(con)
    st = Stream(items)
    ch_of_item = [sent_ch[(s["doc_id"], s["seq"])] for s, _ in items]

    touched = align(st, quotes)
    rq = [q for q in quotes if in_range(q["chapter"])]
    our_q = {qid: i for qid, i in st.quote_first.items() if in_range(ch_of_item[i])}
    missed = [q for q in rq if q["found"] and not q["overlap"]]
    spurious = [qid for qid in our_q if qid not in touched]

    # --- our predictions
    resolve_map = {r["id"]: r["merged_into"] for r in con.execute("SELECT id, merged_into FROM character")}

    def resolve(cid):
        seen = set()
        while cid and resolve_map.get(cid) and cid not in seen:
            seen.add(cid)
            cid = resolve_map[cid]
        return cid

    pred = {}
    for r in con.execute("SELECT quote_id, speaker_id, confidence, source, voice_gender, model_speaker FROM utterance "
                         "WHERE is_quote=1 ORDER BY sentence_id, seq"):
        if r["quote_id"] not in pred:
            sp = r["speaker_id"]
            pred[r["quote_id"]] = {"speaker": sp if sp in NON_SPEAKERS else resolve(sp),
                                   "confidence": r["confidence"], "source": r["source"],
                                   "voice_gender": r["voice_gender"], "who": r["model_speaker"]}
    has_attr = any(qid in pred and pred[qid]["source"] in ATTRIBUTED for qid in our_q)

    # --- cards in range
    chunk_ch = {r["id"]: sent_ch.get((r["doc_id"], r["seq_from"]), 0)
                for r in con.execute("SELECT id, doc_id, seq_from, seq_to, pass1, pass2 FROM chunk")}
    cards = {}
    for r in con.execute("SELECT * FROM character WHERE status IN ('active', 'provisional')"):
        fs = chunk_ch.get(r["first_seen"])
        if fs is None or fs <= hi:
            cards[r["id"]] = {"id": r["id"], "name": r["name"], "status": r["status"],
                              "gender": r["gender"], "first_seen": r["first_seen"], "aliases": [r["name"]]}
    for r in con.execute("SELECT character_id, alias FROM alias"):
        cid = resolve(r["character_id"])
        if cid in cards and r["alias"] not in cards[cid]["aliases"]:
            cards[cid]["aliases"].append(r["alias"])
    index = {}
    for g in gold.values():
        for a in g.aliases:
            n = norm(a)
            if n and n not in ambiguous:
                index[n] = g.name
    alias_err, n_name_alias = [], 0
    for c in cards.values():
        hits = defaultdict(list)
        for a in c["aliases"]:
            if norm(a) in index:
                hits[index[norm(a)]].append(a)
        c["name_hits"] = {g: v for g, v in hits.items()}
        c["map"] = max(hits, key=lambda g: (len(hits[g]), norm(c["name"]) in map(norm, hits[g]))) if hits else None
        n_name_alias += sum(len(v) for v in hits.values())
        for g, v in hits.items():
            if g != c["map"]:
                alias_err += [{"card": c["id"], "alias": a, "belongs_to": g, "card_map": c["map"]} for a in v]
    name_map = {c["id"]: c["map"] for c in cards.values()}

    # --- attribution rows (in range)
    # PDNC labels some quotes with a placeholder instead of a character (_group: several people at once,
    # _unknowable: the annotators could not tell). No answer of ours can match those, so they are left out
    # of the attribution scores and only counted.
    rows, placeholder = [], Counter()
    for q in rq:
        if q["speaker"].startswith("_"):
            placeholder[q["speaker"]] += 1
            continue
        top = q["overlap"].most_common(1)[0][0] if q["overlap"] else None
        p = pred.get(top) if top else None
        pc = p["speaker"] if p and p["source"] in ATTRIBUTED else None
        pg = None if pc in (None, *NON_SPEAKERS) else name_map.get(pc)
        rows.append({"quote": q["id"], "chapter": q["chapter"], "type": q["type"], "gold": q["speaker"],
                     "our_quote": top, "pred_card": pc, "pred": pg, "correct": pg == q["speaker"],
                     "confidence": p["confidence"] if p else None, "aligned": q["aligned"],
                     "sentence": q["sentence"], "text": q["text"].replace("\n", " ")[:100]})

    quote_hits = defaultdict(Counter)
    for r in rows:
        if r["pred_card"] and r["pred_card"] not in NON_SPEAKERS:
            quote_hits[r["pred_card"]][r["gold"]] += 1
    for c in cards.values():
        qh = quote_hits.get(c["id"], Counter())
        c["quote_hits"] = dict(qh.most_common())
        c["quote_map"] = qh.most_common(1)[0][0] if qh else None
        tot = sum(qh.values())
        second = qh.most_common(2)[1][1] if len(qh) > 1 else 0
        c["flags"] = []
        if len(c["name_hits"]) >= 2:
            c["flags"].append("merge(name)")
        if second >= 2 and second / tot >= 0.2:
            c["flags"].append("merge(quotes)")
        if c["quote_map"] and c["map"] and c["quote_map"] != c["map"]:
            c["flags"].append("map-disagree")

    # --- gold characters in range
    present = Counter()
    for q in rq:
        for n in [q["speaker"], *q["addressees"], *q["mentions"]]:
            if n in gold:
                present[n] += 1
    mains = sorted(n for n in present if gold[n].category in MAIN)
    cover = {}
    for g in mains:
        act = [c["id"] for c in cards.values() if c["map"] == g and c["status"] == "active"]
        prov = [c["id"] for c in cards.values() if c["map"] == g and c["status"] == "provisional"]
        cover[g] = {"category": gold[g].category, "active": act, "provisional": prov,
                    "state": "ok" if len(act) == 1 else "split" if len(act) > 1 else
                    "provisional" if prov else "missing"}
    gender_rows = []
    for c in cards.values():
        if c["status"] == "active" and c["map"] and gold[c["map"]].gender in ("M", "F"):
            ours = {"male": "M", "female": "F"}.get((c["gender"] or "").lower(), "?")
            gender_rows.append({"card": c["id"], "gold": c["map"], "ours": ours,
                                "ok": ours == gold[c["map"]].gender})

    # --- candidate coverage: the gold speaker is among the candidates attribute actually offered for the
    # quote: this chunk's and the previous chunk's characters plus the previous chunk's last speakers
    chunk_list = con.execute("SELECT id, doc_id, seq_from, seq_to, characters, scene FROM chunk "
                             "ORDER BY doc_id, seq_from").fetchall()
    chunk_cands = {}
    for i, r in enumerate(chunk_list):
        ids = list(json.loads(r["characters"] or "[]"))
        if i > 0:
            prev = chunk_list[i - 1]
            ids += json.loads(prev["characters"] or "[]")
            ids += json.loads(prev["scene"] or "{}").get("last_speakers", [])
        chunk_chars_of = {name_map.get(resolve(c)) for c in ids}
        chunk_cands[(r["doc_id"], r["seq_from"], r["seq_to"])] = chunk_chars_of
    sent_doc = {r["id"]: (r["doc_id"], r["seq"]) for r in con.execute("SELECT id, doc_id, seq FROM sentence")}

    def cands_for(sid):
        d, q = sent_doc.get(sid, (None, None))
        for (cd, a, b), cands in chunk_cands.items():
            if cd == d and a <= q <= b:
                return cands
        return set()
    cover_rows = [r for r in rows if r["sentence"]]
    for r in cover_rows:
        r["covered"] = r["gold"] in cands_for(r["sentence"])
    uncovered = [{"quote": r["quote"], "gold": r["gold"], "sentence": r["sentence"], "text": r["text"][:80]}
                 for r in cover_rows if not r["covered"]]

    # --- kin speech tags ("said her mother"), from attribute's chunk metrics (latest run per chunk)
    kin_sum = Counter()
    for r in con.execute("SELECT m.chunk_id, m.metrics FROM chunk_metrics m WHERE m.pass=2 AND m.run_id = "
                         "(SELECT MAX(run_id) FROM chunk_metrics x WHERE x.chunk_id=m.chunk_id AND x.pass=2)"):
        if r["chunk_id"] in {c["id"] for c in con.execute("SELECT id FROM chunk")} and \
                lo <= chunk_ch.get(r["chunk_id"], 0) <= hi:
            mm = json.loads(r["metrics"])
            for k in ("kin_tags", "kin_resolved", "kin_agree", "kin_disagree"):
                kin_sum[k] += mm.get(k, 0)

    # --- relations against the gold relation file (docs/DESIGN.md)
    rel = None
    if relations and Path(relations).exists():
        from . import relations as R
        rgold = R.load(relations, pdnc_dir, gold)
        ours = [(resolve(r["a"]), r["type"], resolve(r["b"]), f"{r['source'] or ''}:{r['strength'] or 'strong'}")
                for r in con.execute("SELECT a, type, b, source, strength FROM relation")]
        present = {c["map"] for c in cards.values() if c["map"]}
        g_gold = {g.name: {"M": "male", "F": "female"}.get(g.gender) for g in gold.values()}
        g_ours = {}
        for c in cards.values():
            if c["map"] and (c["gender"] or "").lower() in ("male", "female"):
                g_ours[c["map"]] = c["gender"].lower()
        all_map = dict(name_map)          # cards first seen after the range still name PDNC people
        for r in con.execute("SELECT c.id, a.alias FROM character c JOIN alias a ON a.character_id=c.id "
                             "WHERE c.status IN ('active', 'provisional')"):
            if r["id"] not in all_map and norm(r["alias"]) in index:
                all_map[r["id"]] = index[norm(r["alias"])]
        rel = R.score(rgold, ours, all_map, present, g_gold, g_ours, hi if chapters else None)

    # --- metrics
    def pct(a, b):
        return round(100 * a / b, 1) if b else None

    acc = lambda rs: pct(sum(r["correct"] for r in rs), len(rs))      # noqa: E731
    by_type = {t: {"n": len(rs), "acc": acc(rs)} for t in ("Explicit", "Anaphoric", "Implicit")
               for rs in [[r for r in rows if r["type"] == t]]}
    by_speaker = {g: {"n": len(rs), "acc": acc(rs)} for g in mains
                  for rs in [[r for r in rows if r["gold"] == g]] if rs}
    conf = {}
    for lab, a, b in CONF_BINS:
        rs = [r for r in rows if r["confidence"] is not None and a <= r["confidence"] < b]
        conf[lab] = {"n": len(rs), "acc": acc(rs)}
    wrong = [r for r in rows if not r["correct"]]
    confusion = Counter((r["gold"], r["pred"] or f"[{r['pred_card'] or 'none'}]") for r in wrong)

    chunks_rng = [r for r in con.execute("SELECT * FROM chunk") if lo <= chunk_ch[r["id"]] <= hi]
    rng_ids = {c["id"] for c in chunks_rng}
    warns = [dict(r) for r in con.execute(
        "SELECT code, severity, chunk_id, subject, count, details FROM warning "
        "WHERE status='open' AND severity IN ('warn', 'pause')") if r["chunk_id"] in rng_ids]
    low = [r for r in rows if r["confidence"] is not None and r["confidence"] < 0.6]
    unknown = [r for r in rows if r["pred_card"] == "unknown"]

    alias_prec = pct(n_name_alias - len(alias_err), n_name_alias)
    metrics = {
        "quotes": len(rq),
        "align_pct": pct(sum(q["aligned"] for q in rq), len(rq)),
        "align_pct_book": pct(sum(q["aligned"] for q in quotes), len(quotes)),
        "fuzzy_quotes": sum(1 for q in rq if q.get("fuzzy")),
        "missed_quotes": len(missed), "spurious_quotes": len(spurious),
        "placeholder_quotes": sum(placeholder.values()),
        "main_chars": len(mains),
        "main_recall_pct": pct(sum(v["state"] in ("ok", "split") for v in cover.values()), len(mains)),
        "false_merge": sum(1 for c in cards.values() if {"merge(name)", "merge(quotes)"} & set(c["flags"])),
        "false_split": sum(v["state"] == "split" for v in cover.values()),
        "stuck_provisional": sum(v["state"] == "provisional" for v in cover.values()),
        "alias_precision_pct": alias_prec,
        "gender_pct": pct(sum(r["ok"] for r in gender_rows), len(gender_rows)),
        "attr_pct": acc(rows) if has_attr else None,
        "explicit_pct": by_type["Explicit"]["acc"] if has_attr else None,
        "anaphoric_pct": by_type["Anaphoric"]["acc"] if has_attr else None,
        "implicit_pct": by_type["Implicit"]["acc"] if has_attr else None,
        "high_conf_pct": conf["≥0.8"]["acc"] if has_attr else None,
        "low_conf_catch_pct": pct(sum(1 for r in wrong if r["confidence"] is not None and r["confidence"] < 0.6),
                                  len(wrong)) if has_attr else None,
        "human_load": len(warns) + len(low) + len(unknown),
        "cards": len(cards),
        "candidate_coverage_pct": pct(sum(r["covered"] for r in cover_rows), len(cover_rows)),
        "kin_tags": (f"{kin_sum['kin_resolved']}/{kin_sum['kin_tags']}, agree {kin_sum['kin_agree']}, "
                     f"disagree {kin_sum['kin_disagree']}") if kin_sum["kin_tags"] else None,
        **(rel["metrics"] if rel else {}),
    }
    voices = three_class(con, rq, pred, resolve, gold, spurious, relations, lo, hi, overrides)
    metrics.update(voices["metrics"])
    runs = {
        "prompt_versions": sorted({r[0] for r in con.execute("SELECT DISTINCT prompt_version FROM llm_call")
                                   if r[0]}),
        "models": sorted({r[0] for r in con.execute("SELECT DISTINCT model FROM llm_call") if r[0]}),
        "chunks_in_range": len(chunks_rng),
        "pass1_done": sum(c["pass1"] == "done" for c in chunks_rng),
        "pass2_done": sum(c["pass2"] == "done" for c in chunks_rng),
    }
    return {
        "version": RESULT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "pdnc": str(pdnc_dir), "chapters": [lo, hi] if chapters else None,
        "overrides": {"file": str(overrides) if overrides else None,
                      "sha": _sha(overrides) if overrides and Path(overrides).exists() else None,
                      "applied": applied, "ambiguous_aliases": ambiguous},
        "code": _git_rev(), "config_sha": _sha_obj(cfg) if cfg else None, "run": runs,
        "has_attribution": has_attr,
        "metrics": metrics,
        "by_type": by_type, "by_speaker": by_speaker, "confidence": conf,
        "confusion": [{"gold": g, "pred": p, "n": n} for (g, p), n in confusion.most_common(10)],
        "coverage": cover,
        "cards": sorted(cards.values(), key=lambda c: c["id"]),
        "alias_errors": alias_err, "gender": gender_rows,
        "unaligned": [{"quote": q["id"], "chapter": q["chapter"], "found": q["found"], "subs": len(q["subs"]),
                       "text": q["text"].replace("\n", " ")[:100]} for q in rq if not q["aligned"]],
        "missed": [{"quote": q["id"], "sentence": q["sentence"], "text": q["text"].replace("\n", " ")[:100]}
                   for q in missed],
        "spurious": [{"our_quote": qid, "sentence": st.items[our_q[qid]][0]["id"],
                      "text": " ".join(p.text for s, pcs in st.items[our_q[qid]:our_q[qid] + 3]
                                       for p in pcs if p.quote_id == qid)[:100]} for qid in spurious],
        "warnings": warns, "rows": rows, "uncovered": uncovered,
        "relations": {k: v for k, v in rel.items() if k != "metrics"} if rel else None,
        "voices": {k: v for k, v in voices.items() if k not in ("metrics", "rows")},
        "voice_rows": voices["rows"],
    }


# ---------------------------------------------------------------- three voices (docs/DESIGN.md)

_SALUTE = re.compile(r"^(?:my\s+)?(?:dear(?:est)?|honou?red)\b|^(?:sir|madam|gentlemen)\b|^(?:yours|your\s+"
                     r"(?:affectionate|obedient|sincere|faithful|loving)|believe\s+me|i\s+am,?\s+(?:dear|my|sir|madam))", re.I)


def _letter_like(con, qid: str) -> bool:
    """A quoted passage that looks like a letter (docs/DESIGN.md; a guess from the text,
    not a gold label): starts with a salutation or a closing, is a signature (a short run of capitalized
    words alone in its paragraph), or is a long paragraph that is all quotation (a letter's body).
    Letters are read in the writer's voice by design (section 10) and PDNC does not count them as quotes."""
    rows = con.execute("SELECT u.tts_text, u.is_quote, u.quote_id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                       "WHERE s.doc_id=? AND s.para=? ORDER BY s.seq, u.seq",
                       (int(qid[:3]), int(qid[4:qid.index("q")]))).fetchall()
    text = " ".join(r["tts_text"] for r in rows if r["quote_id"] == qid)
    narration = any(not r["is_quote"] and re.search(r"[A-Za-z]", r["tts_text"]) for r in rows)
    words = text.split()
    if _SALUTE.search(text):
        return True
    if not narration and 1 <= len(words) <= 4 and all(w[:1].isupper() for w in words):
        return True                                # William Collins. / Fitzwilliam Darcy.
    return not narration and len(words) >= 25


def _gold_voice(q: dict, gold: dict, narrator: str | None) -> str | None:
    """male | female | narrator, or None (not scored: _unknowable, gender not M/F)."""
    g = q["speaker"]
    if g == "_unknowable" or not g:
        return None
    if g.startswith("_"):
        return "narrator"                   # _group: several people at once -> narrator (section 1)
    if narrator and g == narrator:
        return "narrator"                   # the first-person narrator's own words (section 1)
    c = gold.get(g)
    return {"M": "male", "F": "female"}.get(c.gender) if c else None


def three_class(con, rq, pred, resolve, gold, spurious, relations, lo, hi, overrides=None) -> dict:
    """Score the voice each PDNC quote gets. v3 books: utterance.speaker_id is the voice and voice_gender /
    confidence the class before the threshold. v2 books: the card's gender of the speaker, all voiced
    ("the v2 answer as is", 3.4), confidence as stored."""
    narrator = None
    for f in (relations, overrides):   # the gold relations file, or (books without one) the overrides file
        if not narrator and f and Path(f).exists():
            narrator = (yaml.safe_load(Path(f).read_text()) or {}).get("narrator")
    card_g = {r["id"]: (r["gender"] or "").lower() for r in con.execute("SELECT id, gender FROM character")}
    v3 = any(p.get("voice_gender") is not None or p["speaker"] in (V3_MALE, V3_FEMALE) for p in pred.values())

    def ours(qid):
        """(final voice, class, confidence, who)"""
        p = pred.get(qid)
        if not p or p["source"] not in ATTRIBUTED:
            return "narrator", None, None, None
        if v3:
            fin = {V3_MALE: "male", V3_FEMALE: "female"}.get(p["speaker"], "narrator")
            cls = p.get("voice_gender")
            if p["source"] == "manual":
                cls = fin
            return fin, cls, p["confidence"], p.get("who")
        g = card_g.get(resolve(p["speaker"]) or "", "") if p["speaker"] not in NON_SPEAKERS else ""
        g = g if g in ("male", "female") else None
        return (g or "narrator"), (g or ("narrator" if p["speaker"] == "narrator" else None)), p["confidence"], \
            p["speaker"]

    rows = []
    for q in rq:
        gv = _gold_voice(q, gold, narrator)
        if gv is None:
            continue
        top = q["overlap"].most_common(1)[0][0] if q["overlap"] else None
        fin, cls, conf, who = ours(top) if top else ("narrator", None, None, None)
        rows.append({"quote": q["id"], "chapter": q["chapter"], "type": q["type"], "gold": q["speaker"],
                     "gold_voice": gv, "voice": fin, "cls": cls, "confidence": conf, "who": who,
                     "words": len(q["text"].split()), "narrator_person": bool(narrator and q["speaker"] == narrator),
                     "text": q["text"].replace("\n", " ")[:100]})
    sp_rows = []
    for qid in spurious:
        fin, cls, conf, who = ours(qid)
        sp_rows.append({"our_quote": qid, "voice": fin, "cls": cls, "confidence": conf, "who": who,
                        "letter": _letter_like(con, qid)})

    def pct(a, b):
        return round(100 * a / b, 2) if b else None

    def score(voiced_of, weight=lambda r: 1):
        tot = sum(weight(r) for r in rows)
        mf = [r for r in rows if r["gold_voice"] in ("male", "female")]
        tot_mf = sum(weight(r) for r in mf)
        v = {r["quote"]: voiced_of(r) for r in rows}
        wrong = sum(weight(r) for r in mf if v[r["quote"]] in ("male", "female") and v[r["quote"]] != r["gold_voice"])
        voiced = sum(weight(r) for r in mf if v[r["quote"]] in ("male", "female"))
        right = sum(weight(r) for r in mf if v[r["quote"]] == r["gold_voice"])
        acc = sum(weight(r) for r in rows if v[r["quote"]] == r["gold_voice"])
        npers = [r for r in rows if r["narrator_person"]]
        nv = sum(1 for r in npers if v[r["quote"]] in ("male", "female"))
        other = sum(1 for r in rows if r["gold_voice"] == "narrator" and not r["narrator_person"]
                    and v[r["quote"]] in ("male", "female"))
        return {"mis_pct": pct(wrong, tot), "mis": wrong, "voiced_pct": pct(voiced, tot_mf),
                "right_pct": pct(right, tot_mf), "acc_pct": pct(acc, tot),
                "narrator_voiced_pct": pct(nv, len(npers)) if npers else None, "narrator_voiced": nv if npers else None,
                "group_voiced": other}

    final = score(lambda r: r["voice"])
    words = score(lambda r: r["voice"], weight=lambda r: r["words"])
    sp_voiced = sum(1 for r in sp_rows if r["voice"] in ("male", "female"))
    sp_letter = sum(1 for r in sp_rows if r["voice"] in ("male", "female") and r["letter"])
    curve = []
    for t in CURVE:
        f = (lambda r, t=t: r["cls"] if r["cls"] in ("male", "female") and (r["confidence"] or 0) >= t else "narrator")
        sc = score(f)
        sc["threshold"] = t
        sc["spurious_voiced"] = sum(1 for r in sp_rows if r["cls"] in ("male", "female") and (r["confidence"] or 0) >= t)
        curve.append(sc)
    conf = Counter()
    for r in rows:
        if r["gold_voice"] in ("male", "female") and r["voice"] in ("male", "female") and r["voice"] != r["gold_voice"]:
            conf[(r["gold"], r["who"] or "?")] += 1
    tasks = list(con.execute("SELECT task, COUNT(*) n, SUM(seconds) s, SUM(completion_tokens) ct, "
                             "SUM(prompt_tokens) pt FROM llm_call WHERE result='ok' GROUP BY task"))
    v3_tasks = [t for t in tasks if (t["task"] or "").startswith("v3_")]
    use = v3_tasks if v3 else [t for t in tasks if not (t["task"] or "").startswith("v3_")]
    metrics = {
        "v3_mis_pct": final["mis_pct"], "v3_mis": final["mis"], "v3_voiced_pct": final["voiced_pct"],
        "v3_right_pct": final["right_pct"], "v3_acc_pct": final["acc_pct"],
        "v3_spurious_voiced": sp_voiced, "v3_spurious_voiced_letter": sp_letter,
        "v3_spurious_voiced_other": sp_voiced - sp_letter, "v3_narrator_voiced_pct": final["narrator_voiced_pct"],
        "v3_words_mis_pct": words["mis_pct"], "v3_words_voiced_pct": words["voiced_pct"],
        "v3_words_right_pct": words["right_pct"],
        "v3_llm_seconds": round(sum(t["s"] or 0 for t in use)), "v3_llm_out_tokens": sum(t["ct"] or 0 for t in use),
    }
    return {"metrics": metrics, "mode": "v3" if v3 else "v2 (card gender)", "narrator": narrator,
            "final": final, "words": words, "curve": curve,
            "spurious": {"n": len(sp_rows), "voiced": sp_voiced, "voiced_letter": sp_letter,
                         "letters": sum(1 for r in sp_rows if r["letter"]),
                         "voiced_other": [r for r in sp_rows if r["voice"] in ("male", "female") and not r["letter"]]},
            "by_source": dict(Counter(pred[q]["source"] for q in pred)),
            "mis_confusion": [{"gold": g, "ours": w, "n": n} for (g, w), n in conf.most_common(15)],
            "llm": [dict(t) for t in use], "rows": rows}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _git_rev() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        rev = subprocess.run(["git", "-C", str(root), "describe", "--always", "--dirty", "--tags"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return rev or None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- reports

SUMMARY = [   # (key, label, higher is better)
    ("align_pct", "aligned %", True), ("missed_quotes", "missed quotes", False), ("spurious_quotes", "extra quotes", False),
    ("placeholder_quotes", "PDNC placeholder speakers (not scored)", False),
    ("main_recall_pct", "main characters found %", True), ("false_merge", "wrong merges", False), ("false_split", "wrong splits", False),
    ("stuck_provisional", "stuck provisional", False), ("alias_precision_pct", "alias precision %", True),
    ("gender_pct", "gender %", True), ("attr_pct", "attribution overall %", True), ("explicit_pct", "Explicit %", True),
    ("anaphoric_pct", "Anaphoric %", True), ("implicit_pct", "Implicit %", True),
    ("high_conf_pct", "accuracy at confidence >= 0.8 %", True), ("low_conf_catch_pct", "errors at low confidence %", True),
    ("human_load", "human load", False), ("cards", "cards", None),
    ("candidate_coverage_pct", "candidate coverage %", True),
    ("kin_tags", "kin tags (resolved / all; agreeing with the model, disagreeing)", None),
    ("rel_precision_pct", "relation precision %", True), ("rel_precision_strong_pct", "strong relation precision %", True), ("rel_recall_pct", "relation recall %", True),
    ("rel_core_recall_pct", "core relation recall %", True), ("kin_lookup_pass", "kin lookups passed", None),
    ("v3_mis_pct", "three voices: wrong gender %", False), ("v3_mis", "three voices: wrong gender (quotes)", False),
    ("v3_voiced_pct", "three voices: voiced %", True), ("v3_right_pct", "three voices: voiced right %", True),
    ("v3_acc_pct", "three voices: all three classes right %", True), ("v3_spurious_voiced", "three voices: extra quotes voiced (pieces)", False),
    ("v3_spurious_voiced_letter", "three voices: of those, letter-like (pieces)", None),
    ("v3_spurious_voiced_other", "three voices: of those, other (pieces)", False),
    ("v3_narrator_voiced_pct", "three voices: narrator's own lines voiced %", False),
    ("v3_words_mis_pct", "three voices (by words): wrong %", False), ("v3_words_voiced_pct", "three voices (by words): voiced %", True),
    ("v3_llm_seconds", "model time (s, summed over calls)", None), ("v3_llm_out_tokens", "model output tokens", None),
]


def _f(v):
    return "—" if v is None else str(v)


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_f(x).replace("|", "\\|") for x in r) + " |" for r in rows]
    return "\n".join(out)


def render_md(res: dict, tag: str, book_id: str) -> str:
    m = res["metrics"]
    L = [f"# PDNC evaluation: {book_id} `{tag}`", "",
         f"- time {res['created']}; chapters {res['chapters']}; eval code {res['code']}; config {res['config_sha']}",
         f"- run: prompt {', '.join(res['run']['prompt_versions']) or '—'}; model {', '.join(res['run']['models']) or '—'}",
         f"- chunks in range {res['run']['chunks_in_range']}: characters done {res['run']['pass1_done']}, "
         f"attribute done {res['run']['pass2_done']}",
         "- gold: PDNC (Vishnubhotla et al. 2022, CC-BY 4.0)", ""]
    if not res["has_attribution"]:
        L += ["> No attribution results in range yet: alignment and roster metrics only.", ""]
    L += ["## Summary", "", _table(["metric", "value"], [(lab, m.get(k)) for k, lab, _ in SUMMARY if k in m]), ""]
    if res["has_attribution"]:
        L += ["## Attribution", "", _table(["type", "quotes", "accuracy %"],
                                       [(t, v["n"], v["acc"]) for t, v in res["by_type"].items()]), "",
              _table(["speaker (main)", "quotes", "accuracy %"],
                     [(g, v["n"], v["acc"]) for g, v in sorted(res["by_speaker"].items(), key=lambda x: -x[1]["n"])]),
              "", _table(["confidence", "quotes", "accuracy %"], [(k, v["n"], v["acc"]) for k, v in res["confidence"].items()]),
              "", "Most confused (gold -> ours):", "",
              _table(["gold", "ours", "count"], [(c["gold"], c["pred"], c["n"]) for c in res["confusion"]]), ""]
    vo = res.get("voices")
    if vo and res["has_attribution"]:
        L += ["## Three voices (male, female, narrator; docs/DESIGN.md)", "",
              f"mode: {vo['mode']}; first-person narrator: {vo['narrator'] or '—'} (his own lines count as narrator); "
              f"extra quotes: {vo['spurious']['n']} pieces ({vo['spurious'].get('letters', '—')} letter-like), voiced "
              f"{vo['spurious']['voiced']} pieces: {vo['spurious'].get('voiced_letter', '—')} letter-like (read in the writer's voice by design), "
              f"other {vo['spurious']['voiced'] - (vo['spurious'].get('voiced_letter') or 0)}", "",
              "Threshold curve (voiced only at confidence >= threshold; the actual result is in the summary):", "",
              _table(["threshold", "wrong %", "wrong", "voiced %", "right %", "all classes right %", "narrator voiced %", "extra voiced"],
                     [(c["threshold"], c["mis_pct"], c["mis"], c["voiced_pct"], c["right_pct"], c["acc_pct"],
                       c["narrator_voiced_pct"], c["spurious_voiced"]) for c in vo["curve"]]), ""]
        if vo["mis_confusion"]:
            L += ["Wrong gender (gold -> our speaker):", "",
                  _table(["gold", "ours", "quotes"], [(c["gold"], c["ours"], c["n"]) for c in vo["mis_confusion"]]), ""]
        other = vo["spurious"].get("voiced_other") or []
        if other:
            L += ["Extra quotes that were voiced and are not letter-like (first 40):", "",
                  _table(["our_quote", "voice", "confidence", "speaker"],
                         [(r["our_quote"], r["voice"], r["confidence"], r["who"]) for r in other[:40]]), ""]
        L += ["Sources: " + ", ".join(f"{k} {v}" for k, v in sorted(vo["by_source"].items(), key=lambda x: -x[1])), ""]
    L += ["## Roster", "", _table(["PDNC character", "category", "status", "confirmed cards", "provisional cards"],
                                 [(g, v["category"], v["state"], ", ".join(v["active"]), ", ".join(v["provisional"]))
                                  for g, v in res["coverage"].items()]), "",
          "Our cards (matched by name / by quotes):", "",
          _table(["card", "status", "by name", "by quotes", "flags", "name hits"],
                 [(c["id"], c["status"], c["map"], c["quote_map"], " ".join(c["flags"]),
                   "; ".join(f"{g}: {', '.join(v)}" for g, v in c["name_hits"].items()))
                  for c in res["cards"] if c["map"] or c["quote_map"] or c["flags"]]), ""]
    unmapped = [c["id"] for c in res["cards"] if not c["map"] and not c["quote_map"]]
    if unmapped:
        L += [f"Cards with no PDNC character ({len(unmapped)}): {', '.join(unmapped)}", ""]
    if res["alias_errors"]:
        L += ["Aliases attached to the wrong person:", "", _table(["card", "alias", "really", "card maps to"],
                                    [(e["card"], e["alias"], e["belongs_to"], e["card_map"])
                                     for e in res["alias_errors"]]), ""]
    bad_g = [g for g in res["gender"] if not g["ok"]]
    if bad_g:
        L += ["Wrong gender: " + ", ".join(f"{g['card']} ({g['ours']}, should be {g['gold']})" for g in bad_g), ""]
    L += ["## Alignment and quote detection", "",
          f"unaligned {len(res['unaligned'])}, missed {len(res['missed'])}, extra {len(res['spurious'])}", ""]
    for title, key, cols in (("Unaligned", "unaligned", ("quote", "chapter", "found", "subs", "text")),
                             ("Missed (in PDNC, narration for us)", "missed", ("quote", "sentence", "text")),
                             ("Extra (ours, not in PDNC)", "spurious", ("our_quote", "sentence", "text"))):
        if res[key]:
            L += [f"### {title}", "", _table(cols, [[x[c] for c in cols] for x in res[key][:60]]), ""]
    if res["warnings"]:
        L += ["## Open warnings (in range)", "", _table(["code", "severity", "chunk", "subject", "count"],
                                           [(w["code"], w["severity"], w["chunk_id"], w["subject"], w["count"])
                                            for w in res["warnings"]]), ""]
    if res.get("uncovered"):
        L += ["## Quotes whose gold speaker was not among the candidates", "",
              _table(["quote", "gold", "sentence", "text"],
                     [(u["quote"], u["gold"], u["sentence"], u["text"]) for u in res["uncovered"]]), ""]
    rel = res.get("relations")
    if rel:
        L += ["## Relations (gold relation file)", "",
              f"written {m.get('rel_written')}; precision {m.get('rel_precision_pct')}%; "
              f"recall {rel['counts']['found']}/{rel['counts']['want']} (core {m.get('rel_core_recall_pct')}%); "
              f"kin lookups {m.get('kin_lookup_pass')}", ""]
        if rel["rows"] or rel["unmapped"]:
            L += [_table(["", "a", "type", "b", "source"],
                         [("✓" if r["correct"] else "✗", r["a"], r["type"], r["b"], r["source"]) for r in rel["rows"]]
                         + [("?", u["a"], u["type"], u["b"], u["source"] + " (no PDNC character)") for u in rel["unmapped"]]),
                  ""]
        if rel["missed"]:
            L += ["Missed (core / stated):", "",
                  _table(["a", "type", "b", "tier", "chapter"],
                         [(x["a"], x["type"], x["b"], x["tier"], x["chapter"]) for x in rel["missed"]]), ""]
        if rel["no_card"]:
            L += [f"People with no card, not counted in recall: {', '.join(rel['no_card'])}", ""]
        bad = [q for q in rel["lookups"] if not q["pass"]]
        if bad:
            L += ["Kin lookups that failed:", "",
                  _table(["anchor", "kin", "expected", "got"],
                         [(q["anchor"], q["kin"], ", ".join(q.get("expect") or []),
                           q.get("why") or ", ".join(q.get("got") or []) or "(empty)") for q in bad]), ""]
    ov = res["overrides"]
    L += ["## overrides", "", f"file {ov['file']} ({ov['sha']})", ""]
    L += [f"- {json.dumps({k: v for k, v in e.items() if k != 'reason'}, ensure_ascii=False)}: {e['reason']}"
          for e in ov["applied"]]
    if ov["ambiguous_aliases"]:
        L += ["", "Aliases shared by several people, not used for matching: " + "; ".join(f"{a} ({', '.join(v)})"
                                                  for a, v in ov["ambiguous_aliases"].items())]
    return "\n".join(L) + "\n"


def write_reports(res: dict, out_dir: Path, tag: str, book_id: str, note: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {**res, "tag": tag, "book": book_id, "note": note}
    js = out_dir / f"{tag}.json"
    if js.exists():          # keep the previous result of the same tag
        old = json.loads(js.read_text())
        hist = out_dir / "history"
        hist.mkdir(exist_ok=True)
        stamp = old.get("created", "old").replace(":", "")
        for ext in (".json", ".md", ".quotes.csv"):
            p = out_dir / f"{tag}{ext}"
            if p.exists():
                p.rename(hist / f"{tag}.{stamp}{ext}")
    js.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (out_dir / f"{tag}.md").write_text(render_md(res, tag, book_id))
    with open(out_dir / f"{tag}.quotes.csv", "w", newline="") as f:
        cols = ["quote", "chapter", "type", "gold", "pred", "pred_card", "correct", "confidence", "aligned",
                "our_quote", "sentence", "text"]
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["rows"])
    return js


def compare(results: list[dict]) -> str:
    """Markdown: metrics side by side (Δ against the first), then per-quote flips first → each later one."""
    names = [f"{r.get('book', '?')}:{r.get('tag', '?')}" for r in results]
    base = results[0]["metrics"]
    rows = []
    for k, lab, better in SUMMARY:
        line = [lab, _f(base.get(k))]
        for r in results[1:]:
            v = r["metrics"].get(k)
            num = isinstance(v, (int, float)) and isinstance(base.get(k), (int, float))
            d = round(v - base[k], 1) if num else None
            mark = ""
            if d and better is not None:
                good = (d > 0) == better
                big = abs(d) >= 2 if k.endswith("_pct") else True
                mark = (" ✅" if good else " ❌") if big else " ·"
            line.append(f"{_f(v)}" + (f" ({d:+g}{mark})" if d else ""))
        rows.append(line)
    L = ["# Version comparison", "", "Δ against the first column; a `_pct` difference under 2 points is marked · (noise).", "",
         _table(["metric", *names], rows), ""]
    L += [_table(["", *names], [
        ["chapters", *[_f(r.get("chapters")) for r in results]],
        ["prompt", *[", ".join(r["run"]["prompt_versions"]) for r in results]],
        ["model", *[", ".join(r["run"]["models"]) for r in results]],
        ["eval code", *[_f(r.get("code")) for r in results]],
        ["overrides", *[_f(r["overrides"]["sha"]) for r in results]],
        ["note", *[_f(r.get("note")) for r in results]],
    ]), ""]
    if len({json.dumps(r.get("chapters")) for r in results}) > 1 or len({r["overrides"]["sha"] for r in results}) > 1:
        L += ["> Warning: the chapter range or the overrides differ, so the numbers are not directly comparable.", ""]
    a = {x["quote"]: x for x in results[0]["rows"]}
    for name, r in zip(names[1:], results[1:]):
        b = {x["quote"]: x for x in r["rows"]}
        fixed = [b[q] for q in b if q in a and b[q]["correct"] and not a[q]["correct"]]
        broke = [(a[q], b[q]) for q in b if q in a and a[q]["correct"] and not b[q]["correct"]]
        changed = [(a[q], b[q]) for q in b if q in a and not a[q]["correct"] and not b[q]["correct"]
                   and a[q]["pred"] != b[q]["pred"]]
        L += [f"## {names[0]} → {name}: fixed {len(fixed)}, broke {len(broke)}, wrong differently {len(changed)}", ""]
        if broke:
            L += ["Broke:", "", _table(["quote", "type", "gold", "before", "after", "text"],
                                     [(y["quote"], y["type"], y["gold"], x["pred"] or x["pred_card"],
                                       y["pred"] or y["pred_card"], y["text"][:60]) for x, y in broke]), ""]
        if fixed:
            L += ["Fixed:", "", _table(["quote", "type", "gold", "before", "text"],
                                     [(y["quote"], y["type"], y["gold"],
                                       a[y["quote"]]["pred"] or a[y["quote"]]["pred_card"], y["text"][:60])
                                      for y in fixed]), ""]
        if changed:
            L += ["Wrong differently:", "", _table(["quote", "gold", "before", "after"],
                                       [(y["quote"], y["gold"], x["pred"] or x["pred_card"],
                                         y["pred"] or y["pred_card"]) for x, y in changed]), ""]
    return "\n".join(L) + "\n"


def list_results(books_root: Path) -> str:
    rows = []
    for p in sorted(books_root.glob("*/reports/eval/*.json")):
        r = json.loads(p.read_text())
        m = r["metrics"]
        rows.append((f"{r.get('book')}:{r.get('tag')}", r["created"], _f(r.get("chapters")), m["align_pct"],
                     m["main_recall_pct"], m["false_merge"], m["false_split"], m["attr_pct"], m["explicit_pct"],
                     m["human_load"], r.get("note") or ""))
    if not rows:
        return "No evaluation results (books/*/reports/eval/*.json)\n"
    return _table(["ref", "time", "chapters", "aligned%", "found%", "merges", "splits", "attribution%", "Explicit%", "human", "note"],
                  rows) + "\n"
