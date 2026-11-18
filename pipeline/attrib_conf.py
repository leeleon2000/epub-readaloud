"""Computed confidence for quote attribution (docs/DESIGN.md: the confidence is not the model's own number).

Signals, in order of strength:
  speech tag naming someone ("said Elizabeth", "Mr. Darcy replied"; also the tag between the two halves
  of “…,” said Elizabeth, “…”, which belongs to both)
      agrees with the model -> 0.95, disagrees -> 0.3
  pronoun tag ("she said")      gender agrees -> 0.8, disagrees -> 0.3, unknown -> 0.6
  no tag: same speaker continuing in the same paragraph -> 0.75,
          A-B-A alternation -> 0.7, otherwise -> 0.5
  narrator (non-speech quote): 0.8 without a tag, 0.3 if a speech tag is present
  unknown -> 0.0
The model's self-reported confidence is not used (it was always 1.0).
"""
from __future__ import annotations

import re

from .memory.db import CharacterDB
from .memory.kin import KIN_WORDS
from .memory.retrieve import candidates

VERBS = (r"said|says|cried|replied|answered|asked|exclaimed|continued|added|returned|observed|whispered|"
         r"shouted|called|thought|rejoined|began|resumed|interrupted|repeated|remarked|inquired|enquired|"
         r"retorted|answered|murmured|muttered|declared|insisted|protested")
TITLE = r"(?:Mr|Mrs|Miss|Ms|Dr|Lady|Sir|Lord|Colonel|Captain|Aunt|Uncle)\.?"
NAME = rf"(?:{TITLE}\s+)?[A-Z][a-z’']+(?:\s+[A-Z][a-z’']+)?"
AFTER = re.compile(rf"^[\s,;:—–-]*(?:(?:{VERBS})\s+(?P<n1>{NAME}|he|she)\b|(?P<n2>{NAME}|he|she|I)\s+(?:{VERBS})\b)")
BEFORE = re.compile(rf"(?P<n>{NAME}|he|she)\s+(?:{VERBS})\b[^.!?]*[,:—–-]?\s*$")
PRONOUN_GENDER = {"he": "male", "she": "female"}
_KIN = "|".join(sorted((re.escape(w) for w in KIN_WORDS if " " not in w), key=len, reverse=True))
KIN_AFTER = re.compile(rf"^[\s,;:—–-]*(?:(?:{VERBS})\s+(?P<p1>his|her|their)\s+(?:\w+\s+)?(?P<k1>{_KIN})\b"
                       rf"|(?P<p2>his|her|their)\s+(?:\w+\s+)?(?P<k2>{_KIN})\s+(?:{VERBS})\b)", re.I)
KIN_BEFORE = re.compile(rf"\b(?P<p>his|her|their)\s+(?:\w+\s+)?(?P<k>{_KIN})\s+(?:{VERBS})\b[^.!?]*[,:—–-]?\s*$", re.I)
POSS_GENDER = {"his": "male", "her": "female", "their": None}


def _kin_tag(after: str, before: str, between: str = ""):
    """("her", "mother") for “…,” said her mother / Her mother said, “…”; else None."""
    for text in (after, between):
        m = KIN_AFTER.search(text or "")
        if m:
            return (m.group("p1") or m.group("p2")).lower(), (m.group("k1") or m.group("k2")).lower()
    m = KIN_BEFORE.search(before or "")
    return (m.group("p").lower(), m.group("k").lower()) if m else None


def _tag(after: str, before: str, between: str = "") -> str | None:
    for text in (after, between):       # between: “…,” said Elizabeth, “<this quote>”
        m = AFTER.search(text or "")
        if m:
            return m.group("n1") or m.group("n2")
    m = BEFORE.search(before or "")
    return m.group("n") if m else None


def score(db: CharacterDB, doc_id: int, quotes: list[dict], kin: dict | None = None,
          stats: dict | None = None) -> list[float]:
    """quotes: [{speaker, after, before, para}] in reading order -> confidence per quote.
    kin: {"graph": memory.kin.Graph, "scene": [card ids]} resolves tags like "said her mother" to the set
    of people the relation names for anyone on the scene (docs/DESIGN.md)."""
    out = []
    st = stats if stats is not None else {}
    for i, q in enumerate(quotes):
        spk = q["speaker"]
        kt = _kin_tag(q.get("after", ""), q.get("before", ""), q.get("between", ""))
        if kt and spk not in ("unknown", "narrator"):
            st["kin_tags"] = st.get("kin_tags", 0) + 1
            if kin:
                g = POSS_GENDER.get(kt[0])
                anchors = [x for x in kin["scene"] if not g or (db.card(x) and db.card(x).gender in (None, "", "unknown", g))]
                strong, every = set(), set()
                for a in anchors:
                    strong |= kin["graph"].lookup(a, kt[1], weak=False)
                    every |= kin["graph"].lookup(a, kt[1])
                if every:
                    st["kin_resolved"] = st.get("kin_resolved", 0) + 1
                    s_ = db.resolve(spk)
                    hit = s_ in every
                    st["kin_agree" if hit else "kin_disagree"] = st.get("kin_agree" if hit else "kin_disagree", 0) + 1
                    if s_ in strong:          # strong relations: 0.95 when they name one person (5.6 point 3)
                        out.append(0.95 if len(every) == 1 else 0.8)
                    elif hit:                 # only through a weak relation: at most 0.8
                        out.append(0.8 if len(every) == 1 else 0.7)
                    elif strong:
                        out.append(0.3)       # a strong relation names someone else
                    else:
                        out.append(0.5)       # only weak relations disagree: no verdict
                    continue
        tag = _tag(q.get("after", ""), q.get("before", ""), q.get("between", ""))
        if spk == "unknown":
            out.append(0.0)
            continue
        if spk == "narrator":
            out.append(0.3 if tag and tag.lower() not in ("i",) else 0.8)
            continue
        if tag:
            low = tag.lower()
            if low in PRONOUN_GENDER:
                card = db.card(db.resolve(spk))
                g = card.gender if card else ""
                out.append(0.6 if g in ("", "unknown") else 0.8 if g == PRONOUN_GENDER[low] else 0.3)
                continue
            if low == "i":
                out.append(0.6)
                continue
            hits = candidates(db, tag, doc_id, [], 3)
            shared = len(hits) > 1 and hits[1][1] >= hits[0][1]   # "said Miss Bennet": a shared alias decides nothing
            if hits and hits[0][1] >= 0.9 and not shared:
                out.append(0.95 if db.resolve(hits[0][0]) == db.resolve(spk) else 0.3)
                continue
        prev1 = quotes[i - 1] if i >= 1 else None
        prev2 = quotes[i - 2] if i >= 2 else None
        if prev1 and prev1["speaker"] == spk and prev1["para"] == q["para"]:
            out.append(0.75)
        elif prev2 and prev1 and prev2["speaker"] == spk and prev1["speaker"] != spk:
            out.append(0.7)
        else:
            out.append(0.5)
    return out
