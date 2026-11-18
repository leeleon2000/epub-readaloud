"""Resolve references through the relation graph (docs/DESIGN.md). Rules only.

The pronoun is not resolved: "her brother" is computed as the set of brothers of every person on the scene
whose gender fits "her". "Mrs. Bennet's sister" uses the named anchor only.

    resolve(graph, cards, "her brother", scene=["miss_bingley", "mrs_hurst", "elizabeth"], gender=...)
        -> ({"mr_bingley"}, ["miss_bingley", "mrs_hurst", "elizabeth"], "brother")
"""
from __future__ import annotations

import re

from .kin import KIN_WORDS, Graph, kin_word

POSS_GENDER = {"his": "male", "her": "female", "their": None, "its": None}
_POSS = re.compile(r"^\s*(?:the\s+)?(?P<poss>his|her|their|my|your|our)\s+(?:[a-z]+\s+){0,2}?(?P<kin>"
                   + "|".join(sorted((re.escape(w) for w in KIN_WORDS), key=len, reverse=True))
                   + r")s?\b", re.I)
_NAMED = re.compile(r"^\s*(?P<name>.+?)['’]s\s+(?:[a-z]+\s+){0,2}?(?P<kin>"
                    + "|".join(sorted((re.escape(w) for w in KIN_WORDS), key=len, reverse=True)) + r")s?\b", re.I)


def parse(text: str):
    """-> (possessive or None, named anchor text or None, kin word) or None."""
    m = _POSS.match(text)
    if m and kin_word(m.group(0)):
        return m.group("poss").lower(), None, m.group("kin").lower()
    m = _NAMED.match(text)
    if m and kin_word(m.group(0)):
        return None, m.group("name"), m.group("kin").lower()
    return None


def resolve(graph: Graph, text: str, scene: list[str], gender: dict, anchor_of=None):
    """-> (targets, anchors, kin) or None when `text` is not a kin reference we can compute.
    scene: card ids present; anchor_of: name text -> card id (for "X's sister").
    "my"/"your"/"our" need the speaker or addressee: not resolved here."""
    p = parse(text)
    if not p:
        return None
    poss, named, kin = p
    if named is not None:
        a = anchor_of(named) if anchor_of else None
        anchors = [a] if a else []
    elif poss in POSS_GENDER:
        g = POSS_GENDER[poss]
        anchors = [x for x in scene if not g or gender.get(x) in (None, "", "unknown", g)]
    else:
        return None
    targets = set()
    for a in anchors:
        targets |= graph.lookup(a, kin)      # never the anchor itself
    return targets, anchors, kin
