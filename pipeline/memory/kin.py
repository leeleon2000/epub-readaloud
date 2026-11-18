"""Relation graph queries (docs/DESIGN.md). Rules only.

Stored relation types (a, type, b):
  spouse_of, sibling_of, in_law_of, cousin_of, friend_of        symmetric
  parent_of (a is parent of b), aunt_uncle_of (a is aunt/uncle of b), employs (a employs b)
Derived on load (5.1 step 6): siblings are transitive; a person's parents are their siblings' parents;
a parent's spouse is also a parent.

    g = Graph(relations, gender={"mrs_bennet": "female", ...})
    g.lookup("elizabeth", "uncle")       -> {"mr_philips", "mr_gardiner"}
    kin_word("brother-in-law")           -> ("in_law", "male")
"""
from __future__ import annotations

import re
from collections import defaultdict

SYMMETRIC = {"spouse_of", "sibling_of", "in_law_of", "cousin_of", "friend_of"}
DIRECTED = {"parent_of", "aunt_uncle_of", "employs"}
TYPES = SYMMETRIC | DIRECTED

# kin word -> (relation from the anchor's point of view, gender of the person meant or None)
KIN_WORDS = {
    "wife": ("spouse", "female"), "husband": ("spouse", "male"), "lady": ("spouse", "female"),
    "spouse": ("spouse", None),
    "mother": ("parent", "female"), "mamma": ("parent", "female"), "mama": ("parent", "female"),
    "father": ("parent", "male"), "papa": ("parent", "male"), "parent": ("parent", None),
    "daughter": ("child", "female"), "son": ("child", "male"), "child": ("child", None),
    "sister": ("sibling", "female"), "brother": ("sibling", "male"), "sibling": ("sibling", None),
    "sister-in-law": ("in_law", "female"), "brother-in-law": ("in_law", "male"),
    "aunt": ("aunt_uncle", "female"), "uncle": ("aunt_uncle", "male"),
    "niece": ("niece_nephew", "female"), "nephew": ("niece_nephew", "male"),
    "cousin": ("cousin", None), "friend": ("friend", None),
    "married to": ("spouse", None),
}
# relation (anchor's view) -> stored type and whether the anchor is `a` (True) or `b` (False) when directed
STORE = {"spouse": ("spouse_of", None), "sibling": ("sibling_of", None), "in_law": ("in_law_of", None),
         "cousin": ("cousin_of", None), "friend": ("friend_of", None),
         "parent": ("parent_of", False), "child": ("parent_of", True),
         "aunt_uncle": ("aunt_uncle_of", False), "niece_nephew": ("aunt_uncle_of", True)}

_KIN_RE = re.compile(r"\b(" + "|".join(sorted((re.escape(w) for w in KIN_WORDS), key=len, reverse=True))
                     + r")(?:s|es)?\b", re.I)


def kin_word(text: str) -> tuple[str, str | None] | None:
    """The kin relation named in a phrase ("her eldest sister", "His brother-in-law") or None.
    In-law words are matched before plain ones."""
    m = _KIN_RE.search(text.replace("’", "'"))
    if not m:
        return None
    w = m.group(1).lower()
    if w == "lady" and not re.search(r"\b(his|your|my)\s+lady\b", text, re.I):
        return None                         # "Lady Lucas" is a title, "his lady" is a wife
    return KIN_WORDS[w]


def kin_words(text: str) -> list[tuple[int, int, str, tuple[str, str | None]]]:
    """Every kin word in text: (start, end, word, (relation, gender))."""
    out = []
    t = text.replace("’", "'")
    for m in _KIN_RE.finditer(t):
        w = m.group(1).lower()
        if w == "lady" and not re.search(r"\b(his|your|my)\s+lady\s*$", t[:m.end(1)], re.I):
            continue
        out.append((m.start(), m.end(), w, KIN_WORDS[w]))
    return out


def to_triple(x: str, rel: str, y: str) -> tuple[str, str, str] | None:
    """x is the <rel> of y  ->  stored (a, type, b)."""
    t = {"spouse": "spouse_of", "sibling": "sibling_of", "in_law": "in_law_of", "cousin": "cousin_of",
         "friend": "friend_of"}.get(rel)
    if t:
        return (min(x, y), t, max(x, y))
    if rel == "parent":
        return (x, "parent_of", y)
    if rel == "child":
        return (y, "parent_of", x)
    if rel == "aunt_uncle":
        return (x, "aunt_uncle_of", y)
    if rel == "niece_nephew":
        return (y, "aunt_uncle_of", x)
    return None


class Graph:
    def __init__(self, relations, gender: dict[str, str | None] | None = None, weak=()):
        """relations: iterable of (a, type, b), strong. weak: relations used only as they are, never to
        derive others or to compose in-laws / aunts / cousins (docs/DESIGN.md)."""
        self.gender = gender or {}
        self.adj: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.adjw: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.stored: set[tuple[str, str, str]] = set()
        for a, t, b in relations:
            if a != b and t in TYPES:
                self._add(a, t, b)
                self.stored.add((a, t, b) if t in DIRECTED else (min(a, b), t, max(a, b)))
        self.derived = self._derive()
        for a, t, b in weak:
            if a != b and t in TYPES:
                self._add(a, t, b, self.adjw)

    def _add(self, a, t, b, adj=None):
        adj = self.adj if adj is None else adj
        if t in SYMMETRIC:
            adj[a][t].add(b)
            adj[b][t].add(a)
        else:
            adj[a][t].add(b)
            adj[b]["~" + t].add(a)

    def _derive(self) -> set[tuple[str, str, str]]:
        """Sibling closure; parents shared by siblings and by spouses. Returns the new triples."""
        before = self.triples()
        changed = True
        while changed:
            changed = False
            for x in list(self.adj):
                sib = self.adj[x]["sibling_of"]
                for s in list(sib):
                    for t in list(self.adj[s]["sibling_of"]):
                        if t != x and t not in sib:
                            self._add(x, "sibling_of", t)
                            changed = True
                parents = set(self.adj[x]["~parent_of"])
                for p in list(parents):
                    for sp in list(self.adj[p]["spouse_of"]):
                        if sp not in parents and sp != x:
                            self._add(sp, "parent_of", x)
                            changed = True
                for s in list(sib):
                    for p in list(self.adj[x]["~parent_of"]):
                        if p not in self.adj[s]["~parent_of"] and p != s:
                            self._add(p, "parent_of", s)
                            changed = True
        return self.triples() - before

    def triples(self) -> set[tuple[str, str, str]]:
        out = set()
        for a, rels in self.adj.items():
            for t, bs in rels.items():
                if t.startswith("~"):
                    continue
                for b in bs:
                    out.add((a, t, b) if t in DIRECTED else (min(a, b), t, max(a, b)))
        return out

    def _r(self, x, t) -> set:
        return set(self.adj[x][t]) if x in self.adj else set()

    def _w(self, x, t) -> set:
        return set(self.adjw[x][t]) if x in self.adjw else set()

    def relation(self, x: str, rel: str, weak: bool = True) -> set[str]:
        """Everyone related to x by `rel` (anchor's view). In-laws, aunts/uncles, nieces/nephews and cousins
        are composed from strong relations only; weak relations count only as themselves."""
        out = self._strong(x, rel)
        if weak:
            direct = {"spouse": "spouse_of", "parent": "~parent_of", "child": "parent_of", "sibling": "sibling_of",
                      "in_law": "in_law_of", "aunt_uncle": "~aunt_uncle_of", "niece_nephew": "aunt_uncle_of",
                      "cousin": "cousin_of", "friend": "friend_of", "employer": "~employs",
                      "employee": "employs"}.get(rel)
            if direct:
                out |= self._w(x, direct) - {x}
        return out

    def _strong(self, x: str, rel: str) -> set[str]:
        sp, sib = self._r(x, "spouse_of"), self._r(x, "sibling_of")
        parents, children = self._r(x, "~parent_of"), self._r(x, "parent_of")
        if rel == "spouse":
            return sp
        if rel == "parent":
            return parents
        if rel == "child":
            return children
        if rel == "sibling":
            return sib
        if rel == "in_law":
            out = self._r(x, "in_law_of")
            for s in sp:
                out |= self._r(s, "sibling_of")
            for s in sib:
                out |= self._r(s, "spouse_of")
            return out - {x}
        if rel == "aunt_uncle":
            out = self._r(x, "~aunt_uncle_of")
            for p in parents:
                for s in self._r(p, "sibling_of"):
                    out |= {s} | self._r(s, "spouse_of")
            return out - parents - {x}
        if rel == "niece_nephew":
            out = self._r(x, "aunt_uncle_of")
            for s in sib | {y for p in sp for y in self._r(p, "sibling_of")}:
                out |= self._r(s, "parent_of")
            return out - {x}
        if rel == "cousin":
            direct = self._r(x, "cousin_of")
            out = set(direct)
            for c in direct:
                out |= self._r(c, "parent_of")            # Mr. Collins and Mr. Bennet's daughters
            for p in parents:
                out |= self._r(p, "cousin_of")            # the daughters and Mr. Collins
                for s in self._r(p, "sibling_of"):
                    out |= self._r(s, "parent_of")        # first cousins
            return out - {x} - sib
        if rel == "friend":
            return self._r(x, "friend_of")
        if rel == "employer":
            return self._r(x, "~employs")
        if rel == "employee":
            return self._r(x, "employs")
        return set()

    def lookup(self, x: str, word: str, weak: bool = True) -> set[str]:
        """People that `word` ("sister", "brother-in-law", "his lady") names when said of x."""
        k = kin_word(word) if word not in KIN_WORDS else KIN_WORDS[word]
        if not k:
            return set()
        rel, g = k
        out = self.relation(x, rel, weak)
        return {y for y in out if not g or self.gender.get(y) in (None, "", "unknown", g)}

    def has(self, a: str, t: str, b: str, weak: bool = True) -> bool:
        """Does the graph (with derivations) hold a -t-> b?"""
        rel = {"spouse_of": "spouse", "sibling_of": "sibling", "in_law_of": "in_law", "cousin_of": "cousin",
               "friend_of": "friend", "parent_of": "child", "aunt_uncle_of": "niece_nephew",
               "employs": "employee"}.get(t)
        return bool(rel) and b in self.relation(a, rel, weak)

    def people(self) -> set[str]:
        return set(self.adj) | set(self.adjw)

    def problems(self) -> list[str]:
        """Whole-graph checks after derivation (5.6 point 5): someone both parent and sibling (or child, or
        spouse) of the same person; more than two parents or two of one gender; a generation cycle; two
        spouses."""
        out = []
        for x in sorted(self.people()):
            par, ch = self.relation(x, "parent"), self.relation(x, "child")
            sib, sp = self.relation(x, "sibling"), self.relation(x, "spouse")
            for name, a, b in (("parent and sibling", par, sib), ("parent and child", par, ch),
                               ("spouse and sibling", sp, sib), ("spouse and parent", sp, par),
                               ("spouse and child", sp, ch), ("child and sibling", ch, sib)):
                both = a & b
                if both:
                    out.append(f"{sorted(both)[0]} is both {name} of {x}")
            if len(par) > 2:
                out.append(f"{x} has {len(par)} parents")
            for g in ("male", "female"):
                if sum(1 for p in par if self.gender.get(p) == g) > 1:
                    out.append(f"{x} has two {g} parents")
            if len(sp) > 1:
                out.append(f"{x} has {len(sp)} spouses")
            seen, todo = set(), list(par)
            while todo:                                   # ancestors
                p = todo.pop()
                if p == x:
                    out.append(f"{x} is their own ancestor")
                    break
                if p not in seen:
                    seen.add(p)
                    todo += list(self.relation(p, "parent"))
        return out
