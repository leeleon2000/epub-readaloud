"""Names: parsing, title compatibility, and name-graph links (docs/DESIGN.md).

Rules only; nothing here calls a model. Used by the P2 roster step and the validation layer.

    parse("Miss Eliza Bennet")      -> Name(title='miss', given='eliza', surname='bennet', kind='name')
    compatible("Miss Lucas", "Lady Lucas").ok        -> False  (Miss vs Mrs./Lady)
    compatible("Lizzy", "Elizabeth").ok              -> True   (nickname)
    conflicts([...names on one card...])             -> incompatible pairs
    link("Lizzy", "Elizabeth")                       -> ("medium", "nickname")   name-graph edge
    full_name_key("Lizzy Bennet")                    -> ("elizabeth", "bennet")

Policy (docs/DESIGN.md): when in doubt, names are *compatible* here and the model
decides with context; `compatible() == False` is a hard rule (different people) that the model's
answer cannot override except through marriage evidence (`married=True`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]

MALE_TITLES = {"mr", "sir", "lord", "master", "colonel", "captain", "general", "major", "lieutenant", "rev",
               "reverend", "father", "uncle", "king", "prince", "duke", "count", "earl", "baron"}
FEMALE_TITLES = {"mrs", "miss", "ms", "lady", "madam", "madame", "mistress", "mother", "aunt", "queen",
                 "princess", "duchess", "countess", "baroness"}
NEUTRAL_TITLES = {"dr", "doctor", "professor", "mx", "st", "saint"}
TITLES = MALE_TITLES | FEMALE_TITLES | NEUTRAL_TITLES
TITLE_ALIASES = {"col": "colonel", "capt": "captain", "prof": "professor", "gen": "general", "lieut": "lieutenant"}
UNMARRIED = {"miss"}
MARRIED = {"mrs", "lady"}          # Miss vs Mrs./Lady: different people unless married (PLAN)

ADDRESS = {"sir", "madam", "madame", "ma'am", "maam", "my dear", "dear", "my dearest", "dearest", "my love",
           "my lord", "my lady", "my child", "my girl", "my boy", "my friend", "my good friend", "darling"}
ADDRESS_PREFIX = {"my", "dear", "dearest", "good", "little", "poor"}   # "my little Lizzy", "poor Jane"
HONORIFICS = {"right", "honourable", "honorable", "hon", "most", "noble", "worshipful"}
KINSHIP = {"mamma", "mama", "mamma's", "papa", "father", "mother", "aunt", "uncle", "cousin", "sister", "brother",
           "grandmamma", "grandmama", "grandpapa", "grandfather", "grandmother", "son", "daughter", "wife",
           "husband", "nephew", "niece"}
POSSESSIVE = {"his", "her", "their", "my", "our", "your", "its"}
ROLE_WORDS = {"ladyship", "lordship", "majesty", "highness", "grace", "honour", "honor", "reverence"}
GENERATION = {"old", "young", "late", "elder", "younger"}
ARTICLES = {"the", "a", "an"}
PARTICLES = {"de", "da", "di", "du", "van", "von", "der", "le", "la", "del", "d'"}
OCCUPATIONS = {"housekeeper", "butler", "servant", "footman", "maid", "housemaid", "coachman", "clergyman",
               "apothecary", "steward", "gardener", "cook", "porter", "waiter", "landlord", "landlady",
               "governess", "nurse", "physician", "surgeon", "lawyer", "attorney", "officer", "soldier",
               "postboy", "postman", "groom", "valet", "page", "messenger", "shopkeeper", "rector", "curate",
               "vicar", "parson", "magistrate", "judge", "sailor", "captain"}
LOOKUP_KINDS = {"name", "given", "nickname", "title_given"}     # may be looked up directly (shared forms of address go to the model)
NAME_KINDS = LOOKUP_KINDS | {"title_surname", "surname", "generation"}


@lru_cache(maxsize=1)
def _nick() -> tuple[dict[str, set[str]], set[str]]:
    """(formal name -> nicknames, all given names)."""
    data = yaml.safe_load((REPO / "data" / "nicknames.yaml").read_text()) or {}
    formal = {k.lower(): {v.lower() for v in vs} for k, vs in data.items()}
    given = set(formal) | {v for vs in formal.values() for v in vs}
    p = REPO / "data" / "given_names.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if not line.startswith("#"):
                given.update(w.lower() for w in line.split())
    return formal, given


def is_given(w: str) -> bool:
    return w.lower() in _nick()[1]


def formal_names(g: str) -> set[str]:
    """Formal given names that `g` can stand for: lizzy -> {elizabeth}; kit -> {catherine, christopher}."""
    g = g.lower()
    formal, _ = _nick()
    out = {f for f, nicks in formal.items() if g in nicks}
    if g in formal or not out:
        out.add(g)
    return out


def given_forms(g: str) -> set[str]:
    """`g`, its formal names and their nicknames."""
    formal, _ = _nick()
    out = {g.lower()}
    for f in formal_names(g):
        out |= {f} | formal.get(f, set())
    return out


def title_gender(title: str | None) -> str | None:
    if title in MALE_TITLES:
        return "male"
    if title in FEMALE_TITLES:
        return "female"
    return None


@dataclass(frozen=True)
class Name:
    text: str
    title: str | None = None
    given: str | None = None
    surname: str | None = None
    kind: str = "other"        # name given nickname title_given title_surname surname generation group
    #                            address kinship relational occupation title_role description other

    @property
    def lookup(self) -> bool:
        return self.kind in LOOKUP_KINDS

    @property
    def is_name(self) -> bool:
        return self.kind in NAME_KINDS

    @property
    def gender(self) -> str | None:
        return title_gender(self.title)

    @property
    def tokens(self) -> set[str]:
        return {w for part in (self.given, self.surname) if part for w in part.split()}


_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")


@lru_cache(maxsize=4096)
def parse(text: str) -> Name:
    raw = [w.replace("’", "'") for w in _WORD.findall(text)]
    if raw and raw[-1].lower().endswith("'s"):
        raw[-1] = raw[-1][:-2]
    raw = [w for w in raw if w]
    low = [w.lower() for w in raw]
    if not low:
        return Name(text)
    if " ".join(low) in ADDRESS:
        return Name(text, kind="address")
    if re.search(r"\b(?:Mr|Mrs|Miss|Lady|Sir|Dr)\.?\s*(?:&|and)\s*(?:Mr|Mrs|Miss|Lady|Sir|Dr)\b", text):
        return Name(text, surname=low[-1], kind="group")          # "Mr. & Mrs. Bennet"
    while len(low) > 1 and low[0] in ADDRESS_PREFIX and low[1] not in ROLE_WORDS and low[1] not in KINSHIP:
        raw, low = raw[1:], low[1:]                          # "My dear Mr. Bennet" -> "Mr. Bennet"
    if " ".join(low) in ADDRESS:
        return Name(text, kind="address")
    if low[0] in POSSESSIVE:
        return Name(text, kind="title_role" if set(low[1:]) & ROLE_WORDS else "relational")
    article = low[0] if low[0] in ARTICLES else None
    if article:
        raw, low = raw[1:], low[1:]
    while len(low) > 1 and low[0] in HONORIFICS:
        raw, low = raw[1:], low[1:]                          # "the Right Honourable Lady Catherine"
    gen = False
    while low and low[0] in GENERATION and len(low) > 1:
        gen, raw, low = True, raw[1:], low[1:]
    if not low:
        return Name(text, kind="description")
    if len(low) == 1 and low[0] in KINSHIP:
        return Name(text, kind="kinship")
    title = None
    t0 = TITLE_ALIASES.get(low[0], low[0])
    if t0 in TITLES and (len(low) > 1 or article):
        title, raw, low = t0, raw[1:], low[1:]
        if not low:
            return Name(text, title=title, kind="title_role")          # "the Colonel"
    if any(not w[0].isupper() and lw not in PARTICLES for w, lw in zip(raw, low)):
        return Name(text, kind="occupation" if low[-1] in OCCUPATIONS else "description")
    if article == "the" and not gen and not title and len(low) == 1 and low[0].endswith("s") \
            and not is_given(low[0]):
        base = low[0][:-2] if low[0].endswith("ses") else low[0][:-1]
        return Name(text, surname=base, kind="group")                  # "the Bennets"
    if len(low) == 1:
        w = low[0]
        if title:
            given, surname, kind = (w, None, "title_given") if is_given(w) else (None, w, "title_surname")
        elif is_given(w):
            formal, _ = _nick()
            nick = w not in formal and any(w in v for v in formal.values())
            given, surname, kind = w, None, ("nickname" if nick else "given")
        else:
            given, surname, kind = None, w, "surname"
    else:
        given, surname, kind = low[0], " ".join(low[1:]), "name"
    if gen:
        kind = "generation"
    return Name(text, title, given, surname, kind)


@dataclass(frozen=True)
class Compat:
    ok: bool
    reason: str


def compatible(a: str, b: str, married: bool = False) -> Compat:
    """Can these two references be the same person? False only on hard evidence."""
    x, y = parse(a), parse(b)
    if not (x.is_name and y.is_name):
        return Compat(True, "not both names")
    gx, gy = x.gender, y.gender
    if gx and gy and gx != gy:
        return Compat(False, f"title gender: {x.title} vs {y.title}")
    if not married and {x.title, y.title} & UNMARRIED and {x.title, y.title} & MARRIED:
        return Compat(False, f"{x.title} vs {y.title}: unmarried and married titles")
    if x.given and y.given and not formal_names(x.given) & formal_names(y.given):
        return Compat(False, f"given names differ: {x.given} vs {y.given}")
    if x.surname and y.surname and x.surname != y.surname and not _spelling(x.surname, y.surname):
        if married and "male" not in (gx, gy):
            return Compat(True, "surname changed by marriage")
        return Compat(False, f"surnames differ: {x.surname} vs {y.surname}")
    if x.kind == "generation" or y.kind == "generation":
        if x.kind != y.kind:
            return Compat(False, "generation (old/young/late) vs plain name")
    return Compat(True, "compatible")


def _spelling(a: str, b: str) -> bool:
    """Spelling variants of one surname (Denny/Denney, Philips/Phillips): one edit, 5+ letters."""
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(p != q for p, q in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def conflicts(names: list[str], married: bool = False) -> list[tuple[str, str, str]]:
    """Pairs of names on one card that cannot belong to one person."""
    out = []
    for a, b in combinations(dict.fromkeys(names), 2):
        c = compatible(a, b, married)
        if not c.ok:
            out.append((a, b, c.reason))
    return out


def link(a: str, b: str) -> tuple[str, str] | None:
    """Name-graph edge between two references, without context (P1 same-chunk evidence is separate):
    ("medium", "contains" | "nickname") when one name extends the other or is its nickname,
    ("weak", "surname") for a shared surname (grouping only), None otherwise."""
    x, y = parse(a), parse(b)
    if not (x.is_name and y.is_name):
        return None
    c = compatible(a, b)
    if c.ok:
        small, big = (x, y) if len(x.tokens) <= len(y.tokens) else (y, x)
        if small.tokens and small.tokens <= big.tokens and (small.given or not big.given):
            return ("medium", "contains")
        if x.given and y.given and x.given != y.given and formal_names(x.given) & formal_names(y.given):
            return ("medium", "nickname")
    if x.surname and y.surname and (x.surname == y.surname or _spelling(x.surname, y.surname)):
        return ("weak", "surname")
    return None


def full_name_key(text: str) -> tuple[str, str] | None:
    """(formal given name, surname) when the reference has both: two cards sharing one are the same
    person or a real clash (ALIAS_CONFLICT)."""
    n = parse(text)
    if n.given and n.surname and n.kind in ("name", "generation"):
        return (min(formal_names(n.given)), n.surname)
    return None
