"""The name -> gender table (docs/DESIGN.md). It is the only "roster" v3 has: it says that
Elizabeth is female and the Jew is male, never who is who.

Sources, in order (the first that decides wins):
  1. manual   genders.yaml `manual:` (human edits always win)
  2. title    Mr., Mrs., Miss, Lady, Sir ... (pipeline/memory/names.py)
  3. noun     the head noun has a gender: the girl, the old gentleman, her mother
  4. pronouns learned from the book: he / she right after the name in narration (nearest antecedent)
  5. name     a full name takes the gender of its given name; a bare surname the gender of the titles it
              is used with in the book (only Mr. Darcy -> Darcy is male)
  6. model    whatever is left, one call per batch of names with two example sentences each (cached)
Names no source decides are "unknown": a quote that needs them falls back to the narrator.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ..memory.names import FEMALE_TITLES, MALE_TITLES, TITLE_ALIASES
from ..store import get_meta, set_meta
from .tags import CAP, NAME, TITLE, norm_phrase

MALE_NOUNS = {"man", "gentleman", "boy", "lad", "fellow", "father", "son", "husband", "brother", "uncle", "nephew",
              "king", "prince", "lord", "sir", "master", "mister", "monsieur", "papa", "grandfather", "grandson",
              "bridegroom", "groom", "landlord", "gentlemen", "men", "boys", "sire", "duke", "earl", "baron", "count",
              "beau", "bachelor", "widower", "monk", "priest", "abbot", "friar", "squire", "chap", "youth",
              "stepfather", "step-father", "father-in-law", "brother-in-law", "son-in-law", "schoolboy", "waiter",
              "headwaiter", "barman", "manservant", "footman", "butler", "valet", "postman", "policeman",
              "fisherman", "boatman", "coachman", "nobleman", "clergyman", "workman",
              "dad", "daddy", "pa", "hero", "emperor", "cardinal", "bishop"}
FEMALE_NOUNS = {"woman", "lady", "girl", "lass", "mother", "daughter", "wife", "sister", "aunt", "niece", "queen",
                "princess", "madam", "madame", "matron", "maid", "maiden", "mistress", "widow", "landlady",
                "governess", "hostess", "actress", "waitress", "mamma", "mama", "grandmother", "granddaughter",
                "bride", "gentlewoman", "dame", "duchess", "countess", "baroness", "heroine", "nun", "abbess",
                "stepmother", "step-mother", "mother-in-law", "sister-in-law", "daughter-in-law", "schoolgirl",
                "housemaid", "chambermaid", "milliner", "laundress", "seamstress", "spinster", "lady's",
                "ladyship", "women", "ladies", "girls", "mum", "mummy", "ma", "wench",
                "damsel", "maidservant", "nursemaid", "dowager", "empress", "señora", "senora", "mademoiselle"}
PRON_M = {"he", "himself"}                     # subjects only: "Darcy took her hand" / "Elizabeth took his
PRON_F = {"she", "herself"}                    # hand" are about someone else
STOP_NAMES = {"I", "The", "A", "An", "He", "She", "It", "They", "We", "You", "But", "And", "Then", "When", "What",
              "Why", "How", "Where", "Who", "Yes", "No", "Oh", "Ah", "Well", "There", "This", "That", "Not", "So",
              "If", "As", "At", "In", "On", "Of", "To", "For", "By", "With", "My", "Your", "His", "Her", "Our",
              "Their", "Its", "Do", "Did", "Is", "Was", "Be", "Are", "Have", "Had", "Will", "Would", "Shall",
              "Should", "Can", "Could", "May", "Might", "Must", "Now", "Here", "Pray", "Indeed", "Very", "Chapter",
              "CHAPTER", "Mr", "Mrs", "Miss", "Sir", "Madam", "Lady", "Lord", "God", "Heaven", "Heavens", "Lord's",
              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Christmas", "English",
              "French", "Spanish", "American", "Americans", "England", "London", "Paris", "Spain", "Good", "Some",
              "All", "One", "Every", "Each", "Any", "Nothing", "Everything", "Only", "Just", "Even", "Still", "Perhaps",
              "Mamma", "Papa", "Sure", "Hullo", "Hello", "Listen", "Look", "Come", "Go", "Let", "Don't", "Please",
              "Thank", "Thanks", "Dear", "Oliver's", "Sir's"}
GENDERS = ("male", "female")
# a name votes only as the subject of its clause: "Elizabeth was silent; but she ..." yes,
# "taking the arm of Mr. Darcy, she ..." no
_SUBJ_BEFORE = re.compile(r"(?:^|[.;:!?—]\s*|[,(]\s*(?:(?:and|but|while|when|as|though|then|so|for|yet)\s+)?"
                          r"|\b(?:and|but|while|when|as|though|then|so|yet)\s+)$")
_SUBJ_AFTER = re.compile(r"\s+(?:had\s+|was\s+|could\s+|would\s+|did\s+|now\s+|then\s+|only\s+|still\s+|soon\s+)?[a-z]")
MAX_GAP = 60                                   # chars between a name and the pronoun that votes for it
_OTHER_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+")   # another capitalized word in between: maybe another person
NAME_RE = re.compile(rf"(?<![A-Za-z’'])({NAME})(?![A-Za-z])")


# Descriptions that name a different person from scene to scene (docs/DESIGN.md): "the other",
# "his companion", "the child" have no gender of their own, so they stay out of the table and the model
# decides from the scene. A possessive description (his friend, her visitor) is always relative.
GENERIC_HEADS = {"other", "others", "companion", "child", "person", "stranger", "servant", "friend", "visitor", "figure",
                 "voice", "one", "speaker", "creature", "individual", "party", "attendant", "guest", "traveller",
                 "traveler", "infant", "baby", "newcomer", "neighbour", "neighbor", "former", "latter", "first",
                 "second", "third", "last", "same", "listener", "questioner", "interlocutor", "partner", "fellow-traveller",
                 "acquaintance", "relation", "relative", "cousin", "patient", "prisoner", "culprit", "victim",
                 "farmer", "landlord's", "wretch", "thing", "kid", "youngster", "orphan", "boarder", "lodger"}
SCENE_DEPENDENT = "description: decided per scene"


def scene_dependent(phrase: str) -> bool:
    words = phrase.replace("’", "'").lower().split()
    if len(words) < 2 or phrase[:1].isupper():
        return False
    if words[0] in ("his", "her", "their", "my", "our", "your", "its"):
        return True
    return words[-1].strip(",.;:!?") in GENERIC_HEADS


def title_of(phrase: str) -> str | None:
    m = re.match(rf"^({TITLE})\s", phrase)
    if not m:
        return None
    t = m.group(1).rstrip(".").lower()
    t = TITLE_ALIASES.get(t, t)
    t = {"mme": "madame", "mlle": "miss", "mister": "mr", "monsieur": "mr"}.get(t, t)
    return "male" if t in MALE_TITLES else "female" if t in FEMALE_TITLES else None


def noun_gender(phrase: str) -> str | None:
    words = [w.lower().strip(",.;:!?") for w in phrase.replace("’", "'").split()]
    if len(words) < 2 and not phrase[:1].islower():
        return None
    head = words[-1] if words else ""
    if head.endswith("'s"):
        return None                  # "her mother's" is not a person here
    return "male" if head in MALE_NOUNS else "female" if head in FEMALE_NOUNS else None


class Table:
    """key (norm_phrase) -> (gender, source). `gender` is male / female / unknown."""

    def __init__(self, entries: dict[str, tuple[str, str]] | None = None):
        self.entries = dict(entries or {})

    def get(self, phrase: str) -> str:
        return self.entries.get(norm_phrase(phrase), ("unknown", ""))[0]

    def source(self, phrase: str) -> str:
        return self.entries.get(norm_phrase(phrase), ("unknown", ""))[1]

    def to_json(self) -> str:
        return json.dumps(self.entries, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str | None) -> "Table":
        return cls({k: tuple(v) for k, v in json.loads(s or "{}").items()})


def path(book_dir: Path) -> Path:
    return book_dir / "genders.yaml"


def load_manual(book_dir: Path) -> dict[str, str]:
    p = path(book_dir)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    out = {}
    for k, v in (data.get("manual") or {}).items():
        v = str(v).lower()
        if v in ("male", "female", "unknown", "narrator"):
            out[norm_phrase(str(k))] = v
    return out


def save(book_dir: Path, table: Table, manual: dict[str, str], counts: Counter) -> None:
    lines = ["# Name -> gender for v3 attribution (docs/DESIGN.md).",
             "# Edit `manual:` (name: male | female | unknown | narrator); it always wins. `auto:` is rewritten",
             "# on every run; the comment says where each answer came from and how often the name is used.",
             "manual:"]
    lines += [f"  {json.dumps(k, ensure_ascii=False)}: {v}" for k, v in sorted(manual.items())] or ["  {}"]
    if not manual:
        lines[-1] = "manual: {}"
        lines.pop(-2)
    lines.append("auto:")
    for k, (g, src) in sorted(table.entries.items(), key=lambda kv: (-counts.get(kv[0], 0), kv[0])):
        if k in manual:
            continue
        lines.append(f"  {json.dumps(k, ensure_ascii=False)}: {g}   # {src}; used {counts.get(k, 0)}x")
    path(book_dir).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- vocabulary

def vocatives(quote_text: str) -> list[str]:
    """Names a speaker addresses: “Come, Darcy,” / “Lizzy, my dear” / “my dear Jane.”"""
    out = []
    for m in re.finditer(rf"(?:^|[,;!?]\s+|\bdear\s+|\bdearest\s+)({NAME})(?=\s*(?:[,!?.;—]|$))", quote_text):
        n = m.group(1)
        if n.split()[0] not in STOP_NAMES:
            out.append(n)
    return out


def narration_names(text: str) -> list[str]:
    """Capitalized names in narration, not counting a word that only starts the sentence."""
    out = []
    for m in NAME_RE.finditer(text):
        n = m.group(1)
        words = n.split()
        while words and words[0] in STOP_NAMES:
            words.pop(0)
        if not words:
            continue
        n2 = " ".join(words)
        start = m.start(1) + (len(n) - len(n2))
        if start == 0 and not title_of(n2 + " ") and len(words) == 1:
            continue                             # sentence-initial single word: could be any capitalized word
        if title_of(n2 + " ") and len(words) == 1:
            continue
        out.append(n2)
    return out


# ---------------------------------------------------------------- learning from pronouns

def learn_pronouns(paragraphs: list[list[tuple[str, bool]]], phrases: set[str]) -> dict[str, Counter]:
    """paragraphs: per paragraph, (narration of the sentence with the quotes cut out, has a quote) per sentence.
    For every he / his / she / herself..., vote for the nearest phrase before it in the same sentence, or
    the only phrase of the previous sentence. A sentence with a quote neither uses nor passes on the
    previous sentence (“Yes,” said Elizabeth. “No,” said he: "he" is not Elizabeth).
    Returns key -> Counter(male=, female=)."""
    if not phrases:
        return {}
    names = sorted({p for p in phrases if p[:1].isupper()}, key=len, reverse=True)
    descs = sorted({p for p in phrases if not p[:1].isupper()}, key=len, reverse=True)
    alts = [re.escape(p) for p in names] + ["(?i:" + re.escape(p).replace(r"\ ", r"\s+") + ")" for p in descs]
    pat = re.compile(r"(?<![A-Za-z’'])(" + "|".join(alts) + r"|(?i:he|himself|she|herself))(?![A-Za-z’'])")
    votes: dict[str, Counter] = defaultdict(Counter)
    for para in paragraphs:
        prev_subject = None
        for sent, has_quote in para:
            last, last_end, seen, first = None, -1, [], None
            for m in pat.finditer(sent):
                w = m.group(1)
                lw = w.lower()
                if lw in PRON_M or lw in PRON_F:
                    who = None
                    if last and m.start() - last_end <= MAX_GAP and not _OTHER_CAP.search(sent[last_end:m.start()]):
                        who = last
                    elif last is None and not has_quote and m.start() == 0:
                        who = prev_subject      # "Elizabeth went in. She ..." (only a sentence-initial pronoun)
                    if who:
                        votes[who]["male" if lw in PRON_M else "female"] += 1
                    last = None                 # the pronoun is now the nearest person
                    continue
                subject = _SUBJ_BEFORE.search(sent[:m.start()]) and _SUBJ_AFTER.match(sent[m.end():])
                last, last_end = (norm_phrase(w) if subject else None), m.end()
                if first is None:
                    first = (last, m.start())
                if last not in seen:
                    seen.append(last)
            prev_subject = first[0] if first and first[1] == 0 and len(seen) == 1 and not has_quote else None
    return votes


def decide_votes(c: Counter, min_votes: int = 4, share: float = 0.85) -> str | None:
    tot = c["male"] + c["female"]
    if tot < min_votes:
        return None
    g = "male" if c["male"] >= c["female"] else "female"
    return g if c[g] / tot >= share else None


# ---------------------------------------------------------------- building the table

def build(phrases: Counter, paragraphs: list[list[tuple[str, bool]]], manual: dict[str, str], examples: dict[str, list[str]],
          llm=None, cached_llm: dict[str, str] | None = None, batch: int = 40, log=print, obs=None,
          workers: int = 4) -> tuple[Table, dict[str, str]]:
    """phrases: surface form -> uses (tags, vocatives, names in narration). Returns (table, model answers)."""
    keys: dict[str, str] = {}
    for p in phrases:
        keys.setdefault(norm_phrase(p), p)
    entries: dict[str, tuple[str, str]] = {}
    for k, v in manual.items():
        entries[k] = (v, "manual")
    for k, p in keys.items():
        if k in entries:
            continue
        g = title_of(p + " ")
        if g:
            entries[k] = (g, "title")
            continue
        g = noun_gender(p)
        if g:
            entries[k] = (g, "noun")
        elif scene_dependent(p):
            entries[k] = ("unknown", SCENE_DEPENDENT)
    # bare names from the titles they take in the book: Mr. Darcy -> Darcy; Miss Elizabeth -> Elizabeth;
    # Mr. and Mrs. Bennet -> Bennet decides nothing. A female title passes on only to a given name: with
    # only "Mrs. Braddocks" in the book, "Braddocks" is still her husband (The Sun Also Rises).
    from ..memory.names import is_given
    titled = defaultdict(set)
    for k, p in keys.items():
        m = re.match(rf"^({TITLE})\s+({CAP}(?:\s+{CAP})*)$", p)
        if m:
            g = title_of(p + " ")
            words = m.group(2).split()
            for i, w in enumerate(words):
                if g == "male" or (g == "female" and i == 0 and len(words) >= 1 and is_given(w.lower())):
                    titled[norm_phrase(w)].add(g)
                elif g == "female":
                    titled[norm_phrase(w)].add("unknown")
    for k, p in keys.items():
        if k not in entries and len(p.split()) == 1 and len(titled.get(k, ())) == 1 and "unknown" not in titled[k]:
            entries[k] = (next(iter(titled[k])), "with title")
    votes = learn_pronouns(paragraphs, {p for k, p in keys.items() if k not in entries})
    weak = {}
    for k, p in keys.items():
        if k in entries:
            continue
        c = votes.get(k, Counter())
        g = decide_votes(c)
        if g:
            entries[k] = (g, f"pronouns {c['male']}:{c['female']}")
        elif decide_votes(c, 3, 0.75):
            weak[k] = (decide_votes(c, 3, 0.75), f"pronouns (weak) {c['male']}:{c['female']}")
    # full names from their given name
    for k, p in keys.items():
        words = p.split()
        if k not in entries and len(words) >= 2 and p[:1].isupper():
            g0 = entries.get(norm_phrase(words[0]), (None,))[0]
            if g0 in GENDERS:
                entries[k] = (g0, f"given name {words[0]}")
    # the model, for what is left
    cached = dict(cached_llm or {})
    left = [k for k in keys if k not in entries]
    ask = [k for k in left if k not in cached]
    if ask and llm is not None:
        from . import prompts
        groups = [ask[i:i + batch] for i in range(0, len(ask), batch)]
        from concurrent.futures import ThreadPoolExecutor

        def one(g):
            items = [{"n": f"N{i + 1}", "name": keys[k], "examples": examples.get(k, [])[:2]} for i, k in enumerate(g)]
            msgs, schema, payload = prompts.genders(items)
            out = llm.complete_json(msgs, schema, "v3_genders", payload, max_tokens=60 * len(g) + 200)
            ans = {d["n"]: d["gender"] for d in out.get("genders", [])}
            return {k: ans.get(f"N{i + 1}", "unknown") for i, k in enumerate(g)}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for res in ex.map(one, groups):
                cached.update(res)
        log(f"voices: asked the model for the gender of {len(ask)} names ({len(groups)} calls)")
    for k in left:
        g = cached.get(k)
        if g in GENDERS:
            entries[k] = (g, "model")
        elif k in weak:
            entries[k] = weak[k]
        else:
            entries[k] = ("unknown", "model: unknown" if g else "no evidence")
    return Table(entries), cached


def load_table(con) -> Table:
    return Table.from_json(get_meta(con, "v3_genders"))


def store_table(con, table: Table, cached_llm: dict[str, str]) -> None:
    set_meta(con, "v3_genders", table.to_json())
    set_meta(con, "v3_gender_llm", json.dumps(cached_llm, ensure_ascii=False))
