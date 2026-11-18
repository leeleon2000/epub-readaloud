"""Combine the evidence for each quotation into a voice and a confidence (docs/DESIGN.md).

Every quote ends with a class and a confidence:
  class   male | female | I (the first-person narrator) | narrator (not speech, or a group) | None
  conf    how sure we are of that class
and is voiced male / female only when conf >= threshold; everything else is read by the narrator.
(The per-person smoothing of 5.6 was removed on 2026-11-14: measured, it changed nothing; 12.2 point 4.) A wrong
gender is the error that matters; falling back to the narrator is cheap (section 2).

Confidence levels (discrete on purpose, so the threshold curve in the eval has clear steps):
  0.95  speech tag with a name / title / noun / pronoun of known gender; "I said"
  0.9   tag whose gender was learned from the book; a non-speech rule (caption, quoted phrase)
  0.85  same paragraph as a tagged quote; scene rules agreeing with the model
  0.8   both model questions agree (ask: both; the second question is the direct voice question, or with
        `voices.second: notes` the `who` question asked again with scene notes and evidence, 15.18);
        in a crowded exchange, the previous paragraph addressed
        the person the model chose
  0.75  scene rule fills a quote the model could not decide
  0.65  model alone, its person's gender agrees with the table
  0.55  model alone, person not in the table (the model's gender is used)
  0.6   a model answer inside a crowded exchange (crowd: that many people of more than one gender)
  0.5   the model contradicts the paragraph's tag
  0.3   the model contradicts a scene rule, or the two model questions disagree, or the quote addresses
        the very person the model says speaks it
  0.2   the model's person and the model's gender disagree with each other
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from . import FEMALE, MALE, NARRATOR
from .prompts import NARRATOR_I, NOT_SPEECH, OTHER_MAN, OTHER_WOMAN, UNKNOWN

PEOPLE = ("male", "female", "I")
GOES_ON = re.compile(r"\b(added|continued|resumed|went on|proceeded|pursued)\b", re.I)
TAG_CONF = {"manual": 0.95, "title": 0.95, "noun": 0.95, "model": 0.85}


def tag_class(q: dict) -> tuple[str | None, float]:
    """From the rules alone: (class, confidence) or (None, 0)."""
    if q.get("rule"):
        return "narrator", 0.9
    kind = q.get("tag_kind")
    if kind == "first_person":
        return "I", 0.95
    if kind == "group":
        return "narrator", 0.9
    g = q.get("tag_gender")
    if kind == "pronoun" and g in ("male", "female"):
        return g, 0.95
    if kind in ("name", "np"):
        if g == "narrator":
            return "I", 0.9                     # genders.yaml: this name is the first-person narrator
        if g in ("male", "female"):
            src = (q.get("tag_gender_source") or "").split(" ")[0].split(":")[0]
            return g, TAG_CONF.get(src, 0.9)
    return None, 0.0


def _person_class(w, gm, tg) -> tuple[str | None, float | None]:
    """Class of a `who` answer: w the person or fixed option, gm the model's gender, tg the table's."""
    if w is None:
        return None, None
    if w == NOT_SPEECH:
        return "narrator", 0.75
    if w == UNKNOWN:
        return None, 0.0
    if w == NARRATOR_I:
        return "I", 0.65
    if w in (OTHER_MAN, OTHER_WOMAN):
        return ("male" if w == OTHER_MAN else "female"), 0.6
    if tg == "narrator":
        return "I", 0.65
    if tg in ("male", "female"):
        if gm in (tg, "unknown", None):
            return tg, 0.65
        return None, 0.2                        # the model's person and its gender disagree
    if gm in ("male", "female"):
        return gm, 0.55
    return None, 0.0


def model_class(q: dict, ask: str) -> tuple[str | None, float]:
    """ask: none | who | voice | both. "voice" is the second question: the direct voice question, or (q["second"]
    == "notes") the `who` question asked again with scene notes and evidence (who2), read the same way as who."""
    def from_who():
        return _person_class(q.get("who"), q.get("who_gender"), q.get("who_table_gender"))

    def from_voice():
        if q.get("second") == "notes":
            return _person_class(q.get("who2"), q.get("who2_gender"), q.get("who2_table_gender"))
        v = q.get("voice_ans")
        if v is None:
            return None, None
        if v in ("male", "female"):
            return v, 0.65
        if v == "narrator":
            return ("I" if q.get("first_person") else "narrator"), 0.6
        return None, 0.0

    if ask == "none":
        return None, None                       # rules only: the baseline of docs/DESIGN.md
    if ask == "who":
        return from_who()
    if ask == "voice":
        return from_voice()
    (a, ca), (b, cb) = from_who(), from_voice()
    if not a and not b:
        return None, (None if ca is None and cb is None else (0.2 if 0.2 in (ca, cb) else 0.0))
    if not a:
        return b, cb                            # one question could not tell: the other one alone
    if not b:
        return a, ca
    if a == b or {a, b} <= {"I", "narrator"}:
        return a, 0.8
    return a, 0.3


def decide(quotes: list[dict], ask: str = "both", threshold: float = 0.7, scene: bool = True, crowd: int = 4,
           vocative: bool = True, narrator_gender: str | None = None) -> dict:
    """quotes: in reading order, each {qid, doc, para, narration, rule, tag_kind, tag_gender, tag_gender_source,
    who, who_gender, who_table_gender, voice_ans, first_person, vocatives}. Sets cls, conf, source, speaker on each.
    narrator_gender: in a first-person book, the narrator's own gender: the same-gender shortcut never fills
    it, because the narrator may be the one speaking (Jake and Bill are both men; 12.6 item 2).
    Returns counts."""
    st = Counter()
    by_para: dict[tuple, list[dict]] = defaultdict(list)
    for q in quotes:
        by_para[(q["doc"], q["para"])].append(q)
        c, v = tag_class(q)
        q.update(cls=c, conf=v, source="tag" if c else None, anchor=c is not None and v >= 0.85)
    for key, qs in by_para.items():
        anchors = {q["cls"] for q in qs if q["anchor"] and q["cls"] in PEOPLE}
        par = next(iter(anchors)) if len(anchors) == 1 else None
        for q in qs:
            if q["cls"] is not None:
                continue
            mc, mv = model_class(q, ask)
            if par:
                if mc in (None, par) or mv is None:
                    q.update(cls=par, conf=0.9 if mc == par else 0.85, source="paragraph")
                elif mc == "narrator":
                    q.update(cls="narrator", conf=mv, source="model")
                else:
                    q.update(cls=par, conf=0.5, source="paragraph")
                    st["paragraph_conflict"] += 1
            elif mv is not None:
                q.update(cls=mc, conf=mv, source="model")
            else:
                q.update(cls=None, conf=0.0, source="none")
    if crowd:
        for run in _runs(quotes):
            if _crowd(run, crowd, st) and vocative:
                _vocatives(run, st)
    if scene:
        for run in _runs(quotes):
            _scene(run, st, narrator_gender)
    for q in quotes:
        q["voiced"] = q["cls"] in ("male", "female") and q["conf"] >= threshold
    for q in quotes:
        q["speaker"] = (MALE if q["cls"] == "male" else FEMALE) if q["voiced"] else NARRATOR
        st[q["source"]] += 1
        st["voiced"] += q["voiced"]
    return dict(st)


def _speech_paras(quotes):
    paras: dict[tuple, list[dict]] = defaultdict(list)
    for q in quotes:
        if q["cls"] != "narrator" or q["source"] not in ("tag",):
            paras[(q["doc"], q["para"])].append(q)
    return paras


def _runs(quotes) -> list[list[list[dict]]]:
    """Runs of consecutive speech paragraphs (same doc), each a list of paragraphs (lists of quotes)."""
    paras = _speech_paras(quotes)
    runs, cur, last = [], [], None
    for key in sorted(paras):
        if cur and not (key[0] == last[0] and key[1] - last[1] == 1):
            runs.append(cur)
            cur = []
        cur.append(paras[key])
        last = key
    if cur:
        runs.append(cur)
    return runs


def _crowd(run: list[list[dict]], n: int, st: Counter) -> None:
    """Several people of both genders in one exchange (a bar, a dinner table): the model loses track of who
    speaks (The Sun Also Rises: Mike's lines given to Brett), so its answers there stay under the threshold."""
    people, genders = set(), set()
    for p in run:
        for q in p:
            if q["source"] == "tag" and q.get("tag_kind") in ("name", "np", "first_person") and q["cls"] in PEOPLE:
                people.add("I" if q["cls"] == "I" else (q.get("tag_text") or "").lower())
                genders.add(q["cls"])
            w = q.get("who")
            if q["source"] == "model" and w and w not in (UNKNOWN, NOT_SPEECH, OTHER_MAN, OTHER_WOMAN) and q["cls"] in PEOPLE:
                people.add("I" if w == NARRATOR_I else w.lower())
                genders.add(q["cls"])
    if len(people) < n or len(genders) < 2:
        return False
    for p in run:
        for q in p:
            if q["source"] == "model":
                q["crowd"] = True
                if q["conf"] > 0.6:
                    q["conf"] = 0.6
                    st["crowd"] += 1
    return True


def _same(a: str | None, b: str | None) -> bool:
    """Two references to one person: equal, or one is a word of the other (Mike / Mike Campbell)."""
    if not a or not b:
        return False
    a, b = a.lower().replace("’", "'"), b.lower().replace("’", "'")
    return a == b or a in b.split() or b in a.split()


def _vocatives(run: list[list[dict]], st: Counter) -> None:
    """Crowded exchange (12.6 item 5): a quote that names someone is not said by them, and the next
    paragraph is usually that person answering. The model's answer is voiced when the address agrees with
    it, and dropped when the quote names its own speaker."""
    flat = [q for p in run for q in p]
    for i, q in enumerate(flat):
        if not q.get("crowd") or q["source"] != "model" or q["cls"] not in ("male", "female"):
            continue
        w = q.get("who")
        if any(_same(w, v) for v in q.get("vocatives") or []):
            q["conf"] = 0.3                      # "Come on, Mike," said by Mike: the model is wrong
            st["vocative_self"] += 1
            continue
        prev = flat[i - 1] if i > 0 else None
        if prev and prev["para"] != q["para"] and any(_same(w, v) for v in prev.get("vocatives") or []):
            q.update(conf=0.8, source="vocative")
            st["vocative_answer"] += 1


def _para_class(qs) -> str | None | bool:
    a = {q["cls"] for q in qs if q["anchor"] and q["cls"] in PEOPLE}
    return None if not a else next(iter(a)) if len(a) == 1 else False


def _scene(run: list[list[dict]], st: Counter, narrator_gender: str | None = None) -> None:
    open_q = [q for p in run for q in p if q["source"] in ("model", "none") and q["cls"] != "narrator"]
    if any(q.get("crowd") for q in open_q):
        return                                  # a crowded exchange: no scene rule either
    if not open_q:
        return
    anchors = [_para_class(p) for p in run]
    if any(a is False for a in anchors):
        return
    known = {a for a in anchors if a}
    said = {q["cls"] for q in open_q if q["cls"] in PEOPLE}
    # same class everywhere: two women talking need no turn-taking to get the gender right
    if len(known) == 1 and said <= known and next(iter(known)) != narrator_gender:
        c = next(iter(known))
        for q in open_q:
            if q["cls"] == c:
                q.update(conf=max(q["conf"], 0.85), source="scene")
                st["scene_same"] += 1
            elif q["cls"] is None and q["conf"] != 0.2:      # 0.2: the model contradicted itself
                q.update(cls=c, conf=0.75, source="scene")
                st["scene_same_fill"] += 1
        return
    # two classes taking turns (split where the narration says the same person goes on)
    for sub in _split_goes_on(run):
        _alternate(sub, st)


def _split_goes_on(run):
    out, cur = [], []
    for p in run:
        if cur and GOES_ON.search(p[0].get("narration") or ""):
            out.append(cur)
            cur = []
        cur.append(p)
    if cur:
        out.append(cur)
    return out


def _alternate(run: list[list[dict]], st: Counter) -> None:
    if len(run) < 3:
        return
    side: dict[int, set] = {0: set(), 1: set()}
    for k, p in enumerate(run):
        a = _para_class(p)
        if a is False:
            return
        if a:
            side[k % 2].add(a)
    if len(side[0]) > 1 or len(side[1]) > 1 or not (side[0] or side[1]) or side[0] == side[1]:
        return
    who = {k: (next(iter(v)) if v else None) for k, v in side.items()}
    for k in (0, 1):
        if who[k] is None:
            c = Counter(q["cls"] for n, p in enumerate(run) if n % 2 == k for q in p
                        if q["cls"] in PEOPLE and q["cls"] != who[1 - k])
            if not c:
                return
            who[k] = c.most_common(1)[0][0]
    used = {q["cls"] for p in run for q in p if q["cls"] in PEOPLE}
    if not used <= {who[0], who[1]}:
        return                                  # a third voice: not a two-person exchange
    for n, p in enumerate(run):
        exp = who[n % 2]
        for q in p:
            if q["source"] not in ("model", "none") or q["cls"] == "narrator":
                continue
            if q["cls"] == exp:
                q.update(conf=max(q["conf"], 0.85), source="scene")
                st["turns_agree"] += 1
            elif q["cls"] is None:
                if q["conf"] != 0.2:            # not a model answer that contradicts itself
                    q.update(cls=exp, conf=0.75, source="scene")
                    st["turns_fill"] += 1
            else:
                q.update(conf=min(q["conf"], 0.3))
                st["turns_conflict"] += 1
