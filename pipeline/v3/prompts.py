"""Prompts and schemas for v3 (docs/DESIGN.md). Each builder returns
(messages, schema, payload); the mock LLM reads only the payload.

Section 15 (asking differently, off by default): `notes()` (scene notes per chunk), `who(notes=, evidence=)`
(the notes of the chunks around, and a quoted piece of evidence + its kind before the answer), `to()` (whom
each quotation is said to, asked on its own so it can disagree with `who`)."""
from __future__ import annotations

import re

from ..memory.prompts import SYSTEM, Sx, _obj

OTHER_MAN, OTHER_WOMAN, NOT_SPEECH, UNKNOWN, NARRATOR_I = (
    "another man", "another woman", "not speech", "unknown", "the narrator (I)")
FIXED = (OTHER_MAN, OTHER_WOMAN, NOT_SPEECH, UNKNOWN)
GENDER3 = ["male", "female", "unknown"]
VOICES = ["male", "female", "narrator", "unknown"]
BASES = ["tag_here", "tag_near", "addressed", "turn", "content", "none"]
SEVERAL, NOBODY = "several people", "nobody"
TO_FIXED = (OTHER_MAN, OTHER_WOMAN, SEVERAL, NOBODY, UNKNOWN)
EVIDENCE_MAX = 80


def plain(marked_text: str) -> str:
    """The passage without the [Qn] marks (the notes are not about who says which quotation)."""
    return re.sub(r"\[/?Q\d+\]", "", marked_text)


def notes(text: str) -> tuple[list[dict], dict, dict]:
    """15.4 step 1, once per chunk: fixed fields, no prose, nothing about who says which quotation."""
    user = f"""Read this passage from a novel and write short scene notes. Do not say who says which quotation.

- narrator: the name of the first-person narrator ("I"), if the passage is told by one and the name is given in it; otherwise "".
- scenes: one entry per scene (a new place or time, or a new group of people talking, starts a new scene). For each scene:
  - where: the place, a few words.
  - people: everyone present, speaking or not, one entry per person (never two people in one entry). For each person: the names, titles or descriptions the passage uses for that one person, copied exactly (e.g. "Mr. Emerson", "the old man", "his father"), and that person's gender.
  - talk: one line, who is talking with whom about what.

PASSAGE:
<<<
{text}
>>>"""
    person = _obj({"names": {"type": "array", "minItems": 1, "maxItems": 6, "items": Sx(60)},
                   "gender": {"type": "string", "enum": GENDER3}})
    schema = _obj({"narrator": Sx(60), "scenes": {"type": "array", "maxItems": 6, "items": _obj({
        "where": Sx(80), "people": {"type": "array", "maxItems": 12, "items": person}, "talk": Sx(200)})}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema, \
        {"task": "v3_notes", "text": text}


def clean_notes(n: dict, text: str) -> dict:
    """Names the passage does not contain are dropped (15.3: only generic checks); people left with no
    name are dropped. Returns a copy with `dropped` listing what went."""
    low = re.sub(r"\s+", " ", text).lower()
    dropped, scenes = [], []
    for sc in n.get("scenes") or []:
        people = []
        for p in sc.get("people") or []:
            keep = [x for x in p.get("names") or [] if x.strip() and re.sub(r"\s+", " ", x.strip()).lower() in low]
            dropped += [x for x in p.get("names") or [] if x not in keep]
            if keep:
                people.append({**p, "names": keep})
        scenes.append({**sc, "people": people})
    nar = n.get("narrator") or ""
    if nar and nar.lower() not in low:
        dropped.append(nar)
        nar = ""
    return {"narrator": nar, "scenes": scenes, "dropped": dropped}


def notes_text(around: list[tuple[str, dict | None]]) -> str:
    """around: [(label, cleaned notes or None)], e.g. previous / this / next passage."""
    out = []
    for label, n in around:
        if not n:
            continue
        lines = [f"{label}:"]
        if n.get("narrator"):
            lines.append(f"  first-person narrator: {n['narrator']}")
        for i, sc in enumerate(n.get("scenes") or [], 1):
            ppl = "; ".join(f"{' / '.join(p['names'])} ({p.get('gender', 'unknown')})" for p in sc.get("people") or [])
            lines.append(f"  scene {i} ({sc.get('where', '')}): {ppl or 'nobody named'}. {sc.get('talk', '')}")
        out.append("\n".join(lines))
    return "\n".join(out) or "(none)"


def _notes_block(notes: str | None) -> str:
    if notes is None:
        return ""
    return f"""SCENE NOTES (written earlier from the previous passage, this passage and the next one; they can be wrong, the passage itself decides):
{notes}

"""


def genders(items: list[dict]) -> tuple[list[dict], dict, dict]:
    """items: [{n, name, examples: [sentence, ...]}] -> {"genders": [{n, gender}]}"""
    lines = []
    for it in items:
        ex = " / ".join(f"“{e[:220]}”" for e in it["examples"]) or "(no example)"
        lines.append(f"{it['n']}: {it['name']}  — e.g. {ex}")
    user = f"""For each name or description below, say whether it refers to a man or boy (male), a woman or girl (female), or cannot be told (unknown). Use the example sentences from the book. Answer "unknown" for groups, things, or when the examples do not tell.

{chr(10).join(lines)}"""
    schema = _obj({"genders": {"type": "array", "maxItems": len(items), "items": _obj({
        "n": {"type": "string", "enum": [it["n"] for it in items]},
        "gender": {"type": "string", "enum": GENDER3}})}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema, \
        {"task": "v3_genders", "items": items}


def who(marked_text: str, todo: list[str], known: list[str], people: list[tuple[str, str]],
        first_person: bool, notes: str | None = None, evidence: bool = False) -> tuple[list[dict], dict, dict]:
    """people: [(name, gender)] named near the scene. -> {"quotes": [{q, speaker, gender}]}
    notes: notes_text() of the passages around (15.4). evidence: each answer starts with
    {evidence, basis} (in that order, before the speaker, so the model finds the evidence first)."""
    plist = "\n".join(f"- {n} ({g if g != 'unknown' else 'gender unknown'})" for n, g in people) or "(none)"
    speakers = [n for n, _ in people] + ([NARRATOR_I] if first_person else []) + list(FIXED)
    fp = (f'"{NARRATOR_I}" when the first-person narrator speaks (“…,” I said); ' if first_person else "")
    user = f"""This novel will be read as an audiobook with three voices: a male voice, a female voice and the narrator. Decide who speaks each quotation listed under TO DECIDE.
Quotations are marked [Q<n>]...[/Q<n>]. One quotation may continue across sentences. The ones under KNOWN are already decided from speech tags: use them as facts.

{_notes_block(notes)}PEOPLE NAMED IN OR NEAR THIS SCENE (one person may appear under several names):
{plist}

KNOWN:
{chr(10).join(known) or '(none)'}

TO DECIDE: {', '.join(todo)}

For each quotation to decide:
{_EVIDENCE if evidence else ""}- speaker: one of the names above; {fp}"{OTHER_MAN}" / "{OTHER_WOMAN}" for a speaker not in the list; "{NOT_SPEECH}" when nobody says it (a quoted word or phrase, a title, a sign, a caption); "{UNKNOWN}" if you cannot tell who speaks.
- gender: the speaker's gender (male, female, or unknown).
A letter read out is speech of its writer; thoughts are speech of the thinker. In an exchange between two people, a new paragraph usually means the other person speaks.

PASSAGE:
<<<
{marked_text}
>>>"""
    item = {"q": {"type": "string", "enum": todo}}
    if evidence:
        item |= {"evidence": Sx(EVIDENCE_MAX), "basis": {"type": "string", "enum": BASES}}
    item |= {"speaker": {"type": "string", "enum": speakers}, "gender": {"type": "string", "enum": GENDER3}}
    schema = _obj({"quotes": {"type": "array", "maxItems": len(todo), "items": _obj(item)}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema, \
        {"task": "v3_who", "text": marked_text, "todo": todo, "people": people, "speakers": speakers,
         "first_person": first_person, "notes": notes is not None, "evidence": evidence}


_EVIDENCE = f"""- evidence: first, copy from the passage the few words (at most {EVIDENCE_MAX} characters) that show who says it: a speech tag ("said Lucy"), narration naming the speaker, a name the speaker is addressed by. Copy them exactly; "" if there are none.
- basis: what the evidence is: "tag_here" a speech tag in the same paragraph; "tag_near" a speech tag or narration in the paragraph before or after; "addressed" the quotation answers someone who was named or addressed; "turn" two people take turns and this is the other one's turn; "content" only what is said tells who says it; "none" nothing tells.
"""


def to(marked_text: str, todo: list[str], known: list[str], people: list[tuple[str, str]],
       first_person: bool, notes: str | None = None) -> tuple[list[dict], dict, dict]:
    """15.4: whom each quotation is said to. Asked on its own (it never sees the `who` answers), so the two
    can disagree. -> {"quotes": [{q, to}]}"""
    plist = "\n".join(f"- {n} ({g if g != 'unknown' else 'gender unknown'})" for n, g in people) or "(none)"
    hearers = [n for n, _ in people] + ([NARRATOR_I] if first_person else []) + list(TO_FIXED)
    fp = f'"{NARRATOR_I}" when it is said to the first-person narrator; ' if first_person else ""
    user = f"""Quotations in this passage from a novel are marked [Q<n>]...[/Q<n>]. One quotation may continue across sentences. For each quotation listed under TO DECIDE, say to whom it is said (the listener), not who says it.
The ones under KNOWN have known speakers: use them as facts.

{_notes_block(notes)}PEOPLE NAMED IN OR NEAR THIS SCENE (one person may appear under several names):
{plist}

KNOWN:
{chr(10).join(known) or '(none)'}

TO DECIDE: {', '.join(todo)}

to: one of the names above; {fp}"{OTHER_MAN}" / "{OTHER_WOMAN}" for a listener not in the list; "{SEVERAL}" when it is said to a group; "{NOBODY}" when it is said to nobody (talking to oneself, a thought, a letter, or not speech at all); "{UNKNOWN}" if you cannot tell.

PASSAGE:
<<<
{marked_text}
>>>"""
    schema = _obj({"quotes": {"type": "array", "maxItems": len(todo), "items": _obj({
        "q": {"type": "string", "enum": todo},
        "to": {"type": "string", "enum": hearers}})}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema, \
        {"task": "v3_to", "text": marked_text, "todo": todo, "people": people, "hearers": hearers,
         "first_person": first_person, "notes": notes is not None}


def evidence_in_text(ev: str, text: str) -> bool:
    """The quoted evidence occurs in the passage (spaces, quote marks and case ignored)."""
    def n(x):
        x = re.sub(r"\[/?Q\d+\]", "", x)
        x = x.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        return re.sub(r"[\s\"']+", " ", x).strip().lower()
    e = n(ev or "")
    return bool(e) and e in n(text)


def voice(marked_text: str, todo: list[str], known: list[str], first_person: bool) -> tuple[list[dict], dict, dict]:
    """The direct question (docs/DESIGN.md): male / female / narrator, no names."""
    fp = " the first-person narrator speaking (“…,” I said), or" if first_person else ""
    user = f"""This novel will be read as an audiobook with three voices: a male voice, a female voice and the narrator. Choose the voice for each quotation listed under TO DECIDE.
Quotations are marked [Q<n>]...[/Q<n>]. One quotation may continue across sentences. The ones under KNOWN are already decided from speech tags: use them as facts.

KNOWN:
{chr(10).join(known) or '(none)'}

TO DECIDE: {', '.join(todo)}

voice: "male" if a man or boy says it, "female" if a woman or girl says it, "narrator" if it is{fp} not speech at all (a quoted word or phrase, a title, a sign, a caption), "unknown" if you cannot tell who speaks.
A letter read out is speech of its writer; thoughts are speech of the thinker. In an exchange between two people, a new paragraph usually means the other person speaks.

PASSAGE:
<<<
{marked_text}
>>>"""
    schema = _obj({"quotes": {"type": "array", "maxItems": len(todo), "items": _obj({
        "q": {"type": "string", "enum": todo},
        "voice": {"type": "string", "enum": VOICES}})}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], schema, \
        {"task": "v3_voice", "text": marked_text, "todo": todo, "first_person": first_person}
