"""Prompts and JSON schemas for the character passes. Each builder returns
(messages, schema, payload); payload is the structured input the mock LLM uses."""
from __future__ import annotations

# Bump when a prompt or schema changes; recorded with every LLM call (llm_call.prompt_version).
PROMPT_VERSION = "2026-11-14.3"   # .3 v3 voices (pipeline/v3/prompts.py); .4 attribute codes C1..Cn; .5/.6 v2 P1 collect; 11-14.1 v2 P2 roster; .2 relations on attribute cards

GENDERS = ["female", "male", "unknown"]
AGES = ["child", "teen", "young adult", "adult", "middle-aged", "elderly", "unknown"]

SYSTEM = ("You are a careful literary analyst preparing a novel for a multi-voice audiobook. "
          "Answer only with JSON matching the schema. Use only information in the given text and cards.")


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or list(props),
            "additionalProperties": False}


S = {"type": "string"}


def Sx(n: int) -> dict:
    """String with a maximum length (enforced by vLLM structured outputs)."""
    return {"type": "string", "maxLength": n}


# ------------------------------------------------------------------ 1. extract mentions

EXTRACT_SCHEMA = _obj({"mentions": {"type": "array", "maxItems": 80, "items": _obj({
    "text": Sx(80),
    "kind": {"type": "string", "enum": ["name", "title", "description", "relational", "group", "generic"]},
    "traits": Sx(300),
})}})


def extract(chunk_text: str) -> tuple[list[dict], dict, dict]:
    user = f"""List every character referred to in the passage: people, animals or beings that act, speak, or are spoken to or about.
Include each distinct way a character is referred to (a proper name, a title such as "the White Rabbit", or a description such as "her sister").
Do not list places, objects, or pronouns alone.

kind:
- name: a proper name (Alice, Mr. Darcy)
- title: a fixed epithet used like a name (the Hatter, the Queen of Hearts)
- description: a descriptive reference to one individual (the old fisherman, the stranger)
- relational: one individual identified only through someone else (his aunt, Charlotte Lucas's papa, her sister)
- group: several people together (the Eltons, Mrs. Long's nieces, the officers, our girls)
- generic: unidentifiable (someone, a voice)
Only list people who exist in the story's world. Do not list the author, critics, or characters of other books mentioned in passing.

traits: facts this passage reveals about that character (gender, age, role, manner of speaking, relationships), or "".

PASSAGE:
<<<
{chunk_text}
>>>"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], EXTRACT_SCHEMA, \
        {"task": "extract", "text": chunk_text}


# ------------------------------------------------------------------ 2. resolve mentions

def resolve_schema(n_mentions: int, profile_chars: int = 400) -> dict:
    return _obj({
        "resolutions": {"type": "array", "maxItems": n_mentions, "items": _obj({
            "mention": {"type": "integer"},
            "action": {"type": "string", "enum": ["existing", "new", "provisional", "anonymous"]},
            "same_as": {"type": "integer"},
            "character_id": Sx(80),
            "name": Sx(80),
            "gender": {"type": "string", "enum": GENDERS},
            "age": {"type": "string", "enum": AGES},
            "accent": Sx(40),
            "profile": Sx(profile_chars),
            "voice_description": Sx(300),
        })},
        "merges": {"type": "array", "maxItems": 5,
                   "items": _obj({"from_id": Sx(80), "into_id": Sx(80), "reason": Sx(200)})},
    })


def resolve(chunk_text: str, mentions: list[dict], roster: list[str], cards: list[str],
            recent: list[str], profile_chars: int) -> tuple[list[dict], dict, dict]:
    mlines = "\n".join(f"{i}. \"{m['text']}\" ({m['kind']}) traits: {m['traits'] or '-'}" for i, m in enumerate(mentions))
    user = f"""Decide which known character each mention refers to, and update the character cards.

KNOWN CHARACTERS (compact roster, id: name (aka aliases)):
{chr(10).join(roster) or '(none yet)'}

CANDIDATE CARDS (most likely matches):
{chr(10).join(cards) or '(none)'}

RECENTLY ON STAGE: {', '.join(recent) or '(none)'}

MENTIONS IN THIS PASSAGE:
{mlines}

For each mention give one resolution:
- action "existing": it is a known character; character_id = that id.
- action "new": a new named or clearly identifiable character; character_id = ""; name = best full name or title.
- action "provisional": someone who may turn out to be a known or later-named character but you cannot tell yet; name = short description.
- action "anonymous": generic or unidentifiable (someone, a voice); not tracked.
same_as: if this mention is the same NEW character as an earlier mention in this list (e.g. "the White Rabbit" and later "the Rabbit"), give that earlier mention's number and copy its action; otherwise -1. Give the full name or title to the first of them.
Infer gender and age from every cue in the passage (e.g. "little girl" means a child; "old man" means elderly). Use "unknown" only when there is no cue.
For "new" and "provisional", fill gender, age, accent (e.g. "British", or ""), profile (who they are, at most {profile_chars} characters) and voice_description (how their voice should sound for an audiobook: pitch, age, timbre, pace, manner).
For "existing", fill profile/voice_description ONLY if this passage adds important new information; then give the full rewritten profile (at most {profile_chars} characters); otherwise "". Keep gender/age as known, or "unknown".
merges: if the passage reveals that two known ids are the same person (e.g. "the stranger was John"), list from_id -> into_id. Otherwise [].

PASSAGE:
<<<
{chunk_text}
>>>"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], \
        resolve_schema(len(mentions), profile_chars), {"task": "resolve", "mentions": mentions, "roster": roster, "recent": recent}


# ------------------------------------------------------------------ 0. classify sections

CATEGORIES = ["story", "front_matter", "toc", "captions", "notes", "boilerplate", "back_matter"]
CLASSIFY_SCHEMA = _obj({"category": {"type": "string", "enum": CATEGORIES}, "confidence": {"type": "number"},
                        "reason": Sx(200)})


def classify(title: str, opening: str, position: str) -> tuple[list[dict], dict, dict]:
    user = f"""Classify one section of a book (an EPUB) so an audiobook pipeline knows what it is.

POSITION: {position}
HEADING: {title or '(no heading)'}
OPENING TEXT (first ~100 words):
<<<
{opening}
>>>

category:
- story: the narrative itself (chapters, prologue, epilogue, letters that are part of the story)
- front_matter: title page, dedication, preface, introduction, foreword, editor's note
- toc: table of contents or a list of chapters
- captions: list of illustrations, or illustration captions / page references
- notes: footnotes, endnotes, annotations
- boilerplate: publisher or e-book licence text, copyright notices, colophons
- back_matter: appendix, index, glossary, afterword, advertisements
confidence: 0.0-1.0."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], CLASSIFY_SCHEMA, \
        {"task": "classify", "title": title, "opening": opening}


# ------------------------------------------------------------------ 3. same person?

SAME_SCHEMA = _obj({"same": {"type": "boolean"}, "confidence": {"type": "number"}, "reason": Sx(300)})


def same_person(card_a: str, card_b: str, evidence_a: str = "", evidence_b: str = "") -> tuple[list[dict], dict, dict]:
    user = f"""Are these two character cards from the same novel the same person?
- A shorter form of a title or name usually refers to the same character (the Rabbit = the White Rabbit, the Queen = the Queen of Hearts, Bennet = Mr. Bennet) unless the text shows two different individuals.
- Be careful: family members often share a surname (Mr. Bennet and Mrs. Bennet are different people).

CARD A:
{card_a}
TEXT WHERE A APPEARS:
{evidence_a or '(none)'}

CARD B:
{card_b}
TEXT WHERE B APPEARS:
{evidence_b or '(none)'}

confidence: 0.0-1.0."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], SAME_SCHEMA, \
        {"task": "same", "a": card_a, "b": card_b}


# ------------------------------------------------------------------ 4. attribute quotes

def attribute_schema(quote_ids: list[str], speaker_ids: list[str]) -> dict:
    return _obj({"quotes": {"type": "array", "maxItems": len(quote_ids), "items": _obj({
        "q": {"type": "string", "enum": quote_ids},
        "speaker": {"type": "string", "enum": speaker_ids},
        "style": Sx(40),
        "confidence": {"type": "number"},
    })}})


def attribute(marked_text: str, quote_ids: list[str], cards: list[str], speaker_ids: list[str],
              last_speakers: list[str], names: dict[str, list[str]] | None = None) -> tuple[list[dict], dict, dict]:
    user = f"""Attribute every quotation in the passage to its speaker.
Quotations are marked [Q<n>]...[/Q<n>]. One quotation may continue across sentences.

CHARACTERS IN OR NEAR THIS SCENE:
{chr(10).join(cards) or '(none)'}

LAST SPEAKERS BEFORE THIS PASSAGE (oldest first): {', '.join(last_speakers) or '(none)'}

For each quotation choose speaker from: the character codes above (C1, C2, ...), "narrator" (the quoted text is not speech, e.g. a sign, a label, a title, or a quoted word), or "unknown" (speech whose speaker cannot be determined).
A character thinking or talking to themselves is that character.
style: a short delivery note for the voice actor, 1-4 words (e.g. "anxious, hurried", "dreamy", "neutral").
confidence: 0.0-1.0.

PASSAGE:
<<<
{marked_text}
>>>"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], \
        attribute_schema(quote_ids, speaker_ids), \
        {"task": "attribute", "text": marked_text, "quote_ids": quote_ids, "speaker_ids": speaker_ids,
         "last_speakers": last_speakers, "names": names or {}}


# ------------------------------------------------------------------ v2 P1: collect (docs/DESIGN.md)

REF_KINDS = ["name", "title", "description", "relational", "address", "group", "generic"]
RELATION_TYPES = ["parent_of", "sibling_of", "spouse_of", "engaged_to", "courting", "employs", "guardian_of",
                  "friend_of"]

COLLECT_SCHEMA = _obj({
    "people": {"type": "array", "maxItems": 60, "items": _obj({
        "refs": {"type": "array", "minItems": 1, "maxItems": 8,
                 "items": _obj({"text": Sx(80), "kind": {"type": "string", "enum": REF_KINDS}})},
        "gender_cue": {"type": "string", "enum": GENDERS},
        "related": {"type": "array", "maxItems": 4, "items": _obj({"relation": Sx(40), "of_ref": Sx(80)})},
        "traits": Sx(160),
    })},
    "relations": {"type": "array", "maxItems": 20, "items": _obj({
        "a": Sx(80), "type": {"type": "string", "enum": RELATION_TYPES}, "b": Sx(80), "evidence": Sx(300)})},
})


def collect(chunk_text: str) -> tuple[list[dict], dict, dict]:
    user = f"""List the people (and other beings that act or speak) in this passage. One entry per individual.

refs: the DIFFERENT ways this passage refers to that individual, copied exactly as written.
  List each distinct wording ONCE, however often it occurs. Never list pronouns (he, she, her, you, they...).
  Skip individuals referred to only by pronouns.
  Put two references in the same entry only if the passage makes clear they are the same individual
  (e.g. "Elizabeth" and "Lizzy" when her father calls her Lizzy). If unsure, make two entries.
  kind:
  - name: a proper name, with or without a title (Elizabeth, Mr. Bennet, Lady Lucas, Lizzy)
  - title: an epithet used like a name (the Colonel, her ladyship)
  - description: describes the individual (the stranger, the young man)
  - relational: identifies them through someone else (her father, his wife, Mrs. Long's niece)
  - address: a form of address in dialogue (my dear, sir, madam)
  - group: several people together (the Bennets, the officers) -- give a group its own entry
  - generic: unidentifiable (someone, a voice)
  A group is one entry; do not list its members' pronouns as separate people.
gender_cue: what the passage shows (title, pronouns, kinship words), else "unknown".
related: this individual's relation to another reference in the passage, only if the passage states it
  (relation: a word such as father, wife, sister; of_ref: that other reference, copied exactly).
traits: one short phrase of what this passage shows about them, or "".
relations: relationships the passage STATES between two named individuals, with the exact sentence as
  evidence (copied word for word). Types: {", ".join(RELATION_TYPES)} (a is parent of b; symmetric types in any order).
Only people in the story's world: not the author, not characters of other books mentioned in passing.

PASSAGE:
<<<
{chunk_text}
>>>"""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], COLLECT_SCHEMA, \
        {"task": "collect", "text": chunk_text}


# ------------------------------------------------------------------ v2 P2: roster per name group (docs/DESIGN.md)

def roster_schema(n: int) -> dict:
    vs = [f"V{i + 1}" for i in range(n)]
    ins = [f"I{i + 1}" for i in range(n)]
    return _obj({
        "individuals": {"type": "array", "maxItems": n, "items": _obj({
            "key": {"type": "string", "enum": ins}, "name": Sx(80),
            "variants": {"type": "array", "minItems": 1, "maxItems": n, "items": {"type": "string", "enum": vs}},
            "gender": {"type": "string", "enum": GENDERS}, "reason": Sx(200)})},
        "shared": {"type": "array", "maxItems": n, "items": _obj({
            "variant": {"type": "string", "enum": vs},
            "who": {"type": "array", "minItems": 2, "maxItems": n, "items": {"type": "string", "enum": ins}}})},
        "relations": {"type": "array", "maxItems": 2 * n, "items": _obj({
            "a": {"type": "string", "enum": ins}, "type": {"type": "string", "enum": RELATION_TYPES},
            "b": {"type": "string", "enum": ins}, "evidence": Sx(300)})},
        "unsure": {"type": "array", "maxItems": n, "items": _obj({
            "variant": {"type": "string", "enum": vs}, "reason": Sx(150)})},
    })


def roster(variants: list[dict], same: list[str], hints: list[str]) -> tuple[list[dict], dict, dict]:
    """variants: [{code, text, count, genders, examples}] in first-appearance order."""
    lines = []
    for v in variants:
        g = ", ".join(f"{k} {n}" for k, n in v["genders"].items() if k != "unknown") or "none"
        ex = " | ".join(f"“{e}”" for e in v["examples"])
        lines.append(f'{v["code"]} "{v["text"]}" -- {v["count"]}x; gender cues: {g}; e.g. {ex}')
    user = f"""Below are all the ways a novel refers to a set of people who share names (usually one family or circle).
Decide how many distinct individuals they are and which reference belongs to whom.

REFERENCES (code, text, how often, gender cues, examples from the book):
{chr(10).join(lines)}

SAME INDIVIDUAL WITHIN ONE PASSAGE (from a passage-by-passage reading; may contain mistakes):
{chr(10).join(same) or '(none)'}

POSSIBLE RELATIONSHIPS NOTED IN PASSAGES (may contain mistakes; keep only what the evidence states):
{chr(10).join(hints) or '(none)'}

Rules:
- Put every reference in exactly one place: one individual's variants, "shared", or "unsure".
- "shared": a reference used for more than one individual (a family surname; "Miss Bennet" if it names
  different sisters at different times); list who (at least two individuals).
- Mr. and Mrs./Miss/Lady are never the same person. Miss and Mrs./Lady are different people unless the
  text shows that one married and changed name. Different first names are different people unless one is
  a nickname of the other (Lizzy and Eliza = Elizabeth, Kitty = Catherine).
- An individual referred to only through a shared reference is still an individual if the examples show
  they exist separately (e.g. Mrs. Bennet next to Mr. Bennet).
- name: the fullest form for that individual, written exactly as one of its references.
- relations: only family and marriage relations between these individuals that an example or a noted
  relationship states, with that sentence copied exactly as evidence. a parent_of b; others in any order.
- reason: one short sentence."""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], roster_schema(len(variants)), \
        {"task": "roster", "variants": variants}
