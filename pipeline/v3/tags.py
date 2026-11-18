"""Speech tags and the "is this dialogue" rule (docs/DESIGN.md). Rules only.

    find_tag(after, before, between) -> Tag("Mr. Darcy", "name") | Tag("she", "pronoun") |
                                        Tag("the old gentleman", "np") | Tag("I", "first_person") |
                                        Tag("they", "group") | None
    not_dialogue(quote) -> reason or None      (a quoted word, a caption, a title)

`after` is the narration right after the quote in the same paragraph, `before` the narration right before
it, `between` the narration between this quote and the previous one when both halves belong to one
sentence (“…,” said Elizabeth, “…”).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

VERBS = ("said|says|say|cried|cries|replied|replies|answered|answers|asked|asks|exclaimed|continued|added|"
         "returned|observed|whispered|shouted|called|thought|rejoined|began|resumed|interrupted|repeated|"
         "remarked|inquired|enquired|retorted|murmured|muttered|declared|insisted|protested|growled|"
         "sobbed|groaned|laughed|sighed|screamed|roared|demanded|urged|pleaded|faltered|stammered|"
         "suggested|responded|interposed|remonstrated|ejaculated|pursued|explained|admitted|agreed|"
         "assented|bawled|bellowed|blurted|chuckled|commanded|complained|confessed|cried out|gasped|grumbled|"
         "hissed|howled|implored|joined in|moaned|mumbled|ordered|panted|persisted|pleaded|rejoined|retorted|"
         "sneered|snapped|snarled|sniffed|stuttered|thundered|wailed|went on|wondered|yelled|called out|"
         "put in|broke in|struck in|cut in")
TITLE = r"(?:Mr|Mrs|Miss|Ms|Dr|Lady|Sir|Lord|Colonel|Col|Captain|Capt|Aunt|Uncle|Madame|Madam|Master|Mister|Count|Mme|Mlle|Monsieur)\.?"
CAP = r"[A-Z][a-zé’'\-]+"
NAME = rf"(?:{TITLE}\s+)?{CAP}(?:\s+(?:de\s+|van\s+|von\s+)?{CAP}){{0,2}}"
DET = r"(?:the|his|her|my|their|our|your|a|an|this|that|its)"
NP = rf"{DET}\s+[A-Za-z’'\-]+(?:\s+[A-Za-z’'\-]+){{0,3}}"
PRON = r"he|she|I|they|we"
SUBJ = rf"{NP}|{NAME}|{PRON}"
ADV = r"(?:[a-z]+ly\s+)?"
_V = rf"(?:{VERBS})"
AFTER = re.compile(rf"^[\s,;:—–\-]*(?:{_V}\s+{ADV}(?P<n1>{SUBJ})\b|(?P<n2>{SUBJ})\s+{ADV}(?:{_V})\b)")
BEFORE = re.compile(rf"(?:^|[\s,;:—–\-(])(?P<n>{SUBJ})\s+{ADV}(?:{_V})\b[^.!?“”\"]*?[,:—–\-]?\s*$")

# words that end a noun phrase captured after the verb ("said the Jew to Oliver" -> "the Jew")
NP_STOP = {"to", "with", "in", "at", "as", "and", "who", "whom", "from", "for", "on", "upon", "when", "while", "but",
           "after", "before", "into", "again", "then", "that", "which", "now", "once", "still", "here", "there",
           "softly", "quietly", "under", "over", "by", "without", "looking", "turning", "taking", "of"}
NOT_SUBJ = {"It", "There", "This", "That", "What", "Then", "Now", "And", "But", "So", "Yes", "No", "Oh", "Well",
            "When", "If", "As", "The", "A", "An", "He", "She", "They", "We", "You", "His", "Her", "My", "Our"}
PRONOUN_GENDER = {"he": "male", "she": "female"}


@dataclass(frozen=True)
class Tag:
    text: str
    kind: str          # name | pronoun | np | first_person | group

    @property
    def key(self) -> str:
        return norm_phrase(self.text)


def norm_phrase(text: str) -> str:
    """Lookup key of a name or description: lowercase, curly apostrophes straightened, his/her/my/... -> the
    for descriptions ("her mother" keeps its own key: the possessive carries no gender of the speaker)."""
    t = re.sub(r"\s+", " ", text.replace("’", "'").strip(" ,.;:!?")).lower()
    return t


def _trim_np(t: str) -> str:
    words = t.split()
    out = []
    for w in words:
        lw = w.lower()
        if out and (lw in NP_STOP or lw.endswith("ly") and len(lw) > 4 or lw.endswith("ing") and len(lw) > 5):
            break
        out.append(w)
    return " ".join(out)


def _classify(t: str) -> Tag | None:
    t = t.strip()
    low = t.lower()
    if low in PRONOUN_GENDER:
        return Tag(low, "pronoun")
    if t == "I":
        return Tag("I", "first_person")
    if low in ("they", "we"):
        return Tag(low, "group")
    first = t.split()[0]
    if first.lower() in ("the", "his", "her", "my", "their", "our", "your", "a", "an", "this", "that", "its"):
        t = _trim_np(t)
        if len(t.split()) < 2:
            return None
        return Tag(t, "np")
    words = t.split()
    while words and words[-1] in NOT_SUBJ:
        words.pop()
    while words and words[0] in NOT_SUBJ:
        words.pop(0)
    if not words:
        return None
    return Tag(" ".join(words), "name")


def find_tag(after: str, before: str, between: str = "") -> Tag | None:
    for text in (after, between):
        m = AFTER.search(text or "")
        if m:
            tag = _classify(m.group("n1") or m.group("n2"))
            if tag:
                return tag
    m = BEFORE.search(before or "")
    if m:
        return _classify(m.group("n"))
    return None


_END_PUNCT = re.compile(r"[.,!?;:—–\-…]\W*$")


def not_dialogue(text: str, whole_paragraph: bool, before_in_sentence: str, after_in_sentence: str) -> str | None:
    """Reason a quotation without a speech tag is not speech, or None.
    - caption / title: the quote is the whole paragraph and ends without punctuation
      ("He came down to see the place", the illustration captions of Gutenberg editions)
    - quoted word or title inside narration: short, with narration on both sides in the same sentence
      (she is “tolerable,” he said... has a tag and never gets here)"""
    words = len(text.split())
    if whole_paragraph and not _END_PUNCT.search(text):
        return "caption"
    has_before = bool(re.search(r"[A-Za-z]", before_in_sentence or ""))
    has_after = bool(re.search(r"[A-Za-z]", after_in_sentence or ""))
    intro = re.search(r"[,:—–]\s*$", before_in_sentence or "")
    if has_before and has_after and not intro and words <= 6:
        return "quoted phrase"
    if has_before and not intro and words <= 3 and not re.search(r"[!?]", text):
        return "quoted phrase"
    return None
