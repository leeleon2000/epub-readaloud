"""Split sentences into narration / quotation pieces (deterministic, no LLM).

Quote state carries across the sentences of a paragraph (a quotation may
span several sentences) and resets at each paragraph. Supported marks:
curly double “ ”, straight double ", and curly single ‘ ’ when a book uses
single quotes for dialogue (apostrophes inside words are not quote marks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[A-Za-z0-9]")


@dataclass
class Piece:
    start: int
    end: int
    text: str           # tts text of the piece, quote marks and outer whitespace removed
    is_quote: bool
    quote_id: str | None


def detect_style(texts: list[str]) -> str:
    """'double' or 'single' depending on which opening mark dominates."""
    dbl = sum(t.count("“") + t.count('"') // 2 for t in texts)
    sgl = sum(len(re.findall(r"(?:^|[\s(—–-])‘", t)) for t in texts)
    return "single" if sgl > dbl else "double"


def _marks(text: str, style: str) -> list[tuple[int, str]]:
    """Positions of quote marks as (index, 'open'|'close'|'toggle')."""
    out = []
    for i, ch in enumerate(text):
        if style == "double":
            if ch == "“":
                out.append((i, "open"))
            elif ch == "”":
                out.append((i, "close"))
            elif ch == '"':
                out.append((i, "toggle"))
        else:
            prev = text[i - 1] if i > 0 else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            if ch == "‘" and not prev.isalnum():
                out.append((i, "open"))
            elif ch == "’" and not (prev.isalnum() and nxt.isalpha()):
                # closing mark: not an in-word apostrophe (don’t, Dinah’ll)
                if prev.isalnum() and nxt.isalnum():
                    continue
                if prev.isspace():
                    continue        # elided word like ’tis
                out.append((i, "close"))
    return out


def split_paragraph(texts: list[str], style: str, para_key: str) -> list[list[Piece]]:
    """Pieces for each sentence of one paragraph, in order."""
    inside = False
    qn = 0
    result = []
    for text in texts:
        pieces: list[Piece] = []
        cur = 0
        seg_inside = inside

        def emit(a: int, b: int, q: bool):
            raw = text[a:b]
            lead = len(raw) - len(raw.lstrip())
            body = raw.strip()
            if not _WORD.search(body):
                return
            pieces.append(Piece(a + lead, a + lead + len(body), body, q, f"{para_key}q{qn}" if q else None))

        for idx, kind in _marks(text, style):
            if kind == "toggle":
                kind = "close" if inside else "open"
            if kind == "open" and not inside:
                emit(cur, idx, seg_inside)
                inside, seg_inside = True, True
                qn += 1
                cur = idx + 1
            elif kind == "close" and inside:
                emit(cur, idx, True)
                inside, seg_inside = False, False
                cur = idx + 1
        emit(cur, len(text), seg_inside)
        result.append(pieces)
    return result
