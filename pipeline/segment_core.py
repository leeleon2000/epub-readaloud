"""Sentence segmentation of one XHTML document (F1).

The only change made to the document is wrapping each sentence in
<span id="...">. When a sentence boundary falls inside an inline element
(e.g. `<em>end. Start</em>`), that element is split into two copies with the
same tag and attributes, so the rendering is unchanged; only the first copy
keeps `id`/`name`. `displaycheck.py` verifies this on every run.

Model of the document:
- A *block* element (p, h1, li, div, ...) is a container. Its inline content
  between block children forms a *run* (one paragraph for pause purposes).
- Characters are counted over displayed text only. Comments, PIs and opaque
  elements (script, style, svg, math) count as zero length and are moved
  whole.
- A zero-length node at offset p belongs to the slice [a, b) with a <= p < b;
  at the very end of a run it belongs to the trailing (unwrapped) slice.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

import pysbd
from lxml import etree

from .xhtml import EPUB_NS, XHTML_NS, local

BLOCK = {
    "address", "article", "aside", "blockquote", "body", "caption", "center", "dd", "details",
    "dialog", "div", "dl", "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1",
    "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "hr", "legend", "li", "main", "nav",
    "ol", "p", "pre", "section", "summary", "table", "tbody", "td", "tfoot", "th", "thead",
    "tr", "ul", "colgroup", "col", "caption",
}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# An illustration caption: text inside <figcaption>, or inside an element whose class has a "caption" token
# (Project Gutenberg writes <div class="caption">). It is displayed, never read aloud and never searched for
# dialogue: captions of an illustrated edition quote lines of the story and carry credits such as "[Copyright ...]".
_CAPTION_CLASS = re.compile(r"(?:^|[-_])caption(?:$|[-_])|figcaption", re.I)


def in_caption(el) -> bool:
    while el is not None and isinstance(el.tag, str):
        if local(el.tag) == "figcaption" or any(_CAPTION_CLASS.search(c) for c in (el.get("class") or "").split()):
            return True
        el = el.getparent()
    return False
# Displayed but zero-length for offsets and moved whole (never split or wrapped).
OPAQUE = {"script", "style", "svg", "math", "template", "noscript", "object", "video",
          "audio", "iframe", "canvas"}
# Displayed, wrapped with their sentence, but not sent to TTS.
SILENT_INLINE = {"rt", "rp"}
# Not traversed at all (left untouched, never spoken).
SKIP_EPUB_TYPES = {"footnote", "endnote", "rearnote", "note", "pagebreak", "toc", "landmarks",
                   "page-list"}
SKIP_ROLES = {"doc-footnote", "doc-endnote", "doc-pagebreak", "doc-toc"}
NOTEREF_TYPES = {"noteref"}

EPUB_TYPE = f"{{{EPUB_NS}}}type"
SPAN = f"{{{XHTML_NS}}}span"
_WORDCHAR = re.compile(r"[A-Za-z0-9]")
_WS = re.compile(r"\s+")

_segmenter = pysbd.Segmenter(language="en", clean=False)


@dataclass
class SentenceInfo:
    sid: str
    para: int            # run index within the document
    kind: str            # 'heading' | 'body' | 'caption' (illustration caption: never read)
    text: str            # displayed text, whitespace collapsed
    tts_text: str


@dataclass
class SegmentResult:
    sentences: list[SentenceInfo] = field(default_factory=list)


# --------------------------------------------------------------------------- node classification

def _is_text_node(n) -> bool:
    return isinstance(n.tag, str)


def _epub_types(el) -> set[str]:
    return set((el.get(EPUB_TYPE) or "").split())


def is_skipped(el) -> bool:
    if not _is_text_node(el):
        return False
    t = local(el.tag)
    if t in ("head",) or t in OPAQUE:
        return True
    if _epub_types(el) & SKIP_EPUB_TYPES:
        return True
    if set((el.get("role") or "").split()) & SKIP_ROLES:
        return True
    return False


def is_opaque(n) -> bool:
    return (not _is_text_node(n)) or local(n.tag) in OPAQUE


def _has_block_descendant(el) -> bool:
    for d in el.iterdescendants():
        if _is_text_node(d) and local(d.tag) in BLOCK:
            return True
    return False


def is_blockish(n) -> bool:
    if not _is_text_node(n) or local(n.tag) in OPAQUE:
        return False
    return local(n.tag) in BLOCK or _has_block_descendant(n)


def content_len(n) -> int:
    """Displayed character count inside n (excluding its tail)."""
    if is_opaque(n):
        return 0
    total = len(n.text or "")
    for c in n:
        total += content_len(c) + len(c.tail or "")
    return total


def content_text(n) -> str:
    if is_opaque(n):
        return ""
    parts = [n.text or ""]
    for c in n:
        parts.append(content_text(c))
        parts.append(c.tail or "")
    return "".join(parts)


# --------------------------------------------------------------------------- slicing

def _sub(s: str | None, start: int, a: int, b: int) -> str:
    if not s:
        return ""
    lo = max(a - start, 0)
    hi = min(b - start, len(s))
    return s[lo:hi] if hi > lo else ""


def _zero_in(p: int, a: int, b: int, last: bool) -> bool:
    return a <= p < b or (last and p == b)


def _clone_shallow(el, keep_ids: bool):
    if not _is_text_node(el):
        return copy.copy(el)
    new = etree.Element(el.tag, attrib=dict(el.attrib), nsmap=None)
    if not keep_ids:
        new.attrib.pop("id", None)
        new.attrib.pop("name", None)
    return new


def _slice(lead: str | None, children, base: int, a: int, b: int, last: bool):
    """Copy of (lead text, children with tails) restricted to [a, b).

    Returns (text, [nodes]) where each node's .tail is already set.
    """
    out_text = _sub(lead, base, a, b)
    pos = base + len(lead or "")
    nodes: list = []

    def add_text(t: str):
        nonlocal out_text
        if not t:
            return
        if nodes:
            nodes[-1].tail = (nodes[-1].tail or "") + t
        else:
            out_text += t

    for child in children:
        L = content_len(child)
        cs, ce = pos, pos + L
        node = None
        if L == 0:
            if _zero_in(cs, a, b, last):
                node = copy.deepcopy(child)
                node.tail = None
        elif cs >= a and ce <= b:
            node = copy.deepcopy(child)
            node.tail = None
        elif b > a and cs < b and ce > a:
            node = _clone_shallow(child, keep_ids=(a <= cs < b))
            t, kids = _slice(child.text, list(child), cs, a, b, last)
            node.text = t or None
            for k in kids:
                node.append(k)
        if node is not None:
            nodes.append(node)
        pos = ce
        tail = child.tail or ""
        add_text(_sub(tail, pos, a, b))
        pos += len(tail)
    return out_text, nodes


# --------------------------------------------------------------------------- runs

@dataclass
class Run:
    """Inline content of `parent` between two block children.

    Bounds are element references (not indices) so that rewriting one run
    does not invalidate the others in the same parent.
    """
    parent: object
    lead_owner: object      # parent (lead = parent.text) or the block child whose .tail leads
    stop: object | None     # block child that ends the run, or None for end of parent

    @property
    def lead(self) -> str:
        return (self.lead_owner.text if self.lead_owner is self.parent else self.lead_owner.tail) or ""

    @property
    def start(self) -> int:
        return 0 if self.lead_owner is self.parent else self.parent.index(self.lead_owner) + 1

    def children(self):
        kids = list(self.parent)
        end = len(kids) if self.stop is None else self.parent.index(self.stop)
        return kids[self.start:end]

    def text(self) -> str:
        parts = [self.lead]
        for c in self.children():
            parts.append(content_text(c))
            parts.append(c.tail or "")
        return "".join(parts)


def collect_runs(container, out: list[Run]) -> None:
    if is_skipped(container):
        return
    lead_owner = container
    for c in list(container):
        if is_blockish(c) and not is_skipped(c):
            out.append(Run(container, lead_owner, c))
            collect_runs(c, out)
            lead_owner = c
        elif _is_text_node(c) and is_skipped(c) and local(c.tag) not in OPAQUE:
            # Skipped non-opaque element (e.g. aside footnote): acts as a run boundary.
            out.append(Run(container, lead_owner, c))
            lead_owner = c
    out.append(Run(container, lead_owner, None))


def _replace_run(run: Run, lead: str, nodes: list) -> None:
    parent = run.parent
    start = run.start
    for c in run.children():
        parent.remove(c)            # lxml removes the tail with the element
    if run.lead_owner is parent:
        parent.text = lead or None
    else:
        run.lead_owner.tail = lead or None
    for k, n in enumerate(nodes):
        parent.insert(start + k, n)


# --------------------------------------------------------------------------- sentence splitting

def split_sentences(raw: str) -> list[tuple[int, int]]:
    """Sentence [start, end) ranges in raw, trimmed of surrounding whitespace."""
    collapsed_chars, idx_map = [], []
    prev_space = True
    for i, ch in enumerate(raw):
        if ch.isspace():
            if not prev_space:
                collapsed_chars.append(" ")
                idx_map.append(i)
            prev_space = True
        else:
            collapsed_chars.append(ch)
            idx_map.append(i)
            prev_space = False
    collapsed = "".join(collapsed_chars)
    if not collapsed.strip():
        return []
    segs = _segmenter.segment(collapsed)
    ranges, cur = [], 0
    for seg in segs:
        s = seg.strip()
        if not s:
            continue
        k = collapsed.find(s, cur)
        if k < 0:       # segmenter altered the text; fall back to one sentence for the rest
            rest = collapsed[cur:].strip()
            k = collapsed.find(rest, cur)
            ranges.append((k, k + len(rest)))
            cur = len(collapsed)
            break
        ranges.append((k, k + len(s)))
        cur = k + len(s)
    out = []
    for cs, ce in ranges:
        out.append((idx_map[cs], idx_map[ce - 1] + 1))
    return out


# --------------------------------------------------------------------------- TTS text

def tts_text_of(span) -> str:
    parts: list[str] = []

    def walk(n, speak: bool):
        if not _is_text_node(n):
            return
        t = local(n.tag)
        silent = (t in SILENT_INLINE or t in OPAQUE or bool(_epub_types(n) & NOTEREF_TYPES)
                  or "doc-noteref" in (n.get("role") or ""))
        sp = speak and not silent
        if t == "br":
            parts.append(" ")
        if sp and n.text:
            parts.append(n.text)
        for c in n:
            walk(c, sp)
            if speak and c.tail:
                parts.append(c.tail)

    walk(span, True)
    return _WS.sub(" ", "".join(parts)).strip()


# --------------------------------------------------------------------------- main entry

def segment_document(tree, id_prefix: str) -> SegmentResult:
    """Wrap sentences of the document in place. Returns sentence metadata in reading order."""
    from .xhtml import body
    root_body = body(tree)
    existing = {e.get("id") for e in tree.iter() if _is_text_node(e) and e.get("id")}
    runs: list[Run] = []
    collect_runs(root_body, runs)

    result = SegmentResult()
    counter = 0
    para = 0
    for run in runs:
        raw = run.text()
        if not _WORDCHAR.search(raw):
            continue
        ranges = [(a, b) for a, b in split_sentences(raw) if _WORDCHAR.search(raw[a:b])]
        if not ranges:
            continue
        total = len(raw)
        lead, kids = run.lead, run.children()
        pieces: list[tuple[str | None, int, int]] = []   # (sid or None for gap, a, b)
        cur = 0
        for a, b in ranges:
            pieces.append((None, cur, a))
            counter += 1
            sid = f"{id_prefix}{counter:05d}"
            while sid in existing:
                sid += "x"
            existing.add(sid)
            pieces.append((sid, a, b))
            cur = b
        pieces.append((None, cur, total))

        new_lead, new_nodes, spans = "", [], []
        for i, (sid, a, b) in enumerate(pieces):
            last = i == len(pieces) - 1
            t, nodes = _slice(lead, kids, 0, a, b, last)
            if sid is None:
                if new_nodes:
                    new_nodes[-1].tail = (new_nodes[-1].tail or "") + t
                else:
                    new_lead += t
                new_nodes.extend(nodes)
            else:
                span = etree.Element(SPAN, id=sid)
                span.text = t or None
                for n in nodes:
                    span.append(n)
                new_nodes.append(span)
                spans.append((sid, span))
        # Clean empty tails
        for n in new_nodes:
            if n.tail == "":
                n.tail = None
        _replace_run(run, new_lead, new_nodes)

        kind = "caption" if in_caption(run.parent) else "heading" if local(run.parent.tag) in HEADINGS else "body"
        for sid, span in spans:
            disp = _WS.sub(" ", content_text(span)).strip()
            tts = tts_text_of(span)
            if not _WORDCHAR.search(tts):
                # Nothing to say (e.g. only a footnote marker): unwrap is not needed; keep span,
                # but don't register it as a sentence. Unwrap to avoid orphan ids.
                _unwrap(span)
                continue
            result.sentences.append(SentenceInfo(sid, para, kind, disp, tts))
        para += 1
    return result


def _unwrap(span) -> None:
    parent = span.getparent()
    idx = parent.index(span)
    text = span.text or ""
    if idx == 0:
        parent.text = (parent.text or "") + text
    else:
        prev = parent[idx - 1]
        prev.tail = (prev.tail or "") + text
    kids = list(span)
    for k, c in enumerate(kids):
        parent.insert(idx + k, c)
    last_tail = span.tail or ""
    if kids:
        kids[-1].tail = (kids[-1].tail or "") + last_tail
    else:
        if idx == 0:
            parent.text = (parent.text or "") + last_tail
        else:
            prev = parent[idx - 1]
            prev.tail = (prev.tail or "") + last_tail
    parent.remove(span) if span.getparent() is not None else None
