"""Display-text check (the displayed text is never altered; docs/DESIGN.md).

Verifies that a segmented document renders exactly like the original body:
ignoring our sentence spans, every character must keep the same text and the
same chain of ancestor elements (tag + attributes other than id/name), and
every non-text node (img, br, comments, script ...) must appear in the same
order with the same context. Split copies of an inline element therefore
compare equal to the original. All original id/name values must survive.
"""
from __future__ import annotations


from .segment_core import OPAQUE
from .xhtml import local


# Attributes that never affect rendering; split copies drop id/name, and the
# EPUB 2 -> 3 upgrade drops obsolete ones.
NON_RENDERING_ATTRS = {"id", "name", "summary"}


class DisplayMismatch(Exception):
    pass


def _img_size_style(el) -> dict:
    """An image's width / height attributes and style as one set of declarations: the EPUB 3 upgrade moves
    width="20%" into style (HTML5 wants an integer there), which renders the same (2026-11-17)."""
    decl = {}
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            decl[k.strip().lower()] = v.strip()
    for attr in ("width", "height"):
        v = (el.get(attr) or "").strip()
        if v:
            decl.setdefault(attr, v + "px" if v.isdigit() else v.replace(" ", ""))
    return decl


def _sig(el) -> tuple:
    if isinstance(el.tag, str) and local(el.tag) == "img":
        attrs = [(k, v) for k, v in el.attrib.items() if k not in NON_RENDERING_ATTRS | {"width", "height", "style"}]
        attrs += [("style:" + k, v) for k, v in _img_size_style(el).items()]
        return (el.tag, tuple(sorted(attrs)))
    attrs = tuple(sorted((k, v) for k, v in el.attrib.items() if k not in NON_RENDERING_ATTRS))
    return (el.tag, attrs)


def _struct(n) -> tuple:
    """Namespace-declaration-independent structure of a subtree (ids kept)."""
    if not isinstance(n.tag, str):
        return ("#", str(n.tag), n.text)
    return (n.tag, tuple(sorted(n.attrib.items())), n.text,
            tuple(_struct(c) + (c.tail,) for c in n))


def _tokens(root, is_ours) -> list[tuple]:
    toks: list[tuple] = []

    def emit_text(text: str, path: tuple):
        if not text:
            return
        if toks and toks[-1][0] == "T" and toks[-1][2] == path:
            toks[-1] = ("T", toks[-1][1] + text, path)
        else:
            toks.append(("T", text, path))

    def walk(n, path: tuple):
        if not isinstance(n.tag, str) or local(n.tag) in OPAQUE:   # comment / PI / svg ...
            s = _struct(n)
            toks.append(("N", s, path))
        else:
            ours = is_ours(n)
            p = path if ours else path + (_sig(n),)
            if not ours and len(n) == 0 and not n.text:
                toks.append(("E", _sig(n), path))
            emit_text(n.text or "", p)
            for c in n:
                walk(c, p)
                emit_text(c.tail or "", p)

    walk(root, ())
    return toks


def _ids(root) -> set[tuple[str, str]]:
    out = set()
    for e in root.iter():
        if isinstance(e.tag, str):
            for a in ("id", "name"):
                if e.get(a):
                    out.add((a, e.get(a)))
    return out


def check(original_body, segmented_body, our_ids: set[str], where: str = "") -> None:
    def is_ours(n) -> bool:
        return local(n.tag) == "span" and n.get("id") in our_ids

    a = _tokens(original_body, lambda n: False)
    b = _tokens(segmented_body, is_ours)
    if a != b:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise DisplayMismatch(f"{where}: token {i} differs\n  original:  {x!r:.300}\n  segmented: {y!r:.300}")
        raise DisplayMismatch(f"{where}: token count differs ({len(a)} vs {len(b)})")
    missing = _ids(original_body) - _ids(segmented_body)
    if missing:
        raise DisplayMismatch(f"{where}: ids lost: {sorted(missing)[:10]}")
