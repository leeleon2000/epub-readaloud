"""XHTML parsing/serialization shared by all stages.

Content documents are parsed as XML (EPUB requires XHTML). Named HTML
entities (&nbsp; etc.) are rewritten to numeric references before parsing,
because there is no DTD to resolve them and EPUB 3 forbids them anyway; this
changes bytes, not displayed text.
"""
from __future__ import annotations

import html.entities
import re
from pathlib import Path

from lxml import etree

XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
SMIL_NS = "http://www.w3.org/ns/SMIL"

HTML5_DOCTYPE = "<!DOCTYPE html>"

_XML_ENTITIES = {"amp", "lt", "gt", "quot", "apos"}
_ENTITY_RE = re.compile(r"&([A-Za-z][A-Za-z0-9]*);")
_ENC_RE = re.compile(rb"""^<\?xml[^>]*encoding=["']([A-Za-z0-9_.-]+)["']""")


def _fix_entities(text: str) -> str:
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name in _XML_ENTITIES or name not in html.entities.name2codepoint:
            return m.group(0)
        return f"&#{html.entities.name2codepoint[name]};"
    return _ENTITY_RE.sub(sub, text)


def _parser() -> etree.XMLParser:
    return etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True,
                           huge_tree=True, remove_blank_text=False)


def parse_bytes(data: bytes) -> etree._ElementTree:
    m = _ENC_RE.match(data)
    enc = m.group(1).decode() if m else "utf-8"
    text = data.decode(enc)
    text = _fix_entities(text)
    # Re-encode as UTF-8 and drop the original declaration's encoding claim.
    text = re.sub(r"^<\?xml[^>]*\?>", "", text, count=1).lstrip("﻿")
    return etree.ElementTree(etree.fromstring(text.encode("utf-8"), _parser()))


def parse(path: Path) -> etree._ElementTree:
    return parse_bytes(Path(path).read_bytes())


def serialize(tree: etree._ElementTree, doctype: str | None = None) -> bytes:
    """UTF-8 with XML declaration. doctype=None keeps the document's own."""
    if doctype is None:
        doctype = tree.docinfo.doctype or None
    return etree.tostring(tree, xml_declaration=True, encoding="utf-8",
                          doctype=doctype or None)


def write(tree: etree._ElementTree, path: Path, doctype: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize(tree, doctype))


def local(tag) -> str:
    """Local name of an element tag; '' for comments/PIs."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def body(tree: etree._ElementTree):
    b = tree.getroot().find(f"{{{XHTML_NS}}}body")
    if b is None:
        raise ValueError("document has no XHTML <body>")
    return b
