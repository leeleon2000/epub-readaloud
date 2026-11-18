"""EPUB package access and EPUB 2 -> 3 upgrade."""
from __future__ import annotations

import datetime as dt
import posixpath
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from . import xhtml
from .xhtml import CONTAINER_NS, DC_NS, NCX_NS, OPF_NS, XHTML_NS

XHTML_TYPES = {"application/xhtml+xml"}
OPF_ATTR_PREFIX = f"{{{OPF_NS}}}"


@dataclass
class Item:
    id: str
    href: str           # relative to OPF dir, URL-decoded
    media_type: str
    properties: set[str] = field(default_factory=set)


@dataclass
class Package:
    root: Path                  # unzipped EPUB root
    opf_path: str               # relative to root (posix)
    tree: etree._ElementTree

    @property
    def opf_dir(self) -> str:
        return posixpath.dirname(self.opf_path)

    @property
    def version(self) -> str:
        return self.tree.getroot().get("version", "2.0")

    def file(self, href: str) -> Path:
        return self.root / posixpath.join(self.opf_dir, href)

    @property
    def manifest(self):
        return self.tree.getroot().find(f"{{{OPF_NS}}}manifest")

    @property
    def metadata(self):
        return self.tree.getroot().find(f"{{{OPF_NS}}}metadata")

    @property
    def spine(self):
        return self.tree.getroot().find(f"{{{OPF_NS}}}spine")

    def items(self) -> dict[str, Item]:
        out = {}
        for el in self.manifest.findall(f"{{{OPF_NS}}}item"):
            from urllib.parse import unquote
            out[el.get("id")] = Item(el.get("id"), unquote(el.get("href")), el.get("media-type"),
                                     set((el.get("properties") or "").split()))
        return out

    def item_el(self, item_id: str):
        for el in self.manifest.findall(f"{{{OPF_NS}}}item"):
            if el.get("id") == item_id:
                return el
        raise KeyError(item_id)

    def spine_docs(self) -> list[Item]:
        items = self.items()
        docs = []
        for ref in self.spine.findall(f"{{{OPF_NS}}}itemref"):
            it = items.get(ref.get("idref"))
            if it and it.media_type in XHTML_TYPES:
                docs.append(it)
        return docs

    def title(self) -> str:
        t = self.metadata.find(f"{{{DC_NS}}}title")
        return (t.text or "").strip() if t is not None else "Untitled"

    def save(self) -> None:
        self.file_opf().write_bytes(etree.tostring(self.tree, xml_declaration=True, encoding="utf-8"))

    def file_opf(self) -> Path:
        return self.root / self.opf_path

    def new_id(self, base: str) -> str:
        used = {e.get("id") for e in self.tree.iter() if isinstance(e.tag, str) and e.get("id")}
        i, cand = 1, base
        while cand in used:
            i += 1
            cand = f"{base}-{i}"
        return cand


def unzip(epub: Path, dest: Path) -> None:
    with zipfile.ZipFile(epub) as z:
        for info in z.infolist():
            p = (dest / info.filename).resolve()
            if not str(p).startswith(str(dest.resolve())):
                raise ValueError(f"unsafe path in EPUB: {info.filename}")
        z.extractall(dest)


def open_package(root: Path) -> Package:
    c = etree.parse(str(root / "META-INF" / "container.xml"))
    rf = c.find(f".//{{{CONTAINER_NS}}}rootfile")
    opf_path = rf.get("full-path")
    tree = etree.parse(str(root / opf_path), etree.XMLParser(remove_blank_text=False))
    return Package(root, opf_path, tree)


def set_modified(pkg: Package) -> None:
    md = pkg.metadata
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for m in md.findall(f"{{{OPF_NS}}}meta"):
        if m.get("property") == "dcterms:modified":
            m.text = now
            return
    m = etree.SubElement(md, f"{{{OPF_NS}}}meta", property="dcterms:modified")
    m.text = now


# --------------------------------------------------------------------------- EPUB 2 -> 3

def upgrade_to_epub3(pkg: Package) -> list[str]:
    """Upgrade OPF metadata, add a nav document, convert content docs to HTML5 doctype.

    Returns notes describing what was changed.
    """
    notes = []
    root = pkg.tree.getroot()
    root.set("version", "3.0")
    md = pkg.metadata
    for t in root.findall(f"{{{OPF_NS}}}tours"):          # EPUB 2 only (deprecated in 2.0.1); not allowed in 3
        root.remove(t)
        notes.append("removed EPUB 2 <tours>")
    for el in list(md):                                    # <dc:subject/>: EPUB 3 wants text in every dc element
        if isinstance(el.tag, str) and el.tag.startswith(f"{{{DC_NS}}}") and not (el.text or "").strip() and not len(el):
            md.remove(el)
            notes.append(f"removed empty {el.tag.split('}')[1]}")

    # opf:role / opf:file-as / opf:scheme / opf:event attributes are EPUB 2 only.
    n_creator = 0
    for el in list(md):
        if not isinstance(el.tag, str):
            continue
        role = el.attrib.pop(OPF_ATTR_PREFIX + "role", None)
        file_as = el.attrib.pop(OPF_ATTR_PREFIX + "file-as", None)
        if el.tag in (f"{{{DC_NS}}}creator", f"{{{DC_NS}}}contributor"):
            # Some EPUB 2 books write role / file-as without the opf: prefix (invalid in both versions;
            # seen in a commercial EPUB, 2026-11-15). The intent is clear: convert them the same way.
            role = role or el.attrib.pop("role", None)
            file_as = file_as or el.attrib.pop("file-as", None)
        el.attrib.pop(OPF_ATTR_PREFIX + "scheme", None)
        el.attrib.pop(OPF_ATTR_PREFIX + "event", None)
        el.attrib.pop("{http://www.w3.org/2001/XMLSchema-instance}type", None)
        if (role or file_as) and el.tag in (f"{{{DC_NS}}}creator", f"{{{DC_NS}}}contributor"):
            if not el.get("id"):
                n_creator += 1
                el.set("id", pkg.new_id(f"creator{n_creator}"))
            idx = list(md).index(el)
            extra = []
            if role:
                m = etree.Element(f"{{{OPF_NS}}}meta", refines="#" + el.get("id"), property="role",
                                  scheme="marc:relators")
                m.text = role
                extra.append(m)
            if file_as:
                m = etree.Element(f"{{{OPF_NS}}}meta", refines="#" + el.get("id"), property="file-as")
                m.text = file_as
                extra.append(m)
            for k, m in enumerate(extra):
                md.insert(idx + 1 + k, m)
    dates = md.findall(f"{{{DC_NS}}}date")
    for d in dates[1:]:
        md.remove(d)
    if len(dates) > 1:
        notes.append(f"kept 1 of {len(dates)} dc:date")
    # dc:date must be W3C-DTF in EPUB 3; keep only if it looks like one.
    if dates and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2}(T[\d:.]+(Z|[+-]\d{2}:\d{2})?)?)?)?", (dates[0].text or "").strip()):
        md.remove(dates[0])
        notes.append("dropped non-W3CDTF dc:date")
    set_modified(pkg)

    items = pkg.items()
    # cover-image property from <meta name="cover">
    for m in md.findall(f"{{{OPF_NS}}}meta"):
        if m.get("name") == "cover" and m.get("content") in items:
            el = pkg.item_el(m.get("content"))
            if items[m.get("content")].media_type.startswith("image/"):
                _add_prop(el, "cover-image")

    # nav document generated from the NCX
    toc_id = pkg.spine.get("toc")
    ncx = items.get(toc_id) if toc_id else None
    if ncx is None:
        ncx = next((i for i in items.values() if i.media_type == "application/x-dtbncx+xml"), None)
    nav_href = _unique_href(pkg, "ab-nav.xhtml")
    pkg.file(nav_href).write_bytes(_nav_from_ncx(pkg, ncx, nav_href))
    etree.SubElement(pkg.manifest, f"{{{OPF_NS}}}item", id=pkg.new_id("ab-nav"), href=nav_href,
                     attrib={"media-type": "application/xhtml+xml", "properties": "nav"})
    notes.append(f"generated {nav_href} from NCX" if ncx else f"generated empty-ish {nav_href} (no NCX)")

    # content documents: HTML5 doctype, manifest properties
    for it in pkg.items().values():
        if it.media_type not in XHTML_TYPES or "nav" in it.properties:
            continue
        path = pkg.file(it.href)
        tree = xhtml.parse(path)
        _html5_fixups(tree)
        xhtml.write(tree, path, doctype=xhtml.HTML5_DOCTYPE)
        props = content_properties(tree)
        if props:
            el = pkg.item_el(it.id)
            for p in props:
                _add_prop(el, p)
    notes.append("content docs: HTML5 doctype")
    pkg.save()
    return notes


def content_properties(tree) -> set[str]:
    props = set()
    for e in tree.iter():
        if not isinstance(e.tag, str):
            continue
        t = xhtml.local(e.tag)
        if e.tag.startswith("{http://www.w3.org/2000/svg}"):
            props.add("svg")
        elif e.tag.startswith("{http://www.w3.org/1998/Math/MathML}"):
            props.add("mathml")
        elif t == "script" and e.tag.startswith(f"{{{XHTML_NS}}}"):
            props.add("scripted")
    return props


def _html5_fixups(tree) -> None:
    """Remove EPUB 2 / XHTML 1.1 constructs that are invalid in EPUB 3 HTML5.
    None of these affect displayed text (see displaycheck.NON_RENDERING_ATTRS)."""
    for t in tree.iter(f"{{{XHTML_NS}}}table"):
        t.attrib.pop("summary", None)
    # <img width="20%"> / height="300px": HTML5 wants a plain integer. Pixel values lose the unit; anything else
    # moves into style. An image holds no text, so the display check is not affected (All We Have Is Now,
    # 88 epubcheck errors, 2026-11-17).
    for img in tree.iter(f"{{{XHTML_NS}}}img"):
        moved = []
        in_style = {d.split(":", 1)[0].strip().lower() for d in (img.get("style") or "").split(";") if ":" in d}
        for attr in ("width", "height"):
            v = (img.get(attr) or "").strip()
            if not v or v.isdigit():
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?\s*px", v):
                img.set(attr, str(round(float(v[:-2].strip()))))
                continue
            if attr not in in_style:                       # a width in style already wins over the attribute
                moved.append(f"{attr}:{v}")
            del img.attrib[attr]
        if moved:
            style = (img.get("style") or "").strip().rstrip(";")
            img.set("style", "; ".join([x for x in [style] if x] + moved))
    head = tree.getroot().find(f"{{{XHTML_NS}}}head")
    if head is not None:
        for m in head.findall(f"{{{XHTML_NS}}}meta"):
            if (m.get("http-equiv") or "").lower() in ("content-style-type", "content-script-type"):
                head.remove(m)
            elif (m.get("http-equiv") or "").lower() == "content-type":
                # HTML5 allows only this value; the files are written as UTF-8 (one book had "UTF-8",
                # All We Have Is Now "http://www.w3.org/1999/xhtml; charset=utf-8")
                m.set("content", "text/html; charset=utf-8")
        for ln in head.findall(f"{{{XHTML_NS}}}link"):
            # Gutenberg's rel="icon" type="image/x-cover" link is not valid HTML5
            if ln.get("type") == "image/x-cover":
                head.remove(ln)


def _add_prop(el, prop: str) -> None:
    props = set((el.get("properties") or "").split())
    props.add(prop)
    el.set("properties", " ".join(sorted(props)))


def _unique_href(pkg: Package, href: str) -> str:
    existing = {i.href for i in pkg.items().values()}
    stem, ext = posixpath.splitext(href)
    k, cand = 1, href
    while cand in existing or pkg.file(cand).exists():
        k += 1
        cand = f"{stem}-{k}{ext}"
    return cand


def _nav_from_ncx(pkg: Package, ncx: Item | None, nav_href: str) -> bytes:
    def rel(target: str) -> str:
        # NCX hrefs are relative to the NCX file; nav sits in the OPF dir.
        base = posixpath.dirname(ncx.href) if ncx else ""
        return posixpath.normpath(posixpath.join(base, target)) if base else target

    def esc(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def points(parent) -> str:
        lis = []
        for np_ in parent.findall(f"{{{NCX_NS}}}navPoint"):
            label = " ".join("".join(np_.find(f"{{{NCX_NS}}}navLabel").itertext()).split()) \
                if np_.find(f"{{{NCX_NS}}}navLabel") is not None else ""
            content = np_.find(f"{{{NCX_NS}}}content")
            href = rel(content.get("src")) if content is not None else ""
            sub = points(np_)
            lis.append(f'<li><a href="{esc(href)}">{esc(label) or "Untitled"}</a>{sub}</li>')
        return f"<ol>{''.join(lis)}</ol>" if lis else ""

    ol = ""
    if ncx is not None:
        t = etree.parse(str(pkg.file(ncx.href)))
        nm = t.getroot().find(f"{{{NCX_NS}}}navMap")
        if nm is not None:
            ol = points(nm)
    if not ol:
        first = pkg.spine_docs()[0].href if pkg.spine_docs() else ""
        ol = f'<ol><li><a href="{esc(first)}">Start</a></li></ol>'
    return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
            f'<html xmlns="{XHTML_NS}" xmlns:epub="http://www.idpf.org/2007/ops">'
            f'<head><title>{esc(pkg.title())}</title></head><body>'
            f'<nav epub:type="toc" id="toc"><h1>Contents</h1>{ol}</nav></body></html>\n').encode()


def zip_epub(src_root: Path, out: Path, stored_exts=(".opus", ".mp3", ".m4a", ".ogg")) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for p in sorted(src_root.rglob("*")):
            # ._name: AppleDouble files macOS tar adds when a book dir is copied to Linux (2026-11-16)
            if p.is_dir() or p.name == "mimetype" and p.parent == src_root or p.name == ".DS_Store" \
                    or p.name.startswith("._"):
                continue
            arc = p.relative_to(src_root).as_posix()
            ct = zipfile.ZIP_STORED if p.suffix.lower() in stored_exts else zipfile.ZIP_DEFLATED
            z.write(p, arc, compress_type=ct)
