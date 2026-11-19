"""Markdown to HTML for the project's own documents (README.md, docs/*.md), so the UI can show them without a
dependency. It covers what those files use: headings, paragraphs, fenced code, tables, bullet and numbered lists
(nested by indentation), inline code, bold, italics, links and images. Anything else comes out as plain text.
"""
from __future__ import annotations

import html
import posixpath
import re

_FENCE = re.compile(r"^(\s*)```(.*)$")
_HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_ITEM = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
_ROW = re.compile(r"^\s*\|.*\|\s*$")
_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def slug(text: str) -> str:
    """The anchor a heading gets on a code-hosting site, so links written for it work here too."""
    t = re.sub(r"[`*_]", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", "-")


def _link(target: str, here: str, base: str = "") -> str:
    """Links between the project's documents become links inside the UI; the rest open as they are. `base` is the
    folder of the document being shown ("docs" for docs/USAGE.md): a picture is named relative to it."""
    if re.match(r"^[a-z]+:", target) or target.startswith("#"):
        return target if not target.startswith("#") else f"#/help/{here}/{target[1:]}"
    path, _, frag = target.partition("#")
    name = path.rsplit("/", 1)[-1]
    if name.lower().endswith(".md"):
        return f"#/help/{name[:-3]}" + (f"/{frag}" if frag else "")
    return "/repo/" + posixpath.normpath(posixpath.join(base, path)).lstrip("./")


def inline(text: str, here: str = "README", base: str = "") -> str:
    out, i = [], 0
    pat = re.compile(r"(`+)(.+?)\1|!\[([^\]]*)\]\(([^)\s]+)\)|\[([^\]]+)\]\(([^)\s]+)\)|\*\*(.+?)\*\*|(?<![\w*])\*([^*\s][^*]*?)\*(?!\w)")
    for m in pat.finditer(text):
        out.append(html.escape(text[i:m.start()], quote=False))
        if m.group(1):
            out.append(f"<code>{html.escape(m.group(2), quote=False)}</code>")
        elif m.group(4):
            out.append(f'<img alt="{html.escape(m.group(3))}" src="{html.escape(_link(m.group(4), here, base))}">')
        elif m.group(6):
            href = _link(m.group(6), here, base)
            ext = ' target="_blank" rel="noopener"' if re.match(r"^https?:", href) else ""
            out.append(f'<a href="{html.escape(href)}"{ext}>{inline(m.group(5), here, base)}</a>')
        elif m.group(7):
            out.append(f"<strong>{inline(m.group(7), here, base)}</strong>")
        else:
            out.append(f"<em>{inline(m.group(8), here, base)}</em>")
        i = m.end()
    out.append(html.escape(text[i:], quote=False))
    return "".join(out)


def _cells(line: str) -> list[str]:
    parts = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [p.strip().replace("\\|", "|") for p in parts]


def render(text: str, here: str = "README", base: str = "") -> tuple[str, list[dict]]:
    """(html, headings). headings: [{level, text, id}] for a table of contents. `base`: the document's folder."""
    return _render(text, lambda t: inline(t, here, base))


def _render(text: str, inl) -> tuple[str, list[dict]]:
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    heads: list[dict] = []
    para: list[str] = []
    lists: list[tuple[int, str]] = []          # open lists: (indent, tag)

    def flush_para() -> None:
        if para:
            out.append(f"<p>{inl(' '.join(s.strip() for s in para))}</p>")
            para.clear()

    def close_lists(to_indent: int = -1) -> None:
        while lists and lists[-1][0] > to_indent:
            out.append(f"</li></{lists.pop()[1]}>")

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line)
        if m:
            flush_para()
            j = i + 1
            body = []
            while j < len(lines) and not _FENCE.match(lines[j]):
                body.append(lines[j][len(m.group(1)):] if lines[j].startswith(m.group(1)) else lines[j])
                j += 1
            lang = m.group(2).strip()
            out.append(f'<pre><code class="lang-{html.escape(lang)}">' if lang else "<pre><code>")
            out.append(html.escape("\n".join(body), quote=False) + "</code></pre>")
            i = j + 1
            continue
        if not line.strip():
            flush_para()
            if lists and not (i + 1 < len(lines) and (_ITEM.match(lines[i + 1]) or lines[i + 1].startswith(" " * (lists[-1][0] + 2)))):
                close_lists()
            i += 1
            continue
        m = _HEAD.match(line)
        if m:
            flush_para()
            close_lists()
            level, title = len(m.group(1)), m.group(2)
            hid = slug(title)
            heads.append({"level": level, "text": re.sub(r"[`*]", "", title), "id": hid})
            out.append(f'<h{level} id="{html.escape(hid)}">{inl(title)}</h{level}>')
            i += 1
            continue
        if _ROW.match(line) and i + 1 < len(lines) and _SEP.match(lines[i + 1]):
            flush_para()
            close_lists()
            out.append("<table><thead><tr>" + "".join(f"<th>{inl(c)}</th>" for c in _cells(line))
                       + "</tr></thead><tbody>")
            i += 2
            while i < len(lines) and _ROW.match(lines[i]):
                out.append("<tr>" + "".join(f"<td>{inl(c)}</td>" for c in _cells(lines[i])) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        m = _ITEM.match(line)
        if m:
            flush_para()
            indent, tag = len(m.group(1)), "ol" if m.group(2)[0].isdigit() else "ul"
            if lists and indent > lists[-1][0]:
                out.append(f"<{tag}><li>")
                lists.append((indent, tag))
            else:
                close_lists(indent)
                if lists and lists[-1][0] == indent:
                    out.append("</li><li>")
                else:
                    out.append(f"<{tag}><li>")
                    lists.append((indent, tag))
            text_ = [m.group(3)]
            i += 1
            while i < len(lines) and lines[i].strip() and not _ITEM.match(lines[i]) and not _FENCE.match(lines[i]) \
                    and lines[i].startswith(" "):
                text_.append(lines[i].strip())
                i += 1
            out.append(inl(" ".join(text_)))
            continue
        if lists and line.startswith(" "):      # a paragraph that continues a list item
            para.append(line)
            i += 1
            continue
        close_lists()
        para.append(line)
        i += 1
    flush_para()
    close_lists()
    return "\n".join(out), heads
