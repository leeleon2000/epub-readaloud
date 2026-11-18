"""Print the file name an audiobook gets when it is copied out of books/<id>/out: the EPUB's own title (dc:title).

  .venv/bin/python scripts/epub_title.py book.epub        ->  Pride and Prejudice.epub

":" becomes " -" and "/" becomes "-" (macOS Finder shows ":" as "/"); other characters that trouble file systems
are dropped; the name is kept under 240 bytes.
"""
from __future__ import annotations

import re
import sys
import zipfile

from lxml import etree

DC = "{http://purl.org/dc/elements/1.1/}"


def title(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        container = etree.fromstring(z.read("META-INF/container.xml"))
        opf = container.find(".//{*}rootfile").get("full-path")
        t = etree.fromstring(z.read(opf)).find(f".//{DC}title")
    return " ".join((t.text or "").split()) if t is not None else ""


def file_name(t: str) -> str:
    t = t.replace(":", " -").replace("/", "-")
    t = re.sub(r'[\x00-\x1f"\\*?<>|]', "", t)
    t = re.sub(r"\s+", " ", t).strip(" .")
    while len(t.encode()) > 240:
        t = t[:-1].rstrip(" .")
    return (t or "untitled") + ".epub"


if __name__ == "__main__":
    print(file_name(title(sys.argv[1])))
