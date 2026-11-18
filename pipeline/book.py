"""Book directory layout (books/<id>/)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import REPO

BOOKS = REPO / "books"


@dataclass
class Book:
    id: str
    dir: Path

    @classmethod
    def get(cls, book_id: str, books_root: Path | None = None) -> "Book":
        d = (books_root or BOOKS) / book_id
        if not (d / "source.epub").exists():
            raise SystemExit(f"no such book: {d} (run `readaloud init <file.epub>` first)")
        return cls(book_id, d)

    source = property(lambda s: s.dir / "source.epub")
    src = property(lambda s: s.dir / "work" / "src")        # pristine unzip
    base = property(lambda s: s.dir / "work" / "base")      # EPUB 3, not segmented
    seg = property(lambda s: s.dir / "work" / "seg")        # segmented content docs only
    build = property(lambda s: s.dir / "work" / "build")    # tree that gets zipped
    chapters = property(lambda s: s.dir / "work" / "chapters")
    takes = property(lambda s: s.dir / "takes")
    reports = property(lambda s: s.dir / "reports")
    out = property(lambda s: s.dir / "out")


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()
    return s[:60] or "book"
