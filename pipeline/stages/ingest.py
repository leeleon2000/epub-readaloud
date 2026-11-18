"""Stage 1: unzip the source EPUB, upgrade EPUB 2 -> 3, register spine documents."""
from __future__ import annotations

import copy
import hashlib
import shutil

from .. import epub, xhtml
from ..book import Book
from ..displaycheck import check
from ..store import get_meta, set_meta


def run(book: Book, con, cfg: dict, log=print) -> None:
    digest = hashlib.sha256(book.source.read_bytes()).hexdigest()
    same_source = get_meta(con, "source_sha256") == digest
    if same_source and book.base.exists():
        log("ingest: up to date")
        return
    for d in (book.src, book.base):
        if d.exists():
            shutil.rmtree(d)
    epub.unzip(book.source, book.src)
    shutil.copytree(book.src, book.base)
    pkg = epub.open_package(book.base)
    version_in = pkg.version
    if version_in.startswith("2"):
        for n in epub.upgrade_to_epub3(pkg):
            log(f"ingest: {n}")
        pkg = epub.open_package(book.base)
    elif not version_in.startswith("3"):
        raise SystemExit(f"unsupported EPUB version {version_in}")

    # Upgrade must not change what is displayed.
    src_pkg = epub.open_package(book.src)
    for it in pkg.spine_docs():
        a = xhtml.body(xhtml.parse(src_pkg.file(it.href)))
        b = xhtml.body(xhtml.parse(pkg.file(it.href)))
        check(copy.deepcopy(a), b, set(), f"upgrade {it.href}")

    spine = [(i, it.href, it.id) for i, it in enumerate(pkg.spine_docs())]
    if same_source and [tuple(r) for r in con.execute("SELECT id, href, manifest_id FROM doc ORDER BY id")] == spine:
        # Same book, only work/ was missing (deleted, or a copied book dir): keep sentences, voices, takes.
        log(f"ingest: work files rebuilt (same source, {len(spine)} spine documents)")
        return
    con.execute("DELETE FROM sentence")
    con.execute("DELETE FROM block")
    con.execute("DELETE FROM doc")
    for i, it in enumerate(pkg.spine_docs()):
        con.execute("INSERT INTO doc(id, href, manifest_id) VALUES (?, ?, ?)", (i, it.href, it.id))
    set_meta(con, "source_sha256", digest)
    set_meta(con, "epub_version_in", version_in)
    set_meta(con, "opf_path", pkg.opf_path)
    set_meta(con, "title", pkg.title())
    con.commit()
    log(f"ingest: EPUB {version_in}, {len(pkg.spine_docs())} spine documents, title {pkg.title()!r}")
