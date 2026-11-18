"""AZW3 / MOBI are converted to EPUB 3 with Calibre before anything else (2026-11-18).
The fixtures are the Alice EPUB converted by Calibre 9.18 (`ebook-convert`, MOBI with --mobi-file-type old)."""
import re
import stat
import zipfile
from pathlib import Path

import pytest

from pipeline import config, convert, store
from pipeline.book import Book
from pipeline.cli import main, run_stages

FIX = Path(__file__).parent / "fixtures"
KINDLE_BOOKS = [FIX / "alice_kf8.azw3", FIX / "alice_mobi6.mobi"]

try:
    convert.ebook_convert()
    HAVE_CALIBRE = True
except convert.ConvertError:
    HAVE_CALIBRE = False
needs_calibre = pytest.mark.skipif(not HAVE_CALIBRE, reason="Calibre (ebook-convert) not installed")


def _opf_version(epub: Path) -> str:
    with zipfile.ZipFile(epub) as z:
        opf = next(n for n in z.namelist() if n.endswith(".opf"))
        return re.search(rb'<package[^>]*\bversion="([^"]+)"', z.read(opf)).group(1).decode()


@needs_calibre
@pytest.mark.parametrize("src", KINDLE_BOOKS, ids=lambda p: p.suffix)
def test_init_converts_kindle_book_to_epub3(tmp_path, src):
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(src), "--id", "alice"])
    b = Book.get("alice", root)
    assert _opf_version(b.dir / "source.epub") == "3.0"
    assert (b.dir / f"source{src.suffix}").read_bytes() == src.read_bytes()   # the original is kept
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    run_stages(b, cfg, until="segment", log=lambda *_: None)                   # display check included
    con = store.connect(b.dir)
    text = " ".join(r[0] for r in con.execute("SELECT text FROM sentence"))
    assert "Down the Rabbit-Hole" in text and "White Rabbit" in text


@needs_calibre
def test_convert_queue_moves_originals(tmp_path):
    nxt, done = tmp_path / "next", tmp_path / "epub"
    nxt.mkdir()
    (nxt / "Alice.azw3").write_bytes(KINDLE_BOOKS[0].read_bytes())
    (nxt / "Other.epub").write_bytes(b"left alone")
    main(["convert", str(nxt), "--done", str(done)])
    assert _opf_version(nxt / "Alice.epub") == "3.0"
    assert (done / "Alice.azw3").exists() and not (nxt / "Alice.azw3").exists()
    assert (nxt / "Other.epub").read_bytes() == b"left alone"
    assert not list(nxt.glob("*.converting.epub"))


def test_drm_and_missing_calibre_are_clear_errors(tmp_path, monkeypatch):
    fake = tmp_path / "ebook-convert"
    fake.write_text("#!/bin/sh\necho 'calibre.ebooks.DRMError: This file is locked with DRM' >&2\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("AB_EBOOK_CONVERT", str(fake))
    with pytest.raises(convert.ConvertError, match="DRM-protected"):
        convert.to_epub3(KINDLE_BOOKS[0], tmp_path / "out.epub")
    assert not list(tmp_path.glob("*.epub"))
    with pytest.raises(SystemExit, match="DRM-protected"):
        main(["--books", str(tmp_path / "books"), "init", str(KINDLE_BOOKS[0]), "--id", "x"])
    assert not (tmp_path / "books" / "x").exists()                            # no half-made book left

    monkeypatch.setenv("AB_EBOOK_CONVERT", str(tmp_path / "missing"))
    monkeypatch.setattr(convert.shutil, "which", lambda _: None)
    monkeypatch.setattr(convert, "APP", str(tmp_path / "missing.app"))
    with pytest.raises(convert.ConvertError, match="install Calibre"):
        convert.ebook_convert()
