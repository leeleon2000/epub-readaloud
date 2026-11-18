"""End-to-end with the mock TTS on a real EPUB 2 book (Alice).

Covers: EPUB 2 -> 3 upgrade, display-text check, epubcheck (inside package),
SMIL references resolve, clips contiguous, durations consistent, and
re-running reuses every take.
"""
import posixpath
import re
import shutil
import zipfile
from pathlib import Path

import pytest
from lxml import etree

from pipeline import config
from pipeline.book import Book
from pipeline.cli import main, run_stages

FIXTURE = Path(__file__).parent / "fixtures" / "alice_epub2.epub"
OPF = "{http://www.idpf.org/2007/opf}"
SMIL = "{http://www.w3.org/ns/SMIL}"


def clock_ms(s: str) -> int:
    h, m, rest = s.split(":")
    sec, ms = rest.split(".")
    return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(ms)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not shutil.which("epubcheck") or not shutil.which("ffmpeg"):
        pytest.skip("needs epubcheck and ffmpeg")
    root = tmp_path_factory.mktemp("books")
    main(["--books", str(root), "init", str(FIXTURE), "--id", "alice"])
    book = Book.get("alice", root)
    cfg = config.load(book.dir)
    cfg["tts"]["adapter"] = "mock"
    cfg["llm"]["adapter"] = "mock"
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000   # the mock's crude name extraction would spike
    run_stages(book, cfg, log=lambda *_: None)
    return book, cfg, next(book.out.glob("*.epub"))


def test_overlays_resolve_and_are_contiguous(built):
    _, _, out = built
    z = zipfile.ZipFile(out)
    assert z.namelist()[0] == "mimetype"
    container = etree.fromstring(z.read("META-INF/container.xml"))
    opf_path = container.find(".//{*}rootfile").get("full-path")
    opf = etree.fromstring(z.read(opf_path))
    opf_dir = posixpath.dirname(opf_path)
    assert opf.get("version") == "3.0"
    items = {i.get("id"): i for i in opf.iter(f"{OPF}item")}
    metas = {(m.get("property"), m.get("refines")): m.text for m in opf.iter(f"{OPF}meta")}
    total = 0
    n_overlays = 0
    for it in items.values():
        mo = it.get("media-overlay")
        if not mo:
            continue
        n_overlays += 1
        smil_path = posixpath.join(opf_dir, items[mo].get("href"))
        smil = etree.fromstring(z.read(smil_path))
        doc_path = posixpath.join(opf_dir, it.get("href"))
        doc_ids = set(re.findall(rb'id="([^"]+)"', z.read(doc_path)))
        prev_end = 0
        pars = list(smil.iter(f"{SMIL}par"))
        assert pars
        for par in pars:
            src = par.find(f"{SMIL}text").get("src")
            target, frag = src.split("#")
            assert posixpath.normpath(posixpath.join(posixpath.dirname(smil_path), target)) == doc_path
            assert frag.encode() in doc_ids
            au = par.find(f"{SMIL}audio")
            b, e = clock_ms(au.get("clipBegin")), clock_ms(au.get("clipEnd"))
            assert b == prev_end and e > b
            prev_end = e
            assert posixpath.normpath(posixpath.join(posixpath.dirname(smil_path), au.get("src"))) in z.namelist()
        dur = clock_ms(metas[("media:duration", "#" + mo)])
        assert dur == prev_end
        total += dur
    assert n_overlays == 13
    assert clock_ms(metas[("media:duration", None)]) == total
    assert metas[("media:active-class", None)]


def test_rerun_reuses_takes(built):
    book, cfg, _ = built
    logs = []
    run_stages(book, cfg, until="assemble", log=logs.append)
    assert any("0 to generate" in ln for ln in logs)
    assert any("0 chapters rebuilt" in ln for ln in logs)
