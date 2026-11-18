"""Kindle books (AZW3, MOBI) -> EPUB 3 with Calibre's ebook-convert (added 2026-11-18).

The pipeline reads EPUB only. A DRM-free Kindle book is converted once, before `readaloud init` (or by it), and the
converted EPUB is the book's source from then on: Calibre's output is not byte-for-byte reproducible (new
timestamps), so a conversion is never redone behind the user's back. Calibre is needed only on the machine that converts
(macOS: `brew install --cask calibre`); a GPU host without it can take the converted EPUB.

  readaloud convert data/next --done data/epub     # every AZW3/MOBI in the queue -> <same name>.epub; original to data/epub
  readaloud init book.azw3 --id x                  # converts into books/x/source.epub, keeps books/x/source.azw3
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

KINDLE = {".azw3", ".azw", ".mobi"}
APP = "/Applications/calibre.app/Contents/MacOS/ebook-convert"
# --no-default-epub-cover: no made-up cover page when the book has no cover image.
OPTIONS = ["--epub-version", "3", "--no-default-epub-cover"]


class ConvertError(Exception):
    pass


def is_kindle(path: Path) -> bool:
    return Path(path).suffix.lower() in KINDLE


def ebook_convert() -> str:
    for c in (os.environ.get("AB_EBOOK_CONVERT"), shutil.which("ebook-convert"), APP):
        if c and Path(c).is_file():
            return c
    raise ConvertError("ebook-convert not found: install Calibre (`brew install --cask calibre`); "
                       "or convert on another machine and copy the EPUB over")


def to_epub3(src: Path, dst: Path, timeout: float = 1800) -> None:
    """Convert `src` (AZW3 / MOBI) to an EPUB 3 at `dst`, written whole or not at all."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.stem + ".converting.epub")         # ebook-convert picks the format by extension
    tmp.unlink(missing_ok=True)
    try:
        r = subprocess.run([ebook_convert(), str(src), str(tmp), *OPTIONS],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        tmp.unlink(missing_ok=True)
        raise ConvertError(f"{src.name}: ebook-convert took over {timeout:.0f} s") from e
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        if "DRM" in out:
            raise ConvertError(f"{src.name} is DRM-protected; only DRM-free books can be converted")
        tail = " | ".join(ln.strip() for ln in out.strip().splitlines()[-3:])
        raise ConvertError(f"{src.name}: ebook-convert failed (exit {r.returncode}): {tail}")
    tmp.replace(dst)


def convert_queue(paths: list[Path], out: Path | None = None, done: Path | None = None, log=print) -> int:
    """Convert every Kindle book among `paths` (files, or folders searched one level deep) to <name>.epub next
    to it (or in `out`); with `done`, move each converted original there. Returns the number of failures."""
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.is_file() and is_kindle(f)) if p.is_dir() else [p]
    failed = 0
    for f in files:
        if not is_kindle(f):
            log(f"{f.name}: not AZW3 / MOBI, left alone")
            continue
        dst = (Path(out) if out else f.parent) / (f.stem + ".epub")
        if dst.exists():
            log(f"{f.name}: {dst} already exists, not converted again")
            continue
        try:
            to_epub3(f, dst)
        except ConvertError as e:
            log(f"{f.name}: {e}")
            failed += 1
            continue
        line = f"{f.name} -> {dst}"
        if done:
            Path(done).mkdir(parents=True, exist_ok=True)
            if (Path(done) / f.name).exists():
                line += f" ({Path(done) / f.name} already exists; original left in place)"
            else:
                shutil.move(str(f), str(Path(done) / f.name))
                line += f" (original moved to {done})"
        log(line)
    return failed
