"""Stage 10: encode audio, write SMIL, patch OPF, zip, run epubcheck (F11)."""
from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lxml import etree

from .. import epub, xhtml
from ..book import Book, slugify
from ..xhtml import OPF_NS, XHTML_NS

ACTIVE_CLASS = "ab-active"
PLAYING_CLASS = "ab-playing"
CSS = f""".{ACTIVE_CLASS} {{ background-color: rgba(255, 213, 79, 0.45); color: inherit; }}
"""
FORMATS = {
    "opus": {"ext": "opus", "mime": "audio/ogg; codecs=opus", "args": ["-c:a", "libopus"]},
    "mp3cbr": {"ext": "mp3", "mime": "audio/mpeg", "args": ["-c:a", "libmp3lame"]},
    # AAC-LC in MP4: the sample table gives exact seeking in WebKit; faststart puts the index first.
    "aac": {"ext": "m4a", "mime": "audio/mp4", "args": ["-c:a", "aac", "-movflags", "+faststart"]},
}


def clock(msec: int) -> str:
    h, rem = divmod(msec, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{frac:03d}"


def _rel(from_file: str, to_file: str) -> str:
    return posixpath.relpath(to_file, posixpath.dirname(from_file) or ".")


def encode(wav: Path, out: Path, fmt: str, bitrate: str) -> None:
    f = FORMATS[fmt]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-ac", "1",
           "-map_metadata", "-1", *f["args"], "-b:a", bitrate, str(out)]
    subprocess.run(cmd, check=True)


def probe_ms(path: Path) -> int:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=True)
    return int(round(float(r.stdout.strip()) * 1000))


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None,
        suffix: str = "") -> Path:
    """Package the book. doc_ids limits which documents get audio (others stay text-only);
    suffix names a separate build dir and output file, e.g. for previews."""
    ocfg = cfg["output"]
    if ocfg.get("granularity", "sentence") != "sentence":
        raise SystemExit("package: only sentence granularity is implemented (word-level is M6)")
    fmt, bitrate = ocfg["format"], ocfg["bitrate"]
    build = book.build.with_name(book.build.name + suffix)
    if build.exists():
        shutil.rmtree(build)
    shutil.copytree(book.base, build)
    pkg = epub.open_package(build)
    opf_dir = pkg.opf_dir

    css_href = "ab/overlay.css"
    pkg.file(css_href).parent.mkdir(parents=True, exist_ok=True)
    pkg.file(css_href).write_text(CSS)
    etree.SubElement(pkg.manifest, f"{{{OPF_NS}}}item", id=pkg.new_id("ab-css"), href=css_href,
                     attrib={"media-type": "text/css"})

    md = pkg.metadata
    total_ms = 0
    docs = con.execute("SELECT id, href, manifest_id FROM doc WHERE has_speech=1 ORDER BY id").fetchall()
    if doc_ids is not None:
        docs = [d for d in docs if d["id"] in doc_ids]
    # One ffmpeg per chapter, several at once (output.encode_jobs; 0 = one per CPU): sequential encoding used
    # one core and took 40 min for a 61 h book (2026-11-17).
    def _enc(d) -> int:
        stem = f"{d['id']:03d}"
        apath = pkg.file(f"ab/audio/{stem}.{FORMATS[fmt]['ext']}")
        apath.parent.mkdir(parents=True, exist_ok=True)
        encode(book.chapters / f"{stem}.wav", apath, fmt, bitrate)
        return probe_ms(apath)

    jobs = int(ocfg.get("encode_jobs", 0)) or (os.cpu_count() or 1)
    with ThreadPoolExecutor(max(1, min(jobs, len(docs) or 1))) as ex:
        enc = dict(zip([d["id"] for d in docs], ex.map(_enc, docs)))
    for d in docs:
        tl = json.loads((book.chapters / f"{d['id']:03d}.json").read_text())
        stem = f"{d['id']:03d}"
        audio_href = f"ab/audio/{stem}.{FORMATS[fmt]['ext']}"
        smil_href = f"ab/smil/{stem}.smil"
        enc_ms = enc[d["id"]]
        if tl["overlay_ms"] >= enc_ms:
            raise SystemExit(f"package: {d['href']}: last clip ends at {tl['overlay_ms']} ms, "
                             f"past encoded audio ({enc_ms} ms)")

        # segmented XHTML + overlay stylesheet link
        tree = xhtml.parse(book.seg / opf_dir / d["href"])
        head = tree.getroot().find(f"{{{XHTML_NS}}}head")
        etree.SubElement(head, f"{{{XHTML_NS}}}link", rel="stylesheet", type="text/css",
                         href=_rel(d["href"], css_href))
        xhtml.write(tree, pkg.file(d["href"]))

        text_rel = _rel(smil_href, d["href"])
        pars = "\n".join(
            f'      <par id="p{k + 1:05d}"><text src="{text_rel}#{sid}"/>'
            f'<audio src="{_rel(smil_href, audio_href)}" clipBegin="{clock(b)}" clipEnd="{clock(e)}"/></par>'
            for k, (sid, b, e) in enumerate(tl["clips"]))
        smil = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">\n'
                f'  <body>\n    <seq id="seq{stem}" epub:textref="{text_rel}">\n{pars}\n    </seq>\n  </body>\n</smil>\n')
        pkg.file(smil_href).parent.mkdir(parents=True, exist_ok=True)
        pkg.file(smil_href).write_text(smil, encoding="utf-8")

        mo_id = pkg.new_id(f"ab-mo{stem}")
        etree.SubElement(pkg.manifest, f"{{{OPF_NS}}}item", id=mo_id, href=smil_href,
                         attrib={"media-type": "application/smil+xml"})
        etree.SubElement(pkg.manifest, f"{{{OPF_NS}}}item", id=pkg.new_id(f"ab-au{stem}"), href=audio_href,
                         attrib={"media-type": FORMATS[fmt]["mime"]})
        pkg.item_el(d["manifest_id"]).set("media-overlay", mo_id)
        dur = tl["clips"][-1][2] - tl["clips"][0][1]
        m = etree.SubElement(md, f"{{{OPF_NS}}}meta", property="media:duration", refines="#" + mo_id)
        m.text = clock(dur)
        total_ms += dur

    for prop, val in (("media:duration", clock(total_ms)), ("media:active-class", ACTIVE_CLASS),
                      ("media:playback-active-class", PLAYING_CLASS)):
        m = etree.SubElement(md, f"{{{OPF_NS}}}meta", property=prop)
        m.text = val
    if not any(m.get("property") == "media:narrator" for m in md.findall(f"{{{OPF_NS}}}meta")):
        m = etree.SubElement(md, f"{{{OPF_NS}}}meta", property="media:narrator")
        from ..adapters.tts import settings
        from .cast import cast_path, load_cast
        nar = load_cast(book, cfg)["narrator"].get("voice") if cast_path(book, cfg).exists() else None
        m.text = f"{cfg['tts']['adapter']} {nar or settings(cfg['tts']).get('voice')}"
    epub.set_modified(pkg)
    pkg.save()

    title = pkg.title()
    out = book.out / f"{slugify(title)}{suffix}.epub"
    epub.zip_epub(build, out)
    log(f"package: {out.relative_to(book.dir)}  ({len(docs)} chapters with audio, "
        f"{clock(total_ms)}, {out.stat().st_size / 1e6:.1f} MB)")
    mode = str((cfg.get("output") or {}).get("epubcheck", "auto"))
    if mode == "off" or (mode == "auto" and not shutil.which("epubcheck")):
        # no epubcheck on this machine (epubcheck 5 needs Java 11+): check later with `readaloud epubcheck <id>`.
        book.reports.mkdir(parents=True, exist_ok=True)
        (book.reports / "epubcheck_skipped.txt").write_text(f"{out.name}: epubcheck {mode}, not run\n")
        log(f"package: epubcheck SKIPPED ({'off' if mode == 'off' else 'not on PATH'}); "
            f"check later with `readaloud epubcheck {book.id}`")
        return out
    _epubcheck(book, out, log)
    (book.reports / "epubcheck_skipped.txt").unlink(missing_ok=True)
    return out


def epubcheck_errors(path: Path) -> tuple[list[str], list[str]]:
    """(error lines normalized to 'CODE file(line,col): message', full output)."""
    exe = shutil.which("epubcheck")
    if not exe:
        raise SystemExit("epubcheck not on PATH (brew install epubcheck)")
    r = subprocess.run([exe, "--locale", "en", "-q", "-e", str(path)], capture_output=True, text=True)
    full = (r.stdout + r.stderr).splitlines()
    errs = []
    for ln in full:
        m = re.match(r"(FATAL|ERROR)\((\w+-\d+)\): (?:.*?\.epub/)?(.*)", ln)
        if m:
            errs.append(f"{m.group(2)} {m.group(3)}")
    return errs, full


def _epubcheck(book: Book, out: Path, log) -> None:
    """Pass if the output has no epubcheck errors beyond those already in the source book.

    Errors are matched by code + file + message (line numbers dropped: our
    edits shift them). Pre-existing errors are reported, not fixed.
    """
    def key(e: str) -> str:
        return re.sub(r"\(\d+,\d+\)", "", e)

    book.reports.mkdir(parents=True, exist_ok=True)
    errs, full = epubcheck_errors(out)
    (book.reports / "epubcheck_output.txt").write_text("\n".join(full))
    base_file = book.reports / "epubcheck_source.txt"
    if not base_file.exists():
        src_errs, _ = epubcheck_errors(book.source)
        base_file.write_text("\n".join(src_errs))
    src_keys = {key(ln) for ln in base_file.read_text().splitlines() if ln}
    new = [e for e in errs if key(e) not in src_keys]
    pre = len(errs) - len(new)
    if new:
        for ln in new[:15]:
            log(f"  {ln[:400]}")
        log(f"package: epubcheck {len(new)} NEW errors (+{pre} already in the source book); "
            f"see reports/epubcheck_output.txt")
        raise SystemExit(1)
    log(f"package: epubcheck passed, no new errors" + (f" ({pre} pre-existing in source book)" if pre else ""))
