"""Turn a finished book into video files for a video site: the cover as a still picture over the audio, with
chapter timestamps and sentence-level captions taken from the book's own timeline.

    python scripts/youtube_export.py pride [--max-hours 11.5] [--parts 3] [--cover cover.jpg] [--repo-url URL]
    python scripts/youtube_export.py pride --cut "CHAPTER XXIV." --cut "CHAPTER XLIII." --part-name Volume

Needs the assemble stage done (books/<id>/work/chapters/NNN.wav and .json) and ffmpeg. A long book is cut into
parts at chapter headings, as evenly as the headings allow, each part no longer than --max-hours. --cut names the
heading a part starts at instead (once per cut), for a book whose own volumes should be the parts. Written to
books/<id>/out/youtube/:

    <slug>-part1.mp4            H.264 still picture + AAC audio
    <slug>-part1.srt            one caption per sentence, as displayed in the book
    <slug>-part1.txt            a description to paste: what it is, credits, and the chapter list with timestamps
    <slug>-thumbnail.jpg        1280x720

Nothing is uploaded. Whether a book may be published where you publish it is yours to check (README).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import soundfile as sf
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
AUDIO_BITRATE = "192k"          # the site encodes again, so give it more than the 48k that goes into the EPUB
FRAME = (1920, 1080)
MIN_CHAPTER_MS = 10_000         # the site drops the whole chapter list when one entry is shorter than 10 s


def hms(ms: int, srt: bool = False) -> str:
    s, ms = divmod(int(ms), 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else (f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}")


def short(title: str, limit: int = 100) -> str:
    """A chapter title for the description: a long heading is cut at a word, not in the middle of one."""
    title = " ".join(title.split())
    if len(title) < limit:
        return title
    return title[:limit - 4].rsplit(" ", 1)[0].rstrip(" ,;:.") + "\u2026"


def load(book_dir: Path):
    """(docs, marks, total_ms). docs: [{id, wav, start, ms, clips}] in reading order, `start` on the book's
    clock. marks: [(ms, title)], one per chapter heading (headings that follow each other are one mark)."""
    con = sqlite3.connect(f"file:{book_dir / 'project.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sent = {r["id"]: r for r in con.execute("SELECT id, doc_id, seq, kind, text FROM sentence")}
    docs, t = [], 0
    for r in con.execute("SELECT id FROM doc WHERE has_speech=1 ORDER BY id"):
        j = book_dir / "work" / "chapters" / f"{r['id']:03d}.json"
        if not j.exists():
            raise SystemExit(f"{j} is missing: run the assemble stage first")
        tl = json.loads(j.read_text())
        docs.append({"id": r["id"], "wav": j.with_suffix(".wav"), "start": t, "ms": int(tl["file_ms"]),
                     "clips": [(sid, int(a), int(b)) for sid, a, b in tl["clips"]]})
        t += int(tl["file_ms"])
    marks, prev = [], None
    for d in docs:
        for sid, a, _ in d["clips"]:
            s = sent[sid]
            if s["kind"] == "heading":
                if prev and prev == (s["doc_id"], s["seq"] - 1):
                    marks[-1] = (marks[-1][0], f"{marks[-1][1]} {s['text']}")
                else:
                    marks.append((d["start"] + a, s["text"]))
                prev = (s["doc_id"], s["seq"])
    marks = [(m, short(title)) for m, title in marks]
    made = con.execute("SELECT adapter, COUNT(*) n FROM take GROUP BY 1 ORDER BY n DESC").fetchone()
    return docs, marks, t, sent, (made["adapter"] if made else None)


def _norm(t: str) -> str:
    return " ".join(t.casefold().split())


def cuts_at(marks: list, total: int, max_ms: int, starts: list[str]) -> list[int]:
    """Part boundaries where the caller says: each of `starts` is the beginning of one chapter heading. "CHAPTER XL"
    is not the beginning of "CHAPTER XLIII": the heading must end there or go on with something that is not a letter
    or a digit."""
    cuts = [0]
    for want in starts:
        w = _norm(want)
        hit = [m for m, t in marks if _norm(t).startswith(w)
               and (not w[-1:].isalnum() or not _norm(t)[len(w):len(w) + 1].isalnum())]
        if len(hit) != 1:
            raise SystemExit(f"--cut {want!r}: {len(hit)} chapter headings start with it, need exactly one"
                             + (f" ({[hms(m) for m in hit]})" if hit else ""))
        if hit[0] <= cuts[-1]:
            raise SystemExit(f"--cut {want!r} ({hms(hit[0])}) is not after the cut before it ({hms(cuts[-1])})")
        cuts.append(hit[0])
    cuts.append(total)
    long = [round((b - a) / 3.6e6, 2) for a, b in zip(cuts, cuts[1:]) if b - a > max_ms]
    if long:
        raise SystemExit(f"a part is longer than {max_ms / 3.6e6:.1f} h: {long} h")
    return cuts


def split_points(marks: list, total: int, max_ms: int, parts: int | None) -> list[int]:
    """Part boundaries on the book's clock, [0, ..., total], each cut at a chapter heading."""
    n = parts or max(1, math.ceil(total / max_ms))
    while True:
        cuts, cand = [0], [m for m, _ in marks if 0 < m < total]
        for k in range(1, n):
            want = k * total / n
            best = min((c for c in cand if c > cuts[-1]), key=lambda c: abs(c - want), default=None)
            if best is None:
                break
            cuts.append(best)
        cuts.append(total)
        if len(cuts) == n + 1 and all(b - a <= max_ms for a, b in zip(cuts, cuts[1:])):
            return cuts
        if parts or n > 60:
            raise SystemExit(f"cannot cut into {n} parts of at most {max_ms / 3.6e6:.1f} h at chapter headings "
                             f"(lengths: {[round((b - a) / 3.6e6, 2) for a, b in zip(cuts, cuts[1:])]})")
        n += 1


def write_part_wav(docs: list, a: int, b: int, out: Path) -> None:
    sr = sf.info(str(docs[0]["wav"])).samplerate
    with sf.SoundFile(str(out), "w", samplerate=sr, channels=1, subtype="PCM_16") as w:
        for d in docs:
            lo, hi = max(a, d["start"]), min(b, d["start"] + d["ms"])
            if lo >= hi:
                continue
            with sf.SoundFile(str(d["wav"])) as f:
                if f.samplerate != sr:
                    raise SystemExit(f"{d['wav']}: sample rate {f.samplerate}, expected {sr}")
                f.seek(int((lo - d["start"]) * sr / 1000))
                left = int((hi - lo) * sr / 1000)
                while left > 0:
                    block = f.read(min(left, sr * 30), dtype="int16")
                    if not len(block):
                        break
                    w.write(block)
                    left -= len(block)


def cover_of(book_dir: Path, out: Path) -> Path:
    """The cover image named in the EPUB's package file, copied next to the videos."""
    with zipfile.ZipFile(book_dir / "source.epub") as z:
        opf_name = etree.fromstring(z.read("META-INF/container.xml")).find(".//{*}rootfile").get("full-path")
        opf = etree.fromstring(z.read(opf_name))
        items = {i.get("id"): i for i in opf.iter("{*}item")}
        it = next((i for i in items.values() if "cover-image" in (i.get("properties") or "")), None)
        if it is None:
            meta = next((m for m in opf.iter("{*}meta") if m.get("name") == "cover"), None)
            it = items.get(meta.get("content")) if meta is not None else None
        if it is None:
            raise SystemExit("no cover image in the EPUB: pass --cover")
        href = (Path(opf_name).parent / it.get("href")).as_posix()
        dst = out / f"cover{Path(href).suffix or '.jpg'}"
        dst.write_bytes(z.read(href))
        return dst


def run(cmd: list) -> None:
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{cmd[0]} failed:\n{p.stderr[-1500:]}")


def still(cover: Path, out: Path, size=FRAME) -> None:
    """The cover centred on a blurred, darkened copy of itself."""
    w, h = size
    run(["ffmpeg", "-y", "-i", cover, "-filter_complex",
         f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=30:4,eq=brightness=-0.3[bg];"
         f"[0:v]scale=-2:{int(h * 0.92)}[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2", "-frames:v", "1", out])


SPEECH = {"drama3": "Built with Fish Audio (Drama 3).", "kokoro": "Speech: Kokoro."}


def description(title, author, k, n, marks, a, b, edition, repo, corrected, adapter, part="Part") -> str:
    L = [f"{title}, by {author}" + (f". {part} {k} of {n}." if n > 1 else "."), ""]
    L += ["A complete, unabridged reading with three synthetic voices: a narrator, a male voice and a female voice.",
          "Captions follow the text sentence by sentence.", ""]
    if edition:
        L += [f"Text: {edition}.", ""]
    L += ["The voices are computer-generated (AI). No human narrator was recorded or imitated."]
    if adapter in SPEECH:
        L += [SPEECH[adapter]]
    if corrected:
        L += ["Which voice reads each line of dialogue was assigned automatically and then corrected with the speaker "
              "labels of the Project Dialogism Novel Corpus (Vishnubhotla, Hammond and Hirst, 2022), CC-BY 4.0."]
    else:
        L += ["Which voice reads each line of dialogue was assigned automatically and may be wrong in places."]
    if repo:
        L += [f"Made with epub-readaloud: {repo}"]
    L += ["", "Chapters"]
    inside = []
    for m, t in marks:
        if a <= m < b:
            if inside and m - a - inside[-1][0] < MIN_CHAPTER_MS:      # a title page's lines: keep the last one
                inside[-1] = (inside[-1][0], t)
            else:
                inside.append((m - a, t))
    if not inside or inside[0][0] >= MIN_CHAPTER_MS:
        inside.insert(0, (0, "Start"))
    inside[0] = (0, inside[0][1])
    L += [f"{hms(m)} {t}" for m, t in inside]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("book")
    ap.add_argument("--books", default=str(ROOT / "books"))
    ap.add_argument("--max-hours", type=float, default=11.5, help="longest part (the site's limit is 12 h)")
    ap.add_argument("--parts", type=int, help="number of parts (default: as few as --max-hours allows)")
    ap.add_argument("--cut", action="append", metavar="HEADING",
                    help='start a part at the chapter heading that begins with this, e.g. "CHAPTER XXIV." '
                         "(repeat for each cut; instead of --parts)")
    ap.add_argument("--part-name", default="Part", help='what a part is called in the description, e.g. "Volume"')
    ap.add_argument("--cover", help="picture to show (default: the EPUB's cover)")
    ap.add_argument("--title")
    ap.add_argument("--author")
    ap.add_argument("--edition", help='where the text comes from, e.g. "Project Gutenberg ebook #1342"')
    ap.add_argument("--repo-url")
    ap.add_argument("--no-video", action="store_true", help="write captions, descriptions and the picture only")
    a = ap.parse_args()
    book_dir = Path(a.books) / a.book
    out = book_dir / "out" / "youtube"
    out.mkdir(parents=True, exist_ok=True)
    docs, marks, total, sent, adapter = load(book_dir)
    if a.cut and a.parts:
        raise SystemExit("--cut and --parts: give one of them")
    max_ms = int(a.max_hours * 3.6e6)
    cuts = cuts_at(marks, total, max_ms, a.cut) if a.cut else split_points(marks, total, max_ms, a.parts)
    n = len(cuts) - 1
    with zipfile.ZipFile(book_dir / "source.epub") as z:
        opf = etree.fromstring(z.read(etree.fromstring(z.read("META-INF/container.xml")).find(".//{*}rootfile").get("full-path")))
    title = a.title or (opf.findtext(".//{*}title") or a.book).strip()
    author = a.author or (opf.findtext(".//{*}creator") or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    ov = book_dir / "attribution_overrides.csv"
    corrected = ov.exists() and ",PDNC " in ov.read_text()
    cover = Path(a.cover) if a.cover else cover_of(book_dir, out)
    frame = out / f"{slug}-frame.png"
    still(cover, frame)
    still(cover, out / f"{slug}-thumbnail.jpg", (1280, 720))
    print(f"{title}: {hms(total)} in {n} part(s), {len(marks)} chapter marks -> {out}")
    for k, (lo, hi) in enumerate(zip(cuts, cuts[1:]), 1):
        stem = out / (f"{slug}-part{k}" if n > 1 else slug)
        caps, i = [], 0
        for d in docs:
            for sid, s, e in d["clips"]:
                t0, t1 = d["start"] + s, d["start"] + e
                if lo <= t0 < hi:
                    i += 1
                    caps.append(f"{i}\n{hms(t0 - lo, True)} --> {hms(min(t1, hi) - lo, True)}\n{sent[sid]['text']}\n")
        stem.with_suffix(".srt").write_text("\n".join(caps), encoding="utf-8")
        stem.with_suffix(".txt").write_text(description(title, author, k, n, marks, lo, hi, a.edition, a.repo_url, corrected, adapter,
                                                        a.part_name),
                                            encoding="utf-8")
        if not a.no_video:
            wav = stem.with_suffix(".part.wav")
            write_part_wav(docs, lo, hi, wav)
            run(["ffmpeg", "-y", "-loop", "1", "-framerate", "2", "-i", frame, "-i", wav, "-c:v", "libx264", "-tune",
                 "stillimage", "-pix_fmt", "yuv420p", "-r", "2", "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest",
                 "-movflags", "+faststart", stem.with_suffix(".mp4")])
            wav.unlink()
        print(f"  part {k}: {hms(hi - lo)}, {len(caps)} captions, {sum(1 for m, _ in marks if lo <= m < hi)} chapters"
              f" -> {stem.name}.mp4" if not a.no_video else f"  part {k}: {hms(hi - lo)}, {len(caps)} captions")


if __name__ == "__main__":
    sys.exit(main())
