#!/usr/bin/env python3
"""
mo_test_epub.py - build small EPUB 3 + Media Overlays test books for checking
how a reading system (Readest, Thorium) handles read-along playback.

One EPUB is written per (granularity x audio format), e.g.
    mo-test_sentence_mp3cbr.epub
    mo-test_word_opus.epub

Usage
    python3 mo_test_epub.py                          # Kokoro, sentence+word, mp3cbr+opus
    python3 mo_test_epub.py --backend tone           # beeps, no TTS, exact word timing
    python3 mo_test_epub.py --formats mp3cbr,mp3vbr,opus,aac
    python3 mo_test_epub.py --text my_excerpt.txt    # your own text (format below)

TTS backends
    kokoro  Natural voice. pip install kokoro soundfile ; apt install espeak-ng
            Word timings come from Kokoro's token timestamps when they line up
            with the text; otherwise they are estimated from word length.
    espeak  espeak-ng command line. Word timings are estimated.
    tone    One beep per word, no TTS. Word timings are exact, so this is the
            best backend for judging highlight sync and seek accuracy.

Audio formats
    mp3cbr  MP3 64 kbps constant bitrate      audio/mpeg
    mp3vbr  MP3 VBR (-q:a 5)                  audio/mpeg   (to compare seek accuracy)
    opus    Opus 32 kbps in Ogg               audio/ogg; codecs=opus
    aac     AAC-LC 64 kbps in MP4 (.m4a)      audio/mp4    (fallback if Opus fails on iOS)

Text file format (UTF-8)
    # Chapter title        starts a chapter with audio (the title is narrated)
    #- Page title          starts a page with no audio (tests how the player skips it)
    One sentence per line. A blank line starts a new paragraph.

Requires Python 3.9+, numpy, and ffmpeg/ffprobe (libmp3lame, libopus).
epubcheck runs automatically if it is on PATH, or pass --epubcheck PATH
(an epubcheck executable or epubcheck.jar).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")

ACTIVE_CLASS = "mo-active"
PLAYING_CLASS = "mo-playing"

FORMATS = {
    "mp3cbr": {"ext": "mp3", "mime": "audio/mpeg",
               "args": ["-c:a", "libmp3lame", "-b:a", "64k", "-ar", "44100"], "enc": "libmp3lame"},
    "mp3vbr": {"ext": "mp3", "mime": "audio/mpeg",
               "args": ["-c:a", "libmp3lame", "-q:a", "5", "-ar", "44100"], "enc": "libmp3lame"},
    "opus":   {"ext": "opus", "mime": "audio/ogg; codecs=opus",
               "args": ["-c:a", "libopus", "-b:a", "32k"], "enc": "libopus"},
    "aac":    {"ext": "m4a", "mime": "audio/mp4",
               "args": ["-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-movflags", "+faststart"], "enc": "aac"},
}

# Original sample text. Chapter Two is long on purpose (auto page turn);
# "Part Two" has no audio on purpose (does the player skip it?).
SAMPLE_TEXT = """\
# Chapter One: The Keeper
Mara Quill had kept the lighthouse on Gannet Rock for eleven years.
In all that time, the lamp had never once gone dark.
She was proud of that, though she rarely said so out loud.

Each evening at six, she climbed the hundred and twelve steps to the lantern room.
She wound the clockwork, trimmed the wick, and polished the great glass lens until it shone.
“There you are,” she would tell it, as if it were an old dog waiting by the door.
The lens, being a lens, never answered.

# Chapter Two: The Long Night
The storm arrived a little after nine, earlier than the radio had promised.
At first it was only a hard rain, rattling against the windows like thrown gravel.
Then the wind found its voice, and the whole tower began to hum.

Mara had weathered storms before.
She knew the sounds the old building made, the groans and ticks and sighs of iron cooling and stone settling.
She knew which windows leaked and which doors would bang if she forgot to latch them.
Tonight, though, there was a new sound, a low and steady knocking that seemed to come from somewhere beneath her feet.
She told herself it was a loose shutter.
She told herself that twice, and then a third time, and then she took the lantern and went down to look.

The stairwell was colder than it should have been.
Water had found its way under the seaward door and spread across the bottom landing in a thin, shining sheet.
The knocking was louder here, slow and patient, like someone who had all night and intended to use it.
Mara set the lantern on the bottom step and listened.
Knock.
A pause.
Knock, knock.
It was coming from the storeroom, the one she used for paint tins and spare wicks and the broken radio she kept meaning to send back to the mainland.
She had not opened that door since the spring.
She opened it now.

Inside, nothing had changed.
The paint tins stood in their crooked rows.
The spare wicks lay coiled in their box like sleeping snakes.
But the small window at the back of the room, the one no bigger than a book, was wide open, and the storm was pouring through it.
Its latch had worked loose, and with every gust the frame swung out and struck the wall.
Knock.
Mara laughed, partly from relief and partly from embarrassment, and the sound echoed strangely off the damp stone.

She wrestled the window shut, wedged it with a folded rag, and stood for a moment with her back against the door.
Her heart was still running faster than it needed to.
“A window,” she said to the empty room.
“Eleven years, and I’m frightened by a window.”
The room, like the lens, did not answer.

When she climbed back to the lantern room, the light was still turning.
Its long white beam swept out across the water, caught the tops of the waves, and moved on.
Somewhere out there, she knew, a ship was watching for it.
She pulled a chair up to the glass and settled in to wait for the morning.

#- Part Two
This page has no audio. The player should move on to the next chapter by itself.

# Chapter Three: Morning
By dawn the storm had blown itself out.
The sea lay flat and grey, as if it had never done anything wrong in its life.
Mara made a pot of strong tea and carried it up to the gallery.

Far off to the east, a fishing boat was coming home, its engine a faint and steady drone.
As it passed the rock, someone on deck raised an arm and waved.
Mara waved back.
“Good morning,” she said, though they could not possibly hear her.
Then she went inside to write up the log: wind from the southwest, heavy rain, one window, no harm done.
"""


# --------------------------------------------------------------------------- model

@dataclass
class Word:
    text: str
    start: int  # char offsets inside Sentence.text
    end: int


@dataclass
class Sentence:
    text: str
    words: List[Word] = field(default_factory=list)
    sid: str = ""
    speech_start: float = 0.0  # seconds on the chapter timeline
    speech_end: float = 0.0
    word_times: List[Tuple[float, float]] = field(default_factory=list)
    timing: str = ""  # "aligned" | "estimated" | "exact"


@dataclass
class Chapter:
    title: str
    has_audio: bool
    heading: Sentence
    paragraphs: List[List[Sentence]] = field(default_factory=list)
    stem: str = ""
    overlay_end: float = 0.0  # seconds; clipEnd of the last clip
    wav: Optional[Path] = None

    def sentences(self) -> List[Sentence]:
        return [self.heading] + [s for p in self.paragraphs for s in p]


def make_sentence(text: str) -> Sentence:
    words = [Word(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    return Sentence(text=text, words=words)


def parse_text(src: str) -> List[Chapter]:
    chapters: List[Chapter] = []
    cur: Optional[Chapter] = None
    para: List[Sentence] = []

    def flush() -> None:
        nonlocal para
        if para and cur is not None:
            cur.paragraphs.append(para)
        para = []

    for raw in src.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            flush()
            no_audio = line.startswith("#-")
            title = (line[2:] if no_audio else line[1:]).strip() or "Untitled"
            cur = Chapter(title=title, has_audio=not no_audio, heading=make_sentence(title))
            chapters.append(cur)
        elif not line:
            flush()
        else:
            if cur is None:
                cur = Chapter(title="Untitled", has_audio=True, heading=make_sentence("Untitled"))
                chapters.append(cur)
            para.append(make_sentence(line))
    flush()
    if not any(c.has_audio for c in chapters):
        sys.exit("No chapter with audio found in the text.")
    return chapters


def assign_ids(chapters: List[Chapter]) -> None:
    for i, ch in enumerate(chapters, 1):
        ch.stem = f"c{i:02d}"
        for n, s in enumerate(ch.sentences()):
            s.sid = f"{ch.stem}s{n:04d}"


def word_id(s: Sentence, k: int) -> str:
    return f"{s.sid}w{k + 1:03d}"


# --------------------------------------------------------------------------- audio utils

def read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        data = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    a = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def trim(audio: np.ndarray, sr: int, times, rel_db: float = -40.0, margin: float = 0.03):
    """Cut leading/trailing silence (relative to peak), keep a small margin, shift word times."""
    if audio.size == 0:
        return audio, times
    mag = np.abs(audio)
    peak = float(mag.max())
    if peak <= 0.0:
        return audio, times
    idx = np.flatnonzero(mag > peak * 10 ** (rel_db / 20))
    a0 = max(0, int(idx[0] - margin * sr))
    a1 = min(audio.size, int(idx[-1] + margin * sr) + 1)
    out = audio[a0:a1]
    if times is not None:
        shift, dur = a0 / sr, out.size / sr
        times = [(min(max(st - shift, 0.0), dur), min(max(et - shift, 0.0), dur)) for st, et in times]
    return out, times


def estimate_word_times(s: Sentence, dur: float) -> List[Tuple[float, float]]:
    """Spread the clip duration over words by length, with extra weight before , ; : dashes."""
    if not s.words:
        return []
    weights = []
    for i, w in enumerate(s.words):
        nxt = s.words[i + 1].start if i + 1 < len(s.words) else len(s.text)
        pause = 4 if re.search(r"[,;:—–]", s.text[w.end:nxt]) and i + 1 < len(s.words) else 0
        weights.append(len(w.text) + 1 + pause)
    lo = min(0.03, dur * 0.1)
    hi = max(dur - 0.03, dur * 0.9)
    total, t, out = float(sum(weights)), lo, []
    for wgt in weights:
        d = (hi - lo) * wgt / total
        out.append((t, t + d))
        t += d
    return out


def _norm(w: str) -> str:
    return w.lower().replace("’", "'")


def tokens_to_word_times(words: List[Word], toks) -> Optional[List[Tuple[float, float]]]:
    """Map (token_text, start, end) to our words. None if they don't line up 1:1."""
    flat = []
    for text, st, et in toks:
        pieces = WORD_RE.findall(text or "")
        if not pieces:
            continue
        total, t = float(sum(len(p) for p in pieces)), st
        for p in pieces:
            d = (et - st) * len(p) / total
            flat.append((_norm(p), t, t + d))
            t += d
    if len(flat) != len(words) or any(f[0] != _norm(w.text) for f, w in zip(flat, words)):
        return None
    return [(st, et) for _, st, et in flat]


# --------------------------------------------------------------------------- TTS backends

class ToneBackend:
    """One beep per word. Pitch changes per sentence. Timings are exact."""
    name = "tone"
    exact = True

    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.label = "tone (beep per word)"
        self._n = 0

    def synth(self, s: Sentence):
        self._n += 1
        if not s.words:
            return np.zeros(int(0.3 * self.sr), np.float32), []
        freq = 440.0 * 2 ** ((self._n % 4) * 3 / 12)
        parts, times, t = [], [], 0.0
        for i, w in enumerate(s.words):
            n = int(min(0.10 + 0.035 * len(w.text), 0.45) * self.sr)
            env = np.ones(n, np.float32)
            fade = min(int(0.005 * self.sr), n // 2)
            env[:fade] = np.linspace(0, 1, fade)
            env[n - fade:] = np.linspace(1, 0, fade)
            tone = 0.5 * np.sin(2 * np.pi * freq * np.arange(n) / self.sr) * env
            parts.append(tone.astype(np.float32))
            times.append((t, t + n / self.sr))
            t += n / self.sr
            if i + 1 < len(s.words):
                g = int(0.07 * self.sr)
                parts.append(np.zeros(g, np.float32))
                t += g / self.sr
        return np.concatenate(parts), times


class EspeakBackend:
    name = "espeak"
    exact = False

    def __init__(self, voice: Optional[str], wpm: int = 160):
        self.exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.exe:
            sys.exit("espeak-ng not found. Install it (sudo apt install espeak-ng) or use --backend tone.")
        self.voice = voice or "en-us"
        self.wpm = wpm
        self.label = f"espeak-ng {self.voice}"
        _, self.sr = self._run("Hello.")

    def _run(self, text: str):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "s.wav"
            subprocess.run([self.exe, "-v", self.voice, "-s", str(self.wpm), "-w", str(out), "--stdin"],
                           input=text.encode("utf-8"), check=True)
            return read_wav(out)

    def synth(self, s: Sentence):
        audio, sr = self._run(s.text)
        if sr != self.sr:
            raise RuntimeError("espeak-ng changed sample rate between calls")
        return audio, None


class KokoroBackend:
    name = "kokoro"
    exact = False

    def __init__(self, voice: Optional[str], speed: float, lang: str, device: Optional[str]):
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError:
            sys.exit("kokoro is not installed: pip install kokoro soundfile  (and sudo apt install espeak-ng)\n"
                     "Or run with --backend tone / --backend espeak.")
        kw = {"lang_code": lang}
        if device:
            kw["device"] = device
        self.pipe = KPipeline(**kw)
        self.voice = voice or "af_heart"
        self.speed = speed
        self.sr = 24000
        self.label = f"Kokoro {self.voice}"

    def synth(self, s: Sentence):
        parts, toks, off = [], [], 0.0
        for r in self.pipe(s.text, voice=self.voice, speed=self.speed):
            a = getattr(r, "audio", None)
            if a is None:
                continue
            a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            a = a.astype(np.float32).reshape(-1)
            for t in getattr(r, "tokens", None) or []:
                st, et = getattr(t, "start_ts", None), getattr(t, "end_ts", None)
                if st is not None and et is not None:
                    toks.append((getattr(t, "text", ""), off + float(st), off + float(et)))
            parts.append(a)
            off += a.size / self.sr
        if not parts:
            raise RuntimeError(f"Kokoro returned no audio for: {s.text!r}")
        return np.concatenate(parts), tokens_to_word_times(s.words, toks)


# --------------------------------------------------------------------------- synthesis

def synth_chapter(ch: Chapter, backend, args, workdir: Path, stats: dict) -> None:
    sr = backend.sr

    def sil(sec: float) -> np.ndarray:
        return np.zeros(int(round(sec * sr)), np.float32)

    parts = [sil(args.lead_in)]
    pos = parts[0].size
    blocks = [[ch.heading]] + ch.paragraphs
    for bi, block in enumerate(blocks):
        for si, s in enumerate(block):
            audio, times = backend.synth(s)
            if times is not None and len(times) != len(s.words):
                times = None
            audio, times = trim(audio, sr, times)
            dur = audio.size / sr
            if times is None:
                times = estimate_word_times(s, dur)
                s.timing = "estimated"
            else:
                s.timing = "exact" if backend.exact else "aligned"
            stats[s.timing] = stats.get(s.timing, 0) + 1
            t0 = pos / sr
            s.speech_start, s.speech_end = t0, t0 + dur
            s.word_times = [(t0 + a, t0 + b) for a, b in times]
            parts.append(audio)
            pos += audio.size
            if si + 1 < len(block):
                gap = args.sentence_gap
            elif bi + 1 < len(blocks):
                gap = args.heading_gap if bi == 0 else args.paragraph_gap
            else:
                gap = 0.0
            if gap > 0:
                g = sil(gap)
                parts.append(g)
                pos += g.size
    last_speech_end = pos / sr
    # Keep the last clipEnd well inside the file: if clipEnd is past the decoded
    # end, some web players never reach it and stall at the end of the chapter.
    ch.overlay_end = last_speech_end + 0.4
    parts.append(sil(1.0))
    audio = np.concatenate(parts)
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio *= 0.708 / peak  # -3 dBFS peak
    ch.wav = workdir / f"{ch.stem}.wav"
    write_wav(ch.wav, audio, sr)


def ms(sec: float) -> int:
    return int(round(sec * 1000))


def clock(msec: int) -> str:
    h, rem = divmod(msec, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, frac = divmod(rem, 1000)
    return f"{h}:{m:02d}:{s:02d}.{frac:03d}"


def build_clips(ch: Chapter, granularity: str) -> List[Tuple[str, int, int]]:
    """Contiguous clips (each ends where the next begins) covering 0..overlay_end."""
    items = []
    for s in ch.sentences():
        if granularity == "word" and s.words:
            items += [(word_id(s, k), st) for k, (st, _) in enumerate(s.word_times)]
        else:
            items.append((s.sid, s.speech_start))
    begins, prev = [], -10
    for i, (_, st) in enumerate(items):
        b = 0 if i == 0 else max(ms(st), prev + 10)
        begins.append(b)
        prev = b
    end = ms(ch.overlay_end)
    if begins[-1] >= end:
        raise RuntimeError(f"{ch.stem}: clip times overflow the chapter audio")
    return [(tid, begins[i], begins[i + 1] if i + 1 < len(items) else end)
            for i, (tid, _) in enumerate(items)]


# --------------------------------------------------------------------------- ffmpeg

def check_ffmpeg(formats: List[str]) -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found on PATH.")
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    missing = [f for f in formats if not re.search(rf"\s{FORMATS[f]['enc']}\s", enc)]
    if missing:
        sys.exit("ffmpeg lacks encoders for: " + ", ".join(f"{f} ({FORMATS[f]['enc']})" for f in missing))


def encode(wav: Path, out: Path, fmt: str) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
           "-ac", "1", "-map_metadata", "-1", *FORMATS[fmt]["args"], str(out)]
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> Optional[float]:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------- EPUB parts

def esc(t: str) -> str:
    return html.escape(t, quote=False)


def render_sentence(s: Sentence, granularity: str, with_ids: bool) -> str:
    if not with_ids:
        return esc(s.text)
    if granularity == "word" and s.words:
        buf, cur = [], 0
        for k, w in enumerate(s.words):
            buf.append(esc(s.text[cur:w.start]))
            buf.append(f'<span id="{word_id(s, k)}">{esc(w.text)}</span>')
            cur = w.end
        buf.append(esc(s.text[cur:]))
        inner = "".join(buf)
    else:
        inner = esc(s.text)
    return f'<span id="{s.sid}">{inner}</span>'


XHTML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="../css/style.css"/>
</head>
<body>
<section epub:type="{etype}"{role}>
{body}
</section>
</body>
</html>
"""

CSS = f"""body {{ font-family: Georgia, serif; line-height: 1.5; }}
h1 {{ font-size: 1.4em; margin: 1.5em 0 1em; }}
p {{ margin: 0 0 0.8em; text-indent: 1.2em; }}
p.meta {{ text-indent: 0; font-family: sans-serif; font-size: 0.85em; }}
/* Light green on purpose: if the highlight shows up light green, the reader is
   using the book's media:active-class. Any other colour is the reader's own style. */
.{ACTIVE_CLASS} {{ background-color: #b9f6ca; color: inherit; border-radius: 0.15em; }}
"""


def render_chapter(ch: Chapter, granularity: str) -> str:
    ids = ch.has_audio
    body = [f"<h1>{render_sentence(ch.heading, granularity, ids)}</h1>"]
    for para in ch.paragraphs:
        body.append("<p>" + " ".join(render_sentence(s, granularity, ids) for s in para) + "</p>")
    if ch.has_audio:
        etype, role = "chapter", ' role="doc-chapter"'
    else:
        etype, role = "part", ' role="doc-part"'
    return XHTML.format(title=esc(ch.title), etype=etype, role=role, body="\n".join(body))


def render_title_page(book_title: str, info: List[Tuple[str, str]]) -> str:
    rows = "\n".join(f'<p class="meta"><b>{esc(k)}:</b> {esc(v)}</p>' for k, v in info)
    body = (f"<h1>{esc(book_title)}</h1>\n"
            '<p class="meta">This title page has no audio. Press play here first: '
            "does playback jump to Chapter One by itself?</p>\n" + rows)
    return XHTML.format(title=esc(book_title), etype="titlepage", role="", body=body)


def render_nav(chapters: List[Chapter]) -> str:
    lis = ['<li><a href="text/title.xhtml">Title page</a></li>']
    lis += [f'<li><a href="text/{c.stem}.xhtml">{esc(c.title)}</a></li>' for c in chapters]
    return ("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="en" lang="en">
<head><meta charset="utf-8"/><title>Contents</title></head>
<body>
<nav epub:type="toc" id="toc" role="doc-toc">
<h1>Contents</h1>
<ol>
""" + "\n".join(lis) + """
</ol>
</nav>
</body>
</html>
""")


def render_smil(ch: Chapter, clips, audio_href: str) -> str:
    pars = "\n".join(
        f'      <par id="par{i + 1:05d}">\n'
        f'        <text src="../text/{ch.stem}.xhtml#{tid}"/>\n'
        f'        <audio src="{audio_href}" clipBegin="{clock(b)}" clipEnd="{clock(e)}"/>\n'
        f"      </par>"
        for i, (tid, b, e) in enumerate(clips))
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<smil xmlns="http://www.w3.org/ns/SMIL" xmlns:epub="http://www.idpf.org/2007/ops" version="3.0">
  <body>
    <seq id="seq_{ch.stem}" epub:textref="../text/{ch.stem}.xhtml" epub:type="chapter">
{pars}
    </seq>
  </body>
</smil>
"""


def render_opf(title: str, narrator: str, chapters: List[Chapter], fmt: str,
               durations_ms: dict) -> str:
    f = FORMATS[fmt]
    modified = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = [f'    <meta property="media:duration" refines="#mo_{c.stem}">{clock(durations_ms[c.stem])}</meta>'
            for c in chapters if c.has_audio]
    meta.append(f'    <meta property="media:duration">{clock(sum(durations_ms.values()))}</meta>')
    meta.append(f'    <meta property="media:narrator">{esc(narrator)}</meta>')
    meta.append(f'    <meta property="media:active-class">{ACTIVE_CLASS}</meta>')
    meta.append(f'    <meta property="media:playback-active-class">{PLAYING_CLASS}</meta>')
    items = ['    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
             '    <item id="css" href="css/style.css" media-type="text/css"/>',
             '    <item id="title" href="text/title.xhtml" media-type="application/xhtml+xml"/>']
    spine = ['    <itemref idref="title"/>']
    for c in chapters:
        mo = f' media-overlay="mo_{c.stem}"' if c.has_audio else ""
        items.append(f'    <item id="x_{c.stem}" href="text/{c.stem}.xhtml" media-type="application/xhtml+xml"{mo}/>')
        if c.has_audio:
            items.append(f'    <item id="mo_{c.stem}" href="smil/{c.stem}.smil" media-type="application/smil+xml"/>')
            items.append(f'    <item id="au_{c.stem}" href="audio/{c.stem}.{f["ext"]}" media-type="{f["mime"]}"/>')
        spine.append(f'    <itemref idref="x_{c.stem}"/>')
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="pub-id" xml:lang="en">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="pub-id">urn:uuid:{uuid.uuid4()}</dc:identifier>
    <dc:title>{esc(title)}</dc:title>
    <dc:language>en</dc:language>
    <dc:creator>mo_test_epub.py</dc:creator>
    <meta property="dcterms:modified">{modified}</meta>
{chr(10).join(meta)}
  </metadata>
  <manifest>
{chr(10).join(items)}
  </manifest>
  <spine>
{chr(10).join(spine)}
  </spine>
</package>
"""


CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def build_epub(path: Path, title: str, narrator: str, chapters: List[Chapter], granularity: str,
               fmt: str, audio_dir: Path, info: List[Tuple[str, str]]) -> None:
    ext = FORMATS[fmt]["ext"]
    durations, smils = {}, {}
    for c in chapters:
        if c.has_audio:
            clips = build_clips(c, granularity)
            durations[c.stem] = clips[-1][2] - clips[0][1]
            smils[c.stem] = render_smil(c, clips, f"../audio/{c.stem}.{ext}")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(zipfile.ZipInfo("mimetype"), "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        def put(name: str, text: str) -> None:
            z.writestr(name, text.encode("utf-8"), compress_type=zipfile.ZIP_DEFLATED)
        put("META-INF/container.xml", CONTAINER)
        put("OEBPS/content.opf", render_opf(title, narrator, chapters, fmt, durations))
        put("OEBPS/nav.xhtml", render_nav(chapters))
        put("OEBPS/css/style.css", CSS)
        put("OEBPS/text/title.xhtml", render_title_page(title, info))
        for c in chapters:
            put(f"OEBPS/text/{c.stem}.xhtml", render_chapter(c, granularity))
            if c.has_audio:
                put(f"OEBPS/smil/{c.stem}.smil", smils[c.stem])
                # Stored, not deflated: already compressed, and lets the reader seek inside the zip.
                z.write(audio_dir / f"{c.stem}.{ext}", f"OEBPS/audio/{c.stem}.{ext}",
                        compress_type=zipfile.ZIP_STORED)


# --------------------------------------------------------------------------- epubcheck

def epubcheck_cmd(arg: Optional[str]) -> Optional[List[str]]:
    if arg:
        return ["java", "-jar", arg] if arg.endswith(".jar") else [arg]
    exe = shutil.which("epubcheck")
    return [exe] if exe else None


def run_epubcheck(cmd: List[str], epub: Path) -> Tuple[bool, List[str]]:
    r = subprocess.run(cmd + [str(epub)], capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("FATAL", "ERROR", "WARNING"))]
    return r.returncode == 0, lines


# --------------------------------------------------------------------------- checklist

CHECKLIST = """# Readest Media Overlay test checklist

Generated: {when}
Voice: {narrator}
Test every file once on PC, Android and iOS.

## Files

{files}

## What to check

1. **It plays**: pressing play gives sound. Watch the Opus files on iOS in particular.
2. **Starting on the title page**: the title page has no audio. Does pressing play there jump to chapter one?
3. **Highlighting**: the text being read is marked.
4. **Highlight style**: light green = Readest applied the book's `media:active-class`; another colour = Readest used its own style.
5. **Word level** (`_word_` files only): does it light up word by word, or the whole sentence or paragraph, or nothing?
6. **Sync**: do sound and highlight match? Still at the end of chapter two? (If seeking is inexact with VBR, it usually shows more the further in. Files made with `--backend tone` are the easiest to judge: one beep per word, and the highlight should start with the beep.)
7. **Automatic page turn**: chapter two is long. Does the page turn by itself at the bottom?
8. **Chapter change, skipping a page without audio**: after chapter two, does it skip "Part Two" (no audio) and go on to chapter three?
9. **Tap to play**: tapping a sentence starts reading from it?
10. **Background play**: switch to another app and lock the screen for a minute during playback. Does the sound stop?
11. **Speed**: is the highlight still in sync at 1.5x and 2x?
12. **Resume**: pause, close the app, open it again. Does it continue from the same place?

## Results

| File | PC | Android | iOS |
|---|---|---|---|
{rows}
"""


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build EPUB 3 Media Overlay test books for Readest/Thorium.")
    ap.add_argument("--backend", choices=["kokoro", "espeak", "tone"], default="kokoro")
    ap.add_argument("--voice", help="kokoro voice (default af_heart) or espeak voice (default en-us)")
    ap.add_argument("--speed", type=float, default=1.0, help="kokoro speed")
    ap.add_argument("--lang", default="a", help="kokoro lang_code: a = American, b = British")
    ap.add_argument("--device", help="kokoro device, e.g. cuda or cpu")
    ap.add_argument("--formats", default="mp3cbr,opus", help="comma list of: " + ",".join(FORMATS))
    ap.add_argument("--granularity", default="sentence,word", help="comma list of: sentence,word")
    ap.add_argument("--text", type=Path, help="your own text file (format in the script header)")
    ap.add_argument("--title", default="MO Test")
    ap.add_argument("--out", type=Path, default=Path("mo_test_out"))
    ap.add_argument("--lead-in", type=float, default=0.3, help="silence before the chapter title (s)")
    ap.add_argument("--heading-gap", type=float, default=1.0, help="pause after the chapter title (s)")
    ap.add_argument("--sentence-gap", type=float, default=0.35, help="pause between sentences (s)")
    ap.add_argument("--paragraph-gap", type=float, default=0.8, help="pause between paragraphs (s)")
    ap.add_argument("--epubcheck", help="path to epubcheck executable or epubcheck.jar")
    ap.add_argument("--keep-work", action="store_true", help="keep WAV and encoded audio in OUT/_work")
    args = ap.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    grans = [g.strip() for g in args.granularity.split(",") if g.strip()]
    bad = [f for f in formats if f not in FORMATS] + [g for g in grans if g not in ("sentence", "word")]
    if bad or not formats or not grans:
        sys.exit(f"Unknown or empty --formats/--granularity value: {bad}")
    check_ffmpeg(formats)

    src = args.text.read_text(encoding="utf-8") if args.text else SAMPLE_TEXT
    chapters = parse_text(src)
    assign_ids(chapters)

    if args.backend == "tone":
        backend = ToneBackend()
    elif args.backend == "espeak":
        backend = EspeakBackend(args.voice)
    else:
        backend = KokoroBackend(args.voice, args.speed, args.lang, args.device)

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "_work"
    work.mkdir(exist_ok=True)

    stats: dict = {}
    for c in chapters:
        if c.has_audio:
            print(f"[tts] {c.stem} {c.title}")
            synth_chapter(c, backend, args, work, stats)

    for fmt in formats:
        d = work / fmt
        d.mkdir(exist_ok=True)
        for c in chapters:
            if not c.has_audio:
                continue
            out = d / f"{c.stem}.{FORMATS[fmt]['ext']}"
            encode(c.wav, out, fmt)
            got = probe_duration(out)
            if got is not None and got < c.overlay_end + 0.05:
                print(f"  ! {fmt} {c.stem}: encoded audio {got:.3f}s is shorter than overlay end "
                      f"{c.overlay_end:.3f}s")

    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    built = []
    for g in grans:
        for fmt in formats:
            name = f"mo-test_{g}_{fmt}.epub"
            title = f"{args.title} · {g} · {fmt}"
            info = [("Highlight granularity", g), ("Audio", f"{fmt} ({FORMATS[fmt]['mime']})"),
                    ("Voice", backend.label), ("Generated", when)]
            path = args.out / name
            build_epub(path, title, backend.label, chapters, g, fmt, work / fmt, info)
            built.append(path)
            print(f"[epub] {path}")

    cmd = epubcheck_cmd(args.epubcheck)
    if cmd:
        for p in built:
            ok, lines = run_epubcheck(cmd, p)
            print(f"[epubcheck] {p.name}: {'OK' if ok else 'FAILED'}")
            for l in lines[:20]:
                print("    " + l)
    else:
        print("[epubcheck] not found, skipped (pass --epubcheck /path/to/epubcheck.jar)")

    timings = {c.stem: {"title": c.title, "overlay_end": round(c.overlay_end, 3), "sentences": [
        {"id": s.sid, "text": s.text, "start": round(s.speech_start, 3), "end": round(s.speech_end, 3),
         "timing": s.timing, "words": [[w.text, round(a, 3), round(b, 3)]
                                       for w, (a, b) in zip(s.words, s.word_times)]}
        for s in c.sentences()]} for c in chapters if c.has_audio}
    (args.out / "timings.json").write_text(json.dumps(timings, ensure_ascii=False, indent=1), encoding="utf-8")

    files = "\n".join(f"- `{p.name}`" for p in built)
    rows = "\n".join(f"| `{p.name}` |  |  |  |" for p in built)
    (args.out / "TEST_CHECKLIST.md").write_text(
        CHECKLIST.format(when=when, narrator=backend.label, files=files, rows=rows), encoding="utf-8")

    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)

    total = sum(c.overlay_end for c in chapters if c.has_audio)
    print(f"\nDone: {len(built)} EPUB(s) in {args.out}/, audio {total / 60:.1f} min per book.")
    print("Word timing sources: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    print(f"Checklist: {args.out / 'TEST_CHECKLIST.md'}")


if __name__ == "__main__":
    main()
