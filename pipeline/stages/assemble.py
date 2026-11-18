"""Stage 9: trim takes, add fixed pauses, normalize, write one WAV + timeline per document (F10).

Clips follow the layout verified on Readest (mo_test_epub.py): contiguous
(each ends where the next begins), the first starts at 0, and the last ends
0.4 s after the final speech while the file runs 1.0 s past that.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .. import audio
from ..book import Book

TAIL_PAD = 1.0
LAST_CLIP_EXTRA = 0.4


def _ms(sec: float) -> int:
    return int(round(sec * 1000))


def run(book: Book, con, cfg: dict, log=print, doc_ids: list[int] | None = None) -> None:
    pauses, acfg = cfg["pauses"], cfg["audio"]
    book.chapters.mkdir(parents=True, exist_ok=True)
    docs = con.execute("SELECT id, href FROM doc WHERE has_speech=1 ORDER BY id").fetchall()
    if doc_ids is not None:
        docs = [d for d in docs if d["id"] in doc_ids]
    built = 0
    for d in docs:
        urows = con.execute(
            "SELECT s.id, s.para, s.kind, u.id uid, t.wav_path, t.key, t.id take_id FROM sentence s "
            "JOIN utterance u ON u.sentence_id = s.id "
            "LEFT JOIN selection sel ON sel.block_id = u.id "
            "LEFT JOIN take t ON t.id = sel.take_id WHERE s.doc_id=? AND s.speak=1 ORDER BY s.seq, u.seq",
            (d["id"],)).fetchall()
        missing = [r["uid"] for r in urows if r["wav_path"] is None]
        if missing:
            raise SystemExit(f"assemble: {d['href']}: {len(missing)} utterances have no take (run synth)")
        rows = []          # one entry per sentence: {id, para, kind, wavs: [...]}
        for r in urows:
            if not rows or rows[-1]["id"] != r["id"]:
                rows.append({"id": r["id"], "para": r["para"], "kind": r["kind"], "wavs": [], "keys": []})
            rows[-1]["wavs"].append(r["wav_path"])
            # the take row too: a take made again (readaloud regen) keeps its key, and the chapter must be rebuilt
            rows[-1]["keys"].append([r["key"], r["take_id"]])
        fp = hashlib.sha256(json.dumps([[r["id"], r["keys"], r["para"], r["kind"]] for r in rows]
                                       + [pauses, acfg], sort_keys=True).encode()).hexdigest()
        tl_path = book.chapters / f"{d['id']:03d}.json"
        wav_path = book.chapters / f"{d['id']:03d}.wav"
        if tl_path.exists() and wav_path.exists() and json.loads(tl_path.read_text()).get("fingerprint") == fp:
            continue

        sr = None
        parts: list[np.ndarray] = []
        starts: list[float] = []
        pos = 0

        def sil(sec: float) -> np.ndarray:
            return np.zeros(int(round(sec * sr)), np.float32)

        for i, r in enumerate(rows):
            pieces = []
            for w in r["wavs"]:          # utterances of one sentence: joined with no added pause
                a, take_sr = audio.read(book.dir / w)
                if sr is None:
                    sr = take_sr
                    lead = sil(pauses.get("chapter_lead_in", 0.0))
                    parts.append(lead)
                    pos += lead.size
                elif take_sr != sr:
                    raise SystemExit(f"assemble: mixed sample rates in {d['href']} ({take_sr} vs {sr})")
                pieces.append(audio.trim(a, sr, acfg["trim_rel_db"], acfg["trim_margin"]))
            a = np.concatenate(pieces)
            starts.append(pos / sr)
            parts.append(a)
            pos += a.size
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                if nxt["para"] == r["para"]:
                    gap = pauses["sentence"]
                elif r["kind"] == "heading":
                    gap = pauses["heading"]
                else:
                    gap = pauses["paragraph"]
                g = sil(gap)
                parts.append(g)
                pos += g.size
        speech_end = pos / sr
        parts.append(sil(TAIL_PAD))
        chapter = np.concatenate(parts)
        chapter, gain_db = audio.normalize(chapter, acfg["target_rms_db"], acfg["peak_db"])
        audio.write(wav_path, chapter, sr)

        begins, prev = [], -10
        for k, st in enumerate(starts):
            b = 0 if k == 0 else max(_ms(st), prev + 10)
            begins.append(b)
            prev = b
        end = _ms(speech_end + LAST_CLIP_EXTRA)
        clips = [(r["id"], begins[k], begins[k + 1] if k + 1 < len(rows) else end) for k, r in enumerate(rows)]
        tl = {"fingerprint": fp, "href": d["href"], "sample_rate": sr, "clips": clips,
              "overlay_ms": end, "file_ms": _ms(chapter.size / sr),
              "rms_db": round(audio.db(audio.rms(chapter)), 2),
              "peak_db": round(audio.db(float(np.abs(chapter).max())), 2), "gain_db": round(gain_db, 2)}
        tl_path.write_text(json.dumps(tl, ensure_ascii=False))
        built += 1
    log(f"assemble: {built} chapters rebuilt, {len(docs) - built} up to date")
