"""Write attribution_overrides.csv for a book from PDNC's speaker labels (a corrected edition, not the pipeline's own result).

The pipeline never reads PDNC: it is evaluation data. This script is the one deliberate exception, for a book that
is published with its voices corrected by hand. It takes the gold speaker of every PDNC quotation that lines up with
one of our quotations and writes the voice that speaker should have (male / female / narrator) into the book's
attribution_overrides.csv, the file for human corrections, which always wins over the model. Run it after the voices
stage, then run the voices stage again (no model calls) so the corrections are applied:

    readaloud run pride --stages ingest,segment,classify,voices
    python scripts/pdnc_overrides.py pride --pdnc data/pdnc/PrideAndPrejudice
    readaloud run pride --stages voices
    readaloud eval pride --pdnc data/pdnc/PrideAndPrejudice --tag corrected      # wrong gender should now be 0

Rows already in the file that this script did not write (their note does not start with "PDNC ") are kept and win.
Not touched: quotations PDNC does not have (letters, quoted phrases), PDNC quotations that did not line up, and
speakers whose gender PDNC does not give. PDNC is CC-BY 4.0 (Vishnubhotla et al. 2022): credit it wherever the
corrected audiobook is published.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import config, store  # noqa: E402
from pipeline.eval import pdnc  # noqa: E402

VOICES = ("male", "female", "narrator")
OURS = {"_generic_male": "male", "_generic_female": "female"}
FIELDS = ["quote_id", "speaker", "style", "note"]


def gold_voices(con, pdnc_dir: Path, overrides: Path | None, cfg: dict | None = None) -> tuple[dict, dict]:
    """({our quote id: (voice, note)}, counts). One of our quotations can cover several PDNC quotations; it is
    corrected only when they all want the same voice."""
    res = pdnc.evaluate(con, pdnc_dir, overrides, None, cfg=cfg)
    gold = {r["quote"]: r for r in res.get("voice_rows", [])}
    want, st = defaultdict(set), Counter()
    notes = {}
    for r in res["rows"]:
        g = gold.get(r["quote"])
        if not r.get("our_quote"):
            st["PDNC quotations that did not line up"] += 1
        elif g is None or g["gold_voice"] not in VOICES:
            st["speaker's gender not in PDNC"] += 1
        else:
            want[r["our_quote"]].add(g["gold_voice"])
            notes.setdefault(r["our_quote"], f"PDNC {r['quote']} {r['gold']}")
    out = {}
    for q, vs in want.items():
        if len(vs) == 1:
            out[q] = (next(iter(vs)), notes[q])
        else:
            st["our quotation covers PDNC quotations with different voices"] += 1
    return out, st


def write(book_dir: Path, con, corrections: dict) -> Counter:
    """Merge into attribution_overrides.csv and say what changes against the voices now in the book."""
    p = book_dir / "attribution_overrides.csv"
    kept = []
    if p.exists():
        with p.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if (r.get("quote_id") or "").strip()
                    and not (r.get("note") or "").startswith("PDNC ")]
    by_hand = {r["quote_id"].strip() for r in kept if not r["quote_id"].startswith("#")}
    now = {}
    for r in con.execute("SELECT quote_id, speaker_id FROM utterance WHERE is_quote=1 AND quote_id IS NOT NULL"):
        now.setdefault(r[0], OURS.get(r[1], "narrator"))
    st = Counter()
    rows = [{k: (r.get(k) or "") for k in FIELDS} for r in kept]
    for q in sorted(corrections):
        voice, note = corrections[q]
        if q in by_hand:
            st["kept: corrected by hand already"] += 1
            continue
        if q not in now:
            st["skipped: quotation not in the book"] += 1
            continue
        rows.append({"quote_id": q, "speaker": voice, "style": "", "note": note})
        was = now[q]
        st["written"] += 1
        st["same voice as before" if was == voice else
           f"{'narrator' if was == 'narrator' else 'wrong gender'} -> {voice}" if voice != "narrator" else
           "voiced -> narrator"] += 1
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("book")
    ap.add_argument("--pdnc", required=True, help="data/pdnc/<Book> (scripts/fetch_pdnc.sh)")
    ap.add_argument("--books", default=str(ROOT / "books"), help="books root (default ./books)")
    ap.add_argument("--overrides", help="PDNC fixes file (default data/pdnc_overrides/<Book>.yaml)")
    a = ap.parse_args()
    book_dir, src = Path(a.books) / a.book, Path(a.pdnc)
    if not (book_dir / "project.db").exists():
        raise SystemExit(f"{book_dir}: no project.db (run the book to the voices stage first)")
    ov = Path(a.overrides) if a.overrides else ROOT / "data" / "pdnc_overrides" / f"{src.name}.yaml"
    con = store.connect(book_dir)
    corrections, skipped = gold_voices(con, src, ov if ov.exists() else None, cfg=config.load(book_dir))
    st = write(book_dir, con, corrections)
    print(f"{book_dir / 'attribution_overrides.csv'}: {st['written']} corrections from PDNC")
    for k, v in sorted(st.items()):
        if k != "written":
            print(f"  {k}: {v}")
    for k, v in sorted(skipped.items()):
        print(f"  not corrected, {k}: {v}")
    print(f"next: readaloud run {a.book} --stages voices   (applies the file; no model calls)")


if __name__ == "__main__":
    main()
