"""PDNC evaluation (pipeline/eval/pdnc.py) on hand-made PDNC-format data (tests/fixtures/pdnc_mini)."""
import json
from pathlib import Path

import pytest

from pipeline import store
from pipeline.cli import main
from pipeline.eval import pdnc
from pipeline.memory.db import norm

MINI = Path(__file__).parent / "fixtures" / "pdnc_mini"

# Our "edition": same story, "to-day" instead of "today" (exercises the fuzzy match).
SENTENCES = [  # (para, kind, text)
    (0, "heading", "CHAPTER I."),
    (1, "body", "“Hello there,” said Tom."),
    (1, "body", "“How are you?”"),
    (2, "body", "“Fine,” said Ann."),
    (3, "body", "“I am quite well to-day, thank you kindly.”"),
    (4, "heading", "CHAPTER II."),
    (5, "body", "“Goodbye,” said Tom."),
]


def make_db(d: Path, speakers: dict[str, tuple[str, float]]):
    """speakers: PDNC quote id -> (our speaker id, confidence) for the quotes in SENTENCES."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "source.epub").touch()                 # Book.get() wants it
    con = store.connect(d)
    for i, (para, kind, t) in enumerate(SENTENCES):
        con.execute("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES (?,?,?,?,?,?,?)",
                    (f"ab001s{i:05d}", 1, i, para, kind, t, t))
    for cid, name, g in (("tom", "Tom", "male"), ("ann", "Ann", "female")):
        con.execute("INSERT INTO character(id, name, gender, status) VALUES (?,?,?, 'active')", (cid, name, g))
    for cid, a in (("tom", "Tom"), ("ann", "Ann"), ("ann", "Lady Smith")):
        con.execute("INSERT INTO alias(character_id, alias, norm) VALUES (?,?,?)", (cid, a, norm(a)))
    # each of our quote pieces belongs to the PDNC quote whose text contains its first word
    first_word = {"Q0": ("Hello", "How"), "Q1": ("Fine",), "Q2": ("I",), "Q3": ("Goodbye",)}
    for s, pcs in pdnc._story_items(con):
        for k, p in enumerate(pcs):
            q = next((q for q, ws in first_word.items() if p.is_quote and p.text.split()[0].strip(",") in ws), None)
            if q in speakers:
                spk, conf = speakers[q]
                con.execute("INSERT INTO utterance(id, sentence_id, seq, start, end, tts_text, is_quote, quote_id, "
                            "speaker_id, confidence, source) VALUES (?,?,?,?,?,?,1,?,?,?,'model')",
                            (f"{s['id']}u{k}", s["id"], k, p.start, p.end, p.text, p.quote_id, spk, conf))
    con.commit()
    return con


BASE = {"Q0": ("tom", 0.9), "Q1": ("tom", 0.5), "Q2": ("ann", 0.7), "Q3": ("tom", 0.9)}


def test_chapter_counter_restarts_per_volume():
    cc = pdnc.ChapterCounter()
    assert [cc.see(n) for n in (1, 2, 3, 1, 2, 1)] == [1, 2, 3, 4, 5, 6]
    assert pdnc.chapter_of_heading("Chapter XIV.") == 14 and pdnc.chapter_of_heading("Chapter: I., II.") is None


def test_alignment_and_metrics(tmp_path):
    con = make_db(tmp_path / "b", BASE)
    res = pdnc.evaluate(con, MINI, MINI / "overrides.yaml", chapters=(1, 1))
    m = res["metrics"]
    assert m["quotes"] == 3 and m["align_pct"] == 100.0       # Q3 is chapter II: out of range
    assert m["fuzzy_quotes"] == 0                              # "to-day" vs "today" is now an exact match
    assert m["missed_quotes"] == 0 and m["spurious_quotes"] == 0
    rows = {r["quote"]: r for r in res["rows"]}
    assert rows["Q0"]["correct"] and not rows["Q1"]["correct"] and rows["Q2"]["correct"]
    assert m["attr_pct"] == 66.7 and m["explicit_pct"] == 50.0 and m["implicit_pct"] == 100.0
    assert res["confusion"][0] == {"gold": "Ann", "pred": "Tom", "n": 1}
    assert m["low_conf_catch_pct"] == 100.0                    # the one error had confidence 0.5
    assert m["main_recall_pct"] == 100.0 and m["false_split"] == 0 and m["gender_pct"] == 100.0
    # overrides split Lady Smith out of Ann, so the alias on our Ann card is now an error
    assert [(e["card"], e["alias"], e["belongs_to"]) for e in res["alias_errors"]] == [("ann", "Lady Smith", "Lady Smith")]
    assert any(e.get("name") == "Lady Smith" for e in res["overrides"]["applied"])
    # Bob's alias "Smith" is not shared, Tom Smith normalizes differently: no ambiguity
    assert res["overrides"]["ambiguous_aliases"] == {}


def test_unmapped_and_unknown_count_as_wrong(tmp_path):
    con = make_db(tmp_path / "b", {**BASE, "Q0": ("unknown", 0.2), "Q2": ("narrator", 0.9)})
    m = pdnc.evaluate(con, MINI, None, chapters=(1, 1))["metrics"]
    assert m["attr_pct"] == 0.0 and m["human_load"] >= 2        # 2 low-confidence (one is unknown too)


def test_without_attribution_only_alignment(tmp_path):
    con = make_db(tmp_path / "b", {})
    res = pdnc.evaluate(con, MINI, None)
    assert res["has_attribution"] is False and res["metrics"]["attr_pct"] is None
    assert res["metrics"]["align_pct"] == 100.0 and res["metrics"]["quotes"] == 4


def test_override_needs_reason(tmp_path):
    bad = tmp_path / "o.yaml"
    bad.write_text("remove_alias:\n  - {character: Tom, alias: Tom}\n")
    with pytest.raises(SystemExit):
        pdnc.load_gold(MINI, bad)


def test_cli_eval_compare_and_history(tmp_path, capsys):
    root = tmp_path / "books"
    make_db(root / "mini", BASE).close()
    fixed = {**BASE, "Q1": ("ann", 0.8)}
    make_db(root / "mini2", fixed).close()
    for bid, tag in (("mini", "v1"), ("mini2", "v2"), ("mini", "v1")):
        main(["--books", str(root), "eval", bid, "--pdnc", str(MINI), "--overrides", str(MINI / "overrides.yaml"),
              "--chapters", "1-1", "--tag", tag, "--note", f"{bid} test"])
    out = root / "mini" / "reports" / "eval"
    assert (out / "v1.json").exists() and (out / "v1.md").exists() and (out / "v1.quotes.csv").exists()
    assert len(list((out / "history").glob("v1.*.json"))) == 1     # same tag again keeps the old result
    assert json.loads((out / "v1.json").read_text())["tag"] == "v1"
    capsys.readouterr()
    main(["--books", str(root), "eval-compare", "mini:v1", "mini2:v2"])
    md = capsys.readouterr().out
    assert "fixed 1, broke 0" in md and "Q1" in md
    assert "66.7" in md and "100.0 (+33.3 ✅)" in md
    main(["--books", str(root), "eval-compare"])
    listing = capsys.readouterr().out
    assert "mini:v1" in listing and "mini2:v2" in listing


def test_mock_pipeline_end_to_end(tmp_path, capsys):
    """Pride fixture chapter I through the mock LLM, then `readaloud eval`: the path works (scores not checked)."""
    from pipeline import config
    from pipeline.book import Book
    from pipeline.cli import run_stages
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "pride_epub2.epub"), "--id", "p"])
    b = Book.get("p", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    assert run_stages(b, cfg, doc_ids=[1], log=lambda *_: None,
                      only=["ingest", "segment", "classify", "characters", "attribute"]) == "ok"
    capsys.readouterr()
    main(["--books", str(root), "eval", "p", "--pdnc", str(Path(__file__).parent / "fixtures" / "pdnc_pride_ch1"),
          "--chapters", "1-1", "--tag", "mock"])
    res = json.loads((b.dir / "reports" / "eval" / "mock.json").read_text())
    assert res["metrics"]["quotes"] == 4 and res["metrics"]["align_pct"] == 100.0
    assert res["has_attribution"] and res["metrics"]["attr_pct"] is not None


def test_oracle_book_mock_attribute(tmp_path, capsys):
    """eval-oracle: PDNC characters as manual cards, chunks marked done, attribute runs and is scored."""
    from pipeline import config
    from pipeline.book import Book
    from pipeline.cli import run_stages
    root = tmp_path / "books"
    pdnc_dir = Path(__file__).parent / "fixtures" / "pdnc_pride_ch1"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "pride_epub2.epub"), "--id", "p"])
    b = Book.get("p", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    run_stages(b, cfg, doc_ids=[1], log=lambda *_: None, only=["ingest", "segment", "classify", "characters"])
    main(["--books", str(root), "eval-oracle", "p", "--pdnc", str(pdnc_dir)])
    o = Book.get("p-oracle", root)
    con = store.connect(o.dir)
    assert {r[0] for r in con.execute("SELECT id FROM character")} == {"mrs_bennet", "mr_bennet"}
    assert con.execute("SELECT COUNT(*) FROM character_event WHERE source != 'manual'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE pass1 != 'done' OR pass2 != 'pending'").fetchone()[0] == 0
    chars = [json.loads(r[0]) for r in con.execute("SELECT characters FROM chunk WHERE characters != '[]'")]
    assert any("mrs_bennet" in c and "mr_bennet" in c for c in chars)
    con.close()
    ocfg = config.load(o.dir)
    ocfg["tts"]["adapter"] = ocfg["llm"]["adapter"] = "mock"
    assert run_stages(o, ocfg, doc_ids=[1], log=lambda *_: None, only=["attribute"]) == "ok"
    main(["--books", str(root), "eval", "p-oracle", "--pdnc", str(pdnc_dir), "--chapters", "1-1", "--tag", "o"])
    res = json.loads((o.dir / "reports" / "eval" / "o.json").read_text())
    assert res["has_attribution"] and res["metrics"]["main_recall_pct"] == 100.0


def test_edition_spellings_of_tonight_align():
    """to-night / tonight are the same word for alignment (Oliver Twist: a short quote that only matched
    a later occurrence dragged the cursor ahead and half the book went unaligned)."""
    from pipeline.eval.pdnc import normtext
    assert normtext("Nothing to-night,")[0] == normtext("Nothing tonight,")[0] == "nothing tonight"
    assert normtext("To-morrow; to-day.")[0] == "tomorrow today"
    assert normtext("go to-wards night")[0] == "go to wards night"          # only those three words


def test_chapter_heading_with_a_title():
    from pipeline.eval.pdnc import chapter_of_heading
    assert chapter_of_heading("CHAPTER IV.") == 4 and chapter_of_heading("Chapter 12") == 12
    assert chapter_of_heading("CHAPTER I. TREATS OF THE PLACE WHERE OLIVER TWIST WAS BORN", heading=True) == 1
    assert chapter_of_heading("Chapter I. of the story was dull", heading=False) is None   # body text is not a heading


def test_pdnc_chapter_headings_with_titles():
    from pipeline.eval.pdnc import PDNC_HEAD
    text = "PART ONE\n\nChapter I: The Bertolini\n\ntext\n\nCHAPTER II.\n\nmore\n\nCHAPTER\n     3\n\nand chapter iv. of the tale\n"
    got = [(m.group(1) or m.group(2)) for m in PDNC_HEAD.finditer(text)]
    assert got == ["I", "II", "3"]              # a sentence that starts a line with "chapter iv. of" is not a heading


def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED_VOICES = {"male": 2, "female": 2}        # Tom's Q0 and Q3, Ann's Q1 and Q2


def test_pdnc_overrides_script_writes_the_gold_voices(tmp_path):
    """scripts/pdnc_overrides.py: a corrected edition takes each quotation's voice from PDNC; hand rows stay."""
    import csv
    from collections import Counter
    po = _load_script("pdnc_overrides")
    d = tmp_path / "b"
    con = make_db(d, {**BASE, "Q1": ("tom", 0.9)})          # Q1 is Ann's: wrong gender before the correction
    fixes, skipped = po.gold_voices(con, MINI, MINI / "overrides.yaml")
    assert Counter(v for v, _ in fixes.values()) == EXPECTED_VOICES and not skipped
    by_hand = next(q for q, (v, _) in fixes.items() if v == "female")
    (d / "attribution_overrides.csv").write_text(f"quote_id,speaker,style,note\n{by_hand},narrator,,my own call\n")
    st = po.write(d, con, fixes)
    rows = list(csv.DictReader((d / "attribution_overrides.csv").open()))
    assert st["kept: corrected by hand already"] == 1 and st["written"] == len(fixes) - 1
    assert {r["quote_id"]: r["speaker"] for r in rows}[by_hand] == "narrator"       # the hand row wins
    assert all(r["note"].startswith("PDNC Q") for r in rows if r["quote_id"] != by_hand)
    st2 = po.write(d, con, fixes)                                                   # running it again changes nothing
    assert list(csv.DictReader((d / "attribution_overrides.csv").open())) == rows and st2["written"] == st["written"]
