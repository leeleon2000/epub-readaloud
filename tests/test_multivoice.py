"""Multi-voice flow on Alice chapter 1 (doc 2) with mock LLM + mock TTS."""
from pathlib import Path

import pytest
import yaml

from pipeline import config, store
from pipeline.book import Book
from pipeline.cli import main, run_stages

FIXTURE = Path(__file__).parent / "fixtures" / "alice_epub2.epub"


@pytest.fixture(scope="module")
def book(tmp_path_factory):
    root = tmp_path_factory.mktemp("books")
    main(["--books", str(root), "init", str(FIXTURE), "--id", "alice"])
    b = Book.get("alice", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = "mock"
    cfg["llm"]["adapter"] = "mock"
    cfg["attribute"]["version"] = "v2"                   # these tests exercise the v2 stages; the default is v3
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000   # the mock's crude name extraction would spike
    run_stages(b, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    return b, cfg


def test_characters_and_cast(book):
    b, _ = book
    con = store.connect(b.dir)
    names = {r["id"] for r in con.execute("SELECT id FROM character WHERE status!='merged'")}
    assert "alice" in names
    cast = yaml.safe_load((b.dir / "cast.yaml").read_text())
    assert "alice" in cast["characters"] and cast["narrator"]["voice"]
    voices = [e["voice"] for e in cast["characters"].values()]
    assert len(set(voices)) == len(voices)          # no two characters share a design
    assert (b.dir / "voices" / "alice.wav").exists()


def test_quotes_split_and_attributed(book):
    b, _ = book
    con = store.connect(b.dir)
    u = con.execute("SELECT * FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                    "WHERE s.doc_id=2 AND u.tts_text LIKE 'Well!%'").fetchone()
    assert u["is_quote"] == 1 and u["speaker_id"] == "alice"     # “Well!” thought Alice
    n_quote = con.execute("SELECT COUNT(*) FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                          "WHERE s.doc_id=2 AND u.is_quote=1").fetchone()[0]
    assert n_quote > 20
    # other chapters were not touched by the LLM passes
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id != 2").fetchone()[0] == 0


def test_multiple_voices_synthesized_and_assembled(book):
    b, _ = book
    con = store.connect(b.dir)
    n_voices = con.execute("SELECT COUNT(DISTINCT t.key) FROM take t").fetchone()[0]
    assert n_voices > 0
    assert (b.chapters / "002.wav").exists()
    assert not (b.chapters / "003.wav").exists()


def test_resume_and_overrides(book):
    b, cfg = book
    con = store.connect(b.dir)
    qid = con.execute("SELECT u.quote_id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                      "WHERE s.doc_id=2 AND u.is_quote=1 LIMIT 1").fetchone()[0]
    (b.dir / "attribution_overrides.csv").write_text(f"quote_id,speaker,style,note\n{qid},white_rabbit,,test\n")
    logs = []
    run_stages(b, cfg, until="attribute", doc_ids=[2], log=logs.append)
    assert any("0 pending" in ln for ln in logs if ln.startswith("characters:"))
    assert any("0 to do" in ln for ln in logs if ln.startswith("attribute:"))
    assert any("applied" in ln for ln in logs)
    con = store.connect(b.dir)
    rows = con.execute("SELECT speaker_id, source FROM utterance WHERE quote_id=?", (qid,)).fetchall()
    assert all(r["speaker_id"] == "white_rabbit" and r["source"] == "manual" for r in rows)
