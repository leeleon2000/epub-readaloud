"""Deleting work/ (or copying a book dir without it) rebuilds the files but keeps the database:
sentences, utterances and their voices survive (2026-11-15: ingest used to wipe the sentences)."""
import shutil
from pathlib import Path

from pipeline import config, store
from pipeline.book import Book
from pipeline.cli import main, run_stages

FIXTURE = Path(__file__).parent / "fixtures" / "alice_epub2.epub"


def test_rebuild_work_keeps_db(tmp_path):
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(FIXTURE), "--id", "alice"])
    b = Book.get("alice", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    run_stages(b, cfg, until="segment", log=lambda *_: None)
    con = store.connect(b.dir)
    con.execute("UPDATE utterance SET speaker_id='_generic_female' WHERE id=(SELECT MIN(id) FROM utterance)")
    con.commit()
    before = con.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM utterance) FROM sentence").fetchone()
    con.close()
    shutil.rmtree(b.dir / "work")
    logs = []
    run_stages(b, cfg, until="segment", log=logs.append)
    assert any("work files rebuilt" in ln for ln in logs)
    con = store.connect(b.dir)
    assert con.execute("SELECT COUNT(*), (SELECT COUNT(*) FROM utterance) FROM sentence").fetchone() == before
    assert con.execute("SELECT speaker_id FROM utterance WHERE id=(SELECT MIN(id) FROM utterance)").fetchone()[0] \
        == "_generic_female"
    assert (b.seg).exists()
