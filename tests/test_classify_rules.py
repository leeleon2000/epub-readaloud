"""Rule-only classification cases (no LLM)."""
from pipeline.stages import classify


def _sec(i, title, texts):
    return {"doc_id": i, "seq_from": 0, "title": title, "texts": texts, "id": f"d{i:03d}s00000"}


def test_copyright_page_is_boilerplate():
    """2026-11-15: a publisher's copyright page was read aloud, and Drama 3 ran away on the ISBNs."""
    page = _sec(1, "THIS IS AN EXAMPLE PRESS BOOK", [
        "THIS IS AN EXAMPLE PRESS BOOK", "Copyright © 2012 by the estate of Jane Example", "All rights reserved.",
        "The Library of Congress has cataloged the earlier printing as follows:",
        "ISBN 978-0-00-000000-2 (alk. paper)", "PS0000.E00X00 2012"])
    story = _sec(2, "THE THIEF", ["He said the copyright was his.", "All rights reserved, she laughed."] * 3)
    long_story = _sec(3, "THE LAWYER", ["“Copyright © 1952,” he read. “All rights reserved.”"]
                      + ["The trial went on."] * 80)
    secs = [page, story, long_story]
    classify._rules(secs)
    assert page["category"] == "boilerplate"
    assert "category" not in story and "category" not in long_story


def test_only_story_carries_over_to_a_headingless_document():
    """2026-11-16, an omnibus with no heading tags: front matter / copyright 'continued' over the whole book."""
    secs = [_sec(1, "", ["Copyright © 1980 by John Example", "All rights reserved.", "ISBN 0-000-00000-0"]),
            _sec(2, "", ["For my sister"]),
            _sec(3, "CHAPTER I", ["The morning began like any other, which should have warned me."]),
            _sec(4, "", ["We went on down the stair."])]
    classify._rules(secs)
    assert secs[0]["category"] == "boilerplate"
    assert "category" not in secs[1]                     # left to the model, not "continues" the copyright page
    assert secs[2]["category"] == "story" and secs[3]["category"] == "story"


def test_illustration_captions_of_an_illustrated_edition_are_not_read(tmp_path):
    """Pride and Prejudice (the 1894 illustrated edition): every picture has a quoted caption and a copyright
    credit inside <div class="caption">. Before 2026-11-18 the ones not next to a heading were read as story."""
    from pathlib import Path

    from pipeline import store
    from pipeline.cli import main
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "pride_epub2.epub"), "--id", "p"])
    main(["--books", str(root), "run", "p", "--stages", "ingest,segment,classify", "--llm", "mock"])
    con = store.connect(root / "p")
    spoken = [r["text"] for r in con.execute("SELECT text FROM sentence WHERE speak=1")]
    assert spoken and not any("Copyright 1894" in t for t in spoken)
    n = con.execute("SELECT COUNT(*) FROM sentence WHERE kind='caption'").fetchone()[0]
    assert n >= 40 and con.execute("SELECT COUNT(*) FROM sentence WHERE kind='caption' AND speak=1").fetchone()[0] == 0
    first = con.execute("SELECT kind, speak FROM sentence WHERE doc_id=2 ORDER BY seq LIMIT 1").fetchone()
    assert tuple(first) == ("caption", 0)                       # the picture at the top of the second file
    assert "It is a truth universally acknowledged" in " ".join(spoken).replace("IT is", "It is")


def test_single_sentences_can_be_skipped_in_sections_yaml(tmp_path):
    """sections.yaml `skip:` keeps one line from being read without changing its section (a printer's imprint
    after the last line of the novel shares the last chapter's section)."""
    from pathlib import Path

    import pytest

    from pipeline import store
    from pipeline.cli import main
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "alice_epub2.epub"), "--id", "a"])
    main(["--books", str(root), "run", "a", "--stages", "ingest,segment,classify", "--llm", "mock"])
    con = store.connect(root / "a")
    sid, sec = tuple(con.execute("SELECT s.id, s.section_id FROM sentence s JOIN section x ON x.id=s.section_id "
                                 "WHERE s.speak=1 AND x.category='story' ORDER BY s.doc_id DESC, s.seq DESC LIMIT 1").fetchone())
    (root / "a" / "sections.yaml").write_text(f"skip: [{sid}]\n")
    main(["--books", str(root), "run", "a", "--stages", "classify", "--llm", "mock"])
    assert tuple(con.execute("SELECT speak, section_id FROM sentence WHERE id=?", (sid,)).fetchone()) == (0, sec)
    main(["--books", str(root), "sections", "a", "--set", f"{sec}=story"])        # --set keeps the skip list
    assert "skip" in (root / "a" / "sections.yaml").read_text()
    assert con.execute("SELECT speak FROM sentence WHERE id=?", (sid,)).fetchone()[0] == 0
    con.close()                                   # the failing run must not find the database held by this test
    (root / "a" / "sections.yaml").write_text("skip: [ab999s00000]\n")
    with pytest.raises(SystemExit):
        main(["--books", str(root), "run", "a", "--stages", "classify", "--llm", "mock"])
