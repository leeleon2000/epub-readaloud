"""Chunk modes (docs/DESIGN.md): `words` is unchanged, `scene` never cuts between two
speech paragraphs."""
import pytest

from pipeline import store
from pipeline.chunks import ensure_chunks

# (paragraph text, is speech); every paragraph is 4 words
PARAS = [("The house was quiet.", 0), ("Rain fell all night.", 0),
         ("“Come in,” said Anne.", 1), ("“Thank you, I will.”", 1), ("“Sit down by me.”", 1),
         ("“Gladly,” said Tom then.", 1), ("They sat in silence.", 0), ("The fire burned low.", 0),
         ("“Well,” she said quietly.", 1), ("Nobody answered her there.", 0)]


def _book(tmp_path):
    con = store.connect(tmp_path)
    con.execute("INSERT INTO doc(id, href, manifest_id, has_speech) VALUES (1, 'a.xhtml', 'a', 1)")
    for i, (t, _) in enumerate(PARAS):
        con.execute("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES (?,1,?,?,'body',?,?)",
                    (f"ab001s{i:05d}", i, i, t, t))
    con.commit()
    return con


def _cuts(con):
    rows = con.execute("SELECT seq_from, seq_to FROM chunk ORDER BY seq_from").fetchall()
    return [(r["seq_from"], r["seq_to"]) for r in rows]


def test_words_mode_unchanged(tmp_path):
    con = _book(tmp_path)
    ensure_chunks(con, [1], 8)                       # default mode: cut once 8 words are reached
    assert _cuts(con) == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def test_scene_mode_waits_for_the_conversation_to_end(tmp_path):
    con = _book(tmp_path)
    ensure_chunks(con, [1], 8, "scene", "double")
    cuts = _cuts(con)
    assert cuts == [(0, 1), (2, 5), (6, 7), (8, 9)]   # words mode cuts after 3: inside the conversation
    speech = {i for i, (_, s) in enumerate(PARAS) if s}
    for (_, a), (b, _) in zip(cuts, cuts[1:]):
        assert not (a in speech and b in speech)
    con.execute("DELETE FROM chunk")
    ensure_chunks(con, [1], 8, "scene", "double")    # same input, same chunks
    assert _cuts(con) == cuts


def test_scene_mode_needs_style_and_a_known_mode(tmp_path):
    con = _book(tmp_path)
    with pytest.raises(ValueError):
        ensure_chunks(con, [1], 8, "scene")
    with pytest.raises(ValueError):
        ensure_chunks(con, [1], 8, "pages", "double")
