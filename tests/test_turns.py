"""Two-speaker alternation rule (pipeline/attrib_turns.py, docs/DESIGN.md)."""
import sqlite3

from pipeline import attrib_turns as T
from pipeline import store


def turn(para, speaker, conf, doc=1):
    return {"doc": doc, "para": para, "quotes": [{"speaker": speaker, "confidence": conf}]}


def expected(turns):
    return {turns[i]["para"]: who for i, who in T.decide(turns)}


def test_two_tagged_speakers_decide_the_untagged_turns():
    # Elizabeth (tag), ?, ?, ?, Darcy (tag) on paragraphs 1..5: the model swapped the middle
    turns = [turn(1, "elizabeth", 0.95), turn(2, "elizabeth", 0.5), turn(3, "darcy", 0.5),
             turn(4, "darcy", 0.7), turn(5, "elizabeth", 0.95)]
    assert expected(turns) == {1: "elizabeth", 2: "darcy", 3: "elizabeth", 4: "darcy", 5: "elizabeth"}


def test_one_tagged_speaker_uses_the_models_other_voice():
    turns = [turn(1, "jane", 0.95), turn(2, "elizabeth", 0.5), turn(3, "elizabeth", 0.5), turn(4, "elizabeth", 0.5)]
    assert expected(turns) == {1: "jane", 2: "elizabeth", 3: "jane", 4: "elizabeth"}


def test_runs_the_rule_leaves_alone():
    # no reliable speaker at all
    assert T.decide([turn(1, "a", 0.5), turn(2, "b", 0.5), turn(3, "a", 0.7)]) == []
    # a third voice: tags name three people
    assert T.decide([turn(1, "a", 0.95), turn(2, "b", 0.95), turn(3, "c", 0.95), turn(4, "a", 0.5)]) == []
    # the model heard a third person in an untagged turn
    assert T.decide([turn(1, "a", 0.95), turn(2, "b", 0.95), turn(3, "c", 0.5), turn(4, "b", 0.5)]) == []
    # the same tagged speaker on both parities (someone speaks twice in a row)
    assert T.decide([turn(1, "a", 0.95), turn(2, "a", 0.95), turn(3, "b", 0.5)]) == []
    # too short, or broken by a paragraph without speech, or another doc
    assert T.decide([turn(1, "a", 0.95), turn(2, "b", 0.5)]) == []
    assert T.decide([turn(1, "a", 0.95), turn(2, "b", 0.5), turn(4, "b", 0.5), turn(5, "a", 0.5)]) == []
    assert T.decide([turn(1, "a", 0.95), turn(2, "b", 0.5), turn(3, "b", 0.5, doc=2)]) == []


def test_unknown_is_not_a_third_voice():
    turns = [turn(1, "a", 0.95), turn(2, "b", 0.95), turn(3, "unknown", 0.0), turn(4, "b", 0.5)]
    assert expected(turns) == {1: "a", 2: "b", 3: "a", 4: "b"}


def _book(rows):
    """rows: (para, speaker, confidence, source) -> a db with one quote per paragraph."""
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    con.execute("INSERT INTO doc(id, href, manifest_id) VALUES (1, 'a.xhtml', 'a')")
    for n, (para, spk, conf, src) in enumerate(rows):
        sid = f"ab001s{n:05d}"
        con.execute("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES (?,?,?,?,?,?,?)",
                    (sid, 1, n, para, "body", "x", "x"))
        con.execute("INSERT INTO utterance(id, sentence_id, seq, start, end, tts_text, is_quote, quote_id, speaker_id, "
                    "confidence, source, model_speaker, model_confidence) VALUES (?,?,0,0,1,'x',1,?,?,?,?,?,?)",
                    (sid + "u0", sid, f"q{n}", spk, conf, src, None if src == "manual" else spk,
                     None if src == "manual" else conf))
    return con


def _state(con):
    return [(r["speaker_id"], r["confidence"], r["source"]) for r in con.execute("SELECT * FROM utterance ORDER BY id")]


ROWS = [(1, "elizabeth", 0.95, "model"), (2, "elizabeth", 0.5, "model"), (3, "elizabeth", 0.5, "model"),
        (4, "darcy", 0.7, "model"), (5, "elizabeth", 0.95, "model")]


def test_apply_fixes_confirms_and_is_repeatable():
    con = _book(ROWS)
    st = T.apply(con, [1], "fix")
    assert st["fixed"] == 1 and st["agree"] == 2
    want = [("elizabeth", 0.95, "model"), ("darcy", 0.5, "turns"), ("elizabeth", 0.8, "model"),
            ("darcy", 0.8, "model"), ("elizabeth", 0.95, "model")]
    assert _state(con) == want
    # the model's own answer is kept, and a second run starts from it
    assert con.execute("SELECT model_speaker, model_confidence FROM utterance WHERE quote_id='q1'").fetchone()[:] == ("elizabeth", 0.5)
    T.apply(con, [1], "fix")
    assert _state(con) == want
    # flag keeps the model's answer and marks it; off restores everything
    T.apply(con, [1], "flag")
    assert _state(con)[1] == ("elizabeth", 0.3, "model")
    T.apply(con, [1], "off")
    assert _state(con) == [(s, c, "model") for _, s, c, _ in ROWS]


def test_manual_rows_are_anchors_and_never_changed():
    rows = [(1, "jane", 1.0, "manual"), (2, "jane", 0.5, "model"), (3, "elizabeth", 1.0, "manual"), (4, "jane", 0.5, "model")]
    con = _book(rows)
    T.apply(con, [1], "fix")
    # paragraph 3 is manual and breaks the alternation jane/?/jane: the manual rows disagree with a simple
    # exchange, so nothing is touched
    assert _state(con) == [(s, c, src) for _, s, c, src in rows]
    rows = [(1, "jane", 1.0, "manual"), (2, "jane", 0.5, "model"), (3, "jane", 0.5, "model")]
    con = _book(rows)
    T.apply(con, [1], "fix")
    assert _state(con) == [("jane", 1.0, "manual"), ("jane", 0.5, "model"), ("jane", 0.5, "model")]   # no second voice known


def test_a_tag_conflict_is_not_raised():
    # paragraph 2 agrees with the alternation but its own speech tag says otherwise (0.3): leave it flagged
    con = _book([(1, "a", 0.95, "model"), (2, "b", 0.3, "model"), (3, "a", 0.5, "model"), (4, "b", 0.95, "model")])
    T.apply(con, [1], "fix")
    assert _state(con)[1] == ("b", 0.3, "model") and _state(con)[2] == ("a", 0.8, "model")


def test_a_speaker_who_goes_on_starts_a_new_run():
    # “…,” he replied. / Elizabeth was too embarrassed to say a word. After a pause, her companion added, “…”
    turns = [turn(1, "elizabeth", 0.95), turn(2, "darcy", 0.95), turn(3, "darcy", 0.5), turn(4, "elizabeth", 0.5)]
    assert expected(turns)[3] == "elizabeth"                      # without the cue the rule would flip it
    turns[2]["narration"] = "After a short pause, her companion added,"
    assert T.decide(turns) == []                                  # two short runs, neither long enough
