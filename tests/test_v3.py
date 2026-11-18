"""Attribution v3: male / female / narrator (docs/DESIGN.md)."""
from collections import Counter
from pathlib import Path

from pipeline import config, store
from pipeline.adapters.llm.mock import MockLLM
from pipeline.book import Book
from pipeline.eval import pdnc
from pipeline.stages import voices
from pipeline.v3 import FEMALE, MALE, NARRATOR, decide as D, genders as G
from pipeline.v3.tags import find_tag, not_dialogue

from test_eval_pdnc import MINI, make_db


# ---------------------------------------------------------------- tags

def test_speech_tags():
    t = lambda a="", b="", m="": find_tag(a, b, m)                       # noqa: E731
    assert t(" said the Jew to Oliver.") == ("the Jew", "np") or t(" said the Jew to Oliver.").text == "the Jew"
    assert t(" said Mr. Darcy.").kind == "name" and t(" said Mr. Darcy.").text == "Mr. Darcy"
    assert t(" replied she,").text == "she" and t(" replied she,").kind == "pronoun"
    assert t(" cried her mother.").text == "her mother"
    assert t(" I said.").kind == "first_person" and t(" said I.").kind == "first_person"
    assert t(" the old gentleman replied, smiling.").text == "the old gentleman"
    assert t(b="Then Bill said:").text == "Bill"
    assert t(b="He said nothing.") is None                     # a full sentence before the quote is no tag
    assert t(" they cried.").kind == "group"
    assert t(m=" said Elizabeth, ").text == "Elizabeth"         # “…,” said Elizabeth, “<this half>”


def test_not_dialogue():
    assert not_dialogue("He came down to see the place", True, "", "") == "caption"
    assert not_dialogue("tolerable", False, "She is ", " but not handsome") == "quoted phrase"
    assert not_dialogue("I am quite well.", True, "", "") is None
    assert not_dialogue("Yes,", False, "He answered, ", " and left") is None     # introduced by a comma


# ---------------------------------------------------------------- genders

def test_title_and_noun():
    assert G.title_of("Mrs. Bennet ") == "female" and G.title_of("Sir William ") == "male"
    assert G.title_of("Elizabeth ") is None
    assert G.noun_gender("the old gentleman") == "male" and G.noun_gender("her mother") == "female"
    assert G.noun_gender("the Jew") is None                    # learned from the book, not built in


def test_pronoun_learning_subjects_only():
    paras = [[("Elizabeth was silent; but she smiled.", False), ("She went in.", False)],
             [("Then taking the arm of Mr. Darcy, she walked on.", False)],     # Darcy is not the subject
             [("Elizabeth looked at him, and she laughed.", False)]]
    v = G.learn_pronouns(paras, {"Elizabeth", "Mr. Darcy"})
    assert v["elizabeth"] == Counter(female=3) and "mr. darcy" not in v
    # “Yes,” said Elizabeth. “No,” said he: "he" is not Elizabeth
    v = G.learn_pronouns([[("said Elizabeth.", True), ("said he.", True)]], {"Elizabeth"})
    assert not v.get("elizabeth")


def test_table_sources_and_manual(tmp_path):
    phrases = Counter({"Mr. Darcy": 5, "Darcy": 3, "Miss Elizabeth": 1, "Elizabeth": 9, "Bennet": 2,
                       "Mr. Bennet": 2, "Mrs. Bennet": 2, "the girl": 2, "the Jew": 4, "Zed": 1})
    paras = [[("The Jew nodded, and he grinned.", False)]] * 4
    table, cached = G.build(phrases, paras, {"zed": "female"}, {}, llm=MockLLM())
    e = table.entries
    assert e["mr. darcy"][0] == "male" and e["darcy"] == ("male", "with title")
    assert e["elizabeth"] == ("female", "with title")             # Miss Elizabeth
    assert e["bennet"][0] == "unknown"                           # Mr. and Mrs. Bennet
    assert e["the girl"] == ("female", "noun") and e["the jew"][0] == "male"
    assert e["zed"] == ("female", "manual")
    G.save(tmp_path, table, {"zed": "female"}, phrases)
    assert G.load_manual(tmp_path) == {"zed": "female"}


# ---------------------------------------------------------------- decide

def q(i, para, **kw):
    return {"qid": f"q{i}", "doc": 1, "para": para, "narration": "", "rule": None, "tag_kind": None,
            "tag_gender": None, "tag_gender_source": "", "who": None, "who_gender": None, "who_table_gender": None,
            "voice_ans": None, "first_person": False, **kw}


def test_tags_and_paragraph():
    qs = [q(1, 1, tag_kind="name", tag_gender="female", tag_gender_source="title"), q(2, 1),
          q(3, 2, rule="caption")]
    D.decide(qs)
    assert [x["speaker"] for x in qs] == [FEMALE, FEMALE, NARRATOR]
    assert qs[1]["source"] == "paragraph"


def test_model_mismatch_falls_back():
    qs = [q(1, 1, who="Darcy", who_gender="female", who_table_gender="male")]
    D.decide(qs, ask="who", scene=False)
    assert qs[0]["speaker"] == NARRATOR and qs[0]["conf"] == 0.2


def test_alternation_and_threshold():
    he = dict(tag_kind="pronoun", tag_gender="male")
    she = dict(tag_kind="pronoun", tag_gender="female")
    qs = [q(1, 1, **he), q(2, 2, **she),
          q(3, 3, who="Bob", who_gender="male", who_table_gender="male"),         # agrees with the turn
          q(4, 4, who="Bob", who_gender="male", who_table_gender="male"),         # contradicts it
          q(5, 5, who="Bob", who_gender="male", who_table_gender="male")]
    D.decide(qs, ask="who")
    assert qs[2]["speaker"] == MALE and qs[2]["source"] == "scene"
    assert qs[3]["speaker"] == NARRATOR and qs[3]["conf"] == 0.3
    # model alone (no scene rules) stays under the default threshold
    qs2 = [dict(x, cls=None) for x in qs[2:3]]
    D.decide(qs2, ask="who", scene=False)
    assert qs2[0]["conf"] == 0.65 and qs2[0]["speaker"] == NARRATOR
    D.decide(qs2, ask="who", scene=False, threshold=0.6)
    assert qs2[0]["speaker"] == MALE


def test_same_gender_scene_and_both():
    she = dict(tag_kind="pronoun", tag_gender="female")
    qs = [q(1, 1, **she), q(2, 2, who="Jane", who_gender="female", who_table_gender="female", voice_ans="female"),
          q(3, 3, who="unknown", who_gender="unknown", voice_ans="female")]
    D.decide(qs, ask="both")
    assert [x["speaker"] for x in qs] == [FEMALE, FEMALE, FEMALE]
    qs = [q(1, 1, who="Jane", who_gender="female", who_table_gender="female", voice_ans="male")]
    D.decide(qs, ask="both", scene=False)
    assert qs[0]["conf"] == 0.3 and qs[0]["speaker"] == NARRATOR


def test_first_person():
    qs = [q(1, 1, tag_kind="first_person"), q(2, 2, who="the narrator (I)", first_person=True)]
    D.decide(qs, ask="who")
    assert [x["speaker"] for x in qs] == [NARRATOR, NARRATOR] and qs[1]["cls"] == "I"


# ---------------------------------------------------------------- stage + eval

def test_voices_stage_and_eval(tmp_path):
    con = make_db(tmp_path / "b", {})
    con.execute("INSERT INTO doc(id, href, manifest_id, has_speech) VALUES (1, 'x.xhtml', 'x', 1)")
    con.commit()
    b = Book.get("b", tmp_path)
    cfg = config.load(b.dir)
    cfg["voices"] = {"ask": "both"}
    voices.run(b, con, cfg, log=lambda *_: None, llm=MockLLM())
    rows = {r["tts_text"]: r for r in con.execute("SELECT * FROM utterance WHERE is_quote=1")}
    assert rows["Hello there,"]["speaker_id"] == NARRATOR     # said Tom, but nothing tells Tom's gender
    (b.dir / "genders.yaml").write_text("manual:\n  Tom: male\n  Ann: female\n")
    voices.run(b, con, cfg, log=lambda *_: None, llm=MockLLM())
    rows = {r["tts_text"]: r for r in con.execute("SELECT * FROM utterance WHERE is_quote=1")}
    assert rows["Hello there,"]["speaker_id"] == MALE                    # said Tom
    assert rows["How are you?"]["speaker_id"] == MALE                    # the tag between two quotes counts for both
    assert rows["Fine,"]["speaker_id"] in (FEMALE, NARRATOR)
    assert (b.dir / "genders.yaml").exists()
    res = pdnc.evaluate(con, MINI, MINI / "overrides.yaml")
    m = res["metrics"]
    assert res["voices"]["mode"] == "v3"
    assert m["v3_mis"] == 0 and m["v3_voiced_pct"] is not None
    assert len(res["voices"]["curve"]) == len(pdnc.CURVE)
    voices.run(b, con, cfg, log=lambda *_: None, llm=MockLLM())         # rerun: answers cached, same result
    assert con.execute("SELECT COUNT(*) FROM v3_quote WHERE who IS NULL AND tag_kind IS NULL AND rule IS NULL") \
        .fetchone()[0] == 0


def test_eval_v2_book_three_class(tmp_path):
    from test_eval_pdnc import BASE
    con = make_db(tmp_path / "b", BASE)
    res = pdnc.evaluate(con, MINI, MINI / "overrides.yaml", chapters=(1, 1))
    m = res["metrics"]
    assert res["voices"]["mode"].startswith("v2")
    # Q1 (Ann) was given to Tom: wrong gender; Q0 Tom, Q2 Ann right
    assert m["v3_mis"] == 1 and m["v3_voiced_pct"] == 100.0


def test_scene_dependent_descriptions():
    assert G.scene_dependent("the other") and G.scene_dependent("his companion") and G.scene_dependent("the child")
    assert not G.scene_dependent("the Jew") and not G.scene_dependent("Mr. Darcy")
    table, _ = G.build(Counter({"his companion": 3, "the other": 2, "her mother": 2}),
                       [[("His companion nodded, and he left.", False)]] * 5, {}, {}, llm=MockLLM())
    assert table.get("his companion") == "unknown" and table.get("the other") == "unknown"
    assert table.get("her mother") == "female"                   # a gendered noun still decides


def test_narrator_guard_and_name():
    he = dict(tag_kind="pronoun", tag_gender="male")
    qs = [q(1, 1, **he), q(2, 2, who="Bill", who_gender="male", who_table_gender="male", first_person=True),
          q(3, 3, **he)]
    D.decide(qs, ask="who", narrator_gender="male")
    assert qs[1]["source"] == "model"                            # no same-gender shortcut: Jake may be speaking
    D.decide(qs, ask="who")
    assert qs[1]["source"] == "scene"
    # the model naming the narrator ("Jake", narrator in the table) is the narrator's voice
    qs = [q(1, 1, who="Jake", who_gender="male", who_table_gender="narrator", first_person=True)]
    D.decide(qs, ask="who")
    assert qs[0]["cls"] == "I" and qs[0]["speaker"] == NARRATOR


def test_vocatives_in_crowd():
    tag = lambda n, g: dict(tag_kind="name", tag_text=n, tag_gender=g, tag_gender_source="title")   # noqa: E731
    qs = [q(1, 1, **tag("Mike", "male")), q(2, 2, **tag("Brett", "female")), q(3, 3, **tag("Bill", "male")),
          q(4, 4, **tag("Cohn", "male"), vocatives=["mike"]),
          q(5, 5, who="Mike", who_gender="male", who_table_gender="male", voice_ans="male"),      # answers Cohn
          q(6, 6, who="Brett", who_gender="female", who_table_gender="female", voice_ans="female",
            vocatives=["brett"])]                                                                # names herself
    D.decide(qs, crowd=4)
    assert qs[4]["source"] == "vocative" and qs[4]["speaker"] == MALE
    assert qs[5]["conf"] == 0.3 and qs[5]["speaker"] == NARRATOR
    D.decide(qs, crowd=4, vocative=False)
    assert qs[4]["speaker"] == NARRATOR and qs[4]["conf"] == 0.6


def test_default_flow_is_v3(tmp_path):
    """The default configuration (2026-11-14) reads a book with three voices: no roster and no attribute
    stage run, and every quote is the narrator's, the generic man's or the generic woman's."""
    import yaml
    from pipeline.cli import main, run_stages
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "alice_epub2.epub"), "--id", "a"])
    b = Book.get("a", root)
    cfg = config.load(b.dir)
    assert cfg["attribute"]["version"] == "v3"
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    logs = []
    assert run_stages(b, cfg, doc_ids=[2], until="assemble", log=logs.append) == "ok"
    ran = {ln.split(":")[0] for ln in logs if ":" in ln}
    assert "voices" in ran and not ran & {"collect", "roster", "characters", "attribute"}
    con = store.connect(b.dir)
    assert con.execute("SELECT COUNT(*) FROM character").fetchone()[0] == 0
    speakers = {r[0] for r in con.execute("SELECT DISTINCT speaker_id FROM utterance")}
    assert speakers <= {NARRATOR, MALE, FEMALE} and NARRATOR in speakers
    cast = yaml.safe_load((b.dir / "cast.yaml").read_text())
    assert cast["narrator"]["voice"] and {MALE, FEMALE} <= set(cast["characters"])
    assert (b.dir / "work" / "chapters" / "002.wav").exists()


def test_second_answer_from_notes_version():
    """voices.second: notes (docs/DESIGN.md): the second answer is who2, read like who."""
    from pipeline.v3.decide import model_class
    base = {"who": "Lucy", "who_gender": "female", "who_table_gender": "female", "voice_ans": "male",
            "second": "notes", "first_person": False}
    agree = dict(base, who2="Miss Honeychurch", who2_gender="female", who2_table_gender="female")
    assert model_class(agree, "both") == ("female", 0.8)          # voice_ans (male) is not looked at
    assert model_class(dict(base, second="voice"), "both") == ("female", 0.3)
    differ = dict(base, who2="Cecil", who2_gender="male", who2_table_gender="male")
    assert model_class(differ, "both") == ("female", 0.3)
    only_first = dict(base, who2="unknown", who2_gender="unknown")
    assert model_class(only_first, "both") == ("female", 0.65)
    assert model_class(dict(base, who2="another man", who2_gender="male"), "voice") == ("male", 0.6)


def test_voices_second_notes_with_mock(tmp_path):
    """second: notes runs with the mock LLM: notes and who2 are written, the direct voice question is not
    asked, and a rerun asks the model nothing. second: voice is the flow as before."""
    con = make_db(tmp_path / "b", {})
    con.execute("INSERT INTO doc(id, href, manifest_id, has_speech) VALUES (1, 'x.xhtml', 'x', 1)")
    con.commit()
    b = Book.get("b", tmp_path)
    cfg = config.load(b.dir)
    cfg["voices"] = {"ask": "both", "second": "notes"}
    llm = MockLLM()
    voices.run(b, con, cfg, log=lambda *_: None, llm=llm)
    open_q = "tag_kind IS NULL AND rule IS NULL"
    assert con.execute(f"SELECT COUNT(*) FROM v3_quote WHERE {open_q}").fetchone()[0] > 0
    assert con.execute(f"SELECT COUNT(*) FROM v3_quote WHERE {open_q} AND (who IS NULL OR who2 IS NULL)").fetchone()[0] == 0
    assert con.execute(f"SELECT COUNT(*) FROM v3_quote WHERE {open_q} AND voice_ans IS NOT NULL").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE notes IS NULL").fetchone()[0] == 0
    calls = llm.stats["calls"]
    voices.run(b, con, cfg, log=lambda *_: None, llm=llm)                # rerun: everything cached
    assert llm.stats["calls"] == calls
    with __import__("pytest").raises(ValueError):
        voices.run(b, con, dict(cfg, voices={"second": "listener"}), log=lambda *_: None, llm=llm)
