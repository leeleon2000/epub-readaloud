"""Observability and error handling (docs/DESIGN.md)."""
import json
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from pipeline import config, store
from pipeline.adapters.llm import LLMError, LLMUnavailable, OpenAICompatLLM
from pipeline.adapters.llm.mock import MockLLM
from pipeline.attrib_conf import score
from pipeline.book import Book
from pipeline.cli import main, run_stages, status_dict
from pipeline.memory.db import CharacterDB
from pipeline.memory.rollback import chunk_range, restore, rollback
from pipeline.memory.validate import Validator
from pipeline.observe import Observer, PipelinePaused
from pipeline.stages import characters, classify

PRIDE = Path(__file__).parent / "fixtures" / "pride_epub2.epub"


# ------------------------------------------------------------------ helpers

def _book(tmp_path, fixture=PRIDE, bid="pride"):
    main(["--books", str(tmp_path), "init", str(fixture), "--id", bid])
    b = Book.get(bid, tmp_path)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = "mock"
    cfg["llm"]["adapter"] = "mock"
    cfg["attribute"]["version"] = "v2"                   # these tests exercise the v2 stages; the default is v3
    return b, cfg


class ScriptedClassifier(MockLLM):
    """Classifies the two undecided Pride front-matter sections like a real model would."""

    def __init__(self):
        super().__init__()
        self.asked = []

    def _classify(self, p):
        self.asked.append(p["title"])
        cat = "toc" if "Chapter" in p["opening"] and "," in p["opening"] else "front_matter"
        return {"category": cat, "confidence": 0.9, "reason": "scripted"}


def _snapshot(con):
    chars = sorted(tuple(r) for r in con.execute("SELECT id, name, status, merged_into, mentions FROM character"))
    aliases = sorted(tuple(r) for r in con.execute("SELECT character_id, norm, strength FROM alias"))
    return chars, aliases


# ------------------------------------------------------------------ classify

def test_classify_keeps_pride_front_matter_out_of_the_story(tmp_path):
    b, cfg = _book(tmp_path)
    run_stages(b, cfg, until="segment", log=lambda *_: None)
    con = store.connect(b.dir)
    llm = ScriptedClassifier()
    classify.run(b, con, cfg, log=lambda *_: None, llm=llm)
    assert len(llm.asked) == 2                         # only the TOC block and the title page needed the model
    story_before_ch1 = con.execute("SELECT COUNT(*) FROM sentence s WHERE s.doc_id=1 AND s.seq < 409 AND "
                                   + classify.story_filter_sql("s")).fetchone()[0]
    assert story_before_ch1 == 0                       # header, TOC, title page, preface, illustrations
    cats = dict(con.execute("SELECT id, category FROM section WHERE doc_id=1").fetchall())
    assert cats["d001s00000"] == "boilerplate" and cats["d001s00083"] == "front_matter"
    assert cats["d001s00205"] == "captions" and cats["d001s00409"] == "story"
    # captions and boilerplate are not narrated; the preface is
    spoken = dict(con.execute("SELECT section_id, MAX(speak) FROM sentence GROUP BY section_id").fetchall())
    assert spoken["d001s00205"] == 0 and spoken["d001s00000"] == 0 and spoken["d001s00083"] == 1
    assert spoken["d002s00034"] == 0                   # caption set as a heading before CHAPTER II.


def test_manual_section_override_wins_and_reports_stale_chunks(tmp_path):
    b, cfg = _book(tmp_path)
    run_stages(b, cfg, until="classify", log=lambda *_: None)
    con = store.connect(b.dir)
    affected = classify.set_categories(b, con, {"d001s00083": "story"})
    assert affected == []                               # characters not run yet
    classify.run(b, con, cfg, log=lambda *_: None)
    assert con.execute("SELECT category, source FROM section WHERE id='d001s00083'").fetchone()[:] == ("story", "manual")


# ------------------------------------------------------------------ spike -> pause -> rollback -> restore

def test_spike_pauses_and_rollback_restore_round_trip(tmp_path):
    b, cfg = _book(tmp_path)
    run_stages(b, cfg, until="classify", log=lambda *_: None)
    con = store.connect(b.dir)
    classify.set_categories(b, con, {"d001s00083": "story"})      # deliberate mistake: preface as story
    logs = []
    assert run_stages(b, cfg, until="characters", log=logs.append) == "paused"
    con = store.connect(b.dir)
    w = con.execute("SELECT * FROM warning WHERE code='CHAR_SPIKE' AND status='open'").fetchone()
    assert w and w["severity"] == "pause"
    st = status_dict(b, con)
    assert st["warnings"][0]["code"] == "CHAR_SPIKE" and st["warnings"][0]["opening"]
    # gate: a second run refuses to continue while the pause is open
    assert run_stages(b, cfg, until="characters", log=lambda *_: None) == "paused"

    before = _snapshot(con)
    done = [r["id"] for r in con.execute("SELECT id FROM chunk WHERE pass1='done' ORDER BY doc_id, seq_from")]
    rid, orphans = rollback(con, done, note="preface was story")
    assert orphans == []
    assert con.execute("SELECT COUNT(*) FROM character").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE pass1='pending' AND id=?", (done[0],)).fetchone()[0] == 1
    restore(con, rid)
    assert _snapshot(con) == before                       # the character list is exactly back
    assert con.execute("SELECT pass1 FROM chunk WHERE id=?", (done[0],)).fetchone()[0] == "done"


def test_manual_events_survive_rollback_and_reset(tmp_path):
    b, cfg = _book(tmp_path, fixture=Path(__file__).parent / "fixtures" / "alice_epub2.epub", bid="alice")
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    run_stages(b, cfg, until="characters", doc_ids=[2], log=lambda *_: None)
    con = store.connect(b.dir)
    db = CharacterDB(con)
    db.apply("profile", {"id": "alice", "age": "child"}, source="manual")
    db.apply("create", {"id": "dodo", "name": "the Dodo"}, source="manual")
    con.commit()
    rid, orphans = characters.reset(b, con)
    assert CharacterDB(con).card("dodo") is not None          # manual create kept
    assert any(o["id"] == "alice" for o in orphans)           # manual edit on a model-made card is orphaned
    restore(con, rid)
    assert CharacterDB(con).card("alice").age == "child"


# ------------------------------------------------------------------ validation layer

@pytest.fixture
def vdb():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    db = CharacterDB(con)
    db.apply("create", {"id": "mr_bennet", "name": "Mr. Bennet", "gender": "male"})
    db.apply("create", {"id": "mrs_bennet", "name": "Mrs. Bennet", "gender": "female"})
    db.apply("create", {"id": "jane", "name": "Jane Bennet", "gender": "female"})
    db.apply("alias", {"id": "jane", "alias": "Miss Bennet", "strength": "medium"})
    return db


def _val(db, text="Mr. Bennet and Mrs. Bennet sat with Jane and Mrs. Long's nieces."):
    return Validator(db, Observer.null(lambda *_: None), "c1", text, {"grounding_reject_warn": 3})


def _res(**kw):
    base = {"mention": 0, "action": "new", "same_as": -1, "character_id": "", "name": "", "gender": "unknown",
            "age": "unknown", "accent": "", "profile": "", "voice_description": ""}
    return {**base, **kw}


def test_validation_rules(vdb):
    v = _val(vdb)
    assert v.check_resolution(_res(name="Mr. Collins"), {"text": "Mr. Collins", "kind": "name"}) is None
    assert v.rejects["not_grounded"] == 1                                       # not in the text
    assert v.check_resolution(_res(action="existing", character_id="ghost"), {"text": "Jane", "kind": "name"}) is None
    assert v.rejects["unknown_id"] == 1                                         # no longer becomes a new character
    assert v.check_resolution(_res(name="nieces"), {"text": "Mrs. Long's nieces", "kind": "group"}) is None
    assert v.check_resolution(_res(action="existing", character_id="mrs_bennet"),
                              {"text": "your daughters", "kind": "relational"}) is None   # plural -> group
    out = v.check_resolution(_res(name="Jane's aunt"), {"text": "Jane", "kind": "relational"})
    assert out["action"] == "provisional"                                       # no real name for her
    out = v.check_resolution(_res(name="Mrs. Bennet"), {"text": "Mrs. Bennet", "kind": "relational"})
    assert out["action"] == "provisional"                                       # name is just the mention
    v2 = _val(vdb, "“My dear Mr. Bennet,” said his lady to him one day. Mrs. Bennet sat down.")
    out = v2.check_resolution(_res(name="Mrs. Bennet"), {"text": "his lady", "kind": "relational"})
    assert out["action"] == "new"                                               # named, and the name is in the text
    assert not v.merge_ok("mrs_bennet", "mr_bennet", 0.95, "same surname")      # two real names
    assert v.obs.memory["warnings"][-1]["code"] == "MERGE_REVIEW"
    assert v.attrs("jane", {"gender": "male", "age": "unknown"}) == {}           # keeps female
    assert v.obs.memory["warnings"][-1]["code"] == "ATTR_CONFLICT"
    assert not v.alias_ok("mrs_bennet", "Miss Bennet", "medium")                 # belongs to Jane
    vdb.apply("constraint", {"kind": "never_merge", "a": "jane", "b": "mrs_bennet"}, source="manual")
    assert not _val(vdb).merge_ok("jane", "mrs_bennet", 0.99, "x")


# ------------------------------------------------------------------ LLM failures

class FlakyLLM(MockLLM):
    def __init__(self, fail_chunks=(), always=False):
        super().__init__()
        self.fail_chunks, self.always, self.failed_once = set(fail_chunks), always, set()

    def complete_json(self, messages, schema, task="", payload=None, max_tokens=None):
        cid = self.obs.chunk_id if self.obs else None
        if self.always or (cid in self.fail_chunks and cid not in self.failed_once):
            self.failed_once.add(cid)
            raise LLMUnavailable("service down (test)")
        return super().complete_json(messages, schema, task, payload, max_tokens)


def _alice_chars(tmp_path, llm):
    b, cfg = _book(tmp_path, fixture=Path(__file__).parent / "fixtures" / "alice_epub2.epub", bid="alice")
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    run_stages(b, cfg, until="classify", log=lambda *_: None)
    con = store.connect(b.dir)
    obs = Observer(b.dir, "test", printer=lambda *_: None)
    try:
        characters.run(b, con, cfg, log=lambda *_: None, doc_ids=[2, 3], llm=llm, obs=obs)
        return con, "ok"
    except PipelinePaused as p:
        obs.flush()
        return con, p.code


def test_failed_chunk_is_isolated_and_retried(tmp_path):
    con, result = _alice_chars(tmp_path, FlakyLLM(fail_chunks={"c002-000"}))
    assert result == "ok"
    assert con.execute("SELECT pass1 FROM chunk WHERE id='c002-000'").fetchone()[0] == "done"   # retried at the end
    assert con.execute("SELECT COUNT(*) FROM warning WHERE code='LLM_FAILED'").fetchone()[0] == 1


def test_llm_down_pauses(tmp_path):
    con, result = _alice_chars(tmp_path, FlakyLLM(always=True))
    assert result == "LLM_DOWN"
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE pass1='failed'").fetchone()[0] >= 3


class _Stub(BaseHTTPRequestHandler):
    script: list = []

    def do_POST(self):
        code, body = _Stub.script.pop(0)
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *a):
        pass


def _stub_server(script):
    _Stub.script = list(script)
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_backoff_on_5xx_then_success_and_no_retry_on_4xx():
    ok = {"choices": [{"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}], "usage": {}}
    srv = _stub_server([(503, {}), (502, {}), (200, ok)])
    llm = OpenAICompatLLM({"endpoint": f"http://127.0.0.1:{srv.server_port}/v1", "backoff_seconds": 0.01})
    obs = Observer.null(lambda *_: None)
    llm.obs = obs
    assert llm.complete_json([], {}, "t") == {"a": 1}
    assert [c["result"] for c in obs.memory["llm_calls"]] == ["error", "error", "ok"]
    srv.shutdown()
    srv = _stub_server([(400, {"error": "bad"})])
    llm = OpenAICompatLLM({"endpoint": f"http://127.0.0.1:{srv.server_port}/v1", "backoff_seconds": 0.01})
    with pytest.raises(LLMError) as e:
        llm.complete_json([], {}, "t")
    assert not isinstance(e.value, LLMUnavailable)
    srv.shutdown()


# ------------------------------------------------------------------ observability

def test_run_records_calls_metrics_and_log(tmp_path):
    b, cfg = _book(tmp_path, fixture=Path(__file__).parent / "fixtures" / "alice_epub2.epub", bid="alice")
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    assert run_stages(b, cfg, until="attribute", doc_ids=[2], log=lambda *_: None) == "ok"
    con = store.connect(b.dir)
    run = con.execute("SELECT * FROM run ORDER BY started DESC").fetchone()
    assert run["result"] == "ok"
    assert con.execute("SELECT COUNT(*) FROM llm_call WHERE run_id=?", (run["run_id"],)).fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM chunk_metrics WHERE pass=1").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM chunk_metrics WHERE pass=2").fetchone()[0] == 2
    assert (b.dir / "work" / "logs" / f"{run['run_id']}.jsonl").exists()
    assert (b.reports / "characters_report.md").exists()
    st = status_dict(b, con)
    assert {"sections", "characters", "attribute", "warnings", "review", "rollbacks"} <= set(st)


# ------------------------------------------------------------------ computed attribution confidence

def test_is_proper_name():
    from pipeline.memory.validate import is_proper_name
    assert is_proper_name("Mrs. Bennet") and is_proper_name("Lizzy")
    assert not is_proper_name("his lady") and not is_proper_name("the Queen") and not is_proper_name("Mr.")


def test_provisional_promoted_when_named(tmp_path):
    from pipeline.stages.characters import _apply_resolution
    con = store.connect(tmp_path)
    db = CharacterDB(con)
    db.apply("create", {"id": "unknown_his_lady_c1", "name": "his lady", "status": "provisional"})
    v = Validator(db, Observer.null(lambda *_: None), "c2", "Mrs. Bennet smiled.", {})
    m = {"existing": 0, "new": 0, "provisional": 0, "new_chars": 0}
    _apply_resolution(db, _res(action="existing", character_id="unknown_his_lady_c1"),
                      {"text": "Mrs. Bennet", "kind": "title"}, {"id": "c2", "doc_id": 1}, v, m)   # model said title
    card = db.card("unknown_his_lady_c1")
    assert card.status == "active" and card.name == "Mrs. Bennet"


def test_attribution_confidence_from_speech_tags(vdb):
    q = lambda spk, after="", before="", para=1: {"speaker": spk, "after": after, "before": before, "para": para}  # noqa
    s = score(vdb, 1, [
        q("mrs_bennet", after="said Mrs. Bennet."),        # tag agrees
        q("mr_bennet", after="said Mrs. Bennet."),         # tag disagrees
        q("jane", after="she cried."),                     # pronoun, gender agrees
        q("mr_bennet", after="she cried."),                # pronoun, gender disagrees
        q("unknown"),
        q("narrator"),
    ])
    assert s == [0.95, 0.3, 0.8, 0.3, 0.0, 0.8]


# ------------------------------------------------------------------ repeated warnings

def test_repeats_are_counted_not_duplicated_and_pause(tmp_path):
    con = store.connect(tmp_path)
    obs = Observer(tmp_path, "test", printer=lambda *_: None)
    obs.monitor = {"repeat_subject_pause": 5, "repeat_code_pause": 20}
    for i in range(4):
        obs.warn("ALIAS_CONFLICT", "warn", f"c{i}", "'Jane' belongs to lydia", alias="Jane", owner="lydia",
                 character_id="jane")
    obs.checkpoint()                                         # 4 repeats: no pause yet
    rows = con.execute("SELECT count, chunk_id, last_chunk FROM warning WHERE code='ALIAS_CONFLICT'").fetchall()
    assert [tuple(r) for r in rows] == [(4, "c0", "c3")]   # one row, counted
    obs.warn("ALIAS_CONFLICT", "warn", "c4", "'Jane' belongs to lydia", alias="Jane", owner="lydia",
             character_id="jane")
    with pytest.raises(PipelinePaused) as e:
        obs.checkpoint()                                     # 5th time: pause after the chunk
    assert e.value.code == "REPEATED"
    rep = con.execute("SELECT details FROM warning WHERE code='REPEATED'").fetchone()
    assert json.loads(rep[0])["of"] == "ALIAS_CONFLICT"


def test_many_different_warnings_of_one_code_pause(tmp_path):
    store.connect(tmp_path)
    obs = Observer(tmp_path, "test", printer=lambda *_: None)
    obs.monitor = {"repeat_subject_pause": 5, "repeat_code_pause": 20, "repeat_code_overrides": {"LLM_FAILED": 3}}
    obs.warn("LLM_FAILED", "warn", "c1", "x")
    obs.warn("LLM_FAILED", "warn", "c7", "x")
    obs.checkpoint()
    obs.warn("LLM_FAILED", "warn", "c9", "x")
    with pytest.raises(PipelinePaused) as e:
        obs.checkpoint()
    assert e.value.code == "REPEATED"


def test_no_pause_records_but_continues_except_llm_down(tmp_path):
    con = store.connect(tmp_path)
    obs = Observer(tmp_path, "test", printer=lambda *_: None)
    obs.monitor = {"no_pause": True, "repeat_subject_pause": 2}
    for i in range(3):
        obs.warn("ALIAS_CONFLICT", "warn", f"c{i}", "'Jane' belongs to lydia", alias="Jane")
        obs.checkpoint()                                     # REPEATED recorded, no raise
    obs.pause("CHAR_SPIKE", "c3", "spike")
    assert {r[0] for r in con.execute("SELECT code FROM warning WHERE severity='pause'")} == {"REPEATED", "CHAR_SPIKE"}
    assert obs.open_pauses() == []                           # the stage gate lets characters/attribute start
    with pytest.raises(PipelinePaused):
        obs.pause("LLM_DOWN", "c4", "down")
    assert [w["code"] for w in obs.open_pauses()] == ["LLM_DOWN"]


def test_speech_tag_between_two_halves_counts(vdb):
    q = {"speaker": "mrs_bennet", "after": "", "before": "said Mrs. Bennet to her husband,", "para": 1}
    assert score(vdb, 1, [{**q, "between": "said Mrs. Bennet to her husband,"}]) == [0.95]
    assert score(vdb, 1, [{**q, "speaker": "mr_bennet", "between": q["before"]}]) == [0.3]
    assert score(vdb, 1, [q]) == [0.5]             # without `between` the tag was missed (v1)


def test_attribute_prompt_uses_neutral_codes(tmp_path):
    """docs/DESIGN.md: the model sees C1..Cn, never our ids; answers map back."""
    from pipeline.stages import attribute
    seen = {}

    class Spy(MockLLM):
        def complete_json(self, messages, schema, name, payload, max_tokens=None):
            seen["prompt"] = messages[-1]["content"]
            seen["enum"] = schema["properties"]["quotes"]["items"]["properties"]["speaker"]["enum"]
            return super().complete_json(messages, schema, name, payload, max_tokens)

    root = tmp_path / "books"
    main(["--books", str(root), "init", str(Path(__file__).parent / "fixtures" / "alice_epub2.epub"), "--id", "a"])
    b = Book.get("a", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, only=["ingest", "segment", "classify", "characters"])
    con = store.connect(b.dir)
    attribute.run(b, con, cfg, log=lambda *_: None, doc_ids=[2], llm=Spy())
    ids = [r[0] for r in con.execute("SELECT id FROM character")]
    assert "alice" in ids and all(f"- {i}:" not in seen["prompt"] for i in ids)
    assert "C1" in seen["enum"] and not set(ids) & set(seen["enum"])
    sp = {r[0] for r in con.execute("SELECT DISTINCT speaker_id FROM utterance WHERE is_quote=1")}
    assert "alice" in sp and not any(s.startswith("C") and s[1:].isdigit() for s in sp)


def test_shared_alias_tag_decides_nothing(vdb):
    vdb.apply("create", {"id": "elizabeth", "name": "Elizabeth Bennet", "gender": "female"})
    vdb.apply("alias", {"id": "elizabeth", "alias": "Miss Bennet", "strength": "weak"})
    vdb.apply("alias", {"id": "jane", "alias": "Miss Bennet", "strength": "weak"})
    q = {"speaker": "elizabeth", "after": "said Miss Bennet.", "before": "", "para": 1}
    assert score(vdb, 1, [q]) == [0.5]          # not 0.95 / 0.3: either sister could be meant


def test_llm_seed_and_temperature_are_sent():
    sent = []
    llm = OpenAICompatLLM({"temperature": 0, "seed": 7})
    llm._post = lambda body: (sent.append(dict(body)) or
                              {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {}})
    llm.complete_json([{"role": "user", "content": "x"}], {"type": "object"}, "t")
    assert sent[0]["temperature"] == 0 and sent[0]["seed"] == 7
    llm2 = OpenAICompatLLM({})
    llm2._post = llm._post
    llm2.complete_json([{"role": "user", "content": "x"}], {"type": "object"}, "t")
    assert "seed" not in sent[1]


def test_a_one_word_question_is_not_a_page_number():
    """“I?” used to match the page-number rule (a roman numeral) and was silently left out of the audio."""
    from pipeline.stages.classify import PAGE
    for line in ("12", "XIV", "[45]", "(iv)".upper(), "PAGE", "27."):
        assert PAGE.match(line), line
    for line in ("“I?”", "“I!”", "I?", '"I!"', "“V.”"):
        assert not PAGE.match(line), line
