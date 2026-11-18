"""v2 P1 collect stage (docs/DESIGN.md): grounding, parallel calls, split on truncation."""
import json
import threading
from pathlib import Path

import pytest

from pipeline import config, store
from pipeline.adapters.llm import LLMTruncated, LLMUnavailable
from pipeline.adapters.llm.mock import MockLLM
from pipeline.book import Book
from pipeline.cli import main, run_stages
from pipeline.stages import collect

PRIDE = Path(__file__).parent / "fixtures" / "pride_epub2.epub"
TEXT = ("“My dear Mr. Bennet,” said his lady to him one day, “have you heard that Netherfield Park is let at last?”"
        "\n\nMr. Bennet replied that he had not.")


def test_ground_keeps_only_verbatim():
    out = {"people": [
        {"refs": [{"text": "Mrs. Bennet", "kind": "name"},          # not in the text: dropped
                  {"text": "his lady", "kind": "relational"},
                  {"text": "she", "kind": "name"}],                  # pronoun: ignored
         "gender_cue": "female", "related": [{"relation": "wife", "of_ref": "Mr. Bennet"},
                                             {"relation": "wife", "of_ref": "Mr. Collins"}], "traits": "eager"},
        {"refs": [{"text": "Mr. Bennet", "kind": "name"}, {"text": "Mr. Bennet,", "kind": "name"}],
         "gender_cue": "male", "related": [], "traits": ""},
        {"refs": [{"text": "Lady Lucas", "kind": "name"}], "gender_cue": "female", "related": [], "traits": ""}],
        "relations": [
            {"a": "Mr. Bennet", "type": "spouse_of", "b": "his lady",
             "evidence": '"My dear Mr. Bennet," said his lady to him one day'},     # straight quotes still match
            {"a": "Mr. Bennet", "type": "parent_of", "b": "his lady", "evidence": "He was her father."}]}
    people, rels, cnt = collect.ground(TEXT, out)
    assert [[r["text"] for r in p["refs"]] for p in people] == [["his lady"], ["Mr. Bennet"]]
    assert people[0]["refs"][0]["rule_kind"] == "relational" and people[1]["refs"][0]["rule_kind"] == "title_surname"
    assert people[0]["related"] == [{"relation": "wife", "of_ref": "Mr. Bennet"}]
    assert [r["type"] for r in rels] == ["spouse_of"]
    assert cnt == {"refs": 2, "refs_dropped": 2, "related_dropped": 1, "relations": 1, "relations_dropped": 1}


@pytest.fixture
def pride(tmp_path):
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(PRIDE), "--id", "p"])
    b = Book.get("p", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    cfg["monitor"]["new_chars_per_chunk_max"] = 10_000
    run_stages(b, cfg, doc_ids=[1, 2], log=lambda *_: None, only=["ingest", "segment", "classify"])
    return b, cfg


def test_collect_stage_parallel(pride):
    b, cfg = pride
    threads = set()

    class Spy(MockLLM):
        def complete_json(self, messages, schema, task="", payload=None, max_tokens=None, **kw):
            threads.add(threading.get_ident())
            if self.obs:
                self.obs.llm_call(task, "t", "mock", {}, {"usage": {}}, 0.0, "stop", 0, "ok")
            return super().complete_json(messages, schema, task, payload, max_tokens)

    from pipeline.observe import Observer
    con = store.connect(b.dir)
    obs = Observer(b.dir, "test", printer=lambda *_: None)
    collect.run(b, con, cfg, log=lambda *_: None, doc_ids=[2], llm=Spy(), obs=obs)
    obs.flush()
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id=2 AND p1 != 'done'").fetchone()[0] == 0
    refs = [json.loads(r[0]) for r in con.execute("SELECT refs FROM p1_person")]
    assert any(r[0]["text"] == "Mr. Bennet" for r in refs)
    # every LLM call was recorded under the chunk it belonged to, even from worker threads
    calls = con.execute("SELECT chunk_id FROM llm_call WHERE task='collect'").fetchall()
    chunk_ids = {r[0] for r in con.execute("SELECT id FROM chunk WHERE doc_id=2")}
    assert calls and {r[0] for r in calls} <= chunk_ids
    assert len(threads) >= 1
    # rerun: nothing pending, no new calls
    n = len(calls)
    collect.run(b, con, cfg, log=lambda *_: None, doc_ids=[2], llm=Spy(), obs=obs)
    obs.flush()
    assert con.execute("SELECT COUNT(*) FROM llm_call WHERE task='collect'").fetchone()[0] == n


def test_truncation_splits_chunk_in_half(pride):
    b, cfg = pride

    class Small(MockLLM):          # "truncates" on long inputs, like a model with too much to say
        def complete_json(self, messages, schema, task="", payload=None, max_tokens=None, **kw):
            if task == "collect" and len(payload["text"].split()) > 700:
                raise LLMTruncated("collect: output hit max_tokens twice")
            return super().complete_json(messages, schema, task, payload, max_tokens)

    con = store.connect(b.dir)
    collect.run(b, con, cfg, log=lambda *_: None, doc_ids=[2], llm=Small())
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id=2 AND p1 != 'done'").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM p1_person").fetchone()[0] > 0


def test_failed_chunk_retried_once_then_left_failed(pride):
    b, cfg = pride
    fail = {"left": 1}

    class Flaky(MockLLM):
        def complete_json(self, messages, schema, task="", payload=None, max_tokens=None, **kw):
            if task == "collect" and "Netherfield" in payload["text"] and fail["left"] > 0:
                fail["left"] -= 1
                raise LLMUnavailable("down")
            return super().complete_json(messages, schema, task, payload, max_tokens)

    con = store.connect(b.dir)
    collect.run(b, con, cfg, log=lambda *_: None, doc_ids=[2], llm=Flaky())
    assert fail["left"] == 0
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id=2 AND p1 != 'done'").fetchone()[0] == 0


def test_collect_only_runs_for_roster_v2(pride):
    b, cfg = pride
    run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, until="characters")
    con = store.connect(b.dir)
    assert con.execute("SELECT COUNT(*) FROM p1_person").fetchone()[0] == 0     # roster v1: skipped
    cfg["character_memory"]["roster"] = "v2"
    run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, only=["collect"])
    assert con.execute("SELECT COUNT(*) FROM p1_person").fetchone()[0] > 0
