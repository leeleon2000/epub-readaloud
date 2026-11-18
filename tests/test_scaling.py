"""Design rule: LLM input size must not grow with the number of characters."""
import itertools

from pipeline import store
from pipeline.adapters.llm.mock import MockLLM
from pipeline.book import Book
from pipeline.stages import characters

SYL = ["bar", "kel", "mor", "dov", "tis", "lun", "fen", "gar", "hol", "pim", "rus", "vex", "zan", "quo",
       "wil", "nod", "cra", "jep", "sul", "tav"]


class RecordingLLM(MockLLM):
    def __init__(self):
        super().__init__()
        self.resolve_sizes = []

    def complete_json(self, messages, schema, task="", payload=None, max_tokens=None):
        if task == "resolve_mentions":
            self.resolve_sizes.append(sum(len(m["content"]) for m in messages))
        return super().complete_json(messages, schema, task, payload)


def test_prompt_size_bounded_with_260_characters(tmp_path):
    names = [(a + b).capitalize() for a, b in itertools.product(SYL, SYL[:13])][:260]
    con = store.connect(tmp_path)
    con.execute("INSERT INTO doc(id, href, manifest_id, has_speech) VALUES (1, 'x.xhtml', 'x', 1)")
    rows = [(f"s{i:04d}", 1, i, i, "body", f"{n} walked slowly to the market that day.",
             f"{n} walked slowly to the market that day.") for i, n in enumerate(names)]
    con.executemany("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    cfg = {"character_memory": {"chunk_words": 80, "max_candidates": 20, "profile_max_tokens": 100,
                                "recent_chunks": 3, "roster_token_budget": 500},
           "monitor": {"new_chars_per_chunk_max": 10_000}}
    llm = RecordingLLM()
    characters.run(Book("synthetic", tmp_path), con, cfg, log=lambda *_: None, llm=llm)
    n_chars = con.execute("SELECT COUNT(*) FROM character").fetchone()[0]
    assert n_chars >= 250
    sizes = llm.resolve_sizes
    # The compact roster grows until it exceeds roster_token_budget, then is dropped in favour of
    # per-mention candidates (<= max_candidates cards). After that the prompt size must plateau:
    # over the last ~80 characters added, it stays flat and below the peak.
    tail = sizes[-8:]
    assert max(tail) < max(sizes), sizes
    assert max(tail) - min(tail) < 0.1 * max(tail), sizes
    assert max(sizes) < 3000 + 4 * 500 + 20 * 400, sizes   # instructions + roster budget + cards


class TruncatingLLM(RecordingLLM):
    """Pretends the output overflows whenever more than 3 mentions are resolved at once."""

    def complete_json(self, messages, schema, task="", payload=None, max_tokens=None):
        from pipeline.adapters.llm import LLMTruncated
        if task == "resolve_mentions" and len(payload["mentions"]) > 3:
            raise LLMTruncated("too long")
        return super().complete_json(messages, schema, task, payload)


def test_truncated_resolve_is_split_not_lost(tmp_path):
    names = [(a + b).capitalize() for a, b in itertools.product(SYL[:4], SYL[:5])]   # 20 names
    con = store.connect(tmp_path)
    con.execute("INSERT INTO doc(id, href, manifest_id, has_speech) VALUES (1, 'x.xhtml', 'x', 1)")
    text = " ".join(f"{n} came in." for n in names)
    con.execute("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES ('s1',1,0,0,'body',?,?)",
                (text, text))
    con.commit()
    cfg = {"character_memory": {"chunk_words": 1500, "max_candidates": 20, "profile_max_tokens": 100,
                                "recent_chunks": 3, "roster_token_budget": 2000, "resolve_batch": 10},
           "monitor": {"new_chars_per_chunk_max": 10_000}}
    characters.run(Book("t", tmp_path), con, cfg, log=lambda *_: None, llm=TruncatingLLM())
    assert con.execute("SELECT COUNT(*) FROM character").fetchone()[0] == 20
    assert con.execute("SELECT pass1 FROM chunk").fetchone()[0] == "done"
