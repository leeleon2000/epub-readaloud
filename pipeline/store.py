"""SQLite project store (the data model). Holds state and caches only;
human-edited inputs live in YAML/CSV files next to it."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS doc (
  id INTEGER PRIMARY KEY,           -- spine order
  href TEXT NOT NULL,               -- path inside the EPUB, relative to the OPF dir
  manifest_id TEXT NOT NULL,
  has_speech INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sentence (
  id TEXT PRIMARY KEY,              -- also the span id in the XHTML
  doc_id INTEGER NOT NULL REFERENCES doc(id),
  seq INTEGER NOT NULL,             -- order within the doc
  para INTEGER NOT NULL,
  kind TEXT NOT NULL,               -- heading | body | caption (illustration caption: never read)
  text TEXT NOT NULL,               -- displayed text (whitespace collapsed)
  tts_text TEXT NOT NULL,
  speaker TEXT,
  style TEXT,
  block_id TEXT,
  section_id TEXT,                  -- set by classify
  speak INTEGER NOT NULL DEFAULT 1  -- 0 = not narrated, not in SMIL (toc, captions, boilerplate, page numbers)
);
CREATE INDEX IF NOT EXISTS sentence_doc ON sentence(doc_id, seq);

-- Content sections (classify): runs of sentences between headings / Gutenberg markers.
CREATE TABLE IF NOT EXISTS section (
  id TEXT PRIMARY KEY,              -- d<doc>s<first seq>
  doc_id INTEGER NOT NULL,
  seq_from INTEGER NOT NULL, seq_to INTEGER NOT NULL,
  title TEXT,
  category TEXT NOT NULL,           -- story | front_matter | toc | captions | notes | boilerplate | back_matter | uncertain
  confidence REAL,
  source TEXT NOT NULL,             -- rule | model | manual
  reason TEXT
);

-- One sentence = one or more utterances (quote vs narration). Default: one narrator utterance.
CREATE TABLE IF NOT EXISTS utterance (
  id TEXT PRIMARY KEY,              -- <sentence id>u<k>
  sentence_id TEXT NOT NULL,
  seq INTEGER NOT NULL,             -- order within the sentence
  start INTEGER NOT NULL,           -- char range in sentence.tts_text
  end INTEGER NOT NULL,
  tts_text TEXT NOT NULL,           -- text sent to TTS (quote marks removed)
  is_quote INTEGER NOT NULL DEFAULT 0,
  quote_id TEXT,                    -- shared by the pieces of one quotation
  speaker_id TEXT NOT NULL DEFAULT 'narrator',
  style TEXT,
  confidence REAL,
  source TEXT NOT NULL DEFAULT 'default',  -- default | rule (narration) | model | turns (alternation rule) | manual
  model_speaker TEXT,               -- the model's own answer, kept when a rule changes speaker_id
  model_confidence REAL,            -- (pipeline/attrib_turns.py always starts from these)
  voice_gender TEXT                 -- v3: class before the threshold (male / female / I / narrator)
);
CREATE INDEX IF NOT EXISTS utterance_sentence ON utterance(sentence_id, seq);

-- v2 P1 (docs/DESIGN.md): per chunk, who is there and how they are referred to.
CREATE TABLE IF NOT EXISTS p1_person (
  chunk_id TEXT NOT NULL,
  idx INTEGER NOT NULL,             -- person number within the chunk
  refs TEXT NOT NULL,               -- JSON [{text, kind, rule_kind}] (all verbatim in the chunk)
  gender_cue TEXT,
  related TEXT,                     -- JSON [{relation, of_ref}]
  traits TEXT,
  PRIMARY KEY (chunk_id, idx)
);
CREATE TABLE IF NOT EXISTS p1_relation (
  chunk_id TEXT NOT NULL,
  a TEXT NOT NULL, type TEXT NOT NULL, b TEXT NOT NULL,
  evidence TEXT NOT NULL            -- verbatim sentence(s) from the chunk
);
CREATE INDEX IF NOT EXISTS p1_relation_chunk ON p1_relation(chunk_id);

-- v2 P2 (docs/DESIGN.md): one cached model answer per name group (key = hash of its input).
CREATE TABLE IF NOT EXISTS p2_group (
  hash TEXT PRIMARY KEY,
  nodes TEXT NOT NULL,              -- JSON list of node keys
  result TEXT,                      -- JSON model answer (NULL until done)
  created TEXT DEFAULT CURRENT_TIMESTAMP
);
-- Basic relations between characters (docs/DESIGN.md); written only through CharacterDB events, computed
-- from relation_claim. Derived relations (sibling closure, in-laws...) are never stored (5.6 point 4).
CREATE TABLE IF NOT EXISTS relation (
  id INTEGER PRIMARY KEY,
  a TEXT NOT NULL, type TEXT NOT NULL, b TEXT NOT NULL,
  evidence TEXT, source TEXT,
  strength TEXT,                    -- strong | weak (docs/DESIGN.md)
  valid_from TEXT                   -- sentence id of the first evidence (spouse_of: married names)
);
-- Every observation of a relation, accepted or not (5.6 point 1); rebuilt by the roster stage.
CREATE TABLE IF NOT EXISTS relation_claim (
  id INTEGER PRIMARY KEY,
  a TEXT NOT NULL, type TEXT NOT NULL, b TEXT NOT NULL,
  kin TEXT,                         -- the kin word in the text
  chunk_id TEXT, sentence_id TEXT,
  kind TEXT NOT NULL,               -- named | p1 | possessive
  evidence TEXT,
  status TEXT NOT NULL,             -- accepted | weak | conflict | retracted
  reason TEXT, run_id TEXT
);

-- A reference without a name resolved through the relation graph ("his lady" -> mrs_bennet), docs/DESIGN.md.
CREATE TABLE IF NOT EXISTS mention_link (
  id INTEGER PRIMARY KEY,
  chunk_id TEXT, mention TEXT NOT NULL,
  anchors TEXT,                     -- JSON card ids tried as "his/her"
  relation TEXT,                    -- kin word
  targets TEXT,                     -- JSON card ids the relation names
  resolved INTEGER NOT NULL,        -- 1 = exactly one target
  source TEXT, run_id TEXT
);

-- v3 (docs/DESIGN.md): per quotation, the rule evidence and the model's answers. The voice itself
-- is decided from these on every run (pipeline/v3/decide.py) and written to utterance.
CREATE TABLE IF NOT EXISTS v3_quote (
  quote_id TEXT PRIMARY KEY,
  chunk_id TEXT, doc_id INTEGER, para INTEGER, ord INTEGER,
  text TEXT, narration TEXT,        -- the quote; the paragraph's narration (turn-taking)
  tag_text TEXT, tag_kind TEXT,     -- speech tag found by rule (pipeline/v3/tags.py)
  rule TEXT,                        -- why it is not dialogue (caption, quoted phrase), or NULL
  who TEXT, who_gender TEXT,        -- model, ask=who: the person (a name or a fixed option) and gender
  voice_ans TEXT,                   -- model, ask=voice: male | female | narrator | unknown
  who2 TEXT, who2_gender TEXT,      -- model, voices.second=notes: who again, with scene notes and evidence
  evidence TEXT, basis TEXT         --   the evidence it quoted and its kind (docs/DESIGN.md)
);
CREATE TABLE IF NOT EXISTS chunk (
  id TEXT PRIMARY KEY,              -- c<doc>-<k>
  doc_id INTEGER NOT NULL,
  seq_from INTEGER NOT NULL,        -- sentence seq range (inclusive)
  seq_to INTEGER NOT NULL,
  pass1 TEXT NOT NULL DEFAULT 'pending',   -- characters: pending | done | failed
  p1 TEXT NOT NULL DEFAULT 'pending',      -- v2 collect (P1): pending | done | failed
  pass2 TEXT NOT NULL DEFAULT 'pending',   -- attribute:  pending | done | failed | stale
  p3 TEXT NOT NULL DEFAULT 'pending',      -- v3 model pass (stage voices): pending | done | failed
  characters TEXT,                  -- JSON list of character ids seen in this chunk (pass 1)
  scene TEXT,                       -- JSON scene-state snapshot after this chunk
  notes TEXT                        -- v3 voices.second=notes: scene notes of the chunk (JSON, 15.18)
);

CREATE TABLE IF NOT EXISTS character (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  gender TEXT, age TEXT, accent TEXT,
  profile TEXT, voice_description TEXT,
  first_seen TEXT, mentions INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active',   -- active | provisional | merged | not_character
  merged_into TEXT,
  embedding BLOB
);
CREATE TABLE IF NOT EXISTS alias (
  character_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  norm TEXT NOT NULL,
  strength TEXT NOT NULL DEFAULT 'strong',  -- strong | medium | weak
  scope_from INTEGER, scope_to INTEGER,     -- doc id range where a weak alias is valid
  PRIMARY KEY (character_id, norm)
);
CREATE VIRTUAL TABLE IF NOT EXISTS alias_fts USING fts5(norm, character_id UNINDEXED, tokenize='trigram');
CREATE TABLE IF NOT EXISTS character_event (
  id INTEGER PRIMARY KEY,
  ts TEXT DEFAULT CURRENT_TIMESTAMP,
  type TEXT NOT NULL,               -- create | alias | profile | mention | merge | split | edit | rejected | constraint
  payload TEXT NOT NULL,            -- JSON
  source TEXT NOT NULL,             -- model | manual
  chunk_id TEXT,
  run_id TEXT,
  superseded_by TEXT                -- rollback id that voided this event (NULL = in effect)
);
CREATE TABLE IF NOT EXISTS rollback (
  id TEXT PRIMARY KEY,
  run_id TEXT,
  created TEXT DEFAULT CURRENT_TIMESTAMP,
  chunks TEXT NOT NULL,             -- JSON list of chunk ids
  event_mark INTEGER,               -- max character_event id when the rollback was made
  note TEXT,
  restored_by TEXT                  -- rollback id of the restore, if undone
);

-- Observability
CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY, command TEXT, started TEXT, ended TEXT, result TEXT, stats TEXT
);
CREATE TABLE IF NOT EXISTS llm_call (
  id INTEGER PRIMARY KEY, run_id TEXT, chunk_id TEXT, task TEXT, prompt_version TEXT, model TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, seconds REAL, finish_reason TEXT, retries INTEGER,
  result TEXT,                      -- ok | truncated | invalid | error
  raw_path TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunk_metrics (
  chunk_id TEXT NOT NULL, pass INTEGER NOT NULL, run_id TEXT NOT NULL, metrics TEXT NOT NULL,
  PRIMARY KEY (chunk_id, pass, run_id)
);
CREATE TABLE IF NOT EXISTS warning (
  id INTEGER PRIMARY KEY, run_id TEXT, code TEXT NOT NULL, severity TEXT NOT NULL,   -- info | warn | pause
  chunk_id TEXT, details TEXT, status TEXT NOT NULL DEFAULT 'open',                  -- open | acknowledged | resolved
  handled_by TEXT, note TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP,
  subject TEXT,                      -- what the warning is about; one open row per (code, subject)
  count INTEGER NOT NULL DEFAULT 1,  -- occurrences (chunk_id = first, last_chunk = latest)
  last_chunk TEXT
);

CREATE TABLE IF NOT EXISTS voice (
  ref TEXT PRIMARY KEY,             -- adapter-specific voice spec
  character_id TEXT,
  path TEXT, sha TEXT, model TEXT, design TEXT, seed INTEGER,
  frozen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS block (
  id TEXT PRIMARY KEY, doc_id INTEGER NOT NULL, utterance_ids TEXT NOT NULL, speakers TEXT
);
CREATE TABLE IF NOT EXISTS take (
  id INTEGER PRIMARY KEY,
  block_id TEXT NOT NULL,
  key TEXT NOT NULL UNIQUE,         -- hash of tts_text + voice + model version + params + seed (+ context take)
  adapter TEXT NOT NULL,
  model_version TEXT,
  params TEXT,                      -- JSON
  seed INTEGER,
  context_take_id INTEGER,
  wav_path TEXT NOT NULL,           -- relative to the book dir
  duration REAL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS timing (
  take_id INTEGER, sentence_id TEXT, start REAL, end REAL, source TEXT,
  PRIMARY KEY (take_id, sentence_id)
);
CREATE TABLE IF NOT EXISTS qa (
  take_id INTEGER, utterance_id TEXT, wer REAL, chars_per_sec REAL, speaker_sim REAL,
  issues TEXT, passed INTEGER,
  PRIMARY KEY (take_id, utterance_id)
);
CREATE TABLE IF NOT EXISTS selection (block_id TEXT PRIMARY KEY, take_id INTEGER, chosen_by TEXT);
CREATE TABLE IF NOT EXISTS override (sentence_id TEXT PRIMARY KEY, wav_path TEXT, note TEXT);
"""


# Columns added after a table first shipped: (table, column, declaration).
ADDED_COLUMNS = [
    ("relation", "strength", "TEXT"),
    ("relation", "valid_from", "TEXT"),
    ("chunk", "p1", "TEXT NOT NULL DEFAULT 'pending'"),     # v2 collect: pending | done | failed
    ("sentence", "section_id", "TEXT"),
    ("sentence", "speak", "INTEGER NOT NULL DEFAULT 1"),
    ("character_event", "run_id", "TEXT"),
    ("character_event", "superseded_by", "TEXT"),
    ("take", "context_take_id", "INTEGER"),
    ("rollback", "event_mark", "INTEGER"),
    ("warning", "subject", "TEXT"),
    ("warning", "count", "INTEGER NOT NULL DEFAULT 1"),
    ("warning", "last_chunk", "TEXT"),
    ("utterance", "model_speaker", "TEXT"),
    ("utterance", "model_confidence", "REAL"),
    ("utterance", "voice_gender", "TEXT"),                  # v3: class before the threshold (male/female/I/narrator)
    ("chunk", "p3", "TEXT NOT NULL DEFAULT 'pending'"),     # v3 model pass: pending | done | failed
    ("chunk", "notes", "TEXT"),                             # v3 voices.second=notes: scene notes (JSON)
    ("v3_quote", "who2", "TEXT"),
    ("v3_quote", "who2_gender", "TEXT"),
    ("v3_quote", "evidence", "TEXT"),
    ("v3_quote", "basis", "TEXT"),
]


def _migrate(con: sqlite3.Connection) -> None:
    has_meta = con.execute("SELECT 1 FROM sqlite_master WHERE name='meta'").fetchone()
    version = 0
    if has_meta:
        r = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(r[0]) if r else (1 if con.execute(
            "SELECT 1 FROM sqlite_master WHERE name='sentence'").fetchone() else 0)
    if version == 1:
        # M1 -> M4: block/character/voice/qa change shape and held no human data;
        # selections were keyed by sentence id, now by utterance block id (synth re-selects by key).
        for t in ("block", "character", "voice", "qa", "selection"):
            con.execute(f"DROP TABLE IF EXISTS {t}")
    for table, col, decl in ADDED_COLUMNS:
        if con.execute(f"SELECT 1 FROM sqlite_master WHERE name='{table}'").fetchone():
            if col not in {r[1] for r in con.execute(f"PRAGMA table_info({table})")}:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    con.executescript(SCHEMA)
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    con.commit()


def connect(book_dir: Path) -> sqlite3.Connection:
    con = sqlite3.connect(book_dir / "project.db", timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    _migrate(con)
    return con


def get_meta(con, key: str, default=None):
    r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_meta(con, key: str, value) -> None:
    con.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, str(value)))
