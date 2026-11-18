"""Observability for a pipeline run (docs/DESIGN.md).

One Observer per `readaloud` invocation. It owns a *separate autocommit* SQLite
connection, so warnings, LLM call records and metrics survive when a stage
rolls back a failed chunk. Because the stage's connection holds a write
transaction while a chunk is in progress, DB writes are queued and written
by `flush()`, which stages call right after each chunk commits or rolls back. Everything is also appended to
work/logs/<run_id>.jsonl; raw LLM requests/responses go to
work/llm/<run_id>/<chunk>/<task>-<n>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import threading
import sqlite3
from pathlib import Path

from collections import Counter

SEVERITIES = ("info", "warn", "pause")
# Codes whose subject is not simply "the chunk": repeats of the same subject across chunks are one problem.
SUBJECT_FIELDS = {
    "ALIAS_CONFLICT": ("alias", "owner", "character_id"),
    "ATTR_CONFLICT": ("character_id", "field", "new"),
    "MERGE_REVIEW": ("pair",),
    "UNKNOWN_ID": ("character_id",),
    "GROUNDING_REJECT": ("mention",),
    "ORPHANED_EDIT": ("event_id",),
}


class PipelinePaused(Exception):
    """A pause-severity warning was raised; the operator must handle it (docs/USAGE.md)."""

    def __init__(self, code: str, chunk_id: str | None, message: str):
        super().__init__(f"{code} at {chunk_id or '-'}: {message}")
        self.code, self.chunk_id = code, chunk_id


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


class Observer:
    def __init__(self, book_dir: Path | None, command: str = "", printer=print):
        self.book_dir = book_dir
        self.printer = printer
        self.run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{random.randrange(16**4):04x}"
        self.chunk_id: str | None = None
        self.memory: dict[str, list] = {"warnings": [], "metrics": [], "llm_calls": []}
        self.con = None
        self._calls = 0
        self._pending: list[tuple[str, tuple]] = []
        self._wstate: dict[tuple[str, str], dict] = {}
        self._code_counts: Counter = Counter()
        self._pause_req = None
        self._tl = threading.local()  # per-thread chunk id for LLM calls made in worker threads
        self._lock = threading.Lock()
        self.monitor: dict = {}       # cfg["monitor"]; set by run_stages
        if book_dir is not None:
            self.con = sqlite3.connect(book_dir / "project.db", timeout=30, isolation_level=None)
            self.con.row_factory = sqlite3.Row
            self.con.execute("INSERT INTO run(run_id, command, started, result) VALUES (?,?,?, 'running')",
                             (self.run_id, command, _now()))
            self.log_path = book_dir / "work" / "logs" / f"{self.run_id}.jsonl"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def null(cls, printer=print) -> "Observer":
        """In-memory observer for tests and direct stage calls."""
        return cls(None, printer=printer)

    # ------------------------------------------------------------------ logging
    def _jsonl(self, rec: dict) -> None:
        if self.con is None:
            return
        rec = {"ts": _now(), "run_id": self.run_id, **rec}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log(self, msg: str, **fields) -> None:
        self.printer(msg)
        self._jsonl({"type": "log", "msg": msg, **fields})

    # ------------------------------------------------------------------ warnings
    def _subject(self, code: str, chunk_id: str | None, details: dict) -> str:
        fields = SUBJECT_FIELDS.get(code)
        if fields:
            return "|".join(json.dumps(details.get(f), sort_keys=True, ensure_ascii=False) for f in fields)
        return chunk_id or details.get("message", "")

    def warn(self, code: str, severity: str, chunk_id: str | None = None, message: str = "", **details) -> None:
        """Record a warning. One open row per (code, subject): repeats raise its count instead of adding
        rows. Too many repeats request a REPEATED pause, raised at the next checkpoint()."""
        assert severity in SEVERITIES, severity
        rec = {"code": code, "severity": severity, "chunk_id": chunk_id, "message": message, **details}
        self._jsonl({"type": "warning", **rec})
        self.memory["warnings"].append(rec)
        key = (code, self._subject(code, chunk_id, details))
        st = self._wstate.get(key)
        if st is None:
            row = None
            if self.con is not None:
                row = self.con.execute("SELECT id, count FROM warning WHERE code=? AND subject=? AND status='open'",
                                       key).fetchone()
            st = {"id": row["id"] if row else None, "count": row["count"] if row else 0, "run_count": 0,
                  "severity": severity, "chunk": chunk_id, "last": chunk_id,
                  "payload": json.dumps(details | {"message": message}, ensure_ascii=False)}
            self._wstate[key] = st
        st["count"] += 1
        st["run_count"] += 1
        st["last"] = chunk_id
        if SEVERITIES.index(severity) > SEVERITIES.index(st["severity"]):
            st["severity"] = severity
        st["dirty"] = True
        if severity != "info" and st["run_count"] == 1:
            self.printer(f"WARNING {code} [{severity}] {chunk_id or ''} {message}".rstrip())
        self._code_counts[code] += 1
        if code != "REPEATED" and severity != "pause" and self._pause_req is None:
            m = self.monitor
            subj_lim = int(m.get("repeat_subject_pause", 5))
            code_lim = int((m.get("repeat_code_overrides") or {}).get(code, m.get("repeat_code_pause", 20)))
            if code in SUBJECT_FIELDS and st["count"] == subj_lim:
                self._pause_req = (chunk_id, f"{code} repeated {st['count']} times for the same subject: {message}",
                                   {"of": code, "subject": key[1], "count": st["count"],
                                    "hint": "an earlier mistake is probably spreading; look at the subject first"})
            elif self._code_counts[code] == code_lim:
                self._pause_req = (chunk_id, f"{code} raised {code_lim} times in this run (latest: {message})",
                                   {"of": code, "count": code_lim, "hint": "probably a systemic problem"})

    def pause(self, code: str, chunk_id: str | None, message: str, **details) -> None:
        """Record a pause warning and stop. Call only after the current chunk is committed.
        With `monitor.no_pause` (measurement runs) only LLM_DOWN stops; the rest are recorded and the run goes on."""
        self.warn(code, "pause", chunk_id, message, **details)
        self.flush()
        if self.no_pause and code != "LLM_DOWN":
            self.printer(f"NO_PAUSE {code} at {chunk_id}: recorded, continuing (monitor.no_pause)")
            return
        raise PipelinePaused(code, chunk_id, message)

    @property
    def no_pause(self) -> bool:
        return bool(self.monitor.get("no_pause"))

    def checkpoint(self) -> None:
        """Call after a chunk committed (or rolled back): write queued rows, then raise a requested pause."""
        self.flush()
        if self._pause_req is not None:
            chunk_id, message, details = self._pause_req
            self._pause_req = None
            self.pause("REPEATED", chunk_id, message, **details)

    def flush(self) -> None:
        """Write queued rows. Call when the stage connection has no open write transaction."""
        if self.con is None:
            return
        with self._lock:
            pending, self._pending = self._pending, []
        for sql, params in pending:
            self.con.execute(sql, params)
        for (code, subject), st in self._wstate.items():
            if not st.pop("dirty", False):
                continue
            if st["id"] is None:
                cur = self.con.execute(
                    "INSERT INTO warning(run_id, code, severity, chunk_id, details, subject, count, last_chunk) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (self.run_id, code, st["severity"], st["chunk"], st["payload"], subject, st["count"], st["last"]))
                st["id"] = cur.lastrowid
            else:
                self.con.execute("UPDATE warning SET count=?, last_chunk=?, severity=? WHERE id=?",
                                 (st["count"], st["last"], st["severity"], st["id"]))

    def open_pauses(self) -> list:
        if self.con is None:
            rows = [w for w in self.memory["warnings"] if w["severity"] == "pause"]
        else:
            rows = self.con.execute("SELECT * FROM warning WHERE severity='pause' AND status='open' ORDER BY id").fetchall()
        return [w for w in rows if w["code"] == "LLM_DOWN"] if self.no_pause else rows

    # ------------------------------------------------------------------ metrics / LLM calls
    def metric(self, chunk_id: str, pass_: int, metrics: dict) -> None:
        self._jsonl({"type": "metrics", "chunk_id": chunk_id, "pass": pass_, **metrics})
        if self.con is None:
            self.memory["metrics"].append((chunk_id, pass_, metrics))
            return
        self._pending.append(("INSERT OR REPLACE INTO chunk_metrics(chunk_id, pass, run_id, metrics) VALUES (?,?,?,?)",
                              (chunk_id, pass_, self.run_id, json.dumps(metrics))))

    def in_chunk(self, chunk_id: str | None):
        """Context manager for worker threads: LLM calls made inside are recorded under chunk_id."""
        obs = self

        class _Ctx:
            def __enter__(self):
                obs._tl.chunk_id = chunk_id

            def __exit__(self, *exc):
                obs._tl.chunk_id = None
        return _Ctx()

    def llm_call(self, task: str, prompt_version: str, model: str, request: dict, response: dict | None,
                 seconds: float, finish_reason: str | None, retries: int, result: str, error: str = "") -> None:
        with self._lock:
            self._llm_call(task, prompt_version, model, request, response, seconds, finish_reason, retries,
                           result, error)

    def _llm_call(self, task, prompt_version, model, request, response, seconds, finish_reason, retries,
                  result, error) -> None:
        usage = (response or {}).get("usage") or {}
        chunk_id = getattr(self._tl, "chunk_id", None) or self.chunk_id
        rec = {"task": task, "chunk_id": chunk_id, "prompt_version": prompt_version, "model": model,
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "seconds": round(seconds, 2), "finish_reason": finish_reason, "retries": retries, "result": result}
        if self.con is None:
            self.memory["llm_calls"].append(rec)
            return
        self._calls += 1
        raw = self.book_dir / "work" / "llm" / self.run_id / (chunk_id or "_") / f"{task}-{self._calls:04d}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(json.dumps({"request": request, "response": response, "error": error}, ensure_ascii=False, indent=1))
        self._pending.append((
            "INSERT INTO llm_call(run_id, chunk_id, task, prompt_version, model, prompt_tokens, completion_tokens, "
            "seconds, finish_reason, retries, result, raw_path) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.run_id, chunk_id, task, prompt_version, model, rec["prompt_tokens"], rec["completion_tokens"],
             rec["seconds"], finish_reason, retries, result, str(raw.relative_to(self.book_dir)))))
        self._jsonl({"type": "llm_call", **rec})

    # ------------------------------------------------------------------ run end
    def finish(self, result: str, **stats) -> None:
        self._jsonl({"type": "run_end", "result": result, **stats})
        if self.con is not None:
            self.flush()
            self.con.execute("UPDATE run SET ended=?, result=?, stats=? WHERE run_id=?",
                             (_now(), result, json.dumps(stats), self.run_id))
