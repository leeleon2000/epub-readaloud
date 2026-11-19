"""Long jobs for the UI: one `readaloud` process per book, which outlives the browser tab and the UI server.

A job's state is two files in books/<id>/ui/: job.json (what was asked, the process id, how it ended) and job.log
(everything the process printed). Closing the page loses nothing; opening it again picks the log up where it is.
Stages are idempotent, so "resume" is the same job started again.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import REPO

EXIT_PAUSED = 3


class JobBusy(Exception):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Jobs:
    def __init__(self, books_root: Path):
        self.root = Path(books_root)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.RLock()     # starting a job, and recording how one ended
        self._io = threading.Lock()        # one writer of a state file at a time

    def _dir(self, book_id: str) -> Path:
        return self.root / book_id / "ui"

    def _write(self, book_id: str, state: dict) -> None:
        d = self._dir(book_id)
        d.mkdir(parents=True, exist_ok=True)
        with self._io:      # the thread watching a job and a request asking about it can both get here at its end
            tmp = d / f"job.json.{os.getpid()}.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(d / "job.json")

    def _read(self, book_id: str) -> dict | None:
        p = self._dir(book_id) / "job.json"
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def state(self, book_id: str) -> dict | None:
        st = self._read(book_id)
        if st and st.get("status") == "running" and not _alive(st.get("pid", -1)):
            with self._lock:                    # if this server started the job, its watcher records the real result
                st = self._read(book_id) or st
                if st.get("status") == "running" and book_id not in self._procs:
                    # the process ended while no UI server was watching it: the exit code is gone
                    st.update(status="ended", ended=st.get("ended") or time.time())
                    self._write(book_id, st)
        return st

    def running(self, book_id: str) -> bool:
        st = self.state(book_id)
        return bool(st and st["status"] == "running")

    def start(self, book_id: str, args: list, label: str) -> dict:
        """Start `readaloud --books <root> <args...>` for a book: one command (a list of arguments), or several run
        one after the other (a list of such lists; the first that fails ends the job). One job per book at a time."""
        commands = [list(c) for c in args] if args and isinstance(args[0], (list, tuple)) else [list(args)]
        with self._lock:
            if self.running(book_id):
                raise JobBusy(f"{book_id}: a job is already running")
            d = self._dir(book_id)
            d.mkdir(parents=True, exist_ok=True)
            log = d / "job.log"
            cmd = [sys.executable, "-u", "-m", "pipeline.ui.runner", "--books", str(self.root)]
            for c in commands:
                cmd += ["--", *c]
            with log.open("wb") as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=REPO,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
            # `args` is the last command (what the form shows again); `commands` is everything the job runs
            st = {"book": book_id, "label": label, "args": commands[-1], "commands": commands, "pid": proc.pid,
                  "status": "running", "started": time.time(), "ended": None, "returncode": None}
            self._write(book_id, st)
            self._procs[book_id] = proc
            threading.Thread(target=self._watch, args=(book_id, proc), daemon=True).start()
            return st

    def _watch(self, book_id: str, proc: subprocess.Popen) -> None:
        rc = proc.wait()
        with self._lock:
            st = self.state(book_id) or {}
            if st.get("pid") != proc.pid:
                return
            if st.get("status") != "stopped":
                st["status"] = "ok" if rc == 0 else "paused" if rc == EXIT_PAUSED else "failed"
            st.update(returncode=rc, ended=time.time())
            self._write(book_id, st)
            self._procs.pop(book_id, None)

    def stop(self, book_id: str, wait: float = 10.0) -> dict | None:
        st = self.state(book_id)
        if not st or st["status"] != "running":
            return st
        pid = st["pid"]
        st["status"] = "stopped"
        self._write(book_id, st)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            end = time.time() + wait
            while time.time() < end and _alive(pid):
                proc = self._procs.get(book_id)
                if proc is not None and proc.poll() is not None:
                    break
                time.sleep(0.1)
            if not _alive(pid) or (self._procs.get(book_id) and self._procs[book_id].poll() is not None):
                break
        st = self.state(book_id) or st
        if st.get("ended") is None:
            st["ended"] = time.time()
            self._write(book_id, st)
        return st

    def log(self, book_id: str, offset: int = 0, limit: int = 200_000) -> tuple[str, int]:
        """The log from a byte offset: (text, next offset). An offset past the end means a new job replaced the log."""
        p = self._dir(book_id) / "job.log"
        if not p.exists():
            return "", 0
        size = p.stat().st_size
        if offset > size:
            offset = 0
        with p.open("rb") as f:
            f.seek(offset)
            data = f.read(limit)
        return data.decode("utf-8", "replace"), offset + len(data)
