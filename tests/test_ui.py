"""The UI server (`readaloud ui`), without a browser: its JSON API, the jobs it starts, the documents it shows.

Everything runs with the mock adapters. tests/test_ui_browser.py drives the same server through a real page.
"""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pipeline import config
from pipeline.cli import planned_stages
from pipeline.config import REPO
from pipeline.ui import api, md, server
from pipeline.ui.jobs import Jobs

ALICE = Path(__file__).parent / "fixtures" / "alice_epub2.epub"


class Client:
    def __init__(self, base: str):
        self.base = base

    def call(self, method: str, path: str, body=None, headers: dict | None = None, raw: bool = False):
        h = {"X-Requested-With": "readaloud", **(headers or {})}
        data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
                return r.status, (b if raw else json.loads(b)), dict(r.headers)
        except urllib.error.HTTPError as e:
            b = e.read()
            return e.code, (json.loads(b) if b else {}), dict(e.headers)

    def get(self, path, **kw):
        return self.call("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.call("POST", path, {} if body is None else body, **kw)

    def wait(self, book: str, timeout: float = 180) -> dict:
        """Follow a job to its end: its final state, with the whole log in ["log"]."""
        offset, log, end = 0, "", time.time() + timeout
        while time.time() < end:
            _, r, _ = self.get(f"/api/books/{book}/job?offset={offset}")
            log, offset = log + r["log"], r["offset"]
            if r["job"] and r["job"]["status"] != "running":
                return {**r["job"], "log": log}
            time.sleep(0.2)
        raise AssertionError(f"job still running after {timeout} s:\n{log[-2000:]}")


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui-books")
    srv = server.make_server(root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    c.root = root
    yield c
    for d in root.iterdir():                 # no job may outlive the test session
        if d.is_dir():
            srv.RequestHandlerClass.jobs.stop(d.name, wait=5)
    srv.shutdown()
    srv.server_close()


# ---- documents -----------------------------------------------------------------------------------------------------

def test_markdown_covers_what_the_documents_use():
    html, heads = md.render("""# Title

Some *text* with `code`, **bold**, a [link](docs/USAGE.md#2-a-step) and [out](https://example.org).

## 2. A step: with punctuation

- one
  - nested `x`
- two
  continued

| A | B |
|---|---|
| 1 | `a\\|b` |

```bash
echo <hi> | cat
```
""")
    assert [h["id"] for h in heads] == ["title", "2-a-step-with-punctuation"]
    assert '<a href="#/help/USAGE/2-a-step">link</a>' in html
    assert 'href="https://example.org" target="_blank"' in html
    assert "<em>text</em>" in html and "<strong>bold</strong>" in html
    assert html.count("<ul>") == 2 and "<li>two continued" in html.replace("\n", "")
    assert "<td><code>a|b</code></td>" in html
    pic, _ = md.render("![a picture](img/x.gif) and ![another](docs/img/y.png)", here="USAGE", base="docs")
    assert 'src="/repo/docs/img/x.gif"' in pic and 'src="/repo/docs/docs/img/y.png"' in pic   # relative to the document
    assert 'src="/repo/docs/img/y.png"' in md.render("![another](docs/img/y.png)")[0]            # README: the repository root
    assert 'src="/repo/x.gif"' in md.render("![up](../x.gif)", base="docs")[0]
    assert "echo &lt;hi&gt; | cat" in html and "```" not in html


@pytest.mark.parametrize("doc_id", ["README", "USAGE", "MODELS", "DESIGN"])
def test_every_project_document_renders(doc_id):
    d = api.doc(doc_id)
    html = d["html"]
    assert d["headings"] and html.count("<h2") >= 3
    assert "```" not in html and "|---" not in html              # no fence or table rule left as text
    assert "<table>" in html and "<pre>" in html
    for src in [m.split('"')[0] for m in html.split('<img alt="')[1:] for m in [m.split('src="')[1]]]:
        assert src.startswith("/repo/docs/img/") and api.repo_file(src.removeprefix("/repo/")).is_file(), src
    assert ".md" not in "".join(m for m in html.split('href="')[1:] if m.startswith(("docs/", "USAGE", "MODELS")))
    for target in [m.split('"')[0] for m in html.split('href="#/help/')[1:]]:
        name, _, anchor = target.partition("/")
        assert name in ("README", "USAGE", "MODELS", "DESIGN")
        if anchor:                                                 # a link into another document lands on a heading
            ids = {md.slug(ln.lstrip("# ")) for ln in (REPO / dict((i, p) for i, p, _ in api.DOCS)[name]).read_text().splitlines()
                   if ln.startswith("#")}
            assert anchor in ids, f"{doc_id} links to {target}, which is not a heading"


def test_documents_are_served(ui):
    st, r, _ = ui.get("/api/docs")
    assert st == 200 and [d["id"] for d in r["docs"]] == ["README", "USAGE", "MODELS", "DESIGN"]
    st, r, _ = ui.get("/api/docs/USAGE")
    assert st == 200 and "readaloud init" in r["html"]
    assert ui.get("/api/docs/NOPE")[0] == 404
    assert ui.get("/repo/pyproject.toml")[0] == 404              # only pictures the documents show
    assert ui.get("/repo/docs/img/../../LICENSE")[0] == 404


# ---- the page and the guards ---------------------------------------------------------------------------------------

def test_the_page_and_its_files_are_served(ui):
    st, body, headers = ui.get("/", raw=True)
    assert st == 200 and b"<main id=\"app\"" in body and headers["Content-Type"].startswith("text/html")
    assert ui.get("/app.js", raw=True)[1].startswith(b"/* readaloud ui")
    assert b"--accent" in ui.get("/app.css", raw=True)[1]
    assert b"<main id=\"app\"" in ui.get("/book/anything", raw=True)[1]     # the page handles its own addresses


def test_a_change_must_come_from_the_page(ui):
    st, r, _ = ui.call("POST", "/api/books/sample", {}, headers={"X-Requested-With": ""})
    assert st == 403 and "own page" in r["error"]


def test_only_this_servers_name_is_answered(ui):
    assert ui.get("/api/books", headers={"Host": "evil.example"})[0] == 403
    assert ui.get("/api/books", headers={"Host": "localhost:1234"})[0] == 200


def test_a_book_id_cannot_leave_the_books_folder(ui):
    for bad in ("..", "..%2F..%2Fetc", "UPPER", "a/b", "-x"):
        assert ui.get(f"/api/books/{bad}")[0] in (400, 404)
    assert ui.get("/api/books/missing")[0] == 404
    assert ui.get("/api/nothing")[0] == 404
    assert ui.call("PUT", "/api/books")[0] == 405


# ---- books and jobs ----------------------------------------------------------------------------------------------------

def test_library_starts_empty_then_takes_an_uploaded_book(ui):
    st, r, _ = ui.get("/api/books")
    assert st == 200 and r["books"] == [] and r["sample"] is True
    assert ui.call("POST", "/api/books?name=notes.txt", b"hello")[0] == 400
    st, r, _ = ui.call("POST", "/api/books?name=My%20Alice.epub&id=upload", ALICE.read_bytes())
    assert st == 200 and r["id"] == "upload"
    assert not (ui.root / ".uploads").exists()                    # the upload is not kept beside the book
    st, r, _ = ui.call("POST", "/api/books?name=again.epub&id=upload", ALICE.read_bytes())
    assert st == 400 and "already exists" in r["error"]
    st, r, _ = ui.get("/api/books/upload")
    assert st == 200 and [s["state"] for s in r["stages"]] == ["todo"] * 8 and r["chapters"] == []
    assert r["config"] == {"tts": "kokoro", "llm": "openai_compat", "multi_voice": True, "own_config": False}


def test_a_job_runs_and_its_log_can_be_followed(ui):
    assert ui.post("/api/books/sample")[1]["id"] == "alice"
    st, r, _ = ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock", "until": "classify"})
    assert st == 200 and r["job"]["status"] == "running" and r["job"]["label"] == "up to classify"
    job = ui.wait("alice")
    assert job["status"] == "ok" and job["returncode"] == 0
    assert job["log"].startswith("$ readaloud run alice --tts mock --llm mock --until classify\n")
    assert "display check passed" in job["log"] and "[classify" in job["log"]
    _, r, _ = ui.get("/api/books/alice")
    assert r["title"] == "Alice's Adventures in Wonderland"
    assert {s["stage"]: s["state"] for s in r["stages"]} == {
        "ingest": "done", "segment": "done", "classify": "done", "cast": "todo", "voices": "todo", "synth": "todo",
        "assemble": "todo", "package": "todo"}
    assert len(r["chapters"]) == 13 and r["chapters"][1]["heading"].startswith("CHAPTER I.")
    _, lib, _ = ui.get("/api/books")
    card = next(b for b in lib["books"] if b["id"] == "alice")
    assert (card["done"], card["planned"], card["job"]["status"]) == (3, 8, "ok")
    _, r, _ = ui.get("/api/books/alice/job?offset=999999999")     # an offset from an older, longer log starts over
    assert r["log"].startswith("$ readaloud run")


def test_stop_then_continue(ui):
    """A second job is refused while one runs; a stopped job leaves the book resumable."""
    assert ui.call("POST", "/api/books?name=a.epub&id=stopme", ALICE.read_bytes())[0] == 200
    assert ui.post("/api/books/stopme/run", {"tts": "mock", "llm": "mock"})[0] == 200
    st, r, _ = ui.post("/api/books/stopme/run", {"tts": "mock", "llm": "mock"})
    assert st == 409 and "already running" in r["error"]
    assert ui.call("DELETE", "/api/books/stopme")[0] == 409
    st, r, _ = ui.post("/api/books/stopme/stop")
    assert st == 200 and r["job"]["status"] == "stopped" and r["job"]["ended"]
    assert ui.get("/api/books/stopme/job")[1]["job"]["status"] == "stopped"
    assert ui.post("/api/books/stopme/run", {"tts": "mock", "llm": "mock", "until": "segment"})[0] == 200
    assert ui.wait("stopme")["status"] == "ok"
    assert ui.call("DELETE", "/api/books/stopme")[0] == 200 and not (ui.root / "stopme").exists()


def test_a_finished_book_can_be_downloaded_in_pieces(ui):
    """One chapter of the sample to the end, then the file with a byte range (what an audio or download bar asks for)."""
    if ui.get("/api/books/alice")[0] != 200:
        ui.post("/api/books/sample")
    st, r, _ = ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock", "docs": "2"})
    assert st == 200 and r["job"]["label"] == "chapters 2" and len(r["job"]["commands"]) == 2
    job = ui.wait("alice", timeout=300)
    assert job["status"] == "ok", job["log"][-1500:]
    first, second = [ln for ln in job["log"].splitlines() if ln.startswith("$ ")]
    assert first == "$ readaloud run alice --tts mock --llm mock --until voices"
    assert second == "$ readaloud run alice --tts mock --llm mock --stages synth,assemble,package --docs 2"
    _, r, _ = ui.get("/api/books/alice")
    assert {s["stage"]: s["state"] for s in r["stages"]} == {
        "ingest": "done", "segment": "done", "classify": "done", "cast": "done", "voices": "done",
        "synth": "partial", "assemble": "partial", "package": "done"}
    # the voices of that chapter are the ones decided from the whole book, not from the chapter alone
    assert ui.get("/api/books/alice/quotes")[1]["total"] == 1109
    ch2 = ui.get("/api/books/alice/quotes?chapter=2")[1]
    assert ch2["matching"] == 39 and sum(1 for q in ch2["quotes"] if q["voice"] == "female") == 33
    assert r["chapters"][1]["assembled"] and not r["chapters"][2]["assembled"]
    out = r["outputs"][0]
    path = f"/api/books/alice/out/{out['name']}"
    st, body, headers = ui.get(path, raw=True)
    assert st == 200 and len(body) == out["bytes"] and body[:2] == b"PK"
    assert headers["Content-Type"] == "application/epub+zip" and "attachment" in headers["Content-Disposition"]
    st, part, headers = ui.get(path, raw=True, headers={"Range": "bytes=10-19"})
    assert st == 206 and part == body[10:20] and headers["Content-Range"] == f"bytes 10-19/{len(body)}"
    st, tail, _ = ui.get(path, raw=True, headers={"Range": "bytes=-5"})
    assert st == 206 and tail == body[-5:]
    assert ui.get(path, raw=True, headers={"Range": f"bytes={len(body)}-"})[0] == 416
    assert ui.get("/api/books/alice/out/..%2Fproject.db")[0] == 404


# ---- pieces --------------------------------------------------------------------------------------------------------------

def test_run_requests_become_command_lines():
    assert api.run_args("b", {}) == ([["run", "b"]], "whole book")
    assert api.run_args("b", {"tts": "mock", "llm": "mock", "until": "package"}) == (
        [["run", "b", "--tts", "mock", "--llm", "mock"]], "whole book")
    assert api.run_args("b", {"until": "voices"}) == ([["run", "b", "--until", "voices"]], "up to voices")
    assert api.run_args("b", {"stages": ["synth", "assemble"], "docs": [4]}) == (
        [["run", "b", "--stages", "synth,assemble", "--docs", "4"]], "stages synth, assemble, chapters 4")


def test_chapters_limit_the_audio_and_never_the_text_stages():
    """A request for some chapters is two commands: the whole book's text stages, then those chapters' audio."""
    cfg = config.load()
    assert api.run_args("b", {"tts": "mock", "docs": "2, 3"}, cfg) == (
        [["run", "b", "--tts", "mock", "--until", "voices"],
         ["run", "b", "--tts", "mock", "--stages", "synth,assemble,package", "--docs", "2,3"]], "chapters 2, 3")
    assert api.run_args("b", {"docs": [4], "until": "assemble"}, cfg) == (
        [["run", "b", "--until", "voices"], ["run", "b", "--stages", "synth,assemble", "--docs", "4"]],
        "chapters 4, up to assemble")
    assert api.run_args("b", {"docs": "2"}, {**cfg, "multi_voice": False})[0][0] == ["run", "b", "--until", "classify"]
    assert api.run_args("b", {"docs": "2", "until": "classify"}, cfg) == ([["run", "b", "--until", "classify"]], "up to classify")


def test_a_job_can_be_several_commands_and_stops_at_the_first_failure(tmp_path):
    from pipeline.ui import runner
    assert runner.split(["--books", "r", "--", "run", "a", "--until", "voices", "--", "run", "a", "--docs", "2"]) == (
        "r", [["run", "a", "--until", "voices"], ["run", "a", "--docs", "2"]])
    for bad in ([], ["--books", "r"], ["--books", "r", "run"], ["--books", "r", "--", "--"]):
        with pytest.raises(SystemExit):
            runner.split(bad)
    jobs = Jobs(tmp_path)
    (tmp_path / "b").mkdir()
    st = jobs.start("b", [["status", "nothing-here"], ["status", "never-reached"]], "two commands")
    assert st["args"] == ["status", "never-reached"] and st["commands"][0] == ["status", "nothing-here"]
    for _ in range(200):
        if not jobs.running("b"):
            break
        time.sleep(0.1)
    st, (log, _) = jobs.state("b"), jobs.log("b")
    assert st["status"] == "failed" and st["returncode"] == 1
    assert log.startswith("$ readaloud status nothing-here\n") and "no such book" in log and "never-reached" not in log
    for bad in ({"tts": "espeak"}, {"llm": "gpt"}, {"until": "publish"}, {"stages": "synth,rm -rf"}):
        with pytest.raises(api.ApiError):
            api.run_args("b", bad)
    with pytest.raises(ValueError):
        api.run_args("b", {"docs": "2; reboot"})
    with pytest.raises(api.ApiError):
        api.run_args("Bad Id", {})


def test_the_ui_plans_the_stages_the_command_line_runs():
    cfg = config.load()
    assert planned_stages(cfg) == ["ingest", "segment", "classify", "cast", "voices", "synth", "assemble", "package"]
    assert planned_stages({**cfg, "multi_voice": False}) == ["ingest", "segment", "classify", "synth", "assemble", "package"]
    assert planned_stages(cfg, until="classify") == ["ingest", "segment", "classify"]
    assert planned_stages(cfg, only=["synth", "assemble"]) == ["synth", "assemble"]
    with pytest.raises(SystemExit):
        planned_stages(cfg, only=["nonsense"])


def test_asking_about_a_job_as_it_ends_neither_fails_nor_loses_its_result(tmp_path):
    """The watcher of a job and requests about it write the same state file at the job's end. Several short jobs,
    each polled hard from four threads: no request may fail, and each job must end as what it was (failed), never as
    "ended", which is only for a job no server was watching."""
    jobs = Jobs(tmp_path)
    (tmp_path / "b").mkdir()
    errors, seen = [], set()

    def poll(stop):
        while not stop.is_set():
            try:
                st = jobs.state("b")
                if st:
                    seen.add(st["status"])
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    for _ in range(4):
        jobs.start("b", ["status", "nothing-here"], "short")
        stop = threading.Event()
        threads = [threading.Thread(target=poll, args=(stop,)) for _ in range(4)]
        for t in threads:
            t.start()
        end = time.time() + 30
        while time.time() < end and (jobs.state("b") or {}).get("returncode") is None:
            time.sleep(0.01)
        stop.set()
        for t in threads:
            t.join()
        assert jobs.state("b")["status"] == "failed" and jobs.state("b")["returncode"] == 1
    assert errors == [] and seen <= {"running", "failed"}
    assert list((tmp_path / "b" / "ui").glob("*.tmp")) == []


def test_a_job_whose_process_is_gone_is_not_shown_as_running(tmp_path):
    jobs = Jobs(tmp_path)
    (tmp_path / "b" / "ui").mkdir(parents=True)
    (tmp_path / "b" / "ui" / "job.json").write_text(json.dumps(
        {"book": "b", "label": "whole book", "args": ["run", "b"], "pid": 2 ** 22 + 12345, "status": "running",
         "started": time.time() - 60, "ended": None, "returncode": None}))
    st = jobs.state("b")
    assert st["status"] == "ended" and st["ended"] and not jobs.running("b")
    assert jobs.state("none") is None and jobs.log("none") == ("", 0)
