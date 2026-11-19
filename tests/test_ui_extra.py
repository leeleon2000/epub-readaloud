"""The UI's last panels (pipeline/ui/extra.py), through the API: whether the model services answer, scoring a book
against known speakers, and video files.

The services are stood in for by two small HTTP servers; the known speakers are the hand-made PDNC-format data of
tests/test_eval_pdnc.py; the video export runs on the sample book with the mock adapters.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pipeline import store
from pipeline.ui import server

from test_eval_pdnc import BASE, make_db
from test_ui import ALICE, Client

FIXTURES = Path(__file__).parent / "fixtures"


def fake_service(routes: dict):
    """A server that answers the given paths with JSON (or with the given bytes)."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = routes.get(self.path)
            if body is None:
                self.send_error(404)
                return
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui-extra-books")
    srv = server.make_server(root, pdnc=FIXTURES)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    c.root = root
    yield c
    for d in root.iterdir():
        if d.is_dir():
            srv.RequestHandlerClass.jobs.stop(d.name, wait=5)
    srv.shutdown()
    srv.server_close()


# ---- do the services answer? ---------------------------------------------------------------------------------------------

def test_services_that_do_not_answer_are_named(ui):
    st, r, _ = ui.get("/api/services")
    by = {s["id"]: s for s in r["services"]}
    assert st == 200 and set(by) == {"llm", "kokoro", "drama3"} and (r["tts"], r["llm"]) == ("kokoro", "openai_compat")
    assert by["llm"]["endpoint"] == "http://localhost:8000/v1" and by["drama3"]["endpoint"] == "http://localhost:8091"
    assert (by["llm"]["used"], by["kokoro"]["used"], by["drama3"]["used"]) == (True, True, False)
    assert all(s["doc"] and s["role"] for s in r["services"])
    assert ui.get("/api/services?book=nothing-here")[0] == 404


def test_services_are_checked_at_a_books_own_addresses(ui):
    llm, llm_url = fake_service({"/v1/models": {"data": [{"id": "qwen3.6"}]}})
    kokoro, kokoro_url = fake_service({"/info": {"model_version": "kokoro-0.9.4-82M", "sample_rate": 24000}})
    drama, drama_url = fake_service({"/v1/models": {"data": [{"id": "some-other-model"}]}})
    page, page_url = fake_service({"/v1/models": b"<html>not an API</html>"})
    try:
        assert ui.call("POST", "/api/books?name=a.epub&id=svc", ALICE.read_bytes())[0] == 200
        cfg = lambda text: (ui.root / "svc" / "config.yaml").write_text(text)       # noqa: E731
        state = lambda: {s["id"]: (s["state"], s["used"], s["detail"]) for s in ui.get("/api/services?book=svc")[1]["services"]}  # noqa: E731
        cfg(f"llm: {{endpoint: {llm_url}/v1}}\ntts: {{endpoint: {kokoro_url}, drama3: {{endpoint: {drama_url}}}}}\n")
        s = state()
        assert s["llm"] == ("up", True, "serves qwen3.6")
        assert s["kokoro"] == ("up", True, "kokoro-0.9.4-82M, 24000 Hz")
        assert s["drama3"][:2] == ("wrong", False) and "the settings ask for fishaudio/drama-3" in s["drama3"][2]
        cfg(f"llm: {{endpoint: {llm_url}/v1, model: a-model-it-does-not-have}}\ntts: {{adapter: drama3, drama3: {{endpoint: {drama_url}}}}}\n")
        s = state()
        assert s["llm"][0] == "wrong" and "a-model-it-does-not-have" in s["llm"][2]
        assert (s["kokoro"][1], s["drama3"][1]) == (False, True)                      # which one the settings use
        cfg(f"llm: {{endpoint: {page_url}/v1}}\n")
        assert state()["llm"][0] == "down" and "not with JSON" in state()["llm"][2]
        cfg("llm: {adapter: mock}\nmulti_voice: false\n")
        assert state()["llm"] == ("off", False, "mock: no model is used")
    finally:
        for srv in (llm, kokoro, drama, page):
            srv.shutdown()


# ---- scoring ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mini(ui):
    make_db(ui.root / "mini", BASE).close()
    return ui


def test_a_book_is_scored_against_known_speakers(mini):
    st, r, _ = mini.get("/api/books/mini/eval")
    assert st == 200 and r["sources"] == ["pdnc_mini", "pdnc_pride_ch1"] and r["results"] == [] and r["corrected"] is False
    st, r, _ = mini.post("/api/books/mini/eval", {"source": "pdnc_mini", "tag": "first", "note": "as it came"})
    assert st == 200 and "eval mini:first" in r["summary"]
    res = r["results"][0]
    assert (res["tag"], res["note"], res["source"], res["quotes"], res["aligned_pct"]) == ("first", "as it came", "pdnc_mini", 4, 100.0)
    assert (res["wrong"], res["wrong_gender_pct"], res["voiced_pct"]) == (1, 25.0, 100.0)    # Q1 is Ann's, read as Tom
    st, r, _ = mini.get("/api/books/mini/eval/first")
    assert st == 200 and "<table>" in r["html"] and "PDNC evaluation" in r["html"] and r["headings"]
    for bad, code in (({"source": "pdnc_mini", "tag": "two words"}, 400), ({"source": "pdnc_mini", "tag": ""}, 400),
                      ({"source": "nothing", "tag": "t"}, 404), ({"source": "../x", "tag": "t"}, 404),
                      ({"source": "pdnc_mini", "tag": "t", "chapters": "one"}, 400)):
        assert mini.post("/api/books/mini/eval", bad)[0] == code
    assert mini.get("/api/books/mini/eval/nothing")[0] == 404 and mini.get("/api/books/mini/eval/..%2Fx")[0] in (400, 404)


def test_two_results_are_compared_quotation_by_quotation(mini):
    con = store.connect(mini.root / "mini")                       # the fix a person would make: Q1 is Ann's
    con.execute("UPDATE utterance SET speaker_id='ann' WHERE tts_text LIKE 'Fine%'")
    con.commit()
    con.close()
    st, r, _ = mini.post("/api/books/mini/eval", {"source": "pdnc_mini", "tag": "second"})
    assert st == 200 and [x["tag"] for x in r["results"]] == ["second", "first"] and r["results"][0]["wrong_gender_pct"] == 0.0
    st, r, _ = mini.post("/api/books/mini/eval-compare", {"tags": ["first", "second"]})
    assert st == 200 and "first" in r["html"] and "second" in r["html"] and "fixed 1, broke 0" in r["html"]
    for bad in ({"tags": ["first"]}, {"tags": ["first", "no such"]}, {}):
        assert mini.post("/api/books/mini/eval-compare", bad)[0] == 400


def test_the_known_speakers_can_be_written_into_the_corrections_file(mini):
    st, r, _ = mini.post("/api/books/mini/corrected", {"source": "pdnc_mini", "apply": False})
    assert st == 200 and r["job"]["label"] == "corrected edition" and r["job"]["args"][:3] == ["script", "pdnc_overrides.py", "mini"]
    job = mini.wait("mini")
    assert job["status"] == "ok", job["log"]
    assert job["log"].startswith("$ python scripts/pdnc_overrides.py mini --pdnc ") and "corrections from PDNC" in job["log"]
    rows = (mini.root / "mini" / "attribution_overrides.csv").read_text().splitlines()
    assert rows[0] == "quote_id,speaker,style,note" and len(rows) >= 4 and all(",PDNC Q" in ln for ln in rows[1:])
    assert mini.get("/api/books/mini/eval")[1]["corrected"] is True
    st, r, _ = mini.post("/api/books/mini/corrected", {"source": "pdnc_mini"})        # as the page asks: and apply it
    assert [c[0] for c in r["job"]["commands"]] == ["script", "run"] and r["job"]["commands"][1][-2:] == ["--stages", "voices"]
    mini.wait("mini")
    assert mini.post("/api/books/mini/corrected", {"source": "nothing"})[0] == 404


def test_only_the_two_scripts_can_be_run_as_a_job(tmp_path):
    from pipeline.ui import runner
    assert runner.main(["--books", str(tmp_path), "--", "script", "drama3_stop.sh"]) == 1
    assert runner.main(["--books", str(tmp_path), "--", "script", "../pipeline/cli.py"]) == 1
    assert runner.main(["--books", str(tmp_path), "--", "script"]) == 1


# ---- video files -------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def whole(ui):
    """The sample book, all of it, with the mock adapters."""
    assert ui.post("/api/books/sample")[0] == 200
    st, r, _ = ui.post("/api/books/alice/video/plan")
    assert st == 200 and r["ready"] is False and r["reason"] == "Run the book first."
    assert ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock", "docs": "2"})[0] == 200
    assert ui.wait("alice", timeout=300)["status"] == "ok"
    r = ui.post("/api/books/alice/video/plan")[1]                   # one chapter's audio is not a book
    assert r["ready"] is False and "Every chapter needs its audio first" in r["reason"]
    st, r, _ = ui.post("/api/books/alice/video")
    assert st == 409 and "Every chapter" in r["error"]
    assert ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock"})[0] == 200
    job = ui.wait("alice", timeout=600)
    assert job["status"] == "ok", job["log"][-1500:]
    return ui


def test_the_parts_of_a_video_are_planned_before_anything_is_encoded(whole):
    plan = lambda **body: whole.post("/api/books/alice/video/plan", body)[1]          # noqa: E731
    shape = lambda r: [(p["chapters"], p["first"][:11]) for p in r["parts"]]          # noqa: E731
    r = plan()
    assert r["ready"] and len(r["marks"]) == 14 and 2 * 3.6e6 < r["total"] < 2.4 * 3.6e6 and len(r["parts"]) == 1
    assert shape(plan(parts=3)) == [(5, "Alice’s Adv"), (4, "CHAPTER V. "), (5, "CHAPTER IX.")]
    cut = plan(cut=["CHAPTER V.", "CHAPTER IX."], part_name="Volume")
    assert shape(cut) == shape(plan(parts=3)) and sum(p["ms"] for p in cut["parts"]) == cut["total"]
    assert all(a["end"] == b["start"] for a, b in zip(cut["parts"], cut["parts"][1:]))
    assert len(plan(max_hours=0.5)["parts"]) == 6
    assert "0 chapter headings start with it" in plan(cut=["CHAPTER XX."])["error"]
    assert "cannot cut into 2 parts" in plan(parts=2, max_hours=0.5)["error"]
    for bad in ({"parts": 2, "cut": ["CHAPTER V."]}, {"max_hours": 40}, {"parts": "many"}):
        assert whole.post("/api/books/alice/video/plan", bad)[0] == 400
    assert not (whole.root / "alice" / "out" / "youtube").exists()                    # planning writes nothing


def test_captions_and_descriptions_are_made_and_can_be_fetched(whole):
    st, r, _ = whole.post("/api/books/alice/video", {"cut": ["CHAPTER V.", "CHAPTER IX."], "part_name": "Volume",
                                                    "edition": "Project Gutenberg ebook no. 11", "no_video": True})
    assert st == 200 and r["job"]["args"][:3] == ["script", "youtube_export.py", "alice"] and "--no-video" in r["job"]["args"]
    job = whole.wait("alice", timeout=300)
    assert job["status"] == "ok", job["log"]
    assert job["log"].startswith("$ python scripts/youtube_export.py alice ") and "in 3 part(s)" in job["log"]
    names = [f["name"] for f in whole.post("/api/books/alice/video/plan")[1]["files"]]
    stem = "alice-s-adventures-in-wonderland"
    assert names == [f"{stem}-part{k}.{ext}" for k in (1, 2, 3) for ext in ("srt", "txt")] + [f"{stem}-thumbnail.jpg", "cover.jpg"]
    st, body, headers = whole.get(f"/api/books/alice/video/{stem}-part2.txt", raw=True)
    text = body.decode()
    assert st == 200 and "attachment" in headers["Content-Disposition"]
    assert text.startswith("Alice's Adventures in Wonderland, by Lewis Carroll. Volume 2 of 3.")
    assert "Text: Project Gutenberg ebook no. 11." in text and "0:00 CHAPTER V." in text
    st, body, _ = whole.get(f"/api/books/alice/video/{stem}-part1.srt", raw=True)
    assert st == 200 and body.startswith(b"1\n00:00:00,000 --> ")
    st, body, headers = whole.get(f"/api/books/alice/video/{stem}-thumbnail.jpg", raw=True)
    assert st == 200 and body[:2] == b"\xff\xd8" and "Content-Disposition" not in headers
    for bad in ("..%2F..%2Fproject.db", "cover.exe", f"{stem}-frame.png", "nothing.srt"):
        assert whole.get(f"/api/books/alice/video/{bad}")[0] == 404
