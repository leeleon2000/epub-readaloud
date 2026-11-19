"""The UI's panels for looking at what the pipeline decided and changing it (pipeline/ui/edit.py), through the API:
warnings, section categories, lines that are not read, the voice of each quotation, the gender table, settings.

One book is run to the voices stage with the mock adapters; every test then edits the same files a person would.
"""
import csv
import threading

import pytest
import yaml

from pipeline import config, store
from pipeline.observe import Observer
from pipeline.stages import classify
from pipeline.ui import server
from pipeline.v3 import genders

from test_ui import ALICE, Client


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui-edit-books")
    srv = server.make_server(root)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = Client(f"http://127.0.0.1:{srv.server_address[1]}")
    c.root, c.dir = root, root / "alice"
    assert c.call("POST", "/api/books?name=alice.epub&id=alice", ALICE.read_bytes())[0] == 200
    st, r, _ = c.get("/api/books/alice/sections")                  # before the first run: nothing yet, and no error
    assert st == 200 and r["sections"] == []
    assert c.get("/api/books/alice/quotes")[1]["total"] == 0
    assert c.post("/api/books/alice/run", {"tts": "mock", "llm": "mock", "until": "voices"})[0] == 200
    job = c.wait("alice")
    assert job["status"] == "ok", job["log"][-1500:]
    yield c
    srv.RequestHandlerClass.jobs.stop("alice", wait=5)
    srv.shutdown()
    srv.server_close()


def spoken(ui, sentence_id: str) -> int:
    con = store.connect(ui.dir)
    try:
        return con.execute("SELECT speak FROM sentence WHERE id=?", (sentence_id,)).fetchone()[0]
    finally:
        con.close()


# ---- sections and lines ------------------------------------------------------------------------------------------------

def test_sections_are_listed_with_what_decided_them(ui):
    st, r, _ = ui.get("/api/books/alice/sections")
    assert st == 200 and len(r["sections"]) == 18 and r["skip"] == []
    assert [c["id"] for c in r["categories"]] == ["story", "front_matter", "notes", "back_matter", "toc", "captions", "boilerplate"]
    assert [c["read"] for c in r["categories"]] == [True, True, True, True, False, False, False]
    first, contents, ch1 = r["sections"][0], r["sections"][2], r["sections"][3]
    assert (first["category"], first["source"], first["spoken"]) == ("boilerplate", "rule", 0)
    assert (contents["title"], contents["category"]) == ("Contents", "toc")
    assert ch1["title"].startswith("CHAPTER I.") and ch1["category"] == "story" and ch1["spoken"] == ch1["sentences"] == 54


def test_a_category_set_in_the_ui_is_a_line_in_sections_yaml(ui):
    sid = "d001s00014"                                              # Contents
    st, r, _ = ui.post("/api/books/alice/sections", {"set": {sid: "front_matter"}})
    sec = next(s for s in r["sections"] if s["id"] == sid)
    assert st == 200 and (sec["category"], sec["source"], sec["spoken"]) == ("front_matter", "manual", 25)
    assert yaml.safe_load((ui.dir / "sections.yaml").read_text()) == {sid: "front_matter"}
    st, r, _ = ui.post("/api/books/alice/sections", {"set": {sid: "toc"}})
    assert next(s for s in r["sections"] if s["id"] == sid)["spoken"] == 0
    for bad in ({"set": {sid: "nonsense"}}, {"set": {"../x": "toc"}}, {"set": {}}, {}):
        assert ui.post("/api/books/alice/sections", bad)[0] == 400
    assert ui.post("/api/books/alice/sections", {"set": {"d099s00000": "toc"}})[0] == 400


def test_a_single_line_can_be_kept_from_being_read(ui):
    st, r, _ = ui.get("/api/books/alice/sections/d002s00000")
    assert st == 200 and len(r["sentences"]) == 54 and all(s["read"] and not s["skipped"] for s in r["sentences"])
    assert r["sentences"][0]["kind"] == "heading"
    target = r["sentences"][5]["id"]
    st, r, _ = ui.post("/api/books/alice/skip", {"add": [target]})
    assert st == 200 and r["skip"] == [target] and spoken(ui, target) == 0
    assert classify.load_skip(type("B", (), {"dir": ui.dir})()) == [target]
    line = next(s for s in ui.get("/api/books/alice/sections/d002s00000")[1]["sentences"] if s["id"] == target)
    assert (line["read"], line["skipped"]) == (False, True)
    st, r, _ = ui.post("/api/books/alice/skip", {"remove": [target]})
    assert st == 200 and r["skip"] == [] and spoken(ui, target) == 1
    assert "skip" not in (yaml.safe_load((ui.dir / "sections.yaml").read_text()) or {})    # and the file still parses
    assert ui.post("/api/books/alice/skip", {"add": ["ab999s99999"]})[0] == 404
    assert ui.post("/api/books/alice/skip", {})[0] == 400
    assert ui.get("/api/books/alice/sections/d099s00000")[0] == 404


def test_the_skip_list_survives_an_empty_category_list(tmp_path):
    """sections.yaml with only `skip:` in it must still be YAML (an empty mapping used to be written as "{}")."""
    book = type("B", (), {"dir": tmp_path})()
    classify.save_skip(book, ["ab002s00003", "ab002s00001", "ab002s00003"])
    assert yaml.safe_load((tmp_path / "sections.yaml").read_text()) == {"skip": ["ab002s00001", "ab002s00003"]}
    classify.save_overrides(book, {"d001s00000": "toc"})
    assert classify.load_skip(book) == ["ab002s00001", "ab002s00003"] and classify.load_overrides(book) == {"d001s00000": "toc"}
    classify.save_skip(book, [])
    assert yaml.safe_load((tmp_path / "sections.yaml").read_text()) == {"d001s00000": "toc"}


# ---- voices ----------------------------------------------------------------------------------------------------------------

def test_quotations_can_be_filtered(ui):
    st, r, _ = ui.get("/api/books/alice/quotes")
    assert st == 200 and r["total"] == r["matching"] == 1109 and len(r["quotes"]) == 100 and r["pending"] == 0
    assert sum(r["counts"].values()) == 1109 and r["chapters"][0] == 2
    q = r["quotes"][0]
    assert (q["text"], q["voice"], q["tag"], q["manual"]) == ("and what is the use of a book,", "female", "Alice", False)
    assert ui.get("/api/books/alice/quotes?offset=1100")[1]["quotes"][-1]["chapter"] == 13
    ch3 = ui.get("/api/books/alice/quotes?chapter=3")[1]
    assert 0 < ch3["matching"] < 1109 and {x["chapter"] for x in ch3["quotes"]} == {3}
    male = ui.get("/api/books/alice/quotes?voice=male")[1]
    assert male["matching"] == r["counts"]["male"] and {x["voice"] for x in male["quotes"]} == {"male"}
    found = ui.get("/api/books/alice/quotes?q=MARMALADE")[1]
    assert found["matching"] == 1 and found["quotes"][0]["why"] == "quoted phrase" and not found["quotes"][0]["unsure"]
    unsure = ui.get("/api/books/alice/quotes?review=1")[1]
    assert 0 < unsure["matching"] < r["counts"]["narrator"] and all(x["unsure"] and x["voice"] == "narrator" for x in unsure["quotes"])
    assert ui.get("/api/books/alice/quotes?chapter=x")[0] == 400


def test_a_voice_change_goes_to_the_overrides_file_and_applies_without_a_model(ui):
    hand = "002p0003q1,narrator,,my own note\n"                       # a row someone wrote by hand is kept
    (ui.dir / "attribution_overrides.csv").write_text("quote_id,speaker,style,note\n" + hand)
    q = ui.get("/api/books/alice/quotes")[1]["quotes"][0]
    assert q["voice"] == "female"
    assert ui.post("/api/books/alice/quotes", {"set": {q["id"]: "male"}})[0] == 200
    with (ui.dir / "attribution_overrides.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"quote_id": "002p0003q1", "speaker": "narrator", "style": "", "note": "my own note"},
                    {"quote_id": q["id"], "speaker": "male", "style": "", "note": "set in the UI"}]
    r = ui.get("/api/books/alice/quotes?changed=1")[1]
    assert r["pending"] == 1 and r["matching"] == 2                    # asked for, not applied yet
    row = next(x for x in r["quotes"] if x["id"] == q["id"])
    assert (row["voice"], row["asked"], row["pending"]) == ("female", "male", True)
    st, r, _ = ui.post("/api/books/alice/quotes/apply")
    assert st == 200 and r["job"]["args"] == ["run", "alice", "--tts", "mock", "--llm", "mock", "--stages", "voices"]
    job = ui.wait("alice")
    assert job["status"] == "ok" and "applied 2 manual overrides" in job["log"]
    row = next(x for x in ui.get("/api/books/alice/quotes?changed=1")[1]["quotes"] if x["id"] == q["id"])
    assert (row["voice"], row["manual"], row["pending"]) == ("male", True, False)
    assert ui.post("/api/books/alice/quotes", {"set": {q["id"]: None}})[0] == 200      # taking it back removes the row
    assert (ui.dir / "attribution_overrides.csv").read_text().splitlines() == ["quote_id,speaker,style,note", hand.strip()]
    for bad in ({"set": {q["id"]: "robot"}}, {"set": {}}, {}):
        assert ui.post("/api/books/alice/quotes", bad)[0] == 400
    assert ui.post("/api/books/alice/quotes", {"set": {"999p0001q1": "male"}})[0] == 404


def test_names_can_be_given_a_gender_without_touching_what_the_pipeline_found(ui):
    st, r, _ = ui.get("/api/books/alice/genders")
    assert st == 200 and r["exists"] and r["manual"] == [] and len(r["auto"]) > 50
    alice = next(a for a in r["auto"] if a["name"] == "alice")
    assert alice["gender"] == "female" and alice["used"] > 100 and alice["source"].startswith("pronouns")
    found = lambda: (ui.dir / "genders.yaml").read_text().split("\nauto:")[1]      # noqa: E731 - the block the pipeline wrote
    before = found()
    assert before.count("\n") > 50
    st, r, _ = ui.post("/api/books/alice/genders", {"set": {"The Dormouse": "male", "Bill": "male"}})
    assert st == 200 and r["manual"] == [{"name": "bill", "gender": "male"}, {"name": "the dormouse", "gender": "male"}]
    assert genders.load_manual(ui.dir) == {"bill": "male", "the dormouse": "male"}     # as the pipeline reads it
    assert found() == before
    st, r, _ = ui.post("/api/books/alice/genders", {"set": {"Bill": None, "the dormouse": None}})
    assert r["manual"] == [] and genders.load_manual(ui.dir) == {} and len(r["auto"]) == len(ui.get("/api/books/alice/genders")[1]["auto"])
    assert yaml.safe_load((ui.dir / "genders.yaml").read_text())["manual"] == {}
    assert found() == before
    for bad in ({"set": {"Bill": "robot"}}, {"set": {"": "male"}}, {}):
        assert ui.post("/api/books/alice/genders", bad)[0] == 400


# ---- warnings ----------------------------------------------------------------------------------------------------------------

def test_warnings_are_explained_and_can_be_handled(ui):
    assert ui.get("/api/books/alice/warnings")[1] == {"warnings": [], "open": 0}
    obs = Observer(ui.dir, "test")
    obs.warn("SECTION_UNCERTAIN", "pause", "d001s00011", "'Alice': model said story", opening="Alice was beginning")
    obs.warn("LLM_FAILED", "warn", "c003_01", "voices: timed out")
    obs.finish("paused")
    st, r, _ = ui.get("/api/books/alice/warnings")
    assert st == 200 and r["open"] == 2 and [w["code"] for w in r["warnings"]] == ["LLM_FAILED", "SECTION_UNCERTAIN"]
    sec = r["warnings"][1]
    assert sec["section"] == "d001s00011" and sec["opening"] == "Alice was beginning" and "Sections tab" in sec["todo"]
    assert r["warnings"][0]["section"] is None and "retried" in r["warnings"][0]["what"]
    assert ui.get("/api/books/alice")[1]["warnings"] == 2            # the tab's badge
    st, r, _ = ui.post(f"/api/books/alice/warnings/{sec['id']}", {"action": "ack", "note": "it is the title page"})
    assert st == 200 and r["open"] == 1
    st, r, _ = ui.post(f"/api/books/alice/warnings/{r['warnings'][0]['id']}", {"action": "resolve"})
    assert st == 200 and r == {"warnings": [], "open": 0}
    handled = ui.get("/api/books/alice/warnings?all=1")[1]["warnings"]
    assert {(w["code"], w["status"], w["note"], w["handled_by"]) for w in handled} == {
        ("SECTION_UNCERTAIN", "acknowledged", "it is the title page", "ui"), ("LLM_FAILED", "resolved", "", "ui")}
    assert ui.post("/api/books/alice/warnings/99999", {"action": "ack"})[0] == 404
    assert ui.post(f"/api/books/alice/warnings/{sec['id']}", {"action": "delete"})[0] == 400


# ---- settings ----------------------------------------------------------------------------------------------------------------

def test_settings_write_only_what_differs_from_the_defaults(ui):
    st, r, _ = ui.get("/api/books/alice/settings")
    values = {f["key"]: f["value"] for f in r["fields"]}
    assert st == 200 and r["text"] == "" and not any(f["own"] for f in r["fields"])
    assert values["multi_voice"] is True and values["tts.adapter"] == "kokoro" and values["voices.threshold"] == 0.7
    assert all(f["help"] and f["label"] for f in r["fields"])
    st, r, _ = ui.call("PUT", "/api/books/alice/settings", {"values": {
        "multi_voice": False, "tts.adapter": "drama3", "tts.drama3.concurrency": 64, "voices.threshold": 0.7, "llm.model": ""}})
    assert st == 200 and yaml.safe_load(r["text"]) == {"multi_voice": False, "tts": {"adapter": "drama3", "drama3": {"concurrency": 64}}}
    assert {f["key"] for f in r["fields"] if f["own"]} == {"multi_voice", "tts.adapter", "tts.drama3.concurrency"}
    cfg = config.load(ui.dir)
    assert cfg["tts"]["drama3"]["concurrency"] == 64 and isinstance(cfg["tts"]["drama3"]["concurrency"], int)
    assert cfg["tts"]["drama3"]["endpoint"] == "http://localhost:8091"   # the rest of the defaults is untouched
    assert ui.get("/api/books/alice")[1]["config"] == {"tts": "drama3", "llm": "openai_compat", "multi_voice": False, "own_config": True}
    st, r, _ = ui.call("PUT", "/api/books/alice/settings", {"values": {
        "multi_voice": True, "tts.adapter": "kokoro", "tts.drama3.concurrency": None}})
    assert st == 200 and r["text"] == "" and not (ui.dir / "config.yaml").exists()      # back to the defaults: no file
    for bad in ({"values": {"tts.adapter": "espeak"}}, {"values": {"voices.threshold": "high"}}, {"values": {"shell": "x"}}):
        assert ui.call("PUT", "/api/books/alice/settings", bad)[0] == 400


def test_the_settings_file_can_be_edited_whole_but_never_left_broken(ui):
    st, r, _ = ui.call("PUT", "/api/books/alice/settings", {"text": "pauses: {sentence: 0.5}\n"})
    assert st == 200 and config.load(ui.dir)["pauses"]["sentence"] == 0.5 and config.load(ui.dir)["pauses"]["paragraph"] == 0.8
    for bad in ("tts: [1, 2\n", "- a\n- b\n", "block_mode: paragraph\n"):
        st, r, _ = ui.call("PUT", "/api/books/alice/settings", {"text": bad})
        assert st == 400 and (ui.dir / "config.yaml").read_text() == "pauses: {sentence: 0.5}\n"   # the old file is back
    assert ui.call("PUT", "/api/books/alice/settings", {"text": ""})[0] == 200 and not (ui.dir / "config.yaml").exists()


# ---- hearing it, and making parts again ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def heard(ui):
    """The same book with the audio of chapter 2."""
    ui.wait("alice")
    assert ui.get("/api/books/alice/cast")[1]["voices"] != []         # the cast stage ran on the way to voices
    assert ui.get("/api/books/alice/listen/2")[1]["audio"] is False
    assert ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock", "docs": "2"})[0] == 200
    job = ui.wait("alice", timeout=300)
    assert job["status"] == "ok", job["log"][-1500:]
    return ui


def test_the_three_voices_have_samples_that_play(heard):
    st, r, _ = heard.get("/api/books/alice/cast")
    assert st == 200 and (r["file"], r["designer"], r["adapter"]) == ("cast.yaml", "kokoro-mix", "mock")
    assert [(v["id"], v["label"], v["sample"], v["can_reseed"]) for v in r["voices"]] == [
        ("narrator", "Narrator", True, False), ("_generic_female", "Female voice", True, True),
        ("_generic_male", "Male voice", True, True)]
    st, wav, headers = heard.get("/api/books/alice/cast/_generic_female/sample", raw=True)
    assert st == 200 and wav[:4] == b"RIFF" and headers["Content-Type"] in ("audio/x-wav", "audio/wav")
    assert "Content-Disposition" not in headers                        # played in the page, not downloaded
    st, part, headers = heard.get("/api/books/alice/cast/narrator/sample", raw=True, headers={"Range": "bytes=0-99"})
    assert st == 206 and len(part) == 100
    for bad in ("nobody", "..%2F..%2Fproject.db", "narrator%2F..%2Fx"):
        assert heard.get(f"/api/books/alice/cast/{bad}/sample")[0] == 404


def test_a_voice_can_be_kept_or_designed_again(heard):
    voice = lambda r, vid: next(v for v in r["voices"] if v["id"] == vid)      # noqa: E731
    before = voice(heard.get("/api/books/alice/cast")[1], "_generic_male")["voice"]
    st, r, _ = heard.post("/api/books/alice/cast/_generic_male", {"action": "freeze"})
    assert st == 200 and voice(r, "_generic_male")["frozen"] and not voice(r, "_generic_female")["frozen"]
    assert yaml.safe_load((heard.dir / "cast.yaml").read_text())["characters"]["_generic_male"]["frozen"] is True
    st, r, _ = heard.post("/api/books/alice/cast/_generic_male", {"action": "reseed"})
    assert st == 400 and "frozen" in r["error"]                        # a kept voice is not replaced
    assert heard.post("/api/books/alice/cast/_generic_male", {"action": "unfreeze"})[0] == 200
    st, r, _ = heard.post("/api/books/alice/cast/_generic_male", {"action": "reseed"})
    after = voice(r, "_generic_male")
    assert st == 200 and after["voice"] != before and after["sample"] and after["voice"] in r["message"]
    st, r, _ = heard.post("/api/books/alice/cast/narrator", {"action": "reseed"})
    assert st == 400 and "tts.voice" in r["error"]                     # the Kokoro narrator is a setting, and it says so
    assert heard.post("/api/books/alice/cast/_generic_male", {"action": "delete"})[0] == 400
    assert heard.post("/api/books/alice/cast/nobody", {"action": "freeze"})[0] == 404


def test_a_chapter_comes_with_the_time_of_every_sentence(heard):
    st, r, _ = heard.get("/api/books/alice/listen/2")
    assert st == 200 and r["audio"] and r["flagged"] == 0 and len(r["sentences"]) == 54
    s = r["sentences"]
    assert (s[0]["kind"], s[0]["start"]) == ("heading", 0) and s[0]["text"].startswith("CHAPTER I.")
    assert all(a["end"] == b["start"] for a, b in zip(s, s[1:]))           # the clips follow each other with no gap
    assert s[-1]["end"] <= r["ms"]
    quote = next(p for x in s for p in x["pieces"] if p["quote"])
    assert quote["voice"] == "female" and quote["made"] and quote["text"] == "and what is the use of a book,"
    assert all(p["made"] for x in s for p in x["pieces"])
    st, wav, headers = heard.get("/api/books/alice/audio/2", raw=True, headers={"Range": "bytes=0-43"})
    assert st == 206 and wav[:4] == b"RIFF" and int(headers["Content-Range"].split("/")[1]) > 1_000_000
    other = heard.get("/api/books/alice/listen/3")[1]                    # a chapter with no audio yet: text, no times
    assert other["audio"] is False and other["sentences"][0]["start"] is None and not other["sentences"][0]["pieces"][0]["made"]
    assert heard.get("/api/books/alice/audio/3")[0] == 404 and heard.get("/api/books/alice/listen/99")[0] == 404


def test_a_line_is_made_again_alone(heard):
    pieces = [p["id"] for p in heard.get("/api/books/alice/listen/2")[1]["sentences"][3]["pieces"]]
    st, r, _ = heard.post("/api/books/alice/redo", {"pieces": pieces, "next_seed": True})
    assert st == 200 and r["chapters"] == [2] and f"{len(pieces)} takes dropped" in r["message"]
    assert r["job"]["args"] == ["run", "alice", "--tts", "mock", "--llm", "mock", "--stages", "synth,assemble,package", "--docs", "2"]
    job = heard.wait("alice", timeout=300)
    assert job["status"] == "ok" and f"{len(pieces)} to generate" in job["log"] and "1 chapters rebuilt" in job["log"]
    skip = yaml.safe_load((heard.dir / "reseed.yaml").read_text())          # "differently": the next seeds, on file
    assert set(skip) == set(pieces) and all(v > 0 for v in skip.values())
    assert all(p["made"] for x in heard.get("/api/books/alice/listen/2")[1]["sentences"] for p in x["pieces"])
    st, r, _ = heard.post("/api/books/alice/redo", {"pieces": pieces[:1], "run": False})   # only drop it; make it later
    assert st == 200 and "job" not in r
    assert [p["made"] for p in heard.get("/api/books/alice/listen/2")[1]["sentences"][3]["pieces"]][0] is False
    for bad, code in (({"pieces": ["x; rm -rf"]}, 400), ({"pieces": ["ab999s00001u0"]}, 404), ({}, 400)):
        assert heard.post("/api/books/alice/redo", bad)[0] == code


def test_lines_the_speech_model_could_not_read_are_listed(heard):
    assert heard.get("/api/books/alice/flagged")[1] == {"flagged": []}
    rep = heard.dir / "reports" / "synth_flagged.tsv"
    rep.parent.mkdir(exist_ok=True)
    rep.write_text("utterance\tseed\twhy\nab002s00005u0\t1236\toutput filled its frame budget\n")
    try:
        st, r, _ = heard.get("/api/books/alice/flagged")
        assert r["flagged"] == [{"id": "ab002s00005u0", "why": "output filled its frame budget", "chapter": 2,
                                 "text": r["flagged"][0]["text"]}] and r["flagged"][0]["text"].startswith("In another moment")
        ch = heard.get("/api/books/alice/listen/2")[1]
        assert ch["flagged"] == 1 and ch["sentences"][4]["pieces"][0]["flagged"] == "output filled its frame budget"
    finally:
        rep.unlink()


# ---- while a job runs -------------------------------------------------------------------------------------------------------

def test_changes_that_need_the_database_wait_for_a_running_job(ui):
    assert ui.post("/api/books/alice/run", {"tts": "mock", "llm": "mock"})[0] == 200
    try:
        assert ui.post("/api/books/alice/sections", {"set": {"d001s00014": "toc"}})[0] == 409
        assert ui.post("/api/books/alice/skip", {"add": ["ab002s00003"]})[0] == 409
        assert ui.call("PUT", "/api/books/alice/settings", {"values": {"multi_voice": False}})[0] == 409
        assert ui.post("/api/books/alice/quotes/apply")[0] == 409
        assert ui.post("/api/books/alice/cast/_generic_male", {"action": "freeze"})[0] == 409
        assert ui.post("/api/books/alice/redo", {"pieces": ["ab002s00002u0"]})[0] == 409
        assert ui.get("/api/books/alice/quotes?chapter=2")[0] == 200       # looking is always possible
        assert ui.get("/api/books/alice/sections")[0] == 200
    finally:
        assert ui.post("/api/books/alice/stop")[1]["job"]["status"] == "stopped"
