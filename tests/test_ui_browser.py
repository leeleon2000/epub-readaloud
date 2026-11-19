"""The UI in a real browser. The scenarios are the ones scripts/ui_demo.py records for the README, run here without
recording and then checked, so the GIFs show operations that are tested.

Skipped when Playwright or a browser it can start is missing (`pip install playwright`; it uses the installed
Google Chrome, or `playwright install chromium`).
"""
import importlib.util
import shutil
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

spec = importlib.util.spec_from_file_location("ui_demo", Path(__file__).parents[1] / "scripts" / "ui_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = demo.launch(p)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"no browser to drive: {str(e).splitlines()[0]}")
        yield b
        b.close()


@pytest.fixture(scope="module")
def books(tmp_path_factory):
    return tmp_path_factory.mktemp("ui-browser-books")


@pytest.fixture(scope="module")
def site(books):
    srv, base = demo.start_server(books, pdnc=Path(__file__).parent / "fixtures")     # hand-made known speakers
    yield base
    demo.stop_server(srv)


@pytest.fixture()
def page(browser):
    pg = browser.new_page(viewport=demo.VIEW, color_scheme="light")
    pg.problems = []
    pg.on("console", lambda m: pg.problems.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: pg.problems.append(str(e)))
    yield pg
    pg.close()


def stage_states(pg) -> dict:
    return {e.get_attribute("data-stage"): e.get_attribute("data-state") for e in pg.query_selector_all("[data-test=stages] li")}


def test_add_the_sample_and_run_a_chapter_to_an_audiobook(page, site):
    demo.scenario_run(page, site, demo.NoRecorder())
    assert page.inner_text("[data-test=book-title]") == "Alice's Adventures in Wonderland"
    assert stage_states(page) == {                                # one chapter: its audio exists, the book's does not yet
        "ingest": "done", "segment": "done", "classify": "done", "cast": "done", "voices": "done",
        "synth": "partial", "assemble": "partial", "package": "done"}
    log = page.inner_text("[data-test=log]")
    assert log.startswith("$ readaloud run alice --tts mock --llm mock --until voices\n")   # the text stages: the whole book
    assert "$ readaloud run alice --tts mock --llm mock --stages synth,assemble,package --docs 2\n" in log
    assert "epubcheck passed, no new errors" in log
    assert "finished" in page.inner_text("[data-test=job-line]")
    assert page.get_attribute("[data-test=download]", "href").endswith("-docs-2.epub")
    with page.expect_download() as dl:                            # the button gives the file itself
        page.click("[data-test=download]")
    path = Path(dl.value.path())
    assert path.read_bytes()[:2] == b"PK" and path.stat().st_size > 100_000
    # a redraw keeps what the job was asked for, and shows nothing that is not there
    assert page.input_value("[data-test=run-tts]") == "mock" and page.input_value("[data-test=run-llm]") == "mock"
    assert page.input_value("[data-test=run-docs]") == "2"
    assert page.is_hidden("[data-test=stop]") and page.is_enabled("[data-test=start]")
    assert "null" not in page.inner_text("#app") and "undefined" not in page.inner_text("#app")
    assert page.problems == []


def test_the_library_shows_the_book_and_reloading_loses_nothing(page, site):
    page.goto(site + "/")
    page.wait_for_selector("[data-test=book-card][data-book=alice]")
    card = page.inner_text("[data-test=book-card][data-book=alice]")
    assert "6 of 8 stages done" in card and "audiobook ready" in card and "finished" in card
    assert page.query_selector("[data-test=add-sample]") is None            # the sample is offered once
    page.click("[data-test=book-card][data-book=alice]")
    page.wait_for_selector("[data-test=stages]")
    page.reload()
    page.wait_for_selector("[data-test=log]")
    assert "epubcheck passed" in page.inner_text("[data-test=log]")         # the log of the finished job is still there
    assert page.problems == []


def test_upload_through_the_page_then_stop_and_continue(page, site):
    page.goto(site + "/")
    page.wait_for_selector("[data-test=drop]")
    page.fill("[data-test=new-id]", "second")
    page.set_input_files("[data-test=file]", str(Path(__file__).parent / "fixtures" / "alice_epub2.epub"))
    page.wait_for_selector("[data-test=stages]")
    assert page.url.endswith("#/book/second") and set(stage_states(page).values()) == {"todo"}
    page.select_option("[data-test=run-tts]", "mock")
    page.select_option("[data-test=run-llm]", "mock")
    page.click("[data-test=start]")
    page.wait_for_selector("[data-test=job-status][data-status=running]")
    assert page.is_visible("[data-test=stop]") and page.is_disabled("[data-test=start]")
    page.click("[data-test=stop]")
    page.wait_for_selector("[data-test=job-status][data-status=stopped]", timeout=30_000)
    assert page.is_hidden("[data-test=stop]") and "stopped" in page.inner_text("[data-test=job-line]")
    page.select_option("[data-test=run-until]", "classify")
    page.click("[data-test=start]")                                           # the same button carries on
    page.wait_for_selector("[data-stage=classify][data-state=done]", timeout=120_000)
    page.wait_for_selector("[data-test=job-status][data-status=ok]")
    assert page.problems == []


def test_the_documents_are_readable_in_the_page(page, site):
    demo.scenario_help(page, site, demo.NoRecorder())
    assert page.inner_text("[data-test=doc] h1") == "Serving the models"
    assert page.locator("[data-test=doc] table").count() >= 3
    page.goto(site + "/#/help/README")
    page.wait_for_selector("[data-test=doc] h1:has-text('epub-readaloud')")
    page.click("[data-test=doc] a[href='#/help/USAGE/3-three-voices-add-a-language-model']")   # a link between documents
    page.wait_for_selector("[data-test=doc] h1:has-text('Using readaloud')")
    assert page.evaluate("document.getElementById('3-three-voices-add-a-language-model').getBoundingClientRect().top") < 200
    page.goto(site + "/#/book/alice")
    page.wait_for_selector("[data-test=run-panel]")
    page.click("text=What each stage does")                                    # from a panel to the section explaining it
    page.wait_for_function("location.hash === '#/help/USAGE/3-three-voices-add-a-language-model'")
    assert page.problems == []


def test_review_and_change_what_was_decided(page, site, books):
    demo.scenario_review(page, site, demo.NoRecorder())
    d = books / "alice"
    rows = (d / "attribution_overrides.csv").read_text().splitlines()
    assert rows[0] == "quote_id,speaker,style,note" and rows[1:] == ["002p0003q1,male,,set in the UI"]
    assert '"the dormouse": male' in (d / "genders.yaml").read_text()
    assert "ab002s00003" not in (d / "sections.yaml").read_text()         # the line it kept from being read is read again
    # the Voices tab shows the applied change as the reader's own, and nothing is waiting
    page.click("[data-test=tab-voices]")
    page.wait_for_selector("[data-test=quotes]")
    page.check("[data-test=voice-changed]")
    page.wait_for_function("(document.querySelector('[data-test=voice-count]') || document.body).textContent.startsWith('1 of ')")
    assert page.locator("[data-test=quote]").count() == 1
    assert page.get_attribute("[data-test=quote]", "data-quote") == "002p0003q1"
    assert page.get_attribute("[data-test=quote]", "data-voice") == "male"
    assert "set by you" in page.inner_text("[data-test=quote]")
    assert page.query_selector("[data-test=apply-voices]") is None
    # searching, and taking a name back
    page.uncheck("[data-test=voice-changed]")
    page.wait_for_function("document.querySelectorAll('[data-test=quote]').length > 1")
    page.fill("[data-test=voice-search]", "marmalade")
    page.click("[data-test=voice-search-go]")
    page.wait_for_function("document.querySelectorAll('[data-test=quote]').length === 1")
    assert "ORANGE MARMALADE" in page.inner_text("[data-test=quotes]")
    page.click("[data-test=gender-remove]")
    page.wait_for_selector("[data-test=gender-manual]", state="detached")
    assert "the dormouse" not in (d / "genders.yaml").read_text().split("\nauto:")[0]
    # a line can be kept from being read, and read again
    page.click("[data-test=tab-sections]")
    page.wait_for_selector("[data-test=sections]")
    page.click("[data-section=d002s00000] [data-test=section-lines]")
    box = "[data-test=line-read][data-sentence=ab002s00005]"
    page.wait_for_selector(f"{box}:checked")
    page.uncheck(box)
    page.wait_for_selector(f"{box}:not(:checked):not([disabled])")
    assert "ab002s00005" in (d / "sections.yaml").read_text()
    assert "not read" in page.inner_text("[data-test=lines-panel] tr.sel")
    page.check(box)
    page.wait_for_selector(f"{box}:checked:not([disabled])")
    assert "ab002s00005" not in (d / "sections.yaml").read_text()
    assert page.problems == []


def test_a_section_category_and_a_setting_can_be_changed_in_the_page(page, site, books):
    demo.ensure_book(page, site)
    page.goto(site + "/#/book/alice/sections")
    page.wait_for_selector("[data-test=sections]")
    pick = "[data-test=section-category][data-section=d001s00014]"
    assert page.input_value(pick) == "toc"
    page.select_option(pick, "front_matter")
    page.wait_for_function("(document.querySelector('[data-test=section][data-section=d001s00014]') || document.body).textContent.includes('set by you')")
    assert page.input_value(pick) == "front_matter"
    assert "d001s00014: front_matter" in (books / "alice" / "sections.yaml").read_text()
    page.select_option(pick, "toc")
    page.wait_for_function("(document.querySelector('[data-test=section][data-section=d001s00014]') || document.body).textContent.includes('0 / 25')")

    page.click("[data-test=tab-settings]")
    page.wait_for_selector("[data-test=settings]")
    box = "[data-test=setting][data-key='voices.threshold']"
    assert page.input_value(box) == "0.7"
    page.fill(box, "0.85")
    page.click("[data-test=settings-save]")
    page.wait_for_function("(document.querySelector('[data-test=settings]') || document.body).textContent.includes('the default is 0.7')")
    assert (books / "alice" / "config.yaml").read_text() == "voices:\n  threshold: 0.85\n"
    page.fill(box, "0.7")
    page.click("[data-test=settings-save]")
    page.wait_for_function("document.querySelector('[data-test=settings]') && !document.querySelector('[data-test=settings]').textContent.includes('the default is')")
    assert not (books / "alice" / "config.yaml").exists()
    assert page.problems == []


def test_warnings_show_what_to_do_and_can_be_handled(page, site, books):
    from pipeline.observe import Observer
    demo.ensure_book(page, site)
    obs = Observer(books / "alice", "test")
    obs.warn("SECTION_UNCERTAIN", "pause", "d001s00011", "'Alice': the model said story", opening="Alice's Adventures")
    obs.finish("paused")
    page.goto(site + "/#/book/alice")
    page.wait_for_selector("[data-test=tab-warnings] .n")
    assert page.inner_text("[data-test=tab-warnings] .n") == "1"
    page.click("[data-test=tab-warnings]")
    page.wait_for_selector("[data-test=warning][data-code=SECTION_UNCERTAIN]")
    text = page.inner_text("[data-test=warning]")
    assert "could not tell whether a section is story" in text and "Alice's Adventures" in text
    assert page.get_attribute("[data-test=warning-section]", "href") == "#/book/alice/sections/d001s00011"
    page.fill("[data-test=warning-note]", "the title page")
    page.click("[data-test=warning-ack]")
    page.wait_for_selector("[data-test=warning]", state="detached")
    assert "No open warnings" in page.inner_text("[data-test=tab-body]")
    assert page.query_selector("[data-test=tab-warnings] .n") is None
    page.click("[data-test=warnings-toggle]")
    page.wait_for_selector("[data-test=warning][data-status=acknowledged]")
    assert "ui: the title page" in page.inner_text("[data-test=warning]")
    assert page.problems == []


def test_hear_the_voices_and_make_a_sentence_again(page, site, books):
    import yaml
    d = books / "alice"
    demo.ensure_book(page, site)
    first = yaml.safe_load((d / "cast.yaml").read_text())["characters"]["_generic_male"]["voice"]
    demo.scenario_listen(page, site, demo.NoRecorder())
    male = yaml.safe_load((d / "cast.yaml").read_text())["characters"]["_generic_male"]
    assert male["voice"] != first and male["frozen"] is True               # another voice, and it is kept
    job = demo.wait_for_job(page, site)
    assert job["label"].startswith("make ") and job["args"][-4:] == ["--stages", "synth,assemble,package", "--docs", "2"]
    assert all(k.startswith("ab002s00004u") for k in yaml.safe_load((d / "reseed.yaml").read_text()))
    # the Cast tab: a kept voice cannot be replaced by accident; the Kokoro narrator points to Settings
    page.click("[data-test=tab-cast]")
    page.wait_for_selector("[data-test=voice-card]")
    card = "[data-test=voice-card][data-voice=_generic_male]"
    assert page.is_disabled(f"{card} [data-test=voice-reseed]") and page.inner_text(f"{card} [data-test=voice-freeze]") == "Let it change"
    assert page.query_selector("[data-voice=narrator] [data-test=voice-reseed]") is None
    assert "Settings" in page.inner_text("[data-voice=narrator]")
    assert page.evaluate("""async () => { const a = document.querySelector('[data-voice=_generic_female] audio');
        a.preload = 'auto'; a.load(); await new Promise((r) => { a.onloadedmetadata = r; a.onerror = r; }); return a.duration; }""") > 3
    page.click(f"{card} [data-test=voice-freeze]")
    page.wait_for_selector(f"{card} [data-test=voice-frozen]", state="detached")
    # the Listen tab: quotations are shown in their voice's colour, and playing one sentence stops at its end
    page.click("[data-test=tab-listen]")
    page.wait_for_selector("[data-test=sentences]")
    assert page.locator("[data-test=sentence]").count() == 54
    assert page.inner_text("[data-sentence=ab002s00002] .v.female").strip() == "and what is the use of a book,"
    assert "sentence-redo" in page.inner_html("[data-sentence=ab002s00002]")
    page.click("[data-sentence=ab002s00001] [data-test=sentence-play]")           # the heading: 2.7 s long
    page.wait_for_function("(() => { const a = document.querySelector('[data-test=chapter-audio]'); return a.paused && a.currentTime > 2; })()",
                           timeout=15_000)
    assert page.evaluate("document.querySelector('[data-test=chapter-audio]').currentTime") < 3.5
    assert page.problems == []


def test_the_models_page_and_a_video_cut_at_chosen_chapters(page, site, books):
    demo.scenario_publish(page, site, demo.NoRecorder())
    out = books / "alice" / "out" / "youtube"
    stem = "alice-s-adventures-in-wonderland"
    assert sorted(p.name for p in out.glob("*.srt")) == [f"{stem}-part{k}.srt" for k in (1, 2, 3)]
    assert "Text: Project Gutenberg ebook no. 11." in (out / f"{stem}-part3.txt").read_text()
    assert "0:00 CHAPTER IX." in (out / f"{stem}-part3.txt").read_text()
    assert not list(out.glob("*.mp4"))                                           # captions only was ticked
    names = [a.inner_text() for a in page.query_selector_all("[data-test=video-file]")]
    assert f"{stem}-part2.srt" in names and f"{stem}-frame.png" not in names
    # the form kept what was chosen, and a wrong choice is explained before anything runs
    assert page.is_checked("[data-test=video-how-cut]") and page.locator("[data-test=video-part]").count() == 3
    page.check("[data-test=video-how-parts]")
    page.fill("[data-test=video-parts]", "2")
    page.fill("[data-test=video-max]", "0.5")
    page.press("[data-test=video-max]", "Tab")
    page.wait_for_selector("[data-test=video-error]")
    assert "cannot cut into 2 parts" in page.inner_text("[data-test=video-error]") and page.is_disabled("[data-test=video-go]")
    # the Models page: each service once, with what the settings use, and a way to the instructions
    page.goto(site + "/#/services")
    page.wait_for_selector("[data-test=service]")
    states = {s.get_attribute("data-service"): s.get_attribute("data-state") for s in page.query_selector_all("[data-test=service]")}
    assert set(states) == {"llm", "kokoro", "drama3"} and set(states.values()) <= {"up", "down", "wrong", "off"}
    assert "these settings use it" in page.inner_text("[data-service=llm]") and "not needed" in page.inner_text("[data-service=drama3]")
    assert page.get_attribute("[data-service=drama3] a", "href") == "#/help/MODELS/fish-audio-drama-3-gpu"
    assert page.inner_text("[data-test=services-verdict]")
    (books / "alice" / "config.yaml").write_text("llm: {adapter: mock}\ntts: {adapter: mock}\n")
    try:
        page.select_option("[data-test=services-book]", "alice")
        page.wait_for_function("(document.querySelector('[data-service=llm]') || document.body).textContent.includes('mock: no model is used')")
        assert page.inner_text("[data-test=services-verdict]") == "Every service these settings need is answering."
    finally:
        (books / "alice" / "config.yaml").unlink()
    page.click("[data-service=kokoro] a")
    page.wait_for_selector("[data-test=doc] h1:has-text('Serving the models')")
    assert page.problems == []


def test_a_book_with_known_speakers_is_scored_and_two_results_compared(page, site, books):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_eval_pdnc import BASE, make_db

    from pipeline import store
    make_db(books / "mini", BASE).close()
    page.goto(site + "/#/book/mini/evaluate")
    page.wait_for_selector("[data-test=eval-run]")
    assert [o.inner_text() for o in page.query_selector_all("[data-test=eval-source] option")] == ["pdnc_mini", "pdnc_pride_ch1"]
    page.click("[data-test=eval-run]")                                            # no name yet
    page.wait_for_function("document.getElementById('toast').textContent.includes('short name')")
    assert len(page.problems) == 1 and "400" in page.problems[0]      # the refused request, as the browser logs it
    page.problems.clear()
    page.fill("[data-test=eval-tag]", "first")
    page.fill("[data-test=eval-note]", "as it came")
    page.click("[data-test=eval-run]")
    page.wait_for_selector("[data-test=eval-result][data-tag=first]")
    row = page.inner_text("[data-test=eval-result][data-tag=first]")
    assert "as it came" in row and page.inner_text("[data-tag=first] [data-test=eval-wrong]") == "25%"
    page.wait_for_selector("[data-test=eval-report][data-tag=first]")             # the new result's report is open
    assert "PDNC evaluation" in page.inner_text("[data-test=eval-report]")
    con = store.connect(books / "mini")                                           # Q1 is Ann's, not Tom's
    con.execute("UPDATE utterance SET speaker_id='ann' WHERE tts_text LIKE 'Fine%'")
    con.commit()
    con.close()
    page.fill("[data-test=eval-tag]", "second")
    page.click("[data-test=eval-run]")
    page.wait_for_selector("[data-test=eval-result][data-tag=second]")
    assert page.inner_text("[data-tag=second] [data-test=eval-wrong]") == "0%"
    assert page.is_disabled("[data-test=eval-compare-go]")
    page.check("[data-tag=first] [data-test=eval-pick]")
    page.check("[data-tag=second] [data-test=eval-pick]")
    page.click("[data-test=eval-compare-go]")
    page.wait_for_selector("[data-test=eval-compare]")
    assert "fixed 1, broke 0" in page.inner_text("[data-test=eval-compare]")
    assert "A corrected edition" in page.inner_text("[data-test=corrected]")
    assert page.problems == []


def test_an_unknown_book_says_so(page, site):
    page.goto(site + "/#/book/nothing-here")
    page.wait_for_selector("[data-test=error]")
    assert "no such book" in page.inner_text("[data-test=error]")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_the_recorder_writes_a_gif(page, site, tmp_path):
    page.goto(site + "/")
    page.wait_for_selector("[data-test=drop]")
    rec = demo.Recorder(page, tmp_path / "frames")
    rec.shot("one", hold=0.3)
    rec.shot("two", hold=0.3, mark="[data-test=drop]")
    out = rec.gif(tmp_path / "x.gif", width=440)
    assert out.read_bytes()[:6] == b"GIF89a" and len(rec.steps) == 2
    assert page.query_selector("[data-demo-mark]") is None                    # the page is left as it was
    assert page.evaluate("document.getElementById('demo-caption').style.display") == "none"
