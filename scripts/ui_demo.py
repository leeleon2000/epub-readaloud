"""Drive the UI in a real browser and record what happens, as the GIFs the README shows.

    python scripts/ui_demo.py [--out docs/img] [--only run,help] [--keep-frames DIR]

Needs `pip install playwright`, Google Chrome (or `playwright install chromium`) and ffmpeg. It starts its own UI
server on a temporary books folder and uses the mock adapters, so it needs no model and touches no real book.

The scenarios below are also what tests/test_ui_browser.py runs and checks, so a GIF shows operations that are
tested. A scenario is a function `(page, base_url, rec)`; `rec` records a frame per step (tests pass one that
records nothing).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VIEW = {"width": 1100, "height": 700}
CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:0;right:0;bottom:0;z-index:99;padding:10px 18px;font:600 17px system-ui,' +
      'sans-serif;background:rgba(22,24,27,.92);color:#fff;letter-spacing:.01em';
    document.body.append(el);
  }
  el.textContent = text;
  el.style.display = text ? 'block' : 'none';
}"""
MARK_JS = """(sel) => {
  document.querySelectorAll('[data-demo-mark]').forEach((e) => { e.style.outline = ''; e.removeAttribute('data-demo-mark'); });
  const el = sel && document.querySelector(sel);
  if (el) {
    el.scrollIntoView({block: 'center'});
    el.style.outline = '3px solid #e8590c';
    el.style.outlineOffset = '3px';
    el.setAttribute('data-demo-mark', '1');
  }
}"""


TOAST_JS = "(show) => { const t = document.getElementById('toast'); if (t) t.style.visibility = show ? '' : 'hidden'; }"


class NoRecorder:
    """What the tests pass: the scenario runs, nothing is recorded."""

    def shot(self, caption: str = "", hold: float = 1.4, mark: str | None = None, toast: bool = False) -> None:
        pass


class Recorder(NoRecorder):
    def __init__(self, page, frames: Path):
        self.page, self.frames, self.steps = page, frames, []
        frames.mkdir(parents=True, exist_ok=True)

    def shot(self, caption: str = "", hold: float = 1.4, mark: str | None = None, toast: bool = False) -> None:
        """One frame: the page as it is, with a caption, and an outline round the control about to be used. The page's
        own pop-up message is left out unless the frame is about it: frames are taken faster than a person works, so
        the message of the step before would still be showing."""
        self.page.evaluate(MARK_JS, mark)
        self.page.evaluate(CAPTION_JS, caption)
        self.page.evaluate(TOAST_JS, toast)
        p = self.frames / f"{len(self.steps):03d}.png"
        self.page.screenshot(path=str(p))
        self.steps.append((p, hold))
        self.page.evaluate(TOAST_JS, True)
        self.page.evaluate(MARK_JS, None)
        self.page.evaluate(CAPTION_JS, "")

    def gif(self, out: Path, width: int = 880) -> Path:
        """Join the frames, each held for its time, into one GIF with a palette of its own."""
        if not self.steps:
            raise SystemExit("no frames were recorded")
        lst = self.frames / "frames.txt"
        lines = []
        for p, hold in self.steps:
            lines += [f"file '{p.as_posix()}'", f"duration {hold}"]
        lines.append(f"file '{self.steps[-1][0].as_posix()}'")       # the concat demuxer needs the last file twice
        lst.write_text("\n".join(lines) + "\n")
        out.parent.mkdir(parents=True, exist_ok=True)
        vf = (f"fps=10,scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-vf", vf,
                            "-loop", "0", str(out)], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"ffmpeg failed:\n{r.stderr[-1500:]}")
        return out


def launch(p):
    """The installed Chrome when there is one, else Playwright's own Chromium."""
    args = ["--autoplay-policy=no-user-gesture-required"]      # the Listen tab plays audio when a sentence is pressed
    try:
        return p.chromium.launch(channel="chrome", headless=True, args=args)
    except Exception:  # noqa: BLE001 - any reason Chrome cannot start: try the bundled browser
        return p.chromium.launch(headless=True, args=args)


def start_server(books_root: Path, pdnc: Path | None = None):
    from pipeline.ui import server
    srv = server.make_server(books_root, pdnc=pdnc)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def stop_server(srv) -> None:
    jobs = srv.RequestHandlerClass.jobs
    for d in Path(jobs.root).iterdir():
        if d.is_dir():
            jobs.stop(d.name, wait=5)
    srv.shutdown()
    srv.server_close()


# ---- scenarios -------------------------------------------------------------------------------------------------------

def scenario_run(page, base: str, rec: NoRecorder) -> None:
    """An empty library, the sample book, one chapter to an audiobook of that chapter, with the mock adapters."""
    page.goto(base + "/")
    page.wait_for_selector("[data-test=empty]")
    rec.shot("readaloud ui: a local page for the whole pipeline", hold=1.8)
    rec.shot("Add a book: drop an EPUB, or start with the sample", mark="[data-test=add-sample]")
    page.click("[data-test=add-sample]")
    page.wait_for_selector("[data-test=stages]")
    rec.shot("Every stage of the pipeline, and where this book stands", hold=1.8)
    page.select_option("[data-test=run-tts]", "mock")
    page.select_option("[data-test=run-llm]", "mock")
    rec.shot("Pick the speech and language models (mock needs none)", mark="[data-test=run-tts]")
    page.fill("[data-test=run-docs]", "2")
    rec.shot("Audio for one chapter first, or for the whole book", mark="[data-test=run-docs]")
    rec.shot("Start", mark="[data-test=start]", hold=1.0)
    page.click("[data-test=start]")
    page.wait_for_function("document.querySelector('[data-test=log]').textContent.includes('[segment')", timeout=120_000)
    page.evaluate("document.querySelector('[data-test=log]').scrollIntoView({block: 'center'})")
    rec.shot("The log follows the job; closing the page does not stop it", hold=1.8)
    page.wait_for_function("document.querySelector('[data-test=log]').textContent.includes('[synth')", timeout=300_000)
    rec.shot("Text stages done; now the audio", hold=1.4)
    page.wait_for_selector("[data-stage=package][data-state=done]", timeout=300_000)
    page.wait_for_selector("[data-test=job-status][data-status=ok]", timeout=30_000)
    page.evaluate("window.scrollTo(0, 0)")
    rec.shot("Finished. Audio exists for that chapter; the rest of the book is still to do", hold=2.2)
    page.wait_for_selector("[data-test=download]")
    rec.shot("Download the EPUB 3 audiobook and open it in a read-along reader", mark="[data-test=download]", hold=2.6)


def scenario_help(page, base: str, rec: NoRecorder) -> None:
    """The project's documents inside the UI, and a link from a panel to the section that explains it."""
    page.goto(base + "/#/help/USAGE")
    page.wait_for_selector("[data-test=doc] h1")
    rec.shot("Help: the project's documents, inside the page", hold=1.8)
    page.click("[data-test=doc-nav] a[href='#/help/USAGE/6-when-it-stops']")
    page.wait_for_function("location.hash.endsWith('6-when-it-stops')")
    page.wait_for_timeout(200)
    rec.shot("Every warning and message is explained", hold=2.0)
    page.click("[data-test=doc-nav] a[href='#/help/MODELS']")
    page.wait_for_selector("[data-test=doc] h1:has-text('Serving the models')")
    rec.shot("How to serve the models, and what was measured", hold=2.2)


def ensure_book(page, base: str, book: str = "alice") -> None:
    """The sample book with its text stages run for the whole book and the audio of chapter 2 (what `scenario_run`
    leaves), made through the API if it is not there."""
    hdr = {"X-Requested-With": "readaloud"}

    def wait() -> None:
        for _ in range(1500):
            if (page.request.get(f"{base}/api/books/{book}/job").json()["job"] or {}).get("status") != "running":
                return
            page.wait_for_timeout(200)
        raise AssertionError("the sample book's job did not finish")

    def run(**body) -> None:
        wait()
        assert page.request.post(f"{base}/api/books/{book}/run", headers=hdr, data={"tts": "mock", "llm": "mock", **body}).ok
        wait()

    r = page.request.get(f"{base}/api/books/{book}")
    if r.status == 404:
        assert page.request.post(f"{base}/api/books/sample", headers=hdr, data={"id": book}).ok
    wait()
    if not page.request.get(f"{base}/api/books/{book}").json()["outputs"]:
        run(docs="2")              # the UI runs the text stages for the whole book, then this chapter's audio


def scenario_review(page, base: str, rec: NoRecorder) -> None:
    """Looking at what was decided and changing it: a line's voice, a name's gender, a line that is not read."""
    ensure_book(page, base)
    page.goto(base + "/#/book/alice/voices")
    page.wait_for_selector("[data-test=quotes]")
    rec.shot("Voices: every quotation, and which of the three voices reads it", hold=2.0)
    page.check("[data-test=voice-review]")
    row = "[data-test=quote][data-quote='002p0003q1']"               # "Oh dear! Oh dear! I shall be late!"
    page.wait_for_function("(document.querySelector('[data-test=quote]') || {dataset: {}}).dataset.quote === '002p0003q1'")
    rec.shot("Lines the pipeline was not sure about stay with the narrator", mark="[data-test=voice-review]", hold=2.0)
    rec.shot("Give a line its voice: the White Rabbit is a he", mark=f"{row} [data-test=voice-male]")
    page.click(f"{row} [data-test=voice-male]")
    page.wait_for_selector(f"{row}[data-pending='1']")
    page.wait_for_selector("[data-test=apply-voices]")
    rec.shot("The change is kept in attribution_overrides.csv", mark="[data-test=apply-voices]", hold=1.8)
    page.click("[data-test=apply-voices]")
    page.wait_for_selector("[data-test=apply-voices]", state="detached", timeout=60_000)
    page.uncheck("[data-test=voice-review]")
    page.check("[data-test=voice-changed]")
    page.wait_for_selector(f"{row}[data-voice=male]:not([data-pending])")
    rec.shot("Applied: the voices stage read the file again, with no model call", hold=2.0)
    page.uncheck("[data-test=voice-changed]")
    page.wait_for_function("document.querySelectorAll('[data-test=quote]').length > 1")      # the list is whole again
    page.fill("[data-test=gender-name]", "the Dormouse")
    page.select_option("[data-test=gender-value]", "male")
    rec.shot("Or settle a name once for the whole book", mark="[data-test=gender-add]", hold=1.8)
    page.click("[data-test=gender-add]")
    page.wait_for_selector("[data-test=gender-manual]")
    page.evaluate("document.querySelector('[data-test=genders]').scrollIntoView({block: 'start'})")
    rec.shot("Kept in genders.yaml; what the pipeline found itself is listed below", hold=1.8)

    page.click("[data-test=tab-sections]")
    page.wait_for_selector("[data-test=sections]")
    rec.shot("Sections: what is story, what the narrator reads, what is left out", hold=2.2)
    page.click("[data-section=d002s00000] [data-test=section-lines]")
    page.wait_for_selector("[data-test=lines-panel]")
    box = "[data-test=line-read][data-sentence=ab002s00003]"
    rec.shot("Any single line can be kept from being read", mark=box, hold=1.8)
    page.uncheck(box)
    page.wait_for_selector(f"{box}:not(:checked):not([disabled])")
    page.evaluate("document.querySelector('[data-test=lines-panel]').scrollIntoView({block: 'start'})")
    rec.shot("The book's text is not changed; only the audio leaves it out", hold=2.0)
    page.check(box)                                  # the book is left as it was found
    page.wait_for_selector(f"{box}:checked:not([disabled])")

    page.click("[data-test=tab-settings]")
    page.wait_for_selector("[data-test=settings]")
    rec.shot("Settings for this book, each one explained", hold=2.4)


def wait_for_job(page, base: str, book: str = "alice") -> dict:
    for _ in range(1500):
        job = page.request.get(f"{base}/api/books/{book}/job").json()["job"] or {}
        if job.get("status") != "running":
            return job
        page.wait_for_timeout(200)
    raise AssertionError("the job did not finish")


def scenario_listen(page, base: str, rec: NoRecorder) -> None:
    """Hearing the result: the three voices' samples, a chapter sentence by sentence, one sentence made again."""
    ensure_book(page, base)
    page.goto(base + "/#/book/alice/cast")
    page.wait_for_selector("[data-test=voice-card]")
    rec.shot("Cast: the three voices that read the book, each with a sample", hold=2.2)
    male = "[data-test=voice-card][data-voice=_generic_male]"
    before = page.inner_text(f"{male} code")
    rec.shot("Not the right voice? Design another", mark=f"{male} [data-test=voice-reseed]")
    page.click(f"{male} [data-test=voice-reseed]")
    page.wait_for_function("([sel, old]) => { const c = document.querySelector(sel + ' code'); return c && c.textContent !== old; }",
                           arg=[male, before], timeout=120_000)
    rec.shot("A different male voice, with a new sample to hear", mark=f"{male} audio", hold=2.0)
    page.click(f"{male} [data-test=voice-freeze]")
    page.wait_for_selector(f"{male} [data-test=voice-frozen]")
    rec.shot("Keep this one: later runs will not change it", mark=f"{male} [data-test=voice-frozen]", hold=2.0)

    page.click("[data-test=tab-listen]")
    page.wait_for_selector("[data-test=sentences]")
    rec.shot("Listen: a chapter, sentence by sentence, in the voices it was given", hold=2.2)
    row = "[data-test=sentence][data-sentence=ab002s00004]"
    rec.shot("Play any sentence", mark=f"{row} [data-test=sentence-play]")
    page.click(f"{row} [data-test=sentence-play]")
    page.wait_for_selector(f"{row}.now")
    page.wait_for_function("document.querySelector('[data-test=chapter-audio]').currentTime > 33.9")
    rec.shot("The sentence being read is marked, as in a read-along reader", hold=2.4)
    page.evaluate("document.querySelector('[data-test=chapter-audio]').pause()")
    rec.shot("A sentence that came out wrong is made again, alone", mark=f"{row} [data-test=sentence-reseed]", hold=2.0)
    page.click(f"{row} [data-test=sentence-reseed]")
    page.wait_for_function("document.getElementById('toast').textContent.startsWith('Making it again')")
    job = wait_for_job(page, base)
    assert job.get("status") == "ok", job
    page.wait_for_function("document.getElementById('toast').textContent.startsWith('Made again')", timeout=30_000)
    rec.shot("Only that sentence was generated; its chapter was put together again", hold=2.6, toast=True)


def ensure_whole_book(page, base: str, book: str = "alice") -> None:
    """The sample book with the audio of every chapter (the mock adapters make two hours of tone in seconds)."""
    ensure_book(page, base, book)
    if not all(c["assembled"] for c in page.request.get(f"{base}/api/books/{book}").json()["chapters"]):
        assert page.request.post(f"{base}/api/books/{book}/run", headers={"X-Requested-With": "readaloud"},
                                 data={"tts": "mock", "llm": "mock"}).ok
        job = wait_for_job(page, base, book)
        assert job.get("status") == "ok", job


def scenario_publish(page, base: str, rec: NoRecorder) -> None:
    """Whether the model services answer, and a finished book as video files cut at chosen chapters."""
    ensure_whole_book(page, base)
    page.goto(base + "/#/services")
    page.wait_for_selector("[data-test=service]")
    rec.shot("Models: each service the pipeline talks to, and whether it answers", hold=2.4)
    page.goto(base + "/#/book/alice/video")
    page.wait_for_selector("[data-test=video-parts-table]")
    rec.shot("Video: a finished book as video files, with captions and chapters", hold=2.2)
    rec.shot("Cut it where the book's own parts begin", mark="[data-test=video-how-cut]")
    page.check("[data-test=video-how-cut]")
    page.wait_for_selector("[data-test=video-marks]")
    page.check("[data-test=video-mark][data-title^='CHAPTER V.']")
    page.wait_for_function("document.querySelectorAll('[data-test=video-part]').length === 2")
    page.check("[data-test=video-mark][data-title^='CHAPTER IX.']")
    page.wait_for_function("document.querySelectorAll('[data-test=video-part]').length === 3")
    page.evaluate("document.querySelector('[data-test=video-parts-table]').scrollIntoView({block: 'center'})")
    rec.shot("The parts are shown before anything is encoded", mark="[data-test=video-parts-table]", hold=2.4)
    page.fill("[data-test=video-edition]", "Project Gutenberg ebook no. 11")
    page.check("[data-test=video-novideo]")
    rec.shot("Captions and descriptions only, or the videos as well", mark="[data-test=video-go]", hold=1.8)
    page.click("[data-test=video-go]")
    page.wait_for_function("location.hash.endsWith('/run')")
    page.wait_for_function("(document.querySelector('[data-test=log]') || document.body).textContent.includes('in 3 part(s)')",
                           timeout=120_000)
    job = wait_for_job(page, base)
    assert job.get("status") == "ok", job
    page.click("[data-test=tab-video]")
    page.wait_for_selector("[data-test=video-file]")
    page.evaluate("document.querySelector('[data-test=video-files]').scrollIntoView({block: 'start'})")
    rec.shot("For each part: captions, and a description with the chapter list", hold=2.6)


SCENARIOS = {"run": scenario_run, "review": scenario_review, "listen": scenario_listen, "publish": scenario_publish,
             "help": scenario_help}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "img"), help="folder for the GIFs (default docs/img)")
    ap.add_argument("--only", help="comma-separated scenarios (default: all): " + ", ".join(SCENARIOS))
    ap.add_argument("--keep-frames", help="also keep the PNG frames in this folder")
    a = ap.parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("this needs Playwright: pip install playwright") from None
    if not shutil.which("ffmpeg"):
        raise SystemExit("this needs ffmpeg on the PATH")
    names = a.only.split(",") if a.only else list(SCENARIOS)
    bad = [n for n in names if n not in SCENARIOS]
    if bad:
        raise SystemExit(f"unknown scenario: {bad[0]} (there are: {', '.join(SCENARIOS)})")
    with tempfile.TemporaryDirectory(prefix="readaloud-demo-") as tmp, sync_playwright() as p:
        tmp = Path(tmp)
        srv, base = start_server(tmp / "books")
        browser = launch(p)
        try:
            for name in names:
                page = browser.new_page(viewport=VIEW, color_scheme="light", device_scale_factor=1)
                frames = Path(a.keep_frames) / name if a.keep_frames else tmp / "frames" / name
                rec = Recorder(page, frames)
                SCENARIOS[name](page, base, rec)
                out = rec.gif(Path(a.out) / f"ui-{name}.gif")
                print(f"{out}  ({len(rec.steps)} frames, {sum(h for _, h in rec.steps):.0f} s, {out.stat().st_size / 1e3:.0f} kB)")
                page.close()
        finally:
            browser.close()
            stop_server(srv)


if __name__ == "__main__":
    main()
