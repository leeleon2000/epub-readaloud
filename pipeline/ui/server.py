"""The UI's HTTP server: a JSON API under /api/, the page's static files, and the audio and EPUB files of the books.

Standard library only. It is meant for one person on one machine:
- it listens on 127.0.0.1 unless told otherwise;
- a request must name this server in its Host header (a page on another site cannot reach it by a made-up name);
- anything that changes state must carry the header the page's own script sends, which a plain cross-site form
  cannot add.
"""
from __future__ import annotations

import json
import mimetypes
import re
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .. import config
from . import api, edit, extra, listen
from .api import ApiError
from .jobs import JobBusy, Jobs

STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 600 * 1024 * 1024
LOCAL = ("127.0.0.1", "localhost", "::1", "[::1]")
ROUTES: list[tuple[str, re.Pattern, object]] = []


def route(method: str, pattern: str):
    def deco(fn):
        ROUTES.append((method, re.compile("^" + pattern + "$"), fn))
        return fn
    return deco


class Ctx:
    """One request, as the handlers see it."""

    def __init__(self, handler, root: Path, jobs: Jobs, query: dict, body: bytes):
        self.h, self.root, self.jobs, self.query, self.body = handler, root, jobs, query, body

    def q(self, key: str, default: str = "") -> str:
        return (self.query.get(key) or [default])[0]

    def json(self) -> dict:
        if not self.body:
            return {}
        try:
            data = json.loads(self.body)
        except ValueError as e:
            raise ApiError("the request body is not JSON") from e
        if not isinstance(data, dict):
            raise ApiError("the request body must be a JSON object")
        return data


# ---- books -----------------------------------------------------------------------------------------------------

@route("GET", r"/api/books")
def _books(c: Ctx):
    return {"books": api.books(c.root, c.jobs), "root": str(c.root), "sample": api.sample_book() is not None}


@route("POST", r"/api/books")
def _add_book(c: Ctx):
    """The file is the request body; its name (and an optional id) are in the query."""
    if not c.body:
        raise ApiError("no file was sent")
    src = api.save_upload(c.root, c.q("name", "book.epub"), c.body)
    try:
        return api.add_book(c.root, src, c.q("id") or None)
    finally:
        api.remove_upload(src)


@route("POST", r"/api/books/sample")
def _add_sample(c: Ctx):
    src = api.sample_book()
    if src is None:
        raise ApiError("the sample book is not in this installation", 404)
    return api.add_book(c.root, src, c.json().get("id") or "alice")


@route("GET", r"/api/books/([^/]+)")
def _book(c: Ctx, book_id: str):
    return api.book(c.root, book_id, c.jobs)


@route("DELETE", r"/api/books/([^/]+)")
def _delete_book(c: Ctx, book_id: str):
    api.delete_book(c.root, book_id, c.jobs)
    return {"deleted": book_id}


@route("POST", r"/api/books/([^/]+)/run")
def _run(c: Ctx, book_id: str):
    d = api.book_dir(c.root, book_id)
    commands, label = api.run_args(book_id, c.json(), config.load(d))
    return {"job": c.jobs.start(book_id, commands, label)}


@route("POST", r"/api/books/([^/]+)/stop")
def _stop(c: Ctx, book_id: str):
    api.book_dir(c.root, book_id)
    return {"job": c.jobs.stop(book_id)}


@route("GET", r"/api/books/([^/]+)/job")
def _job(c: Ctx, book_id: str):
    api.book_dir(c.root, book_id)
    try:
        offset = max(0, int(c.q("offset", "0")))
    except ValueError:
        offset = 0
    text, nxt = c.jobs.log(book_id, offset)
    return {"job": c.jobs.state(book_id), "log": text, "offset": nxt}


@route("GET", r"/api/books/([^/]+)/out/([^/]+)")
def _output(c: Ctx, book_id: str, name: str):
    return api.output_file(c.root, book_id, name)


# ---- what the pipeline decided, and changing it ------------------------------------------------------------------

@route("GET", r"/api/books/([^/]+)/warnings")
def _warnings(c: Ctx, book_id: str):
    return edit.warnings(api.book_dir(c.root, book_id), include_handled=c.q("all") == "1")


@route("POST", r"/api/books/([^/]+)/warnings/(\d+)")
def _warning_action(c: Ctx, book_id: str, wid: str):
    b = c.json()
    return edit.warning_action(c.root, api.book_dir(c.root, book_id), c.jobs, int(wid), b.get("action", ""), b.get("note", ""))


@route("GET", r"/api/books/([^/]+)/sections")
def _sections(c: Ctx, book_id: str):
    return edit.sections(api.book_dir(c.root, book_id))


@route("POST", r"/api/books/([^/]+)/sections")
def _set_sections(c: Ctx, book_id: str):
    return edit.set_sections(c.root, api.book_dir(c.root, book_id), c.jobs, c.json().get("set") or {})


@route("GET", r"/api/books/([^/]+)/sections/([^/]+)")
def _section(c: Ctx, book_id: str, sid: str):
    return edit.section_sentences(api.book_dir(c.root, book_id), sid)


@route("POST", r"/api/books/([^/]+)/skip")
def _skip(c: Ctx, book_id: str):
    b = c.json()
    return edit.set_skip(c.root, api.book_dir(c.root, book_id), c.jobs, b.get("add") or [], b.get("remove") or [])


@route("GET", r"/api/books/([^/]+)/quotes")
def _quotes(c: Ctx, book_id: str):
    try:
        chapter = int(c.q("chapter")) if c.q("chapter") else None
        offset = max(0, int(c.q("offset", "0")))
    except ValueError:
        raise ApiError("chapter and offset are numbers") from None
    return edit.quotes(api.book_dir(c.root, book_id), chapter=chapter, voice=c.q("voice"), search=c.q("q"),
                       review=c.q("review") == "1", changed=c.q("changed") == "1", offset=offset)


@route("POST", r"/api/books/([^/]+)/quotes")
def _set_quotes(c: Ctx, book_id: str):
    return edit.set_quotes(api.book_dir(c.root, book_id), c.json().get("set") or {})


@route("POST", r"/api/books/([^/]+)/quotes/apply")
def _apply_quotes(c: Ctx, book_id: str):
    """The voices stage again: it reads attribution_overrides.csv and genders.yaml; answers already given are reused."""
    api.book_dir(c.root, book_id)
    args = ["run", api.check_id(book_id), *edit.last_adapters(c.jobs, book_id), "--stages", "voices"]
    return {"job": c.jobs.start(book_id, args, "apply voice changes")}


@route("GET", r"/api/books/([^/]+)/genders")
def _genders(c: Ctx, book_id: str):
    return edit.genders(api.book_dir(c.root, book_id))


@route("POST", r"/api/books/([^/]+)/genders")
def _set_genders(c: Ctx, book_id: str):
    return edit.set_genders(api.book_dir(c.root, book_id), c.json().get("set") or {})


@route("GET", r"/api/books/([^/]+)/settings")
def _settings(c: Ctx, book_id: str):
    return edit.settings(api.book_dir(c.root, book_id))


@route("PUT", r"/api/books/([^/]+)/settings")
def _save_settings(c: Ctx, book_id: str):
    return edit.save_settings(api.book_dir(c.root, book_id), c.jobs, c.json())


# ---- hearing it, and making parts again ----------------------------------------------------------------------------

@route("GET", r"/api/books/([^/]+)/cast")
def _cast(c: Ctx, book_id: str):
    return listen.cast(api.book_dir(c.root, book_id), c.jobs)


@route("GET", r"/api/books/([^/]+)/cast/([^/]+)/sample")
def _cast_sample(c: Ctx, book_id: str, vid: str):
    return listen.cast_sample(api.book_dir(c.root, book_id), c.jobs, vid)


@route("POST", r"/api/books/([^/]+)/cast/([^/]+)")
def _cast_action(c: Ctx, book_id: str, vid: str):
    return listen.cast_action(c.root, api.book_dir(c.root, book_id), c.jobs, vid, c.json().get("action", ""))


@route("GET", r"/api/books/([^/]+)/listen/(\d+)")
def _listen(c: Ctx, book_id: str, n: str):
    return listen.chapter(api.book_dir(c.root, book_id), int(n))


@route("GET", r"/api/books/([^/]+)/audio/(\d+)")
def _audio(c: Ctx, book_id: str, n: str):
    return listen.chapter_audio(api.book_dir(c.root, book_id), int(n))


@route("GET", r"/api/books/([^/]+)/flagged")
def _flagged(c: Ctx, book_id: str):
    return listen.flagged(api.book_dir(c.root, book_id))


@route("POST", r"/api/books/([^/]+)/redo")
def _redo(c: Ctx, book_id: str):
    """Drop the takes of some pieces and, unless told not to, make them again for the chapters they are in."""
    d, b = api.book_dir(c.root, book_id), c.json()
    out = listen.redo(c.root, d, c.jobs, b.get("pieces") or [], bool(b.get("next_seed")))
    if b.get("run", True):
        n = len(b["pieces"])
        out["job"] = c.jobs.start(book_id, listen.redo_args(book_id, c.jobs, out["chapters"]),
                                  f"make {n} piece{'s' if n > 1 else ''} of audio again")
    return out


# ---- services, scoring, video ------------------------------------------------------------------------------------------

@route("GET", r"/api/services")
def _services(c: Ctx):
    return extra.services(c.root, c.q("book") or None)


@route("GET", r"/api/books/([^/]+)/eval")
def _eval(c: Ctx, book_id: str):
    return extra.evaluation(api.book_dir(c.root, book_id), c.h.pdnc)


@route("POST", r"/api/books/([^/]+)/eval")
def _eval_run(c: Ctx, book_id: str):
    return extra.evaluate(c.root, api.book_dir(c.root, book_id), c.jobs, c.json(), c.h.pdnc)


@route("GET", r"/api/books/([^/]+)/eval/([^/]+)")
def _eval_report(c: Ctx, book_id: str, tag: str):
    return extra.report(api.book_dir(c.root, book_id), tag)


@route("POST", r"/api/books/([^/]+)/eval-compare")
def _eval_compare(c: Ctx, book_id: str):
    return extra.compare(c.root, api.book_dir(c.root, book_id), c.json().get("tags") or [])


@route("POST", r"/api/books/([^/]+)/corrected")
def _corrected(c: Ctx, book_id: str):
    """A corrected edition: PDNC's speakers into the corrections file, then the voices stage to apply them."""
    d, b = api.book_dir(c.root, book_id), c.json()
    cmds = [extra.corrected_args(d, str(b.get("source") or ""), c.h.pdnc)]
    if b.get("apply", True):
        cmds.append(["run", d.name, *edit.last_adapters(c.jobs, book_id), "--stages", "voices"])
    return {"job": c.jobs.start(book_id, cmds, "corrected edition")}


@route("POST", r"/api/books/([^/]+)/video/plan")
def _video_plan(c: Ctx, book_id: str):
    return extra.video(api.book_dir(c.root, book_id), c.json())


@route("POST", r"/api/books/([^/]+)/video")
def _video(c: Ctx, book_id: str):
    d, b = api.book_dir(c.root, book_id), c.json()
    plan = extra.video(d, b)
    if not plan["ready"] or plan.get("error"):
        raise ApiError(plan.get("error") or plan["reason"], 409)
    return {"job": c.jobs.start(book_id, extra.video_args(d, b), "video files")}


@route("GET", r"/api/books/([^/]+)/video/([^/]+)")
def _video_file(c: Ctx, book_id: str, name: str):
    return extra.video_file(api.book_dir(c.root, book_id), name)


# ---- documents ---------------------------------------------------------------------------------------------------

@route("GET", r"/api/docs")
def _docs(c: Ctx):
    return {"docs": api.doc_list()}


@route("GET", r"/api/docs/([A-Za-z]+)")
def _doc(c: Ctx, doc_id: str):
    return api.doc(doc_id)


@route("GET", r"/repo/(.+)")
def _repo_file(c: Ctx, rel: str):
    return api.repo_file(rel)


# ---- the server ---------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "readaloud-ui"
    protocol_version = "HTTP/1.1"
    root: Path
    jobs: Jobs
    pdnc: Path | None = None        # where the PDNC books are (default data/pdnc); tests point it at their fixture
    quiet = True

    def log_message(self, fmt, *args):  # noqa: D401 - the console stays readable; errors are printed below
        if not self.quiet:
            super().log_message(fmt, *args)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0] if not (self.headers.get("Host") or "").startswith("[") \
            else (self.headers.get("Host") or "").split("]")[0] + "]"
        bound = self.server.server_address[0]
        return host in LOCAL or host == bound or bound not in LOCAL

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):     # the page went elsewhere before the answer was read
            self.close_connection = True

    def _json(self, status: int, obj) -> None:
        self._send(status, api.dumps(obj), "application/json; charset=utf-8")

    def _file(self, path: Path, download: bool = False) -> None:
        """A file, with byte ranges so that audio can be played from the middle."""
        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".epub":
            ctype = "application/epub+zip"
        start, end, status = 0, size - 1, 200
        rng = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
        if rng and (rng.group(1) or rng.group(2)):
            if rng.group(1):
                start = int(rng.group(1))
                end = min(int(rng.group(2)), size - 1) if rng.group(2) else size - 1
            else:
                start = max(0, size - int(rng.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 256 * 1024))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def _static(self, rel: str) -> None:
        rel = rel.lstrip("/") or "index.html"
        p = (STATIC / rel).resolve()
        if STATIC.resolve() not in p.parents or not p.is_file():
            p = STATIC / "index.html"           # the page handles its own addresses
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(200, p.read_bytes(), ctype)

    def _handle(self) -> None:
        if not self._host_ok():
            self._json(403, {"error": "this server answers only to its own address"})
            return
        url = urlparse(self.path)
        path = unquote(url.path)
        method = "GET" if self.command == "HEAD" else self.command
        if method != "GET" and self.headers.get("X-Requested-With") != "readaloud":
            self._json(403, {"error": "changes must come from the UI's own page"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD:
            self._json(413, {"error": "the file is too large"})
            return
        body = self.rfile.read(length) if length else b""
        if not (path.startswith("/api/") or path.startswith("/repo/")):
            if method == "GET":
                self._static(path)
            else:
                self._json(404, {"error": "not found"})
            return
        allowed = False
        for m, pat, fn in ROUTES:
            hit = pat.match(path)
            if not hit:
                continue
            allowed = True
            if m != method:
                continue
            try:
                res = fn(Ctx(self, self.root, self.jobs, parse_qs(url.query), body), *hit.groups())
            except ApiError as e:
                self._json(e.status, {"error": str(e)})
            except JobBusy as e:
                self._json(409, {"error": str(e)})
            except SystemExit as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - one broken request must not take the server down
                traceback.print_exc()
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
            else:
                if isinstance(res, Path):
                    self._file(res, download=res.suffix in (".epub", ".mp4", ".srt", ".txt"))
                else:
                    self._json(200, res)
            return
        self._json(405 if allowed else 404, {"error": "method not allowed" if allowed else "not found"})

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = _handle


def make_server(books_root: Path, host: str = "127.0.0.1", port: int = 0, pdnc: Path | None = None) -> ThreadingHTTPServer:
    """A server ready to run (`serve_forever()`); port 0 picks a free one (tests)."""
    root = Path(books_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    handler = type("BoundHandler", (Handler,), {"root": root, "jobs": Jobs(root), "pdnc": pdnc})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    return srv


def serve(books_root: Path, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    srv = make_server(books_root, host, port)
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    url = f"http://{shown}:{srv.server_address[1]}/"
    print(f"readaloud ui: {url}   (books in {Path(books_root).resolve()}; Ctrl-C to stop)")
    if host not in LOCAL:
        print("  listening beyond this machine: anyone who can reach this address can use it")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped (running jobs carry on; `readaloud ui` again to watch them)")
    finally:
        srv.server_close()
