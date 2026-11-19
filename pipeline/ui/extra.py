"""The rest of the command line, for the UI: whether the model services answer, scoring a book against known
speakers, and turning a finished book into video files.

As everywhere in the UI, nothing is computed here that the command line does not compute: a score is
`readaloud eval`, a comparison is `readaloud eval-compare`, a corrected edition is scripts/pdnc_overrides.py, a video
is scripts/youtube_export.py. This module finds their inputs, starts them, and reads what they wrote.
"""
from __future__ import annotations

import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import config
from ..adapters.tts import settings as tts_settings
from ..config import REPO
from . import api, edit, md
from .api import ApiError

TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
VIDEO_FILES = (".mp4", ".srt", ".txt", ".jpg", ".png")


# ---- do the model services answer? ---------------------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 3.0):
    t0 = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
            body = r.read(200_000)
        return json.loads(body), None, int((time.time() - t0) * 1000)
    except urllib.error.HTTPError as e:
        return None, f"it answered with an error ({e.code} {e.reason})", int((time.time() - t0) * 1000)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        return None, f"no answer ({reason})", int((time.time() - t0) * 1000)
    except ValueError:
        return None, "it answered, but not with JSON: is this the right address?", int((time.time() - t0) * 1000)


def _models(data) -> list[str]:
    return [m.get("id", "") for m in (data or {}).get("data", []) if isinstance(m, dict)] if isinstance(data, dict) else []


def _check_llm(cfg: dict) -> dict:
    c = cfg["llm"]
    out = {"id": "llm", "name": "Language model", "role": "decides which voice reads a quotation; classifies odd sections",
           "configured": c["adapter"], "endpoint": c.get("endpoint", ""), "doc": "the-language-model"}
    if c["adapter"] == "mock":
        return {**out, "state": "off", "detail": "mock: no model is used"}
    data, err, ms = _get_json(c["endpoint"].rstrip("/") + "/models")
    if err:
        return {**out, "state": "down", "detail": err, "ms": ms}
    ids = _models(data)
    ok = c.get("model") in ids
    return {**out, "state": "up" if ok else "wrong", "ms": ms,
            "detail": f"serves {', '.join(ids) or 'no model'}" + ("" if ok else f"; the settings ask for {c.get('model')}")}


def _check_kokoro(cfg: dict) -> dict:
    t = tts_settings({**cfg["tts"], "adapter": "kokoro"})
    out = {"id": "kokoro", "name": "Kokoro", "role": "speech, on a laptop", "configured": cfg["tts"]["adapter"],
           "endpoint": t.get("endpoint", ""), "doc": "kokoro-laptop"}
    data, err, ms = _get_json(t["endpoint"].rstrip("/") + "/info")
    if err:
        return {**out, "state": "down", "detail": err, "ms": ms}
    return {**out, "state": "up", "ms": ms, "detail": f"{data.get('model_version', 'Kokoro')}, {data.get('sample_rate', '?')} Hz"}


def _check_drama3(cfg: dict) -> dict:
    t = tts_settings({**cfg["tts"], "adapter": "drama3"})
    out = {"id": "drama3", "name": "Fish Audio Drama 3", "role": "speech and voice design, on a GPU",
           "configured": cfg["tts"]["adapter"], "endpoint": t.get("endpoint", ""), "doc": "fish-audio-drama-3-gpu"}
    data, err, ms = _get_json(t["endpoint"].rstrip("/") + "/v1/models")
    if err:
        return {**out, "state": "down", "detail": err, "ms": ms}
    ids = _models(data)
    ok = t.get("model") in ids
    return {**out, "state": "up" if ok else "wrong", "ms": ms,
            "detail": f"serves {', '.join(ids) or 'no model'}" + ("" if ok else f"; the settings ask for {t.get('model')}")}


def services(root: Path, book_id: str | None = None) -> dict:
    """One request to each service named in the settings (the defaults, or a book's)."""
    cfg = config.load(api.book_dir(root, book_id)) if book_id else config.load()
    with ThreadPoolExecutor(max_workers=3) as ex:
        checks = list(ex.map(lambda f: f(cfg), (_check_llm, _check_kokoro, _check_drama3)))
    used = {"llm": cfg["llm"]["adapter"] != "mock" and bool(cfg.get("multi_voice")),
            "kokoro": cfg["tts"]["adapter"] == "kokoro", "drama3": cfg["tts"]["adapter"] == "drama3"}
    for c in checks:
        c["used"] = used[c["id"]]
    return {"services": checks, "book": book_id, "tts": cfg["tts"]["adapter"], "llm": cfg["llm"]["adapter"],
            "multi_voice": bool(cfg.get("multi_voice"))}


# ---- scoring against known speakers --------------------------------------------------------------------------------------

def pdnc_root() -> Path:
    return REPO / "data" / "pdnc"


def pdnc_books(root: Path | None = None) -> list[str]:
    r = root or pdnc_root()
    return sorted(d.name for d in r.iterdir() if (d / "quotation_info.csv").exists()) if r.exists() else []


def _pdnc_dir(name: str, root: Path | None = None) -> Path:
    if name not in pdnc_books(root):
        raise ApiError("no such PDNC book: fetch it with scripts/fetch_pdnc.sh", 404)
    return (root or pdnc_root()) / name


def _results(d: Path) -> list[dict]:
    out = []
    ev = d / "reports" / "eval"
    for p in sorted(ev.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True) if ev.exists() else []:
        try:
            res = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        m = res.get("metrics", {})
        out.append({"tag": p.stem, "modified": p.stat().st_mtime, "created": res.get("created") or "",
                    "note": res.get("note") or "", "source": Path(str(res.get("pdnc") or "")).name,
                    "quotes": m.get("quotes"), "aligned_pct": m.get("align_pct"),
                    "wrong": m.get("v3_mis"), "wrong_gender_pct": m.get("v3_mis_pct"),
                    "voiced_pct": m.get("v3_voiced_pct")})
    return out


def evaluation(d: Path, root: Path | None = None) -> dict:
    return {"sources": pdnc_books(root), "results": _results(d), "where": str(root or pdnc_root()),
            "corrected": any(r["note"].startswith("PDNC ") for r in edit.read_overrides(d))}


def evaluate(books_root: Path, d: Path, jobs, body: dict, root: Path | None = None) -> dict:
    """`readaloud eval`: score the book as it is now, under a name."""
    tag = str(body.get("tag") or "").strip()
    if not TAG.match(tag):
        raise ApiError("give the result a short name: letters, digits, dots, hyphens")
    src = _pdnc_dir(str(body.get("source") or ""), root)
    args = ["eval", d.name, "--pdnc", str(src), "--tag", tag]
    ch = str(body.get("chapters") or "").strip()
    if ch:
        if not re.match(r"^\d+-\d+$", ch):
            raise ApiError("chapters are a range such as 1-13")
        args += ["--chapters", ch]
    if body.get("note"):
        args += ["--note", str(body["note"])[:200]]
    edit._idle(d, jobs)
    rc, out = api.cli(books_root, *args, timeout=900)
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "the evaluation failed")
    return {"summary": out, **evaluation(d, root)}


def report(d: Path, tag: str) -> dict:
    if not TAG.match(tag):
        raise ApiError("not a result name")
    p = d / "reports" / "eval" / f"{tag}.md"
    if not p.is_file():
        raise ApiError("no such result", 404)
    body, heads = md.render(p.read_text(encoding="utf-8"), here="DESIGN")
    return {"tag": tag, "html": body, "headings": [h for h in heads if h["level"] <= 3]}


def compare(books_root: Path, d: Path, tags: list) -> dict:
    """`readaloud eval-compare`: results side by side, with every quotation that changed."""
    tags = [str(t) for t in tags or []]
    if len(tags) < 2 or any(not TAG.match(t) for t in tags):
        raise ApiError("choose two or more results to compare")
    rc, out = api.cli(books_root, "eval-compare", *[f"{d.name}:{t}" for t in tags], timeout=300)
    if rc:
        raise ApiError(out.splitlines()[-1] if out else "the comparison failed")
    body, _ = md.render(out, here="DESIGN")
    return {"tags": tags, "html": body}


def corrected_args(d: Path, source: str, root: Path | None = None) -> list[str]:
    """scripts/pdnc_overrides.py: write the known speakers' voices into attribution_overrides.csv (a corrected edition)."""
    return ["script", "pdnc_overrides.py", d.name, "--pdnc", str(_pdnc_dir(source, root))]


# ---- video files -----------------------------------------------------------------------------------------------------------

def _exporter():
    p = REPO / "scripts" / "youtube_export.py"
    if not p.exists():
        raise ApiError("scripts/youtube_export.py is not in this installation", 404)
    spec = importlib.util.spec_from_file_location("youtube_export", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _video_options(body: dict) -> dict:
    try:
        max_hours = float(body.get("max_hours") or 11.5)
        parts = int(body["parts"]) if str(body.get("parts") or "").strip() else None
    except (TypeError, ValueError):
        raise ApiError("the longest part is a number of hours; the number of parts is a whole number") from None
    if not 0.01 <= max_hours <= 12:
        raise ApiError("a part can be at most 12 hours")
    cuts = [str(c).strip() for c in (body.get("cut") or []) if str(c).strip()]
    if cuts and parts:
        raise ApiError("give a number of parts, or the headings to cut at, not both")
    return {"max_hours": max_hours, "parts": parts, "cut": cuts, "part_name": str(body.get("part_name") or "Part")[:20] or "Part",
            "edition": str(body.get("edition") or "")[:200], "repo_url": str(body.get("repo_url") or "")[:200],
            "no_video": bool(body.get("no_video"))}


def video(d: Path, body: dict | None = None) -> dict:
    """Where the parts would be cut (nothing is written), and the files of an earlier export."""
    ye = _exporter()
    out = {"ready": False, "reason": "", "marks": [], "parts": [], "total": 0, "files": video_files(d)}
    try:
        docs, marks, total, _, _ = ye.load(d)
    except SystemExit as e:
        return {**out, "reason": "Every chapter needs its audio first: run the whole book on the Run tab. "
                                 f"({str(e).split('/')[-1]})"}
    except Exception as e:  # noqa: BLE001 - a book that has not been run at all
        return {**out, "reason": f"Run the book first. ({type(e).__name__})"}
    if not docs:
        return {**out, "reason": "Run the book first."}
    out.update(ready=True, total=total, marks=[{"ms": m, "title": t} for m, t in marks])
    o = _video_options(body or {})
    try:
        max_ms = int(o["max_hours"] * 3.6e6)
        cuts = ye.cuts_at(marks, total, max_ms, o["cut"]) if o["cut"] else ye.split_points(marks, total, max_ms, o["parts"])
    except SystemExit as e:
        return {**out, "error": str(e)}
    out["parts"] = [{"start": a, "end": b, "ms": b - a, "first": next((t for m, t in marks if m >= a), ""),
                     "chapters": sum(1 for m, _ in marks if a <= m < b)} for a, b in zip(cuts, cuts[1:])]
    return out


def video_args(d: Path, body: dict) -> list[str]:
    o = _video_options(body)
    args = ["script", "youtube_export.py", d.name, "--max-hours", str(o["max_hours"]), "--part-name", o["part_name"]]
    if o["parts"]:
        args += ["--parts", str(o["parts"])]
    for c in o["cut"]:
        args += ["--cut", c]
    for flag, key in (("--edition", "edition"), ("--repo-url", "repo_url")):
        if o[key]:
            args += [flag, o[key]]
    if o["no_video"]:
        args.append("--no-video")
    return args


def _is_video_file(p: Path) -> bool:
    """What an export is for. The full-size still it encodes from (<title>-frame.png) is a working file."""
    return p.suffix.lower() in VIDEO_FILES and not p.name.endswith("-frame.png") and p.is_file()


def video_files(d: Path) -> list[dict]:
    y = d / "out" / "youtube"
    files = sorted(p for p in y.iterdir() if _is_video_file(p)) if y.exists() else []
    return [{"name": p.name, "bytes": p.stat().st_size, "modified": p.stat().st_mtime} for p in files]


def video_file(d: Path, name: str) -> Path:
    p = d / "out" / "youtube" / Path(name).name
    if not _is_video_file(p):
        raise ApiError("no such file", 404)
    return p
