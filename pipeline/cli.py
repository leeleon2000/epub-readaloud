"""`readaloud` command line (see docs/USAGE.md).

    readaloud init <book.epub|book.azw3|book.mobi> [--id ID]          (AZW3 / MOBI: converted to EPUB 3 with Calibre)
    readaloud convert <file|folder ...> [--out DIR] [--done DIR]      (AZW3 / MOBI -> <name>.epub, e.g. data/next)
    readaloud run <id> [--until STAGE | --stages a,b] [--docs 2,3] [--tts mock|kokoro|drama3] [--llm mock|openai_compat]
    readaloud preview <id> --docs 2[,3]
    readaloud regen <id> [UTTERANCE ...] [--flagged] [--next-seed]     (drop takes; the next synth makes them again)
    readaloud status <id> [--json]
    readaloud sections <id> [--set SECTION=CATEGORY ...]
    readaloud warnings <id> [--all] [--ack ID --note "..."] [--resolve ID --note "..."]
    readaloud characters <id> [--merge FROM INTO | --split ID | --set ID KEY=VALUE ... |
                        --not-character ID | --never-merge A B |
                        --rollback-from CHUNK [--to CHUNK] | --restore ROLLBACK_ID | --reset --yes]
    readaloud cast <id> [--play ID | --reseed ID | --freeze ID | --unfreeze ID]
    readaloud review <id> [--docs 2] [--sample N]
    readaloud relations <id> [--retract A TYPE B | --assert A TYPE B] [--note "..."] [--audit N]
    readaloud eval <id> --pdnc data/pdnc/BOOK [--chapters 1-13] --tag v1 [--note "..."]
    readaloud eval-compare [BOOK:TAG | path.json ...] [--out FILE]   (no refs: list all results)
    readaloud eval-oracle <id> --pdnc data/pdnc/BOOK [--out ID]      (copy with PDNC's characters; then run attribute)
    readaloud ui [--port 8765] [--no-browser]                         (the same operations in a browser, on this machine only)

Exit codes: 0 ok, 1 error, 3 paused by a warning (see `readaloud status <id> --json`, docs/USAGE.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import config, convert, store
from .book import BOOKS, Book, slugify
from .observe import Observer, PipelinePaused
from .stages import (assemble, attribute, cast, characters, classify, collect, ingest, package, roster, segment, synth,
                     voices)

STAGES = ["ingest", "segment", "classify", "collect", "roster", "characters", "cast", "attribute", "voices", "synth",
          "assemble", "package"]
MULTI_VOICE_ONLY = {"collect", "roster", "characters", "cast", "attribute", "voices"}
V3_SKIPS = {"collect", "roster", "characters", "attribute"}   # attribute.version: v3 (docs/DESIGN.md)
V3_ONLY = {"voices"}
V2_ONLY = {"collect", "roster"}                # run only with character_memory.roster: v2 (or named in --stages)
V1_ONLY = {"characters"}                       # replaced by collect + roster when roster: v2
GATED = {"collect", "roster", "characters", "attribute", "voices"}   # refuse to start while pause warnings are open
TAKES_OBS = {"classify", "collect", "roster", "characters", "attribute", "voices"}
RUNNERS = {"ingest": ingest.run, "segment": segment.run, "classify": classify.run, "collect": collect.run, "roster": roster.run,
           "characters": characters.run,
           "cast": cast.run, "attribute": attribute.run, "voices": voices.run, "synth": synth.run, "assemble": assemble.run,
           "package": package.run}
EXIT_PAUSED = 3


def _book(a) -> Book:
    return Book.get(a.id, Path(a.books) if a.books else None)


def _cfg(a, book: Book) -> dict:
    cfg = config.load(book.dir, Path(a.config) if getattr(a, "config", None) else None)
    if getattr(a, "tts", None):
        cfg["tts"]["adapter"] = a.tts
    if getattr(a, "llm", None):
        cfg["llm"]["adapter"] = a.llm
    return cfg


def _docs(a) -> list[int] | None:
    return [int(x) for x in a.docs.split(",")] if getattr(a, "docs", None) else None


def cmd_init(a) -> None:
    src = Path(a.epub)
    if not src.is_file():
        raise SystemExit(f"not a file: {src}")
    book_id = a.id or slugify(src.stem)
    d = (Path(a.books) if a.books else BOOKS) / book_id
    if (d / "source.epub").exists() and not a.force:
        raise SystemExit(f"{d} already exists (use --force to replace the source)")
    d.mkdir(parents=True, exist_ok=True)
    for ext in convert.KINDLE:   # --force over a book that came from another format
        (d / f"source{ext}").unlink(missing_ok=True)
    if convert.is_kindle(src):   # AZW3 / MOBI: converted once to EPUB 3; the original is kept beside it
        try:
            convert.to_epub3(src, d / "source.epub")
        except convert.ConvertError as e:
            if not any(d.iterdir()):
                d.rmdir()
            raise SystemExit(str(e)) from e
        shutil.copyfile(src, d / f"source{src.suffix.lower()}")
        print(f"converted {src.name} to EPUB 3 (Calibre)")
    else:
        shutil.copyfile(src, d / "source.epub")
    store.connect(d).close()
    print(f"initialized {d}\nnext: readaloud run {book_id}")


def cmd_convert(a) -> None:
    failed = convert.convert_queue([Path(p) for p in a.paths], Path(a.out) if a.out else None,
                                   Path(a.done) if a.done else None)
    if failed:
        raise SystemExit(f"{failed} book(s) not converted")


def planned_stages(cfg: dict, until: str = "package", only: list[str] | None = None) -> list[str]:
    """The stages a run with this configuration goes through, in order (the command line and the UI share this)."""
    if only:
        bad = [x for x in only if x not in STAGES]
        if bad:
            raise SystemExit(f"unknown stage: {bad[0]} (stages: {', '.join(STAGES)})")
    todo = [st for st in STAGES if st in only] if only else STAGES[: STAGES.index(until) + 1]
    v2 = cfg["character_memory"].get("roster", "v1") == "v2"
    v3 = str(cfg.get("attribute", {}).get("version", "v2")) == "v3"
    out = []
    for st in todo:
        named = bool(only and st in only)
        if st in MULTI_VOICE_ONLY and not cfg.get("multi_voice") and not (st == "cast" and cast.needs_reference(cfg)):
            continue
        if st in V2_ONLY and not named and not v2:
            continue
        if st in V1_ONLY and v2:
            continue
        if st in V3_SKIPS and v3 and not named:
            continue
        if st in V3_ONLY and not v3 and not named:
            continue
        out.append(st)
    return out


def run_stages(book: Book, cfg: dict, until: str = "package", log=print, doc_ids: list[int] | None = None,
               command: str = "run", only: list[str] | None = None) -> str:
    """Run stages in order (all up to `until`, or just the ones in `only`).
    Returns 'ok', or 'paused' when a pause warning stopped the run."""
    todo = planned_stages(cfg, until, only)
    con = store.connect(book.dir)
    obs = Observer(book.dir, command, printer=log)
    obs.monitor = cfg.get("monitor", {})
    result = "error"
    try:
        for st in todo:
            if st in GATED and obs.open_pauses():
                w = obs.open_pauses()[0]
                raise PipelinePaused(w["code"], w["chunk_id"], "open pause warning(s); handle them first "
                                     f"(`readaloud warnings {book.id}`)")
            t0 = time.time()
            kw = {"doc_ids": doc_ids} if st not in ("ingest", "segment", "classify") else {}
            if st in TAKES_OBS:
                kw["obs"] = obs
            if st == "package" and doc_ids is not None:
                kw["suffix"] = "-docs-" + "-".join(map(str, doc_ids))
            RUNNERS[st](book, con, cfg, log=obs.log, **kw)
            obs.checkpoint()
            obs.log(f"  [{st} {time.time() - t0:.1f}s]", stage=st)
        result = "ok"
    except PipelinePaused as p:
        con.rollback()
        result = "paused"
        obs.log(f"PAUSED {p}. Next: `readaloud status {book.id} --json`, handle per docs/USAGE.md, "
                f"then `readaloud warnings {book.id} --ack <id> --note ...` and rerun.", code=p.code)
    finally:
        obs.finish(result)
        con.close()
    return result


def cmd_run(a) -> None:
    book = _book(a)
    cmdline = "run " + " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "run"
    only = a.stages.split(",") if a.stages else None
    if run_stages(book, _cfg(a, book), a.until, doc_ids=_docs(a), command=cmdline, only=only) == "paused":
        raise SystemExit(EXIT_PAUSED)


def cmd_regen(a) -> None:
    """Drop the takes of some utterances so the next synth generates them again (M2a, minimal).
    --flagged: the ones listed in reports/synth_flagged.tsv (cut after running away on every seed)."""
    book = _book(a)
    con = store.connect(book.dir)
    ids = list(a.utterances or [])
    rep = book.reports / "synth_flagged.tsv"
    if a.flagged and rep.exists():
        ids += [ln.split("\t", 1)[0] for ln in rep.read_text().splitlines()[1:] if ln.strip()]
    if not ids:
        raise SystemExit("nothing to regenerate (give utterance ids or --flagged)")
    takes = {r["id"]: r["wav_path"] for r in con.execute(
        f"SELECT t.id, t.wav_path FROM selection s JOIN take t ON t.id=s.take_id "
        f"WHERE s.block_id IN ({','.join('?' * len(ids))}) AND s.chosen_by != 'manual'", ids)}
    for tid, wav in takes.items():
        con.execute("DELETE FROM selection WHERE take_id=?", (tid,))
        con.execute("DELETE FROM take WHERE id=?", (tid,))
        (book.dir / wav).unlink(missing_ok=True)      # else synth would adopt the file again
    con.commit()
    if a.next_seed:
        # same seed, same audio for a lone request: move these utterances on to the next seeds (reseed.yaml)
        import yaml

        from .stages import synth
        skip = synth.load_reseed(book)
        for i in ids:
            skip[i] = skip.get(i, 0) + synth.RESEEDS + 1
        synth.reseed_path(book).write_text(
            "# utterance id -> seeds to skip (readaloud regen --next-seed); delete a line to go back to the first seed\n"
            + yaml.safe_dump(skip, sort_keys=True))
    if a.flagged and rep.exists():
        rep.rename(rep.with_name("synth_flagged.regenerated.tsv"))
    print(f"{len(takes)} takes dropped for {len(ids)} utterances; now "
          f"`readaloud run {book.id} --tts <adapter> --stages synth,assemble,package`")


def cmd_epubcheck(a) -> None:
    """epubcheck an EPUB already built (e.g. packaged on a machine without epubcheck)."""
    book = _book(a)
    path = Path(a.file) if a.file else max(book.out.glob("*.epub"), key=lambda q: q.stat().st_mtime, default=None)
    if path is None or not path.exists():
        raise SystemExit(f"no EPUB in {book.out}")
    print(f"epubcheck {path}")
    package._epubcheck(book, path, print)
    (book.reports / "epubcheck_skipped.txt").unlink(missing_ok=True)


def cmd_preview(a) -> None:
    """Package only the given documents from takes that already exist (generates nothing)."""
    book = _book(a)
    cfg = _cfg(a, book)
    doc_ids = _docs(a)
    con = store.connect(book.dir)
    try:
        synth.select_existing(book, con, cfg, doc_ids=doc_ids)
        assemble.run(book, con, cfg, doc_ids=doc_ids)
        package.run(book, con, cfg, doc_ids=doc_ids, suffix="-preview-" + "-".join(map(str, doc_ids)))
    finally:
        con.close()


def status_dict(book: Book, con) -> dict:
    """Machine-readable state for the operating agent (docs/USAGE.md)."""
    q1 = lambda sql, *p: con.execute(sql, p).fetchone()[0]  # noqa: E731
    chunks = {}
    for col in ("pass1", "pass2"):
        chunks[col] = {r[0]: r[1] for r in con.execute(f"SELECT {col}, COUNT(*) FROM chunk GROUP BY {col}")}
    sections = {r[0]: {"sections": r[1], "sentences": r[2]} for r in con.execute(
        "SELECT category, COUNT(*), SUM(seq_to - seq_from + 1) FROM section GROUP BY category")}
    warnings = []
    for w in con.execute("SELECT * FROM warning WHERE status='open' ORDER BY id"):
        d = json.loads(w["details"] or "{}")
        item = {"id": w["id"], "code": w["code"], "severity": w["severity"], "chunk_id": w["chunk_id"],
                "count": w["count"], "last_chunk": w["last_chunk"], "created": w["created"], **d}
        ch = con.execute("SELECT * FROM chunk WHERE id=?", (w["chunk_id"],)).fetchone() if w["chunk_id"] else None
        if ch and "opening" not in item:
            item["opening"] = [r[0][:120] for r in con.execute(
                "SELECT text FROM sentence WHERE doc_id=? AND seq BETWEEN ? AND ? ORDER BY seq LIMIT 3",
                (ch["doc_id"], ch["seq_from"], ch["seq_to"]))]
        warnings.append(item)
    last = con.execute("SELECT * FROM run ORDER BY started DESC LIMIT 1").fetchone()
    by_code = {r[0]: r[1] for r in con.execute(
        "SELECT code, COUNT(*) FROM warning WHERE status='open' GROUP BY code")}
    return {
        "book": book.id, "title": store.get_meta(con, "title", ""),
        "last_run": dict(last) if last else None,
        "sections": sections,
        "characters": {"chunks_pass1": chunks["pass1"], "count": q1(
            "SELECT COUNT(*) FROM character WHERE status IN ('active','provisional')"),
            "provisional": q1("SELECT COUNT(*) FROM character WHERE status='provisional'")},
        "attribute": {"chunks_pass2": chunks["pass2"],
                      "quotes": q1("SELECT COUNT(DISTINCT quote_id) FROM utterance WHERE is_quote=1"),
                      "unknown": q1("SELECT COUNT(DISTINCT quote_id) FROM utterance WHERE is_quote=1 "
                                    "AND speaker_id='unknown'"),
                      "low_confidence": q1("SELECT COUNT(DISTINCT quote_id) FROM utterance WHERE is_quote=1 "
                                           "AND confidence < 0.6 AND source != 'manual'")},
        "synth": {"utterances": q1("SELECT COUNT(*) FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                                   "WHERE s.speak=1"),
                  "selected": q1("SELECT COUNT(*) FROM selection")},
        "warnings": warnings,
        "review": {"merge_review": by_code.get("MERGE_REVIEW", 0), "attr_conflict": by_code.get("ATTR_CONFLICT", 0),
                   "alias_conflict": by_code.get("ALIAS_CONFLICT", 0),
                   "section_uncertain": by_code.get("SECTION_UNCERTAIN", 0)},
        "rollbacks": [dict(r) for r in con.execute(
            "SELECT id, created, note, restored_by, json_array_length(chunks) n_chunks FROM rollback "
            "ORDER BY created DESC LIMIT 5")],
        "outputs": [str(o) for o in sorted(book.out.glob("*.epub"))] if book.out.exists() else [],
    }


def cmd_status(a) -> None:
    book = _book(a)
    con = store.connect(book.dir)
    if a.json:
        print(json.dumps(status_dict(book, con), ensure_ascii=False, indent=1, default=str))
        return
    print(f"{book.id}: {store.get_meta(con, 'title', '?')}  (EPUB {store.get_meta(con, 'epub_version_in', '?')} in)")
    rows = con.execute(
        "SELECT d.id, d.href, COUNT(DISTINCT s.id) n, COUNT(u.id) nu, COUNT(sel.take_id) synth "
        "FROM doc d LEFT JOIN sentence s ON s.doc_id=d.id AND s.speak=1 "
        "LEFT JOIN utterance u ON u.sentence_id=s.id LEFT JOIN selection sel ON sel.block_id=u.id "
        "WHERE d.has_speech=1 GROUP BY d.id ORDER BY d.id").fetchall()
    chunks = {r["doc_id"]: (r["p1"], r["p2"], r["n"]) for r in con.execute(
        "SELECT doc_id, SUM(pass1='done') p1, SUM(pass2='done') p2, COUNT(*) n FROM chunk GROUP BY doc_id")}
    print("  doc  file                                     sentences  utter. synth  chunks(p1/p2)")
    for r in rows:
        built = "assembled" if (book.chapters / f"{r['id']:03d}.wav").exists() else ""
        ch = chunks.get(r["id"])
        cs = f"{ch[0]}/{ch[1]} of {ch[2]}" if ch else "-"
        print(f"  {r['id']:3d}  {r['href'][-40:]:40s} {r['n']:9d} {r['nu']:7d} {r['synth']:5d}  {cs:12s} {built}")
    st = status_dict(book, con)
    print(f"  sections: " + ", ".join(f"{k} {v['sentences']}" for k, v in st["sections"].items()))
    print(f"  characters: {st['characters']['count']} ({st['characters']['provisional']} provisional)")
    if st["warnings"]:
        print(f"  open warnings: {len(st['warnings'])} -> `readaloud warnings {book.id}`")
    for o in st["outputs"]:
        print(f"  output: {o}")


def cmd_sections(a) -> None:
    book = _book(a)
    con = store.connect(book.dir)
    if a.set:
        assign = dict(x.split("=", 1) for x in a.set)
        affected = classify.set_categories(book, con, assign)
        cfg = _cfg(a, book)
        try:
            classify.run(book, con, cfg, log=lambda *_: None)
        except PipelinePaused:
            pass
        print(f"set {len(assign)} section(s)")
        if affected:
            print(f"character pass is out of date for: {', '.join(affected)}\n"
                  f"  -> readaloud characters {book.id} --rollback-from {affected[0]} --to {affected[-1]}   then rerun")
    for r in con.execute("SELECT s.*, (SELECT text FROM sentence x WHERE x.doc_id=s.doc_id AND x.seq=s.seq_from) t "
                         "FROM section s ORDER BY doc_id, seq_from"):
        if a.category and r["category"] != a.category:
            continue
        print(f"  {r['id']}  {r['seq_from']:5d}-{r['seq_to']:<5d} {r['category']:12s} {r['source']:6s} "
              f"{(r['confidence'] or 0):.2f}  {(r['t'] or '')[:60]}")


def cmd_warnings(a) -> None:
    book = _book(a)
    con = store.connect(book.dir)
    for wid, status in ((a.ack, "acknowledged"), (a.resolve, "resolved")):
        if wid:
            n = con.execute("UPDATE warning SET status=?, handled_by=?, note=? WHERE id=?",
                            (status, a.by, a.note or "", wid)).rowcount
            con.commit()
            print(f"warning {wid}: {status}" if n else f"no warning {wid}")
    where = "" if a.all else "WHERE status='open'"
    for w in con.execute(f"SELECT * FROM warning {where} ORDER BY id"):
        d = json.loads(w["details"] or "{}")
        cnt = f"x{w['count']}" if (w["count"] or 1) > 1 else ""
        print(f"  #{w['id']:<4d} {w['code']:18s} {w['severity']:5s} {w['status']:12s} {w['chunk_id'] or '':12s} "
              f"{cnt:5s} {d.get('message', '')[:90]}")


def cmd_characters(a) -> None:
    from .memory.db import CharacterDB
    from .memory.rollback import chunk_range, mark_stale, restore, rollback
    book = _book(a)
    con = store.connect(book.dir)
    db = CharacterDB(con, run_id="manual")
    orphans = None
    if a.reset:
        if not a.yes:
            raise SystemExit("--reset voids ALL model character results (manual edits are kept; undo with "
                             "--restore); add --yes")
        rid, orphans = characters.reset(book, con, run_id="manual")
        print(f"reset as rollback {rid} (undo: readaloud characters {book.id} --restore {rid}); rerun `readaloud run`")
    elif a.rollback_from:
        chunks = chunk_range(con, a.rollback_from, a.to)
        rid, orphans = rollback(con, chunks, run_id="manual", note=a.note or "")
        print(f"rollback {rid}: {len(chunks)} chunks back to pending ({chunks[0]} .. {chunks[-1]}); "
              f"undo: readaloud characters {book.id} --restore {rid}")
    elif a.restore:
        rid, orphans = restore(con, a.restore, run_id="manual")
        print(f"restored {a.restore} (as {rid})")
    elif a.merge:
        src, dst = a.merge
        db.apply("merge", {"from": src, "into": dst, "reason": "manual"}, source="manual")
    elif a.split:
        db.apply("split", {"id": a.split}, source="manual")
        mark_stale(con, char_ids={a.split, db.resolve(a.split)})
    elif a.set:
        cid, *pairs = a.set
        upd = dict(p.split("=", 1) for p in pairs)
        db.apply("profile", {"id": cid, **upd}, source="manual")
    elif a.not_character:
        db.apply("constraint", {"kind": "not_character", "id": a.not_character}, source="manual")
        mark_stale(con, char_ids={a.not_character})
    elif a.never_merge:
        x, y = a.never_merge
        db.apply("constraint", {"kind": "never_merge", "a": x, "b": y}, source="manual")
        if db.resolve(x) == db.resolve(y):
            print(f"note: {x} and {y} are currently merged; split them with --split")
    con.commit()
    if orphans:
        print(f"{len(orphans)} manual edits refer to characters that no longer exist (kept, flagged for review):")
        obs = Observer(book.dir, "characters")
        for o in orphans:
            print(f"  event #{o['event_id']} {o['type']} {o.get('id') or o.get('from')}")
            obs.warn("ORPHANED_EDIT", "warn", None, f"manual {o['type']} on missing character", **o)
        obs.finish("ok")
    characters.export_yaml(book, db)
    characters.write_report(book, con, db)
    for cd in db.roster():
        aka = ", ".join(x for x in cd.aliases if x != cd.name)[:60]
        print(f"  {cd.id:28s} {cd.mentions:4d}  {cd.gender or '?':7s} {cd.age or '?':12s} {cd.status:11s} {aka}")
    for m in con.execute("SELECT id, merged_into, status FROM character WHERE status IN ('merged','not_character')"):
        print(f"  ({m['status']}) {m['id']}" + (f" -> {m['merged_into']}" if m["merged_into"] else ""))


def cmd_cast(a) -> None:
    book = _book(a)
    cfg = _cfg(a, book)
    data = cast.load_cast(book, cfg)
    chars = data.get("characters", {})
    entries = {"narrator": data.get("narrator") or {}, **chars}
    path = cast.cast_path(book, cfg).name
    if a.play:
        p = cast.samples_dir(book, cfg) / f"{a.play}.wav"
        if not p.exists():
            raise SystemExit(f"no sample {p}; run `readaloud run {book.id} --stages cast`")
        subprocess.run(["afplay", str(p)], check=False)
        return
    target = a.reseed or a.freeze or a.unfreeze
    if target and target not in entries:
        raise SystemExit(f"{target} not in {path}")
    if a.reseed:
        e = entries[a.reseed]
        if e.get("frozen"):
            raise SystemExit(f"{a.reseed} is frozen; --unfreeze first")
        if a.reseed == "narrator" and cast.designer_name(cfg) == "kokoro-mix":
            raise SystemExit(f"the Kokoro narrator is tts.voice (or `narrator.voice` in {path}); edit it there")
        con = store.connect(book.dir)
        from .memory.db import CharacterDB
        card = CharacterDB(con).card(a.reseed) if a.reseed != "narrator" else None
        e["seed"] = int(e.get("seed", 0)) + 7919
        designer = cast.designer_for(cfg, book)
        taken = {p for k, v in entries.items() if k != a.reseed and v.get("voice") for p in designer.primaries(v["voice"])}
        if a.reseed == "narrator":
            g, age = (cfg.get("cast") or {}).get("narrator_gender", ""), "adult"
        else:
            g, age = (card.gender, card.age) if card else cast.GENERIC.get(a.reseed, ("", ""))
        e["voice"], e["design"] = designer.design(g, age, card.accent if card else "", e["seed"], taken)
        e["basis"] = cast._narrator_basis(cfg) if a.reseed == "narrator" else f"{g}/{age}/{card.accent if card else ''}"
        e["auto_voice"] = e["voice"]
        cast.save_cast(book, data, cfg)
        cast._samples(book, con, cfg, data, print, getattr(designer, "tts", None))
        print(f"{a.reseed}: {e['voice']}  ({e['design']})")
        return
    if a.freeze or a.unfreeze:
        entries[target]["frozen"] = bool(a.freeze)
        cast.save_cast(book, data, cfg)
    print(f"  {path}")
    for cid, e in entries.items():
        print(f"  {cid:28s} {'F' if e.get('frozen') else ' '} {str(e.get('voice')):42s} {e.get('design', '')}")


def cmd_review(a) -> None:
    """Things a human should look at: low-confidence / unknown attributions, uncertain merges, rare characters."""
    import json
    book = _book(a)
    con = store.connect(book.dir)
    doc_ids = _docs(a)
    if a.sample:
        path = _sample_attributions(book, con, doc_ids, a.sample)
        print(f"wrote {path}: fill `correct` (y/n) and `correct_speaker`, then count")
        return
    q = "" if doc_ids is None else f"AND s.doc_id IN ({','.join(map(str, doc_ids))})"
    rows = con.execute(
        "SELECT u.quote_id, u.speaker_id, u.style, MIN(u.confidence) c, GROUP_CONCAT(u.tts_text, ' ') t "
        f"FROM utterance u JOIN sentence s ON s.id=u.sentence_id WHERE u.is_quote=1 AND u.source!='manual' {q} "
        "GROUP BY u.quote_id HAVING c < ? OR u.speaker_id='unknown' ORDER BY s.doc_id, s.seq",
        (a.min_confidence,)).fetchall()
    print(f"attributions to check ({len(rows)}); fix in attribution_overrides.csv (quote_id,speaker,style,note):")
    for r in rows:
        print(f"  {r['quote_id']:16s} {r['speaker_id']:18s} conf {r['c'] if r['c'] is not None else '-':<4} {r['t'][:70]}")
    ev = [json.loads(r["payload"]) for r in con.execute(
        "SELECT payload FROM character_event WHERE type='edit' AND payload LIKE '%needs_review\": true%'")]
    if ev:
        print("uncertain merges (readaloud characters --merge A B to accept):")
        for e in ev:
            print(f"  {e['pair'][0]} ? {e['pair'][1]}  ({e['confidence']}) {e['reason'][:80]}")
    rare = con.execute("SELECT id, name, mentions FROM character WHERE status IN ('active','provisional') "
                       "AND mentions <= 2 ORDER BY id").fetchall()
    if rare:
        print("characters seen only once or twice:")
        for r in rare:
            print(f"  {r['id']:28s} {r['name']} ({r['mentions']})")


def _sample_attributions(book: Book, con, doc_ids, n: int, seed: int = 7) -> Path:
    """Random sample of attributed quotations with context, as a CSV for manual scoring."""
    import csv
    import random
    q = "" if doc_ids is None else f"AND s.doc_id IN ({','.join(map(str, doc_ids))})"
    quotes = con.execute(
        "SELECT u.quote_id, u.speaker_id, u.style, MIN(s.doc_id) doc, MIN(s.seq) seq, MAX(s.seq) seq2, "
        "GROUP_CONCAT(u.tts_text, ' ') t FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
        f"WHERE u.is_quote=1 {q} GROUP BY u.quote_id ORDER BY doc, seq").fetchall()
    pick = sorted(random.Random(seed).sample(range(len(quotes)), min(n, len(quotes))))
    names = {r["id"]: r["name"] for r in con.execute("SELECT id, name FROM character")}
    path = book.reports / "attribution_sample.csv"
    book.reports.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quote_id", "speaker", "speaker_name", "style", "quote", "context", "correct", "correct_speaker"])
        for i in pick:
            r = quotes[i]
            ctx = con.execute("SELECT tts_text FROM sentence WHERE doc_id=? AND seq BETWEEN ? AND ? ORDER BY seq",
                              (r["doc"], r["seq"] - 2, r["seq2"] + 1)).fetchall()
            w.writerow([r["quote_id"], r["speaker_id"], names.get(r["speaker_id"], r["speaker_id"]), r["style"] or "",
                        r["t"], " ".join(c["tts_text"] for c in ctx), "", ""])
    return path


def cmd_relations(a) -> None:
    """List the relation graph; record a human retraction/assertion in relations.yaml (docs/DESIGN.md; the roster stage applies it); --audit N prints N accepted relations with their evidence."""
    import random
    import yaml
    from .memory.db import CharacterDB
    from .memory.kin import DIRECTED, TYPES
    book = _book(a)
    con = store.connect(book.dir)
    db = CharacterDB(con)
    triple = a.retract or a.assert_
    if triple:
        x, typ, y = triple
        if typ not in TYPES:
            raise SystemExit(f"type must be one of {', '.join(sorted(TYPES))}")
        for cid in (x, y):
            if not db.exists(cid):
                raise SystemExit(f"no such character: {cid}")
        x, y = db.resolve(x), db.resolve(y)
        if typ not in DIRECTED:
            x, y = min(x, y), max(x, y)
        p = book.dir / "relations.yaml"
        data = (yaml.safe_load(p.read_text()) if p.exists() else None) or {}
        key = "retract" if a.retract else "assert"
        other = "assert" if a.retract else "retract"
        entry = {"a": x, "type": typ, "b": y, **({"note": a.note} if a.note else {})}
        data[other] = [r for r in (data.get(other) or []) if (r["a"], r["type"], r["b"]) != (x, typ, y)]
        if all((r["a"], r["type"], r["b"]) != (x, typ, y) for r in (data.get(key) or [])):
            data[key] = (data.get(key) or []) + [entry]
        p.write_text("# Human decisions on relations (docs/DESIGN.md). assert: always in; retract: never.\n"
                     "# Applied by the roster stage: rerun `readaloud run <id> --stages roster,attribute`.\n"
                     + yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
        db.apply(f"relation_{key}", {"a": x, "type": typ, "b": y, "note": a.note or ""}, source="manual")
        con.commit()
        print(f"{key}: {x} {typ} {y} -> {p.name}; rerun `readaloud run {book.id} --stages roster,attribute`")
        return
    rows = con.execute("SELECT * FROM relation ORDER BY strength, type, a").fetchall()
    if a.audit:
        pick = random.Random(0).sample(rows, min(a.audit, len(rows)))
        for r in pick:
            print(f"{r['a']} {r['type']} {r['b']}  [{r['strength']}, {r['source']}]")
            for c in con.execute("SELECT kind, chunk_id, evidence FROM relation_claim WHERE a=? AND type=? AND b=? "
                                 "LIMIT 4", (r["a"], r["type"], r["b"])):
                print(f"    {c['kind']:<10} {c['chunk_id'] or '-':<9} {c['evidence'][:150]}")
        return
    claims = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM relation_claim GROUP BY 1")}
    print(f"{book.id}: {len(rows)} relations; observations: "
          + ", ".join(f"{k} {v}" for k, v in sorted(claims.items())))
    for r in rows:
        print(f"  {r['strength'] or 'strong':<6} {r['a']} {r['type']} {r['b']}"
              + (f"  (from {r['valid_from']})" if r["valid_from"] and r["type"] == "spouse_of" else ""))
    from .store import get_meta
    if get_meta(con, "relations_disabled") == "1":
        print("RELATION_ANOMALY: attribution does not use this graph (see reports/roster_report.md)")


def cmd_eval(a) -> None:
    from .eval import pdnc
    book = _book(a)
    con = store.connect(book.dir)
    src = Path(a.pdnc)
    ov = Path(a.overrides) if a.overrides else Path(__file__).resolve().parents[1] / "data" / "pdnc_overrides" / f"{src.name}.yaml"
    chapters = tuple(int(x) for x in a.chapters.split("-")) if a.chapters else None
    relf = Path(a.relations) if a.relations else ov.with_name(f"{src.name}.relations.yaml")
    res = pdnc.evaluate(con, src, ov if ov.exists() else None, chapters, cfg=_cfg(a, book),
                        relations=relf if relf.exists() else None)
    js = pdnc.write_reports(res, book.dir / "reports" / "eval", a.tag, book.id, a.note)
    m = res["metrics"]
    print(f"eval {book.id}:{a.tag}  quotes {m['quotes']}  aligned {m['align_pct']}%  "
          f"main recall {m['main_recall_pct']}%  merge {m['false_merge']}  split {m['false_split']}  "
          f"attribution {m['attr_pct']}%  explicit {m['explicit_pct']}%  human {m['human_load']}  "
          f"coverage {m['candidate_coverage_pct']}%"
          + (f"  relations P {m['rel_precision_pct']}% R {m['rel_recall_pct']}% core {m['rel_core_recall_pct']}% "
             f"lookups {m['kin_lookup_pass']}" if "rel_precision_pct" in m else ""))
    if res["run"]["pass2_done"] < res["run"]["chunks_in_range"]:
        print(f"  note: attribute done for {res['run']['pass2_done']}/{res['run']['chunks_in_range']} chunks in range")
    print(f"  {js.with_suffix('.md')}")


def cmd_eval_oracle(a) -> None:
    from .eval import oracle
    book = _book(a)
    src = Path(a.pdnc)
    ov = Path(a.overrides) if a.overrides else Path(__file__).resolve().parents[1] / "data" / "pdnc_overrides" / f"{src.name}.yaml"
    out = book.dir.parent / (a.out or f"{book.id}-oracle")
    oracle.make_oracle_book(book.dir, out, src, ov if ov.exists() else None)
    print(f"next: readaloud run {out.name} --stages attribute [--docs ...]; "
          f"readaloud eval {out.name} --pdnc {a.pdnc} --tag <tag>-oracle")


def _eval_ref(a, ref: str) -> dict:
    p = Path(ref)
    if p.suffix == ".json" and p.exists():
        return json.loads(p.read_text())
    if ":" not in ref:
        raise SystemExit(f"eval-compare: expected BOOK:TAG or a .json path, got {ref!r}")
    bid, tag = ref.split(":", 1)
    p = Book.get(bid, Path(a.books) if a.books else None).dir / "reports" / "eval" / f"{tag}.json"
    if not p.exists():
        raise SystemExit(f"eval-compare: no result {p}")
    return json.loads(p.read_text())


def cmd_eval_compare(a) -> None:
    from .eval import pdnc
    if not a.refs:
        print(pdnc.list_results(Path(a.books) if a.books else BOOKS), end="")
        return
    md = pdnc.compare([_eval_ref(a, r) for r in a.refs])
    if a.out:
        Path(a.out).write_text(md)
        print(a.out)
    else:
        print(md, end="")


def cmd_ui(a) -> None:
    from .ui import server
    server.serve(Path(a.books) if a.books else BOOKS, a.host, a.port, open_browser=not a.no_browser)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="readaloud", description="EPUB -> EPUB 3 Media Overlays audiobook")
    ap.add_argument("--books", help="books root (default: ./books)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="new book from an EPUB (or an AZW3 / MOBI, converted to EPUB 3 with Calibre)")
    p.add_argument("epub")
    p.add_argument("--id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("convert", help="AZW3 / MOBI -> EPUB 3 with Calibre (files, or folders like data/next)")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", help="folder for the EPUBs (default: next to each original)")
    p.add_argument("--done", help="move each converted original here (e.g. data/epub)")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("run")
    p.add_argument("id")
    p.add_argument("--until", choices=STAGES, default="package")
    p.add_argument("--stages", help="run only these stages, e.g. characters,attribute (in pipeline order)")
    p.add_argument("--docs", help="limit LLM passes, synthesis and packaging to these doc ids (e.g. 2,3)")
    p.add_argument("--tts", choices=["mock", "kokoro", "drama3"], help="override tts.adapter")
    p.add_argument("--llm", choices=["mock", "openai_compat"], help="override llm.adapter")
    p.add_argument("--config", help="base config file (default: repo config.yaml)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("preview", help="package selected documents (ids from `readaloud status`) from existing takes")
    p.add_argument("id")
    p.add_argument("--docs", required=True, help="comma-separated doc ids, e.g. 2 or 2,3")
    p.add_argument("--tts", choices=["mock", "kokoro", "drama3"], help="override tts.adapter")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_preview)

    p = sub.add_parser("regen", help="drop utterance takes so the next synth generates them again")
    p.add_argument("id")
    p.add_argument("utterances", nargs="*", help="utterance ids, e.g. ab063s00061u0")
    p.add_argument("--flagged", action="store_true", help="the ones in reports/synth_flagged.tsv")
    p.add_argument("--next-seed", action="store_true",
                   help="also move them on to the next seeds, for a take that came out wrong the same way again")
    p.set_defaults(fn=cmd_regen)

    p = sub.add_parser("epubcheck", help="epubcheck a built EPUB against the source book's own errors")
    p.add_argument("id")
    p.add_argument("--file", help="EPUB to check (default: the newest in out/)")
    p.set_defaults(fn=cmd_epubcheck)

    p = sub.add_parser("status")
    p.add_argument("id")
    p.add_argument("--json", action="store_true", help="machine-readable state for agents")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("sections", help="list / set content categories of sections")
    p.add_argument("id")
    p.add_argument("--set", nargs="+", metavar="SECTION=CATEGORY")
    p.add_argument("--category", help="only list this category")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_sections)

    p = sub.add_parser("warnings", help="list / acknowledge / resolve warnings")
    p.add_argument("id")
    p.add_argument("--all", action="store_true", help="include handled warnings")
    p.add_argument("--ack", type=int, metavar="ID")
    p.add_argument("--resolve", type=int, metavar="ID")
    p.add_argument("--note")
    p.add_argument("--by", default="agent")
    p.set_defaults(fn=cmd_warnings)

    p = sub.add_parser("characters", help="list / merge / split / edit characters")
    p.add_argument("id")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--merge", nargs=2, metavar=("FROM", "INTO"))
    g.add_argument("--split", metavar="ID")
    g.add_argument("--set", nargs="+", metavar="ID KEY=VALUE", help="e.g. --set alice gender=female age=child")
    g.add_argument("--reset", action="store_true", help="void all model results, keep manual edits (needs --yes)")
    g.add_argument("--rollback-from", metavar="CHUNK")
    g.add_argument("--restore", metavar="ROLLBACK_ID")
    g.add_argument("--not-character", metavar="ID")
    g.add_argument("--never-merge", nargs=2, metavar=("A", "B"))
    p.add_argument("--to", metavar="CHUNK", help="last chunk for --rollback-from (default: end of book)")
    p.add_argument("--note")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(fn=cmd_characters)

    p = sub.add_parser("cast", help="list / play / reseed / freeze character voices")
    p.add_argument("id")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--play", metavar="ID")
    g.add_argument("--reseed", metavar="ID")
    g.add_argument("--freeze", metavar="ID")
    g.add_argument("--unfreeze", metavar="ID")
    p.add_argument("--tts", choices=["mock", "kokoro", "drama3"])
    p.add_argument("--config")
    p.set_defaults(fn=cmd_cast)

    p = sub.add_parser("review", help="list attributions and characters that need a human look")
    p.add_argument("id")
    p.add_argument("--docs")
    p.add_argument("--min-confidence", type=float, default=0.6)
    p.add_argument("--sample", type=int, metavar="N", help="write N random attributions to reports/attribution_sample.csv")
    p.set_defaults(fn=cmd_review)

    p = sub.add_parser("relations", help="list / retract / assert relations; --audit N shows evidence")
    p.add_argument("id")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--retract", nargs=3, metavar=("A", "TYPE", "B"))
    g.add_argument("--assert", dest="assert_", nargs=3, metavar=("A", "TYPE", "B"))
    g.add_argument("--audit", type=int, metavar="N")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_relations)

    p = sub.add_parser("eval", help="score characters/attribution against PDNC (docs/DESIGN.md)")
    p.add_argument("id")
    p.add_argument("--pdnc", required=True, help="PDNC book dir, e.g. data/pdnc/PrideAndPrejudice")
    p.add_argument("--overrides", help="default: data/pdnc_overrides/<book>.yaml")
    p.add_argument("--relations", help="gold relations; default: data/pdnc_overrides/<book>.relations.yaml")
    p.add_argument("--chapters", help="e.g. 1-13 (by chapter headings)")
    p.add_argument("--tag", required=True, help="result name; same tag again moves the old one to history/")
    p.add_argument("--note")
    p.add_argument("--config")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("eval-oracle", help="copy a book with PDNC's character list (upper-bound experiment)")
    p.add_argument("id")
    p.add_argument("--pdnc", required=True)
    p.add_argument("--overrides")
    p.add_argument("--out", help="new book id (default <id>-oracle)")
    p.set_defaults(fn=cmd_eval_oracle)

    p = sub.add_parser("eval-compare", help="compare eval results side by side (no refs: list them)")
    p.add_argument("refs", nargs="*", metavar="BOOK:TAG|path.json")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_eval_compare)

    p = sub.add_parser("ui", help="the same operations in a browser (a local web page; nothing leaves this machine)")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(fn=cmd_ui)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
