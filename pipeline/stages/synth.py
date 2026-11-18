"""Stage: synth. One take per block; a block is one utterance (block_mode: sentence).

Each utterance's voice comes from cast.yaml via its speaker (narrator voice
when there is no cast). Takes are cached by key = hash(tts_text, voice,
adapter, model version, params, seed); an existing key with its WAV on disk
is never regenerated. Requests run concurrently so a batching TTS server
(vLLM-Omni) stays busy.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import audio
from ..adapters.tts import SynthRequest, Utterance, get_tts, settings
from ..book import Book
from ..chunks import scope_docs
from ..memory.db import CharacterDB


def take_key(tts_text: str, voice: str, adapter: str, model_version: str, params: dict, seed: int) -> str:
    blob = json.dumps({"text": tts_text, "voice": voice, "adapter": adapter, "model": model_version,
                       "params": params, "seed": seed}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def plan(book: Book, con, cfg: dict, tts, doc_ids: list[int] | None = None) -> list[dict]:
    """Utterances in scope with their voice and take key, in reading order."""
    from .cast import cast_path, load_cast, needs_reference, voice_for
    t = settings(cfg["tts"])
    book_cast = load_cast(book, cfg) if cast_path(book, cfg).exists() else None
    if book_cast is None and needs_reference(cfg):
        raise SystemExit(f"synth: {tts.name} clones reference voices; run the cast stage first "
                         f"(`readaloud run {book.id} --stages cast`), then listen to the samples")
    db = CharacterDB(con)
    docs = scope_docs(con, doc_ids)
    q = ",".join("?" * len(docs))
    out = []
    for r in con.execute(
            f"SELECT u.id, u.tts_text, u.speaker_id, u.style FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
            f"WHERE s.doc_id IN ({q}) AND s.speak=1 ORDER BY s.doc_id, s.seq, u.seq", docs):
        voice = voice_for(book_cast, db, r["speaker_id"]) if book_cast else t["voice"]
        out.append({"id": r["id"], "text": r["tts_text"], "voice": voice, "style": r["style"],
                    "key": take_key(r["tts_text"], voice, tts.name, tts.model_version, t.get("params", {}), t["seed"])})
    return out


def select_existing(book: Book, con, cfg: dict, tts=None, doc_ids=None) -> None:
    """Point utterances at already-generated takes without generating anything (for previews)."""
    from .cast import refs_dir
    tts = tts or get_tts(cfg["tts"], refs_dir(book))
    select(con, {u["id"]: u["key"] for u in plan(book, con, cfg, tts, doc_ids)})


def run(book: Book, con, cfg: dict, log=print, tts=None, doc_ids: list[int] | None = None) -> None:
    from .cast import refs_dir
    t = settings(cfg["tts"])
    tts = tts or get_tts(cfg["tts"], refs_dir(book))
    items = plan(book, con, cfg, tts, doc_ids)
    have = {r["key"] for r in con.execute("SELECT key FROM take")}
    todo, seen, adopted = [], set(), 0
    for u in items:
        wav = book.takes / f"{u['key']}.wav"
        if (u["key"] in have and wav.exists()) or u["key"] in seen:
            continue
        seen.add(u["key"])
        if u["key"] not in have and wav.exists():
            # Written by a run that stopped before saving its row. audio.write renames a finished temp
            # file into place, so the file is complete; the seed it used is unknown.
            _save_take(con, u, tts, t, None, audio.duration(wav))
            adopted += 1
            continue
        todo.append(u)
    con.commit()
    voices = len({u["voice"] for u in items})
    log(f"synth: {len(items)} utterances, {voices} voices, {len(items) - len(todo)} cached"
        + (f" ({adopted} adopted from takes/)" if adopted else "") + f", {len(todo)} to generate "
        f"({tts.name} {tts.model_version}, concurrency {t.get('concurrency', 1)})")
    if todo:
        _generate(book, con, t, tts, todo, log)
    select(con, {u["id"]: u["key"] for u in items})


def select(con, keys: dict[str, str]) -> None:
    """Point each utterance block at the take for the current config (manual picks win)."""
    ids = {r["key"]: r["id"] for r in con.execute("SELECT id, key FROM take")}
    manual = {r["block_id"] for r in con.execute("SELECT block_id FROM selection WHERE chosen_by='manual'")}
    con.executemany("INSERT OR REPLACE INTO selection(block_id, take_id, chosen_by) VALUES (?,?, 'auto')",
                    [(bid, ids[k]) for bid, k in keys.items() if k in ids and bid not in manual])
    con.commit()


RESEEDS = 2          # extra seeds tried when the adapter says a take failed in a way another seed may fix
COMMIT_EVERY = 20    # takes; also at least every COMMIT_SECONDS, so a killed run loses little
COMMIT_SECONDS = 30.0


def _save_take(con, u, tts, t, seed, dur) -> None:
    con.execute("INSERT OR REPLACE INTO take(block_id, key, adapter, model_version, params, seed, "
                "wav_path, duration) VALUES (?,?,?,?,?,?,?,?)",
                (u["id"], u["key"], tts.name, tts.model_version, json.dumps(t.get("params", {})), seed,
                 f"takes/{u['key']}.wav", dur))


CUT_CPS = 9.0        # on_runaway: cut keeps len(text) / CUT_CPS + 1 s of the last runaway take


def reseed_path(book: Book):
    return book.dir / "reseed.yaml"


def load_reseed(book: Book) -> dict[str, int]:
    """utterance id -> how many seeds to skip (`readaloud regen --next-seed`). A lone request with the same seed
    gives the same audio again, so a take that came out wrong but did not run away needs other seeds."""
    import yaml
    p = reseed_path(book)
    return {str(k): int(v) for k, v in ((yaml.safe_load(p.read_text()) if p.exists() else None) or {}).items()}


def _generate(book: Book, con, t: dict, tts, todo, log) -> None:
    flagged = []
    skip = load_reseed(book)

    def job(u):
        # The key names the slot (text, voice, model, params, base seed); take.seed records the seed used.
        for k in range(RESEEDS + 1):
            seed = t["seed"] + skip.get(u["id"], 0) + k
            try:
                res = tts.synthesize(SynthRequest([Utterance(u["text"], u["voice"], u["style"])], seed=seed,
                                                  params=t.get("params", {})))
                a = res.audio
                break
            except Exception as e:
                if not getattr(e, "reseed", False):
                    raise
                if k == RESEEDS:
                    # Every seed ran away. Unattended runs (tts.<adapter>.on_runaway: cut) keep the start of the
                    # last take, where the text usually is, and flag it for a listen instead of blocking the book.
                    if t.get("on_runaway", "fail") != "cut" or getattr(e, "audio", None) is None:
                        raise
                    a = e.audio[: int(tts.sample_rate * (len(u["text"]) / CUT_CPS + 1.0))]
                    flagged.append((u["id"], seed, f"{e}"))
                    log(f"synth: {u['id']}: every seed ran away; kept the first {a.size / tts.sample_rate:.1f} s, "
                        f"flagged in reports/synth_flagged.tsv")
                    break
                log(f"synth: {u['id']}: {e}; retrying with seed {seed + 1}")
        audio.write(book.takes / f"{u['key']}.wav", a, tts.sample_rate)
        return a.size / tts.sample_rate, seed

    t0, done, failed, audio_sec, down = time.time(), 0, [], 0.0, None
    made = set()                 # utterances that got a take in this run
    last_commit = t0
    step = max(10, len(todo) // 20)
    with ThreadPoolExecutor(max(1, int(t.get("concurrency", 1)))) as ex:
        futs = {ex.submit(job, it): it for it in todo}
        for f in as_completed(futs):
            if f.cancelled():
                continue
            u = futs[f]
            try:
                dur, seed = f.result()
            except Exception as e:  # keep going; report at the end
                failed.append((u["id"], f"{type(e).__name__}: {e}"))
                if getattr(e, "service_down", False) and down is None:
                    # The service is gone: stop instead of failing every remaining utterance one by one.
                    down = e
                    for g in futs:
                        g.cancel()
                    log(f"synth: TTS service down, stopping after the requests in flight: {e}")
                continue
            _save_take(con, u, tts, t, seed, dur)
            made.add(u["id"])
            done += 1
            audio_sec += dur
            if done % COMMIT_EVERY == 0 or time.time() - last_commit > COMMIT_SECONDS:
                con.commit()
                last_commit = time.time()
            if done % step == 0 or done == len(todo):
                el = time.time() - t0
                log(f"synth: {done}/{len(todo)}  {el:.0f}s elapsed, RTF {el / max(audio_sec, 1e-9):.2f}")
    con.commit()
    # The report lists the takes that are cut now: a line generated again in this run replaces its old entry,
    # and drops out of the report when the new take is whole.
    rep = book.reports / "synth_flagged.tsv"
    old = [ln for ln in rep.read_text().splitlines()[1:] if ln.strip()] if rep.exists() else []
    kept = [ln for ln in old if ln.split("\t", 1)[0] not in made]
    lines = kept + [f"{uid}\t{seed}\t{why}" for uid, seed, why in flagged]
    if lines:
        book.reports.mkdir(parents=True, exist_ok=True)
        rep.write_text("utterance\tseed\treason (listen; regenerate with another voice seed or fix the text)\n"
                       + "\n".join(lines) + "\n")
    elif rep.exists():
        rep.unlink()
    if flagged:
        log(f"synth: {len(flagged)} takes cut after running away on every seed; listen to them: {rep}")
    if down is not None:
        raise SystemExit(f"synth: TTS service down ({down}); {done} takes saved this run, rerun to resume")
    if failed:
        for sid, err in failed[:10]:
            log(f"synth: FAILED {sid}: {err}")
        raise SystemExit(f"synth: {len(failed)} utterances failed; rerun to retry")
