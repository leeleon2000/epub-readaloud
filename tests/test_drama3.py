"""Drama 3 adapter, the Drama 3 voice designer and the cast/synth flow, against a fake server."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline import audio, config, store
from pipeline.adapters.tts import SynthRequest, Utterance, get_tts, settings
from pipeline.adapters.tts.drama3 import TTSError, TTSTruncated, TTSUnavailable
from pipeline.adapters.voice.drama3 import pitch_gender
from pipeline.book import Book
from pipeline.cli import main, run_stages

from fake_drama3 import FakeDrama3

FIXTURE = Path(__file__).parent / "fixtures" / "alice_epub2.epub"


@pytest.fixture
def fake():
    f = FakeDrama3()
    yield f
    f.close()


def _tts_cfg(url: str) -> dict:
    t = config.load()["tts"]
    t["adapter"] = "drama3"
    t["drama3"]["endpoint"] = url
    t["drama3"]["backoff_seconds"] = 0.01
    return t


def _ref(dirpath: Path, f0: float, text="Hello there.") -> str:
    sr = 44100
    t = np.arange(sr * 2) / sr
    a = sum(0.2 / k * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 6)).astype(np.float32)
    dirpath.mkdir(parents=True, exist_ok=True)
    audio.write(dirpath / "abc123.wav", a, sr)
    (dirpath / "abc123.txt").write_text(text)
    return "ref:abc123"


def test_settings_override_by_adapter():
    t = config.load()["tts"]
    t["adapter"] = "drama3"
    s = settings(t)
    assert s["endpoint"].endswith(":8091") and s["concurrency"] == 16 and s["params"] == {}
    t["adapter"] = "kokoro"
    assert settings(t)["endpoint"].endswith(":9001")


def test_upload_once_budget_and_errors(fake, tmp_path):
    tts = get_tts(_tts_cfg(fake.url), tmp_path / "refs")
    assert tts.sample_rate == 44100 and tts.max_speakers == 1
    ref = _ref(tmp_path / "refs", 120)
    for text in ("One short line.", "Another line, a little longer than the first."):
        res = tts.synthesize(SynthRequest([Utterance(text, ref)], seed=7))
        assert res.audio.size / 44100 == pytest.approx(0.06 * len(text), abs=0.01)
        assert abs(audio.median_f0(res.audio, 44100) - 120) < 3
    assert fake.uploads == 1                                       # the reference went up once
    body = fake.requests[-1]
    assert body["voice"] == "ab-abc123" and body["seed"] == 7 and body["response_format"] == "flac"
    assert body["max_new_tokens"] == tts.frame_budget("Another line, a little longer than the first.")
    assert tts.frame_budget("x" * 2000) == 4000 and tts.frame_budget("Hi.") == 120
    # a fresh adapter (new run) sees the voice on the server and does not upload again
    tts2 = get_tts(_tts_cfg(fake.url), tmp_path / "refs")
    tts2.synthesize(SynthRequest([Utterance("Again.", ref)], seed=7))
    assert fake.uploads == 1

    with pytest.raises(TTSTruncated):
        tts.synthesize(SynthRequest([Utterance("RUNAWAY text", ref)], seed=7))
    fake.fail_next = 2                                             # 503 twice, then fine
    tts.synthesize(SynthRequest([Utterance("Retry me.", ref)], seed=7))
    fake.fail_next = 10
    with pytest.raises(TTSUnavailable):
        tts.synthesize(SynthRequest([Utterance("Down.", ref)], seed=7))
    fake.fail_next = 0
    with pytest.raises(TTSError, match="Kokoro"):
        tts.synthesize(SynthRequest([Utterance("Wrong spec.", "b|bf_emma:0.70+bf_lily:0.30")], seed=7))


def test_pitch_gender():
    assert pitch_gender(115) == "male" and pitch_gender(210) == "female"
    assert pitch_gender(160) == "ambiguous" and pitch_gender(0) == ""


@pytest.fixture
def book(tmp_path):
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(FIXTURE), "--id", "alice"])
    return Book.get("alice", root)


def _cfg(book, url):
    cfg = config.load(book.dir)
    cfg["tts"] = _tts_cfg(url)
    cfg["llm"]["adapter"] = "mock"
    return cfg


def test_cast_and_synth_with_drama3(fake, book):
    cfg = _cfg(book, fake.url)
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    assert not (book.dir / "cast.yaml").exists()                  # Kokoro's cast file is left alone
    data = yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())
    voices = {"narrator": data["narrator"], **data["characters"]}
    assert set(voices) == {"narrator", "_generic_female", "_generic_male"}   # v3: no voices for v2 cards
    refs = book.dir / "voices" / "refs"
    for cid, e in voices.items():
        assert e["voice"].startswith("ref:")
        sha = e["voice"][4:]
        assert (refs / f"{sha}.wav").exists() and (refs / f"{sha}.txt").exists()
        assert (book.dir / "voices" / "drama3" / f"{cid}.wav").exists()
    f0 = {cid: audio.median_f0(*audio.read(refs / f"{e['voice'][4:]}.wav")) for cid, e in voices.items()}
    assert pitch_gender(f0["_generic_male"]) == "male" and pitch_gender(f0["_generic_female"]) == "female"
    assert 150 <= f0["narrator"] <= 172                           # narrator_gender: neutral (default)
    pitches = sorted(f0.values())
    assert all(np.log(b / a) >= 0.14 for a, b in zip(pitches, pitches[1:]))   # no two voices alike
    con = store.connect(book.dir)
    takes = con.execute("SELECT adapter, wav_path FROM take").fetchall()
    assert takes and {t["adapter"] for t in takes} == {"drama3"}
    assert {r["voice"] for r in fake.requests if "voice" in r} >= {f"ab-{e['voice'][4:]}" for e in voices.values()}
    assert (book.chapters / "002.wav").exists()

    # rerun: nothing designed or generated again
    n = len(fake.requests)
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    assert len(fake.requests) == n


def test_single_voice_drama3_casts_only_the_narrator(fake, book):
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    data = yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())
    assert data["narrator"]["voice"].startswith("ref:") and not data["characters"]
    con = store.connect(book.dir)
    assert con.execute("SELECT COUNT(*) FROM take").fetchone()[0] > 0


def test_truncated_take_retried_with_another_seed(fake, book):
    """A runaway take (fills the frame budget) is retried with the next seed; take.seed records it."""
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    con = store.connect(book.dir)
    uid = con.execute("SELECT u.id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                      "WHERE s.doc_id=2 AND s.speak=1 ORDER BY s.seq LIMIT 1").fetchone()[0]
    con.execute("UPDATE utterance SET tts_text='Runs away the first time.' WHERE id=?", (uid,))
    con.commit()
    fake.runaway_once.add("Runs away the first time.")
    logs = []
    run_stages(book, cfg, until="synth", doc_ids=[2], log=logs.append)
    base = cfg["tts"]["seed"]
    seeds = [r["seed"] for r in fake.requests if r["input"] == "Runs away the first time."]
    assert seeds == [base, base + 1]
    assert any("retrying with seed" in ln for ln in logs)
    assert con.execute("SELECT t.seed FROM take t JOIN selection s ON s.take_id=t.id WHERE s.block_id=?",
                       (uid,)).fetchone()[0] == base + 1


def test_service_down_stops_the_run_and_takes_are_adopted(fake, book):
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    con = store.connect(book.dir)
    n_utt = con.execute("SELECT COUNT(*) FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                        "WHERE s.doc_id=2 AND s.speak=1").fetchone()[0]
    n0 = len(fake.requests)
    fake.fail_next = 10 ** 6                                      # the service answers 503 from now on
    logs = []
    with pytest.raises(SystemExit, match="service down"):
        run_stages(book, cfg, until="synth", doc_ids=[2], log=logs.append)
    assert len(fake.requests) == n0                               # 503s are not recorded as speech requests
    assert any("stopping after the requests in flight" in ln for ln in logs)

    fake.fail_next = 0
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    n1 = len(fake.requests)
    assert con.execute("SELECT COUNT(*) FROM take").fetchone()[0] >= n_utt - 5   # (identical texts share a take)
    # A run killed before saving its rows: the files are still there and are taken back, not regenerated.
    con.execute("DELETE FROM take")
    con.commit()
    logs = []
    run_stages(book, cfg, until="synth", doc_ids=[2], log=logs.append)
    assert len(fake.requests) == n1
    assert any("adopted from takes/" in ln for ln in logs)
    assert con.execute("SELECT COUNT(*) FROM take WHERE seed IS NULL").fetchone()[0] > 0


def test_changing_narrator_gender_redesigns_only_the_narrator(fake, book):
    cfg = _cfg(book, fake.url)
    cfg["cast"]["narrator_gender"] = "female"
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    before = yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())
    cfg["cast"]["narrator_gender"] = "neutral"
    logs = []
    run_stages(book, cfg, until="cast", log=logs.append)
    after = yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())
    assert after["narrator"]["voice"] != before["narrator"]["voice"]
    assert after["characters"] == before["characters"]
    assert any("narrator designed" in ln for ln in logs)
    nf0 = audio.median_f0(*audio.read(book.dir / "voices" / "refs" / f"{after['narrator']['voice'][4:]}.wav"))
    assert 150 <= nf0 <= 172
    # a frozen narrator is kept whatever the setting says
    after["narrator"]["frozen"] = True
    (book.dir / "cast.drama3.yaml").write_text(yaml.safe_dump(after))
    cfg["cast"]["narrator_gender"] = "male"
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    assert yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())["narrator"]["voice"] == after["narrator"]["voice"]


def test_package_without_epubcheck_says_so(fake, book):
    """No epubcheck on the machine: the book is still packaged, the check is skipped and recorded."""
    import shutil as _sh
    if not _sh.which("ffmpeg"):
        pytest.skip("needs ffmpeg")
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    cfg["output"]["epubcheck"] = "off"
    logs = []
    run_stages(book, cfg, until="package", doc_ids=[2], log=logs.append)
    assert any("epubcheck SKIPPED" in ln for ln in logs)
    assert (book.reports / "epubcheck_skipped.txt").exists()
    assert list(book.out.glob("*.epub"))


def test_on_runaway_cut_keeps_the_start_and_flags_it(fake, book):
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    con = store.connect(book.dir)
    uid = con.execute("SELECT u.id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                      "WHERE s.doc_id=2 AND s.speak=1 ORDER BY s.seq LIMIT 1").fetchone()[0]
    con.execute("UPDATE utterance SET tts_text='RUNAWAY ISBN 978-1' WHERE id=?", (uid,))
    con.commit()
    with pytest.raises(SystemExit, match="failed"):                 # default: fail
        run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    cfg["tts"]["drama3"]["on_runaway"] = "cut"
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    dur = con.execute("SELECT t.duration FROM take t JOIN selection s ON s.take_id=t.id WHERE s.block_id=?",
                      (uid,)).fetchone()[0]
    assert dur == pytest.approx(len("RUNAWAY ISBN 978-1") / 9 + 1.0, abs=0.05)
    rep = (book.reports / "synth_flagged.tsv").read_text()
    assert uid in rep
    assert (book.chapters / "002.wav").exists()


def test_request_timeout_follows_the_frame_budget(fake, tmp_path, monkeypatch):
    tts = get_tts(_tts_cfg(fake.url), tmp_path / "refs")
    seen = []
    real = tts._request

    def spy(method, path, body=None, ctype=None, timeout=None):
        seen.append((path, timeout))
        return real(method, path, body, ctype, timeout)
    monkeypatch.setattr(tts, "_request", spy)
    long_text = "word " * 300                                    # 1500 chars: budget 4000 frames (~186 s)
    tts.synthesize(SynthRequest([Utterance(long_text, "zero")], seed=1))
    tts.synthesize(SynthRequest([Utterance("Short.", "zero")], seed=1))
    (p1, t1), (p2, t2) = [x for x in seen if x[0] == "/v1/audio/speech"]
    assert t1 == pytest.approx(tts.frame_budget(long_text) * 3 / 21.5) and t1 > 500
    assert t2 == 300


def test_request_timeout_grows_with_concurrency(fake, tmp_path):
    """2026-11-18, a whole-book run: at 64 streams a stream runs at ~0.2x real time; 3x the budget timed out."""
    cfg = _tts_cfg(fake.url)
    for streams, per_frame in [(16, 3), (32, 32 / 6), (64, 64 / 6)]:
        cfg["drama3"] = {**cfg.get("drama3", {}), "concurrency": streams}
        tts = get_tts(cfg, tmp_path / "refs")
        assert tts.timeout_per_frame == pytest.approx(per_frame / 21.5)


def test_male_voice_is_designed_low(fake, book):
    """2026-11-17: a 128 Hz male voice sounded like a woman; designs stay at or under male_max_pitch."""
    cfg = _cfg(book, fake.url)
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    data = yaml.safe_load((book.dir / "cast.drama3.yaml").read_text())
    ref = data["characters"]["_generic_male"]["voice"][4:]
    f0 = audio.median_f0(*audio.read(book.dir / "voices" / "refs" / f"{ref}.wav"))
    assert f0 <= 115 + 2


def test_brackets_are_not_sent_to_drama3(fake, tmp_path):
    """Drama 3 reads [..] as a tag; bracketed lines ran away on every seed (2026-11-17)."""
    from pipeline.adapters.tts.drama3 import spoken_text
    assert spoken_text("(TO GABRIEL:)") == "TO GABRIEL:"
    assert spoken_text("[Introduced to Disprove Count 2: Knowing]") == "Introduced to Disprove Count 2: Knowing"
    assert spoken_text("He said (quietly) no.") == "He said quietly no."
    assert spoken_text("()") == "()"
    tts = get_tts(_tts_cfg(fake.url), tmp_path / "refs")
    tts.synthesize(SynthRequest([Utterance("(Shakes her.)", "zero")], seed=1))
    assert fake.requests[-1]["input"] == "Shakes her."


def test_regen_flagged_generates_the_cut_lines_again(fake, book):
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    cfg["tts"]["drama3"]["on_runaway"] = "cut"
    run_stages(book, cfg, until="cast", log=lambda *_: None)
    con = store.connect(book.dir)
    uid = con.execute("SELECT u.id FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
                      "WHERE s.doc_id=2 AND s.speak=1 ORDER BY s.seq LIMIT 1").fetchone()[0]
    con.execute("UPDATE utterance SET tts_text='Runs away three times.' WHERE id=?", (uid,))
    con.commit()
    fake.runaway_once.add("Runs away three times.")       # first request runs away ...
    real_once = fake.runaway_once
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    # the first seed ran away, the second did not: nothing flagged, so flag it by hand to exercise regen
    (book.reports).mkdir(parents=True, exist_ok=True)
    (book.reports / "synth_flagged.tsv").write_text(f"utterance\tseed\treason\n{uid}\t1234\ttest\n")
    n = len(fake.requests)
    main(["--books", str(book.dir.parent), "regen", "alice", "--flagged"])
    assert not (book.reports / "synth_flagged.tsv").exists()
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    assert len(fake.requests) == n + 1                       # exactly that line again
    assert real_once is fake.runaway_once


def test_regen_next_seed_moves_one_utterance_to_other_seeds(fake, book):
    """A lone request with the same seed gives the same audio, so a take that is wrong without running away
    (a phrase said twice) needs other seeds: `regen --next-seed` records that in reseed.yaml."""
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    con = store.connect(book.dir)
    uid, key, seed = tuple(con.execute("SELECT t.block_id, t.key, t.seed FROM take t JOIN utterance u ON u.id=t.block_id "
                                       "JOIN sentence s ON s.id=u.sentence_id WHERE s.doc_id=2 LIMIT 1").fetchone())
    base = settings(cfg["tts"])["seed"]
    assert seed == base
    for n in (1, 2):
        main(["--books", str(book.dir.parent), "regen", "alice", uid, "--next-seed"])
        run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
        k2, s2 = tuple(con.execute("SELECT key, seed FROM take WHERE block_id=?", (uid,)).fetchone())
        assert k2 == key and s2 == base + 3 * n                 # the same slot, the next three seeds
    assert f"{uid}: 6" in (book.dir / "reseed.yaml").read_text()


def test_flagged_report_drops_a_line_once_its_new_take_is_whole(fake, book):
    """reports/synth_flagged.tsv lists the takes that are cut now, not every take that ever was."""
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    con = store.connect(book.dir)
    a, b = [r[0] for r in con.execute("SELECT t.block_id FROM take t JOIN utterance u ON u.id=t.block_id "
                                      "JOIN sentence s ON s.id=u.sentence_id WHERE s.doc_id=2 ORDER BY t.id LIMIT 2")]
    book.reports.mkdir(parents=True, exist_ok=True)
    rep = book.reports / "synth_flagged.tsv"
    rep.write_text(f"utterance\tseed\treason\n{a}\t1236\tcut earlier\n{b}\t1236\tcut earlier\n")
    main(["--books", str(book.dir.parent), "regen", "alice", a])          # one of the two is made again, whole
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    assert [ln.split("\t")[0] for ln in rep.read_text().splitlines()[1:]] == [b]
    main(["--books", str(book.dir.parent), "regen", "alice", b])
    run_stages(book, cfg, until="synth", doc_ids=[2], log=lambda *_: None)
    assert not rep.exists()


def test_a_take_made_again_gets_into_the_chapter(fake, book):
    """2026-11-18: three cut lines of a novel were regenerated and the packaged book still had the cut ones:
    a regenerated take keeps its key, and assemble compared keys only."""
    import json
    cfg = _cfg(book, fake.url)
    cfg["multi_voice"] = False
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    tl = book.chapters / "002.json"
    before = json.loads(tl.read_text())["fingerprint"]
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=lambda *_: None)
    assert json.loads(tl.read_text())["fingerprint"] == before            # nothing changed: not rebuilt
    con = store.connect(book.dir)
    uid = con.execute("SELECT t.block_id FROM take t JOIN utterance u ON u.id=t.block_id JOIN sentence s "
                      "ON s.id=u.sentence_id WHERE s.doc_id=2 LIMIT 1").fetchone()[0]
    con.close()
    main(["--books", str(book.dir.parent), "regen", "alice", uid])
    logs = []
    run_stages(book, cfg, until="assemble", doc_ids=[2], log=logs.append)
    assert json.loads(tl.read_text())["fingerprint"] != before
    assert any("1 chapters rebuilt" in str(m) for m in logs)
