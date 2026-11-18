"""v2 P2 roster (docs/DESIGN.md): name graph, groups, validation, events, stage."""
import json
import sqlite3
from pathlib import Path

from pipeline import config, store
from pipeline.adapters.llm.mock import MockLLM
from pipeline.book import Book
from pipeline.cli import main, run_stages
from pipeline.memory.db import CharacterDB
from pipeline.observe import Observer
from pipeline.stages import roster

PRIDE = Path(__file__).parent / "fixtures" / "pride_epub2.epub"


def P(chunk, *refs, g="unknown"):
    return {"chunk_id": chunk, "refs": [{"text": r} for r in refs], "gender_cue": g, "traits": ""}


PERSONS = [
    P("c1", "Mr. Bennet", g="male"), P("c1", "Mrs. Bennet", g="female"),
    P("c1", "Elizabeth", "Lizzy", g="female"), P("c1", "Jane", "Miss Bennet"),
    P("c2", "Catherine", "Lydia"),                      # wrong P1 claim: different first names
    P("c2", "Lady Lucas"), P("c2", "Charlotte", "Miss Lucas"), P("c2", "Sir William Lucas"),
    P("c3", "Mr. Darcy"), P("c3", "Darcy"), P("c3", "Mrs. Long"), P("c3", "her father"),
]


def test_graph_and_groups():
    nodes, edges, p1c = roster.build_graph(PERSONS)
    assert "her father" not in nodes                                   # not a name
    e = lambda a, b: edges.get(tuple(sorted((a, b))), {}).get("strength")    # noqa: E731
    assert e("elizabeth", "lizzy") == "strong"            # P1 + nickname rule
    assert e("jane", "miss bennet") == "hint"              # one passage, no name rule: shown to the model only
    assert e("catherine", "lydia") is None and p1c[("catherine", "lydia")] == 1
    assert e("mr darcy", "darcy") == "medium" and e("mr bennet", "mrs bennet") == "weak"
    groups = {frozenset(g) for g in roster.name_groups(nodes, edges)}
    assert frozenset({"mr darcy", "darcy"}) in groups and frozenset({"mrs long"}) in groups
    lucas = next(g for g in groups if "lady lucas" in g)
    assert {"miss lucas", "sir william lucas"} <= lucas and "charlotte" not in lucas   # Charlotte = Miss Lucas: one passage only
    assert frozenset({"catherine"}) in groups and frozenset({"lydia"}) in groups


def _db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    return CharacterDB(con)


def test_validation_splits_incompatible_names_and_checks_relations():
    persons = [P("c1", "Mrs. Hurst", "Louisa"), P("c1", "Miss Bingley", "Caroline"), P("c1", "Mr. Hurst"),
               P("c1", "Bingley")]
    nodes, edges, _ = roster.build_graph(persons)
    g = list(nodes)                                    # mrs hurst, louisa, miss bingley, caroline, mr hurst, bingley
    v = {nodes[k].text: f"V{i + 1}" for i, k in enumerate(g)}
    ans = {"individuals": [
        {"key": "I1", "name": "Mrs. Hurst", "variants": [v["Mrs. Hurst"], v["Louisa"], v["Miss Bingley"], v["Caroline"]],
         "gender": "female", "reason": "the model merged the sisters"},
        {"key": "I2", "name": "Mr. Hurst", "variants": [v["Mr. Hurst"]], "gender": "male", "reason": ""},
        {"key": "I3", "name": "Mr. Bingley", "variants": [v["Bingley"]], "gender": "male", "reason": ""}],
        "shared": [], "unsure": [],
        "relations": [   # no marriage claimed: Miss Bingley cannot be Mrs. Hurst
            {"a": "I3", "type": "sibling_of", "b": "I1", "evidence": "Mr. Hurst and his wife"},
            {"a": "I1", "type": "sibling_of", "b": "I3", "evidence": "Mr. Hurst and his wife"},     # duplicate
            {"a": "I2", "type": "parent_of", "b": "I1", "evidence": "not in the book"}]}
    db = _db()
    obs = Observer.null(lambda *_: None)
    created, rep = roster._apply_group(db, g, nodes, ans, "Mr. Hurst and his wife sat down.", obs)
    by_name = {db.card(c).name: sorted(db.card(c).aliases) for c in created}
    assert by_name["Mrs. Hurst"] == ["Louisa", "Mrs. Hurst"]
    assert sorted(by_name["Caroline"] if "Caroline" in by_name else by_name["Miss Bingley"]) == ["Caroline", "Miss Bingley"]
    assert len(rep["splits"]) == 1 and [w["code"] for w in obs.memory["warnings"]] == ["NAME_CONFLICT"]
    # relations are only claims here; memory/relgraph.py decides (tests/test_relgraph.py)
    assert db.con.execute("SELECT COUNT(*) FROM relation").fetchone()[0] == 0 and len(rep["claims"]) == 3
    assert db.card("bingley").gender == "male"     # model name "Mr. Bingley" is not a reference: "Bingley"


def test_name_and_gender_rules():
    nodes, _, _ = roster.build_graph([P("c1", "Lizzy", "Elizabeth Bennet", "Miss Eliza", g="female")])
    keys = list(nodes)
    assert roster._name(keys, nodes, "Elizabeth") == "Elizabeth Bennet"     # model name not a reference
    assert roster._name(keys, nodes, "Miss Eliza") == "Miss Eliza"
    assert roster._gender(keys, nodes, "male") == "female"                   # title (Miss) beats the model


def test_stage_end_to_end_with_mock(tmp_path):
    root = tmp_path / "books"
    main(["--books", str(root), "init", str(PRIDE), "--id", "p"])
    b = Book.get("p", root)
    cfg = config.load(b.dir)
    cfg["tts"]["adapter"] = cfg["llm"]["adapter"] = "mock"
    cfg["character_memory"]["roster"] = "v2"
    cfg["attribute"]["version"] = "v2"                   # these tests exercise the v2 stages; the default is v3
    assert run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, until="attribute") == "ok"
    con = store.connect(b.dir)
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id=2 AND (p1!='done' OR pass1!='done' OR pass2!='done')"
                       ).fetchone()[0] == 0
    ids = {r[0] for r in con.execute("SELECT id FROM character WHERE status != 'merged'")}
    assert "mr_bennet" in ids and "mrs_bennet" in ids
    assert not con.execute("SELECT 1 FROM llm_call WHERE task LIKE 'extract%' OR task LIKE 'resolve%'").fetchone()
    assert (b.dir / "characters.yaml").exists() and (b.dir / "reports" / "roster_report.md").exists()
    cands = [json.loads(r[0]) for r in con.execute("SELECT characters FROM chunk WHERE doc_id=2")]
    assert any("mr_bennet" in c for c in cands)
    # rerun: groups answered from the cache, same roster
    n_calls = con.execute("SELECT COUNT(*) FROM llm_call WHERE task='roster'").fetchone()[0]
    run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, only=["roster"])
    assert con.execute("SELECT COUNT(*) FROM llm_call WHERE task='roster'").fetchone()[0] == n_calls
    assert {r[0] for r in con.execute("SELECT id FROM character WHERE status != 'merged'")} == ids
    # a rebuilt roster makes the attribution out of date
    assert con.execute("SELECT COUNT(*) FROM chunk WHERE doc_id=2 AND pass2 != 'stale'").fetchone()[0] == 0
    # manual edits survive a rebuild
    main(["--books", str(root), "characters", "p", "--set", "mr_bennet", "age=middle-aged"])
    run_stages(b, cfg, doc_ids=[2], log=lambda *_: None, only=["roster"])
    con = store.connect(b.dir)
    assert con.execute("SELECT age FROM character WHERE id='mr_bennet'").fetchone()[0] == "middle-aged"


def test_marriage_name_change_needs_the_spouse_surname():
    persons = [P("c1", "Charlotte Lucas", "Miss Lucas", "Mrs. Collins"), P("c1", "Mr. Collins"), P("c1", "Mrs. Long")]
    nodes, _, _ = roster.build_graph(persons)
    ch = ["charlotte lucas", "miss lucas", "mrs collins"]
    assert len(roster._partition(ch, nodes)) == 2                         # no marriage known: split
    assert len(roster._partition(ch, nodes, {"collins"})) == 1            # married to a Collins: one person
    assert len(roster._partition(ch + ["mrs long"], nodes, {"collins"})) == 2   # but not any Mrs.


def test_model_giving_up_loses_nobody():
    """First v2 run: the model put Jane in "shared" with no holders and Sir William in "unsure"."""
    persons = [P("c1", "Mr. Bennet"), P("c1", "Mrs. Bennet"), P("c1", "Jane"), P("c1", "Miss Bennet"),
               P("c1", "Sir William"), P("c1", "Sir William Lucas"), P("c1", "Longbourn")]
    nodes, _, _ = roster.build_graph(persons)
    g = list(nodes)
    v = {nodes[k].text: f"V{i + 1}" for i, k in enumerate(g)}
    ans = {"individuals": [{"key": "I1", "name": "Mr. Bennet", "variants": [v["Mr. Bennet"]], "gender": "male",
                            "reason": ""}],
           "shared": [{"variant": v["Jane"], "who": ["I7"]}, {"variant": v["Miss Bennet"], "who": ["I1", "I1"]}],
           "unsure": [{"variant": v["Sir William"], "reason": "?"}, {"variant": v["Longbourn"], "reason": "?"}],
           "relations": []}
    db = _db()
    created, rep = roster._apply_group(db, g, nodes, ans, "", Observer.null(lambda *_: None))
    for keys in rep["deferred"]:                       # as run() does once every group is applied
        created.update(roster._shared_or_card(db, keys, nodes, {}, created, rep, Observer.null(lambda *_: None)))
    cards = {db.card(c).name: sorted(db.card(c).aliases) for c in created}
    assert "Jane" in cards and "Mrs. Bennet" in cards and cards["Mr. Bennet"] == ["Mr. Bennet"]
    assert cards["Sir William Lucas"] == ["Sir William", "Sir William Lucas"]     # grouped, not split
    assert "Longbourn" not in cards and "Longbourn" in rep["unsure"]


def test_relation_merges_and_contradictions():
    """5.2: Charles joins Mr. Bingley (one-passage same-person claim + both Miss Bingley's brother in
    different passages); Mrs. Bennet never joins Jane (Elizabeth's mother vs Elizabeth's sister)."""
    persons = [P("c1", "Mr. Bingley"), P("c1", "Mr. Bingley"), P("c2", "Mr. Bingley", "Charles"), P("c3", "Charles"),
               P("c1", "Miss Bingley"), P("c1", "Elizabeth"), P("c1", "Jane"), P("c4", "Jane", "Mrs. Bennet"),
               P("c4", "Mrs. Bennet"), P("c5", "Mrs. Bennet")]
    nodes, edges, _ = roster.build_graph(persons)
    db = _db()
    created = {}
    for cid, name, keys in (("mr_bingley", "Mr. Bingley", ["mr bingley"]), ("charles", "Charles", ["charles"]),
                            ("miss_bingley", "Miss Bingley", ["miss bingley"]), ("elizabeth", "Elizabeth", ["elizabeth"]),
                            ("jane", "Jane", ["jane"]), ("mrs_bennet", "Mrs. Bennet", ["mrs bennet"])):
        db.apply("create", {"id": cid, "name": name, "gender": None})
        db.apply("mention", {"id": cid, "n": 10 if cid != "charles" else 1})
        created[cid] = keys
    S = lambda *ch, strong=False: {"chunks": set(ch), "strong": strong, "modes": {"B"}, "evidence": []}  # noqa: E731
    support = {("miss_bingley", "sibling_of", "mr_bingley"): S("c1", "c3", strong=True),
               ("charles", "sibling_of", "miss_bingley"): S("c2"),
               ("elizabeth", "sibling_of", "jane"): S("c1", "c2", strong=True),
               ("elizabeth", "sibling_of", "mrs_bennet"): S("c4"),
               ("mrs_bennet", "parent_of", "elizabeth"): S("c5", strong=True)}
    accepted = {t: {"strength": "strong"} for t, v in support.items() if v["strong"]}
    merges = roster._relation_merges(db, edges, nodes, created, support, accepted)
    assert [(f, t) for f, t, _ in merges] == [("charles", "mr_bingley")]
    assert db.resolve("charles") == "mr_bingley" and db.resolve("mrs_bennet") == "mrs_bennet"


def test_unplaced_title_surname_becomes_shared():
    persons = [P("c1", "Jane", "Miss Bennet"), P("c2", "Elizabeth Bennet", "Miss Bennet"), P("c3", "Mrs. Long")]
    nodes, edges, _ = roster.build_graph(persons)
    db = _db()
    created = {}
    for cid, name, k in (("jane", "Jane", "jane"), ("elizabeth", "Elizabeth Bennet", "elizabeth bennet"),
                         ("mrs_long", "Mrs. Long", "mrs long")):
        db.apply("create", {"id": cid, "name": name})
        created[cid] = [k]
    rep = {"shared": [], "individuals": []}
    out = roster._shared_or_card(db, ["miss bennet"], nodes, edges, created, rep, Observer.null(lambda *_: None))
    assert sorted(out) == ["elizabeth", "jane"] and rep["shared"][0]["name"] == "Miss Bennet"
    assert not db.exists("miss_bennet")
