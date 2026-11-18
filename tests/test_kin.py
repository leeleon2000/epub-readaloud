"""Relation graph queries (pipeline/memory/kin.py) and gold-relation scoring (pipeline/eval/relations.py)."""
from pathlib import Path

import pytest

from pipeline.eval import pdnc
from pipeline.eval import relations as R
from pipeline.memory.kin import Graph, kin_word

G = {"mr_bennet": "male", "mrs_bennet": "female", "jane": "female", "elizabeth": "female", "kitty": "female",
     "mrs_philips": "female", "mr_philips": "male", "mr_gardiner": "male", "mr_collins": "male",
     "mr_bingley": "male", "miss_bingley": "female", "mrs_hurst": "female", "mr_hurst": "male"}
RELS = [("mr_bennet", "spouse_of", "mrs_bennet"), ("mrs_bennet", "parent_of", "jane"),
        ("jane", "sibling_of", "elizabeth"), ("elizabeth", "sibling_of", "kitty"),
        ("mrs_bennet", "sibling_of", "mrs_philips"), ("mrs_bennet", "sibling_of", "mr_gardiner"),
        ("mr_philips", "spouse_of", "mrs_philips"), ("mr_collins", "cousin_of", "mr_bennet"),
        ("mr_bingley", "sibling_of", "miss_bingley"), ("miss_bingley", "sibling_of", "mrs_hurst"),
        ("mr_hurst", "spouse_of", "mrs_hurst")]


def test_derivations_and_lookups():
    g = Graph(RELS, G)
    # siblings transitive; parents shared by siblings and by the spouse
    assert g.lookup("jane", "sister") == {"elizabeth", "kitty"}
    assert g.lookup("kitty", "father") == {"mr_bennet"} and g.lookup("elizabeth", "mother") == {"mrs_bennet"}
    assert g.lookup("mrs_bennet", "daughter") == {"jane", "elizabeth", "kitty"}
    assert g.lookup("elizabeth", "uncle") == {"mr_philips", "mr_gardiner"}
    assert g.lookup("elizabeth", "aunt") == {"mrs_philips"}
    assert g.lookup("mr_philips", "niece") == {"jane", "elizabeth", "kitty"}
    assert g.lookup("mr_bingley", "brother-in-law") == {"mr_hurst"}
    assert g.lookup("mr_hurst", "sister-in-law") == {"miss_bingley"}
    assert g.lookup("miss_bingley", "brother") == {"mr_bingley"}          # not Mr. Hurst
    assert g.lookup("mr_collins", "cousin") == {"mr_bennet", "jane", "elizabeth", "kitty"}
    assert g.lookup("elizabeth", "cousin") == {"mr_collins"}
    assert g.lookup("elizabeth", "brother") == set()
    assert ("mr_bennet", "parent_of", "kitty") in g.derived
    assert g.has("mrs_hurst", "sibling_of", "mr_bingley") and not g.has("mr_hurst", "sibling_of", "mrs_hurst")


@pytest.mark.parametrize("text,out", [
    ("her eldest sister", ("sibling", "female")), ("His brother-in-law", ("in_law", "male")),
    ("his lady", ("spouse", "female")), ("Lady Lucas", None), ("their uncle", ("aunt_uncle", "male")),
    ("my cousin", ("cousin", None)), ("her sisters", ("sibling", "female")), ("the stranger", None),
])
def test_kin_word(text, out):
    assert kin_word(text) == out


MINI = Path(__file__).parent / "fixtures" / "pdnc_mini"


def test_gold_relations_load_and_score():
    gold, *_ = pdnc.load_gold(MINI, MINI / "overrides.yaml")
    rg = R.load(MINI / "relations.yaml", MINI, gold)
    assert ("Tom", "spouse_of", "Ann", "core", 1) in rg["triples"]
    names = {"tom": "Tom", "ann": "Ann", "bob": "Bob"}
    ours = [("tom", "spouse_of", "ann", "p3"), ("tom", "sibling_of", "ann", "p3"), ("tom", "friend_of", "x", "p3")]
    res = R.score(rg, ours, names, {"Tom", "Ann", "Bob"}, {"Tom": "male", "Ann": "female"}, {}, max_chapter=1)
    assert [r["correct"] for r in res["rows"]] == [True, False] and len(res["unmapped"]) == 1
    assert res["metrics"]["rel_precision_pct"] == 33.3
    assert res["metrics"]["rel_recall_pct"] == 100.0          # friend_of is chapter 2: out of range
    assert res["metrics"]["kin_lookup_pass"] == "1/1"


def test_gold_relations_file_is_checked(tmp_path):
    gold, *_ = pdnc.load_gold(MINI, MINI / "overrides.yaml")
    bad = tmp_path / "r.yaml"
    bad.write_text("relations:\n  - {a: Tom, type: friend_of, b: Nobody, tier: core, chapter: 1, text: not in the book}\n")
    with pytest.raises(SystemExit) as e:
        R.load(bad, MINI, gold)
    assert "Nobody" in str(e.value) and "not in novel_text" in str(e.value)


PRIDE = Path(__file__).resolve().parents[1] / "data" / "pdnc" / "PrideAndPrejudice"


@pytest.mark.skipif(not PRIDE.exists(), reason="PDNC not downloaded (scripts/fetch_pdnc.sh)")
def test_pride_gold_relations_are_consistent():
    """The hand-written gold file passes its own checks, and its kin lookups hold on the gold graph."""
    ov = PRIDE.parents[2] / "data" / "pdnc_overrides"
    gold, *_ = pdnc.load_gold(PRIDE, ov / "PrideAndPrejudice.yaml")
    rg = R.load(ov / "PrideAndPrejudice.relations.yaml", PRIDE, gold)
    gender = {g.name: {"M": "male", "F": "female"}.get(g.gender) for g in gold.values()}
    g = Graph([t[:3] for t in rg["triples"]], gender)
    bad = [(q["anchor"], q["kin"], sorted(g.lookup(q["anchor"], q["kin"])), q["expect"]) for q in rg["lookups"]
           if not R.lookup_ok(g.lookup(q["anchor"], q["kin"]), set(q["expect"]), q.get("later", []))]
    assert bad == []


OTHER_GOLD = sorted(p.name.split(".")[0] for p in (PRIDE.parents[2] / "data" / "pdnc_overrides").glob("*.relations.yaml")
                    if not p.name.startswith("PrideAndPrejudice"))


@pytest.mark.parametrize("book", OTHER_GOLD)
def test_other_gold_relations_are_consistent(book):
    """Every other gold file loads (names, verbatim evidence, chapters) and its lookups hold on its own graph."""
    pdnc_dir = PRIDE.parent / book
    if not pdnc_dir.exists():
        pytest.skip(f"PDNC {book} not downloaded (scripts/fetch_pdnc.sh {book})")
    ov = PRIDE.parents[2] / "data" / "pdnc_overrides"
    gold, *_ = pdnc.load_gold(pdnc_dir, ov / f"{book}.yaml")
    rg = R.load(ov / f"{book}.relations.yaml", pdnc_dir, gold)
    gender = {g.name: {"M": "male", "F": "female"}.get(g.gender) for g in gold.values()}
    g = Graph([t[:3] for t in rg["triples"]], gender)
    bad = [(q["anchor"], q["kin"], sorted(g.lookup(q["anchor"], q["kin"])), q["expect"]) for q in rg["lookups"]
           if not R.lookup_ok(g.lookup(q["anchor"], q["kin"]), set(q["expect"]), q.get("later", []))]
    assert bad == []
    for u in rg["unscored"]:                 # what the types cannot express stays out of the scored graph
        for a in R._as_list(u["a"]):
            for b in R._as_list(u["b"]):
                assert not any(g.has(a, t, b) for t in ("sibling_of", "spouse_of", "parent_of")), (a, u["type"], b)


def test_unscored_pairs_are_neither_right_nor_wrong():
    gold = {"triples": [("Tom", "friend_of", "Dick", "stated", 1)], "lookups": [], "wrong": [],
            "unscored": [{"a": "Monks", "type": "half_sibling_of", "b": "Oliver", "accept": ["sibling_of"]}]}
    ours = [("tom", "friend_of", "dick", "p3"), ("monks", "sibling_of", "oliver", "p3"), ("monks", "spouse_of", "oliver", "p3")]
    name_of = {"tom": "Tom", "dick": "Dick", "monks": "Monks", "oliver": "Oliver"}
    res = R.score(gold, ours, name_of, set(name_of.values()), {}, {}, None)
    assert [r["type"] for r in res["ignored"]] == ["sibling_of"]                 # tolerated: not counted either way
    assert [(r["type"], r["correct"]) for r in res["rows"]] == [("friend_of", True), ("spouse_of", False)]
    assert res["metrics"]["rel_precision_pct"] == 50.0


def test_kinref_resolves_through_the_scene():
    from pipeline.memory import kinref
    g = Graph(RELS, G)
    targets, anchors, kin = kinref.resolve(g, "his lady", ["mr_bennet", "mrs_bennet"], G)
    assert targets == {"mrs_bennet"} and anchors == ["mr_bennet"]
    # "her brother" with three women on the scene: only the Bingley sisters have a brother
    assert kinref.resolve(g, "her brother", ["miss_bingley", "mrs_hurst", "elizabeth"], G)[0] == {"mr_bingley"}
    assert kinref.resolve(g, "her mother", ["mrs_bennet", "elizabeth"], G)[0] == {"mrs_bennet"}
    assert kinref.resolve(g, "Mrs. Bennet's sister", [], G, anchor_of=lambda t: "mrs_bennet")[0] == {"mrs_philips"}
    assert kinref.resolve(g, "my sister", ["elizabeth"], G) is None          # needs the speaker
    assert kinref.resolve(g, "the stranger", ["elizabeth"], G) is None


def test_kin_speech_tags_set_confidence():
    import sqlite3
    from pipeline import store
    from pipeline.attrib_conf import score
    from pipeline.memory.db import CharacterDB
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    db = CharacterDB(con)
    for cid, gender in G.items():
        db.apply("create", {"id": cid, "name": cid, "gender": gender})
    kin = {"graph": Graph(RELS, G), "scene": ["miss_bingley", "mrs_hurst", "elizabeth", "mr_bingley", "mr_hurst"]}
    q = lambda spk, after: {"speaker": spk, "after": after, "before": "", "para": 1}   # noqa: E731
    st = {}
    out = score(db, 1, [q("mr_bingley", "cried her brother."), q("mr_hurst", "cried her brother."),
                        q("miss_bingley", "added her sister."), q("elizabeth", "added her sister.")], kin, st)
    assert out == [0.95, 0.3, 0.8, 0.3]          # one brother; several sisters (Mrs. Hurst / Miss Bingley / Jane...)
    assert st == {"kin_tags": 4, "kin_resolved": 4, "kin_agree": 2, "kin_disagree": 2}


def test_relations_on_candidate_cards():
    from pipeline.stages.attribute import relations_text
    g = Graph(RELS, G)
    assert relations_text(g, "mrs_bennet", "female", {"mr_bennet": "C1", "jane": "C5", "kitty": "C6"}) == \
        "wife of C1; mother of C5, C6"
    assert relations_text(g, "mr_hurst", "male", {"mr_bingley": "C2"}) == "brother-in-law of C2"


def test_weak_relations_count_only_as_themselves():
    g = Graph([("mr_bennet", "spouse_of", "mrs_bennet")], G,
              weak=[("mrs_bennet", "parent_of", "jane"), ("jane", "sibling_of", "elizabeth")])
    assert g.lookup("jane", "mother") == {"mrs_bennet"} and g.lookup("jane", "mother", weak=False) == set()
    assert g.lookup("elizabeth", "mother") == set()              # no "shared parents" through a weak sibling
    assert g.lookup("jane", "father") == set()                   # no "parent's spouse" through a weak parent
    assert Graph([("a", "parent_of", "b"), ("b", "sibling_of", "a")], {}).problems()
