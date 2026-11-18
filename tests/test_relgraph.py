"""Relation graph (pipeline/memory/relgraph.py, docs/DESIGN.md)."""
import json
import sqlite3

import yaml

from pipeline import store
from pipeline.memory import relgraph as RG
from pipeline.memory.db import CharacterDB


def _db(cards):
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(store.SCHEMA)
    db = CharacterDB(con)
    for cid, name, g, aliases in cards:
        db.apply("create", {"id": cid, "name": name, "gender": g})
        for a in aliases:
            db.apply("alias", {"id": cid, "alias": a, "strength": "strong"})
    return db


CARDS = [("mr_bingley", "Mr. Bingley", "male", ["Bingley"]), ("mr_darcy", "Mr. Darcy", "male", ["Darcy"]),
         ("mr_hurst", "Mr. Hurst", "male", []), ("mrs_hurst", "Mrs. Hurst", "female", []),
         ("miss_bingley", "Miss Bingley", "female", ["Caroline"]), ("mr_philips", "Mr. Philips", "male", []),
         ("mrs_philips", "Mrs. Philips", "female", []), ("jane", "Jane", "female", []),
         ("elizabeth", "Elizabeth", "female", ["Lizzy"]), ("kitty", "Kitty", "female", []),
         ("mr_bennet", "Mr. Bennet", "male", []), ("mrs_bennet", "Mrs. Bennet", "female", []),
         ("charlotte", "Charlotte Lucas", "female", ["Charlotte", "Miss Lucas"]),
         ("mr_collins", "Mr. Collins", "male", ["Collins"]), ("mrs_collins", "Mrs. Collins", "female", [])]


def test_evidence_kinds():
    db = _db(CARDS)
    c = RG.Cards(db)
    # named: kin word tied to a name; the other one named in the sentence, or the claim's own evidence
    a = RG._attach("Mr. Bingley came; but his friend Mr. Darcy soon drew the attention", "mr_bingley", "mr_darcy", c)
    assert ("mr_darcy", "mr_bingley", "friend", None, "named") in a
    assert RG._attach("But his friend Mr. Darcy soon drew the attention", "mr_bingley", "mr_darcy", c, modes="AD") == []
    a = RG._attach("She had a sister married to a Mr. Philips", "mrs_philips", "mr_philips", c, paired=True)
    assert ("mr_philips", "mrs_philips", "spouse", None, "named") in a
    a = RG._attach("Mrs. Hurst sang with her sister", "mrs_hurst", "miss_bingley", c)
    assert a == [("miss_bingley", "mrs_hurst", "sibling", "female", "possessive")]
    s = "“That will not do for a compliment to Darcy, Caroline,” cried her brother, “because he does not. Do not you, Darcy?”"
    assert not [x for x in RG._attach(s, "miss_bingley", "mr_darcy", c) if x[4] == "named" and x[2] == "sibling"]


def _book(db, texts, persons=()):
    con = db.con
    con.execute("INSERT INTO chunk(id, doc_id, seq_from, seq_to) VALUES ('c1', 1, 0, 99)")
    con.execute("INSERT INTO chunk(id, doc_id, seq_from, seq_to) VALUES ('c2', 2, 0, 99)")
    for i, (chunk, t) in enumerate(texts):
        doc = 1 if chunk == "c1" else 2
        con.execute("INSERT INTO sentence(id, doc_id, seq, para, kind, text, tts_text) VALUES (?,?,?,?,'body',?,?)",
                    (f"s{i}", doc, i, i, t, t))
    for k, (chunk, refs, related) in enumerate(persons):
        con.execute("INSERT INTO p1_person(chunk_id, idx, refs, related) VALUES (?,?,?,?)",
                    (chunk, k, json.dumps([{"text": r, **({"rule_kind": "relational"} if r[0].islower() else {})}
                                           for r in refs]),
                     json.dumps([{"relation": "x", "of_ref": o} for o in related])))
    chunks = con.execute("SELECT * FROM chunk ORDER BY id").fetchall()
    sents = {c["id"]: con.execute("SELECT * FROM sentence WHERE doc_id=? ORDER BY seq",
                                  (1 if c["id"] == "c1" else 2,)).fetchall() for c in chunks}
    return chunks, sents


def test_strong_weak_and_no_stored_derivations():
    db = _db(CARDS)
    chunks, sents = _book(db, [
        ("c1", "Elizabeth and her sister Jane talked."), ("c1", "Elizabeth and her sister Kitty laughed."),
        ("c1", "Mrs. Hurst sang with her sister."), ("c2", "Mrs. Hurst played with her sister.")],
        persons=[("c1", ["Elizabeth"], ["Jane", "Kitty"]), ("c1", ["Mrs. Hurst"], ["Miss Bingley"]),
                 ("c2", ["Mrs. Hurst"], ["Miss Bingley"])])
    rep = RG.build(db, db.con, chunks, sents, [])
    rels = {(r["a"], r["type"], r["b"]): r["strength"] for r in db.con.execute("SELECT * FROM relation")}
    assert rels[("elizabeth", "sibling_of", "jane")] == "strong"                # named
    assert rels[("miss_bingley", "sibling_of", "mrs_hurst")] == "weak"         # possessive only, two passages
    assert ("jane", "sibling_of", "kitty") not in rels                         # derived: computed, not stored
    g = RG.graph_from_db(db.con)
    assert g.lookup("jane", "sister") == {"elizabeth", "kitty"}
    assert g.lookup("mrs_hurst", "sister", weak=False) == set() and g.lookup("mrs_hurst", "sister") == {"miss_bingley"}
    kinds = {r["kind"] for r in db.con.execute("SELECT kind FROM relation_claim")}
    assert kinds == {"named", "possessive"} and rep["accepted"]


def test_whole_graph_check_rejects_contradictions():
    """Kitty cannot become Elizabeth's mother once she is her sister (pride-full); no third parent."""
    db = _db(CARDS)
    c = RG.Cards(db)
    S = lambda n, strong: {"obs": [], "chunks": set(f"c{i}" for i in range(n)), "kinds": set(),  # noqa: E731
                           "strong": strong, "first": None}
    cands = {("elizabeth", "sibling_of", "kitty"): S(5, True), ("mrs_bennet", "parent_of", "elizabeth"): S(9, True),
             ("kitty", "parent_of", "elizabeth"): S(10, False), ("elizabeth", "sibling_of", "mrs_bennet"): S(8, False),
             ("mr_bennet", "parent_of", "kitty"): S(4, True), ("mr_darcy", "parent_of", "elizabeth"): S(3, True),
             ("mr_bennet", "spouse_of", "mrs_bennet"): S(6, True)}
    dec = RG.decide(cands, c)
    acc = set(dec["accepted"])
    assert ("mrs_bennet", "parent_of", "elizabeth") in acc and ("elizabeth", "sibling_of", "kitty") in acc
    dropped = {tuple(x["dropped"]) for x in dec["conflicts"]}
    # Kitty is Elizabeth's sister, so not her mother; Mrs. Bennet is her mother, so not her sister;
    # Mr. Bennet is her (derived) father, so Mr. Darcy cannot be another one
    assert {("kitty", "parent_of", "elizabeth"), ("elizabeth", "sibling_of", "mrs_bennet"),
            ("mr_darcy", "parent_of", "elizabeth")} <= dropped


def test_share_threshold_for_weak():
    db = _db(CARDS)
    c = RG.Cards(db)
    S = lambda n, strong: {"obs": [], "chunks": set(f"c{i}" for i in range(n)), "kinds": set(),  # noqa: E731
                           "strong": strong, "first": None}
    # single-valued roles compete (Elizabeth has one mother); siblings do not (Mary is named less than Jane)
    dec = RG.decide({("mrs_bennet", "parent_of", "elizabeth"): S(20, True),
                     ("charlotte", "parent_of", "elizabeth"): S(2, False),
                     ("elizabeth", "sibling_of", "jane"): S(20, True), ("elizabeth", "sibling_of", "kitty"): S(2, False)}, c)
    assert dec["weak"] == [("charlotte", "parent_of", "elizabeth")]
    assert ("elizabeth", "sibling_of", "kitty") in dec["accepted"]


def test_manual_relations_yaml(tmp_path):
    db = _db(CARDS)
    c = RG.Cards(db)
    (tmp_path / "relations.yaml").write_text(yaml.safe_dump({
        "retract": [{"a": "jane", "type": "sibling_of", "b": "elizabeth"}],
        "assert": [{"a": "mr_bingley", "type": "friend_of", "b": "mr_darcy"}]}))
    manual = RG.load_manual(tmp_path)
    S = {"obs": [], "chunks": {"c1"}, "kinds": {"named"}, "strong": True, "first": None}
    dec = RG.decide({("elizabeth", "sibling_of", "jane"): S}, c, manual)
    assert ("elizabeth", "sibling_of", "jane") not in dec["accepted"]           # tombstone: never learned back
    assert dec["accepted"][("mr_bingley", "friend_of", "mr_darcy")]["source"] == "manual"


def test_married_name_alias_and_merge():
    db = _db(CARDS)
    acc = {("charlotte", "spouse_of", "mr_collins"): {"strength": "strong", "valid_from": "s9"}}
    out = RG.married_names(db, acc)
    assert "Mrs. Collins" in db.card("charlotte").aliases
    assert db.resolve("mrs_collins") == "charlotte" and any(m.get("merged") == "mrs_collins" for m in out)


def test_anomaly():
    db = _db(CARDS)
    c = RG.Cards(db)
    dec = {"accepted": {("jane", "sibling_of", "elizabeth"): {"strength": "strong"}},
           "conflicts": [{"dropped": ["x", "parent_of", "y"], "strength": "strong"}] * 3, "weak": [], "retracted": []}
    assert RG.anomaly(dec, c, set(), False)                                    # 3 strong conflicts for 1 accepted
    dec["conflicts"] = [{"dropped": ["x", "parent_of", "y"], "strength": "weak"}] * 30
    assert RG.anomaly(dec, c, set(), False) == []                              # bounced weak noise is normal
    dec["conflicts"] = []
    assert RG.anomaly(dec, c, set(), False) == []
    assert RG.anomaly(dec, c, {("a", "sibling_of", "b"), ("c", "sibling_of", "d")}, True)    # changed 100%


def test_ab_relations_retract_assert_audit(tmp_path, capsys):
    from pipeline.cli import main
    d = tmp_path / "books" / "b"
    d.mkdir(parents=True)
    (d / "source.epub").touch()
    con = store.connect(d)
    db = CharacterDB(con)
    for cid, name, g, _ in CARDS[:4]:
        db.apply("create", {"id": cid, "name": name, "gender": g})
    db.apply("relation", {"a": "mr_bingley", "type": "friend_of", "b": "mr_darcy", "strength": "strong", "source": "p3"})
    con.execute("INSERT INTO relation_claim(a, type, b, kind, evidence, status) VALUES "
                "('mr_bingley', 'friend_of', 'mr_darcy', 'named', 'his friend Mr. Darcy', 'accepted')")
    con.commit()
    root = str(tmp_path / "books")
    main(["--books", root, "relations", "b", "--retract", "mr_hurst", "spouse_of", "mrs_hurst", "--note", "test"])
    main(["--books", root, "relations", "b", "--assert", "mrs_hurst", "spouse_of", "mr_hurst"])
    data = yaml.safe_load((d / "relations.yaml").read_text())
    assert data["retract"] == [] and data["assert"][0]["a"] == "mr_hurst"     # the later decision wins
    kinds = [r[0] for r in store.connect(d).execute("SELECT type FROM character_event WHERE source='manual'")]
    assert kinds == ["relation_retract", "relation_assert"]
    capsys.readouterr()
    main(["--books", root, "relations", "b", "--audit", "1"])
    assert "his friend Mr. Darcy" in capsys.readouterr().out


def test_married_name_with_first_name():
    db = _db(CARDS + [("lydia", "Lydia", "female", ["Lydia Bennet"]), ("wickham", "Mr. Wickham", "male", ["Wickham"]),
                      ("lydia_wickham", "Lydia Wickham", "female", []), ("mary", "Mary Wickham", "female", [])])
    RG.married_names(db, {("lydia", "spouse_of", "wickham"): {"strength": "strong", "valid_from": None}})
    assert db.resolve("lydia_wickham") == "lydia" and db.resolve("mary") == "mary"     # another first name: no
