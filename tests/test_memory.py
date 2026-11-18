import sqlite3

import pytest

from pipeline.memory.db import CharacterDB, norm, norm_core
from pipeline.memory.retrieve import candidates
from pipeline.store import SCHEMA


@pytest.fixture
def db():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    d = CharacterDB(con)
    for cid, name, aliases in [
        ("elizabeth_bennet", "Elizabeth Bennet", ["Elizabeth", "Lizzy", "Eliza"]),
        ("jane_bennet", "Jane Bennet", ["Jane", "Miss Bennet"]),
        ("mr_bennet", "Mr. Bennet", []),
        ("mrs_bennet", "Mrs. Bennet", []),
        ("william_collins", "William Collins", ["Mr. Collins"]),
    ]:
        d.apply("create", {"id": cid, "name": name})
        for a in aliases:
            d.apply("alias", {"id": cid, "alias": a, "strength": "strong"})
    d.apply("create", {"id": "fisherman", "name": "the old fisherman"})
    d.apply("alias", {"id": "fisherman", "alias": "the fisherman", "strength": "weak",
                      "scope_from": 3, "scope_to": 3})
    return d


def ids(c):
    return [cid for cid, _ in c]


def test_norm():
    assert norm("Mrs. Bennet's") == "mrs bennet"
    assert norm_core("Mrs. Bennet") == "bennet"
    assert norm("the White Rabbit") == "white rabbit"


def test_exact_and_title(db):
    assert ids(candidates(db, "Miss Bennet", 1, []))[0] == "jane_bennet"
    assert ids(candidates(db, "MR. BENNET", 1, []))[0] == "mr_bennet"


def test_surname_recalls_whole_family(db):
    got = set(ids(candidates(db, "Bennet", 1, [])))
    assert {"elizabeth_bennet", "jane_bennet", "mr_bennet", "mrs_bennet"} <= got


def test_nickname_dictionary(db):
    assert "william_collins" in ids(candidates(db, "Bill", 1, []))
    assert "elizabeth_bennet" in ids(candidates(db, "Beth", 1, []))


def test_typo_fuzzy(db):
    assert "william_collins" in ids(candidates(db, "Colins", 1, []))


def test_weak_alias_scoped_to_its_doc(db):
    assert "fisherman" in ids(candidates(db, "the fisherman", 3, []))
    # outside its doc only the strong name can match, not the weak alias
    assert candidates(db, "the fisherman", 9, []) == [] or \
        dict(candidates(db, "the fisherman", 9, [])).get("fisherman", 0) < 1.0


def test_recent_characters_offered_for_descriptions(db):
    assert ids(candidates(db, "her sister", 1, ["jane_bennet"])) == ["jane_bennet"]


def test_merge_redirects_and_keeps_history(db):
    db.apply("create", {"id": "stranger", "name": "the stranger"})
    db.apply("merge", {"from": "stranger", "into": "william_collins"})
    assert db.resolve("stranger") == "william_collins"
    assert "william_collins" in ids(candidates(db, "the stranger", 1, []))
    db.apply("split", {"id": "stranger"})
    assert db.resolve("stranger") == "stranger"
    n = db.con.execute("SELECT COUNT(*) FROM character_event").fetchone()[0]
    assert n > 10
