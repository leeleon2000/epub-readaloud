"""pipeline/memory/names.py: name parsing, title compatibility, name-graph link strength
(docs/DESIGN.md)."""
import pytest

from pipeline.memory import names as N


@pytest.mark.parametrize("text,title,given,surname,kind", [
    ("Mr. Bennet", "mr", None, "bennet", "title_surname"),
    ("Mrs. Bennet", "mrs", None, "bennet", "title_surname"),
    ("Miss Lucas", "miss", None, "lucas", "title_surname"),
    ("Lady Lucas", "lady", None, "lucas", "title_surname"),
    ("Lady Catherine", "lady", "catherine", None, "title_given"),
    ("Lady Catherine de Bourgh", "lady", "catherine", "de bourgh", "name"),
    ("Sir William Lucas", "sir", "william", "lucas", "name"),
    ("Sir William", "sir", "william", None, "title_given"),
    ("Elizabeth Bennet", None, "elizabeth", "bennet", "name"),
    ("Miss Eliza Bennet", "miss", "eliza", "bennet", "name"),
    ("Elizabeth", None, "elizabeth", None, "given"),
    ("Lizzy", None, "lizzy", None, "nickname"),
    ("Bingley", None, None, "bingley", "surname"),
    ("Mr. Darcy's", "mr", None, "darcy", "title_surname"),
    ("the late Mr. Darcy", "mr", None, "darcy", "generation"),
    ("young Lucas", None, None, "lucas", "generation"),
    ("aunt Philips", "aunt", None, "philips", "title_surname"),
    ("My dear Mr. Bennet", "mr", None, "bennet", "title_surname"),
    ("my dear", None, None, None, "address"),
    ("sir", None, None, None, "address"),
    ("Mamma", None, None, None, "kinship"),
    ("her father", None, None, None, "relational"),
    ("his wife", None, None, None, "relational"),
    ("the housekeeper", None, None, None, "occupation"),
    ("the Colonel", "colonel", None, None, "title_role"),
    ("her ladyship", None, None, None, "title_role"),
    ("the stranger", None, None, None, "description"),
    ("the Bennets", None, None, "bennet", "group"),
])
def test_parse(text, title, given, surname, kind):
    n = N.parse(text)
    assert (n.title, n.given, n.surname, n.kind) == (title, given, surname, kind)


def test_nicknames_map_to_formal_given_names():
    assert "elizabeth" in N.given_forms("lizzy") and "elizabeth" in N.given_forms("eliza")
    assert "catherine" in N.given_forms("kitty")
    assert N.given_forms("jane") == {"jane", "janey", "jenny"} or "jane" in N.given_forms("jane")
    assert not N.given_forms("lydia") & N.given_forms("jane")


@pytest.mark.parametrize("a,b,ok", [
    # the cases named in docs/DESIGN.md
    ("Miss Lucas", "Lady Lucas", False),
    ("Lizzy", "Elizabeth", True),
    ("Mr. Bennet", "Mrs. Bennet", False),
    # the v1 false merges on Pride
    ("Miss Bingley", "Mrs. Hurst", False),
    ("Mr. Philips", "Mrs. Philips", False),
    ("Jane", "Elizabeth", False),
    ("Lydia", "Kitty", False),
    ("Caroline", "Louisa", False),
    # the same person
    ("Elizabeth Bennet", "Miss Eliza Bennet", True),
    ("Miss Elizabeth", "Lizzy", True),
    ("Mr. Darcy", "Darcy", True),
    ("Bingley", "Mr. Bingley", True),
    ("Sir William", "Sir William Lucas", True),
    ("Charlotte", "Miss Lucas", True),
    ("Kitty", "Catherine Bennet", True),
    ("Lady Catherine", "Lady Catherine de Bourgh", True),
    # different surnames, different given names
    ("Mrs. Long", "Mrs. Bennet", False),
    ("Charles Bingley", "William Collins", False),
    # title gender
    ("Sir William", "Lady Lucas", False),
    ("Colonel Forster", "Mrs. Forster", False),
])
def test_same_person_possible(a, b, ok):
    assert N.compatible(a, b).ok is ok, N.compatible(a, b).reason
    assert N.compatible(b, a).ok is ok


def test_marriage_allows_miss_to_mrs():
    assert not N.compatible("Miss Lucas", "Mrs. Collins").ok
    assert N.compatible("Miss Lucas", "Mrs. Collins", married=True).ok
    assert N.compatible("Charlotte Lucas", "Mrs. Collins", married=True).ok
    assert not N.compatible("Mr. Collins", "Mrs. Collins", married=True).ok      # gender still differs


def test_conflicts_in_one_card():
    """All names on one card must be pairwise compatible: what v1's merged cards looked like."""
    bad = N.conflicts(["Charlotte Lucas", "Miss Lucas", "Charlotte", "Lady Lucas"])
    assert {frozenset((a, b)) for a, b, _ in bad} == {frozenset(("Miss Lucas", "Lady Lucas"))}
    assert N.conflicts(["Elizabeth", "Lizzy", "Eliza", "Miss Elizabeth Bennet", "Miss Eliza"]) == []
    eliz = N.conflicts(["Elizabeth", "Lizzy", "Jane", "Lydia"])
    assert {frozenset((a, b)) for a, b, _ in eliz} >= {frozenset(("Elizabeth", "Jane")), frozenset(("Lizzy", "Lydia"))}
    # non-name references never conflict (they are not looked up)
    assert N.conflicts(["Mrs. Bennet", "my dear", "her mother", "Mamma"]) == []


@pytest.mark.parametrize("a,b,strength,why", [
    ("Elizabeth", "Elizabeth Bennet", "medium", "contains"),
    ("Bingley", "Mr. Bingley", "medium", "contains"),
    ("Lizzy", "Elizabeth", "medium", "nickname"),
    ("Kitty", "Catherine Bennet", "medium", "nickname"),
    ("Mr. Bennet", "Mrs. Bennet", "weak", "surname"),       # same family: grouped, never merged by the rule
    ("Jane Bennet", "Miss Bennet", "weak", "surname"),
    ("Miss Lucas", "Lady Lucas", "weak", "surname"),
    ("Mr. Darcy", "Mr. Bingley", None, None),
    ("Jane", "Lydia", None, None),
])
def test_link(a, b, strength, why):
    assert N.link(a, b) == ((strength, why) if strength else None)


def test_full_name_key():
    assert N.full_name_key("Miss Elizabeth Bennet") == N.full_name_key("Lizzy Bennet") == ("elizabeth", "bennet")
    assert N.full_name_key("Mr. Bennet") is None and N.full_name_key("Jane") is None


def test_lookup_kinds():
    """Only name-like kinds may be looked up directly (docs/DESIGN.md: name/nickname); the rest go to the model."""
    assert N.parse("Elizabeth Bennet").lookup and N.parse("Lizzy").lookup
    assert not N.parse("Miss Bennet").lookup and not N.parse("my dear").lookup
    assert not N.parse("her father").lookup and not N.parse("Bennet").lookup


@pytest.mark.parametrize("text,title,given,surname,kind", [
    ("the Right Honourable Lady Catherine de Bourgh", "lady", "catherine", "de bourgh", "name"),
    ("my little Lizzy", None, "lizzy", None, "nickname"),
    ("poor Jane", None, "jane", None, "given"),
    ("A Young Lucas", None, None, "lucas", "generation"),
    ("the Lucases", None, None, "lucas", "group"),
    ("Mr. & Mrs. Bennet", None, None, "bennet", "group"),
    ("Mr. and Mrs. Hurst", None, None, "hurst", "group"),
])
def test_parse_found_on_pride(text, title, given, surname, kind):
    n = N.parse(text)
    assert (n.title, n.given, n.surname, n.kind) == (title, given, surname, kind)


def test_surname_spelling_variants():
    assert N.compatible("Mr. Denny", "Mr. Denney").ok and N.compatible("Mrs. Philips", "Mrs. Phillips").ok
    assert not N.compatible("Mr. Long", "Mr. Lang").ok           # too short to call it a spelling variant
    assert N.link("Mr. Philips", "Mrs. Phillips") == ("weak", "surname")
