"""Gold relations for evaluation (docs/DESIGN.md): load, check, expand, score.

The gold file (data/pdnc_overrides/<Book>.relations.yaml) is hand-written and used only here.
Loading checks it the same way it was written: every name is a PDNC character (after overrides),
every evidence text occurs verbatim in novel_text.txt, and the first one in the chapter given.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import yaml

from ..memory.kin import Graph
from .pdnc import PDNC_HEAD, ChapterCounter, roman

TIER_RANK = {"core": 0, "stated": 1, "inferred": 2}
TYPE_ALIASES = {"patron_of": "employs"}     # PLAN has no patron type; employs counts as right


def _flat(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def load(path: Path, pdnc_dir: Path, gold_chars: dict) -> dict:
    """-> {"triples": [(a, type, b, tier, chapter)], "lookups": [...], "wrong": [...]}; SystemExit on problems."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    text = (Path(pdnc_dir) / "novel_text.txt").read_text()
    # Chapter of a position in the flattened text. Headings are found the way pdnc.load_gold finds them
    # ("CHAPTER I." on a line, or "CHAPTER" with the number on the next line), then each chapter is
    # flattened on its own so the offsets line up.
    cc = ChapterCounter()
    marks = [(m.start(), cc.see(roman(m.group(1) or m.group(2)))) for m in PDNC_HEAD.finditer(text)]
    bounds = [0] + [o for o, _ in marks] + [len(text)]
    parts, heads, pos = [], [], 0
    for (a, b), n in zip(zip(bounds, bounds[1:]), [0] + [n for _, n in marks]):
        seg = _flat(text[a:b])
        heads.append((pos, n))
        parts.append(seg)
        pos += len(seg) + 1
    flat = " ".join(parts)
    problems: list[str] = []

    def chapters_of(t: str) -> set[int]:
        out, i = set(), flat.find(_flat(t))
        while i != -1:
            out.add(max((n for o, n in heads if o <= i), default=0))
            i = flat.find(_flat(t), i + 1)
        return out

    def check(where: str, names: list[str], texts, chapter):
        for n in names:
            if n not in gold_chars:
                problems.append(f"{where}: {n!r} is not a PDNC character")
        texts = _as_list(texts or [])
        for k, t in enumerate(texts):
            chs = chapters_of(t)
            if not chs:
                problems.append(f"{where}: text not in novel_text: {t[:60]!r}")
            elif k == 0 and chapter is not None and chapter not in chs:
                problems.append(f"{where}: first text is in chapter {sorted(chs)}, not {chapter}: {t[:60]!r}")

    triples: list[tuple] = []

    def add(a, t, b, tier, chapter):
        t = TYPE_ALIASES.get(t, t)
        triples.append((a, t, b, tier, chapter))

    def weaker(x, y):
        return x if TIER_RANK[x["tier"]] >= TIER_RANK[y["tier"]] else y

    for f in data.get("families", []):
        sp = f.get("spouses", [])
        check(f"family {f['id']}", sp, f.get("text"), f.get("chapter"))
        if len(sp) == 2:
            add(sp[0], "spouse_of", sp[1], f["tier"], f["chapter"])
        kids = f.get("children", [])
        for c in kids:
            check(f"family {f['id']} child {c['name']}", [c["name"]], c.get("text"), c.get("chapter"))
            for p in sp:
                add(p, "parent_of", c["name"], c["tier"], c["chapter"])
        for i, x in enumerate(kids):
            for y in kids[i + 1:]:
                add(x["name"], "sibling_of", y["name"], weaker(x, y)["tier"], max(x["chapter"], y["chapter"]))
    for s in data.get("siblings", []):
        ms = s.get("members", [])
        for m in ms:
            check(f"siblings {s['id']} {m['name']}", [m["name"]], m.get("text"), m.get("chapter"))
        for i, x in enumerate(ms):
            for y in ms[i + 1:]:
                add(x["name"], "sibling_of", y["name"], weaker(x, y)["tier"], max(x["chapter"], y["chapter"]))
    for r in data.get("relations", []):
        a_s, b_s = _as_list(r["a"]), _as_list(r["b"])
        check(f"relation {r['type']}", a_s + b_s, r.get("text"), r.get("chapter"))
        for a in a_s:
            for b in b_s:
                add(a, r["type"], b, r["tier"], r["chapter"])
                for alt in r.get("also", []):
                    add(a, alt, b, "inferred", r["chapter"])
    lookups = data.get("kin_lookups", [])
    for q in lookups:
        check(f"lookup {q['anchor']} {q['kin']}", [q["anchor"], *q["expect"], *q.get("later", [])],
              q.get("text"), q.get("chapter"))
    wrong = data.get("wrong", [])
    for w in wrong:
        check(f"wrong {w['type']}", _as_list(w["a"]) + _as_list(w["b"]), None, None)
    # Relationships the relation types cannot express (half-siblings, lovers, engagements, a gang). They are
    # checked like the rest but not scored; `accept` lists the stored types that are neither right nor wrong
    # for that pair (a half-brother written as sibling_of).
    unscored = data.get("unscored", [])
    for u in unscored:
        check(f"unscored {u['type']}", _as_list(u["a"]) + _as_list(u["b"]), u.get("text"), u.get("chapter"))
    if data.get("narrator"):
        check("narrator", [data["narrator"]], None, None)
    if problems:
        raise SystemExit(f"{path}: {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return {"triples": triples, "lookups": lookups, "wrong": wrong, "unscored": unscored,
            "narrator": data.get("narrator"), "file": str(path)}


def lookup_ok(got: set, expect: set, later=()) -> bool:
    """Everyone expected is there; anyone extra is someone who only becomes right later in the book."""
    return set(expect) <= set(got) <= set(expect) | set(later)


def score(gold: dict, ours: list[tuple[str, str, str, str]], name_of: dict, present: set[str],
          gender_gold: dict, gender_ours: dict, max_chapter: int | None) -> dict:
    """ours: [(a_id, type, b_id, source)]; name_of: card id -> PDNC name (None if unmapped);
    present: PDNC names that have a card. Returns metrics and detail lists."""
    G = Graph([(a, t, b) for a, t, b, *_ in gold["triples"]], gender_gold)
    named, unmapped = [], []
    for a, t, b, src in ours:
        na, nb = name_of.get(a), name_of.get(b)
        if na and nb and na != nb:
            named.append((na, t, nb, src))
        else:
            unmapped.append({"a": a, "type": t, "b": b, "source": src})
    wrong_named = {(w_a, w["type"], w_b) for w in gold["wrong"] for w_a in _as_list(w["a"]) for w_b in _as_list(w["b"])}
    tolerated = {(frozenset((a, b)), t) for u in gold.get("unscored", []) for t in u.get("accept", [])
                 for a in _as_list(u["a"]) for b in _as_list(u["b"])}
    ignored = [{"a": a, "type": t, "b": b, "source": src} for a, t, b, src in named
               if (frozenset((a, b)), t) in tolerated]
    named = [x for x in named if (frozenset((x[0], x[2])), x[1]) not in tolerated]
    rows = []
    for a, t, b, src in named:
        ok = G.has(a, t, b) and (a, t, b) not in wrong_named and (b, t, a) not in wrong_named
        rows.append({"a": a, "type": t, "b": b, "source": src, "correct": ok})
    O = Graph([(a, t, b) for a, t, b, src in named if not src.endswith(":weak")], gender_ours,
              weak=[(a, t, b) for a, t, b, src in named if src.endswith(":weak")])
    want = [x for x in gold["triples"] if x[3] in ("core", "stated")
            and (max_chapter is None or x[4] <= max_chapter) and x[0] in present and x[2] in present]
    want = list(dict.fromkeys(want))
    found = [x for x in want if O.has(x[0], x[1], x[2])]
    missed = [{"a": a, "type": t, "b": b, "tier": tier, "chapter": ch} for a, t, b, tier, ch in want
              if (a, t, b, tier, ch) not in found]
    no_card = sorted({n for x in gold["triples"] if x[3] in ("core", "stated")
                      and (max_chapter is None or x[4] <= max_chapter) for n in (x[0], x[2]) if n not in present})
    lk = []
    for q in gold["lookups"]:
        if max_chapter is not None and q["chapter"] > max_chapter + 3:      # doc 4 runs past chapter 13
            continue
        if q["anchor"] not in present:
            lk.append({**q, "got": None, "pass": False, "why": "anchor has no card"})
            continue
        expect = {e for e in q["expect"] if e in present}
        got = O.lookup(q["anchor"], q["kin"])
        lk.append({"anchor": q["anchor"], "kin": q["kin"], "expect": sorted(expect), "got": sorted(got),
                   "pass": lookup_ok(got, expect, q.get("later", []))})

    def pct(x, n):
        return round(100 * x / n, 1) if n else None
    n_core = sum(1 for x in want if x[3] == "core")
    strong_rows = [r for r in rows if not r["source"].endswith(":weak")]
    m = {"rel_written": len(ours),
         "rel_precision_strong_pct": pct(sum(r["correct"] for r in strong_rows), len(strong_rows)), "rel_precision_pct": pct(sum(r["correct"] for r in rows) + 0, len(rows) + len(unmapped)),
         "rel_recall_pct": pct(len(found), len(want)),
         "rel_core_recall_pct": pct(sum(1 for x in found if x[3] == "core"), n_core),
         "kin_lookup_pass": f"{sum(q['pass'] for q in lk)}/{len(lk)}"}
    by_source = Counter(r["source"] for r in rows)
    return {"metrics": m, "rows": rows, "unmapped": unmapped, "ignored": ignored, "missed": missed, "no_card": no_card,
            "lookups": lk, "by_source": dict(by_source),
            "counts": {"want": len(want), "found": len(found), "core": n_core}}
