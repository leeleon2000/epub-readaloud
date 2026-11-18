"""Candidate retrieval for a mention: recall over precision; the LLM decides.

Signals, merged by best score per character:
  exact normalized alias (1.0) / title-stripped core match (0.95)
  nickname dictionary (0.9)
  FTS5 trigram on aliases, re-ranked with rapidfuzz (score/100)
  token overlap with any alias, e.g. surname only (0.8)
  recently seen characters (0.5, so description-style mentions still get candidates)
Weak aliases only count inside their doc scope.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from ..config import REPO
from .db import CharacterDB, norm, norm_core


@lru_cache(maxsize=1)
def nicknames() -> dict[str, set[str]]:
    """name -> set of names it may stand for (both directions)."""
    data = yaml.safe_load((REPO / "data" / "nicknames.yaml").read_text()) or {}
    out: dict[str, set[str]] = {}
    for formal, nicks in data.items():
        group = {formal, *nicks}
        for n in group:
            out.setdefault(n, set()).update(group - {n})
    return out


def _alias_rows(db: CharacterDB, doc_id: int | None):
    rows = db.con.execute(
        "SELECT a.character_id, a.alias, a.norm, a.strength, a.scope_from, a.scope_to FROM alias a "
        "JOIN character c ON c.id = a.character_id WHERE c.status != 'merged'").fetchall()
    for r in rows:
        if r["strength"] == "weak" and doc_id is not None and r["scope_from"] is not None:
            if not (r["scope_from"] <= doc_id <= (r["scope_to"] if r["scope_to"] is not None else r["scope_from"])):
                continue
        yield r


def candidates(db: CharacterDB, mention: str, doc_id: int | None, recent: list[str],
               limit: int = 20) -> list[tuple[str, float]]:
    m, mc = norm(mention), norm_core(mention)
    if not m:
        return []
    scores: dict[str, float] = {}

    def bump(cid: str, s: float):
        cid = db.resolve(cid)
        if s > scores.get(cid, 0):
            scores[cid] = s

    nick = nicknames()
    m_tokens = set(mc.split())
    for r in _alias_rows(db, doc_id):
        an, ac = r["norm"], norm_core(r["alias"])
        if an == m:
            bump(r["character_id"], 1.0)
        elif ac == mc:
            bump(r["character_id"], 0.95)
        elif m_tokens & nick.get(ac, set()) or nick.get(mc, set()) & set(ac.split()):
            bump(r["character_id"], 0.9)
        elif m_tokens and m_tokens & set(ac.split()):
            bump(r["character_id"], 0.8)
        else:
            s = fuzz.WRatio(mc, ac) / 100
            if s >= 0.75:
                bump(r["character_id"], s * 0.9)

    if len(mc) >= 3:
        try:
            q = '"' + mc.replace('"', "") + '"'
            for r in db.con.execute("SELECT character_id, norm FROM alias_fts WHERE alias_fts MATCH ? LIMIT 50", (q,)):
                bump(r["character_id"], max(0.6, fuzz.WRatio(mc, r["norm"]) / 100 * 0.9))
        except Exception:
            pass   # FTS syntax edge cases: other signals still apply

    for cid in recent:
        if db.exists(cid):
            bump(cid, 0.5)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:limit]
