"""Validation layer between model output and CharacterDB.apply() (docs/DESIGN.md).

Refused proposals are recorded as `rejected` events (with a reason) and never
reach the character tables. Conflicts that need a human are raised as
warnings and the existing value is kept.
"""
from __future__ import annotations

import re
from collections import Counter

from .db import ARTICLES, TITLES, CharacterDB, norm
from .prompts import AGES

_WS = re.compile(r"\s+")


def _flat(t: str) -> str:
    return _WS.sub(" ", t.replace("’", "'").replace("‘", "'").lower())


PLURAL_PEOPLE = re.compile(
    r"\b(?:daughters|sons|sisters|brothers|girls|boys|children|parents|cousins|nieces|nephews|aunts|uncles|"
    r"friends|ladies|gentlemen|officers|servants|neighbours|neighbors|families|relations|guests|women|men)\b", re.I)


def is_proper_name(name: str) -> bool:
    """Has a capitalised word that is not a title or article ('Mrs. Bennet' yes, 'his lady' / 'the Queen' no)."""
    for w in re.findall(r"[A-Za-z’']+", name or ""):
        if w[0].isupper() and w.lower().strip("'’") not in TITLES | ARTICLES | {"his", "her", "their", "my", "our"}:
            return True
    return False


class Validator:
    def __init__(self, db: CharacterDB, obs, chunk_id: str | None, chunk_text: str, monitor: dict):
        self.db, self.obs, self.chunk_id = db, obs, chunk_id
        self.text = _flat(chunk_text)
        self.not_char, self.never = db.constraints()
        self.monitor = monitor
        self.rejects: Counter = Counter()

    # ------------------------------------------------------------------ helpers
    def reject(self, reason: str, code: str | None = None, **payload) -> None:
        self.rejects[reason] += 1
        self.db.apply("rejected", {"reason": reason, **payload}, chunk_id=self.chunk_id)
        if code:
            self.obs.warn(code, "info", self.chunk_id, f"{reason}: {payload}", **payload)

    def grounded(self, *names: str) -> bool:
        return any(n and _flat(n) in self.text for n in names)

    def finish_chunk(self) -> None:
        n = self.rejects["not_grounded"]
        if n > self.monitor.get("grounding_reject_warn", 3):
            self.obs.warn("GROUNDING_REJECT", "warn", self.chunk_id,
                          f"{n} mentions were not found in the chunk text", count=n)

    # ------------------------------------------------------------------ resolutions
    def check_resolution(self, res: dict, mention: dict) -> dict | None:
        """Return the (possibly adjusted) resolution to apply, or None if refused."""
        kind, action = mention.get("kind"), res["action"]
        if kind == "relational" and PLURAL_PEOPLE.search(mention["text"]):
            kind = "group"            # "your daughters", "his sisters": several people
        if kind == "group":
            self.reject("group", mention=mention["text"])
            return None
        if action == "anonymous" or kind == "generic":
            return None
        if action == "existing":
            cid = self.db.resolve(res["character_id"])
            if not self.db.exists(cid):
                self.reject("unknown_id", "UNKNOWN_ID", mention=mention["text"], character_id=res["character_id"])
                return None
            if cid in self.not_char or not self.db.is_real(cid):
                self.reject("not_character", mention=mention["text"], character_id=cid)
                return None
            return {**res, "character_id": cid}
        # new / provisional
        if not self.grounded(mention["text"], res.get("name", "")):
            self.reject("not_grounded", "GROUNDING_REJECT", mention=mention["text"], name=res.get("name", ""))
            return None
        # A relational mention ("his lady", "Charlotte Lucas's papa") only makes a full character when the
        # model names the person and that name is in the text; otherwise it stays provisional.
        if kind == "relational" and action == "new":
            name = res.get("name", "")
            if not (is_proper_name(name) and self.grounded(name) and norm(name) != norm(mention["text"])):
                res = {**res, "action": "provisional"}
        n = norm(res.get("name") or mention["text"])
        for cid in self.not_char:
            if n and n in {r["norm"] for r in self.db.con.execute("SELECT norm FROM alias WHERE character_id=?", (cid,))}:
                self.reject("not_character", mention=mention["text"], character_id=cid)
                return None
        return res

    def alias_ok(self, cid: str, alias: str, strength: str) -> bool:
        """A strong/medium alias may belong to one live character only."""
        if strength == "weak":
            return True
        n = norm(alias)
        rows = self.db.con.execute(
            "SELECT a.character_id FROM alias a JOIN character c ON c.id=a.character_id "
            "WHERE a.norm=? AND a.strength IN ('strong','medium') AND c.status IN ('active','provisional') "
            "AND a.character_id != ?", (n, cid)).fetchall()
        if rows:
            other = rows[0]["character_id"]
            self.obs.warn("ALIAS_CONFLICT", "warn", self.chunk_id,
                          f"'{alias}' already belongs to {other}; not added to {cid}",
                          alias=alias, character_id=cid, owner=other)
            return False
        return True

    def attrs(self, cid: str, proposed: dict) -> dict:
        """Filter gender/age updates: fill unknowns, keep existing values on conflict (ATTR_CONFLICT)."""
        card = self.db.card(cid)
        out = {}
        for k in ("gender", "age"):
            new = proposed.get(k)
            old = getattr(card, k) if card else ""
            if not new or new == "unknown" or new == old:
                continue
            if old in ("", "unknown"):
                out[k] = new
                continue
            if k == "age" and old in AGES and new in AGES and abs(AGES.index(old) - AGES.index(new)) < 2:
                continue   # neighbouring age bands: noise, keep the old one silently
            self.obs.warn("ATTR_CONFLICT", "warn", self.chunk_id, f"{cid}.{k}: keeps '{old}', model said '{new}'",
                          character_id=cid, field=k, old=old, new=new)
        return out

    # ------------------------------------------------------------------ merges
    def merge_ok(self, a: str, b: str, confidence: float | None, reason: str) -> bool:
        """Auto-merge only if allowed; otherwise raise MERGE_REVIEW and keep both."""
        if frozenset((a, b)) in self.never or a in self.not_char or b in self.not_char:
            self.reject("never_merge", **{"from": a, "into": b})
            return False
        na, nb = self.db.strong_names(a), self.db.strong_names(b)
        why = None
        if na and nb and not (na & nb):
            why = "both cards have their own real names"
        elif confidence is not None and confidence < 0.8:
            why = f"low confidence {confidence:.2f}"
        if why:
            self.obs.warn("MERGE_REVIEW", "warn", self.chunk_id, f"{a} ? {b}: {why} ({reason})",
                          pair=[a, b], confidence=confidence, reason=reason, why=why)
            return False
        return True
