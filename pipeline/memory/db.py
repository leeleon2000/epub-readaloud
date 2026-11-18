"""Character memory: long-term store (character / alias / character_event).

Every change goes through `apply()` and is recorded as an append-only event,
so model and manual edits share one history. Merges never delete: the old
card gets status 'merged' + merged_into, and `resolve()` follows the chain.

The tables are a projection of the events: `replay()` rebuilds them from all
events that are not superseded. Rollback = mark events superseded + replay
(see pipeline/memory/rollback.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

TITLES = {"mr", "mrs", "miss", "ms", "mx", "dr", "doctor", "sir", "lady", "lord", "madam", "madame",
          "master", "mistress", "captain", "capt", "colonel", "col", "professor", "prof", "rev",
          "reverend", "father", "mother", "aunt", "uncle", "king", "queen", "prince", "princess", "duke",
          "duchess", "count", "countess", "st", "saint"}
ARTICLES = {"the", "a", "an", "old", "young", "little"}


def norm(alias: str) -> str:
    """Lowercase, drop possessives/punctuation and leading articles; keep titles (see norm_core)."""
    s = alias.lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    words = s.split()
    while words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(words)


def norm_core(alias: str) -> str:
    """norm() without titles and descriptive adjectives: 'Mrs. Bennet' -> 'bennet'."""
    words = [w for w in norm(alias).split() if w not in TITLES and w not in ARTICLES]
    return " ".join(words) or norm(alias)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", norm(name)).strip("_") or "character"


@dataclass
class Card:
    id: str
    name: str
    aliases: list[str]
    gender: str
    age: str
    accent: str
    profile: str
    voice_description: str
    mentions: int
    status: str

    def compact(self) -> str:
        aka = [a for a in self.aliases if a != self.name]
        return f"{self.id}: {self.name}" + (f" (aka {', '.join(aka[:6])})" if aka else "")

    def full(self) -> str:
        aka = [a for a in self.aliases if a != self.name]
        return (f"- id: {self.id}\n  name: {self.name}\n  aliases: {aka[:10]}\n"
                f"  gender: {self.gender or 'unknown'}, age: {self.age or 'unknown'}"
                f"{', accent: ' + self.accent if self.accent else ''}\n"
                f"  profile: {self.profile or ''}\n  status: {self.status}")


    def coded(self, code: str, relations: str = "") -> str:
        """Card for the attribute prompt under a neutral code (no internal id)."""
        aka = [a for a in self.aliases if a != self.name]
        return (f"- {code}: {self.name}" + (f" ({relations})" if relations else "") + f"\n  aliases: {aka[:10]}\n"
                f"  gender: {self.gender or 'unknown'}, age: {self.age or 'unknown'}\n"
                f"  profile: {self.profile or ''}")


class CharacterDB:
    def __init__(self, con, profile_max_chars: int = 400, run_id: str | None = None):
        self.con = con
        self.profile_max_chars = profile_max_chars
        self.run_id = run_id

    # ------------------------------------------------------------------ reads
    def resolve(self, cid: str) -> str:
        seen = set()
        while cid and cid not in seen:
            seen.add(cid)
            r = self.con.execute("SELECT merged_into FROM character WHERE id=?", (cid,)).fetchone()
            if not r or not r["merged_into"]:
                return cid
            cid = r["merged_into"]
        return cid

    def card(self, cid: str) -> Card | None:
        r = self.con.execute("SELECT * FROM character WHERE id=?", (cid,)).fetchone()
        if not r:
            return None
        aliases = [a["alias"] for a in self.con.execute(
            "SELECT alias FROM alias WHERE character_id=? ORDER BY strength='strong' DESC, alias", (cid,))]
        return Card(r["id"], r["name"], aliases, r["gender"] or "", r["age"] or "", r["accent"] or "",
                    r["profile"] or "", r["voice_description"] or "", r["mentions"], r["status"])

    def is_real(self, cid: str) -> bool:
        """Exists and is a character (not merged away, not marked 'not a character')."""
        r = self.con.execute("SELECT status FROM character WHERE id=?", (cid,)).fetchone()
        return bool(r) and r["status"] in ("active", "provisional")

    def strong_names(self, cid: str) -> set[str]:
        return {r["norm"] for r in self.con.execute(
            "SELECT norm FROM alias WHERE character_id=? AND strength='strong'", (cid,))}

    def constraints(self) -> tuple[set[str], set[frozenset]]:
        """Manual constraints in effect: (not-character ids, never-merge pairs)."""
        notc, never = set(), set()
        for r in self.con.execute("SELECT payload FROM character_event WHERE type='constraint' "
                                  "AND superseded_by IS NULL ORDER BY id"):
            p = json.loads(r["payload"])
            if p["kind"] == "not_character":
                notc.add(p["id"])
            elif p["kind"] == "never_merge":
                never.add(frozenset((p["a"], p["b"])))
        return notc, never

    def active_ids(self) -> list[str]:
        return [r["id"] for r in self.con.execute(
            "SELECT id FROM character WHERE status IN ('active','provisional') ORDER BY mentions DESC, id")]

    def roster(self) -> list[Card]:
        return [self.card(i) for i in self.active_ids()]

    def exists(self, cid: str) -> bool:
        return self.con.execute("SELECT 1 FROM character WHERE id=?", (cid,)).fetchone() is not None

    def new_id(self, name: str, provisional_tag: str | None = None) -> str:
        base = slug(name)
        if provisional_tag:
            base = f"unknown_{base}_{provisional_tag}"
        cand, k = base, 1
        while self.exists(cand):
            k += 1
            cand = f"{base}_{k}"
        return cand

    # ------------------------------------------------------------------ writes (all via events)
    def apply(self, etype: str, payload: dict, source: str = "model", chunk_id: str | None = None) -> None:
        getattr(self, f"_ev_{etype}")(payload)
        self.con.execute("INSERT INTO character_event(type, payload, source, chunk_id, run_id) VALUES (?,?,?,?,?)",
                         (etype, json.dumps(payload, ensure_ascii=False), source, chunk_id, self.run_id))

    def replay(self) -> list[dict]:
        """Rebuild character/alias tables from events in effect. Returns orphaned manual events
        (ones whose character no longer exists after the replay)."""
        for t in ("character", "alias", "alias_fts", "relation"):
            self.con.execute(f"DELETE FROM {t}")
        orphans = []
        rows = self.con.execute("SELECT id, type, payload, source FROM character_event "
                                "WHERE superseded_by IS NULL ORDER BY id").fetchall()
        for r in rows:
            p = json.loads(r["payload"])
            target = p.get("id") or p.get("from")
            if r["type"] not in ("create", "edit", "rejected", "constraint", "relation", "relation_assert",
                                 "relation_retract") and target \
                    and not self.exists(target):
                if r["source"] == "manual":
                    orphans.append({"event_id": r["id"], "type": r["type"], **p})
                continue
            if r["type"] == "constraint" and p["kind"] == "not_character" and not self.exists(p["id"]):
                orphans.append({"event_id": r["id"], "type": "constraint", **p})
                continue
            getattr(self, f"_ev_{r['type']}")(p)
        return orphans

    def _ev_create(self, p: dict) -> None:
        self.con.execute(
            "INSERT INTO character(id, name, gender, age, accent, profile, voice_description, first_seen, "
            "mentions, status) VALUES (?,?,?,?,?,?,?,?,0,?)",
            (p["id"], p["name"], p.get("gender"), p.get("age"), p.get("accent"),
             (p.get("profile") or "")[: self.profile_max_chars], p.get("voice_description"),
             p.get("first_seen"), p.get("status", "active")))
        self._ev_alias({"id": p["id"], "alias": p["name"], "strength": "strong"})

    def _ev_alias(self, p: dict) -> None:
        n = norm(p["alias"])
        if not n:
            return
        cur = self.con.execute("SELECT strength FROM alias WHERE character_id=? AND norm=?", (p["id"], n)).fetchone()
        rank = {"weak": 0, "medium": 1, "strong": 2}
        if cur and rank[cur["strength"]] >= rank[p.get("strength", "strong")]:
            return
        self.con.execute("INSERT OR REPLACE INTO alias(character_id, alias, norm, strength, scope_from, scope_to) "
                         "VALUES (?,?,?,?,?,?)", (p["id"], p["alias"], n, p.get("strength", "strong"),
                                                   p.get("scope_from"), p.get("scope_to")))
        self.con.execute("DELETE FROM alias_fts WHERE character_id=? AND norm=?", (p["id"], n))
        self.con.execute("INSERT INTO alias_fts(norm, character_id) VALUES (?,?)", (n, p["id"]))

    def _ev_relation(self, p: dict) -> None:
        """Basic relation a -type-> b (parent_of, sibling_of, spouse_of, ...), with evidence."""
        self.con.execute("INSERT INTO relation(a, type, b, evidence, source, strength, valid_from) "
                         "VALUES (?,?,?,?,?,?,?)", (p["a"], p["type"], p["b"], p.get("evidence"), p.get("source"),
                                                    p.get("strength", "strong"), p.get("valid_from")))

    def _ev_relation_assert(self, p: dict) -> None:
        """History only: relations.yaml is what the roster stage reads (`readaloud relations --assert`)."""

    def _ev_relation_retract(self, p: dict) -> None:
        """History only (`readaloud relations --retract`); the tombstone lives in relations.yaml."""

    def _ev_profile(self, p: dict) -> None:
        sets, vals = [], []
        for k in ("profile", "voice_description", "gender", "age", "accent", "name", "status"):
            if p.get(k):
                sets.append(f"{k}=?")
                vals.append(p[k][: self.profile_max_chars] if k == "profile" else p[k])
        if sets:
            self.con.execute(f"UPDATE character SET {', '.join(sets)} WHERE id=?", (*vals, p["id"]))

    def _ev_mention(self, p: dict) -> None:
        self.con.execute("UPDATE character SET mentions = mentions + ? WHERE id=?", (p.get("n", 1), p["id"]))

    def _ev_merge(self, p: dict) -> None:
        src, dst = p["from"], self.resolve(p["into"])
        if src == dst or not self.exists(dst):
            return
        self.con.execute("UPDATE character SET status='merged', merged_into=? WHERE id=?", (dst, src))
        m = self.con.execute("SELECT mentions FROM character WHERE id=?", (src,)).fetchone()
        self.con.execute("UPDATE character SET mentions = mentions + ? WHERE id=?", (m["mentions"] if m else 0, dst))
        for a in self.con.execute("SELECT alias, strength FROM alias WHERE character_id=?", (src,)).fetchall():
            self._ev_alias({"id": dst, "alias": a["alias"], "strength": a["strength"]})

    def _ev_split(self, p: dict) -> None:
        """Undo a merge: the card becomes active again (its own aliases were never deleted)."""
        self.con.execute("UPDATE character SET status='active', merged_into=NULL WHERE id=?", (p["id"],))

    def _ev_edit(self, p: dict) -> None:
        """Record-only event (e.g. a merge-scan verdict kept for review)."""

    def _ev_rejected(self, p: dict) -> None:
        """Record-only: a model proposal the validation layer refused (payload has `reason`)."""

    def _ev_constraint(self, p: dict) -> None:
        """Manual constraint. not_character hides the card; never_merge is checked by validation."""
        if p["kind"] == "not_character":
            self.con.execute("UPDATE character SET status='not_character' WHERE id=?", (p["id"],))
