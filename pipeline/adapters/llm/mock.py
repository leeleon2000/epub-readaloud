"""Rule-based fake LLM for tests. Reads the structured `payload` the prompt
builders pass alongside the messages; never looks at the prose prompt.

- extract: capitalized name sequences (minus common sentence-starting words)
- resolve: match against roster names/aliases by shared word, else new
- same: never the same person
- attribute: "said/asked/cried/thought <Name>" right after a quote, else the
  previous quote's speaker, else unknown
"""
from __future__ import annotations

import re

STOP = {"The", "A", "An", "I", "It", "He", "She", "They", "We", "You", "Who", "What", "Why", "How",
        "Only", "Oh", "Well", "Yes", "No", "But", "And", "So", "Then", "There", "This", "That", "Chapter",
        "CHAPTER", "When", "If", "In", "On", "At", "Not", "My", "Please", "Come", "Here", "Now"}
NAME = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*")
SAID = re.compile(r"^[\s,.!?—–\-”’\"]*(?:said|asked|cried|thought|replied|shouted|whispered)\s+([A-Z][a-z]+)")
SAID_BEFORE = re.compile(r"([A-Z][a-z]+)\s+(?:said|asked|cried|thought|replied)[^\[]*$")


def _names(text: str) -> list[str]:
    out = []
    for m in NAME.finditer(text):
        words = [w for w in m.group(0).split() if w not in STOP]
        if words:
            n = " ".join(words)
            if n not in out:
                out.append(n)
    return out


class MockLLM:
    name = "mock"

    def __init__(self, cfg: dict | None = None):
        self.stats = {"calls": 0}
        self.obs = None

    def complete_json(self, messages, schema, task="", payload=None, max_tokens=None, truncation_retries=1) -> dict:
        self.stats["calls"] += 1
        p = payload or {}
        out = getattr(self, "_" + p["task"])(p)
        if self.obs is not None:
            self.obs.llm_call(task, "mock", "mock", {"payload_task": p["task"]}, {"mock": True}, 0.0, "stop", 0, "ok")
        return out

    def _classify(self, p):
        t = (p["title"] + " " + p["opening"]).lower()
        if "gutenberg" in t:
            return {"category": "boilerplate", "confidence": 0.9, "reason": "mock"}
        return {"category": "story", "confidence": 0.8, "reason": "mock"}

    def _extract(self, p):
        return {"mentions": [{"text": n, "kind": "name", "traits": ""} for n in _names(p["text"])]}

    def _resolve(self, p):
        roster = []
        for line in p["roster"]:
            cid, _, rest = line.partition(": ")
            names = re.split(r"[(),]|aka", rest)
            roster.append((cid, {w.lower() for n in names for w in n.split() if w.strip()}))
        res = []
        for i, m in enumerate(p["mentions"]):
            words = {w.lower() for w in m["text"].split()}
            hit = next((cid for cid, ws in roster if words & ws), None)
            if hit == "__pending__":
                res.append(_r(i, "anonymous"))           # same new person mentioned twice in this chunk
            elif hit:
                res.append(_r(i, "existing", hit))
            else:
                res.append(_r(i, "new", "", m["text"]))
                roster.append(("__pending__", words))
        return {"resolutions": res, "merges": []}

    def _collect(self, p):
        """One person per distinct capitalized name (titles kept); kinds by rule. No relations."""
        people, seen = [], set()
        for m in re.finditer(r"\b(?:(?:Mr|Mrs|Miss|Lady|Sir)\.?\s+)?[A-Z][a-z]+(?: [A-Z][a-z]+)*", p["text"]):
            t = m.group()
            if t.split()[0] in STOP or t in seen:
                continue
            seen.add(t)
            g = "female" if t.startswith(("Mrs", "Miss", "Lady")) else "male" if t.startswith(("Mr", "Sir")) \
                else "unknown"
            people.append({"refs": [{"text": t, "kind": "name"}], "gender_cue": g, "related": [], "traits": ""})
        return {"people": people[:60], "relations": []}

    def _roster(self, p):
        """Rule-based: variants join an individual they are compatible and linked with (names.py)."""
        from ...memory import names as N
        vs = sorted(p["variants"], key=lambda v: (-len(v["text"]), v["code"]))
        inds: list[dict] = []
        for v in vs:
            home = next((d for d in inds if all(N.compatible(v["text"], t).ok for t in d["texts"])
                         and any(N.link(v["text"], t) and N.link(v["text"], t)[0] != "weak" for t in d["texts"])),
                        None)
            if home is None:
                home = {"key": f"I{len(inds) + 1}", "name": v["text"], "variants": [], "texts": [],
                        "gender": N.parse(v["text"]).gender or "unknown", "reason": "mock"}
                inds.append(home)
            home["variants"].append(v["code"])
            home["texts"].append(v["text"])
        return {"individuals": [{k: d[k] for k in ("key", "name", "variants", "gender", "reason")} for d in inds],
                "shared": [], "relations": [], "unsure": []}

    def _same(self, p):
        return {"same": False, "confidence": 0.9, "reason": "mock"}

    def _attribute(self, p):
        text, ids = p["text"], p["speaker_ids"]
        last = next((s for s in reversed(p.get("last_speakers", [])) if s in ids), None)
        out = []
        for q in p["quote_ids"]:
            m = re.search(rf"\[{q}\](.*?)\[/{q}\]", text, re.S)
            after = text[m.end(): m.end() + 80] if m else ""
            before = text[max(0, m.start() - 80): m.start()] if m else ""
            name = (SAID.search(after) or SAID_BEFORE.search(before))
            spk = None
            if name:
                first = name.group(1).lower()
                names = p.get("names") or {}
                words = {i: ({w.lower() for n in names[i] for w in re.findall(r"[A-Za-z]+", n)}
                             if i in names else set(i.split("_"))) for i in ids}
                spk = next((i for i in ids if first in words[i]), None)
            spk = spk or last or "unknown"
            if spk not in ("narrator", "unknown"):
                last = spk
            out.append({"q": q, "speaker": spk, "style": "neutral", "confidence": 0.9 if name else 0.4})
        return {"quotes": out}


    # ---- v3 (docs/DESIGN.md)
    def _v3_genders(self, p):
        return {"genders": [{"n": it["n"], "gender": "unknown"} for it in p["items"]]}

    def _v3_who(self, p):
        """The nearest listed name before the quote in the passage, else unknown."""
        text, out = p["text"], []
        people = [n for n, _ in p["people"]]
        gender = dict(p["people"])
        for q in p["todo"]:
            m = re.search(rf"\[{q}\]", text)
            before = text[:m.start()] if m else ""
            best, pos = None, -1
            for n in people:
                i = before.rfind(n)
                if i > pos:
                    best, pos = n, i
            spk = best or "unknown"
            g = gender.get(spk, "unknown")
            d = {"q": q}
            if p.get("evidence"):
                d |= {"evidence": best or "", "basis": "tag_near" if best else "none"}
            out.append(d | {"speaker": spk, "gender": g})
        return {"quotes": out}

    def _v3_voice(self, p):
        return {"quotes": [{"q": q, "voice": "unknown"} for q in p["todo"]]}

    def _v3_notes(self, p):
        """One scene with every capitalised two-word name in the passage, gender unknown."""
        names = sorted(set(re.findall(r"\b(?:Mr|Mrs|Miss|Dr)\. [A-Z][a-z]+", p["text"])))[:12]
        return {"narrator": "", "scenes": [{"where": "", "talk": "",
                                            "people": [{"names": [n], "gender": "unknown"} for n in names]}]}

    def _v3_to(self, p):
        return {"quotes": [{"q": q, "to": "unknown"} for q in p["todo"]]}


def _r(i, action, cid="", name=""):
    return {"mention": i, "action": action, "same_as": -1, "character_id": cid, "name": name, "gender": "unknown",
            "age": "unknown", "accent": "", "profile": "", "voice_description": ""}
