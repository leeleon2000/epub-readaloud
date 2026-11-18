"""Two-speaker alternation (docs/DESIGN.md).

In a run of consecutive paragraphs that each hold speech, a new paragraph normally means the other
person speaks. When the quotes with a reliable speaker (speech tag, kin tag, manual: confidence >= 0.8)
name exactly two people and sit on alternating paragraphs, the paragraph position decides who speaks in
the untagged ones.

The model gets these wrong a whole run at a time (Elizabeth and Darcy swapped for six turns), and asking
again repeats the mistake, so this is the one rule that may change the model's answer. Measured on the
whole of Pride and Prejudice: the rule disagreed with the model on 21 quotes and was right on 19; it
agreed on 344, 339 of them right.

A run is used only if all of this holds:
  - at least three consecutive speech paragraphs in the same doc;
  - a paragraph whose narration says the speaker goes on ("her companion added", "he continued") starts
    a new run: the same person may be speaking twice in a row;
  - the reliable quotes name one person per parity, and not the same person on both;
  - the model itself used no third speaker in the run (a third voice breaks the alternation).

Input is always the model's own answer (utterance.model_speaker / model_confidence), so running this
again gives the same result. Manual rows are never changed; they count as reliable.
"""
from __future__ import annotations

import re
from collections import Counter

ANCHOR = 0.8          # a quote this confident fixes who speaks in its paragraph
AGREE = 0.8           # confidence when the rule and the model agree
FIXED = 0.5           # confidence when the rule changed the answer: still worth a listen
FLAGGED = 0.3         # mode "flag": the model's answer is kept but marked
TAG_CONFLICT = 0.5    # below this the model's answer already contradicts a speech tag: never raised
MIN_RUN = 3
NOT_PEOPLE = ("narrator", "unknown")
GOES_ON = re.compile(r"\b(added|continued|resumed|went on|proceeded|pursued)\b", re.I)


def decide(turns: list[dict]) -> list[tuple[int, str]]:
    """turns: speech paragraphs in reading order, each {"doc", "para", "quotes": [{"speaker", "confidence"}],
    "narration": text of the paragraph outside the quotes (optional)}.
    -> [(turn index, expected speaker)] for every turn of a run the rule applies to."""
    out: list[tuple[int, str]] = []
    run: list[int] = []
    for i, t in enumerate(turns):
        if run and (t["doc"] != turns[run[-1]]["doc"] or t["para"] - turns[run[-1]]["para"] != 1
                    or GOES_ON.search(t.get("narration") or "")):
            out += _run(turns, run)
            run = []
        run.append(i)
    out += _run(turns, run)
    return out


def _anchor(t: dict):
    """The reliable speaker of a paragraph; None if there is none; False if two disagree."""
    who = {q["speaker"] for q in t["quotes"] if (q["confidence"] or 0) >= ANCHOR and q["speaker"] not in NOT_PEOPLE}
    return None if not who else next(iter(who)) if len(who) == 1 else False


def _run(turns: list[dict], idx: list[int]) -> list[tuple[int, str]]:
    if len(idx) < MIN_RUN:
        return []
    side: dict[int, set] = {0: set(), 1: set()}
    for k, i in enumerate(idx):
        a = _anchor(turns[i])
        if a is False:
            return []                       # two reliable speakers in one paragraph: not a simple exchange
        if a:
            side[k % 2].add(a)
    if len(side[0]) > 1 or len(side[1]) > 1 or not (side[0] or side[1]) or side[0] == side[1]:
        return []
    who = {k: (next(iter(v)) if v else None) for k, v in side.items()}
    said = [Counter(q["speaker"] for q in turns[i]["quotes"] if q["speaker"] != "narrator") for i in idx]
    for k in (0, 1):                        # only one side has a tag: the other side is whoever the model heard there
        if who[k] is None:
            c = Counter()
            for n, cnt in enumerate(said):
                if n % 2 == k:
                    c.update({s: v for s, v in cnt.items() if s not in NOT_PEOPLE and s != who[1 - k]})
            if not c:
                return []
            who[k] = c.most_common(1)[0][0]
    used = {s for cnt in said for s in cnt if s != "unknown"}
    if not used <= {who[0], who[1]}:
        return []                           # the model heard a third person
    return [(i, who[k % 2]) for k, i in enumerate(idx)]


def apply(con, doc_ids: list[int], mode: str = "fix") -> dict:
    """Run the rule over the story quotes of these docs and write speaker / confidence / source.
    mode: fix (change the answer) | flag (keep it, confidence 0.3) | off (restore the model's answers)."""
    q = ",".join("?" * len(doc_ids))
    # first run on a book attributed before this rule existed: the current answer is the model's
    con.execute("UPDATE utterance SET model_speaker=speaker_id, model_confidence=confidence "
                "WHERE is_quote=1 AND source='model' AND model_speaker IS NULL")
    rows = con.execute(
        f"SELECT u.id, u.quote_id, u.source, u.speaker_id, u.confidence, u.model_speaker, u.model_confidence, "
        f"u.is_quote, u.tts_text, s.doc_id, s.para FROM utterance u JOIN sentence s ON s.id=u.sentence_id "
        f"WHERE s.doc_id IN ({q}) ORDER BY s.doc_id, s.seq, u.seq", doc_ids).fetchall()
    narration: dict[tuple, list[str]] = {}
    for r in rows:
        if not r["is_quote"]:
            narration.setdefault((r["doc_id"], r["para"]), []).append(r["tts_text"])
    turns: list[dict] = []
    quotes: dict[str, dict] = {}
    for r in rows:
        if not r["is_quote"]:
            continue
        manual = r["source"] == "manual"
        spk = r["speaker_id"] if manual else r["model_speaker"]
        if spk is None:
            continue
        if r["quote_id"] not in quotes:
            qd = {"id": r["quote_id"], "speaker": spk, "manual": manual,
                  "confidence": 1.0 if manual else r["model_confidence"]}
            quotes[r["quote_id"]] = qd
            if spk == "narrator":
                continue                    # a quoted phrase the narrator reads: not a turn
            if turns and (turns[-1]["doc"], turns[-1]["para"]) == (r["doc_id"], r["para"]):
                turns[-1]["quotes"].append(qd)
            else:
                turns.append({"doc": r["doc_id"], "para": r["para"], "quotes": [qd],
                              "narration": " ".join(narration.get((r["doc_id"], r["para"]), []))})
    stats = Counter()
    new: dict[str, tuple[str, float, str]] = {}          # quote id -> (speaker, confidence, source)
    if mode != "off":
        for i, expected in decide(turns):
            for qd in turns[i]["quotes"]:
                c = qd["confidence"] or 0
                if qd["manual"] or c >= ANCHOR:
                    continue
                if qd["speaker"] == expected:
                    if c >= TAG_CONFLICT:
                        new[qd["id"]] = (expected, AGREE, "model")
                        stats["agree"] += 1
                elif mode == "fix":
                    new[qd["id"]] = (expected, FIXED, "turns")
                    stats["fixed"] += 1
                else:
                    new[qd["id"]] = (qd["speaker"], FLAGGED, "model")
                    stats["flagged"] += 1
    for qid, qd in quotes.items():
        if qd["manual"]:
            continue
        spk, conf, src = new.get(qid, (qd["speaker"], qd["confidence"], "model"))
        con.execute("UPDATE utterance SET speaker_id=?, confidence=?, source=? WHERE quote_id=? AND source!='manual'",
                    (spk, conf, src, qid))
    con.commit()
    stats["turns"] = len(turns)
    return dict(stats)
