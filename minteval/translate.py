"""Back-translation: reference program -> colloquial English trading instruction q_s."""
from __future__ import annotations

import json
import re

from .assembler import describe
from .primitives import REGISTRY

NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 8: "eight",
             10: "ten", 12: "twelve", 16: "sixteen", 20: "twenty", 24: "twenty-four", 32: "thirty-two",
             48: "forty-eight", 64: "sixty-four", 96: "ninety-six"}

TRANSLATOR_SYS = ("You are a veteran crypto futures trader who explains strategies to a junior quant "
                  "the way traders talk on a desk chat. You never write code.")

TRANSLATOR_USER = '''Rewrite the following strategy as ONE short message a trader would send to a quant
developer who will code it. The developer works under these desk conventions, so you do NOT need to
restate them (but must not contradict them): entry price = close of the signal bar; "ATR" = 14-period
ATR, and distances in ATR use the ATR at entry; stops/targets are live from the signal bar on; profit
conditions are judged on bar closes; "above/below/breaks/crosses" are strict comparisons and
"at least/at most/within" inclusive; breakouts ("closes above the prior N-bar high", "breaks the box")
are level conditions checked every bar while only "crosses" is an event; volatility sizing is phrased
"risk r% of equity per k ATR" (fraction = r x close / (k x ATR)); only enter when flat; an opposite signal just closes the trade; bars
are 15-minute candles.

STRATEGY (precise spec):
{desc}

REFERENCE IMPLEMENTATION (for your understanding only; never mention code):
```python
{src}
```

HARD REQUIREMENTS
1. {lo}-{hi} words. Casual trader voice, like a chat message. No bullet points, no headings.
2. Use at least 3 expressions from this desk slang list (verbatim, case-insensitive):
{jargon}
3. No code, no variable names, no function names, no snake_case, no brackets with code.
4. Every number in the spec must appear (digits are fine; for bar counts you may say hours/days if
   exact, e.g. 16 bars = 4 hours; fractions of equity may be percents, e.g. 0.5 = 50%).
5. Talk naturally; never write "strictly". By desk convention "above/below/breaks/crosses" are strict
   and "at least/at most/no more than/within" are inclusive, so pick the word that matches the spec.
   Never mention parameter letters or internal names (no "k", "r", "lookback parameter n").
6. Voice: {voice} Do not start with "Yo", "Hey" or "Alright".
7. Keep all the logic: anything that changes behaviour must be recoverable from your message alone
   (together with the conventions above). Do not add rules that are not in the spec.
Reply with the message only.'''

VOICES = ["terse and clipped, like a busy PM between meetings.",
          "chatty, a bit informal, thinking out loud.",
          "matter-of-fact, like a senior trader writing a ticket.",
          "impatient, uses fragments and abbreviations.",
          "friendly mentor tone explaining to a junior.",
          "dry and sarcastic but precise.",
          "like a voice note transcribed into text (run-on, conversational).",
          "structured but casual, walks through entry, size, then exits."]


def load_jargon(path="data/jargon.txt", include_pending=True) -> list[str]:
    out = []
    for line in open(path):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.rstrip("\n").split("\t")
        status = parts[2] if len(parts) > 2 else "PENDING"
        if status == "DROP" or (status == "PENDING" and not include_pending):
            continue
        expr = status[5:] if status.startswith("EDIT:") else parts[0]
        out.append(expr.strip())
    return out


def jargon_hits(text: str, jargon: list[str]) -> list[str]:
    low = text.lower()
    hits = []
    for j in jargon:
        jj = j.lower().replace("max n a day", "")
        if not jj:
            continue
        if re.search(r"(?<![a-z])" + re.escape(jj) + r"(?![a-z])", low):
            hits.append(j)
    return hits


def _renderings(name, v, pid):
    """Acceptable textual renderings of a parameter value."""
    r = set()
    if isinstance(v, bool) or isinstance(v, str):
        return None
    if isinstance(v, (tuple, list)):
        return [_renderings(name, x, pid) for x in v]
    fv = float(v)
    r.add(f"{fv:g}")
    if fv.is_integer():
        iv = int(fv)
        r.add(str(iv))
        if iv in NUM_WORDS:
            r.add(NUM_WORDS[iv])
        if pid == "htf_trend" and name == "minutes":
            h = iv // 60
            r |= {f"{h}h", f"{h}-hour", f"{h} hour", f"{h}hr", f"{h}-hr"}
            if h == 1:
                r |= {"hourly", "one-hour", "one hour", "1h"}
            if h == 4:
                r |= {"four-hour", "four hour"}
        bar_params = {("time_stop", "n"), ("cooldown", "n"), ("breakout_pullback", "wait"),
                      ("breakout_pullback", "n"), ("donchian_break", "n"), ("box_breakout", "m")}
        if (pid, name) in bar_params:
            if iv % 4 == 0:
                h = iv // 4
                r |= {f"{h}h", f"{h} hour", f"{h}-hour", f"{h} hr", f"{h}hr"}
                if h in NUM_WORDS:
                    r |= {f"{NUM_WORDS[h]} hour", f"{NUM_WORDS[h]}-hour"}
            if iv % 96 == 0:
                d = iv // 96
                r |= {f"{d} day", f"{d}-day", f"{d}d"}
                if d == 1:
                    r |= {"a day", "one day", "a full day", "24h", "24 hour", "24-hour"}
                if d in NUM_WORDS:
                    r |= {f"{NUM_WORDS[d]} day", f"{NUM_WORDS[d]}-day"}
    if fv < 1 or (pid in ("fixed_frac", "pyramid") and name in ("f", "f0")):
        pct = fv * 100
        r |= {f"{pct:g}%", f"{pct:g} %", f"{pct:g} percent", f"{pct:g} pct"}
        if fv == 0.5:
            r |= {"half"}
        if fv == 0.25:
            r |= {"quarter"}
        if fv == 0.75:
            r |= {"three quarters", "three-quarters", "3/4"}
        if fv == 1.0:
            r |= {"full size", "full port", "all in", "100%", "full"}
    if fv == 0.5 and name in ("a",):
        r |= {"half"}
    if fv == 1.5:
        r |= {"one and a half", "1 and a half", "one-and-a-half"}
    return r


def required_numbers(spec: dict) -> list[tuple[str, set]]:
    req = []
    parts = [(spec["signal"]["id"], spec["signal"]["params"]), (spec["filter"]["id"], spec["filter"]["params"]),
             (spec["sizing"]["id"], spec["sizing"]["params"])] + [(r["id"], r["params"]) for r in spec["risk"]]
    for pid, vals in parts:
        for name, v in vals.items():
            rr = _renderings(name, v, pid)
            if rr is None:
                continue
            if isinstance(rr, list):
                for i, x in enumerate(rr):
                    req.append((f"{pid}.{name}[{i}]", x))
            else:
                req.append((f"{pid}.{name}", rr))
    return req


def _present(text, rends):
    low = text.lower()
    for r in rends:
        rl = r.lower()
        if re.search(r"(?<![0-9a-z.])" + re.escape(rl) + r"(?![0-9a-z])", low) or (
                rl.endswith("%") and rl in low):
            return True
    return False


CODEY = [re.compile(p) for p in (r"\b[a-z]+_[a-z0-9_]+\b", r"\w+\(\s*[\w.]*\s*[,)]", r"\bdef\b",
                                    r"\bstate\[", r"\bind\.", r"\bhist\.", r"[=]{1,2}", r"\breturn\b",
                                    r"`")]


def validate(text: str, spec: dict, jargon: list[str], lo=40, hi=120) -> dict:
    words = len(re.findall(r"\S+", text))
    hits = jargon_hits(text, jargon)
    missing = [lab for lab, rends in required_numbers(spec) if not _present(text, rends)]
    codey = [p.pattern for p in CODEY if p.search(text)]
    problems = []
    if not (lo <= words <= hi):
        problems.append(f"length is {words} words; it must be between {lo} and {hi}")
    if len(set(hits)) < 3:
        problems.append(f"only {len(set(hits))} slang expressions from the list were used verbatim: {hits}")
    if missing:
        problems.append("these parameter values are missing (state them with digits): " + ", ".join(missing))
    if codey:
        problems.append("text looks like code (patterns: " + ", ".join(codey) + ")")
    if spec["sizing"]["id"] == "atr_risk" and not re.search(
            r"\brisk(ing|s)?\b[^;\n]{0,40}?%[^;\n]{0,40}?\bper\b[^;\n]{0,25}?ATR", text, re.I):
        problems.append('volatility sizing must be phrased "risk <r>% of equity per <k> ATR" (desk convention)')
    return {"ok": not problems, "words": words, "n_jargon": len(set(hits)), "jargon": sorted(set(hits)),
            "missing": missing, "codey": codey, "problems": problems}


def _slot_diff(ref, got):
    from .metrics import spec_match
    diffs = []
    if not isinstance(got, dict):
        return ["the reader could not recover any strategy"]
    for slot in ("signal", "filter", "sizing"):
        if spec_match(ref, {**ref, slot: got.get(slot)}) < 1:
            diffs.append(f"{slot}: reader got {json.dumps(got.get(slot))}, intended {json.dumps(ref[slot])}")
    if ref["direction"] != got.get("direction"):
        diffs.append(f"direction: reader got {got.get('direction')}, intended {ref['direction']}")
    if spec_match(ref, {**ref, "risk": got.get("risk")}) < 1:
        diffs.append(f"risk rules: reader got {json.dumps(got.get('risk'))}, intended {json.dumps(ref['risk'])}")
    return diffs


def translate_task(task: dict, client, jargon: list[str], lo=40, hi=120, max_attempts=6,
                   temperature=0.7, seed=0, reader=None) -> dict:
    desc = describe(task["spec"])
    voice = VOICES[int(task["strategy_id"][1:]) % len(VOICES)]
    user = TRANSLATOR_USER.format(desc=desc, src=task["source"], lo=lo, hi=hi,
                                  jargon=", ".join(jargon), voice=voice)
    messages = [{"role": "system", "content": TRANSLATOR_SYS}, {"role": "user", "content": user}]
    attempts = []
    best = None
    for a in range(max_attempts):
        out = client.chat(messages, temperature=temperature, max_tokens=4000, seed=seed + a,
                          tag=f"translate-{task['strategy_id']}-{a}")
        text = (out.get("text") or "").strip().strip('"')
        v = validate(text, task["spec"], jargon, lo, hi) if text else {"ok": False, "problems": [
            f"empty response: {out.get('error')}"], "words": 0, "n_jargon": 0}
        if v["ok"] and reader is not None:
            rt = roundtrip_check(task, text, reader, tag_suffix=str(a))
            v["recovery"] = rt["recovery"]
            if rt["recovery"] < 1.0:   # NaN (reader failure) passes through ungated
                v["ok"] = False
                v["problems"] = ["a reader misunderstood it: " + "; ".join(
                    _slot_diff(task["spec"], rt["recovered_spec"])) +
                    ". Make those parts unambiguous in trader language (do not use block ids or JSON)"]
        attempts.append({"attempt": a, "text": text, **v})
        if v["ok"]:
            best = attempts[-1]
            break
        messages = messages + [{"role": "assistant", "content": text or "(empty)"},
                               {"role": "user", "content": "Revise it. Problems: " + "; ".join(
                                   v["problems"]) + ". Reply with the revised message only."}]
    if best is None:   # keep the attempt with the fewest problems, flagged
        best = min(attempts, key=lambda x: (len(x.get("missing", []) or []) * 10 + len(x["problems"])))
    return {"strategy_id": task["strategy_id"], "q": best["text"], "valid": bool(best["ok"]),
            "n_jargon": best.get("n_jargon", 0), "prompt_len": best.get("words", 0),
            "n_attempts": len(attempts), "attempts": attempts, "spec_desc": desc,
            "recovery": best.get("recovery")}


ROUNDTRIP_PROMPT = '''Below is a trader's instruction. Using the building-block menu, recover the exact
strategy spec as JSON (same schema as below).

MENU
{menu}

INSTRUCTION
"""{q}"""

Reply with ONLY the JSON object: {{"signal": {{"id":..,"params":{{..}}}}, "direction": "long_only"|"long_short",
"filter": {{"id":..,"params":{{..}}}}, "sizing": {{"id":..,"params":{{..}}}}, "risk": [{{"id":..,"params":{{..}}}}]}}'''


def roundtrip_check(task: dict, q: str, client, tag_suffix: str = "") -> dict:
    """Third-model recovery check: q -> spec; returns spec_match recovery rate (v0: off by default)."""
    from .metrics import spec_match
    from .prompts import extract_json, menu_text
    out = client.chat([{"role": "user", "content": ROUNDTRIP_PROMPT.format(menu=menu_text(), q=q)}],
                      temperature=0.0, max_tokens=16000, tag=f"roundtrip-{task['strategy_id']}-{tag_suffix}")
    spec = extract_json(out.get("text"))
    if out.get("error") or out.get("finish_reason") == "length":
        # reader failure is not evidence against the translation: do not gate on it
        return {"recovered_spec": spec, "recovery": float("nan"), "reader_error": out.get("error") or "length"}
    return {"recovered_spec": spec, "recovery": spec_match(task["spec"], spec)}
