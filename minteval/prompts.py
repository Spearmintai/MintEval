"""Prompts shown to the models under test (open + closed settings) and helpers."""
from __future__ import annotations

import json
import re

from .primitives import REGISTRY, by_family

INTERFACE_DOC = r'''You are implementing a trading strategy for a backtest engine. Write ONE Python function:

    def strategy(hist, state, pos, ind) -> dict

ENGINE
- Market: BTCUSDT, 15-minute bars. The engine calls strategy() once per bar, at the bar's close,
  starting after a 960-bar warm-up (full history before that is available in `hist`).
- Return {"target": float, "stop": float or None, "take": float or None}.
  * target: desired position as a fraction of equity in [-1, 1] (negative = short). It is executed at
    the NEXT bar's open. Targets are rounded to a 0.05 grid. Returning the current `pos` means "no change".
  * stop / take: absolute price levels for resting exit orders that are active during the NEXT bar only
    (return them again on every bar you want them to stay active). For a long, the stop triggers if the
    bar's low <= stop and the take triggers if its high >= take; mirrored for a short. Fills at the level,
    or at the open if the bar gaps through it. If both trigger in one bar, the stop wins. A triggered order
    closes the whole position.
- pos: the position currently held, a float on the 0.05 grid (0.0 after a stop/take fill).
- hist: read-only history up to and including the current bar: hist.open, hist.high, hist.low,
  hist.close, hist.volume, hist.time (bar open time, ms since epoch, UTC). Index -1 is the current bar,
  -2 the previous one. len(hist) = number of bars so far. Accessing a future bar raises LookaheadError.
  hist.htf(minutes) returns the same kind of object for COMPLETED higher-timeframe candles (UTC-aligned,
  e.g. hist.htf(60), hist.htf(240)); the candle in progress is not included.
- ind: indicator functions; each returns a numpy array aligned with its input (last element = current bar):
    ind.sma(x, n)      mean of the last n values (NaN for the first n-1)
    ind.ema(x, n)      e[0]=x[0]; e[i]=a*x[i]+(1-a)*e[i-1], a=2/(n+1)
    ind.rma(x, n)      same as ema with a=1/n (Wilder)
    ind.atr(h, n)      rma of true range, h is hist (or hist.htf(..)); TR[0]=high-low
    ind.rsi(x, n)      Wilder RSI
    ind.macd(x, fast, slow, signal) -> (macd_line, signal_line, histogram)
    ind.highest(x, n)  max of the last n values including the current one
    ind.lowest(x, n)   min of the last n values including the current one
  x is a series such as hist.close or hist.htf(240).close. Use these functions (they are cached);
  recomputing indicators over the whole history yourself on every bar will exceed the time limit.
- state: a dict that persists between calls. Keys must be str and values must be float, int or bool
  (no None, lists, dicts or other objects). `del state[k]`, `k in state`, `state.get(k, default)` work.
  This is the ONLY way to remember anything between bars.

RULES
- Only `import math` and `import numpy as np` are allowed. No global variables, no closures or nested
  functions, no classes, no mutable default arguments, no attribute assignment, no names starting with
  an underscore, no file or network access. Module-level helper functions and constant literals are fine.
- The whole backtest (~70,000 bars) must finish within 10 seconds.

DESK CONVENTIONS (apply unless the instruction explicitly says otherwise)
- "Entry price" = the close of the bar on which you decide to enter (the signal bar), even though the
  fill happens at the next open. "ATR" = ind.atr(hist, 14) and "entry ATR" = its value on the signal
  bar, frozen for the life of the trade. Distances such as "2 ATR" use the entry ATR.
- Stops and take-profits are sent from the signal bar onward, on every bar while the trade is live.
  If several stop rules apply, use the tightest one; if several take-profit levels apply, the nearest.
- Profit conditions ("once it is 1 ATR in profit") are evaluated on bar closes, inclusive (>=).
- Wording: "above", "below", "breaks", "crosses above/below" are strict comparisons (a cross means
  at-or-below on the previous bar and strictly above now); "at least", "at most", "no more than",
  "within" are inclusive. "The prior N bars" / "N-bar high" for breakouts means the N bars BEFORE the
  current bar.
- Breakout conditions ("closes above the prior N-bar high", "breaks the box") are LEVEL conditions,
  checked on every bar (true on every bar the close is beyond the level), not one-time cross events.
  Only the word "cross"/"crosses" denotes an event (at-or-below on the previous bar, above now).
- Volatility sizing "risk r% of equity per k ATR" means: position fraction = r * close / (k * ATR) on the
  signal bar (r as a fraction, e.g. 1% -> 0.01), then apply any stated cap/floor.
- Only open a trade when flat. A signal in the direction opposite to the open trade closes it (target 0)
  at the next open; it does not reverse the position.
- "Bars" are 15-minute bars. Bar counts are measured with bar indices (t = len(hist) - 1).
'''

OPEN_TASK = '''{doc}
TRADER'S INSTRUCTION
"""{q}"""

Implement exactly this strategy. Reply with a single ```python code block containing the full module.'''

STEP1_TASK = '''{doc}
BUILDING-BLOCK MENU
The desk's strategies are combinations of the following blocks. Every numeric parameter takes one of
the listed grid values.
{menu}

A strategy = one signal block + direction ("long_only" or "long_short") + one filter block + one sizing
block + a list of 1-3 risk blocks.

TRADER'S INSTRUCTION
"""{q}"""

Step 1: identify the blocks and parameters. Reply with ONLY a JSON object of the form
{{"signal": {{"id": ..., "params": {{...}}}}, "direction": "long_only"|"long_short",
  "filter": {{"id": ..., "params": {{...}}}}, "sizing": {{"id": ..., "params": {{...}}}},
  "risk": [{{"id": ..., "params": {{...}}}}, ...]}}
Use exactly the ids and parameter names from the menu. A parameter that is a tuple (e.g. fsg) is a JSON list.'''

STEP2_TASK = '''Step 2: now implement the strategy you specified, following the interface, rules and desk
conventions above. Reply with a single ```python code block containing the full module.'''


def menu_text() -> str:
    lines = []
    for fam in ("signal", "filter", "sizing", "risk"):
        lines.append(f"[{fam}]")
        for p in by_family(fam):
            grid = ", ".join(f"{k} in {list(v)}" for k, v in p.params.items()) or "no parameters"
            ex = {k: v[0] for k, v in p.params.items()}
            desc = p.describe(ex)
            lines.append(f"- {p.id} ({grid}). Semantics, shown with the first grid values: {desc}")
    return "\n".join(lines)


def open_prompt(q: str) -> str:
    return OPEN_TASK.format(doc=INTERFACE_DOC, q=q)


def closed_prompt_step1(q: str) -> str:
    return STEP1_TASK.format(doc=INTERFACE_DOC, menu=menu_text(), q=q)


_CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    blocks = _CODE_RE.findall(text)
    if blocks:
        for b in reversed(blocks):
            if "def strategy" in b:
                return b
        return blocks[-1]
    if "def strategy" in text:
        return text[text.index("def strategy") - 0:] if "import" not in text else text
    return None


def extract_json(text: str | None):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    cand = m.group(1) if m else text
    i, j = cand.find("{"), cand.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(cand[i:j + 1])
    except Exception:  # noqa: BLE001
        return None
