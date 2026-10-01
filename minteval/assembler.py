"""Compose primitives into reference programs, sample candidates, compute K_bits."""
from __future__ import annotations

import json
import math
import textwrap

import numpy as np

from .primitives import REGISTRY, by_family

FAMILIES = ("signal", "filter", "sizing", "risk")
DIRECTIONS = ("long_only", "long_short")
MAX_RISK = 3
BASE_TRADE_KEYS = ("side", "ref", "ref_atr")

HEADER = '''import math


def tighter_stop(current, level, side):
    if current is None:
        return level
    return max(current, level) if side > 0 else min(current, level)


def nearer_take(current, level, side):
    if current is None:
        return level
    return min(current, level) if side > 0 else max(current, level)


def strategy(hist, state, pos, ind):
    close = hist.close
    high = hist.high
    low = hist.low
    t = len(close) - 1
    price = close[-1]
    atr = ind.atr(hist, 14)[-1]
    side = 1 if pos > 0 else (-1 if pos < 0 else 0)
'''


def _ind(code: str, level: int) -> str:
    code = textwrap.dedent(code).strip("\n")
    if not code:
        return ""
    return textwrap.indent(code, "    " * level) + "\n"


def risk_layers(spec):
    return [(r["id"], r["params"]) for r in spec["risk"]]


def all_parts(spec):
    parts = [(spec["signal"]["id"], spec["signal"]["params"]),
             (spec["filter"]["id"], spec["filter"]["params"]),
             (spec["sizing"]["id"], spec["sizing"]["params"])]
    return parts + risk_layers(spec)


def render(spec: dict) -> str:
    """Render a structured spec to reference-program source."""
    parts = [(REGISTRY[i], v) for i, v in all_parts(spec)]

    def hook(name, level):
        return "".join(_ind(p.render(name, v), level) for p, v in parts if p.hooks.get(name))

    trade_keys = list(BASE_TRADE_KEYS)
    for p, _ in parts:
        trade_keys += [k for k in p.trade_keys if k not in trade_keys]
    long_only = spec["direction"] == "long_only"
    sizing = REGISTRY[spec["sizing"]["id"]]
    size_expr = sizing.render("size", spec["sizing"]["params"])

    src = HEADER
    src += "\n    # entry signal\n" + hook("signal", 1)
    src += "\n    # a trade we opened is over (exit order filled, closed, or never filled)\n"
    src += '    if "side" in state and state["side"] != side:\n'
    src += hook("on_exit", 2)
    src += f"        for key in {tuple(trade_keys)!r}:\n"
    src += "            if key in state:\n                del state[key]\n"
    src += "\n    target = 0.0\n    if side == 0:\n"
    cond = "sig > 0" if long_only else "sig != 0"
    src += f"        if {cond}:\n            ok = True\n"
    src += hook("gate", 3)
    src += "            if ok:\n"
    src += f"                size = {size_expr}\n"
    src += ('                state["side"] = sig\n'
            '                state["ref"] = price\n'
            '                state["ref_atr"] = atr\n')
    src += hook("on_entry", 4)
    src += "                target = sig * size\n"
    src += "    else:\n        target = pos\n"
    src += hook("hold", 2)
    src += hook("exit", 2)
    src += "        if sig == -side:\n            target = 0.0\n"
    src += "\n    stop = None\n    take = None\n"
    orders = hook("orders", 2)
    if orders:
        src += ('    if "side" in state:\n'
                '        s = state["side"]\n'
                '        ref = state["ref"]\n'
                '        ra = state["ref_atr"]\n')
        src += orders
    src += '    return {"target": target, "stop": stop, "take": take}\n'
    return src


def describe(spec: dict) -> str:
    """Precise structured English description (input to back-translation)."""
    lines = [REGISTRY[spec["signal"]["id"]].describe(spec["signal"]["params"])]
    lines.append("Direction: long only (bearish signals only close longs)." if spec["direction"] == "long_only"
                 else "Direction: long and short (a bullish signal opens a long when flat, a bearish one "
                      "opens a short when flat).")
    if spec["filter"]["id"] != "no_filter":
        lines.append(REGISTRY[spec["filter"]["id"]].describe(spec["filter"]["params"]))
    lines.append(REGISTRY[spec["sizing"]["id"]].describe(spec["sizing"]["params"]))
    for rid, v in risk_layers(spec):
        lines.append(REGISTRY[rid].describe(v))
    lines.append("A signal in the opposite direction of an open trade closes it at the next open.")
    return "\n".join(f"- {l}" for l in lines)


# ------------------------------------------------------------------------- sampling
def _valid(ids: list[str]) -> bool:
    for i in ids:
        p = REGISTRY[i]
        if any(e in ids for e in p.excludes):
            return False
    return True


def sample_spec(rng: np.random.Generator, risk_size_p=(0.25, 0.35, 0.40), filter_p=0.35) -> dict:
    while True:
        sig = rng.choice([p.id for p in by_family("signal")])
        direction = DIRECTIONS[int(rng.integers(2))]
        filt = "htf_trend" if rng.random() < filter_p else "no_filter"
        sizing = rng.choice([p.id for p in by_family("sizing")])
        k = int(rng.choice([1, 2, 3], p=list(risk_size_p)))
        risk_ids = sorted(rng.choice([p.id for p in by_family("risk")], size=k, replace=False).tolist())
        ids = [sig, filt, sizing] + risk_ids
        if not _valid(ids):
            continue
        spec = {"signal": {"id": str(sig), "params": _draw(REGISTRY[sig], rng)},
                "direction": direction,
                "filter": {"id": filt, "params": _draw(REGISTRY[filt], rng)},
                "sizing": {"id": str(sizing), "params": _draw(REGISTRY[sizing], rng)},
                "risk": [{"id": r, "params": _draw(REGISTRY[r], rng)} for r in risk_ids]}
        # conditional variant: trailing gated on breakeven requires breakeven in the stack
        for r in spec["risk"]:
            if r["id"] == "trailing_hwm" and r["params"]["after_be"] and "breakeven" not in risk_ids:
                r["params"]["after_be"] = False
        return spec


def _draw(p, rng):
    out = {}
    for name, grid in p.params.items():
        v = grid[int(rng.integers(len(grid)))]
        out[name] = tuple(int(x) for x in v) if isinstance(v, tuple) else (
            v.item() if hasattr(v, "item") else v)
    return out


def k_bits(spec: dict) -> float:
    """Description length of the program in bits.

    structural: log2(#signals) + log2(#directions) + log2(#filter options) + log2(#sizing)
                + log2(MAX_RISK)  [number of risk layers] + sum_layers log2(#risk primitives)
    parametric: sum over the chosen primitives' params of log2(grid size)
                (trailing_hwm.after_be only counts when breakeven is present, since it is
                 forced to False otherwise)
    """
    bits = (math.log2(len(by_family("signal"))) + math.log2(len(DIRECTIONS))
            + math.log2(len(by_family("filter"))) + math.log2(len(by_family("sizing")))
            + math.log2(MAX_RISK) + len(spec["risk"]) * math.log2(len(by_family("risk"))))
    risk_ids = [r["id"] for r in spec["risk"]]
    for pid, vals in all_parts(spec):
        for name, grid in REGISTRY[pid].params.items():
            if pid == "trailing_hwm" and name == "after_be" and "breakeven" not in risk_ids:
                continue
            bits += math.log2(len(grid))
    return bits


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)
