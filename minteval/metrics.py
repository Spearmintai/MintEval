"""Behavioural-equivalence metrics between a reference run and a generated run."""
from __future__ import annotations

import math

import numpy as np

from .engine import perf_stats

NAN = float("nan")
METRIC_COLS = ("spec_match", "action_match", "trade_f1", "abs_es", "delta_sharpe", "delta_maxdd",
               "R_llm", "sharpe_llm", "maxdd_llm")


def active_mask(ref: dict, gen: dict, warmup: int) -> np.ndarray:
    """Bars (after warm-up) where either program holds or targets a non-zero position."""
    m = (ref["pos_q"] != 0) | (gen["pos_q"] != 0) | (ref["target_q"] != 0) | (gen["target_q"] != 0)
    m[:warmup] = False
    return m


def action_match(ref, gen, warmup):
    m = active_mask(ref, gen, warmup)
    if not m.any():
        return NAN
    return float((ref["target_q"][m] == gen["target_q"][m]).mean())


def trade_f1(ref_trades, gen_trades):
    a = {tuple(x[:3]) for x in ref_trades}
    b = {tuple(x[:3]) for x in gen_trades}
    if not a and not b:
        return 1.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def compare(ref: dict, gen: dict, cfg, spec_match=NAN) -> dict:
    """ref/gen: result dicts (equity, target_q, pos_q, trades, error)."""
    if gen.get("error"):
        return {"compile_fail": 1, **{c: NAN for c in METRIC_COLS}, "spec_match": spec_match}
    sr = perf_stats(ref["equity"], cfg.initial_equity, cfg.bars_per_year)
    sg = perf_stats(gen["equity"], cfg.initial_equity, cfg.bars_per_year)
    return {"compile_fail": 0, "spec_match": spec_match,
            "action_match": action_match(ref, gen, cfg.warmup_bars),
            "trade_f1": trade_f1(ref["trades"], gen["trades"]),
            "abs_es": abs(sr["R"] - sg["R"]) * 1e4,
            "delta_sharpe": sg["sharpe"] - sr["sharpe"],
            "delta_maxdd": sg["maxdd"] - sr["maxdd"],
            "R_llm": sg["R"], "sharpe_llm": sg["sharpe"], "maxdd_llm": sg["maxdd"]}


def spec_match(ref_spec: dict, gen_spec: dict | None) -> float:
    """Fraction of the 4 slots (signal+direction, filter, sizing, risk set) exactly right."""
    if not isinstance(gen_spec, dict):
        return 0.0

    def norm_params(p):
        out = {}
        for k, v in (p or {}).items():
            if isinstance(v, (list, tuple)):
                v = tuple(float(x) for x in v)
            elif isinstance(v, bool):
                v = bool(v)
            elif isinstance(v, (int, float)):
                v = float(v)
            out[str(k)] = v
        return out

    def slot(x):
        if not isinstance(x, dict):
            return None
        return (str(x.get("id")), tuple(sorted(norm_params(x.get("params")).items())))

    try:
        ok_sig = (slot(ref_spec["signal"]) == slot(gen_spec.get("signal"))
                  and ref_spec["direction"] == gen_spec.get("direction"))
        ok_filter = slot(ref_spec["filter"]) == slot(gen_spec.get("filter") or {"id": "no_filter", "params": {}})
        ok_size = slot(ref_spec["sizing"]) == slot(gen_spec.get("sizing"))
        gr = gen_spec.get("risk") or []
        ok_risk = sorted(map(slot, ref_spec["risk"])) == sorted(map(slot, gr)) if all(
            isinstance(r, dict) for r in gr) else False
    except Exception:  # noqa: BLE001
        return 0.0
    return (ok_sig + ok_filter + ok_size + ok_risk) / 4.0
