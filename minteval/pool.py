"""Generate the candidate pool: sample specs, render, backtest with state tracking, filter."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from .assembler import k_bits, render, sample_spec, spec_key
from .complexity import state_complexity, static_complexity
from .data import load_prices
from .engine import EngineConfig, perf_stats, run_backtest

_P = None
_CFG = None


def _init(price_file, cfg_dict):
    global _P, _CFG
    _P = load_prices(price_file)
    _CFG = EngineConfig(**cfg_dict)


def compile_strategy(src: str):
    g = {"__name__": "ref"}
    exec(compile(src, "<ref>", "exec"), g)
    return g["strategy"]


def evaluate_reference(item):
    idx, spec = item
    src = render(spec)
    r = run_backtest(compile_strategy(src), _P, _CFG, track_state=True)
    st = perf_stats(r.equity, _CFG.initial_equity, _CFG.bars_per_year)
    n = len(r.equity)
    early = int(0.05 * n)
    bust_early = bool(np.nanmin(r.equity[:max(early, 1)]) < 0.5 * _CFG.initial_equity)
    row = {"cand_id": idx, "spec": spec, "error": r.error, "K_bits": k_bits(spec),
           **state_complexity(r.state_spans), **static_complexity(src),
           "R_bench": st["R"], "sharpe_bench": st["sharpe"], "maxdd_bench": st["maxdd"],
           "n_trades_bench": len(r.trades), "n_fills": r.n_fills,
           "frac_bars_in_pos": float((r.pos_q != 0).mean()),
           "always_flat": bool((r.target_q == 0).all()), "bust_early": bust_early,
           "spans_by_key": {k: int(max(v)) for k, v in r.state_spans.items()}}
    return row


def build_pool(cfg: dict, n_candidates: int, seed: int, out_path: str, workers: int | None = None):
    rng = np.random.default_rng(seed)
    specs, seen = [], set()
    while len(specs) < n_candidates:
        sp = sample_spec(rng, **cfg.get("sampler", {}))
        key = spec_key(sp)
        if key in seen:
            continue
        seen.add(key)
        specs.append(sp)
    ecfg = {k: v for k, v in cfg["engine"].items()}
    workers = workers or max(1, os.cpu_count() - 1)
    with mp.get_context("fork").Pool(workers, _init, (cfg["data"]["price_file"], ecfg)) as pool:
        rows = pool.map(evaluate_reference, list(enumerate(specs)), chunksize=4)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows
