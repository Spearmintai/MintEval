"""Run model generations through the sandbox and score them against the references."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import EngineConfig, run_backtest
from .metrics import compare, spec_match
from .sandbox import run_sandboxed

CSV_COLS = ["strategy_id", "model", "setting", "K_bits", "tau_max", "tau_p90", "n_registers",
            "nesting_depth", "mccabe", "halstead", "n_jargon", "prompt_len", "realized_vol",
            "R_bench", "sharpe_bench", "maxdd_bench", "n_trades_bench",
            "R_llm", "sharpe_llm", "maxdd_llm", "compile_fail",
            "spec_match", "action_match", "trade_f1", "abs_es", "delta_sharpe", "delta_maxdd",
            "error_type", "tau_bin", "signal_id", "n_risk"]


def load_tasks(path="results/tasks/tasks.jsonl"):
    return [json.loads(l) for l in open(path)]


# ------------------------------------------------------------------ reference runs
def _ref_job(args):
    sid, src, price_file, cfg = args
    from .pool import compile_strategy
    from .data import load_prices
    global _PRICES
    if "_PRICES" not in globals():
        _PRICES = load_prices(price_file)
    r = run_backtest(compile_strategy(src), _PRICES, EngineConfig(**cfg))
    return sid, {"equity": r.equity, "target_q": r.target_q, "pos_q": r.pos_q,
                 "trades": r.trades, "error": r.error}


def reference_runs(tasks, price_file, cfg: dict, cache_dir="results/cache/ref", workers=None):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    todo = [(t["strategy_id"], t["source"], price_file, cfg) for t in tasks
            if not (cache / f"{t['strategy_id']}_{tag}.pkl").exists()]
    if todo:
        with mp.get_context("fork").Pool(workers or max(1, os.cpu_count() - 1)) as pool:
            for sid, res in pool.imap_unordered(_ref_job, todo, chunksize=2):
                with open(cache / f"{sid}_{tag}.pkl", "wb") as f:
                    pickle.dump(res, f)
    out = {}
    for t in tasks:
        with open(cache / f"{t['strategy_id']}_{tag}.pkl", "rb") as f:
            out[t["strategy_id"]] = pickle.load(f)
    return out


# ------------------------------------------------------------------ generated runs
def _gen_job(args):
    sid, model, code, price_file, cfg, timeout = args
    if code is None:
        return sid, model, {"error": "NoCode: no code block in model output", "error_bar": -1}
    if code.startswith("# TRUNCATED"):
        return sid, model, {"error": "Truncated: output hit max_tokens before any code", "error_bar": -1}
    if code.startswith("# API_ERROR"):
        return sid, model, {"error": "APIError: request failed after retries", "error_bar": -1}
    return sid, model, run_sandboxed(code, price_file, cfg, timeout_s=timeout)


def error_type(err: str | None) -> str:
    if not err:
        return ""
    return err.split(":", 1)[0]


def score(tasks, generations: dict, refs: dict, price_file: str, cfg: dict, timeout=10.0,
          workers=None, setting="open", extra_cols: dict | None = None) -> pd.DataFrame:
    """generations: {(sid, model): {"code": str|None, "spec": dict|None}}"""
    ecfg = EngineConfig(**cfg)
    jobs = [(sid, m, g.get("code"), price_file, cfg, timeout) for (sid, m), g in generations.items()]
    results = {}
    with mp.get_context("fork").Pool(workers or max(1, os.cpu_count() - 1)) as pool:
        for sid, m, res in pool.imap_unordered(_gen_job, jobs, chunksize=1):
            results[(sid, m)] = res
    tmap = {t["strategy_id"]: t for t in tasks}
    rows = []
    for (sid, m) in sorted(results):
        t = tmap[sid]
        g = generations[(sid, m)]
        sm = spec_match(t["spec"], g.get("spec")) if setting == "closed" else float("nan")
        met = compare(refs[sid], results[(sid, m)], ecfg, spec_match=sm)
        row = {"strategy_id": sid, "model": m, "setting": setting,
               **{k: t.get(k) for k in ("K_bits", "tau_max", "tau_p90", "n_registers",
                                        "nesting_depth", "mccabe", "halstead", "R_bench",
                                        "sharpe_bench", "maxdd_bench", "n_trades_bench",
                                        "tau_bin", "realized_vol")},
               "n_jargon": t.get("n_jargon"), "prompt_len": t.get("prompt_len"),
               **met, "error_type": error_type(results[(sid, m)].get("error")),
               "signal_id": t["spec"]["signal"]["id"], "n_risk": len(t["spec"]["risk"])}
        if extra_cols:
            row.update(extra_cols.get((sid, m), {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[[c for c in CSV_COLS if c in df.columns] + [c for c in df.columns if c not in CSV_COLS]]


def write_csv(df: pd.DataFrame, path: str) -> str:
    df = df.sort_values(["setting", "model", "strategy_id"]).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.10g")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
