"""Port checks for the MintEval "Real" subset (pilot).

For every task directory data/real/tasks/<id>/ this script
  1. checks reference.py against the sandbox's static rules (minteval.sandbox.check_source);
  2. runs it in-process through minteval.engine.run_backtest (engine section of configs/base.yaml),
     recording every (pos, target, stop, take) the strategy returns;
  3. runs it again through minteval.sandbox.run_sandboxed and checks the target path is identical;
  4. runs the independent second implementation second_impl.py (plain numpy/pandas, own indicator
     code, different structure) on the SAME realised position path and compares, bar by bar,
     the target grid index, the entry decisions and the stop/take levels;
  5. writes the results into meta.json["port_checks"].

Usage: python data/real/check_real.py [task_id ...]
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minteval.data import load_prices  # noqa: E402
from minteval.engine import EngineConfig, perf_stats, run_backtest  # noqa: E402
from minteval.pool import compile_strategy  # noqa: E402
from minteval.sandbox import SandboxViolation, check_source, run_sandboxed, soft_violations  # noqa: E402

TASKS = ROOT / "data" / "real" / "tasks"


def recorder(strategy, n):
    rec = {"pos": np.full(n, np.nan), "target": np.full(n, np.nan),
           "stop": np.full(n, np.nan), "take": np.full(n, np.nan)}

    def wrapped(hist, state, pos, ind):
        out = strategy(hist, state, pos, ind)
        t = len(hist) - 1
        rec["pos"][t] = pos
        rec["target"][t] = out["target"]
        s, k = out.get("stop"), out.get("take")
        rec["stop"][t] = np.nan if s is None else s
        rec["take"][t] = np.nan if k is None else k
        return out
    return wrapped, rec


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem + "_" + path.parent.name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def same_level(a, b, rtol=1e-9):
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=rtol, atol=0.0)
    return both_nan | close


def check_task(tdir: Path, P: dict, cfg: dict, price_file: str) -> dict:
    src = (tdir / "reference.py").read_text()
    out = {"checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        check_source(src)
        out["sandbox_static_ok"] = True
    except SandboxViolation as e:
        out["sandbox_static_ok"] = False
        out["sandbox_static_error"] = str(e)
        return out
    out["soft_violations"] = soft_violations(src)
    ecfg = EngineConfig(**cfg)
    n = len(P["close"])
    strat, rec = recorder(compile_strategy(src), n)
    t0 = time.time()
    r = run_backtest(strat, P, ecfg)
    out["engine_seconds"] = round(time.time() - t0, 2)
    out["engine_error"] = r.error
    w = ecfg.warmup_bars
    out["n_trades"] = len(r.trades)
    out["n_long_trades"] = sum(1 for x in r.trades if x[0] > 0)
    out["n_short_trades"] = sum(1 for x in r.trades if x[0] < 0)
    out["frac_bars_in_position"] = float((r.pos_q[w:] != 0).mean())
    out["n_fills"] = r.n_fills
    out["distinct_abs_position_levels"] = sorted({round(abs(int(k)) * ecfg.quant_delta, 2)
                                                  for k in np.unique(r.pos_q[w:]) if k != 0})
    st = perf_stats(r.equity, ecfg.initial_equity, ecfg.bars_per_year)
    out.update({"R": st["R"], "sharpe": st["sharpe"], "maxdd": st["maxdd"]})
    hold = [x[2] - x[1] for x in r.trades]
    out["median_trade_bars"] = float(np.median(hold)) if hold else None
    # stop/take fills: position went to zero without the strategy asking for it
    tq = r.target_q
    pq = r.pos_q
    exits_by_order = int(sum(1 for t in range(w + 1, n) if pq[t - 1] != 0 and pq[t] == 0 and tq[t - 1] != 0))
    out["exits_by_stop_or_take"] = exits_by_order
    # sandbox run
    t0 = time.time()
    sb = run_sandboxed(src, price_file, cfg, timeout_s=10.0)
    out["sandbox_seconds"] = round(time.time() - t0, 2)
    out["sandbox_error"] = sb.get("error")
    out["sandbox_ok"] = sb.get("error") is None and bool(np.array_equal(sb["target_q"], r.target_q))
    # second implementation
    si = tdir / "second_impl.py"
    if si.exists():
        mod = load_module(si)
        pos_held = r.pos_q.astype(float) * ecfg.quant_delta
        t0 = time.time()
        res = mod.decide(P, pos_held, w)
        out["second_impl_seconds"] = round(time.time() - t0, 2)
        tgt2 = np.asarray(res["target"], float)
        k2 = np.clip(np.round(tgt2 / ecfg.quant_delta), -20, 20)
        rng = np.arange(w, n)
        agree = (k2[rng] == tq[rng])
        out["second_impl_target_agreement"] = float(agree.mean())
        out["second_impl_target_mismatch_bars"] = [int(x) for x in rng[~agree][:20]]
        flat = pq[rng] == 0
        ref_entry = flat & (tq[rng] != 0)
        si_entry = flat & (k2[rng] != 0)
        out["entries_ref"] = int(ref_entry.sum())
        out["entries_second"] = int(si_entry.sum())
        out["entries_both"] = int((ref_entry & si_entry).sum())
        denom = max(1, int((ref_entry | si_entry).sum()))
        out["entry_jaccard"] = (ref_entry & si_entry).sum() / denom
        out["entry_bar_agreement_while_flat"] = float((ref_entry == si_entry)[flat].mean()) if flat.any() else None
        for lvl in ("stop", "take"):
            if lvl in res:
                a = np.asarray(res[lvl], float)[rng]
                b = rec[lvl][rng]
                ok = same_level(a, b)
                out[f"second_impl_{lvl}_agreement"] = float(ok.mean())
                out[f"second_impl_{lvl}_mismatch_bars"] = [int(x) for x in rng[~ok][:10]]
    return out


def main(ids=None):
    cfg_all = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    price_file = str(ROOT / cfg_all["data"]["price_file"])
    cfg = cfg_all["engine"]
    P = load_prices(price_file)
    P["time"] = P["open_time"].astype(np.float64)
    dirs = sorted(d for d in TASKS.iterdir() if d.is_dir() and (d / "reference.py").exists())
    if ids:
        dirs = [d for d in dirs if d.name in ids]
    summary = {}
    for d in dirs:
        res = check_task(d, P, cfg, price_file)
        summary[d.name] = res
        mp = d / "meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            meta["port_checks"] = res
            mp.write_text(json.dumps(meta, indent=1, default=float) + "\n")
        keys = ("n_trades", "frac_bars_in_position", "engine_seconds", "sandbox_ok", "sandbox_error",
                "second_impl_target_agreement", "entry_jaccard", "second_impl_stop_agreement",
                "second_impl_take_agreement", "engine_error", "sandbox_static_error")
        print(d.name, json.dumps({k: res.get(k) for k in keys}, default=float), flush=True)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:] or None)
