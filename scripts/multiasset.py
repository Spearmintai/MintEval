"""Re-run the existing reference programs and generated programs on other instruments (no new generation).
Usage: multiasset.py <asset_name> [stage...]   stages: refs gens inert   (default: all)
Outputs under results/multiasset/<asset>/ ; the combined CSV is built by multiasset_collect.py."""
import sys, json, os, copy, multiprocessing as mp
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from minteval.assembler import render
from minteval.complexity import state_complexity
from minteval.data import load_prices
from minteval.engine import EngineConfig, perf_stats, run_backtest
from minteval.evaluate import load_tasks, reference_runs, score
from minteval.pool import compile_strategy

cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]
asset = sys.argv[1]; A = MA["assets"][asset]
FORCE = "--force" in sys.argv
stages = [a for a in sys.argv[2:] if a != "--force"] or ["refs", "gens", "inert"]
OUT = f"results/multiasset/{asset}"; os.makedirs(OUT, exist_ok=True)
_done = {"refs": "refs.csv", "gens": "gens.csv", "inert": "inert_blocks.csv"}
if not FORCE:   # skip stages already completed (lets the overnight supervisor resume without redoing work)
    stages = [st for st in stages if not os.path.exists(f"{OUT}/{_done[st]}")]
    print(asset, "stages to run:", stages, flush=True)
ecfg = dict(cfg["engine"], fee_bp=A["fee_bp"], slippage_bp=A["slippage_bp"], bars_per_year=A["bars_per_year"])
PF = A["price_file"]
tasks = load_tasks()
_P = None

def _init():
    global _P
    _P = load_prices(PF)

def ref_row(t):
    ec = EngineConfig(**ecfg)
    r = run_backtest(compile_strategy(t["source"]), _P, ec, track_state=True)
    st = perf_stats(r.equity, ec.initial_equity, ec.bars_per_year)
    n = len(r.equity); early = max(int(0.05 * n), 1)
    c = _P["close"]; lr = np.diff(np.log(c), prepend=np.log(c[0])); m = r.pos_q != 0
    held_days = pd.to_datetime(_P["open_time"][m], unit="ms", utc=True).normalize().unique()
    return {"strategy_id": t["strategy_id"], "error": r.error, **state_complexity(r.state_spans),
            "R_bench": st["R"], "sharpe_bench": st["sharpe"], "maxdd_bench": st["maxdd"], "n_trades_bench": len(r.trades),
            "bust_early": bool(np.nanmin(r.equity[:early]) < 0.5 * ec.initial_equity),
            "always_flat": bool((r.target_q == 0).all()), "frac_in_pos": float(m.mean()),
            "realized_vol": float(lr[m].std() * np.sqrt(ec.bars_per_year)) if m.sum() > 1 else float("nan"),
            "held_days": [str(d.date()) for d in held_days]}

def inert_rows(t):
    """For each risk block: is behaviour identical with the block removed? (block never effective here)"""
    ec = EngineConfig(**ecfg); out = []
    base = run_backtest(compile_strategy(t["source"]), _P, ec).target_q
    for i, rb in enumerate(t["spec"]["risk"]):
        sp = copy.deepcopy(t["spec"]); del sp["risk"][i]
        if not sp["risk"]:
            continue    # a program needs >= 1 risk layer in the template; single-layer tasks are skipped
        alt = run_backtest(compile_strategy(render(sp)), _P, ec).target_q
        out.append({"strategy_id": t["strategy_id"], "block": rb["id"], "inert": bool(np.array_equal(base, alt))})
    return out

if "refs" in stages:
    with mp.get_context("fork").Pool(max(1, os.cpu_count() - 1), _init) as pool:
        R = pool.map(ref_row, tasks, chunksize=2)
    R = pd.DataFrame(R)
    R["pass_filter"] = R.error.isna() & (R.n_trades_bench >= cfg["pool"]["min_trades"]) & ~R.bust_early & ~R.always_flat
    vix = pd.read_csv(MA["vix_file"]); vix["d"] = pd.to_datetime(vix.DATE).dt.strftime("%Y-%m-%d"); V = dict(zip(vix.d, vix.CLOSE))
    R["vix_mean_held"] = R.held_days.map(lambda ds: float(np.mean([V[d] for d in ds if d in V])) if any(d in V for d in ds) else np.nan)
    R["frac_held_days_high_vix"] = R.held_days.map(lambda ds: float(np.mean([V[d] > MA["high_vix"] for d in ds if d in V])) if any(d in V for d in ds) else np.nan)
    R.drop(columns=["held_days"]).to_csv(f"{OUT}/refs.csv", index=False)
    print(asset, "refs: pass_filter", int(R.pass_filter.sum()), "/", len(R), flush=True)

if "gens" in stages:
    R = pd.read_csv(f"{OUT}/refs.csv").set_index("strategy_id")
    T = []
    for t in tasks:
        t = dict(t); r = R.loc[t["strategy_id"]]
        for k in ("tau_max", "tau_p90", "n_registers", "R_bench", "sharpe_bench", "maxdd_bench", "n_trades_bench", "realized_vol"):
            t[k] = r[k]
        T.append(t)
    refs = reference_runs(T, PF, ecfg, cache_dir=f"results/cache/ref_{asset}")   # per-asset cache (key hashes cfg only)
    n_bars = len(load_prices(PF)["close"])
    timeout = MA["timeout_s_per_70k_bars"] * n_bars / 70000
    frames = []
    for d, setting_files in (("results/generations", "__open.jsonl"), ("results/generations_frontier", "__open.jsonl")):
        for f in sorted(os.listdir(d)):
            if not f.endswith(setting_files): continue
            G = [json.loads(l) for l in open(f"{d}/{f}")]
            gens = {(g["strategy_id"], g["model"]): {"code": g["code"], "spec": g["spec"]} for g in G}
            Ts = [t for t in T if (t["strategy_id"], G[0]["model"]) in gens]
            frames.append(score(Ts, gens, refs, PF, ecfg, timeout=timeout, setting="open"))
            print(asset, "scored", f, flush=True)
    S = pd.concat(frames); S["asset"] = asset; S["timeout_s"] = timeout
    S.to_csv(f"{OUT}/gens.csv", index=False)

if "inert" in stages:
    with mp.get_context("fork").Pool(max(1, os.cpu_count() - 1), _init) as pool:
        rows = [r for rs in pool.map(inert_rows, tasks, chunksize=2) for r in rs]
    pd.DataFrame(rows).to_csv(f"{OUT}/inert_blocks.csv", index=False)
    print(asset, "inert check done", len(rows), flush=True)
