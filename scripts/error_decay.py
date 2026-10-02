"""Do implementation errors diversify? Daily error series e[s,m,t] = r_gen - r_ref (daily returns from end-of-day
equity), per asset and model; average pairwise correlation across strategies; variance of an equal-weight portfolio
of k random strategies relative to the mean single-strategy variance, vs 1/k (independent) and rho+(1-rho)/k.
Usage: error_decay.py <asset> [<asset> ...]   -> results/decay/<asset>_daily.npz, results/decay/decay.json"""
import sys, json, os, pickle, hashlib, multiprocessing as mp
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from minteval.data import load_prices
from minteval.sandbox import run_sandboxed
from minteval.evaluate import load_tasks, reference_runs

cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]
os.makedirs("results/decay", exist_ok=True)
CSV = pd.read_csv("results/multiasset/minteval_multiasset.csv")
KS = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]

def day_index(asset, ot):
    t = pd.to_datetime(ot, unit="ms", utc=True)
    if MA["assets"][asset]["kind"] == "equity":
        t = t.tz_convert("America/New_York")
    return pd.Index(t.strftime("%Y-%m-%d"))

def daily_returns(eq, days, warm):
    s = pd.Series(eq[warm:], index=days[warm:]).groupby(level=0).last()
    return s.pct_change().iloc[1:]

def _gen(args):
    code, pf, ecfg, timeout = args
    r = run_sandboxed(code, pf, ecfg, timeout_s=timeout)
    return None if r.get("error") else r["equity"]

def build(asset):
    A = MA["assets"][asset]; pf = A["price_file"]
    ecfg = dict(cfg["engine"], fee_bp=A["fee_bp"], slippage_bp=A["slippage_bp"], bars_per_year=A["bars_per_year"])
    P = load_prices(pf); days = day_index(asset, P["open_time"]); warm = ecfg["warmup_bars"]
    rows = CSV[(CSV.asset == asset) & CSV.pass_filter & (CSV.compile_fail == 0)]
    tasks = {t["strategy_id"]: t for t in load_tasks()}
    refs = reference_runs([tasks[s] for s in rows.strategy_id.unique()], pf, ecfg, cache_dir=f"results/cache/ref_{asset}")
    code = {}
    for d in ("results/generations", "results/generations_frontier"):
        for f in os.listdir(d):
            if f.endswith("__open.jsonl"):
                for l in open(f"{d}/{f}"):
                    g = json.loads(l); code[(g["strategy_id"], g["model"])] = g["code"]
    timeout = MA["timeout_s_per_70k_bars"] * len(P["close"]) / 70000
    keys = list(zip(rows.strategy_id, rows.model))
    with mp.get_context("fork").Pool(max(1, os.cpu_count() - 1)) as pool:
        eqs = pool.map(_gen, [(code[k], pf, ecfg, timeout) for k in keys], chunksize=2)
    out = {}
    for (sid, m), eq in zip(keys, eqs):
        if eq is None: continue
        rg = daily_returns(eq, days, warm); rr = daily_returns(refs[sid]["equity"], days, warm)
        out.setdefault(m, {})[sid] = (rg - rr).values
    dates = daily_returns(refs[rows.strategy_id.iloc[0]]["equity"], days, warm).index
    np.savez_compressed(f"results/decay/{asset}_daily.npz", dates=np.array(dates),
                        **{f"{m}|{sid}": v for m, dd in out.items() for sid, v in dd.items()})
    return out

def analyse(asset, out, rng):
    res = {}
    for m, dd in out.items():
        E = np.column_stack(list(dd.values()))                        # T x N
        v = E.var(axis=0, ddof=1); keep = v > 0
        E = E[:, keep]; n = E.shape[1]
        C = np.corrcoef(E, rowvar=False); iu = np.triu_indices(n, 1)
        rho = float(np.nanmean(C[iu]))
        # day-block bootstrap CI for rho (resample 20-day blocks)
        T = E.shape[0]; B = 20; nb = T // B; bs = []
        for _ in range(200):
            idx = np.concatenate([np.arange(b * B, b * B + B) for b in rng.integers(0, nb, nb)])
            Cb = np.corrcoef(E[idx], rowvar=False); bs.append(np.nanmean(Cb[iu]))
        curve = {}
        vbar = E.var(axis=0, ddof=1).mean()
        for k in KS:
            if k > n: continue
            draws = [E[:, rng.choice(n, k, replace=False)].mean(axis=1).var(ddof=1) / vbar for _ in range(300 if k < n else 1)]
            curve[k] = {"ratio": float(np.mean(draws)), "indep": 1.0 / k, "equicorr": rho + (1 - rho) / k}
        # also on standardised errors (each desk's error scaled to unit variance)
        Z = (E - E.mean(0)) / E.std(0, ddof=1)
        zfull = float(Z.mean(axis=1).var(ddof=1))
        res[m] = {"n_strategies": int(n), "n_zero_var_dropped": int((~keep).sum()), "T_days": int(T), "rho_mean": rho,
                  "rho_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "rho_median": float(np.nanmedian(C[iu])), "share_pairs_pos": float(np.nanmean(C[iu] > 0)),
                  "curve": curve, "std_errors_portfolio_var_all": zfull, "std_errors_indep_benchmark": 1.0 / n}
        print(f"{asset} {m:18s} N={n:4d} T={T} rho={rho:+.4f} [{res[m]['rho_ci'][0]:+.4f},{res[m]['rho_ci'][1]:+.4f}]  "
              + "  ".join(f"k={k}:{c['ratio']:.3f}(1/k {c['indep']:.3f})" for k, c in curve.items() if k in (10, 100, 200)))
    return res

if __name__ == "__main__":
    rng = np.random.default_rng(20261003)
    path = "results/decay/decay.json"; R = json.load(open(path)) if os.path.exists(path) else {}
    for asset in sys.argv[1:]:
        R[asset] = analyse(asset, build(asset), rng)
        json.dump(R, open(path, "w"), indent=1)
