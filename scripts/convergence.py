"""Cross-model convergence: do different models make the SAME mistakes?
L1 task level : silent-failure indicators of two models across tasks (phi, observed/expected co-failure, Fisher).
L2 bar level  : on tasks where both fail silently, among bars where BOTH deviate from the reference, the share with the
                identical (wrong) action, vs a circular-time-shift null (shift >= 1 day, 200 draws) that keeps each
                series' structure but breaks alignment.
Usage: convergence.py <asset> ...   (re-runs generated programs to get bar-level targets; cached in results/convergence/)"""
import sys, json, os, itertools, multiprocessing as mp
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from scipy.stats import fisher_exact
from minteval.sandbox import run_sandboxed
from minteval.evaluate import load_tasks, reference_runs
from minteval.data import load_prices

cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]
CSV = pd.read_csv("results/multiasset/minteval_multiasset.csv")
ids200 = {json.loads(l)["strategy_id"] for l in open("results/tasks/frontier200.jsonl")}
FRONTIER = ["claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro"]
LOWCOST = ["gpt-5.4-mini", "claude-haiku", "qwen2.5-coder-32b", "qwen2.5-coder-7b"]
os.makedirs("results/convergence", exist_ok=True)
WARM = cfg["engine"]["warmup_bars"]

def _gen(a):
    code, pf, ecfg, to = a
    r = run_sandboxed(code, pf, ecfg, timeout_s=to)
    return None if r.get("error") else r["target_q"].astype(np.int8)

def targets(asset):
    path = f"results/convergence/{asset}_targets.npz"
    if os.path.exists(path):
        Z = np.load(path); return {tuple(k.split("|")): Z[k] for k in Z.files}
    A = MA["assets"][asset]; pf = A["price_file"]
    ecfg = dict(cfg["engine"], fee_bp=A["fee_bp"], slippage_bp=A["slippage_bp"], bars_per_year=A["bars_per_year"])
    rows = CSV[(CSV.asset == asset) & CSV.pass_filter & (CSV.compile_fail == 0)]
    code = {}
    for d in ("results/generations", "results/generations_frontier"):
        for f in os.listdir(d):
            if f.endswith("__open.jsonl"):
                for l in open(f"{d}/{f}"):
                    g = json.loads(l); code[(g["strategy_id"], g["model"])] = g["code"]
    to = MA["timeout_s_per_70k_bars"] * len(load_prices(pf)["close"]) / 70000
    keys = list(zip(rows.strategy_id, rows.model))
    with mp.get_context("fork").Pool(max(1, os.cpu_count() - 1)) as pool:
        tq = pool.map(_gen, [(code[k], pf, ecfg, to) for k in keys], chunksize=2)
    tasks = {t["strategy_id"]: t for t in load_tasks()}
    refs = reference_runs([tasks[s] for s in rows.strategy_id.unique()], pf, ecfg, cache_dir=f"results/cache/ref_{asset}")
    out = {(s, m): t for (s, m), t in zip(keys, tq) if t is not None}
    for s in rows.strategy_id.unique():
        out[(s, "reference")] = refs[s]["target_q"].astype(np.int8)
    np.savez_compressed(path, **{f"{s}|{m}": v for (s, m), v in out.items()})
    return out

def analyse(asset, T, rng):
    S = CSV[(CSV.asset == asset) & CSV.pass_filter]
    res = {"L1": {}, "L2": {}}
    pairs = list(itertools.combinations(FRONTIER + LOWCOST, 2))
    for m1, m2 in pairs:
        frontier_involved = m1 in FRONTIER or m2 in FRONTIER
        a = S[S.model == m1].set_index("strategy_id"); b = S[S.model == m2].set_index("strategy_id")
        ids = a.index.intersection(b.index)
        if frontier_involved: ids = ids.intersection(list(ids200))
        a, b = a.loc[ids], b.loc[ids]
        ok = (a.compile_fail == 0) & (b.compile_fail == 0)
        f1 = (a.action_match[ok] < 0.9).values; f2 = (b.action_match[ok] < 0.9).values
        n = len(f1); both = int((f1 & f2).sum())
        exp = f1.mean() * f2.mean() * n
        phi = float(np.corrcoef(f1, f2)[0, 1]) if f1.std() > 0 and f2.std() > 0 else float("nan")
        tab = [[both, int((f1 & ~f2).sum())], [int((~f1 & f2).sum()), int((~f1 & ~f2).sum())]]
        odds, p = fisher_exact(tab)
        key = f"{m1}|{m2}"
        res["L1"][key] = {"n_tasks": n, "both_fail": both, "expected_if_indep": float(exp), "obs_over_exp": float(both / exp) if exp else float("nan"),
                          "phi": phi, "odds_ratio": float(odds), "fisher_p": float(p)}
        # L2 on co-failed tasks
        cof = [s for s, x, y in zip(np.array(ids)[ok.values], f1, f2) if x and y]
        obs_n = obs_same = 0; null = np.zeros(200); null_n = np.zeros(200); per_task = []
        for s in cof:
            if (s, m1) not in T or (s, m2) not in T: continue
            r, x, y = T[(s, "reference")][WARM:], T[(s, m1)][WARM:], T[(s, m2)][WARM:]
            dx, dy = x != r, y != r
            both_dev = dx & dy
            k = int(both_dev.sum())
            if k == 0: continue
            same = int((x[both_dev] == y[both_dev]).sum()); obs_n += k; obs_same += same
            L = len(r); shifts = rng.integers(96, L - 96, 200)
            ts = []
            for i, sh in enumerate(shifts):
                ys = np.roll(y, sh); dys = ys != np.roll(r, sh)        # shift m2 together with its own deviation pattern
                bd = dx & (np.roll(y, sh) != r)                          # m2-shifted deviates from the CURRENT reference
                kk = int(bd.sum())
                if kk:
                    null[i] += (x[bd] == ys[bd]).sum(); null_n[i] += kk
                    ts.append((x[bd] == ys[bd]).mean())
            per_task.append({"s": s, "obs": same / k, "null_mean": float(np.mean(ts)) if ts else float("nan")})
        if obs_n:
            nr = null / np.maximum(null_n, 1)
            res["L2"][key] = {"n_cofailed_tasks": len(per_task), "bars_both_deviate": int(obs_n),
                              "same_wrong_action": obs_same / obs_n, "null_mean": float(nr.mean()),
                              "null_p95": float(np.percentile(nr, 95)), "ratio": float((obs_same / obs_n) / nr.mean()),
                              "perm_p": float((np.sum(nr >= obs_same / obs_n) + 1) / (len(nr) + 1)),
                              "share_tasks_above_null": float(np.mean([t["obs"] > t["null_mean"] for t in per_task if np.isfinite(t["null_mean"])]))}
        l1 = res["L1"][key]; l2 = res["L2"].get(key, {})
        print(f"{asset[:3]} {m1[:12]:12s}~{m2[:12]:12s} L1 n={n:3d} both={both:3d} obs/exp={l1['obs_over_exp']:.2f} phi={phi:+.2f} p={p:.1g}"
              + (f" | L2 tasks={l2['n_cofailed_tasks']:3d} same={l2['same_wrong_action']:.3f} null={l2['null_mean']:.3f} x{l2['ratio']:.2f} p={l2['perm_p']:.3f}" if l2 else ""), flush=True)
    return res

if __name__ == "__main__":
    rng = np.random.default_rng(20261004)
    path = "results/convergence/convergence.json"; R = json.load(open(path)) if os.path.exists(path) else {}
    for asset in sys.argv[1:]:
        R[asset] = analyse(asset, targets(asset), rng); json.dump(R, open(path, "w"), indent=1)
