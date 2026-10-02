"""Combine per-asset runs into results/multiasset/minteval_multiasset.csv (+ report json).
Per asset: filter survival, tau-vs-R_bench correlation (re-measured tau), ActionMatch ~ log tau + R_bench
regression per model group, inert-block shares."""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]
frames, rep = [], {}
FRONTIER = {"claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro"}
for asset, A in MA["assets"].items():
    d = f"results/multiasset/{asset}"
    if not os.path.exists(f"{d}/gens.csv"): continue
    R = pd.read_csv(f"{d}/refs.csv"); G = pd.read_csv(f"{d}/gens.csv")
    keep = R[["strategy_id", "pass_filter", "frac_in_pos", "vix_mean_held", "frac_held_days_high_vix"]]
    G = G.merge(keep, on="strategy_id"); G["asset"] = asset; G["asset_kind"] = A["kind"]
    G["fee_bp"] = A["fee_bp"]; G["slippage_bp"] = A["slippage_bp"]
    G["subset"] = np.where(G.model.isin(FRONTIER), "frontier200", "full800")
    frames.append(G)
    S = R[R.pass_filter]
    rho, p = spearmanr(np.log10(S.tau_max), S.R_bench)
    r = {"n_refs": len(R), "survive_filter": int(R.pass_filter.sum()),
         "fail_few_trades": int((R.n_trades_bench < cfg["pool"]["min_trades"]).sum()),
         "fail_bust_early": int(R.bust_early.sum()), "fail_always_flat": int(R.always_flat.sum()),
         "survivors_by_btc_tau_bin": None, "spearman_logtau_R": float(rho), "spearman_p": float(p),
         "R_bench_median_survivors": float(S.R_bench.median()), "share_profitable": float((S.R_bench > 0).mean())}
    bins = {json.loads(l)["strategy_id"]: json.loads(l)["tau_bin"] for l in open("results/tasks/tasks.jsonl")}
    r["survivors_by_btc_tau_bin"] = S.strategy_id.map(bins).value_counts().sort_index().to_dict()
    ok = G[G.pass_filter & (G.compile_fail == 0)].copy(); ok["ltau"] = np.log10(ok.tau_max)
    r["regressions"] = {}
    for grp, gg in (("low_cost", ok[~ok.model.isin(FRONTIER)]), ("frontier", ok[ok.model.isin(FRONTIER)])):
        if len(gg) < 50: continue
        for name, f in (("tau", "action_match ~ ltau + C(model)"), ("tau_R", "action_match ~ ltau + R_bench + C(model)")):
            m = smf.ols(f, gg).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(gg.strategy_id)[0]})
            r["regressions"][f"{grp}|{name}"] = {k: {"b": float(m.params[k]), "se": float(m.bse[k]), "p": float(m.pvalues[k])}
                                                 for k in ("ltau", "R_bench") if k in m.params}
    r["by_model"] = {m: {"n": int(len(g)), "compile_ok": float(1 - g.compile_fail.mean()),
                         "AM": float(g[g.compile_fail == 0].action_match.mean()),
                         "silent": float(((g.compile_fail == 0) & (g.action_match < 0.9)).mean()),
                         "abs_es_med": float(g[g.compile_fail == 0].abs_es.median())}
                     for m, g in G[G.pass_filter].groupby("model")}
    if os.path.exists(f"{d}/inert_blocks.csv"):
        I = pd.read_csv(f"{d}/inert_blocks.csv")
        r["inert_share_by_block"] = I.groupby("block").inert.mean().round(3).to_dict()
    rep[asset] = r
os.makedirs("results/multiasset", exist_ok=True)
D = pd.concat(frames, ignore_index=True).sort_values(["asset", "subset", "model", "strategy_id"])
D.to_csv("results/multiasset/minteval_multiasset.csv", index=False, float_format="%.10g")
json.dump(rep, open("results/multiasset/report.json", "w"), indent=1, default=float)
print(D.groupby(["asset", "subset", "model"]).size().to_string())
for a, r in rep.items():
    print(f"\n== {a}: survive {r['survive_filter']}/{r['n_refs']}  spearman(logtau,R)={r['spearman_logtau_R']:.3f} (p={r['spearman_p']:.2g})"
          f"  R_med={r['R_bench_median_survivors']:.3f} profitable={r['share_profitable']:.2f}")
    for k, v in r["regressions"].items(): print("  ", k, {kk: (round(vv["b"], 3), f"{vv['p']:.2g}") for kk, vv in v.items()})
    for m, v in r["by_model"].items(): print(f"   {m:18s} cOK {v['compile_ok']:.3f} AM {v['AM']:.3f} silent {v['silent']:.3f}")
    if "inert_share_by_block" in r: print("   inert:", r["inert_share_by_block"])
