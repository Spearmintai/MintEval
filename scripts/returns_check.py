"""Why negative reference returns do not drive the conclusions: gross (frictionless) reference returns,
ActionMatch by reference-return band, tau effect controlling for R_bench. -> results/returns_check.json"""
import sys, json, yaml
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from minteval.evaluate import load_tasks, reference_runs
from minteval.engine import perf_stats, EngineConfig
cfg = yaml.safe_load(open("configs/base.yaml")); pf = cfg["data"]["price_file"]
tasks = load_tasks()
gross_cfg = dict(cfg["engine"], fee_bp=0.0, slippage_bp=0.0)
refs0 = reference_runs(tasks, pf, gross_cfg, cache_dir="results/cache/ref_gross")
ec = EngineConfig(**gross_cfg)
gross = {sid: perf_stats(r["equity"], ec.initial_equity, ec.bars_per_year)["R"] for sid, r in refs0.items()}
d = pd.read_csv("results/minteval_v0.csv")
t1 = d.drop_duplicates("strategy_id")[["strategy_id", "R_bench", "n_trades_bench"]].copy()
t1["R_gross"] = t1.strategy_id.map(gross)
out = {"R_net_median": float(t1.R_bench.median()), "R_gross_median": float(t1.R_gross.median()),
       "share_net_pos": float((t1.R_bench > 0).mean()), "share_gross_pos": float((t1.R_gross > 0).mean()),
       "R_gross_q25": float(t1.R_gross.quantile(.25)), "R_gross_q75": float(t1.R_gross.quantile(.75))}
ok = d[d.compile_fail == 0].copy()
ok["r_band"] = pd.qcut(ok.R_bench, 5, labels=False)
out["am_by_rband"] = {f"{m}|{s}": g.groupby("r_band").action_match.mean().round(4).tolist()
                      for (m, s), g in ok.groupby(["model", "setting"])}
ok["log_tau"] = np.log10(ok.tau_max)
for s in ["open", "closed"]:
    ds = ok[ok.setting == s]
    r = smf.ols("action_match ~ log_tau + R_bench + C(model)", ds).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(ds.strategy_id)[0]})
    out[f"reg_{s}"] = {k: {"coef": float(r.params[k]), "p": float(r.pvalues[k])} for k in ("log_tau", "R_bench")}
    r2 = smf.ols("action_match ~ R_bench + C(model)", ds).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ds.strategy_id)[0]})
    out[f"reg_{s}_R_only"] = {"coef": float(r2.params["R_bench"]), "p": float(r2.pvalues["R_bench"])}
json.dump(out, open("results/returns_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
