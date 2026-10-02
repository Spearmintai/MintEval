"""Where do frontier silent failures concentrate? Opus 5.5 (and GPT-5.4-mini on the same 200 tasks)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf
T = {json.loads(l)["strategy_id"]: json.loads(l)["spec"] for l in open("results/tasks/tasks.jsonl")}
fr = pd.read_csv("results/minteval_v0_frontier.csv"); full = pd.read_csv("results/minteval_v0.csv")
D = pd.concat([fr, full[(full.setting == "open") & full.strategy_id.isin(set(fr.strategy_id))]])
D = D[D.compile_fail == 0].copy()
RISK = ["fixed_stop", "breakeven", "trailing_hwm", "time_stop", "take_profit", "partial_take", "box_shift", "cooldown", "daily_cap"]
def feats(sid):
    sp = T[sid]; r = [x["id"] for x in sp["risk"]]
    f = {"signal": sp["signal"]["id"], "sizing": sp["sizing"]["id"],
         "htf_filter": int(sp["filter"]["id"] == "htf_trend"), "long_short": int(sp["direction"] == "long_short"),
         # risk layers reading each other's state: trailing stop armed by the breakeven flag
         "cross_state": int(any(x["id"] == "trailing_hwm" and x["params"].get("after_be") for x in sp["risk"])),
         # several layers co-own the stop/take level (min/max merge across layers)
         "multi_stop_layers": int(sum(x in r for x in ["fixed_stop", "breakeven", "trailing_hwm", "box_shift"]) >= 2),
         "stateful_signal": int(sp["signal"]["id"] == "breakout_pullback")}
    f.update({f"has_{k}": int(k in r) for k in RISK})
    return f
F = pd.DataFrame([{"strategy_id": s, **feats(s)} for s in D.strategy_id.unique()])
D = D.merge(F, on="strategy_id"); D["silent"] = (D.action_match < 0.9).astype(int)
out = {}
for m in ["claude-opus-5.5", "gpt-5.4-mini"]:
    g = D[D.model == m]; base = g.silent.mean(); rows = []
    for col in ["htf_filter", "long_short", "cross_state", "multi_stop_layers", "stateful_signal"] + [f"has_{k}" for k in RISK]:
        a, b = g[g[col] == 1], g[g[col] == 0]
        if len(a) < 5 or len(b) < 5: continue
        tab = [[a.silent.sum(), len(a) - a.silent.sum()], [b.silent.sum(), len(b) - b.silent.sum()]]
        rows.append({"feature": col, "n_with": len(a), "silent_with": round(a.silent.mean(), 3),
                     "silent_without": round(b.silent.mean(), 3), "fisher_p": float(f"{fisher_exact(tab)[1]:.3g}")})
    for col in ["signal", "sizing", "n_risk"]:
        for v, h in g.groupby(col):
            rows.append({"feature": f"{col}={v}", "n_with": len(h), "silent_with": round(h.silent.mean(), 3),
                         "silent_without": round(g[g[col] != v].silent.mean(), 3), "fisher_p": None})
    R = pd.DataFrame(rows)
    lr = smf.logit("silent ~ C(signal) + C(sizing) + n_risk + htf_filter + long_short + " +
                   " + ".join(f"has_{k}" for k in RISK), g).fit(disp=0, method="bfgs", maxiter=2000)
    out[m] = {"n": len(g), "silent_rate": base, "features": R.to_dict(orient="records"),
              "logit": {k: {"b": float(lr.params[k]), "p": float(lr.pvalues[k])} for k in lr.params.index}}
    print(f"\n=== {m}: n={len(g)} silent={base:.3f}")
    print(R.sort_values("silent_with", ascending=False).to_string(index=False))
json.dump(out, open("results/opus_structure.json", "w"), indent=1, default=float)
