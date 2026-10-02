"""H2: implementation error in money terms. winsorized |ES| ~ log tau + vol + log tau x vol, per model.
Compiled runs only. vol = realized_vol (z-scored over tasks), tau = log10 tau_max (centered).
Winsorization at the 99th pct within model x setting (robustness: 95th; HC1 SEs)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
full = pd.read_csv("results/minteval_v0.csv")
fr = pd.read_csv("results/minteval_v0_frontier.csv")
frames = [full.assign(subset="full800"), fr.assign(subset="frontier200")]
ids200 = set(fr.strategy_id)
frames.append(full[(full.setting == "open") & full.strategy_id.isin(ids200)].assign(subset="frontier200"))
D = pd.concat(frames)
D = D[D.compile_fail == 0].copy()
t = D.drop_duplicates("strategy_id")
mu_t, mu_v, sd_v = np.log10(t.tau_max).mean(), t.realized_vol.mean(), t.realized_vol.std()
D["ltau"] = np.log10(D.tau_max) - mu_t
D["vol"] = (D.realized_vol - mu_v) / sd_v
D["hivol"] = (D.realized_vol > t.realized_vol.quantile(0.75)).astype(int)
D["ltrades"] = np.log(D.n_trades_bench)
out = {}
for (sub, m, s), g in D.groupby(["subset", "model", "setting"]):
    if sub == "frontier200" and s != "open": continue
    res = {"n": len(g)}
    for w in (0.99, 0.95):
        g = g.copy(); cap = g.abs_es.quantile(w); g["wes"] = g.abs_es.clip(upper=cap)
        for name, f in {"base": "wes ~ ltau * vol", "ctrl": "wes ~ ltau * vol + ltrades + K_bits",
                        "hivol": "wes ~ ltau * hivol + ltrades + K_bits"}.items():
            r = smf.ols(f, g).fit(cov_type="HC1")
            res[f"w{int(w*100)}_{name}"] = {k: {"b": round(float(r.params[k]), 1), "se": round(float(r.bse[k]), 1),
                                              "p": float(f"{r.pvalues[k]:.3g}")} for k in r.params.index if k != "Intercept"}
            res[f"w{int(w*100)}_{name}"]["r2"] = round(float(r.rsquared), 3)
    out[f"{sub}|{m}|{s}"] = res
json.dump(out, open("results/es_regression.json", "w"), indent=1)
print("coef in bp of initial equity; ltau per decade of tau_max; vol per 1 SD of realized_vol\n")
for k, v in out.items():
    b = v["w99_base"]; c = v["w99_ctrl"]
    fmt = lambda d, x: f"{d[x]['b']:>7}({d[x]['p']:.2g})" if x in d else "   -"
    print(f"{k:42s} n={v['n']:4d} | base: tau {fmt(b,'ltau')} vol {fmt(b,'vol')} txv {fmt(b,'ltau:vol')} | +ctrl: tau {fmt(c,'ltau')} vol {fmt(c,'vol')} txv {fmt(c,'ltau:vol')}")
