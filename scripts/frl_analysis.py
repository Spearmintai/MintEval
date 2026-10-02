"""FRL paper: every number in paper_frl/ from results/frl/minteval_v0_frl.csv.

Sample: open setting; four low-cost models on full800, Claude Opus 5.5 on frontier200. Compiled runs only
unless stated. ES = R_bench - R_llm in bp of initial equity. Winsorisation is within model.
Regressors: ltau = log10(tau_max) centred on the 800-task mean (coef = per decade of tau_max);
vol = realised vol z-scored over the 800 tasks (coef = per 1 SD); controls K_bits, log trade count,
McCabe, R_bench; block FE = signal id + one dummy per risk block. SEs clustered by task.
Outputs: paper_frl/numbers.tex, paper_frl/frl_analysis.json, paper_frl/figs/fig3_tau.pdf."""
import sys, json, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT = "paper_frl"
os.makedirs(f"{OUT}/figs", exist_ok=True)
LOW = ["gpt-5.4-mini", "claude-haiku", "qwen2.5-coder-32b", "qwen2.5-coder-7b"]
OPUS = "claude-opus-5.5"
NAMES = {"gpt-5.4-mini": "GPTMini", "claude-haiku": "Haiku", "qwen2.5-coder-32b": "QwenL",
         "qwen2.5-coder-7b": "QwenS", OPUS: "Opus"}

d = pd.read_csv("results/frl/minteval_v0_frl.csv")
d = d[d.setting == "open"]
d = d[((d.subset == "full800") & d.model.isin(LOW)) | ((d.subset == "frontier200") & (d.model == OPUS))].copy()
tasks = {t["strategy_id"]: t for t in map(json.loads, open("results/tasks/tasks.jsonl"))}
RISK = sorted({r["id"] for t in tasks.values() for r in t["spec"]["risk"]})
for r in RISK:
    d[f"rk_{r}"] = d.strategy_id.map(lambda s: int(any(x["id"] == r for x in tasks[s]["spec"]["risk"])))
t800 = d.drop_duplicates("strategy_id")
mu_t, mu_v, sd_v = np.log10(t800.tau_max).mean(), t800.realized_vol.mean(), t800.realized_vol.std()
d["ltau"] = np.log10(d.tau_max) - mu_t
d["vol"] = (d.realized_vol - mu_v) / sd_v
d["ltrades"] = np.log(d.n_trades_bench)
d["es"] = (d.R_bench - d.R_llm) * 1e4
d["Rg"] = d.R_bench_gross


def wins(s, lo, hi):
    return s.clip(s.quantile(lo), s.quantile(hi))


for lo, hi, tag in ((0.01, 0.99, ""), (0.005, 0.995, "_w05")):
    for col in ("abs_es", "es"):
        d[f"w{col}{tag}"] = d.groupby("model")[col].transform(lambda s: wins(s, lo, hi))
# compile failures imputed as maximum loss: the model's 99th-percentile |ES| (its winsorisation cap)
cap = d.groupby("model").abs_es.transform(lambda s: s.quantile(0.99))
d["wabs_es_imp"] = np.where(d.compile_fail == 1, cap, d.wabs_es)
d["am"] = d.action_match * 1e2   # percentage points, so coefficients are readable

CMP = d[d.compile_fail == 0]
N = {}
res = {"centering": {"mean_log10_tau": mu_t, "vol_mean": mu_v, "vol_sd": sd_v}}

# ---------------------------------------------------------------- Table 1 + signed ES
tab = {}
for m in LOW + [OPUS]:
    g, c = d[d.model == m], CMP[CMP.model == m]
    n = len(g)
    es = c.wes
    tab[m] = dict(n=n, compile=1 - g.compile_fail.mean(), med_abs=c.abs_es.median(),
                  mean_es=es.mean(), t_es=es.mean() / (es.std() / np.sqrt(len(es))),
                  med_es=c.es.median(), share_pos=(c.es > 0).mean(), mean_abs=c.wabs_es.mean(),
                  am=c.action_match.mean(), exact=(c.action_match == 1).sum() / n,
                  silent=(c.action_match < 0.9).sum() / n, sd_es=es.std(),
                  rllm_med=c.R_llm.median(), rref_med=c.R_bench.median(),
                  rllm_gt_ref=(c.R_llm > c.R_bench).mean(),
                  # share of the intended outcome rewritten: |ES| / |R_ref|, per task (median is robust to small R_ref)
                  ratio_med=(c.abs_es / (c.R_bench.abs() * 1e4)).median(),
                  ratio_gt1=(c.abs_es > c.R_bench.abs() * 1e4).mean())
res["table1"] = tab

# ---------------------------------------------------------------- Table 2
CTRL = "K_bits + ltrades + mccabe + R_bench"
FE = "C(signal_id) + " + " + ".join(f"rk_{r}" for r in RISK)


def fit(f, g):
    return smf.ols(f, g).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g.strategy_id)[0]})


def spec(col, y="wabs_es", data=None, ctrl_extra=""):
    data = CMP if data is None else data
    mfe = " + C(model)" if col in (1, 2, 3, 4, 5) else ""
    rhs = {1: "ltau", 2: "ltau + vol", 3: "ltau * vol"}.get(col, "ltau * vol + " + CTRL + ctrl_extra + " + " + FE)
    if col == 5:
        data = data[data.model != OPUS]
    if col == 6:
        data = data[data.model == OPUS]
    return fit(f"{y} ~ {rhs}{mfe}", data)


def summ(r):
    k = {x: {"b": float(r.params[x]), "t": float(r.tvalues[x]), "p": float(r.pvalues[x])}
         for x in ("ltau", "vol", "ltau:vol") if x in r.params}
    return {**k, "n": int(r.nobs), "r2a": float(r.rsquared_adj)}


res["table2"] = {c: summ(spec(c)) for c in range(1, 7)}
rob = {"am": ("am", None, ""), "imputed": ("wabs_es_imp", d, ""), "gross": ("wabs_es", None, " + Rg"),
       "w05": ("wabs_es_w05", None, "")}
res["robust"] = {k: {c: summ(spec(c, y=y, data=dat, ctrl_extra=ce)) for c in (4, 5, 6)}
                 for k, (y, dat, ce) in rob.items()}

# ---------------------------------------------------------------- economics / misc
fr = t800.assign(drag=(t800.R_bench_gross - t800.R_bench) * 1e4)
res["misc"] = {"friction_drag_med_bp": float(fr.drag.median()), "window": "2022-01-01 to 2023-12-31",
               "n_low_compiled": int((CMP.model != OPUS).sum()), "n_opus": int((CMP.model == OPUS).sum()),
               "ltau_sd_decades": float(np.log10(t800.tau_max).std())}
# median |ES| and ActionMatch by tau quintile
q = {}
for m in LOW + [OPUS]:
    c = CMP[CMP.model == m]
    q[m] = {int(b): {"med": float(g.abs_es.median()), "am": float(g.action_match.mean()),
                     "mean": float(g.wabs_es.mean()), "n": len(g)} for b, g in c.groupby("tau_bin")}
res["by_quintile"] = q
json.dump(res, open(f"{OUT}/frl_analysis.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- Fig 3
rng = np.random.default_rng(20261002)


def boot_med(x, B=2000):
    x = np.asarray(x)
    b = np.median(rng.choice(x, (B, len(x))), axis=1)
    return np.median(x), *np.percentile(b, [2.5, 97.5])


plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.2, 3.3))
LAB = {"gpt-5.4-mini": "GPT-5.4-mini", "claude-haiku": "Claude Haiku", "qwen2.5-coder-32b": "Qwen2.5-Coder 32B",
       "qwen2.5-coder-7b": "Qwen2.5-Coder 7B", OPUS: "Claude Opus 5.5"}
MK = {"gpt-5.4-mini": "o", "claude-haiku": "s", "qwen2.5-coder-32b": "^", "qwen2.5-coder-7b": "v", OPUS: "D"}
off = dict(zip(LOW + [OPUS], np.linspace(-0.16, 0.16, 5)))
for m in LOW + [OPUS]:
    c = CMP[CMP.model == m]
    st = np.array([boot_med(g.abs_es) for _, g in c.groupby("tau_bin")])
    x = np.arange(1, 6) + off[m]
    col, lw = ("black", 1.4) if m == OPUS else ("0.55", 1.0)
    ax.errorbar(x, st[:, 0], yerr=[st[:, 0] - st[:, 1], st[:, 2] - st[:, 0]], color=col, lw=lw, marker=MK[m],
                ms=4, capsize=2, mfc="white" if m != OPUS else "black", label=LAB[m])
ax.set_xticks(range(1, 6), [f"Q{i}" for i in range(1, 6)])
ax.set_xlabel(r"$\tau_{\max}$ quintile (state span)")
ax.set_ylabel(r"Median $|ES|$ (bp of initial equity)")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_ylim(bottom=-100)
fig.tight_layout()
fig.savefig(f"{OUT}/figs/fig3_tau.pdf"); fig.savefig(f"{OUT}/figs/fig3_tau.png", dpi=200)

# ---------------------------------------------------------------- print
pd.set_option("display.width", 200)
print(pd.DataFrame(tab).T.round(3))
for c, r in res["table2"].items():
    print(c, {k: (round(v["b"], 1), round(v["t"], 2)) if isinstance(v, dict) else round(v, 3) for k, v in r.items()})
for k, v in res["robust"].items():
    for c, r in v.items():
        print("rob", k, c, {kk: (round(vv["b"], 1), round(vv["t"], 2)) if isinstance(vv, dict) else round(vv, 3) for kk, vv in r.items()})
print(res["misc"])
for m, v in q.items():
    print(m, {b: (round(x["med"]), round(x["am"], 3), x["n"]) for b, x in v.items()})
