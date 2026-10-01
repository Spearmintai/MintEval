"""All paper numbers, tables and figures from the results CSV (nothing typed by hand).
Usage: analyze.py <results.csv> <out_dir>   -> figs/*.pdf, table1.tex, numbers.tex, analysis.json"""
import sys, json
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

csv, out = sys.argv[1], Path(sys.argv[2]); (out / "figs").mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv)
tasks = [json.loads(l) for l in open("results/tasks/tasks.jsonl")]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; MARK = ["o", "s", "^", "D"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "font.family": "serif", "pdf.fonttype": 42})
models = [m for m in ["gpt-5.4-mini", "claude-haiku", "qwen2.5-coder-32b", "qwen2.5-coder-7b"] if m in set(df.model)]
models += sorted(set(df.model) - set(models))
style = {m: (COLORS[i % 4], MARK[i % 4]) for i, m in enumerate(models)}
A = {}

def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x[~np.isnan(x)]); rng = np.random.default_rng(seed)
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

# ------------------------------------------------------------- Table 1 (per setting)
rows = []
for (m, s), g in df.groupby(["model", "setting"]):
    ok = g[g.compile_fail == 0]
    rows.append({"model": m, "setting": s, "n": len(g), "CompileOK": 1 - g.compile_fail.mean(),
                 "SpecMatch": g.spec_match.mean() if s == "closed" else np.nan,
                 "ActionMatch": ok.action_match.mean(), "ActionMatch_all0": g.action_match.fillna(0).mean(),
                 "TradeF1": ok.trade_f1.mean(), "ES_med": ok.abs_es.median(),
                 "exact": (ok.action_match == 1).sum() / len(g),
                 "silent": ((g.compile_fail == 0) & (g.action_match < 0.9)).mean()})
T = pd.DataFrame(rows); T.to_csv(out / "table1.csv", index=False)
A["table1"] = T.to_dict(orient="records")
def f3(x): return "--" if pd.isna(x) else f"{x:.3f}"
lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
         r"Model & Set. & CompileOK & SpecMatch & ActionMatch & TradeF1 & $|$ES$|$ med (bp) & Silent \\", r"\midrule"]
for s in ["open", "closed"]:
    for m in models:
        r = T[(T.model == m) & (T.setting == s)]
        if r.empty: continue
        r = r.iloc[0]
        lines.append(f"{m} & {s} & {f3(r.CompileOK)} & {f3(r.SpecMatch)} & {f3(r.ActionMatch)} & {f3(r.TradeF1)} & "
                     f"{'--' if pd.isna(r.ES_med) else f'{r.ES_med:.0f}'} & {f3(r.silent)} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"; lines.append(r"\end{tabular}")
(out / "table1.tex").write_text("\n".join(lines) + "\n")

# ------------------------------------------------------------- Fig 1: ActionMatch by tau bin
fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
A["fig1"] = {}
for ax, s in zip(axes, ["open", "closed"]):
    for m in models:
        g = df[(df.model == m) & (df.setting == s) & (df.compile_fail == 0)]
        if g.empty: continue
        st = [boot_ci(g[g.tau_bin == b].action_match.values) for b in range(5)]
        A["fig1"][f"{m}|{s}"] = st
        mu = np.array([x[0] for x in st]); lo = np.array([x[1] for x in st]); hi = np.array([x[2] for x in st])
        c, mk = style[m]; xs = np.arange(5)
        ax.errorbar(xs, mu, yerr=[mu - lo, hi - mu], color=c, marker=mk, ms=5, capsize=2, lw=1.5, label=m,
                    markeredgecolor="white", markeredgewidth=0.8)
    ax.set_title(f"{s} setting", color=INK); ax.set_xticks(range(5)); ax.set_xlabel(r"$\tau_{\max}$ quintile (1 = shortest state span)")
    ax.set_xticklabels([str(i + 1) for i in range(5)])
axes[0].set_ylabel("ActionMatch (compiled runs)")
axes[1].legend(frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(out / "figs/fig1_tau.pdf"); plt.close(fig)

# ------------------------------------------------------------- Fig 2: SpecMatch vs ActionMatch
g = df[(df.setting == "closed") & (df.compile_fail == 0)]
fig, ax = plt.subplots(figsize=(3.4, 2.6))
rng = np.random.default_rng(0)
ax.axvspan(0.97, 1.03, ymin=0, ymax=0.9 / 1.05, color="#e34948", alpha=0.08, lw=0)
for m in models:
    h = g[g.model == m]; c, mk = style[m]
    ax.scatter(h.spec_match + rng.uniform(-0.025, 0.025, len(h)), h.action_match, s=8, color=c, marker=mk,
               alpha=0.55, lw=0, label=m)
ax.axhline(0.9, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("SpecMatch (fraction of 4 slots exact)"); ax.set_ylabel("ActionMatch")
ax.set_xlim(-0.08, 1.08); ax.set_ylim(0, 1.05); ax.legend(frameon=False, fontsize=6, markerscale=1.5, loc="lower right")
ax.text(0.94, 0.45, "spec right,\nbehaviour wrong", ha="right", fontsize=6.5, color=INK)
fig.tight_layout(); fig.savefig(out / "figs/fig2_spec_vs_action.pdf"); plt.close(fig)
sm1 = g[g.spec_match == 1]
A["fig2"] = {m: {"n_spec1": int((sm1.model == m).sum()),
                 "frac_spec1_am_lt_0_9": float((sm1[sm1.model == m].action_match < 0.9).mean()) if (sm1.model == m).any() else None}
             for m in models}
A["fig2"]["all"] = {"n_spec1": int(len(sm1)), "frac_spec1_am_lt_0_9": float((sm1.action_match < 0.9).mean()) if len(sm1) else None}

# ------------------------------------------------------------- Fig 3: tau_max histogram
taus = np.array([t["tau_max"] for t in tasks])
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.hist(np.log10(taus), bins=30, color=COLORS[0], edgecolor="white", linewidth=1)
ax.set_xlabel(r"$\log_{10}\,\tau_{\max}$ (bars)"); ax.set_ylabel("tasks")
fig.tight_layout(); fig.savefig(out / "figs/fig3_tau_hist.pdf"); plt.close(fig)

# ------------------------------------------------------------- regressions (primitive fixed effects)
tk = pd.DataFrame([{"strategy_id": t["strategy_id"], "risk_ids": "+".join(r["id"] for r in t["spec"]["risk"])}
                   for t in tasks])
for t in tasks:
    pass
d = df[df.compile_fail == 0].merge(tk, on="strategy_id")
for rid in ["fixed_stop", "breakeven", "trailing_hwm", "time_stop", "take_profit", "partial_take", "box_shift", "cooldown", "daily_cap"]:
    d[f"has_{rid}"] = d.risk_ids.str.contains(rid).astype(int)
d["log_tau"] = np.log10(d.tau_max)
reg = {}
fe = " + ".join([f"has_{r}" for r in ["fixed_stop", "breakeven", "trailing_hwm", "time_stop", "take_profit", "partial_take", "box_shift", "cooldown", "daily_cap"]])
for s in ["open", "closed"]:
    ds = d[d.setting == s]
    if ds.empty: continue
    for name, f in {"tau_only": "action_match ~ log_tau + C(model)",
                    "tau_K": "action_match ~ log_tau + K_bits + C(model)",
                    "tau_K_FE": f"action_match ~ log_tau + K_bits + n_registers + C(model) + C(signal_id) + {fe}",
                    "tau_K_FE_controls": f"action_match ~ log_tau + K_bits + n_registers + mccabe + np.log(n_trades_bench) + C(model) + C(signal_id) + {fe}"}.items():
        r = smf.ols(f, ds).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ds.strategy_id)[0]})
        reg[f"{s}|{name}"] = {k: {"coef": float(r.params[k]), "se": float(r.bse[k]), "p": float(r.pvalues[k])}
                              for k in r.params.index if k in ("log_tau", "K_bits", "n_registers", "mccabe", "np.log(n_trades_bench)")}
        reg[f"{s}|{name}"]["n"] = int(r.nobs); reg[f"{s}|{name}"]["r2"] = float(r.rsquared)
A["regressions"] = reg
# within-primitive tau effect: parameter-only tau variation (time_stop n, cooldown n)
A["corr_tau_K"] = float(np.corrcoef(np.log10(taus), [t["K_bits"] for t in tasks])[0, 1])
(out / "analysis.json").write_text(json.dumps(A, indent=1, default=float))

# ------------------------------------------------------------- numbers.tex macros
def mac(name, v, fmt="{:.3f}"):
    return f"\\newcommand{{\\{name}}}{{{fmt.format(v) if v == v else '--'}}}"
L = [mac("NTasks", len(tasks), "{}"), mac("NRows", len(df), "{}"), mac("CorrTauK", A["corr_tau_K"], "{:.2f}")]
key = lambda m: "".join(ch for ch in m.title() if ch.isalpha())
for r in A["table1"]:
    k = key(r["model"]) + r["setting"].title()
    for col in ["CompileOK", "ActionMatch", "TradeF1", "silent", "exact"]:
        L.append(mac(f"{col.replace('silent','Silent').replace('exact','Exact')}{k}", r[col]))
    L.append(mac(f"ESmed{k}", r["ES_med"], "{:.0f}"))
    if r["setting"] == "closed": L.append(mac(f"SpecMatch{k}", r["SpecMatch"]))
if A["fig2"]["all"]["frac_spec1_am_lt_0_9"] is not None:
    L.append(mac("SpecOneBehaviourWrong", A["fig2"]["all"]["frac_spec1_am_lt_0_9"]))
    L.append(mac("NSpecOne", A["fig2"]["all"]["n_spec1"], "{}"))
for k, v in reg.items():
    if "log_tau" in v:
        n = "Reg" + "".join(ch for ch in k.title() if ch.isalpha())
        L += [mac(n + "Tau", v["log_tau"]["coef"]), mac(n + "TauSE", v["log_tau"]["se"]), mac(n + "TauP", v["log_tau"]["p"], "{:.2g}")]
        if "K_bits" in v: L.append(mac(n + "K", v["K_bits"]["coef"], "{:.4f}"))
from minteval.primitives import by_family
for fam in ["signal", "filter", "sizing", "risk"]:
    L.append(mac(f"NBlocks{fam.title()}", len(by_family(fam)), "{}"))
sel = json.load(open("results/tasks/selection_report.json"))
L += [mac("NPool", sel["pool_size"], "{}"), mac("NPoolKept", sel["after_filter"], "{}"),
      mac("NFewTrades", sel["filtered_out"]["few_trades"], "{}"), mac("NAlwaysFlat", sel["filtered_out"]["always_flat"], "{}")]
for b in sel["bins"]:
    L.append(mac(f"RMedBin{'ABCDE'[b['tau_bin']]}", b["R_med"], "{:.2f}"))
PR = [json.loads(l) for l in open("results/tasks/prompts.jsonl")]
rec = np.array([np.nan if p.get("recovery") is None else p["recovery"] for p in PR], float)
L += [mac("NValidPrompts", sum(p["valid"] for p in PR), "{}"), mac("NRoundTripOK", int((rec == 1).sum()), "{}"),
      mac("NRoundTripNA", int(np.isnan(rec).sum()), "{}"),
      mac("MedWords", float(np.median([p["prompt_len"] for p in PR])), "{:.0f}"),
      mac("MedJargon", float(np.median([p["n_jargon"] for p in PR])), "{:.0f}"),
      mac("FirstTryValid", float(np.mean([p["n_attempts"] == 1 and p["valid"] for p in PR])), "{:.2f}")]
(out / "numbers.tex").write_text("% AUTO-GENERATED by scripts/analyze.py from " + csv + " -- do not edit\n" + "\n".join(L) + "\n")
print(T.round(3).to_string()); print(json.dumps(reg, indent=0)[:3000])
