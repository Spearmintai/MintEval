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
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "figs/fig1_tau.pdf"); fig.savefig(out / "figs/fig1_tau.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------- Fig 2: SpecMatch vs ActionMatch
g = df[(df.setting == "closed") & (df.compile_fail == 0)]
fig, ax = plt.subplots(figsize=(3.4, 2.7))
levels = [0.0, 0.25, 0.5, 0.75, 1.0]
w = 0.18
ax.add_patch(plt.Rectangle((4 - 0.45, 0), 0.9, 0.9, color="#e34948", alpha=0.08, lw=0, zorder=0))
for i, m in enumerate(models):
    c, mk = style[m]
    data = [g[(g.model == m) & (g.spec_match == lv)].action_match.values for lv in levels]
    pos = [j + (i - (len(models) - 1) / 2) * w for j in range(len(levels))]
    keep = [(p_, d_) for p_, d_ in zip(pos, data) if len(d_) >= 5]
    if not keep: continue
    bp = ax.boxplot([d_ for _, d_ in keep], positions=[p_ for p_, _ in keep], widths=w * 0.85, patch_artist=True,
                    showfliers=False, medianprops={"color": "white", "lw": 1.2}, whiskerprops={"color": c, "lw": 1},
                    capprops={"color": c, "lw": 1})
    for patch in bp["boxes"]:
        patch.set_facecolor(c); patch.set_edgecolor(c)
    ax.plot([], [], color=c, marker="s", ls="", ms=5, label=m)
ax.axhline(0.9, color=MUTED, lw=0.8, ls="--")
ax.set_xticks(range(len(levels))); ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
ax.set_xlabel("SpecMatch (fraction of 4 slots exact)"); ax.set_ylabel("ActionMatch")
ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, 4.5)
fig.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(out / "figs/fig2_spec_vs_action.pdf"); fig.savefig(out / "figs/fig2_spec_vs_action.png", dpi=200); plt.close(fig)
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
fig.tight_layout(); fig.savefig(out / "figs/fig3_tau_hist.pdf"); fig.savefig(out / "figs/fig3_tau_hist.png", dpi=200); plt.close(fig)

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
KEYS = {"gpt-5.4-mini": "Gpt", "claude-haiku": "Haiku", "qwen2.5-coder-32b": "QwenL", "qwen2.5-coder-7b": "QwenS",
        "claude-opus-5.5": "Opus", "qwen3.8-max": "QMax", "deepseek-v4-pro": "DSPro"}
FRONTIER = ["claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro"]
key = lambda m: KEYS.get(m) or "".join(ch for ch in m.title() if ch.isalpha())
assert len({key(m) for m in models}) == len(models), "macro key collision"
models = [m for m in models if m not in FRONTIER]   # main tables: full-800 models only
for r in A["table1"]:
    k = key(r["model"]) + r["setting"].title()
    for col in ["CompileOK", "ActionMatch", "TradeF1", "silent", "exact"]:
        L.append(mac(f"{col.replace('silent','Silent').replace('exact','Exact').replace('TradeF1','TradeFone')}{k}", r[col]))
    L.append(mac(f"ESmed{k}", r["ES_med"], "{:.0f}"))
    if r["setting"] == "closed": L.append(mac(f"SpecMatch{k}", r["SpecMatch"]))
if A["fig2"]["all"]["frac_spec1_am_lt_0_9"] is not None:
    L.append(mac("SpecOneBehaviourWrong", A["fig2"]["all"]["frac_spec1_am_lt_0_9"]))
    L.append(mac("NSpecOne", A["fig2"]["all"]["n_spec1"], "{}"))
for m in models:
    f2 = A["fig2"].get(m)
    if f2 and f2["frac_spec1_am_lt_0_9"] is not None:
        L += [mac(f"SpecOneWrong{key(m)}", f2["frac_spec1_am_lt_0_9"]), mac(f"NSpecOne{key(m)}", f2["n_spec1"], "{}")]
    for s_ in ["open", "closed"]:
        st = A["fig1"].get(f"{m}|{s_}")
        if st:
            L += [mac(f"AMQone{key(m)}{s_.title()}", st[0][0]), mac(f"AMQfive{key(m)}{s_.title()}", st[4][0])]
T["silent_n"] = 0
L.append(mac("SilentMin", T.silent.min()))
L.append(mac("SilentMax", T.silent.max()))
L.append(mac("BestAMOpen", T[T.setting == "open"].ActionMatch.max()))
L.append(mac("BestAMClosed", T[T.setting == "closed"].ActionMatch.max()))
L.append(mac("BestExactOpen", T[T.setting == "open"].exact.max()))
import os
if os.path.exists("results/sensitivity_state_summary.csv"):
    S = pd.read_csv("results/sensitivity_state_summary.csv")
    for _, r in S.iterrows():
        k = key(r.model) + r.setting.title()
        L += [mac(f"CompileLenient{k}", r.CompileOK_lenient), mac(f"AMLenient{k}", r.AM_lenient),
              mac(f"NStateFail{k}", r.n_state_type_fail, "{}")]
# silent-failure threshold sensitivity (share of all tasks: compiled and ActionMatch < thr)
SIL = {}
for thr in (0.8, 0.9, 0.95, 0.99):
    for (m, s_), g in df[df.model.isin(models)].groupby(["model", "setting"]):
        SIL[(thr, m, s_)] = float(((g.compile_fail == 0) & (g.action_match < thr)).mean())
A["silent_thresholds"] = {f"{t}|{m}|{s_}": v for (t, m, s_), v in SIL.items()}
for thr, nm in ((0.8, "Eighty"), (0.95, "NinetyFive"), (0.99, "NinetyNine")):
    vals = [v for (t, m, s_), v in SIL.items() if t == thr]
    L += [mac(f"Silent{nm}Min", min(vals)), mac(f"Silent{nm}Max", max(vals))]
L += [mac("SilentGptOpenEighty", SIL[(0.8, "gpt-5.4-mini", "open")]),
      mac("SilentGptOpenNinetyNine", SIL[(0.99, "gpt-5.4-mini", "open")])]
if os.path.exists("results/returns_check.json"):
    RC = json.load(open("results/returns_check.json"))
    L += [mac("RNetMed", 100 * RC["R_net_median"], "{:.1f}"), mac("RGrossMed", 100 * RC["R_gross_median"], "{:.1f}"),
          mac("ShareNetPos", 100 * RC["share_net_pos"], "{:.1f}"), mac("ShareGrossPos", 100 * RC["share_gross_pos"], "{:.1f}"),
          mac("RegOpenTauGivenR", RC["reg_open"]["log_tau"]["coef"]), mac("RegClosedTauGivenR", RC["reg_closed"]["log_tau"]["coef"]),
          mac("RegOpenRonly", RC["reg_open_R_only"]["coef"], "{:.2f}"), mac("RegOpenRonlyP", RC["reg_open_R_only"]["p"], "{:.1g}")]
# frontier subset (200 tasks, 40 per tau bin, open setting) incl. Opus
if os.path.exists("results/minteval_v0_frontier.csv"):
    FR = pd.read_csv("results/minteval_v0_frontier.csv"); ids = set(FR.strategy_id)
    SUB = pd.concat([FR, df[(df.setting == "open") & df.strategy_id.isin(ids)]])
    for m_, g in SUB.groupby("model"):
        ok = g[g.compile_fail == 0]; k = "Sub" + key(m_)
        L += [mac(f"CompileOK{k}", 1 - g.compile_fail.mean()), mac(f"AM{k}", ok.action_match.mean()),
              mac(f"Exact{k}", (ok.action_match == 1).sum() / len(g)),
              mac(f"Silent{k}", ((g.compile_fail == 0) & (g.action_match < 0.9)).mean())]
        bt = ok.groupby("tau_bin").action_match.mean()
        L += [mac(f"AMQone{k}", bt.get(0, float("nan"))), mac(f"AMQfive{k}", bt.get(4, float("nan")))]
    L.append(mac("NSub", len(ids), "{}"))
    rows = []
    for m_ in FRONTIER + models:
        g = SUB[SUB.model == m_]
        if g.empty: continue
        ok = g[g.compile_fail == 0]
        rows.append(f"{m_} & {1-g.compile_fail.mean():.3f} & {ok.action_match.mean():.3f} & {(ok.action_match==1).sum()/len(g):.3f} & "
                    f"{((g.compile_fail==0)&(g.action_match<0.9)).mean():.3f} & "
                    + " & ".join(f"{v:.2f}" for v in ok.groupby('tau_bin').action_match.mean().reindex(range(5)).values) + r" \\")
    (out / "table2.tex").write_text("\n".join([r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
        r"Model & CompileOK & ActionMatch & Exact & Silent & $\tau$1 & $\tau$2 & $\tau$3 & $\tau$4 & $\tau$5 \\", r"\midrule"]
        + rows + [r"\bottomrule", r"\end{tabular}"]) + "\n")
if os.path.exists("results/judge_vs_behaviour.json"):
    JV = json.load(open("results/judge_vs_behaviour.json"))
    for m_, v in JV["by_model"].items():
        k = KEYS.get(m_, m_)
        L += [mac(f"JudgeN{k}", v["n"], "{}"), mac(f"JudgePass{k}", v["judge_pass"]),
              mac(f"JudgePassBad{k}", v["pass_and_bad"], "{}"), mac(f"JudgePassOk{k}", v["pass_and_ok"], "{}"),
              mac(f"JudgeFailAll{k}", v["fail_and_ok"] + v["fail_and_bad"], "{}"),
              mac(f"JudgeSilentShare{k}", v["share_silent_among_pass"]), mac(f"JudgeAMPass{k}", v["AM_mean_among_pass"])]
    L.append(mac("JudgeMissing", JV["api_errors"], "{}"))
    NC = json.load(open("results/judge_negative_control.json"))
    nc = [r for r in NC if r["judge_pass"] is not None]
    L += [mac("NegCtlN", len(nc), "{}"), mac("NegCtlRejected", sum(not r["judge_pass"] for r in nc), "{}")]
# ---- post-v0 analyses (each from its own results/*.json; skipped if absent)
def J(path):
    return json.load(open(path)) if os.path.exists(path) else None
ES = J("results/es_regression.json")
if ES:
    for kk, nm in [("full800|gpt-5.4-mini|open", "GptOpen"), ("full800|claude-haiku|open", "HaikuOpen"),
                   ("full800|gpt-5.4-mini|closed", "GptClosed"), ("full800|claude-haiku|closed", "HaikuClosed"),
                   ("frontier200|claude-opus-5.5|open", "OpusSub")]:
        c = ES[kk]["w99_ctrl"]
        L += [mac(f"ESvol{nm}", c["vol"]["b"], "{:.0f}"), mac(f"ESvolP{nm}", c["vol"]["p"], "{:.2g}"),
              mac(f"EStau{nm}", c["ltau"]["b"], "{:.0f}"), mac(f"EStauP{nm}", c["ltau"]["p"], "{:.2g}"),
              mac(f"EStxv{nm}", c["ltau:vol"]["b"], "{:.0f}"), mac(f"EStxvP{nm}", c["ltau:vol"]["p"], "{:.2g}")]
    ps = [v["w99_ctrl"]["ltau:vol"]["p"] for v in ES.values()]
    L += [mac("ESnCells", len(ps), "{}"), mac("ESnTxvSig", sum(p < 0.05 for p in ps), "{}")]
ST = J("results/opus_structure.json")
if ST:
    F = {r["feature"]: r for r in ST["claude-opus-5.5"]["features"]}
    for feat, nm in [("stateful_signal", "Retest"), ("htf_filter", "Htf"), ("has_trailing_hwm", "Trail"), ("long_short", "LS"),
                     ("multi_stop_layers", "MultiStop"), ("has_box_shift", "Box"), ("has_cooldown", "Cool")]:
        r = F[feat]
        L += [mac(f"OpusSil{nm}With", r["silent_with"]), mac(f"OpusSil{nm}Without", r["silent_without"]),
              mac(f"OpusSil{nm}P", r["fisher_p"], "{:.1g}"), mac(f"OpusSil{nm}N", r["n_with"], "{}")]
AC = J("results/ambiguity_check.json")
if AC and "claude-opus-5.5" in AC:
    a = AC["claude-opus-5.5"]
    L += [mac("AmbNoCloseN", a["noclose"]["n"], "{}"), mac("AmbNoCloseSil", a["noclose"]["silent"], "{}"),
          mac("AmbCloseN", a["close"]["n"], "{}"), mac("AmbCloseSil", a["close"]["silent"], "{}")]
TC = J("results/tau_ci_frontier.json")
if TC:
    for m_, v in TC.items():
        k = {"claude-opus-5.5": "Opus", "qwen3.8-max": "QMax", "deepseek-v4-pro": "DSPro", "gpt-5.4-mini": "Gpt", "claude-haiku": "Haiku"}[m_]
        L += [mac(f"Slope{k}", v["slope"], "{:+.3f}"), mac(f"SlopeLo{k}", v["slope_ci"][0], "{:+.3f}"),
              mac(f"SlopeHi{k}", v["slope_ci"][1], "{:+.3f}"), mac(f"SlopeP{k}", v["slope_p"], "{:.2g}")]
if os.path.exists("results/mutants/mutant_scores.csv"):
    MS = pd.read_csv("results/mutants/mutant_scores.csv")
    for t_, g in MS.groupby("type"):
        k = "".join(w.title() for w in t_.split("_"))
        L += [mac(f"Mut{k}AM", g.action_match.mean()), mac(f"Mut{k}Bad", (g.action_match < 0.9).mean()),
              mac(f"Mut{k}ES", g.abs_es.median(), "{:.0f}")]
if os.path.exists("results/judges/kimi_k2.5.csv"):
    KJ = pd.read_csv("results/judges/kimi_k2.5.csv"); KJ = KJ[KJ.judge_pass.notna()]
    KJ["jp"] = KJ.judge_pass.astype(bool); KJ["bad"] = KJ.action_match < 0.9
    for (kind, who), g in KJ.groupby(["kind", "who"]):
        k = ("Mut" + "".join(w.title() for w in who.split("_"))) if kind == "mutant" else \
            {"claude-opus-5.5": "Opus", "qwen3.8-max": "QMax", "deepseek-v4-pro": "DSPro", "gpt-5.4-mini": "Gpt",
             "claude-haiku": "Haiku", "qwen2.5-coder-32b": "QwenL", "qwen2.5-coder-7b": "QwenS"}[who]
        L += [mac(f"Kimi{k}Pass", g.jp.mean()), mac(f"Kimi{k}PassBad", int((g.jp & g.bad).sum()), "{}"),
              mac(f"Kimi{k}Bad", int(g.bad.sum()), "{}"), mac(f"Kimi{k}N", len(g), "{}")]
    mu = KJ[KJ.kind == "mutant"]
    L += [mac("KimiMutPassBad", int((mu.jp & mu.bad).sum()), "{}"), mac("KimiMutBad", int(mu.bad.sum()), "{}")]
MAR = J("results/multiasset/report.json")
if MAR:
    AK = {"BTCUSDT_2022_2023": "Btc", "ETHUSDT_2018_2023": "Eth", "SPY_CFD_2018_2023": "Spy", "NDX_CFD_2018_2023": "Ndx"}
    for a_, r_ in MAR.items():
        k = AK[a_]
        L += [mac(f"MA{k}Survive", r_["survive_filter"], "{}"), mac(f"MA{k}Rho", r_["spearman_logtau_R"], "{:.2f}"),
              mac(f"MA{k}RhoP", r_["spearman_p"], "{:.1g}"), mac(f"MA{k}RMed", 100 * r_["R_bench_median_survivors"], "{:.1f}"),
              mac(f"MA{k}Profitable", r_["share_profitable"], "{:.2f}")]
        for grp, gk in (("low_cost", "Low"), ("frontier", "Front")):
            rr = r_["regressions"].get(f"{grp}|tau_R")
            if rr:
                L += [mac(f"MA{k}{gk}Tau", rr["ltau"]["b"], "{:+.3f}"), mac(f"MA{k}{gk}TauP", rr["ltau"]["p"], "{:.1g}")]
        for m_, v_ in r_["by_model"].items():
            L += [mac(f"MA{k}AM{key(m_)}", v_["AM"]), mac(f"MA{k}Silent{key(m_)}", v_["silent"])]
        for blk, v_ in (r_.get("inert_share_by_block") or {}).items():
            L.append(mac(f"MA{k}Inert{''.join(w.title() for w in blk.split('_'))}", v_, "{:.2f}"))
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
bad_names = [l for l in L if any(ch.isdigit() for ch in l.split("}")[0])]
assert not bad_names, f"macro names with digits: {bad_names[:3]}"
(out / "numbers.tex").write_text("% AUTO-GENERATED by scripts/analyze.py from " + csv + " -- do not edit\n" + "\n".join(L) + "\n")
print(T.round(3).to_string()); print(json.dumps(reg, indent=0)[:3000])
