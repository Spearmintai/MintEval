"""FRL minor revision (FRL-D-26-05997), Reviewer 1 comments 1-3. Everything is computed from
results/multiasset/minteval_multiasset.csv and the task metadata; nothing is typed by hand.

Conventions (identical to scripts/frl_numbers.py and to the submitted manuscript):
- IS = R_ref - R_llm in bp of initial equity (eq. 1 of the paper; positive = generated program earns less).
- Analysis sample D: references passing the per-instrument filter and compiled generated runs (N = 11,029).
- |IS| winsorised 1/99 within instrument for regressions; log10 tau centred; sigma standardised within instrument;
  SEs clustered by strategy.
- Mean holding duration = bars in position / round trips of the reference (as in the paper's exposure control).

Outputs (paper_frl/):
  numbers_frl_rev.tex            all macros (prefix Rev)
  table_frl_unamb.tex            Table 3 col. 4 spec, with/without exposure controls, full vs unambiguous vs strict
  table_frl_unamb_table2.tex     Table 2 restricted to unambiguous tasks
  table_frl_amb_silent.tex       silent-failure share on ambiguous vs unambiguous tasks, attributable fraction
  table_frl_riskadj.tex          annualised, risk-adjusted and turnover-adjusted shortfall by model
  table_frl_vif.tex              correlations and VIFs of log tau, log holding duration, share in position
  table_frl_reg_rev.tex          complete exposure-controlled regression (6 columns)
  table_frl_matched_dur.tex      duration-matched complexity test (register-count median split within quintile)
  table_frl_matched_dur_alt.tex  same, alternative definition (single risk layer and <= 5 registers)
  table_frl_matched_dur_lit.tex  same, literal definition (<= 2 registers and single risk layer; thin)
  results/frl/expert_audit_100.csv   100-task stratified audit sample; rater columns left BLANK for the raters
"""
import sys, os, json, re, math
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr, spearmanr, ttest_ind
from minteval.data import load_prices

OUT = "paper_frl"
os.makedirs(OUT, exist_ok=True); os.makedirs("results/frl", exist_ok=True)

cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]["assets"]
d = pd.read_csv("results/multiasset/minteval_multiasset.csv")
tasks = {json.loads(l)["strategy_id"]: json.loads(l) for l in open("results/tasks/tasks.jsonl")}
spec = {k: v["spec"] for k, v in tasks.items()}
prompts = {json.loads(l)["strategy_id"]: json.loads(l) for l in open("results/tasks/prompts.jsonl")}
f200 = [json.loads(l) for l in open("results/tasks/frontier200.jsonl")]
ids200 = {x["strategy_id"] for x in f200}

RISK = ["fixed_stop", "breakeven", "trailing_hwm", "time_stop", "take_profit", "partial_take", "box_shift", "cooldown", "daily_cap"]
d["sizing_id"] = d.strategy_id.map(lambda s: spec[s]["sizing"]["id"])
d["filter_id"] = d.strategy_id.map(lambda s: spec[s]["filter"]["id"])
d["direction"] = d.strategy_id.map(lambda s: spec[s]["direction"])
for r in RISK:
    d[f"has_{r}"] = d.strategy_id.map(lambda s: int(any(x["id"] == r for x in spec[s]["risk"])))
d["is_bp"] = (d.R_bench - d.R_llm) * 1e4            # IS = R_ref - R_llm (paper eq. 1)
ok_rows = d.compile_fail == 0
assert np.allclose(d.loc[ok_rows, "is_bp"].abs(), d.loc[ok_rows, "abs_es"], atol=0.01)

FRONTIER = ["claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro"]
LOWCOST = ["gpt-5.4-mini", "claude-haiku", "qwen2.5-coder-32b", "qwen2.5-coder-7b"]
MK = {"claude-opus-5.5": "Opus", "qwen3.8-max": "QMax", "deepseek-v4-pro": "DSPro", "gpt-5.4-mini": "Gpt",
      "claude-haiku": "Haiku", "qwen2.5-coder-32b": "QwenL", "qwen2.5-coder-7b": "QwenS"}
MNAME = {"claude-opus-5.5": "Claude Opus 5.5", "qwen3.8-max": "Qwen3.8-Max", "deepseek-v4-pro": "DeepSeek-V4-Pro",
         "gpt-5.4-mini": "GPT-5.4-mini", "claude-haiku": "Claude Haiku", "qwen2.5-coder-32b": "Qwen2.5-Coder-32B",
         "qwen2.5-coder-7b": "Qwen2.5-Coder-7B"}
AK = {"BTCUSDT_2022_2023": "Btc", "ETHUSDT_2018_2023": "Eth", "SPY_CFD_2018_2023": "Spy", "NDX_CFD_2018_2023": "Ndx"}
nbars, YEARS = {}, {}
for a in AK:
    pr_ = load_prices(MA[a]["price_file"]); nbars[a] = len(pr_["close"])
    t0, t1 = pd.to_datetime(pr_["open_time"][[0, -1]], unit="ms", utc=True); YEARS[a] = (t1 - t0).days / 365.25

M = []
def mac(name, val, fmt="{:.3f}"):
    assert not any(c.isdigit() for c in name), name
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        M.append(f"\\newcommand{{\\{name}}}{{\\todo{{{name}}}}}"); return
    M.append(f"\\newcommand{{\\{name}}}{{{fmt.format(val)}}}")
num = lambda x: "{:,}".format(int(x)).replace(",", "{,}")
def grp_sample(D_, m):
    return D_[(D_.model == m) & (D_.strategy_id.isin(ids200) if m in FRONTIER else (D_.subset == "full800"))]
for a, k in AK.items():
    mac(f"RevYears{k}", YEARS[a], "{:.2f}")

# =====================================================================================================
# COMMENT 1. Ambiguity: flag, count, audit sample, unambiguous-subset re-estimation
# =====================================================================================================
# (a) round-trip reader failed or unverifiable (the paper's 780/800)
rt_fail = {s for s, p in prompts.items() if p.get("recovery") != 1.0 or not p.get("valid", True)}
# (b) breakout-then-retest tasks whose message does not say the breakout is measured on the CLOSE
# (c) higher-timeframe filter tasks whose message does not say the filter reads the COMPLETED candle
RE_CLOSE = re.compile(r"clos\w*[^.]{0,40}(above|over|beyond|past|below|under|through)|(above|below)[^.]{0,30}clos")
RE_DONE = re.compile(r"completed|last closed|closed\s+(1h|4h|60|240|hour|htf|higher)|finished\s+(1h|4h|60|240|hour|candle|bar)|(1h|4h|60|240)[^.]{0,40}(completed|closed)")
bp_all = {s for s in spec if spec[s]["signal"]["id"] == "breakout_pullback"}
htf_all = {s for s in spec if spec[s]["filter"]["id"] == "htf_trend"}
bp_amb = {s for s in bp_all if not RE_CLOSE.search(prompts[s]["q"].lower())}
htf_amb = {s for s in htf_all if not RE_DONE.search(prompts[s]["q"].lower())}
amb_ids = rt_fail | bp_amb | htf_amb
unamb_ids = set(tasks) - amb_ids
strict_amb_ids = rt_fail | bp_all | htf_all            # drop every retest / HTF task regardless of wording
strict_unamb_ids = set(tasks) - strict_amb_ids

mac("RevAmbCount", len(amb_ids), "{}"); mac("RevAmbShare", 100 * len(amb_ids) / len(tasks), "{:.1f}")
mac("RevAmbRtCount", len(rt_fail), "{}")
mac("RevAmbBpCount", len(bp_amb), "{}"); mac("RevAmbBpTotal", len(bp_all), "{}")
mac("RevAmbHtfCount", len(htf_amb), "{}"); mac("RevAmbHtfTotal", len(htf_all), "{}")
mac("RevAmbSumFlags", len(rt_fail) + len(bp_amb) + len(htf_amb), "{}")
mac("RevUnambCount", len(unamb_ids), "{}"); mac("RevUnambShare", 100 * len(unamb_ids) / len(tasks), "{:.1f}")
mac("RevStrictCount", len(strict_unamb_ids), "{}"); mac("RevStrictShare", 100 * len(strict_unamb_ids) / len(tasks), "{:.1f}")
mac("RevStrictDropped", len(strict_amb_ids), "{}")

# audit sample: 20 tasks per tau quintile from the 200-task core set; rater columns are left EMPTY on purpose
rows = []
for x in f200:
    s = x["strategy_id"]; t = tasks[s]
    rows.append({"strategy_id": s, "tau_bin": t["tau_bin"], "tau_max": t["tau_max"], "n_registers": t["n_registers"],
                 "signal_id": spec[s]["signal"]["id"], "filter_id": spec[s]["filter"]["id"], "sizing_id": spec[s]["sizing"]["id"],
                 "risk_ids": "+".join(r["id"] for r in spec[s]["risk"]), "roundtrip_ok": int(s not in rt_fail),
                 "auto_flag_retest_no_close": int(s in bp_amb), "auto_flag_htf_no_completed": int(s in htf_amb),
                 "auto_flag_ambiguous": int(s in amb_ids), "prompt": x["q"], "spec_desc": x["spec_desc"],
                 "rater1_unambiguous": "", "rater1_notes": "", "rater2_unambiguous": "", "rater2_notes": ""})
F = pd.DataFrame(rows)
sub100 = F.groupby("tau_bin", group_keys=False).apply(lambda g: g.sample(20, random_state=20261002)).sort_values("strategy_id")
sub100.to_csv("results/frl/expert_audit_100.csv", index=False)
mac("RevAuditN", len(sub100), "{}"); mac("RevAuditAutoFlagged", int(sub100.auto_flag_ambiguous.sum()), "{}")

d["ambiguous"] = d.strategy_id.isin(amb_ids); d["strict_ambiguous"] = d.strategy_id.isin(strict_amb_ids)

# analysis sample, exactly as frl_numbers.py
D = d[d.pass_filter & (d.compile_fail == 0)].copy()
D["wes"] = D.groupby("asset").abs_es.transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
u = D.drop_duplicates(["asset", "strategy_id"])
tau_mean = float(np.log10(u.tau_max).mean())
D["ltau"] = np.log10(D.tau_max) - tau_mean
vs = u.groupby("asset").realized_vol.agg(["mean", "std"])
D["sig"] = (D.realized_vol - D.asset.map(vs["mean"])) / D.asset.map(vs["std"])
D["ltrades"] = np.log(D.n_trades_bench)
D["hold_dur"] = D.frac_in_pos * D.asset.map(nbars) / D.n_trades_bench
D["log_hold"] = np.log(D.hold_dur)
D["am_pp"] = 100 * D.action_match
assert len(D) == 11029, len(D)

CTRL = "K_bits + ltrades + mccabe"
FE = "C(signal_id) + C(sizing_id) + C(filter_id) + C(direction) + " + " + ".join(f"has_{r}" for r in RISK)
BASE = f"wes ~ ltau * sig + {CTRL} + {FE} + C(model) + C(asset)"        # Table 3 col. 4 (no R_ref)
EXPO = BASE + " + log_hold + frac_in_pos"                               # Table 4 last rows
cl = lambda data: {"groups": pd.factorize(data.strategy_id)[0]}
ols = lambda f, data: smf.ols(f, data).fit(cov_type="cluster", cov_kwds=cl(data))

D_un = D[~D.ambiguous]; D_st = D[~D.strict_ambiguous]
SAMPLES = {"Full": (BASE, D), "Unamb": (BASE, D_un), "Strict": (BASE, D_st),
           "FullExpo": (EXPO, D), "UnambExpo": (EXPO, D_un), "StrictExpo": (EXPO, D_st),
           "UnambLow": (BASE, D_un[D_un.model.isin(LOWCOST)]),
           "UnambFront": (BASE, D_un[D_un.model.isin(FRONTIER) & D_un.strategy_id.isin(ids200)]),
           "StrictLow": (BASE, D_st[D_st.model.isin(LOWCOST)])}
R = {k: ols(f, data) for k, (f, data) in SAMPLES.items()}
TERMS = [("ltau", "Tau"), ("sig", "Sig"), ("ltau:sig", "Int"), ("log_hold", "Dur"), ("frac_in_pos", "Pos"),
         ("K_bits", "K"), ("ltrades", "Ltr"), ("mccabe", "Mcc")]
def reg_macros(prefix, r, data):
    for term, tk in TERMS:
        if term in r.params:
            mac(f"{prefix}{tk}", r.params[term], "{:.1f}"); mac(f"{prefix}{tk}T", r.tvalues[term], "{:.2f}")
    mac(f"{prefix}N", r.nobs, "{:.0f}"); mac(f"{prefix}AdjR", r.rsquared_adj, "{:.3f}")
    mac(f"{prefix}Clust", data.strategy_id.nunique(), "{}")
for k, (f, data) in SAMPLES.items():
    reg_macros(f"Rev{k}", R[k], data)

LAB = [("ltau", r"$\log_{10}\tau_{\max}$ (centred)"), ("sig", r"Volatility $\sigma$"), ("ltau:sig", r"$\log_{10}\tau_{\max}\times\sigma$"),
       ("log_hold", r"log mean holding duration"), ("frac_in_pos", r"Share of bars in position")]
def reg_table(cols, header_rows, fname, terms=LAB, extra_rows=()):
    L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule"] + header_rows + [r"\midrule"]
    for term, lab in terms:
        L.append(lab + " & " + " & ".join(f"{r.params[term]:.1f}" if term in r.params else "--" for r, _ in cols) + r" \\")
        L.append(" & " + " & ".join(f"({r.tvalues[term]:.2f})" if term in r.params else "" for r, _ in cols) + r" \\")
    L += [r"\midrule"] + list(extra_rows)
    L.append("$N$ & " + " & ".join(num(r.nobs) for r, _ in cols) + r" \\")
    L.append("Strategies (clusters) & " + " & ".join(str(dd.strategy_id.nunique()) for _, dd in cols) + r" \\")
    L.append(r"Adj.\ $R^2$ & " + " & ".join(f"{r.rsquared_adj:.3f}" for r, _ in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/{fname}", "w").write("\n".join(L) + "\n")

reg_table([(R["Full"], D), (R["Unamb"], D_un), (R["Strict"], D_st), (R["FullExpo"], D), (R["UnambExpo"], D_un), (R["StrictExpo"], D_st)],
          [r" & \multicolumn{3}{c}{Table 3, column (4) specification} & \multicolumn{3}{c}{\ldots\ with exposure controls} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
           r" & All tasks & Unambiguous & Strict & All tasks & Unambiguous & Strict \\"],
          "table_frl_unamb.tex",
          extra_rows=[r"Block, model and instrument FE & Yes & Yes & Yes & Yes & Yes & Yes \\"])

# Table 2 on unambiguous tasks (same construction as frl_numbers.t1: all surviving references, not only compiled)
P = d[d.pass_filter]
def t2(sample, tag):
    for m in FRONTIER + LOWCOST:
        g = grp_sample(sample, m); okk = g[g.compile_fail == 0]; es = okk.is_bp
        tval = es.mean() / (es.std(ddof=1) / math.sqrt(len(es)))
        k = f"Rev{tag}{MK[m]}"
        mac(f"{k}N", len(g), "{}"); mac(f"{k}Compile", 1 - g.compile_fail.mean())
        mac(f"{k}ISMed", okk.abs_es.median(), "{:.0f}"); mac(f"{k}ISMean", es.mean(), "{:.0f}"); mac(f"{k}ISMeanT", tval, "{:.2f}")
        mac(f"{k}AM", okk.action_match.mean()); mac(f"{k}Exact", (okk.action_match == 1).sum() / len(g))
        mac(f"{k}Silent", ((g.compile_fail == 0) & (g.action_match < 0.9)).mean())
t2(P[~P.ambiguous], "Unamb"); t2(P[P.ambiguous], "Amb"); t2(P[~P.strict_ambiguous], "Strict"); t2(P, "All")
L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
     r"Model & $N$ & Compile & med.\ $|IS|$ (bp) & mean $IS$ (bp) [$t$] & ActionMatch & Exact & Silent \\", r"\midrule",
     r"\multicolumn{8}{l}{\emph{Unambiguous tasks (\RevUnambCount{} of 800), four instruments pooled, surviving references}} \\"]
for m in FRONTIER + LOWCOST:
    k = f"RevUnamb{MK[m]}"
    L.append(f"{MNAME[m]} & \\{k}N & \\{k}Compile & \\{k}ISMed & \\{k}ISMean{{}} [\\{k}ISMeanT] & \\{k}AM & \\{k}Exact & \\{k}Silent \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_unamb_table2.tex", "w").write("\n".join(L) + "\n")

# silent-failure share on ambiguous vs unambiguous tasks and the share of silent failures attributable to ambiguity
# (population-attributable fraction: excess silent rate on ambiguous tasks x number of ambiguous tasks / all silent failures)
L = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
     r"Model & \multicolumn{2}{c}{Silent share} & \multicolumn{2}{c}{Median $|IS|$ (bp)} & Silent failures & Attributable \\",
     r" & unambig. & ambig. & unambig. & ambig. & (count) & to ambiguity \\", r"\midrule"]
tot_sil = tot_attr = 0.0
sil = lambda x: ((x.compile_fail == 0) & (x.action_match < 0.9))
for m in FRONTIER + LOWCOST:
    g = grp_sample(P, m); g_un, g_am = g[~g.ambiguous], g[g.ambiguous]
    s_un, s_am = sil(g_un).mean(), sil(g_am).mean(); n_sil = int(sil(g).sum())
    attr = max(0.0, s_am - s_un) * len(g_am) / n_sil
    tot_sil += n_sil; tot_attr += max(0.0, s_am - s_un) * len(g_am)
    k = MK[m]
    mac(f"RevAttr{k}Share", attr); mac(f"RevAttr{k}SilN", n_sil, "{}")
    mac(f"RevAttr{k}SilentUn", s_un); mac(f"RevAttr{k}SilentAm", s_am)
    med_un = g_un[g_un.compile_fail == 0].abs_es.median(); med_am = g_am[g_am.compile_fail == 0].abs_es.median()
    mac(f"RevAttr{k}MedUn", med_un, "{:.0f}"); mac(f"RevAttr{k}MedAm", med_am, "{:.0f}")
    L.append(f"{MNAME[m]} & {s_un:.3f} & {s_am:.3f} & {med_un:.0f} & {med_am:.0f} & {n_sil} & {attr:.3f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_amb_silent.tex", "w").write("\n".join(L) + "\n")
mac("RevAttrAllShare", tot_attr / tot_sil)
for tag, g in [("Low", P[P.model.isin(LOWCOST)]), ("Front", P[P.model.isin(FRONTIER) & P.strategy_id.isin(ids200)])]:
    s = sil(g); s_un, s_am = s[~g.ambiguous].mean(), s[g.ambiguous].mean()
    mac(f"RevAttr{tag}SilentUn", s_un); mac(f"RevAttr{tag}SilentAm", s_am)
    mac(f"RevAttr{tag}Share", max(0.0, s_am - s_un) * g.ambiguous.sum() / s.sum())

# =====================================================================================================
# COMMENT 2. Annualised, risk-adjusted and turnover-adjusted shortfall
# =====================================================================================================
D["years"] = D.asset.map(YEARS)
D["is_ann"] = D.is_bp / D.years; D["abs_is_ann"] = D.abs_es / D.years
D["d_sharpe"] = D.sharpe_bench - D.sharpe_llm                     # positive = generated program has the lower Sharpe
D["d_maxdd_bp"] = (D.maxdd_bench - D.maxdd_llm) * 1e4            # maxdd stored <= 0: positive = generated program's drawdown deeper
D["is_per_trade"] = D.abs_es / D.n_trades_bench
# Sharpe of programs that lose their whole equity is a numerical artefact (sentinel values to -999):
mac("RevBustRuns", int((D.R_llm <= -0.999).sum()), "{}"); mac("RevSharpeBelowFifty", int((D.sharpe_llm <= -50).sum()), "{}")
for c in ["d_sharpe", "d_maxdd_bp"]:          # winsorise the two risk-adjusted differences 1/99 within model
    D[f"w_{c}"] = D.groupby("model")[c].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))

def tstat(x): return x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
     r"Model & $N$ & med.\ $|IS|$/yr & mean $IS$/yr [$t$] & med.\ $|\Delta\mathrm{SR}|$ & mean $\Delta\mathrm{SR}$ [$t$] & med.\ $|\Delta\mathrm{MDD}|$ & mean $\Delta\mathrm{MDD}$ [$t$] & med.\ $|IS|$/trade \\",
     r" & & (bp) & (bp) & & & (bp) & (bp) & (bp) \\", r"\midrule",
     r"\multicolumn{9}{l}{\emph{Low-cost models, all tasks, four instruments pooled, compiled runs}} \\"]
FM = dict(N="{}", AnnMed="{:.0f}", AnnMean="{:.0f}", AnnMeanT="{:.2f}", AnnAbsMean="{:.0f}", DSMed="{:.2f}", DSMean="{:.2f}", DSMeanT="{:.2f}",
          DSAbsMean="{:.2f}", DSMeanRaw="{:.2f}", DDMed="{:.0f}", DDMean="{:.0f}", DDMeanT="{:.2f}", DDAbsMean="{:.0f}", TurnMed="{:.2f}", TurnMean="{:.2f}",
          SharpeRef="{:.2f}", SharpeLlm="{:.2f}", MddRef="{:.0f}", MddLlm="{:.0f}")
for i, models in enumerate([LOWCOST, FRONTIER]):
    if i: L += [r"\midrule", r"\multicolumn{9}{l}{\emph{Frontier models, 200-task subset, four instruments pooled, compiled runs}} \\"]
    for m in models:
        g = grp_sample(D, m); k = MK[m]
        v = dict(N=len(g), AnnMed=g.abs_is_ann.median(), AnnMean=g.is_ann.mean(), AnnMeanT=tstat(g.is_ann), AnnAbsMean=g.abs_is_ann.mean(),
                 DSMed=g.d_sharpe.abs().median(), DSMean=g.w_d_sharpe.mean(), DSMeanT=tstat(g.w_d_sharpe), DSAbsMean=g.w_d_sharpe.abs().mean(),
                 DSMeanRaw=g.d_sharpe.mean(),
                 DDMed=g.d_maxdd_bp.abs().median(), DDMean=g.w_d_maxdd_bp.mean(), DDMeanT=tstat(g.w_d_maxdd_bp), DDAbsMean=g.w_d_maxdd_bp.abs().mean(),
                 TurnMed=g.is_per_trade.median(), TurnMean=g.is_per_trade.mean(),
                 SharpeRef=g.sharpe_bench.median(), SharpeLlm=g.sharpe_llm.median(), MddRef=g.maxdd_bench.median() * 1e4, MddLlm=g.maxdd_llm.median() * 1e4)
        for kk, val in v.items(): mac(f"RevRisk{k}{kk}", val, FM[kk])
        L.append(f"{MNAME[m]} & {v['N']} & {v['AnnMed']:.0f} & {v['AnnMean']:.0f} [{v['AnnMeanT']:.2f}] & {v['DSMed']:.2f} & "
                 f"{v['DSMean']:.2f} [{v['DSMeanT']:.2f}] & {v['DDMed']:.0f} & {v['DDMean']:.0f} [{v['DDMeanT']:.2f}] & {v['TurnMed']:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_riskadj.tex", "w").write("\n".join(L) + "\n")

# =====================================================================================================
# COMMENT 3. Correlations / VIF, complete exposure-controlled regression, duration-matched complexity test
# =====================================================================================================
u = D.drop_duplicates(["asset", "strategy_id"]).copy()                 # one row per reference x instrument
X3 = sm.add_constant(u[["ltau", "log_hold", "frac_in_pos"]]); X2 = sm.add_constant(u[["ltau", "log_hold"]])
vif3 = {c: float(variance_inflation_factor(X3.values, i)) for i, c in enumerate(X3.columns) if c != "const"}
vif2 = {c: float(variance_inflation_factor(X2.values, i)) for i, c in enumerate(X2.columns) if c != "const"}
X3p = sm.add_constant(D[["ltau", "log_hold", "frac_in_pos"]])
vif3p = {c: float(variance_inflation_factor(X3p.values, i)) for i, c in enumerate(X3p.columns) if c != "const"}
pr = lambda a, b: pearsonr(u[a], u[b])[0]; sr = lambda a, b: spearmanr(u[a], u[b])[0]
mac("RevCorrTauDur", pr("ltau", "log_hold")); mac("RevSpearmanTauDur", sr("ltau", "log_hold"))
mac("RevCorrTauPos", pr("ltau", "frac_in_pos")); mac("RevSpearmanTauPos", sr("ltau", "frac_in_pos"))
mac("RevCorrDurPos", pr("log_hold", "frac_in_pos")); mac("RevSpearmanDurPos", sr("log_hold", "frac_in_pos"))
mac("RevVifTau", vif3["ltau"], "{:.2f}"); mac("RevVifDur", vif3["log_hold"], "{:.2f}"); mac("RevVifPos", vif3["frac_in_pos"], "{:.2f}")
mac("RevVifTwoTau", vif2["ltau"], "{:.2f}"); mac("RevVifPooledTau", vif3p["ltau"], "{:.2f}")
mac("RevVifUnitsN", len(u), "{}")
r_aux = sm.OLS(u.ltau, X3[["const", "log_hold", "frac_in_pos"]]).fit(); mac("RevTauOnExpoR", r_aux.rsquared)
L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
     r" & \multicolumn{3}{c}{Pearson $r$ (Spearman $\rho$ below)} & VIF & VIF \\", r"\cmidrule(lr){2-4}",
     r" & $\log_{10}\tau_{\max}$ & log hold.\ dur. & share in pos. & (3 vars) & ($\tau$, dur.) \\", r"\midrule"]
V = {"ltau": (r"$\log_{10}\tau_{\max}$", vif3["ltau"], vif2["ltau"]), "log_hold": (r"log mean holding duration", vif3["log_hold"], vif2["log_hold"]),
     "frac_in_pos": (r"Share of bars in position", vif3["frac_in_pos"], None)}
for a, (lab, v3, v2) in V.items():
    L.append(lab + " & " + " & ".join("1" if a == b else f"{pr(a, b):.3f}" for b in V) + f" & {v3:.2f} & " + (f"{v2:.2f}" if v2 else "--") + r" \\")
    L.append(" & " + " & ".join("" if a == b else f"({sr(a, b):.3f})" for b in V) + r" & & \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_vif.tex", "w").write("\n".join(L) + "\n")

# complete exposure-controlled regression
Cdata = {"TauOnly": D, "Base": D, "DurOnly": D, "Expo": D, "ExpoLow": D[D.model.isin(LOWCOST)],
         "ExpoFront": D[D.model.isin(FRONTIER) & D.strategy_id.isin(ids200)]}
CREG = {"TauOnly": ols("wes ~ ltau", D), "Base": R["Full"], "DurOnly": ols(BASE + " + log_hold", D), "Expo": R["FullExpo"],
     "ExpoLow": ols(EXPO, Cdata["ExpoLow"]), "ExpoFront": ols(EXPO, Cdata["ExpoFront"])}
for k in CREG: reg_macros(f"RevReg{k}", CREG[k], Cdata[k])
WR = BASE.replace(CTRL, CTRL + " + R_bench")          # Appendix C variant with the reference return (585.2, t = 4.33)
reg_macros("RevRegWithRBase", ols(WR, D), D); reg_macros("RevRegWithRExpo", ols(WR + " + log_hold + frac_in_pos", D), D)
reg_table([(CREG[k], Cdata[k]) for k in ["TauOnly", "Base", "DurOnly", "Expo", "ExpoLow", "ExpoFront"]],
          [r" & (1) & (2) & (3) & (4) & (5) & (6) \\", r" & $\tau$ only & Table 3 (4) & $+$ duration & $+$ both & low-cost & frontier \\"],
          "table_frl_reg_rev.tex",
          terms=LAB + [("K_bits", "Description length $K$ (bits)"), ("ltrades", "log reference trades"), ("mccabe", "McCabe complexity")],
          extra_rows=[r"Block FE & No & Yes & Yes & Yes & Yes & Yes \\", r"Model and instrument FE & No & Yes & Yes & Yes & Yes & Yes \\"])

# duration-matched complexity test
u["dur_q"] = pd.qcut(u.hold_dur, 5, labels=[1, 2, 3, 4, 5])
qb = u.groupby("dur_q", observed=False).hold_dur.agg(["min", "max", "median"])
qmed_reg = u.groupby("dur_q", observed=False).n_registers.median()
D = D.merge(u[["asset", "strategy_id", "dur_q"]], on=["asset", "strategy_id"])
D["low_med"] = D.n_registers <= D.dur_q.map(qmed_reg).astype(float)            # primary: register count <= quintile median
D["low_alt"] = (D.n_registers <= 5) & (D.n_risk == 1)                           # alternative: one risk layer, <= 5 registers
D["low_lit"] = (D.n_registers <= 2) & (D.n_risk == 1)                           # literal: <= 2 registers, one risk layer
mac("RevRegCorrTau", pearsonr(u.n_registers, u.ltau)[0]); mac("RevRegCorrDur", pearsonr(u.n_registers, u.log_hold)[0])
mac("RevRegCorrPos", pearsonr(u.n_registers, u.frac_in_pos)[0])
mac("RevLitTasks", d[(d.n_registers <= 2) & (d.n_risk == 1)].strategy_id.nunique(), "{}")
mac("RevAltTasks", d[(d.n_registers <= 5) & (d.n_risk == 1)].strategy_id.nunique(), "{}")
mac("RevRegMin", int(u.n_registers.min()), "{}"); mac("RevRegMax", int(u.n_registers.max()), "{}"); mac("RevRegMedian", float(u.n_registers.median()), "{:.0f}")
QN = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
SUB = {"Low": D.model.isin(LOWCOST), "Front": D.model.isin(FRONTIER) & D.strategy_id.isin(ids200)}
mac("RevDurNLow", int(SUB["Low"].sum()), "{}"); mac("RevDurNFront", int(SUB["Front"].sum()), "{}")
for q in [1, 2, 3, 4, 5]:
    mac(f"RevDurQ{QN[q]}MedBars", qb.loc[q, "median"], "{:.1f}"); mac(f"RevDurQ{QN[q]}MinBars", qb.loc[q, "min"], "{:.1f}")
    mac(f"RevDurQ{QN[q]}MaxBars", qb.loc[q, "max"], "{:.1f}"); mac(f"RevDurQ{QN[q]}MedRegs", qmed_reg.loc[q], "{:.0f}")
def matched_table(col, tag, fname):
    L = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r"Holding-duration & median & \multicolumn{4}{c}{Low-cost models} & \multicolumn{4}{c}{Frontier models} & low $=$ \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"quintile & bars & $n_{\mathrm{L}}/n_{\mathrm{H}}$ & AM$_{\mathrm{L}}$ / AM$_{\mathrm{H}}$ & $\Delta$AM ($t$) & med.\ $|IS|_{\mathrm{L}}$ / $|IS|_{\mathrm{H}}$ & "
         r"$n_{\mathrm{L}}/n_{\mathrm{H}}$ & AM$_{\mathrm{L}}$ / AM$_{\mathrm{H}}$ & $\Delta$AM ($t$) & med.\ $|IS|_{\mathrm{L}}$ / $|IS|_{\mathrm{H}}$ & regs.\ $\le$ \\", r"\midrule"]
    for q in [1, 2, 3, 4, 5]:
        cells = []
        for g_tag, sub in SUB.items():
            g = D[sub & (D.dur_q == q)]; lo, hi = g[g[col]], g[~g[col]]
            p = f"RevDur{tag}Q{QN[q]}{g_tag}"
            if len(lo) < 5 or len(hi) < 5:
                cells.append("-- & -- & -- & --")
                for s in ["NLo", "NHi", "AMLo", "AMHi", "AMDiff", "AMT", "ISLo", "ISHi"]: mac(f"{p}{s}", None)
                continue
            dAM = lo.action_match.mean() - hi.action_match.mean()
            t = float(ttest_ind(lo.action_match, hi.action_match, equal_var=False).statistic)
            mac(f"{p}NLo", len(lo), "{}"); mac(f"{p}NHi", len(hi), "{}")
            mac(f"{p}AMLo", lo.action_match.mean()); mac(f"{p}AMHi", hi.action_match.mean()); mac(f"{p}AMDiff", dAM, "{:+.3f}"); mac(f"{p}AMT", t, "{:.2f}")
            mac(f"{p}ISLo", lo.abs_es.median(), "{:.0f}"); mac(f"{p}ISHi", hi.abs_es.median(), "{:.0f}")
            mac(f"{p}DurLo", lo.hold_dur.median(), "{:.1f}"); mac(f"{p}DurHi", hi.hold_dur.median(), "{:.1f}")
            cells.append(f"{len(lo)}/{len(hi)} & {lo.action_match.mean():.3f} / {hi.action_match.mean():.3f} & {dAM:+.3f} ({t:.2f}) & {lo.abs_es.median():.0f} / {hi.abs_es.median():.0f}")
        thr = f"{qmed_reg.loc[q]:.0f}" if col == "low_med" else ("5, 1 layer" if col == "low_alt" else "2, 1 layer")
        L.append(f"Q{q} & {qb.loc[q, 'median']:.1f} & " + " & ".join(cells) + f" & {thr} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/{fname}", "w").write("\n".join(L) + "\n")
matched_table("low_med", "Med", "table_frl_matched_dur.tex")
matched_table("low_alt", "Alt", "table_frl_matched_dur_alt.tex")
matched_table("low_lit", "Lit", "table_frl_matched_dur_lit.tex")
# pooled within-quintile effects (duration-quintile x model x instrument FE, strategy clusters)
for g_tag, sub in SUB.items():
    g = D[sub]
    ra = ols("am_pp ~ n_registers + C(dur_q) + C(model) + C(asset)", g); ri = ols("wes ~ n_registers + C(dur_q) + C(model) + C(asset)", g)
    rt = ols("am_pp ~ ltau + C(dur_q) + C(model) + C(asset)", g); rti = ols("wes ~ ltau + C(dur_q) + C(model) + C(asset)", g)
    rb = ols("am_pp ~ n_registers + ltau + log_hold + C(dur_q) + C(model) + C(asset)", g)
    mac(f"RevDurPool{g_tag}AMReg", ra.params["n_registers"], "{:.2f}"); mac(f"RevDurPool{g_tag}AMRegT", ra.tvalues["n_registers"], "{:.2f}")
    mac(f"RevDurPool{g_tag}ISReg", ri.params["n_registers"], "{:.1f}"); mac(f"RevDurPool{g_tag}ISRegT", ri.tvalues["n_registers"], "{:.2f}")
    mac(f"RevDurPool{g_tag}AMTau", rt.params["ltau"], "{:.2f}"); mac(f"RevDurPool{g_tag}AMTauT", rt.tvalues["ltau"], "{:.2f}")
    mac(f"RevDurPool{g_tag}ISTau", rti.params["ltau"], "{:.1f}"); mac(f"RevDurPool{g_tag}ISTauT", rti.tvalues["ltau"], "{:.2f}")
    mac(f"RevDurPool{g_tag}AMBothReg", rb.params["n_registers"], "{:.2f}"); mac(f"RevDurPool{g_tag}AMBothRegT", rb.tvalues["n_registers"], "{:.2f}")
    mac(f"RevDurPool{g_tag}AMBothTau", rb.params["ltau"], "{:.2f}"); mac(f"RevDurPool{g_tag}AMBothTauT", rb.tvalues["ltau"], "{:.2f}")

hdr = ["% AUTO-GENERATED by scripts/frl_revision.py (FRL-D-26-05997 minor revision, Reviewer 1 comments 1-3) -- do not edit.",
       "% Conventions: see the docstring of scripts/frl_revision.py.", r"\providecommand{\todo}[1]{{\color{red}\textbf{[TODO: #1]}}}"]
open(f"{OUT}/numbers_frl_rev.tex", "w").write("\n".join(hdr + M) + "\n")
print(len(M), "macros ->", f"{OUT}/numbers_frl_rev.tex")
print(f"ambiguous {len(amb_ids)} ({100*len(amb_ids)/800:.1f}%): roundtrip {len(rt_fail)}, retest-no-close {len(bp_amb)}/{len(bp_all)}, htf-no-completed {len(htf_amb)}/{len(htf_all)}; strict subset {len(strict_unamb_ids)}")
for k in SAMPLES:
    r = R[k]; g = lambda t: f"{r.params[t]:8.1f} ({r.tvalues[t]:5.2f})" if t in r.params else " " * 16
    print(f"{k:10s} tau {g('ltau')} dur {g('log_hold')} pos {g('frac_in_pos')} N={int(r.nobs)}")
print("VIF (unit level):", {k: round(v, 2) for k, v in vif3.items()}, " 2-var:", {k: round(v, 2) for k, v in vif2.items()}, " pooled:", {k: round(v, 2) for k, v in vif3p.items()})
print("corr(n_registers, ltau) =", round(pearsonr(u.n_registers, u.ltau)[0], 3), " corr(n_registers, log_hold) =", round(pearsonr(u.n_registers, u.log_hold)[0], 3))
