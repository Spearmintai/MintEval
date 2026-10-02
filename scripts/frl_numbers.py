"""All FRL numbers from results/multiasset/minteval_multiasset.csv -> paper_frl/numbers_frl.tex (one macro per
number) + paper_frl/table_frl_*.tex (tables reference the macros). Nothing is typed by hand; any quantity that
cannot be computed is emitted as \\todo{...}.

Conventions (also written into the .tex header):
- Analysis sample: references passing the per-asset filter (pass_filter) and compiled generated runs.
- |ES| = |R_llm - R_bench| in bp of initial equity, winsorized at the 1st/99th percentile WITHIN each asset.
- tau = log10(tau_max) (re-measured per asset), centred at its mean over the analysis sample (asset x strategy).
- sigma = annualised realised volatility over the reference's holding bars, standardised within each asset
  (mean/sd over that asset's strategies). Used for crypto and equities alike.
- SEs clustered by strategy_id (same program across models and assets).
- Signed ES = (R_llm - R_bench) in bp; t = mean / (sd / sqrt(n)) over compiled tasks.
"""
import sys, json, os, math
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
import statsmodels.formula.api as smf
from minteval.data import load_prices

OUT = "paper_frl"; os.makedirs(OUT, exist_ok=True)
# robustness mode: --vix uses VIX (mean over the reference's holding days) as sigma for the equity instruments;
# crypto keeps realised vol. Only the regression/cell outputs are produced, under prefix FrlVix / suffix _vix.
VIX = "--vix" in sys.argv
PFX = "FrlVix" if VIX else "Frl"
SFX = "_vix" if VIX else ""
cfg = yaml.safe_load(open("configs/base.yaml")); MA = cfg["multiasset"]["assets"]
d = pd.read_csv("results/multiasset/minteval_multiasset.csv")
spec = {json.loads(l)["strategy_id"]: json.loads(l)["spec"] for l in open("results/tasks/tasks.jsonl")}
RISK = ["fixed_stop", "breakeven", "trailing_hwm", "time_stop", "take_profit", "partial_take", "box_shift", "cooldown", "daily_cap"]
d["sizing_id"] = d.strategy_id.map(lambda s: spec[s]["sizing"]["id"])
d["filter_id"] = d.strategy_id.map(lambda s: spec[s]["filter"]["id"])
d["direction"] = d.strategy_id.map(lambda s: spec[s]["direction"])
for r in RISK:
    d[f"has_{r}"] = d.strategy_id.map(lambda s: int(any(x["id"] == r for x in spec[s]["risk"])))
d["es_bp"] = (d.R_llm - d.R_bench) * 1e4
FRONTIER = ["claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro"]
LOWCOST = ["gpt-5.4-mini", "claude-haiku", "qwen2.5-coder-32b", "qwen2.5-coder-7b"]
MK = {"claude-opus-5.5": "Opus", "qwen3.8-max": "QMax", "deepseek-v4-pro": "DSPro", "gpt-5.4-mini": "Gpt",
      "claude-haiku": "Haiku", "qwen2.5-coder-32b": "QwenL", "qwen2.5-coder-7b": "QwenS"}
MNAME = {"claude-opus-5.5": "Claude Opus 5.5", "qwen3.8-max": "Qwen3.8-Max", "deepseek-v4-pro": "DeepSeek-V4-Pro",
         "gpt-5.4-mini": "GPT-5.4-mini", "claude-haiku": "Claude Haiku", "qwen2.5-coder-32b": "Qwen2.5-Coder-32B",
         "qwen2.5-coder-7b": "Qwen2.5-Coder-7B"}
AK = {"BTCUSDT_2022_2023": "Btc", "ETHUSDT_2018_2023": "Eth", "SPY_CFD_2018_2023": "Spy", "NDX_CFD_2018_2023": "Ndx"}
ANAME = {"BTCUSDT_2022_2023": "BTC/USDT", "ETHUSDT_2018_2023": "ETH/USDT", "SPY_CFD_2018_2023": "SPY (CFD proxy)",
         "NDX_CFD_2018_2023": "Nasdaq-100 (CFD proxy)"}
ids200 = {json.loads(l)["strategy_id"] for l in open("results/tasks/frontier200.jsonl")}

M = []           # macro lines
def mac(name, val, fmt="{:.3f}"):
    if VIX and name.startswith("Frl") and not name.startswith("FrlVix"):
        name = "FrlVix" + name[3:]
    assert not any(c.isdigit() for c in name), name
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        M.append(f"\\newcommand{{\\{name}}}{{\\todo{{{name}}}}}"); return
    M.append(f"\\newcommand{{\\{name}}}{{{fmt.format(val)}}}")
num = lambda x: "{:,}".format(int(x)).replace(",", "{,}")

# ---------------------------------------------------------------- analysis sample
D = d[d.pass_filter & (d.compile_fail == 0)].copy()
D["wes"] = D.groupby("asset").abs_es.transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
u = D.drop_duplicates(["asset", "strategy_id"])
tau_mean = float(np.log10(u.tau_max).mean())
D["ltau"] = np.log10(D.tau_max) - tau_mean
EQUITY = {"SPY_CFD_2018_2023", "NDX_CFD_2018_2023"}
D["sig_raw"] = np.where(VIX & D.asset.isin(EQUITY), D.vix_mean_held, D.realized_vol)
u = D.drop_duplicates(["asset", "strategy_id"])
vs = u.groupby("asset").sig_raw.agg(["mean", "std"])
D["sig"] = (D.sig_raw - D.asset.map(vs["mean"])) / D.asset.map(vs["std"])
D["ltrades"] = np.log(D.n_trades_bench)
mac("FrlTauMean", tau_mean, "{:.3f}")
mac("FrlNAnalysis", len(D), "{:,}".replace(",", "{,}") if False else "{}")
for a, k in AK.items():
    lo, hi = d[d.pass_filter & (d.compile_fail == 0) & (d.asset == a)].abs_es.quantile([0.01, 0.99])
    mac(f"FrlWinsLo{k}", lo, "{:.0f}"); mac(f"FrlWinsHi{k}", hi, "{:.0f}")
    mac(f"FrlSigMean{k}", vs.loc[a, "mean"], "{:.3f}"); mac(f"FrlSigSd{k}", vs.loc[a, "std"], "{:.3f}")

# ---------------------------------------------------------------- 1. regression table
CTRL = "K_bits + ltrades + mccabe + R_bench"
FE = "C(signal_id) + C(sizing_id) + C(filter_id) + C(direction) + " + " + ".join(f"has_{r}" for r in RISK)
FULL = f"wes ~ ltau * sig + {CTRL} + {FE} + C(model) + C(asset)"
COLS = [("One", "wes ~ ltau", D),
        ("Two", "wes ~ ltau + sig", D),
        ("Three", "wes ~ ltau * sig", D),
        ("Four", FULL, D),
        ("Five", FULL, D[D.model.isin(LOWCOST)]),
        ("Six", FULL, D[D.model.isin(FRONTIER) & D.strategy_id.isin(ids200)]),
        ("Seven", FULL, D[D.asset.isin(["SPY_CFD_2018_2023", "NDX_CFD_2018_2023"])]),
        ("Eight", FULL, D[D.asset.isin(["BTCUSDT_2022_2023", "ETHUSDT_2018_2023"])])]
REG = {}
for col, f, data in COLS:
    try:
        r = smf.ols(f, data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.strategy_id)[0]})
        REG[col] = r
        for term, tk in (("ltau", "Tau"), ("sig", "Sig"), ("ltau:sig", "Int"), ("R_bench", "Rbench"),
                         ("K_bits", "K"), ("ltrades", "Ltr"), ("mccabe", "Mcc")):
            if term in r.params:
                mac(f"FrlReg{col}{tk}", r.params[term], "{:.1f}"); mac(f"FrlReg{col}{tk}T", r.tvalues[term], "{:.2f}")
            # absent terms get no macro; the table writer leaves those cells empty
        mac(f"FrlReg{col}N", r.nobs, "{:.0f}"); mac(f"FrlReg{col}AdjR", r.rsquared_adj, "{:.3f}")
        mac(f"FrlReg{col}Clust", data.strategy_id.nunique(), "{}")
    except Exception as e:   # noqa: BLE001
        for tk in ("Tau", "Sig", "Int", "Rbench", "K", "Ltr", "Mcc", "N", "AdjR", "Clust"):
            mac(f"FrlReg{col}{tk}", None); mac(f"FrlReg{col}{tk}T", None)
        print("regression failed", col, e)

def cell(col, tk):
    return f"\\FrlReg{col}{tk}" + (f" \\\\ & ({chr(92)}FrlReg{col}{tk}T)" if False else "")
cols = [c for c, _, _ in COLS]
L = [r"\begin{tabular}{l" + "r" * 8 + "}", r"\toprule",
     " & " + " & ".join(f"({i+1})" for i in range(8)) + r" \\",
     r" & $\tau$ & $+\sigma$ & $+\tau{\times}\sigma$ & full & low-cost & frontier & equity & crypto \\", r"\midrule"]
TERM = {"Tau": "ltau", "Sig": "sig", "Int": "ltau:sig", "Rbench": "R_bench", "K": "K_bits", "Ltr": "ltrades", "Mcc": "mccabe"}
has = lambda c, tk: c in REG and TERM[tk] in REG[c].params
for tk, lab in (("Tau", r"$\log_{10}\tau_{\max}$ (centred)"), ("Sig", r"$\sigma$ (std.\ within asset)"),
                ("Int", r"$\tau\times\sigma$"), ("Rbench", "reference return"), ("K", "$K$ (bits)"),
                ("Ltr", "log trades"), ("Mcc", "McCabe")):
    L.append(lab + " & " + " & ".join(f"\\{PFX}Reg{c}{tk}" if has(c, tk) else "" for c in cols) + r" \\")
    L.append(" & " + " & ".join(f"(\\{PFX}Reg{c}{tk}T)" if has(c, tk) else "" for c in cols) + r" \\")
L += [r"\midrule",
      "block FE & -- & -- & -- & yes & yes & yes & yes & yes \\\\",
      "model FE & -- & -- & -- & yes & yes & yes & yes & yes \\\\",
      "asset FE & -- & -- & -- & yes & yes & yes & yes & yes \\\\",
      "$N$ & " + " & ".join(f"\\{PFX}Reg{c}N" for c in cols) + r" \\",
      "strategies & " + " & ".join(f"\\{PFX}Reg{c}Clust" for c in cols) + r" \\",
      r"adj.\ $R^2$ & " + " & ".join(f"\\{PFX}Reg{c}AdjR" for c in cols) + r" \\",
      r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_reg{SFX}.tex", "w").write("\n".join(L) + "\n")

# sensitivity: |ES| contains R_bench mechanically (|R_llm - R_bench|), so conditioning on it may be a bad
# control. Report the full specification without R_bench (all samples of columns 4-8).
FULL_NOR = FULL.replace(" + R_bench", "")
assert FULL_NOR != FULL
for col, f, data in COLS[3:]:
    r = smf.ols(FULL_NOR, data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.strategy_id)[0]})
    for term, tk in (("ltau", "Tau"), ("sig", "Sig"), ("ltau:sig", "Int")):
        mac(f"FrlRegNoR{col}{tk}", r.params[term], "{:.1f}"); mac(f"FrlRegNoR{col}{tk}T", r.tvalues[term], "{:.2f}")
    mac(f"FrlRegNoR{col}N", r.nobs, "{:.0f}"); mac(f"FrlRegNoR{col}AdjR", r.rsquared_adj, "{:.3f}")
L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r" & (4) full & (5) low-cost & (6) frontier & (7) equity & (8) crypto \\", r"\midrule"]
for tk, lab in (("Tau", r"$\log_{10}\tau_{\max}$"), ("Sig", r"$\sigma$"), ("Int", r"$\tau\times\sigma$")):
    L.append(lab + " & " + " & ".join(f"\\{PFX}RegNoR{c}{tk}" for c, _, _ in COLS[3:]) + r" \\")
    L.append(" & " + " & ".join(f"(\\{PFX}RegNoR{c}{tk}T)" for c, _, _ in COLS[3:]) + r" \\")
L += [r"\midrule", r"$N$ & " + " & ".join(f"\\{PFX}RegNoR{c}N" for c, _, _ in COLS[3:]) + r" \\",
      r"adj.\ $R^2$ & " + " & ".join(f"\\{PFX}RegNoR{c}AdjR" for c, _, _ in COLS[3:]) + r" \\", r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_reg_noR{SFX}.tex", "w").write("\n".join(L) + "\n")   # columns (4)-(8) without R_bench

# ---------------------------------------------------------------- robustness: alternative outcomes (column-4 spec without
# R_bench, all four assets, clustered by strategy)
if not VIX:
    RHS = FULL_NOR.split("~", 1)[1]
    def rob(name, dep, data):
        r = smf.ols(f"{dep} ~{RHS}", data).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data.strategy_id)[0]})
        mac(f"FrlRob{name}Tau", r.params["ltau"], "{:.2f}"); mac(f"FrlRob{name}TauT", r.tvalues["ltau"], "{:.2f}")
        mac(f"FrlRob{name}N", r.nobs, "{:.0f}")
        print(f"Rob {name:8s} tau {r.params['ltau']:9.3f} (t {r.tvalues['ltau']:5.2f}) N={int(r.nobs)}")
    # (a) |ES| / realised vol of the reference's holding bars, winsorized 1/99 within asset
    Ds = D.copy(); Ds["sc"] = Ds.abs_es / Ds.realized_vol
    Ds["wsc"] = Ds.groupby("asset").sc.transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    rob("Scaled", "wsc", Ds)
    # (b) ActionMatch in percentage points
    Da = D.copy(); Da["am_pp"] = 100 * Da.action_match
    rob("AM", "am_pp", Da)
    # (c) compile failures kept, |ES| := p99 of compiled |ES| of the same model on the same asset (raw, before winsorizing)
    Dm = d[d.pass_filter].copy()
    p99 = Dm[Dm.compile_fail == 0].groupby(["model", "asset"]).abs_es.quantile(0.99)
    fill = Dm.set_index(["model", "asset"]).index.map(p99.to_dict().get)
    Dm["es_imp"] = np.where(Dm.compile_fail == 1, np.asarray(fill, dtype=float), Dm.abs_es)
    Dm["wmax"] = Dm.groupby("asset").es_imp.transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
    Dm["ltau"] = np.log10(Dm.tau_max) - tau_mean
    Dm["sig"] = (Dm.realized_vol - Dm.asset.map(vs["mean"])) / Dm.asset.map(vs["std"])
    Dm["ltrades"] = np.log(Dm.n_trades_bench)
    mac("FrlRobMaxLossNImputed", int((Dm.compile_fail == 1).sum()), "{}")
    rob("MaxLoss", "wmax", Dm)

# ---------------------------------------------------------------- ActionMatch on tau (no block FE / low-cost with block FE),
# exposure-controlled |ES| (column-4 spec w/o R_bench + share of bars in position + log mean holding duration)
if not VIX:
    CT = "K_bits + ltrades + mccabe"
    FEB = "C(signal_id) + C(sizing_id) + C(filter_id) + C(direction) + " + " + ".join(f"has_{r}" for r in RISK)
    Da = D.copy(); Da["am_pp"] = 100 * Da.action_match
    cl = lambda data: {"groups": pd.factorize(data.strategy_id)[0]}
    r = smf.ols(f"am_pp ~ ltau * sig + {CT} + C(model) + C(asset)", Da).fit(cov_type="cluster", cov_kwds=cl(Da))
    mac("FrlAMNoFETau", r.params["ltau"], "{:.2f}"); mac("FrlAMNoFETauT", r.tvalues["ltau"], "{:.2f}")
    Dl = Da[Da.model.isin(LOWCOST)]
    r = smf.ols(f"am_pp ~ ltau * sig + {CT} + {FEB} + C(model) + C(asset)", Dl).fit(cov_type="cluster", cov_kwds=cl(Dl))
    mac("FrlAMLowFETau", r.params["ltau"], "{:.2f}"); mac("FrlAMLowFETauT", r.tvalues["ltau"], "{:.2f}")
    # exposure controls: bars in position = frac_in_pos x bars of the asset; mean holding duration = that / round trips
    nbars = {a: len(load_prices(MA[a]["price_file"])["close"]) for a in AK}
    De = D.copy()
    De["log_hold"] = np.log(De.frac_in_pos * De.asset.map(nbars) / De.n_trades_bench)
    r = smf.ols(FULL_NOR + " + frac_in_pos + log_hold", De).fit(cov_type="cluster", cov_kwds=cl(De))
    mac("FrlExpoTau", r.params["ltau"], "{:.1f}"); mac("FrlExpoTauT", r.tvalues["ltau"], "{:.2f}")
    mac("FrlExpoN", r.nobs, "{:.0f}")
    print(f"Expo tau {r.params['ltau']:.1f} (t {r.tvalues['ltau']:.2f})  frac_in_pos {r.params['frac_in_pos']:.1f} (t {r.tvalues['frac_in_pos']:.2f})  log_hold {r.params['log_hold']:.1f} (t {r.tvalues['log_hold']:.2f})")
    # diversification ranges from results/decay/decay.json (all assets x models)
    DJ = json.load(open("results/decay/decay.json")) if os.path.exists("results/decay/decay.json") else None
    if DJ:
        rhos = [v["rho_mean"] for a in DJ.values() for v in a.values()]
        r100 = [100 * v["curve"]["100"]["ratio"] for a in DJ.values() for v in a.values() if "100" in v["curve"]]
        mac("FrlDivCorrMin", min(rhos), "{:.3f}"); mac("FrlDivCorrMax", max(rhos), "{:.3f}")
        mac("FrlDivVarMin", min(r100), "{:.0f}"); mac("FrlDivVarMax", max(r100), "{:.0f}")
        # tiny portfolio-diversification table: per model, averaged over the four assets
        TL = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & $\bar\rho$ & Var$_{10}$ (\%) & Var$_{50}$ (\%) \\", r"\midrule"]
        for m in FRONTIER + LOWCOST:
            vals = [a[m] for a in DJ.values() if m in a]
            if not vals: continue
            k = MK[m]
            mac(f"FrlDivRho{k}", float(np.mean([v["rho_mean"] for v in vals])), "{:.3f}")
            for kk, nm in (("10", "Ten"), ("50", "Fifty")):
                xs = [100 * v["curve"][kk]["ratio"] for v in vals if kk in v["curve"]]
                mac(f"FrlDivVar{nm}{k}", float(np.mean(xs)) if len(xs) == len(vals) else None, "{:.1f}")
            TL.append(f"{MNAME[m]} & \\FrlDivRho{k} & \\FrlDivVarTen{k} & \\FrlDivVarFifty{k} \\\\")
        mac("FrlDivIndepTen", 100 / 10, "{:.1f}"); mac("FrlDivIndepFifty", 100 / 50, "{:.1f}"); mac("FrlDivIndepRho", 0.0, "{:.0f}")
        TL += [r"\midrule", r"Independent errors & 0 & \FrlDivIndepTen & \FrlDivIndepFifty \\", r"\bottomrule", r"\end{tabular}"]
        open(f"{OUT}/table_frl_div.tex", "w").write("\n".join(TL) + "\n")
    else:
        for n in ("FrlDivCorrMin", "FrlDivCorrMax", "FrlDivVarMin", "FrlDivVarMax"): mac(n, None)
else:
    # column (7) aliases for the VIX run
    r7 = REG.get("Seven")
    mac("FrlEqSig", r7.params["sig"] if r7 is not None else None, "{:.1f}"); mac("FrlEqSigT", r7.tvalues["sig"] if r7 is not None else None, "{:.2f}")
    mac("FrlEqInt", r7.params["ltau:sig"] if r7 is not None else None, "{:.1f}"); mac("FrlEqIntT", r7.tvalues["ltau:sig"] if r7 is not None else None, "{:.2f}")

# ---------------------------------------------------------------- 2. interaction by model x asset
rows = []
for (m, a), g in D.groupby(["model", "asset"]):
    if m in FRONTIER: g = g[g.strategy_id.isin(ids200)]
    try:
        r = smf.ols(f"wes ~ ltau * sig + {CTRL}", g).fit(cov_type="HC1")   # one obs per strategy in a cell
        rows.append({"model": m, "asset": a, "n": int(r.nobs), "b": float(r.params["ltau:sig"]),
                     "t": float(r.tvalues["ltau:sig"]), "p": float(r.pvalues["ltau:sig"])})
    except Exception as e:  # noqa: BLE001
        rows.append({"model": m, "asset": a, "n": len(g), "b": np.nan, "t": np.nan, "p": np.nan})
C = pd.DataFrame(rows); C.to_csv(f"results/multiasset/frl_interaction_cells{SFX}.csv", index=False)
ok = C.dropna(subset=["p"])
mac("FrlCellsN", len(C), "{}"); mac("FrlCellsEst", len(ok), "{}")
mac("FrlCellsSig", int((ok.p < 0.05).sum()), "{}")
mac("FrlCellsSigPos", int(((ok.p < 0.05) & (ok.b > 0)).sum()), "{}")
mac("FrlCellsSigNeg", int(((ok.p < 0.05) & (ok.b < 0)).sum()), "{}")
mac("FrlCellsExpected", 0.05 * len(ok), "{:.1f}")
for r in C.itertuples():
    k = MK[r.model] + AK[r.asset]
    mac(f"FrlCell{k}B", r.b, "{:.1f}"); mac(f"FrlCell{k}T", r.t, "{:.2f}"); mac(f"FrlCell{k}N", r.n, "{}")
L = [r"\begin{tabular}{l" + "r" * len(AK) + "}", r"\toprule", "Model & " + " & ".join(ANAME[a] for a in AK) + r" \\", r"\midrule"]
for m in FRONTIER + LOWCOST:
    L.append(MNAME[m] + " & " + " & ".join(f"\\{PFX}Cell{MK[m]}{AK[a]}B" for a in AK) + r" \\")
    L.append(" & " + " & ".join(f"(\\{PFX}Cell{MK[m]}{AK[a]}T)" for a in AK) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_cells{SFX}.tex", "w").write("\n".join(L) + "\n")

if VIX:
    hdr = ["% AUTO-GENERATED by scripts/frl_numbers.py --vix: robustness with VIX as sigma for SPY/NDX (crypto: realised vol)."]
    open(f"{OUT}/numbers_frl_vix.tex", "w").write("\n".join(hdr + M) + "\n")
    for c in [c for c, _, _ in COLS]:
        r = REG.get(c)
        if r is not None:
            g = lambda t: f"{r.params[t]:8.1f} ({r.tvalues[t]:5.2f})" if t in r.params else " " * 16
            print(f"({c:5s}) tau {g('ltau')} sig {g('sig')} int {g('ltau:sig')}  N={int(r.nobs)} adjR2={r.rsquared_adj:.3f}")
    print("cells", C.assign(sig=C.p < 0.05).groupby("asset").sig.sum().to_dict(), int((ok.p < 0.05).sum()), "/", len(ok))
    sys.exit(0)
# ---------------------------------------------------------------- 3. Table 1 (per asset sample definitions)
EMPTY = set()
def t1(sample, tag):
    out = []
    for m in FRONTIER + LOWCOST:
        g = sample[sample.model == m]
        k = f"Tone{tag}{MK[m]}"
        if g.empty:
            EMPTY.add(k); continue
        okk = g[g.compile_fail == 0]; es = okk.es_bp
        tval = es.mean() / (es.std(ddof=1) / math.sqrt(len(es))) if len(es) > 1 and es.std() > 0 else float("nan")
        mac(f"{k}N", len(g), "{}"); mac(f"{k}Compile", 1 - g.compile_fail.mean())
        mac(f"{k}ESMed", okk.abs_es.median(), "{:.0f}"); mac(f"{k}ESMean", es.mean(), "{:.0f}"); mac(f"{k}ESMeanT", tval, "{:.2f}")
        mac(f"{k}AM", okk.action_match.mean()); mac(f"{k}Exact", (okk.action_match == 1).sum() / len(g))
        mac(f"{k}Silent", ((g.compile_fail == 0) & (g.action_match < 0.9)).mean())
def t1_table(tags, fname, caption_note, drop_empty=False):
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Model & $N$ & Compile & med.\ $|$ES$|$ (bp) & mean ES (bp) [$t$] & ActionMatch & Exact & Silent \\"]
    for tag, title in tags:
        L += [r"\midrule", f"\\multicolumn{{8}}{{l}}{{\\emph{{{title}}}}} \\\\"]
        for m in FRONTIER + LOWCOST:
            k = f"Tone{tag}{MK[m]}"
            if k in EMPTY and drop_empty:
                continue
            if k in EMPTY:
                L.append(f"{MNAME[m]} & " + " & ".join(["--"] * 7) + r" \\"); continue
            L.append(f"{MNAME[m]} & \\{k}N & \\{k}Compile & \\{k}ESMed & \\{k}ESMean{{}} [\\{k}ESMeanT] & \\{k}AM & \\{k}Exact & \\{k}Silent \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/{fname}", "w").write("\n".join(L) + "\n")      # tabular only (note: caption_note is for the author)
B = d[(d.asset == "BTCUSDT_2022_2023") & d.pass_filter]
t1(B[B.subset == "full800"], "BtcFull"); t1(B[B.strategy_id.isin(ids200)], "BtcSub")
t1_table([("BtcFull", "BTC/USDT 2022--2023, all 800 tasks (frontier models were run on the 200-task subset only)"),
          ("BtcSub", "BTC/USDT 2022--2023, 200-task subset (40 per $\\tau$ quintile)")], "table_frl_table1.tex",
         "Table 1 on the benchmark instrument (BTC), open setting.")
P = d[d.pass_filter]
t1(P[P.subset == "full800"], "AllFull"); t1(P[P.strategy_id.isin(ids200)], "AllSub")
t1_table([("AllFull", "All four instruments pooled, surviving references, all tasks"),
          ("AllSub", "All four instruments pooled, 200-task subset")], "table_frl_table1_pooled.tex",
         "Table 1 pooled over BTC, ETH, SPY and Nasdaq-100 (surviving references only).", drop_empty=True)

# ---------------------------------------------------------------- 4. multi-asset description
rep = json.load(open("results/multiasset/report.json"))
L = [r"\begin{tabular}{lllrrrr}", r"\toprule",
     r"Instrument & Window (first--last bar, UTC) & Frictions (bp) & Bars & Surviving refs & Median ref.\ return & Profitable \\", r"\midrule"]
for a, k in AK.items():
    pr = load_prices(MA[a]["price_file"]); t0, t1_ = pd.to_datetime(pr["open_time"][[0, -1]], unit="ms", utc=True)
    M.append(f"\\newcommand{{\\Frl{k}Start}}{{{t0.strftime('%Y-%m-%d')}}}"); M.append(f"\\newcommand{{\\Frl{k}End}}{{{t1_.strftime('%Y-%m-%d')}}}")
    mac(f"Frl{k}Bars", len(pr["close"]), "{}"); mac(f"Frl{k}Fee", MA[a]["fee_bp"], "{:g}"); mac(f"Frl{k}Slip", MA[a]["slippage_bp"], "{:g}")
    r = rep[a]
    mac(f"Frl{k}Survive", r["survive_filter"], "{}"); mac(f"Frl{k}Refs", r["n_refs"], "{}")
    mac(f"Frl{k}RMed", 100 * r["R_bench_median_survivors"], "{:.1f}"); mac(f"Frl{k}Profitable", r["share_profitable"], "{:.2f}")
    L.append(f"{ANAME[a]} & \\Frl{k}Start--\\Frl{k}End & \\Frl{k}Fee{{}} + \\Frl{k}Slip & \\Frl{k}Bars & \\Frl{k}Survive/\\Frl{k}Refs & \\Frl{k}RMed\\% & \\Frl{k}Profitable \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/table_frl_assets.tex", "w").write("\n".join(L) + "\n")

hdr = ["% AUTO-GENERATED by scripts/frl_numbers.py from results/multiasset/minteval_multiasset.csv -- do not edit.",
       "% Conventions: see the docstring of scripts/frl_numbers.py (winsorization within asset, sigma standardised within",
       "% asset, tau centred, SEs clustered by strategy).",
       r"\providecommand{\todo}[1]{{\color{red}\textbf{[TODO: #1]}}}"]
open(f"{OUT}/numbers_frl.tex", "w").write("\n".join(hdr + M) + "\n")
print(len(M), "macros")
for c in cols:
    r = REG.get(c)
    if r is not None:
        g = lambda t: f"{r.params[t]:8.1f} ({r.tvalues[t]:5.2f})" if t in r.params else " " * 16
        print(f"({c:5s}) tau {g('ltau')} sig {g('sig')} int {g('ltau:sig')}  N={int(r.nobs)} adjR2={r.rsquared_adj:.3f}")
print(C.assign(sig=C.p < 0.05).groupby("asset").sig.sum().to_dict(), "sig cells", int((ok.p < 0.05).sum()), "/", len(ok))
