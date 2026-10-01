"""Sensitivity: re-score rows that failed with StateTypeError under a lenient engine (None/str allowed).
Main results are unchanged; output: results/sensitivity_state.csv + printed summary."""
import sys, json, glob, yaml
sys.path.insert(0, ".")
import pandas as pd
from minteval.evaluate import load_tasks, reference_runs, score
cfg = yaml.safe_load(open("configs/base.yaml")); pf = cfg["data"]["price_file"]
main = pd.read_csv("results/minteval_v0.csv")
bad = main[main.error_type == "StateTypeError"]
tasks = [t for t in load_tasks() if t["strategy_id"] in set(bad.strategy_id)]
refs = reference_runs(tasks, pf, cfg["engine"])
ecfg = dict(cfg["engine"], lenient_state=True)
frames = []
for f in sorted(glob.glob("results/generations/*.jsonl")):
    G = [json.loads(l) for l in open(f)]
    s = G[0]["setting"]; m = G[0]["model"]
    keys = set(map(tuple, bad[(bad.model == m) & (bad.setting == s)][["strategy_id", "model"]].values))
    gens = {(g["strategy_id"], g["model"]): {"code": g["code"], "spec": g["spec"]} for g in G if (g["strategy_id"], g["model"]) in keys}
    if gens:
        frames.append(score(tasks, gens, refs, pf, ecfg, setting=s))
L = pd.concat(frames); L.to_csv("results/sensitivity_state.csv", index=False)
merged = main.set_index(["model", "setting", "strategy_id"]).copy()
for _, r in L.iterrows():
    for c in ["compile_fail", "action_match", "trade_f1", "abs_es"]:
        merged.loc[(r.model, r.setting, r.strategy_id), c] = r[c]
merged = merged.reset_index()
out = []
for (m, s), g in merged.groupby(["model", "setting"]):
    g0 = main[(main.model == m) & (main.setting == s)]
    out.append({"model": m, "setting": s, "n_state_type_fail": int((g0.error_type == "StateTypeError").sum()),
                "recovered_compile": int(((L.model == m) & (L.setting == s) & (L.compile_fail == 0)).sum()),
                "CompileOK_strict": 1 - g0.compile_fail.mean(), "CompileOK_lenient": 1 - g.compile_fail.mean(),
                "AM_strict": g0[g0.compile_fail == 0].action_match.mean(), "AM_lenient": g[g.compile_fail == 0].action_match.mean(),
                "AM_rescued_rows": L[(L.model == m) & (L.setting == s) & (L.compile_fail == 0)].action_match.mean()})
O = pd.DataFrame(out); O.to_csv("results/sensitivity_state_summary.csv", index=False); print(O.round(3).to_string())
