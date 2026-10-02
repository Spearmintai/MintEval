import sys, json, yaml
sys.path.insert(0, ".")
import pandas as pd
from minteval.evaluate import load_tasks, reference_runs, score
cfg = yaml.safe_load(open("configs/base.yaml")); pf = cfg["data"]["price_file"]
M = [json.loads(l) for l in open("results/mutants/mutants.jsonl")]
ids = {m["strategy_id"] for m in M}
tasks = [t for t in load_tasks() if t["strategy_id"] in ids]
refs = reference_runs(tasks, pf, cfg["engine"])
gens = {(m["strategy_id"], m["type"]): {"code": m["code"]} for m in M}
df = score(tasks, gens, refs, pf, cfg["engine"], setting="mutant").rename(columns={"model": "type"})
df.to_csv("results/mutants/mutant_scores.csv", index=False)
print(df.groupby("type").agg(n=("action_match", "size"), compile_fail=("compile_fail", "sum"),
      AM_mean=("action_match", "mean"), AM_median=("action_match", "median"),
      AM_lt_0_9=("action_match", lambda x: (x < 0.9).mean()), AM_eq_1=("action_match", lambda x: (x == 1).mean()),
      ES_med=("abs_es", "median")).round(3).to_string())
