"""Acceptance test 1: every reference program, fed back as the 'model output' through the
full evaluation path (static check -> sandbox subprocess -> metrics), must give
ActionMatch = 1, TradeF1 = 1, |ES| = 0. Also acceptance test 4 (determinism): the
scoring CSV is produced twice and the hashes must match."""
import sys, yaml, json
sys.path.insert(0, ".")
from minteval.evaluate import load_tasks, reference_runs, score, write_csv
cfg = yaml.safe_load(open("configs/base.yaml"))
ecfg = cfg["engine"]
pf = cfg["data"]["price_file"]
tasks = load_tasks()
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(tasks)
tasks = tasks[:n]
refs = reference_runs(tasks, pf, ecfg)
gens = {(t["strategy_id"], "reference"): {"code": t["source"], "spec": t["spec"]} for t in tasks}
hashes = []
for rep in range(2):
    df = score(tasks, gens, refs, pf, ecfg, setting="closed")
    hashes.append(write_csv(df, f"results/selftest/self_match_run{rep}.csv"))
summary = {"n": len(df), "compile_fail": int(df.compile_fail.sum()),
           "action_match_min": float(df.action_match.min()), "trade_f1_min": float(df.trade_f1.min()),
           "abs_es_max": float(df.abs_es.max()), "spec_match_min": float(df.spec_match.min()),
           "errors": df.error_type[df.error_type != ""].value_counts().to_dict(),
           "csv_sha256": hashes, "deterministic": hashes[0] == hashes[1]}
print(json.dumps(summary, indent=1))
ok = (summary["compile_fail"] == 0 and summary["action_match_min"] == 1.0 and summary["trade_f1_min"] == 1.0
      and summary["abs_es_max"] == 0.0 and summary["deterministic"])
print("SELF-MATCH + DETERMINISM:", "PASS" if ok else "FAIL")
json.dump(summary, open("results/selftest/summary.json", "w"), indent=1)
sys.exit(0 if ok else 1)
