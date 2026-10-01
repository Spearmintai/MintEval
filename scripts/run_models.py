"""Usage: run_models.py <prompts.jsonl> [model,...|all] [open,closed] [n_tasks|all]"""
import sys, json, yaml
sys.path.insert(0, ".")
from minteval.runner import run_model
cfg = yaml.safe_load(open("configs/base.yaml"))
P = [json.loads(l) for l in open(sys.argv[1])]
names = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else [m["name"] for m in cfg["models"]]
settings = sys.argv[3].split(",") if len(sys.argv) > 3 else cfg["generation"]["settings"]
if len(sys.argv) > 4 and sys.argv[4] != "all":
    P = P[:int(sys.argv[4])]
prompts = {p["strategy_id"]: p["q"] for p in P}
out_dir = sys.argv[5] if len(sys.argv) > 5 else "results/generations"
for m in cfg["models"]:
    if m["name"] not in names: continue
    for s in settings:
        res = run_model(m, prompts, s, cfg["generation"], out_dir)
        nc = sum(r["code"] is None for r in res.values())
        print(m["name"], s, "tasks", len(res), "no-code", nc, flush=True)
