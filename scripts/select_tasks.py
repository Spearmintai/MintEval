import sys, json, yaml
sys.path.insert(0, ".")
from minteval.select import select_tasks
cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"))
tasks, rep = select_tasks("results/pool/candidates.jsonl", "results/tasks", cfg["pool"], cfg["seed"])
print(json.dumps({k: v for k, v in rep.items() if k != "bins"}, indent=1))
import pandas as pd; print(pd.DataFrame(rep["bins"]).round(3).to_string())
