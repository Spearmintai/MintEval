import sys, json, yaml
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from minteval.evaluate import load_tasks
from minteval.llm import client_from_cfg
from minteval.translate import load_jargon, translate_task
cfg = yaml.safe_load(open("configs/base.yaml"))
tc = cfg["translate"]
tasks = load_tasks()
ids = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
out = sys.argv[2] if len(sys.argv) > 2 else "results/tasks/prompts.jsonl"
if ids: tasks = [t for t in tasks if t["strategy_id"] in ids]
cl = client_from_cfg(tc["model"]); jg = load_jargon()
rd = client_from_cfg(tc["roundtrip"]["model"]) if tc["roundtrip"]["enabled"] else None
f = lambda t: translate_task(t, cl, jg, tc["min_words"], tc["max_words"], tc["max_attempts"], tc["temperature"], cfg["seed"], reader=rd)
with ThreadPoolExecutor(16) as ex: res = list(ex.map(f, tasks))
with open(out, "w") as fh:
    for r in res: fh.write(json.dumps(r) + "\n")
print("recovery", [r.get("recovery") for r in res][:40]); print(f"valid {sum(r['valid'] for r in res)}/{len(res)}; attempts", [r["n_attempts"] for r in res][:40])
