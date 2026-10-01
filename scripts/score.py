"""Score generations -> CSV. Usage: score.py <gen_dir> <prompts.jsonl> <out.csv>"""
import sys, json, glob, yaml
sys.path.insert(0, ".")
import pandas as pd
from minteval.evaluate import load_tasks, reference_runs, score, write_csv
cfg = yaml.safe_load(open("configs/base.yaml"))
gen_dir, prompts_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
pr = {p["strategy_id"]: p for p in map(json.loads, open(prompts_path))}
tasks = [t for t in load_tasks() if t["strategy_id"] in pr]
for t in tasks:
    t["n_jargon"] = pr[t["strategy_id"]]["n_jargon"]; t["prompt_len"] = pr[t["strategy_id"]]["prompt_len"]
    t["prompt_valid"] = int(pr[t["strategy_id"]]["valid"]); t["roundtrip"] = pr[t["strategy_id"]].get("recovery")
refs = reference_runs(tasks, cfg["data"]["price_file"], cfg["engine"])
import numpy as np
from minteval.data import load_prices
C = load_prices(cfg["data"]["price_file"])["close"]
lr = np.diff(np.log(C), prepend=np.log(C[0]))
for t in tasks:   # annualised realised vol of BTC over the bars the reference holds a position
    m = refs[t["strategy_id"]]["pos_q"] != 0
    t["realized_vol"] = float(lr[m].std() * np.sqrt(cfg["engine"]["bars_per_year"])) if m.sum() > 1 else float("nan")
frames = []
for f in sorted(glob.glob(f"{gen_dir}/*.jsonl")):
    G = [json.loads(l) for l in open(f)]
    setting = G[0]["setting"]
    gens = {(g["strategy_id"], g["model"]): {"code": g["code"], "spec": g["spec"]} for g in G if g["strategy_id"] in pr}
    frames.append(score(tasks, gens, refs, cfg["data"]["price_file"], cfg["engine"],
                        timeout=cfg["sandbox"]["timeout_s"], setting=setting))
df = pd.concat(frames)
print(write_csv(df, out))
print(df.groupby(["model", "setting"])[["compile_fail", "spec_match", "action_match", "trade_f1", "abs_es"]]
      .agg(["mean", "median"]).round(3).to_string())
