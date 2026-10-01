"""Score generations for the Real subset -> CSV (open setting only).

Mirrors scripts/score.py, with two differences that avoid touching minteval/*.py and scripts/score.py:
  * tasks come from data/real/real_tasks.jsonl (score.py hard-codes results/tasks/tasks.jsonl), and
    t["source"] is set to the reference CODE (t["source_code"]) in memory, as evaluate.reference_runs
    expects;
  * reference runs are cached under results/cache/ref_real/<hash of all reference sources> so an
    edited reference can never be served from a stale cache.
Closed-setting generation files are skipped: the building-block menu and spec_match do not apply to
real strategies.

Generate first (no paid API is called by this script):
  python scripts/run_models.py data/real/real_tasks.jsonl <model> open all results/generations_real
Usage: python data/real/score_real.py results/generations_real results/real_scores.csv
"""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minteval.data import load_prices  # noqa: E402
from minteval.evaluate import reference_runs, score, write_csv  # noqa: E402

TASK_FILE = ROOT / "data/real/real_tasks.jsonl"


def load_real_tasks(path=TASK_FILE):
    tasks = [json.loads(l) for l in open(path)]
    for t in tasks:
        t["provenance"] = t["source"]
        t["source"] = t["source_code"]
    return tasks


def main(gen_dir, out):
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    tasks = load_real_tasks()
    tag = hashlib.sha256("".join(t["source"] for t in tasks).encode()).hexdigest()[:10]
    refs = reference_runs(tasks, cfg["data"]["price_file"], cfg["engine"],
                          cache_dir=f"results/cache/ref_real/{tag}")
    C = load_prices(cfg["data"]["price_file"])["close"]
    lr = np.diff(np.log(C), prepend=np.log(C[0]))
    for t in tasks:
        m = refs[t["strategy_id"]]["pos_q"] != 0
        t["realized_vol"] = float(lr[m].std() * np.sqrt(cfg["engine"]["bars_per_year"])) if m.sum() > 1 else float("nan")
    ids = {t["strategy_id"] for t in tasks}
    frames = []
    for f in sorted(glob.glob(f"{gen_dir}/*.jsonl")):
        G = [json.loads(l) for l in open(f)]
        if not G or G[0]["setting"] != "open":
            print("skip (not open setting):", f)
            continue
        gens = {(g["strategy_id"], g["model"]): {"code": g["code"], "spec": g["spec"]}
                for g in G if g["strategy_id"] in ids}
        if gens:
            frames.append(score(tasks, gens, refs, cfg["data"]["price_file"], cfg["engine"],
                                timeout=cfg["sandbox"]["timeout_s"], setting="open"))
    if not frames:
        print("no open-setting generations for real tasks found in", gen_dir)
        return
    df = pd.concat(frames)
    print(write_csv(df, out))
    print(df.groupby("model")[["compile_fail", "action_match", "trade_f1", "abs_es"]]
          .agg(["mean", "median"]).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
