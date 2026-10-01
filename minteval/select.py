"""Filter the candidate pool and draw the tau-stratified task set."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assembler import render


def select_tasks(cand_path: str, out_dir: str, pcfg: dict, seed: int):
    rows = [json.loads(l) for l in open(cand_path)]
    df = pd.DataFrame(rows)
    keep = (df.error.isna() & (df.n_trades_bench >= pcfg["min_trades"]) & ~df.bust_early & ~df.always_flat)
    pool = df[keep].copy()
    col = pcfg["bin_on"]
    edges = np.quantile(pool[col], np.linspace(0, 1, pcfg["n_bins"] + 1))
    edges[-1] += 1
    pool["tau_bin"] = np.clip(np.searchsorted(edges, pool[col], side="right") - 1, 0, pcfg["n_bins"] - 1)
    rng = np.random.default_rng(seed)
    picks, alarms = [], []
    counts = pool.tau_bin.value_counts().sort_index()
    rm = pcfg.get("r_match")
    if rm:
        r_edges = np.linspace(rm["lo"], rm["hi"], rm["n_bands"] + 1)
        pool["r_band"] = np.searchsorted(r_edges, pool.R_bench, side="right") - 1
        pool.loc[(pool.R_bench < rm["lo"]) | (pool.R_bench >= rm["hi"]), "r_band"] = -1
    for b in range(pcfg["n_bins"]):
        sub = pool[pool.tau_bin == b]
        if len(sub) < 120:
            alarms.append(f"ALARM: tau bin {b} has only {len(sub)} pool samples (< 120)")
        if not rm:
            k = min(pcfg["per_bin"], len(sub))
            if k < pcfg["per_bin"]:
                alarms.append(f"ALARM: tau bin {b} can supply only {k} < {pcfg['per_bin']} tasks")
            picks.append(sub.iloc[np.sort(rng.choice(len(sub), size=k, replace=False))])
            continue
        per_band = pcfg["per_bin"] // rm["n_bands"]
        taken = {}
        for rb in range(rm["n_bands"]):
            cell = sub[sub.r_band == rb]
            k = min(per_band, len(cell))
            if k < per_band:
                alarms.append(f"ALARM: cell (tau bin {b}, R band {rb}) has only {len(cell)} < {per_band}")
            taken[rb] = cell.iloc[np.sort(rng.choice(len(cell), size=k, replace=False))]
        # spill a cell's deficit over to the nearest band(s) of the same tau bin
        for rb in range(rm["n_bands"]):
            deficit = per_band - len(taken[rb])
            for nb in sorted(range(rm["n_bands"]), key=lambda x: (abs(x - rb), x)):
                if deficit <= 0:
                    break
                if nb == rb:
                    continue
                rest = sub[(sub.r_band == nb) & ~sub.index.isin(taken[nb].index)]
                k = min(deficit, len(rest))
                extra = rest.iloc[np.sort(rng.choice(len(rest), size=k, replace=False))]
                taken[nb] = pd.concat([taken[nb], extra])
                alarms.append(f"spill: {k} task(s) for (tau bin {b}, R band {rb}) taken from R band {nb}")
                deficit -= k
        picks.extend(taken[rb] for rb in range(rm["n_bands"]))
    tasks = pd.concat(picks).reset_index(drop=True)
    tasks["strategy_id"] = [f"S{i:04d}" for i in range(len(tasks))]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tasks.jsonl", "w") as f:
        for _, r in tasks.iterrows():
            d = r.to_dict()
            d["source"] = render(d["spec"])
            f.write(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    summ = tasks.groupby("tau_bin").agg(
        n=("strategy_id", "size"), tau_lo=(col, "min"), tau_hi=(col, "max"),
        R_med=("R_bench", "median"), R_q1=("R_bench", lambda x: x.quantile(.25)),
        R_q3=("R_bench", lambda x: x.quantile(.75)), K_med=("K_bits", "median"),
        trades_med=("n_trades_bench", "median"))
    report = {"pool_size": int(len(df)), "after_filter": int(keep.sum()),
              "filtered_out": {"error": int(df.error.notna().sum()),
                               "few_trades": int((df.n_trades_bench < pcfg["min_trades"]).sum()),
                               "bust_early": int(df.bust_early.sum()),
                               "always_flat": int(df.always_flat.sum())},
              "bin_edges": edges.tolist(), "pool_bin_counts": counts.to_dict(),
              "alarms": alarms,
              "cell_counts": (pd.crosstab(pool.tau_bin, pool.r_band).to_dict() if rm else None), "bins": summ.reset_index().to_dict(orient="records")}
    (out / "selection_report.json").write_text(json.dumps(report, indent=1, default=str))
    return tasks, report
