"""Acceptance tests 1 (self-match, small sample), 4 (determinism) and 6 (tau distribution)."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from minteval.assembler import render, sample_spec
from minteval.engine import EngineConfig
from conftest import PRICE_FILE, ROOT

TASKS = ROOT / "results/tasks/tasks.jsonl"


def _cfg():
    return {"initial_equity": 10000.0, "fee_bp": 5.0, "slippage_bp": 1.0, "quant_delta": 0.05,
            "bars_per_year": 35040, "warmup_bars": 960}


@pytest.mark.skipif(not TASKS.exists(), reason="task set not built")
def test_self_match_sample():
    from minteval.evaluate import load_tasks, reference_runs, score
    tasks = load_tasks(str(TASKS))[::80]          # 10 tasks spread across tau bins
    refs = reference_runs(tasks, PRICE_FILE, _cfg())
    gens = {(t["strategy_id"], "reference"): {"code": t["source"], "spec": t["spec"]} for t in tasks}
    df = score(tasks, gens, refs, PRICE_FILE, _cfg(), setting="closed", workers=4)
    print(df[["strategy_id", "action_match", "trade_f1", "abs_es", "spec_match"]].to_string())
    assert (df.compile_fail == 0).all()
    assert (df.action_match == 1.0).all() and (df.trade_f1 == 1.0).all() and (df.abs_es == 0.0).all()


@pytest.mark.skipif(not TASKS.exists(), reason="task set not built")
def test_mutation_is_detected():
    """Changing one numeric parameter of the reference must (almost always) break equivalence."""
    from minteval.evaluate import load_tasks, reference_runs, score
    from minteval.primitives import REGISTRY
    tasks = load_tasks(str(TASKS))[3::100]
    refs = reference_runs(tasks, PRICE_FILE, _cfg())
    gens = {}
    for t in tasks:
        sp = copy.deepcopy(t["spec"])
        r = sp["risk"][0]
        grid = REGISTRY[r["id"]].params
        name = next(n for n, g in grid.items() if not isinstance(g[0], bool))
        g = grid[name]
        r["params"][name] = g[(g.index(r["params"][name]) + 1) % len(g)]
        gens[(t["strategy_id"], "mutant")] = {"code": render(sp)}
    df = score(tasks, gens, refs, PRICE_FILE, _cfg(), workers=4)
    print(df[["strategy_id", "action_match", "trade_f1", "abs_es"]].to_string())
    assert (df.action_match < 1.0).mean() >= 0.75


def test_pool_deterministic(tmp_path):
    import yaml
    from minteval.pool import build_pool
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    cfg["data"]["price_file"] = PRICE_FILE
    hs = []
    for i in range(2):
        out = tmp_path / f"c{i}.jsonl"
        build_pool(cfg, 24, cfg["seed"], str(out), workers=4)
        hs.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert hs[0] == hs[1]


@pytest.mark.skipif(not TASKS.exists(), reason="task set not built")
def test_tau_distribution_bins():
    """Acceptance test 6: histogram of tau_max; alarm if any bin < 120."""
    rep = json.loads((ROOT / "results/tasks/selection_report.json").read_text())
    tasks = [json.loads(l) for l in open(TASKS)]
    taus = np.array([t["tau_max"] for t in tasks])
    bins = np.array([t["tau_bin"] for t in tasks])
    counts = np.bincount(bins, minlength=5)
    hist, edges = np.histogram(np.log10(taus), bins=20)
    print("tau_max log10 histogram:", list(zip(np.round(edges[:-1], 2).tolist(), hist.tolist())))
    print("per-bin counts:", counts.tolist(), "pool bin counts:", rep["pool_bin_counts"])
    assert (counts >= 120).all(), f"ALARM: bin counts {counts.tolist()}"
    assert all(v >= 120 for v in rep["pool_bin_counts"].values())
