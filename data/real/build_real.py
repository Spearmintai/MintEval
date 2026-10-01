"""Collect the Real-subset tasks (data/real/tasks/<id>/) into data/real/real_tasks.jsonl.

One JSON line per task with
  strategy_id   "R0xx"
  source        provenance string (framework / repository / strategy name)
  url, license, status
  source_code   reference.py text (the reference program run by the scorer)
  q             the instruction (intent_task.txt without the placeholder header and the
                bracketed added-facts note)
plus the fields the existing pipeline reads:
  scripts/run_models.py reads  strategy_id, q                       -> works on this file as is
  scripts/score.py      reads  n_jargon, prompt_len from the prompts file, and from the TASK file
                               source (= reference CODE), spec.signal.id, spec.risk, K_bits, tau_*,
                               n_registers, nesting_depth, mccabe, halstead, R_bench, ... (t.get)
NOTE: in results/tasks/tasks.jsonl the key "source" holds the reference CODE (evaluate.reference_runs
uses t["source"] as code), while this file uses "source" for provenance as requested. scripts/score.py
also hard-codes load_tasks() = results/tasks/tasks.jsonl. Use data/real/score_real.py, which maps
source_code -> source in memory and scores the open setting; nothing in minteval/ is modified.
`spec` is a stub ({"signal": {"id": "real"}, ..., "risk": []}) so evaluate.score() can fill
signal_id / n_risk; spec_match (closed setting) is meaningless for real tasks.

Usage: python data/real/build_real.py [--all]   (default: only tasks whose port checks pass)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minteval.complexity import state_complexity, static_complexity  # noqa: E402
from minteval.data import load_prices  # noqa: E402
from minteval.engine import EngineConfig, perf_stats, run_backtest  # noqa: E402
from minteval.pool import compile_strategy  # noqa: E402
from minteval.sandbox import check_source  # noqa: E402
from minteval.translate import jargon_hits, load_jargon  # noqa: E402

TASKS = ROOT / "data/real/tasks"
OUT = ROOT / "data/real/real_tasks.jsonl"
MIN_TRADES = 20


def instruction_text(raw: str) -> str:
    """intent_task.txt minus the placeholder header and the trailing [Facts added ...] note."""
    txt = re.sub(r"^\s*PLACEHOLDER[^\n]*\n", "", raw)
    txt = re.sub(r"\n\s*\[Facts added beyond[^\]]*\]\s*$", "", txt.strip(), flags=re.S)
    return " ".join(txt.split())


def build(include_all: bool = False):
    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ecfg = EngineConfig(**cfg["engine"])
    P = load_prices(str(ROOT / cfg["data"]["price_file"]))
    jargon = load_jargon(str(ROOT / "data/jargon.txt"))
    rows = []
    for d in sorted(p for p in TASKS.iterdir() if (p / "reference.py").exists()):
        meta = json.loads((d / "meta.json").read_text())
        code = (d / "reference.py").read_text()
        check_source(code)
        r = run_backtest(compile_strategy(code), P, ecfg, track_state=True)
        if r.error:
            raise RuntimeError(f"{d.name}: reference failed: {r.error}")
        st = perf_stats(r.equity, ecfg.initial_equity, ecfg.bars_per_year)
        pc = meta.get("port_checks", {})
        ok = (len(r.trades) >= MIN_TRADES and pc.get("sandbox_ok") is True
              and pc.get("second_impl_target_agreement") == 1.0)
        if not ok and not include_all:
            print(f"skip {d.name}: trades={len(r.trades)} port_checks={ {k: pc.get(k) for k in ('sandbox_ok', 'second_impl_target_agreement')} }")
            continue
        q = instruction_text((d / "intent_task.txt").read_text())
        hits = sorted(set(jargon_hits(q, jargon)))
        rows.append({
            "strategy_id": meta["id"],
            "source": meta["source"],
            "url": meta["url"],
            "license": meta["license"],
            "status": meta["status"],
            "q_status": "PLACEHOLDER_PENDING_HUMAN_REWRITE",
            "source_code": code,
            "q": q,
            "n_jargon": len(hits),
            "prompt_len": len(q.split()),
            "spec": {"signal": {"id": "real", "params": {}}, "direction": "real",
                     "filter": {"id": "no_filter", "params": {}}, "sizing": {"id": "real", "params": {}},
                     "risk": []},
            "state_features": meta.get("state_features", []),
            **state_complexity(r.state_spans), **static_complexity(code),
            "K_bits": None, "tau_bin": None,
            "R_bench": st["R"], "sharpe_bench": st["sharpe"], "maxdd_bench": st["maxdd"],
            "n_trades_bench": len(r.trades),
            "frac_bars_in_pos": float((r.pos_q[ecfg.warmup_bars:] != 0).mean()),
        })
    with open(OUT, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} tasks -> {OUT.relative_to(ROOT)}")
    return rows


if __name__ == "__main__":
    build(include_all="--all" in sys.argv)
