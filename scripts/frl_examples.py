"""FRL Fig 2, step 1: find clean silent-failure examples.

Candidates: open-setting silent failures (compiled, ActionMatch < 0.9) of GPT-5.4-mini and Claude Opus 5.5
with ActionMatch in [0.5, 0.85]. Both programs are re-run with a wrapper that logs the stop level each
returns, and the first bar on which their positions differ (t0) is located. Kept if the equity gap opened
in the 300 bars from t0 (returns rebased at t0-1) is >= 300 bp. Writes results/frl/fig2_candidates.csv and a pickle
of per-bar traces for plotting (scripts/frl_fig2.py)."""
import sys, json, pickle, multiprocessing as mp
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from minteval.engine import EngineConfig, run_backtest
from minteval.data import load_prices
from minteval.sandbox import check_source

cfg = yaml.safe_load(open("configs/base.yaml"))
P = load_prices(cfg["data"]["price_file"])
ECFG = EngineConfig(**cfg["engine"])
tasks = {t["strategy_id"]: t for t in map(json.loads, open("results/tasks/tasks.jsonl"))}
GEN = {}
for f, m in (("results/generations/gpt-5.4-mini__open.jsonl", "gpt-5.4-mini"),
             ("results/generations_frontier/claude-opus-5.5__open.jsonl", "claude-opus-5.5")):
    for g in map(json.loads, open(f)):
        GEN[(g["strategy_id"], m)] = g["code"]


def traced(src, check=False, prices=None, ecfg=None):
    prices, ecfg = prices or P, ecfg or ECFG
    if check:
        check_source(src)
    g = {"__name__": "s"}
    exec(compile(src, "<s>", "exec"), g)
    f = g["strategy"]
    lv = {k: np.full(len(prices["close"]), np.nan) for k in ("stop", "take")}

    def w(hist, state, pos, ind):
        out = f(hist, state, pos, ind)
        for k in lv:
            v = out.get(k) if isinstance(out, dict) else None
            if v is not None:
                lv[k][len(hist.close) - 1] = float(v)
        return out
    r = run_backtest(w, prices, ecfg)
    return r, lv


def job(key):
    sid, m = key
    try:
        rr, sr = traced(tasks[sid]["source"])
        rg, sg = traced(GEN[key], check=True)
    except Exception as e:  # noqa: BLE001
        return key, {"err": repr(e)}
    if rg.error:
        return key, {"err": rg.error}
    diff = np.nonzero(rr.pos_q != rg.pos_q)[0]
    if not len(diff):
        return key, {"err": "no divergence"}
    t0 = int(diff[0])
    same = np.nonzero(rr.pos_q[t0:] == rg.pos_q[t0:])[0]
    t1 = t0 + int(same[0]) if len(same) else len(rr.pos_q) - 1
    e0 = ECFG.initial_equity
    gp = lambda t: (rr.equity[t] / rr.equity[t0 - 1] - rg.equity[t] / rg.equity[t0 - 1]) * 1e4
    t3 = min(t0 + 300, len(rr.equity) - 1)
    # first bar (<= t0) at which the stop levels the two programs return differ
    a, b = sr["stop"][:t0 + 1], sg["stop"][:t0 + 1]
    sd = np.nonzero(~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-6)))[0]
    kind = ("missed_entry" if rr.pos_q[t0] != 0 and rg.pos_q[t0] == 0 and rr.pos_q[t0 - 1] == 0 else
            "extra_entry" if rg.pos_q[t0] != 0 and rr.pos_q[t0] == 0 and rg.pos_q[t0 - 1] == 0 else
            "llm_exits_early" if rg.pos_q[t0] == 0 else "llm_holds_longer" if rr.pos_q[t0] == 0 else "size")
    tr = {"ref": dict(pos=rr.pos_q, eq=rr.equity, **sr), "llm": dict(pos=rg.pos_q, eq=rg.equity, **sg)}
    return key, {"t0": t0, "t1": t1, "episode": t1 - t0, "gap_ep": float(gp(t1)), "gap_300": float(gp(t3)),
                 "kind": kind, "stop_div": int(sd[0]) if len(sd) else -1,
                 "stop_ref": int(np.isfinite(sr["stop"]).any()), "stop_llm": int(np.isfinite(sg["stop"]).any()),
                 "trace": tr}


if __name__ == "__main__":
    d = pd.read_csv("results/frl/minteval_v0_frl.csv")
    d = d[(d.setting == "open") & d.model.isin(["gpt-5.4-mini", "claude-opus-5.5"]) & (d.compile_fail == 0)]
    d = d[d.action_match.between(0.5, 0.85)]
    keys = list(zip(d.strategy_id, d.model))
    rows, traces = [], {}
    with mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1)) as pool:
        for key, r in pool.imap_unordered(job, keys):
            tr = r.pop("trace", None)
            rows.append({"strategy_id": key[0], "model": key[1], **r})
            if tr is not None:
                traces[key] = tr
    c = pd.DataFrame(rows).merge(d[["strategy_id", "model", "action_match", "signal_id"]], on=["strategy_id", "model"])
    c["risk"] = c.strategy_id.map(lambda s: "+".join(r["id"] for r in tasks[s]["spec"]["risk"]))
    c["keep"] = (c.gap_300.abs() >= 300)
    c.sort_values(["keep", "gap_300"], ascending=False).to_csv("results/frl/fig2_candidates.csv", index=False)
    pickle.dump({k: v for k, v in traces.items() if k in set(zip(c[c.keep].strategy_id, c[c.keep].model))},
                open("results/frl/fig2_traces.pkl", "wb"))
    print(len(c), "candidates;", int(c.keep.sum()), "kept")
    print(c[c.keep].drop(columns=["err"], errors="ignore").to_string())
