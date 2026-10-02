"""FRL Fig 2, equity panel: re-run reference and generated programs on SPY/QQQ regular-hours 15m bars
(2024-10-02 .. 2026-10-01, Polygon) and look for silent failures that first diverge on an overnight gap.

The generated programs are the existing open-setting BTC generations (GPT-5.4-mini, Claude Opus 5.5);
they take only bars and indicators as input, so no new model calls are needed. Engine settings are the
v0 ones (5 bp fee + 1 bp slippage, warm-up 960 bars), bars_per_year = 26 * 252.
A candidate is a compiled run whose positions first differ (t0) on the first bar of a session.
`ref_gap_stop` = the reference was in a position at the close of t0-1 and its stop was gapped through
at the open of t0 (so the engine filled it at the open).
Output: results/frl/equity_candidates.csv."""
import sys, json, multiprocessing as mp
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from minteval.engine import EngineConfig
from minteval.data import load_prices
from minteval.metrics import compare
from scripts.frl_examples import traced, tasks, GEN

cfg = yaml.safe_load(open("configs/base.yaml"))
ECFG = EngineConfig(**dict(cfg["engine"], bars_per_year=26 * 252))
EQ = {t: load_prices(f"data/prices/{t}_15m_rth_2024_2026.csv") for t in ("SPY", "QQQ")}
DAY = {t: pd.to_datetime(p["open_time"], unit="ms", utc=True).tz_convert("America/New_York").date
       for t, p in EQ.items()}


def job(arg):
    tk, sid, m = arg
    P = EQ[tk]
    try:
        rr, lr = traced(tasks[sid]["source"], prices=P, ecfg=ECFG)
        rg, lg = traced(GEN[(sid, m)], check=True, prices=P, ecfg=ECFG)
    except Exception as e:  # noqa: BLE001
        return {"ticker": tk, "strategy_id": sid, "model": m, "err": repr(e)[:200]}
    row = {"ticker": tk, "strategy_id": sid, "model": m, "err": rg.error or rr.error}
    if rg.error or rr.error:
        return row
    pack = lambda r: {"equity": r.equity, "target_q": r.target_q, "pos_q": r.pos_q, "trades": r.trades, "error": r.error}
    met = compare(pack(rr), pack(rg), ECFG)
    row.update(action_match=met["action_match"], abs_es=met["abs_es"], n_trades_ref=len(rr.trades))
    diff = np.nonzero(rr.pos_q != rg.pos_q)[0]
    if not len(diff):
        return row
    t0 = int(diff[0])
    O = P["open"]
    d = DAY[tk]
    s_prev = lr["stop"][t0 - 1]
    held = rr.pos_q[t0 - 1]
    gap_stop = bool(held != 0 and rr.pos_q[t0] == 0 and np.isfinite(s_prev) and
                    ((held > 0 and O[t0] <= s_prev) or (held < 0 and O[t0] >= s_prev)))
    t3 = min(t0 + 300, len(O) - 1)
    gp = (rr.equity[t3] / rr.equity[t0 - 1] - rg.equity[t3] / rg.equity[t0 - 1]) * 1e4
    row.update(t0=t0, session_open=bool(d[t0] != d[t0 - 1]), ref_gap_stop=gap_stop,
               llm_pos_t0=int(rg.pos_q[t0]), ref_pos_prev=int(held), stop_ref_prev=s_prev,
               stop_llm_prev=lg["stop"][t0 - 1], open_t0=O[t0], close_prev=P["close"][t0 - 1], gap_300=gp)
    return row


if __name__ == "__main__":
    d = pd.read_csv("results/frl/minteval_v0_frl.csv")
    d = d[(d.setting == "open") & d.model.isin(["gpt-5.4-mini", "claude-opus-5.5"]) & (d.compile_fail == 0)]
    args = [(tk, s, m) for tk in EQ for s, m in zip(d.strategy_id, d.model)]
    with mp.get_context("fork").Pool(max(1, mp.cpu_count() - 1)) as pool:
        rows = list(pool.imap_unordered(job, args, chunksize=4))
    c = pd.DataFrame(rows)
    c.to_csv("results/frl/equity_candidates.csv", index=False)
    ok = c[c.err.isna()]
    print(len(c), "runs;", len(ok), "ran;", "errors:", c.err.notna().sum())
    print(ok.groupby(["ticker", "model"]).action_match.agg(["mean", "count"]))
    g = ok[ok.ref_gap_stop.fillna(False).astype(bool)]
    print("first divergence = reference stop gapped through at the open:", len(g))
    print(g.sort_values("gap_300", key=abs, ascending=False).head(30)[
        ["ticker", "strategy_id", "model", "t0", "action_match", "llm_pos_t0", "ref_pos_prev",
         "stop_ref_prev", "stop_llm_prev", "close_prev", "open_t0", "gap_300"]].to_string())
