"""Differential check of the jesse-derived ports (R001-R005) against jesse's OWN indicator code.

The unmodified indicator modules of jesse-ai/jesse @ 7a01f999 (MIT) live in
data/real/sources/jesse-core-7a01f999/jesse/indicators (git-ignored clone area) together with a
minimal stub of jesse.helpers. For a sample of bars (every bar where the reference opened or closed a
trade + random bars) the original strategy conditions are re-evaluated with jesse's functions on
jesse-format candles ([ts, open, close, high, low, volume], same slicing as jesse: non-sequential
calls see the last 240 candles) and compared with the decision the reference port took on that bar
in the MintEval engine run.

Usage: python data/real/diff_jesse.py [n_random]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/real/sources/jesse-core-7a01f999"))

import jesse.indicators as ta  # noqa: E402

from minteval.data import load_prices  # noqa: E402
from minteval.engine import EngineConfig, run_backtest  # noqa: E402
from minteval.pool import compile_strategy  # noqa: E402

TASKS = ROOT / "data/real/tasks"
SEQ_TAIL = 1000          # candles passed to sequential=True indicators (EMA/RSI seeds converge)


def crossed(s1, s2, direction):
    if direction == "above":
        return s1[-2] <= s2 and s1[-1] > s2 if np.isscalar(s2) else (s1[-2] <= s2[-2] and s1[-1] > s2[-1])
    return s1[-2] >= s2 and s1[-1] < s2 if np.isscalar(s2) else (s1[-2] >= s2[-2] and s1[-1] < s2[-1])


def j_r001(C, t, held):
    c = C[: t + 1]
    price = c[-1, 2]
    if held > 0:
        return "exit" if price > ta.sma(c, 5) else "hold"
    if held < 0:
        return "exit" if price < ta.sma(c, 5) else "hold"
    r = ta.rsi(c, 2)
    s = ta.sma(c, 200)
    if price < s and r >= 90:
        return "short"
    if price > s and r <= 10:
        return "long"
    return "flat"


def j_r002(C, t, held):
    c = C[max(0, t + 1 - SEQ_TAIL): t + 1]
    r = ta.rsi(c, 5, sequential=True)
    if held > 0:
        return "exit" if (crossed(r, 75, "below") or crossed(r, 10, "below")) else "hold"
    return "long" if crossed(r, 35, "above") else "flat"


def j_r003(C, t, held):
    c = C[: t + 1]
    price = c[-1, 2]
    bb = ta.bollinger_bands(c, source_type="hl2")
    if held > 0:
        return "exit" if price < bb[1] else "hold"
    ich = ta.ichimoku_cloud(c)
    return "long" if (price > bb[0] and price > ich.span_a and price > ich.span_b) else "flat"


def j_r004(C, C4, n4_done, t, held, last_closed_ok):
    """Entry conditions only (stop ratchet compared through ATR separately)."""
    if held != 0:
        return "hold"
    c = C[: t + 1]
    price = c[-1, 2]
    dc = ta.donchian(c[:-1], period=20)
    c4 = C4[: n4_done[t]]
    ma = ta.sma(c4, 200) if len(c4) >= 200 else np.nan
    adx = ta.adx(c) > 30
    chop = ta.chop(c) < 40
    if price > dc.upperband and price > ma and adx and chop and last_closed_ok:
        return "long"
    if price < dc.lowerband and price < ma and adx and chop and last_closed_ok:
        return "short"
    return "flat"


def j_r005(C, t, held, entry):
    c = C[max(0, t + 1 - SEQ_TAIL): t + 1]
    f = ta.ema(c, 8, sequential=True)
    s = ta.ema(c, 22, sequential=True)
    if held > 0:
        pnl = (c[-1, 2] - entry) / entry * 100
        any_cross = crossed(f, s, "above") or crossed(f, s, "below")
        return "exit" if (pnl > 15.3 or any_cross) else "hold"

    def is_dp(i):
        return abs(c[i, 1] - c[i, 2]) * 100 / c[i, 1] > 2.3
    dp = is_dp(-1) or is_dp(-2) or is_dp(-3) or abs(c[-3, 1] - c[-1, 2]) * 100 / c[-3, 1] > 2.3
    return "long" if (crossed(f, s, "above") and not dp) else "flat"


def ref_label(pos, k, delta):
    tgt = k * delta
    if pos == 0:
        return "flat" if k == 0 else ("long" if k > 0 else "short")
    if k == 0:
        return "exit"
    return "hold"


def main(n_random=1500):
    cfg_all = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ecfg = EngineConfig(**cfg_all["engine"])
    P = load_prices(str(ROOT / cfg_all["data"]["price_file"]))
    n = len(P["close"])
    C = np.column_stack([P["open_time"], P["open"], P["close"], P["high"], P["low"], P["volume"]]).astype(float)
    # completed 4h candles in jesse format + number completed at each 15m close
    tt = P["open_time"].astype(np.int64)
    ms4 = 240 * 60000
    b = tt // ms4
    starts = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    ends = np.r_[starts[1:], n]
    C4 = np.column_stack([b[starts] * ms4, P["open"][starts], P["close"][ends - 1],
                          np.maximum.reduceat(P["high"], starts), np.minimum.reduceat(P["low"], starts),
                          np.add.reduceat(P["volume"], starts)])
    idx = np.repeat(np.arange(len(starts)), ends - starts)
    n4_done = np.where(((tt + 15 * 60000) % ms4) == 0, idx + 1, idx)
    rng = np.random.default_rng(20261001)
    report = {}
    for tid in ("R001", "R002", "R003", "R004", "R005"):
        src = (TASKS / tid / "reference.py").read_text()
        r = run_backtest(compile_strategy(src), P, ecfg)
        pos = r.pos_q * ecfg.quant_delta
        k = r.target_q
        w = ecfg.warmup_bars
        events = [t for t in range(w, n) if (pos[t] == 0) != (k[t] == 0) or (pos[t] != 0 and k[t] == 0)]
        rand = list(rng.choice(np.arange(w, n), size=n_random, replace=False))
        sample = sorted(set(events) | set(int(x) for x in rand))
        # state needed by R004 (cooldown) and R005 (entry price)
        entry = np.full(n, np.nan)
        cool_ok = np.ones(n, bool)
        cur = np.nan
        for t in range(w, n):
            if pos[t] == 0:
                cur = np.nan
            if not np.isnan(cur):
                entry[t] = cur
            if pos[t] == 0 and k[t] != 0:
                cur = P["close"][t]
            if t > w and pos[t] == 0 and pos[t - 1] != 0 and k[t - 1] != 0:
                cool_ok[t] = False           # stopped out during bar t
        mism = []
        for t in sample:
            ref = ref_label(pos[t], int(k[t]), ecfg.quant_delta)
            if tid == "R001":
                j = j_r001(C, t, pos[t])
            elif tid == "R002":
                j = j_r002(C, t, pos[t])
            elif tid == "R003":
                j = j_r003(C, t, pos[t])
            elif tid == "R004":
                j = j_r004(C, C4, n4_done, t, pos[t], cool_ok[t])
            else:
                j = j_r005(C, t, pos[t], entry[t])
            if j != ref:
                mism.append((int(t), ref, j))
        report[tid] = {"bars_checked": len(sample), "event_bars": len(events), "mismatches": len(mism),
                       "agreement": 1 - len(mism) / len(sample), "first_mismatches": mism[:10]}
        print(tid, json.dumps(report[tid]), flush=True)
    # ATR used by the R004 stop ratchet: jesse atr vs ind.atr
    from minteval.indicators import _atr
    a_ref = _atr(P["high"], P["low"], P["close"], 14)
    ts = rng.choice(np.arange(1000, n), size=500, replace=False)
    rel = max(abs(ta.atr(C[: t + 1]) / a_ref[t] - 1) for t in ts)
    report["R004_atr_max_rel_diff"] = float(rel)
    print("R004 ATR max rel diff (500 bars):", rel)
    out = ROOT / "data/real/diff_jesse_report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
