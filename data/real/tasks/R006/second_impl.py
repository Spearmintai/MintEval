"""Independent second implementation of R006 (Dual Thrust alpha + 3 % drawdown risk).

Different structure: 30m bars are rebuilt with pandas groupby on the timestamp bucket, the dual-thrust
lines are computed for each completed 30m bar and forward-mapped onto the 15m bars, then a loop runs
an explicit insight object (direction, generated time, close time) like LEAN's InsightCollection."""
import numpy as np
import pandas as pd


def decide(P, pos_held, warmup):
    t_open = P["time"].astype(np.int64)
    ms30 = 30 * 60000
    df = pd.DataFrame({"b": t_open // ms30, "h": P["high"], "l": P["low"], "c": P["close"]})
    g = df.groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    hh = g.h.rolling(20).max()
    ll = g.l.rolling(20).min()
    hc = g.c.rolling(20).max()
    lc = g.c.rolling(20).min()
    rng = np.maximum(hh - lc, hc - ll)
    up_by_bucket = (g.c + 0.63 * rng).to_dict()
    lo_by_bucket = (g.c - 0.63 * rng).to_dict()
    close_time = t_open + 15 * 60000
    c = P["close"]
    n = len(c)
    tgt = np.zeros(n)
    stop = np.full(n, np.nan)
    insight = None            # dict(dir, gen, close)
    prev_tgt = 0.0
    for t in range(warmup, n):
        now = close_time[t]
        held = np.sign(pos_held[t])
        if prev_tgt != 0 and held == 0:
            insight = None                      # risk model liquidated + cancelled
        last_done = (now // ms30) - 1           # last 30m bucket completed at this time
        up = up_by_bucket.get(last_done, np.nan)
        lo = lo_by_bucket.get(last_done, np.nan)
        if c[t] > up and held != 1:
            insight = {"dir": 1, "gen": now, "close": now + 5 * 86400000, "entry": c[t]}
        elif c[t] < lo and held != -1:
            insight = {"dir": -1, "gen": now, "close": now + 5 * 86400000, "entry": c[t]}
        if insight is not None and insight["close"] < now:
            insight = None
        tgt[t] = 0.0 if insight is None else insight["dir"]
        if insight is not None:
            stop[t] = insight["entry"] * (1 - 0.03 * insight["dir"])
        prev_tgt = tgt[t]
    return {"target": tgt, "stop": stop}
