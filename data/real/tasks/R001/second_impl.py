"""Independent second implementation of R001 (RSI2) for differential checking.

Structure: all indicators precomputed with pandas / own numpy code (jesse-style SMA-seeded Wilder RSI),
then signal arrays built vectorised; the only loop applies the exit-then-entry rule to the realised
position path handed in by the checker (pos_held[t] = position held at the close of bar t).
"""
import numpy as np
import pandas as pd


def wilder_rsi_sma_seed(x, n):
    d = np.diff(x)
    g = np.where(d > 0, d, 0.0)
    l = np.where(d < 0, -d, 0.0)
    ag = np.full(len(x), np.nan)
    al = np.full(len(x), np.nan)
    a, b = g[:n].mean(), l[:n].mean()
    ag[n], al[n] = a, b
    for i in range(n, len(d)):
        a = (a * (n - 1) + g[i]) / n
        b = (b * (n - 1) + l[i]) / n
        ag[i + 1], al[i + 1] = a, b
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + ag / al)
    return np.where(al == 0, 100.0, rsi)


def decide(P, pos_held, warmup):
    c = pd.Series(P["close"])
    sma200 = c.rolling(200).mean().to_numpy()
    sma5 = c.rolling(5).mean().to_numpy()
    rsi = wilder_rsi_sma_seed(P["close"], 2)
    px = P["close"]
    long_sig = (px > sma200) & (rsi <= 10)
    short_sig = (px < sma200) & (rsi >= 90)
    exit_long = px > sma5
    exit_short = px < sma5
    n = len(px)
    tgt = np.zeros(n)
    for t in range(warmup, n):
        p = pos_held[t]
        if p > 0 and not exit_long[t]:
            tgt[t] = p
            continue
        if p < 0 and not exit_short[t]:
            tgt[t] = p
            continue
        tgt[t] = -1.0 if short_sig[t] else (1.0 if long_sig[t] else 0.0)
    return {"target": tgt}
