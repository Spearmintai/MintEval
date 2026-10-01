"""Independent second implementation of R002 (TradingView_RSI): vectorised cross events with a
jesse-style SMA-seeded RSI; one loop over the realised position path for the bracket levels."""
import numpy as np


def rsi_sma_seed(x, n):
    d = np.diff(x)
    g, l = np.clip(d, 0, None), np.clip(-d, 0, None)
    out = np.full(len(x), np.nan)
    a, b = g[:n].mean(), l[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            a = (a * (n - 1) + g[i - 1]) / n
            b = (b * (n - 1) + l[i - 1]) / n
        out[i] = 100.0 if b == 0 else 100 - 100 / (1 + a / b)
    return out


def decide(P, pos_held, warmup):
    c = P["close"]
    r = rsi_sma_seed(c, 5)
    rp = np.r_[np.nan, r[:-1]]
    up35 = (rp <= 35) & (r > 35)
    dn75 = (rp >= 75) & (r < 75)
    dn10 = (rp >= 10) & (r < 10)
    n = len(c)
    tgt = np.zeros(n)
    stop = np.full(n, np.nan)
    take = np.full(n, np.nan)
    entry = np.nan
    for t in range(warmup, n):
        long_now = pos_held[t] > 0
        if not long_now:
            entry = np.nan
        if long_now and not (dn75[t] or dn10[t]):
            tgt[t] = pos_held[t]
        else:
            entry = np.nan
            if up35[t]:
                tgt[t] = 1.0
                entry = c[t]
        if not np.isnan(entry):
            stop[t], take[t] = entry * 0.95, entry * 1.10
    return {"target": tgt, "stop": stop, "take": take}
