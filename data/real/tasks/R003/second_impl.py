"""Independent second implementation of R003 (SimpleBollinger): pandas rolling Bollinger on hl2 and
an Ichimoku cloud computed the textbook way (spans shifted forward by 26-1 bars), loop only over the
realised position path."""
import numpy as np
import pandas as pd


def decide(P, pos_held, warmup):
    h, l, c = pd.Series(P["high"]), pd.Series(P["low"]), P["close"]
    hl2 = (h + l) / 2
    mid = hl2.rolling(20).mean().to_numpy()
    sd = hl2.rolling(20).std(ddof=0).to_numpy()
    upper = mid + 2 * sd

    def midpoint(n):
        return ((h.rolling(n).max() + l.rolling(n).min()) / 2)
    span_a = ((midpoint(9) + midpoint(26)) / 2).shift(25).to_numpy()
    span_b = midpoint(52).shift(25).to_numpy()
    entry = (c > upper) & (c > span_a) & (c > span_b)
    exit_ = c < mid
    n = len(c)
    tgt = np.zeros(n)
    for t in range(warmup, n):
        if pos_held[t] > 0:
            tgt[t] = 0.0 if exit_[t] else pos_held[t]
        else:
            tgt[t] = 1.0 if entry[t] else 0.0
    return {"target": tgt}
