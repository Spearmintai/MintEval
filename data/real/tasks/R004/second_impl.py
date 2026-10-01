"""Independent second implementation of R004 (TurtleAI).

Different structure: ADX is computed with a literal transcription of jesse's _calculate_adx loop on
the 240-candle window (no rma trick), CHOP with jesse's formula on the window, ATR with an SMA-seeded
Wilder recursion over the full history, the 4h SMA200 from a pandas resample of the 15m data (a 4h
candle is used once the 15m bar closing its bucket has closed)."""
import math

import numpy as np
import pandas as pd


def jesse_adx_loop(h, l, c, period=14):
    n = len(c)
    TR = np.zeros(n)
    pDM = np.zeros(n)
    mDM = np.zeros(n)
    for i in range(1, n):
        TR[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        hd = h[i] - h[i - 1]
        ld = l[i - 1] - l[i]
        pDM[i] = hd if (hd > ld and hd > 0) else 0.0
        mDM[i] = ld if (ld > hd and ld > 0) else 0.0

    def ws(a):
        r = np.full(n, np.nan)
        r[period] = a[1:period + 1].sum()
        for i in range(period + 1, n):
            r[i] = r[i - 1] - r[i - 1] / period + a[i]
        return r
    tr, p, m = ws(TR), ws(pDM), ws(mDM)
    DX = np.full(n, np.nan)
    for i in range(period, n):
        if tr[i] != 0:
            dp, dm = 100 * p[i] / tr[i], 100 * m[i] / tr[i]
            DX[i] = 100 * abs(dp - dm) / (dp + dm) if dp + dm != 0 else 0.0
        else:
            DX[i] = 0.0
    adx = np.mean(DX[period:2 * period])
    for i in range(2 * period + 1, n):
        adx = (adx * (period - 1) + DX[i]) / period
    return adx


def atr_sma_seed(h, l, c, n=14):
    tr = np.r_[h[0] - l[0], np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])]
    out = np.full(len(c), np.nan)
    a = tr[:n].mean()
    out[n - 1] = a
    for i in range(n, len(c)):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def htf_sma_completed(P, minutes=240, n=200):
    t = P["time"].astype(np.int64)
    ms = minutes * 60000
    df = pd.DataFrame({"b": t // ms, "c": P["close"]})
    last = df.groupby("b")["c"].last()
    sma = last.rolling(n).mean()
    # bucket b is complete at bar i iff bar i closes the bucket (or a later bucket has started)
    closes = ((t + 15 * 60000) % ms) == 0
    b = t // ms
    out = np.full(len(t), np.nan)
    idx = {k: v for k, v in zip(sma.index, sma.to_numpy())}
    keys = np.array(sma.index)
    for i in range(len(t)):
        done = b[i] if closes[i] else b[i] - 1
        # last completed bucket <= done
        j = np.searchsorted(keys, done, side="right") - 1
        out[i] = idx[keys[j]] if j >= 0 else np.nan
    return out


def decide(P, pos_held, warmup):
    h, l, c = P["high"], P["low"], P["close"]
    n = len(c)
    atr = atr_sma_seed(h, l, c)
    up = pd.Series(h).shift(1).rolling(20).max().to_numpy()
    lo = pd.Series(l).shift(1).rolling(20).min().to_numpy()
    sma4h = htf_sma_completed(P)
    tgt = np.zeros(n)
    stop = np.full(n, np.nan)
    cur_stop = np.nan
    in_trade = False
    last_closed = -1
    for t in range(warmup, n):
        side = np.sign(pos_held[t])
        if in_trade and side == 0:
            last_closed = t
            in_trade = False
            cur_stop = np.nan
        if side != 0:
            cand = c[t] - side * 2.5 * atr[t]
            cur_stop = max(cur_stop, cand) if side > 0 else min(cur_stop, cand)
            tgt[t] = pos_held[t]
            stop[t] = cur_stop
            continue
        if t - last_closed <= 0:
            continue
        sig = 1 if c[t] > up[t] else (-1 if c[t] < lo[t] else 0)
        if sig == 0 or not ((sig > 0 and c[t] > sma4h[t]) or (sig < 0 and c[t] < sma4h[t])):
            continue
        w = slice(t - 239, t + 1)
        adx = jesse_adx_loop(h[w], l[w], c[w])
        hh, ll = h[t - 13:t + 1].max(), l[t - 13:t + 1].min()
        trs = np.maximum.reduce([h[t - 13:t + 1] - l[t - 13:t + 1], abs(h[t - 13:t + 1] - c[t - 14:t]),
                                 abs(l[t - 13:t + 1] - c[t - 14:t])])
        ch = 100 * (math.log10(trs.sum()) - math.log10(hh - ll)) / math.log10(14)
        if adx > 30 and ch < 40:
            tgt[t] = sig * min(1.0, 1.8 * min(1.0, 0.03 * c[t] / (2.5 * atr[t])))
            cur_stop = c[t] - sig * 2.5 * atr[t]
            stop[t] = cur_stop
            in_trade = True
    return {"target": tgt, "stop": stop}
