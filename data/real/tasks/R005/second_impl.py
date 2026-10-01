"""Independent second implementation of R005 (SimplEma): pandas ewm (SMA-seeded variant, as jesse),
vectorised pump/dump filter and cross arrays; the loop replays the jesse event handlers
(on_stop_loss) on the realised position path."""
import numpy as np
import pandas as pd


def ema_sma_seed(x, n):
    a = 2 / (n + 1)
    out = np.full(len(x), np.nan)
    e = x[:n].mean()
    out[n - 1] = e
    for i in range(n, len(x)):
        e = a * x[i] + (1 - a) * e
        out[i] = e
    return out


def decide(P, pos_held, warmup):
    o, c = P["open"], P["close"]
    f, s = ema_sma_seed(c, 8), ema_sma_seed(c, 22)
    fp, sp = np.r_[np.nan, f[:-1]], np.r_[np.nan, s[:-1]]
    up = (fp <= sp) & (f > s)
    dn = (fp >= sp) & (f < s)
    body = (np.abs(o - c) * 100 / o) > 2.3
    b1 = np.r_[False, body[:-1]]
    b2 = np.r_[False, False, body[:-2]]
    o2 = np.r_[np.nan, np.nan, o[:-2]]
    multi = (np.abs(o2 - c) * 100 / o2) > 2.3
    dp = body | b1 | b2 | multi
    n = len(c)
    tgt = np.zeros(n)
    stop = np.full(n, np.nan)
    losses, mult = 0, 1.0
    entry = None
    asked_exit = False
    for t in range(warmup, n):
        held = pos_held[t] > 0
        if entry is not None and not held:
            if not asked_exit:
                losses += 1
                mult *= 1.66
            entry, asked_exit = None, False
        if held:
            if (c[t] - entry) / entry * 100 > 15.3 or up[t] or dn[t]:
                asked_exit = True
                tgt[t] = 0.0
            else:
                tgt[t] = pos_held[t]
                stop[t] = entry * (1 - 0.172)
            continue
        if up[t] and not dp[t]:
            tgt[t] = 0.1 * (mult if losses <= 4 else 1.0)
            entry = c[t]
            stop[t] = entry * (1 - 0.172)
    return {"target": tgt, "stop": stop}
