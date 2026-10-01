"""Independent second implementation of R010: integer unit counter (no floats), 30m bars from a
pandas groupby, event list (30m-bar events and end-of-day events) processed in time order."""
import numpy as np
import pandas as pd


def decide(P, pos_held, warmup):
    t_open = P["time"].astype(np.int64)
    close_ms = t_open + 15 * 60000
    df = pd.DataFrame({"b": t_open // 1800000, "c": P["close"]})
    c30 = df.groupby("b")["c"].last().to_dict()
    n = len(t_open)
    tgt = np.zeros(n)
    last = None
    for t in range(warmup, n):
        units = int(round(pos_held[t] / 0.1))
        ms = close_ms[t]
        if ms % 86400000 == 23 * 3600000 + 45 * 60000:      # OnEndOfDay
            last = None
            tgt[t] = 0.0
            continue
        if ms % 1800000:
            tgt[t] = pos_held[t]
            continue
        bar = c30[ms // 1800000 - 1]
        if last is not None and bar > last and units < 10:
            units += 1
        elif last is not None and bar < last and units > -10:
            units -= 1
        last = bar
        tgt[t] = units * 0.1
    return {"target": tgt}
