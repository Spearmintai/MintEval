"""Independent second implementation of R008: hourly bars and New York local time via pandas
(tz database instead of hand-coded DST rule); insight objects with UTC close times like LEAN."""
import numpy as np
import pandas as pd


def decide(P, pos_held, warmup):
    t_open = P["time"].astype(np.int64)
    close_ms = t_open + 15 * 60000
    df = pd.DataFrame({"b": t_open // 3600000, "c": P["close"]})
    hourly = df.groupby("b")["c"].last()
    sma = hourly.rolling(5).mean()
    sma_by_hour = sma.to_dict()
    n = len(t_open)
    ts = pd.to_datetime(close_ms, unit="ms", utc=True).tz_convert("America/New_York")
    tgt = np.zeros(n)
    prev_dir = 0
    insight = None
    held = 0.0
    for t in range(warmup, n):
        if close_ms[t] % 3600000 != 0:
            tgt[t] = pos_held[t]
            continue
        hour_bucket = close_ms[t] // 3600000 - 1         # the hour that just completed
        price = P["close"][t]
        s = sma_by_hour.get(hour_bucket, np.nan)
        local = ts[t]
        tod = local.hour * 60 + local.minute
        if 600 <= tod <= 900:
            d = 1 if (not np.isnan(s) and price < round(s * 1.001, 6)) else -1
            if d != prev_dir:
                prev_dir = d
                close_local = local.replace(hour=15, minute=1, second=0)
                insight = (d, close_local.tz_convert("UTC").value // 10**6)
        if insight is not None and insight[1] < close_ms[t]:
            insight = None
        tgt[t] = 0.0 if insight is None else float(insight[0])
    return {"target": tgt}
