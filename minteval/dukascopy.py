"""Dukascopy free historical 1-minute candles -> 15-minute regular-session bars.

URL: https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/BID_candles_min_1.bi5
  MM0 is the ZERO-BASED month (verified: '2019/02/15' is 15 March 2019).
Record (big-endian, 24 bytes): int32 seconds-from-midnight-UTC, int32 open, close, low, high (x1000), float32 volume.
These are CFD prices on the ETF/index, i.e. a proxy for the exchange-traded instrument (documented)."""
from __future__ import annotations

import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed"


def _url(sym, d: date):
    return f"{BASE}/{sym}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"


def download(sym: str, start: str, end: str, out_dir: str, workers: int = 2, pause_s: float = 1.0):
    out = Path(out_dir) / sym; out.mkdir(parents=True, exist_ok=True)
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    days = [d0 + timedelta(i) for i in range((d1 - d0).days) if (d0 + timedelta(i)).weekday() < 5]
    log = []

    def get(d):
        f = out / f"{d.isoformat()}.bi5"
        if f.exists():
            return d, "cached"
        for k in range(9):                       # the feed throttles: be gentle, back off up to ~5 min
            time.sleep(pause_s)
            try:
                r = requests.get(_url(sym, d), headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
                if r.status_code == 200:
                    f.write_bytes(r.content); return d, "ok"
                if r.status_code == 404:
                    f.write_bytes(b""); return d, "404"
            except requests.RequestException:
                pass
            time.sleep(min(300, 2 ** (k + 1)))
        return d, "fail"
    with ThreadPoolExecutor(workers) as ex:
        for d, st in ex.map(get, days):
            log.append((d.isoformat(), st))
    return log


def build_15m(sym: str, raw_dir: str, out_csv: str, scale: float = 1000.0) -> dict:
    rows = []
    for f in sorted((Path(raw_dir) / sym).glob("*.bi5")):
        b = f.read_bytes()
        if not b:
            continue
        raw = lzma.decompress(b)
        a = np.frombuffer(raw, dtype=">i4").reshape(-1, 6)
        vol = np.frombuffer(raw, dtype=">f4").reshape(-1, 6)[:, 5]
        day = pd.Timestamp(f.stem, tz="UTC")
        t = day + pd.to_timedelta(a[:, 0], unit="s")
        rows.append(pd.DataFrame({"t": t, "open": a[:, 1] / scale, "close": a[:, 2] / scale, "low": a[:, 3] / scale,
                                  "high": a[:, 4] / scale, "volume": vol.astype(float)}))
    m = pd.concat(rows).set_index("t").sort_index()
    ny = m.index.tz_convert("America/New_York")
    mins = ny.hour * 60 + ny.minute
    m = m[(mins >= 570) & (mins < 960)]                     # regular session 09:30-16:00 ET
    m = m[m.volume > 0]                                     # Dukascopy pads non-trading minutes with flat zero-vol rows
    b = m.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    bny = b.index.tz_convert("America/New_York"); bm = bny.hour * 60 + bny.minute
    b = b[(bm >= 570) & (bm < 960)]
    per_day = b.groupby(b.index.tz_convert("America/New_York").date).size()
    out = b.reset_index().rename(columns={"t": "open_time"})
    out["open_time"] = out.open_time.astype("int64") // 10**6
    out[["open_time", "open", "high", "low", "close", "volume"]].to_csv(out_csv, index=False)
    bad = int(((out.high < out[["open", "close"]].max(axis=1)) | (out.low > out[["open", "close"]].min(axis=1))).sum())
    return {"n_bars": len(out), "n_days": int(len(per_day)), "bars_per_day_median": float(per_day.median()),
            "short_days(<26)": int((per_day < 26).sum()), "bad_ohlc": bad,
            "first": str(b.index[0]), "last": str(b.index[-1])}
