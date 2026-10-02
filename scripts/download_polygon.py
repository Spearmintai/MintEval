"""Download 15-minute SPY/QQQ bars from Polygon (massive.com) into data/prices/raw_equity/.

Usage: POLYGON_API_KEY=... python scripts/download_polygon.py [start] [end]
The key is read from the environment only. Free plans allow 5 requests/min and roughly two years of
history, so requests are spaced 13 s apart and the window is fetched in quarterly chunks.
Bars are split-adjusted and include pre/post-market; `session` marks regular-hours bars
(09:30-16:00 America/New_York) so either view can be built later."""
import os, sys, time, json
from pathlib import Path
import pandas as pd
import requests

KEY = os.environ["POLYGON_API_KEY"]
START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-10-02")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-10-01")
OUT = Path("data/prices/raw_equity")
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://api.polygon.io/v2/aggs/ticker/{t}/range/15/minute/{a}/{b}"


def get(url, params):
    for attempt in range(6):
        r = requests.get(url, params=params, headers={"Authorization": f"Bearer {KEY}"}, timeout=60)
        j = r.json()
        if j.get("status") in ("OK", "DELAYED"):
            return j
        if "maximum requests" in str(j.get("error", "")):
            time.sleep(65)
            continue
        raise RuntimeError(f"{url}: {j}")
    raise RuntimeError(f"{url}: rate limit retries exhausted")


def chunks(a, b):
    s = a
    while s <= b:
        e = min(s + pd.offsets.QuarterEnd(0), b)
        yield s, e
        s = e + pd.Timedelta(days=1)


report = {}
for t in ("SPY", "QQQ"):
    rows = []
    for a, b in chunks(START, END):
        url = URL.format(t=t, a=a.date(), b=b.date())
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        while url:
            j = get(url, params)
            rows += j.get("results", [])
            url, params = j.get("next_url"), {}
            time.sleep(13)
        print(t, a.date(), b.date(), len(rows), flush=True)
    df = pd.DataFrame(rows).rename(columns={"t": "open_time", "o": "open", "h": "high", "l": "low",
                                            "c": "close", "v": "volume", "vw": "vwap", "n": "n_trades"})
    df = df.drop_duplicates("open_time").sort_values("open_time")
    et = pd.to_datetime(df.open_time, unit="ms", utc=True).dt.tz_convert("America/New_York")
    mins = et.dt.hour * 60 + et.dt.minute
    df["session"] = ((mins >= 570) & (mins < 960)).map({True: "regular", False: "extended"})
    df.to_csv(OUT / f"{t}_15m.csv", index=False)
    reg = df[df.session == "regular"]
    report[t] = {"bars": len(df), "regular_bars": len(reg), "first": str(et.iloc[0]), "last": str(et.iloc[-1]),
                 "regular_days": int(et[df.session == "regular"].dt.date.nunique())}
json.dump(report, open(OUT / "download_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
