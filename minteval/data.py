"""Load and clean Binance BTCUSDT spot klines (data.binance.vision monthly zips)."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_base", "taker_quote", "ignore"]


def _read_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        df = pd.read_csv(io.BytesIO(z.read(name)), header=None, names=COLS)
    return df


def build_price_file(raw_dir: str, out_path: str, start: str, end: str, freq_ms: int) -> dict:
    """Concatenate monthly zips, keep OHLCV, log (do not fill) gaps. Returns a gap report."""
    raw = Path(raw_dir)
    frames = [_read_zip(p) for p in sorted(raw.glob("*.zip"))]
    df = pd.concat(frames, ignore_index=True)
    # Binance switched spot timestamps to microseconds from 2025; normalise to ms defensively.
    df.loc[df.open_time > 10**14, "open_time"] //= 1000
    df = df[["open_time", "open", "high", "low", "close", "volume"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    t0 = pd.Timestamp(start, tz="UTC").value // 10**6
    t1 = pd.Timestamp(end, tz="UTC").value // 10**6
    df = df[(df.open_time >= t0) & (df.open_time < t1)].reset_index(drop=True)
    diffs = np.diff(df.open_time.values)
    gap_idx = np.where(diffs != freq_ms)[0]
    gaps = [{"after_bar": int(i),
             "from": str(pd.Timestamp(int(df.open_time[i]), unit="ms")),
             "to": str(pd.Timestamp(int(df.open_time[i + 1]), unit="ms")),
             "missing_bars": int(diffs[i] // freq_ms - 1)} for i in gap_idx]
    expected = (t1 - t0) // freq_ms
    report = {"n_bars": len(df), "expected_bars": int(expected),
              "missing_bars": int(expected - len(df)), "gaps": gaps,
              "bad_ohlc": int(((df.high < df[["open", "close"]].max(axis=1)) |
                               (df.low > df[["open", "close"]].min(axis=1))).sum())}
    df.to_parquet(out_path, index=False) if out_path.endswith(".parquet") else df.to_csv(out_path, index=False)
    Path(out_path).with_suffix(".gaps.json").write_text(json.dumps(report, indent=1))
    return report


def load_prices(path: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {c: df[c].to_numpy(dtype=np.float64) for c in ["open", "high", "low", "close", "volume"]} | {
        "open_time": df.open_time.to_numpy(dtype=np.int64)}
