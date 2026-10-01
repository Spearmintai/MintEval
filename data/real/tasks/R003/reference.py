"""R003 - Simple Bollinger Bands with Ichimoku-cloud trend filter
(jesse example-strategies/SimpleBollinger, MIT; origin gabrielweich/jesse-strategies, MIT).

Source: https://github.com/jesse-ai/example-strategies/blob/7c91e0a3/SimpleBollinger/__init__.py

Original logic (jesse, docstring says 1h):
  Bollinger Bands = jesse ta.bollinger_bands defaults: period 20, 2 std up/down, SMA middle,
  population std (np.std, ddof=0), source hl2 = (high+low)/2.
  filter: close > Ichimoku span A and close > span B (jesse ta.ichimoku_cloud defaults 9/26/52,
  displacement 26: spans are the values computed on the candles that end 25 candles before the
  current one, i.e. the cloud plotted under the current candle).
  long when close > upper band (level condition, every bar while flat).
  exit (market) when close < middle band. Long only, whole balance.

Approximations / porting decisions:
  * Timeframe: 1h in the source -> ported to 15m bars with the same bar counts.
  * jesse ichimoku_cloud (numba version, commit 7a01f999 of jesse-ai/jesse, matches the pre-numba
    python version): span_a = (mid(9) + mid(26)) / 2 and span_b = mid(52), where mid(n) =
    (max high + min low)/2 over the n candles ending at index -26 (candles[:-(26-1)]).
    Reproduced exactly with ind.highest/ind.lowest read at index -26.
  * Size: whole balance (~99.85 % after jesse's fee factor) -> 1.0. Fill at next open (universal).
  * Same-bar exit + re-entry cannot happen here (close < middle excludes close > upper).
"""
import math

import numpy as np

BB_N = 20
BB_K = 2.0
TENKAN = 9
KIJUN = 26
SENKOU_B = 52
SHIFT = 26          # spans read at index -SHIFT (candles ending SHIFT-1 bars ago)


def strategy(hist, state, pos, ind):
    close = hist.close
    price = close[-1]
    if len(close) < 80:
        return {"target": 0.0, "stop": None, "take": None}
    hl2 = (hist.high[-BB_N:] + hist.low[-BB_N:]) / 2.0
    mid = float(np.mean(hl2))
    sd = float(np.std(hl2))
    upper = mid + BB_K * sd

    target = pos
    if pos > 0:
        if price < mid:
            target = 0.0
        return {"target": target, "stop": None, "take": None}

    target = 0.0
    if price > upper:
        hh = ind.highest(hist.high, TENKAN)[-SHIFT]
        ll = ind.lowest(hist.low, TENKAN)[-SHIFT]
        conv = (hh + ll) / 2.0
        base = (ind.highest(hist.high, KIJUN)[-SHIFT] + ind.lowest(hist.low, KIJUN)[-SHIFT]) / 2.0
        span_a = (conv + base) / 2.0
        span_b = (ind.highest(hist.high, SENKOU_B)[-SHIFT] + ind.lowest(hist.low, SENKOU_B)[-SHIFT]) / 2.0
        if price > span_a and price > span_b:
            target = 1.0
    return {"target": target, "stop": None, "take": None}
