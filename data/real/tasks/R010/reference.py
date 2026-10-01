"""R010 - 30-minute momentum stepping with daily flatten (QuantConnect LEAN
Algorithm.Python/DataConsolidationAlgorithm.py, Apache-2.0). Reference port.

Source: https://github.com/QuantConnect/Lean/blob/41c6e603/Algorithm.Python/DataConsolidationAlgorithm.py
        Copyright 2014 QuantConnect Corporation, Apache-2.0.

Original trading logic (SPY minute data; the rest of the file only demonstrates consolidators):
  thirty_minute_bar_handler (every completed 30-minute bar):
      if last is not None and bar.close > last.close: order +100 shares
      elif last is not None and bar.close < last.close: order -100 shares
      last = bar
  on_end_of_day: liquidate SPY and set last = None ("close up shop each day and reset our 'last'").
  => the position accumulates one fixed unit per 30m bar in the direction of the last 30m change
     (pyramiding / stepwise target, can go net short), flattened once per day.

Approximations / porting decisions:
  * Unit: 100 SPY shares has no BTC analogue. ADDED FACT: one unit = 10 % of equity (target steps of
    0.10); the net position is capped at +/-100 % - orders that would exceed it are skipped (LEAN
    would reject orders beyond buying power; the cap level is an added fact).
  * Units are re-expressed as a fraction of current equity at each change (the engine works in
    equity fractions), not a fixed coin quantity.
  * 30m bars = hist.htf(30) (UTC-aligned); a decision is taken only on the 15m bar that completes a
    30m bar; orders fill at the next open.
  * End of day: BTC trades 24/7; LEAN fires OnEndOfDay 10 minutes before the market's daily close,
    i.e. 23:50 UTC for a UTC-exchange crypto symbol. Ported as: at the close of the 15m bar ending
    23:45 UTC the position is flattened and `last` is cleared; the 30m bar ending 00:00 UTC is then
    only stored as `last` (no order), exactly as the handler does when last is None.
"""
import math

UNIT = 0.1
CAP = 1.0
CONS_MS = 1800000
DAY_MS = 86400000
EOD_MS = 85500000         # 23:45 UTC as ms after midnight


def strategy(hist, state, pos, ind):
    now = int(hist.time[-1]) + 900000                 # close time of this 15m bar
    target = pos
    if now % DAY_MS == EOD_MS:
        if "last" in state:
            del state["last"]
        return {"target": 0.0, "stop": None, "take": None}
    if now % CONS_MS != 0:
        return {"target": target, "stop": None, "take": None}
    bar_close = hist.htf(30).close[-1]
    if "last" in state:
        if bar_close > state["last"] and pos + UNIT <= CAP + 1e-9:
            target = pos + UNIT
        elif bar_close < state["last"] and pos - UNIT >= -CAP - 1e-9:
            target = pos - UNIT
    state["last"] = bar_close
    return {"target": target, "stop": None, "take": None}
