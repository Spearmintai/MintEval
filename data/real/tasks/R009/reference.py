"""R009 - Always-long with a one-cancels-other exit bracket (QuantConnect LEAN
Algorithm.Python/OneCancelsOtherOrderRegressionAlgorithm.py, Apache-2.0). Reference port.

Source: https://github.com/QuantConnect/Lean/blob/41c6e603/Algorithm.Python/OneCancelsOtherOrderRegressionAlgorithm.py
        Copyright 2014 QuantConnect Corporation, Apache-2.0.

Original logic (SPY, minute bars):
  on_data:
    if an exit set exists: if any of its orders is filled/canceled/invalid -> forget the set and
       RETURN (nothing else on this data point); otherwise return.
    else: market-buy 100 shares, then submit an OCO set using the current price p:
       limit sell at p*1.003 ("Take Profit"), stop-market sell at p*0.997 ("Stop Loss"),
       stop-limit sell stop p*0.99 / limit p*0.98 ("Far Stop Loss"). The first to fill cancels the rest.
  => always long; after each exit the position is re-opened on the following data point.

Approximations / porting decisions:
  * Size: 100 SPY shares (~17 % of the 100k account in Oct 2013) has no BTC analogue; ADDED FACT:
    the position is 100 % of equity (target 1.0).
  * Data: minute -> 15m bars. Reference price p = close of the bar on which the entry is decided
    (desk convention); the entry fills at the next open (LEAN fills the market order immediately).
  * The take-profit LIMIT order is modelled as the engine's take level (fills at the level, or at
    the open on a gap - a limit sell would fill at the better open price too). LEAN equity fill
    models use strict comparisons (high > limit, low < stop); the engine uses >= / <=.
  * The far stop-limit (0.99/0.98) can never be the first order to trigger (the 0.997 stop always
    triggers at or before it, and the engine's stop wins ties), so it is dropped.
  * The one-data-point pause after an exit is reproduced: the bar on which the stop/take fills is
    the bar on which LEAN notices the completed set and returns; the re-entry is decided on the
    next bar's close.
"""
import math

TAKE = 1.003
STOP = 0.997
SIZE = 1.0


def strategy(hist, state, pos, ind):
    price = hist.close[-1]
    if pos > 0:
        return {"target": pos, "stop": state["ref"] * STOP, "take": state["ref"] * TAKE}
    if state.get("set_open", False):
        # the exit set completed during this bar: forget it and do nothing else now
        state["set_open"] = False
        return {"target": 0.0, "stop": None, "take": None}
    state["set_open"] = True
    state["ref"] = price
    return {"target": SIZE, "stop": price * STOP, "take": price * TAKE}
