"""R007 - Constant long insight + 1 % trailing-stop risk model (QuantConnect LEAN
TrailingStopRiskFrameworkRegressionAlgorithm.py on BaseFrameworkRegressionAlgorithm, Apache-2.0).

Sources (commit 41c6e603, Copyright 2014 QuantConnect Corporation, Apache-2.0):
  Algorithm.Python/TrailingStopRiskFrameworkRegressionAlgorithm.py   (risk model 0.01)
  Algorithm.Python/BaseFrameworkRegressionAlgorithm.py               (ConstantAlphaModel UP, 31 days,
                                                                      EqualWeighting PCM, hourly data)
  Algorithm.Framework/Risk/TrailingStopRiskManagementModel.py
  Algorithm.Framework/Alphas/ConstantAlphaModel.py

Original logic:
  * ConstantAlphaModel emits an UP insight for the symbol at the first data point and then again
    whenever at least 31 days have passed since the previous emission (utc - generated >= period).
    Each insight lives 31 days, so emissions chain without a gap.
  * EqualWeighting PCM -> 100 % long while an UP insight is active.
  * TrailingStopRiskManagementModel(0.01): while invested, track the highest holdings value since
    the position was opened (initial value = holdings cost); on each data point, if the value made a
    new high just record it, otherwise if (high - value)/high > 1 % -> cancel the insights and
    liquidate. Not invested -> state removed.
  * After a liquidation the algorithm stays flat until the alpha's next 31-day emission.

Approximations / porting decisions:
  * Universe: the base regression trades AAPL with universe churn of 4 equities; collapsed to one
    symbol (BTCUSDT) -> equal weight 1/1 = target 1.0.
  * Data: hourly bars -> 15m bars; the holdings value is evaluated at each 15m close (value is
    proportional to the close for a fixed quantity, so the drawdown test uses closes).
  * Initial high = holdings cost = fill price in LEAN; here the signal-bar close (desk convention).
  * The liquidation is a market order at the next open (LEAN: market order at the next data point).
  * The 31-day emission clock uses bar timestamps (decision time = 15m bar close time) and starts at
    the first strategy call after the 960-bar warm-up.
"""
import math

PERIOD_MS = 2678400000        # 31 days
TRAIL = 0.01


def strategy(hist, state, pos, ind):
    price = hist.close[-1]
    now = hist.time[-1] + 900000          # decision time = bar close time
    invested = pos > 0

    if not invested and "peak" in state:
        del state["peak"]                 # risk model forgets the holding once flat

    target = pos
    # alpha: (re-)emit an UP insight every 31 days
    if "emitted" not in state or now - state["emitted"] >= PERIOD_MS:
        state["emitted"] = now
        state["active"] = True
    if state["active"] and not invested:
        target = 1.0
        state["peak"] = price             # holdings cost (entry price)
        return {"target": target, "stop": None, "take": None}

    # risk model on the held position
    if invested:
        if "peak" not in state:
            state["peak"] = price
        if price > state["peak"]:
            state["peak"] = price
        elif (state["peak"] - price) / state["peak"] > TRAIL:
            state["active"] = False       # insight cancelled
            del state["peak"]
            target = 0.0
    return {"target": target, "stop": None, "take": None}
