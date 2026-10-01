"""R006 - Dual Thrust alpha with 3 % per-position drawdown stop (QuantConnect LEAN
Algorithm.Python/Alphas/VIXDualThrustAlpha.py, Apache-2.0). Reference port.

Source: https://github.com/QuantConnect/Lean/blob/41c6e603/Algorithm.Python/Alphas/VIXDualThrustAlpha.py
        + Algorithm.Framework/Risk/MaximumDrawdownPercentPerSecurity.py,
        Common/Algorithm/Framework/Alphas/Insight.cs (IsExpired: CloseTimeUtc < utcTime),
        EqualWeightingPortfolioConstructionModel. Copyright 2014 QuantConnect Corporation, Apache-2.0.

Original logic (LEAN framework, minute data, one symbol):
  * 30-minute consolidated bars; RollingWindow of the last 20 such bars. On each new 30m bar:
      HH = max high, HC = max close, LC = min close, LL = min low over the 20 bars,
      range = max(HH - LC, HC - LL); upper = close_of_that_bar + 0.63*range; lower = close - 0.63*range.
  * Every minute: if price > upper and the holding is not long -> UP insight valid 5 days;
                  if price < lower and the holding is not short -> DOWN insight valid 5 days.
  * EqualWeighting portfolio construction with one symbol: latest active insight -> 100 % long or
    100 % short (an opposite insight REVERSES the position); an expired insight -> 0 (flat).
  * Risk: MaximumDrawdownPercentPerSecurity(0.03): if unrealised PnL < -3 % -> liquidate and cancel
    the symbol's insights (so no target until the alpha emits a new insight).

Approximations / porting decisions:
  * Data: minute bars -> 15m bars. The range lines come from hist.htf(30) (completed 30m candles,
    UTC-aligned), the breakout test uses each 15m close instead of each minute's price.
  * Risk model: LEAN checks the unrealised PnL on every minute close and liquidates at market; here a
    resting stop at entry*(1 -/+ 0.03) (entry = signal-bar close, desk convention) checked on the next
    bars' high/low. After a stop fill the insight is treated as cancelled (pos == 0 and no new signal).
  * Insight expiry: Insight.IsExpired = CloseTimeUtc < now (strict); with close = emission + 5 days the
    position is closed at the first 15m decision whose time is strictly later than emission + 5 days.
    Time is measured with bar timestamps (the data has one 5-bar gap). NOTE: the source passes a UTC
    time to the closeTimeLocal overload; for a UTC-exchange crypto symbol this is exactly +5 days.
  * Size: equal weighting of one insight = 100 % of portfolio value -> target +/-1.0.
  * Fees: the source uses ConstantFeeModel(0); the MintEval engine always charges its own costs.
"""
import math

K1 = 0.63
K2 = 0.63
RANGE_N = 20
CONS_MIN = 30
EXPIRY_MS = 432000000         # 5 days in ms
MAX_DD = 0.03


def strategy(hist, state, pos, ind):
    price = hist.close[-1]
    now = hist.time[-1] + 15 * 60 * 1000          # decision time = close time of the 15m bar
    side = 1 if pos > 0 else (-1 if pos < 0 else 0)

    # stop-out by the risk model: we targeted a position on the previous bar but are flat now
    # -> MaximumDrawdownPercentPerSecurity cancelled the insight
    if state.get("last_target", 0) != 0 and side == 0 and "dir" in state:
        del state["dir"]
        del state["emitted"]

    # alpha model: lines from the last 20 completed 30m bars; new insight only if not already
    # holding that direction
    bars = hist.htf(CONS_MIN)
    if len(bars) >= RANGE_N:
        hh = ind.highest(bars.high, RANGE_N)[-1]
        ll = ind.lowest(bars.low, RANGE_N)[-1]
        hc = ind.highest(bars.close, RANGE_N)[-1]
        lc = ind.lowest(bars.close, RANGE_N)[-1]
        rng = max(hh - lc, hc - ll)
        upper = bars.close[-1] + K1 * rng
        lower = bars.close[-1] - K2 * rng
        if price > upper and side <= 0:
            state["dir"] = 1
            state["emitted"] = now
            state["entry"] = price
        elif price < lower and side >= 0:
            state["dir"] = -1
            state["emitted"] = now
            state["entry"] = price

    # portfolio construction: latest insight while active (expiry is strict: close time < now)
    target = 0.0
    if "dir" in state:
        if state["emitted"] + EXPIRY_MS < now:
            del state["dir"]
            del state["emitted"]
        else:
            target = float(state["dir"])
    state["last_target"] = target

    # risk model: 3 % adverse move from the entry price
    stop = None
    if target > 0:
        stop = state["entry"] * (1.0 - MAX_DD)
    elif target < 0:
        stop = state["entry"] * (1.0 + MAX_DD)
    return {"target": target, "stop": stop, "take": None}
