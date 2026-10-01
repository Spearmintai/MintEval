"""R001 - Connors RSI(2) (jesse example-strategies/RSI2, MIT). Reference port for MintEval.

Source: https://github.com/jesse-ai/example-strategies/blob/7c91e0a3/RSI2/__init__.py
        (author FengkieJ, 2020; MIT License, Copyright (c) 2020 jesse-ai)

Original logic (jesse, any timeframe):
  long  when close > SMA(200) and RSI(2) <= 10       (whole balance)
  short when close < SMA(200) and RSI(2) >= 90       (whole balance)
  exit long when close > SMA(5); exit short when close < SMA(5)  (market, "liquidate")
  Entries are only checked while flat; jesse checks entries again on the SAME candle right after a
  liquidation (Strategy._check: update_position -> market fill -> should_long/should_short), so an
  exit and a new entry (also in the other direction) can happen on one bar.

Approximations / porting decisions (documented for the human verifier):
  * Timeframe: the original is timeframe-agnostic; ported to 15m bars with the same bar counts.
  * Fills: jesse fills market orders at the candle close; our engine fills at the next open
    (universal engine difference).
  * Size: jesse sizes with utils.size_to_qty(balance, price, fee_rate) = balance*(1-3*fee)/price,
    i.e. ~99.85 % of equity -> target 1.0 (the 0.05 grid absorbs the fee factor).
  * RSI: jesse rsi() = Wilder RSI seeded with the simple mean of the first `period` changes, computed
    on the last 240 candles (helpers.slice_candles warm-up window); ind.rsi = Wilder RSI seeded with
    the first value over the full history. For period 2 the seed weight after 240 candles is
    0.5**238 -> numerically identical except for float rounding at the exact thresholds.
  * SMA: identical definition (simple mean of the last n closes incl. the current one).
  * Exit and same-bar re-entry: reproduced (exit check first, then the entry check when flat).
"""
import math

SLOW = 200
FAST = 5
RSI_N = 2
OVERSOLD = 10.0
OVERBOUGHT = 90.0


def strategy(hist, state, pos, ind):
    close = hist.close
    price = close[-1]
    slow = ind.sma(close, SLOW)[-1]
    fast = ind.sma(close, FAST)[-1]
    rsi = ind.rsi(close, RSI_N)[-1]

    side = 1 if pos > 0 else (-1 if pos < 0 else 0)
    target = pos
    # update_position: exits
    if side > 0 and price > fast:
        target = 0.0
        side = 0
    elif side < 0 and price < fast:
        target = 0.0
        side = 0
    # entries (only when flat, also on the bar of an exit)
    if side == 0:
        target = 0.0
        if not math.isnan(slow):
            if price < slow and rsi >= OVERBOUGHT:
                target = -1.0
            elif price > slow and rsi <= OVERSOLD:
                target = 1.0
    return {"target": target, "stop": None, "take": None}
