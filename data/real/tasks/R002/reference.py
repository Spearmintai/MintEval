"""R002 - "RSI Trend Crypto" (jesse example-strategies/TradingView_RSI, MIT). Reference port.

Source: https://github.com/jesse-ai/example-strategies/blob/7c91e0a3/TradingView_RSI/__init__.py
        (MIT License, Copyright (c) 2020 jesse-ai; docstring cites TradingView script Ru7qOVtp,
        whose Pine code was NOT inspected - only the jesse re-implementation is ported)

Original logic (jesse, any timeframe, default hyperparameters):
  RSI period = 5 (hp 'rsi' default 5), stop_loss = 0.95, take_profit = 1.10, xparam = 75.
  long  when RSI crosses above 35 (utils.crossed: prev <= 35 and now > 35), whole balance.
  on entry: stop-loss order at entry*0.95, take-profit order at entry*1.10 (whole position).
  exit (market) when RSI crosses below 75 or crosses below 10 (prev >= x and now < x).
  long only.

Approximations / porting decisions:
  * Timeframe: original timeframe-agnostic (results files are for BTC/ETH, timeframe not stated);
    ported to 15m bars.
  * Entry price for the bracket = close of the signal candle (jesse: market order filled at close;
    stop/take computed from self.price) = our desk convention. Fill at next open (universal).
  * Stop/take are resting orders checked inside the following bars (jesse simulates them on 1m
    candles; we on 15m high/low, stop wins a tie).
  * RSI: jesse Wilder RSI on a 240-candle window with SMA seed vs ind.rsi (seed = first value, full
    history). Seed weight after 240 candles (4/5)**235 ~ 1e-23: identical up to float rounding.
  * Size: size_to_qty(balance, price, precision 3, fee) ~ 99.85 % of equity -> 1.0. The
    available-margin guard in should_long is always true when flat and is dropped.
  * After a stop/take fill jesse evaluates should_long on the same candle close; reproduced (the
    strategy sees pos = 0 at that close and may enter again).
"""
import math

RSI_N = 5
ENTRY_LVL = 35.0
EXIT_LVL = 75.0
EMERGENCY_LVL = 10.0
STOP_MULT = 0.95
TAKE_MULT = 1.10


def strategy(hist, state, pos, ind):
    close = hist.close
    price = close[-1]
    rsi = ind.rsi(close, RSI_N)
    r_now = rsi[-1]
    r_prev = rsi[-2]

    side = 1 if pos > 0 else 0
    if side == 0 and "entry" in state:
        del state["entry"]          # stop or take filled

    target = pos
    if side > 0:
        crossed_exit = r_prev >= EXIT_LVL and r_now < EXIT_LVL
        crossed_emerg = r_prev >= EMERGENCY_LVL and r_now < EMERGENCY_LVL
        if crossed_exit or crossed_emerg:
            target = 0.0
            side = 0
            del state["entry"]
    if side == 0:
        target = 0.0
        if r_prev <= ENTRY_LVL and r_now > ENTRY_LVL:
            target = 1.0
            state["entry"] = price

    stop = None
    take = None
    if "entry" in state:
        stop = state["entry"] * STOP_MULT
        take = state["entry"] * TAKE_MULT
    return {"target": target, "stop": stop, "take": take}
