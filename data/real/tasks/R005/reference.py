"""R005 - SimplEma (ysdede/jesse_strategies, simplema/strategies/SimplEma, CC0-1.0). Reference port.

Source: https://github.com/ysdede/jesse_strategies/blob/ade9f4ba/simplema/strategies/SimplEma/__init__.py
        README: https://github.com/ysdede/jesse_strategies/blob/ade9f4ba/simplema/README.md
        License: CC0 1.0 Universal (public-domain dedication).

Original logic (jesse, route Binance Futures ETH-USDT 30m, default hyperparameters):
  EMA fast 8, EMA slow 22 (sequential EMAs of close).
  long when EMA8 crosses above EMA22 (prev <= and now >) and NOT dump_pump, where dump_pump is true
       if any of the last 3 candles has |open-close|/open > 2.3 %, or |open[-3] - close[-1]|/open[-3]
       > 2.3 % (3-candle move). should_short returns False (long only). Donchian filter disabled.
  size = capital/10 (10 % of the balance; 10 * number of routes = 10), multiplied by `multiplier`
       while (not last_was_profitable and lose_count <= 4); plus 0.001 coin.
  stop-loss order at entry*(1 - 0.172).
  exits (market): unrealised PnL% > 15.3 % (pnl_percentage/leverage, leverage 1) or ANY EMA8/EMA22
       cross (either direction) -> liquidate.
  on_stop_loss: lose_count += 1; multiplier *= 1 + 33/50 (= 1.66); last_was_profitable = False.
  on_take_profit: reset - but it is NEVER called, because profits are taken with liquidate() (a market
       order), not with a take-profit order. So last_was_profitable stays False forever and the
       multiplier only resets... never. Faithful consequence: size = 0.1 * 1.66**k after k stop-loss
       hits while k <= 4, and 0.1 for ever once k >= 5.

Approximations / porting decisions:
  * Timeframe: 30m ETH -> 15m BTC bars with the same bar counts.
  * 0.001-coin size bump dropped (ETH ~2-6 USD at 2021 balance scale; negligible); fee factor of
    size_to_qty dropped. Target sizes are rounded to the 0.05 grid by the engine: 0.1, 0.166->0.15,
    0.2756->0.3, 0.4575->0.45, 0.7594->0.75.
  * "capital" = current balance (jesse alias) -> fraction of current equity.
  * PnL take-profit evaluated on candle closes (jesse update_position runs at candle close) -> market
    exit at next open. Entry price = signal candle close (jesse market fill price) = desk convention.
  * EMA: jesse EMA is SMA-seeded on the full candle history; ind.ema is seeded with the first close;
    after the 960-bar warm-up the seed weight is (20/22)**960 ~ 1e-40: identical.
  * Stop-loss resting order checked on the next bars' lows (jesse: 1m simulation).
  * jesse re-checks should_long on the candle of a liquidation; a cross-up cannot occur on such a
    candle (while long EMA8 > EMA22 on every bar since entry), so the port skips that check.
"""
import math

FAST = 8
SLOW = 22
DP_THRESHOLD = 2.3          # percent
STOP = 0.172
TARGET_PNL = 15.3           # percent
BASE_SIZE = 0.1
MULT_STEP = 1.66            # 1 + carpan/50 with carpan = 33
LOSE_LIMIT = 4


def is_dp(o, c):
    return abs(o - c) * 100.0 / o > DP_THRESHOLD


def strategy(hist, state, pos, ind):
    close = hist.close
    opn = hist.open
    price = close[-1]
    ef = ind.ema(close, FAST)
    es = ind.ema(close, SLOW)
    cross_up = ef[-2] <= es[-2] and ef[-1] > es[-1]
    cross_dn = ef[-2] >= es[-2] and ef[-1] < es[-1]

    if "lose_count" not in state:
        state["lose_count"] = 0
        state["multiplier"] = 1.0

    in_pos = pos > 0
    if "entry" in state and not in_pos:
        if not state.get("exit_sent", False):
            # flat without having asked for it -> the stop-loss order filled
            state["lose_count"] = state["lose_count"] + 1
            state["multiplier"] = state["multiplier"] * MULT_STEP
        del state["entry"]
        if "exit_sent" in state:
            del state["exit_sent"]

    target = pos
    if in_pos:
        pnl = (price - state["entry"]) / state["entry"] * 100.0
        if pnl > TARGET_PNL or cross_up or cross_dn:
            target = 0.0
            state["exit_sent"] = True
            return {"target": target, "stop": None, "take": None}
        return {"target": target, "stop": state["entry"] * (1.0 - STOP), "take": None}

    target = 0.0
    if cross_up:
        dump_pump = (is_dp(opn[-1], close[-1]) or is_dp(opn[-2], close[-2]) or is_dp(opn[-3], close[-3])
                     or is_dp(opn[-3], close[-1]))
        if not dump_pump:
            size = BASE_SIZE
            if state["lose_count"] <= LOSE_LIMIT:
                size = BASE_SIZE * state["multiplier"]
            target = size
            state["entry"] = price
            return {"target": target, "stop": price * (1.0 - STOP), "take": None}
    return {"target": target, "stop": None, "take": None}
