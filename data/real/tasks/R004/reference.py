"""R004 - TurtleAI (jesse-ai/gpt-instructions, instructions.md "Example Strategy #6", MIT).

Source: https://github.com/jesse-ai/gpt-instructions/blob/64873bfd/instructions.md (lines ~1666-1738)
        MIT License, Copyright (c) 2025 Jesse.

Original logic (jesse, timeframe-agnostic, 4h candles as the long-term feed):
  long  when close > prior-20-candle Donchian upper band (ta.donchian(candles[:-1], 20): highest
        high of the 20 candles BEFORE the current one) and close > SMA(200) of 4h closes and
        ADX(14) > 30 and CHOP(14) < 40 and the strategy did not close a position on this candle.
  short mirrored (close < prior-20 lowest low, close < 4h SMA200, same ADX/CHOP, same cooldown).
  size  = utils.risk_to_qty(available_margin, 3 %, entry, entry -/+ 2.5*ATR(14)) * 1.8
          (risk_to_size caps the notional at the capital -> fraction = 1.8*min(1, 0.03*P/(2.5*ATR))).
  stop  = entry -/+ 2.5*ATR(14) at the open; then on every later candle the stop is ratcheted:
          long: max(previous stop, close - 2.5*ATR(14) of that candle); short: min(prev, close+2.5*ATR).
  no other exit (no take-profit, no signal exit).
  cooldown: passed_time = index - last_closed_index > 0; on_close_position stores self.index, and
          jesse increments index AFTER each strategy call, so a stop filled during candle k blocks
          entries at the close of candle k only (one-bar cooldown).

Approximations / porting decisions:
  * Timeframe: base timeframe unspecified -> 15m bars; long-term feed = hist.htf(240) COMPLETED 4h
    candles (jesse backtests add a bigger-timeframe candle to the store only once it is complete).
  * Size: 1.8*min(1, 0.03*close/(2.5*ATR)) exceeds 1 almost always on 15m BTC -> capped at the engine
    maximum 1.0 (jesse would apply up to 1.8x leverage on a futures account). jesse's fee factors
    (1-3*fee twice) are dropped (absorbed by the 0.05 grid).
  * ATR(14): jesse = Wilder ATR with SMA seed on the last 240 candles; ind.atr = Wilder with TR[0] seed
    over full history -> identical up to ~1e-7 relative (seed weight (13/14)**226).
  * ADX(14): reproduced EXACTLY as jesse computes it (on the last 240 candles; Wilder sums seeded with
    the plain sum of the first 14 values; ADX seeded with mean(DX[14:28]) and DX[28] skipped), using
    ind.rma on window arrays (Wilder sum / n == rma when seeded with the mean).
  * CHOP(14): jesse definition 100*log10(sum TR(14)/(HH14-LL14))/log10(14), drift 1.
  * Stops are resting orders (engine: next bar's high/low, stop wins ties, gap fills at open);
    jesse simulates them on 1m candles. Fill of entries at next open (universal).
  * Entry ATR for the initial stop = ATR on the signal candle (jesse: on_open_position runs on the
    same candle close as the market fill).
"""
import math

import numpy as np

DC_N = 20
HTF_MIN = 240
HTF_SMA = 200
ADX_N = 14
ADX_MIN = 30.0
CHOP_N = 14
CHOP_MAX = 40.0
ATR_N = 14
STOP_ATR = 2.5
RISK = 0.03
SIZE_MULT = 1.8
WIN = 240


def jesse_adx(hist, ind):
    h = hist.high[-WIN:]
    l = hist.low[-WIN:]
    c = hist.close[-WIN:]
    n = ADX_N
    tr = np.maximum(np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1])), np.abs(l[1:] - c[:-1]))
    up = h[1:] - h[:-1]
    dn = l[:-1] - l[1:]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    # Wilder running sums seeded with sum(first n) == n * rma(seeded with the mean)
    s_tr = ind.rma(np.concatenate(([tr[:n].mean()], tr[n:])), n)
    s_p = ind.rma(np.concatenate(([pdm[:n].mean()], pdm[n:])), n)
    s_m = ind.rma(np.concatenate(([mdm[:n].mean()], mdm[n:])), n)
    di_p = np.where(s_tr != 0, 100.0 * s_p / np.where(s_tr != 0, s_tr, 1.0), 0.0)
    di_m = np.where(s_tr != 0, 100.0 * s_m / np.where(s_tr != 0, s_tr, 1.0), 0.0)
    den = di_p + di_m
    dx = np.where(den != 0, 100.0 * np.abs(di_p - di_m) / np.where(den != 0, den, 1.0), 0.0)
    # dx[k] corresponds to jesse DX[n + k]; ADX[2n] = mean(DX[n:2n]); then Wilder from DX[2n+1]
    seed = dx[:n].mean()
    adx = ind.rma(np.concatenate(([seed], dx[n + 1:])), n)
    return adx[-1]


def chop(hist):
    h = hist.high[-(CHOP_N + 1):]
    l = hist.low[-(CHOP_N + 1):]
    c = hist.close[-(CHOP_N + 1):]
    tr = np.maximum(np.maximum(h[1:] - l[1:], np.abs(h[1:] - c[:-1])), np.abs(l[1:] - c[:-1]))
    rng = np.max(h[1:]) - np.min(l[1:])
    if rng <= 0:
        return 100.0
    return 100.0 * (math.log10(np.sum(tr)) - math.log10(rng)) / math.log10(CHOP_N)


def strategy(hist, state, pos, ind):
    close = hist.close
    price = close[-1]
    t = len(close) - 1
    atr = ind.atr(hist, ATR_N)[-1]
    side = 1 if pos > 0 else (-1 if pos < 0 else 0)

    # trade ended by its stop during this bar -> one-bar cooldown
    if "side" in state and side == 0:
        state["last_closed"] = t
        del state["side"]
        del state["stop"]

    if side != 0:
        if side > 0:
            state["stop"] = max(state["stop"], price - STOP_ATR * atr)
        else:
            state["stop"] = min(state["stop"], price + STOP_ATR * atr)
        return {"target": pos, "stop": state["stop"], "take": None}

    target = 0.0
    if t - state.get("last_closed", -1) > 0 and len(close) > DC_N + 1:
        upper = ind.highest(hist.high, DC_N)[-2]
        lower = ind.lowest(hist.low, DC_N)[-2]
        sig = 1 if price > upper else (-1 if price < lower else 0)
        if sig != 0:
            htf = hist.htf(HTF_MIN)
            ma = ind.sma(htf.close, HTF_SMA)[-1] if len(htf) >= HTF_SMA else float("nan")
            trend_ok = (sig > 0 and price > ma) or (sig < 0 and price < ma)
            if trend_ok and jesse_adx(hist, ind) > ADX_MIN and chop(hist) < CHOP_MAX:
                frac = SIZE_MULT * min(1.0, RISK * price / (STOP_ATR * atr))
                target = sig * min(1.0, frac)
                state["side"] = sig
                state["stop"] = price - sig * STOP_ATR * atr
    stop = state["stop"] if "stop" in state else None
    return {"target": target, "stop": stop, "take": None}
