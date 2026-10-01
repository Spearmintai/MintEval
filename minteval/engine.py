"""Bar-by-bar backtest engine.

Timeline for bar t:
  1. at the open of bar t, the decision made at the close of bar t-1 is executed
     (rebalance to the quantised target at open, with fee + adverse slippage);
  2. the stop/take levels returned at bar t-1 are checked inside bar t using
     high/low (stop has priority; a gap through the level fills at the open);
  3. bar t is revealed to ``hist`` and the strategy is called at its close.

Positions are fractions of equity in [-1, 1] on a grid of ``quant_delta``.
``pos`` handed to the strategy is the quantised position currently held
(0 after a stop/take fill), not the mark-to-market drifted fraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import hist as _hist
from .hist import Feed, LookaheadError
from .indicators import Indicators
from .tracking import TrackedState


class StrategyOutputError(ValueError):
    pass


@dataclass
class EngineConfig:
    initial_equity: float = 10000.0
    fee_bp: float = 5.0
    slippage_bp: float = 1.0
    quant_delta: float = 0.05
    bars_per_year: int = 35040
    warmup_bars: int = 960          # strategy is first called at the close of this bar
    lenient_state: bool = False     # sensitivity analysis only: also accept None / str state values
    interval_minutes: int = 15

    @classmethod
    def from_dict(cls, d: dict, minutes: int = 15):
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in keys}, interval_minutes=minutes)


@dataclass
class BacktestResult:
    equity: np.ndarray                  # mark-to-market equity at each bar close
    target_q: np.ndarray                # int grid index returned at bar t (decision for t+1)
    pos_q: np.ndarray                   # int grid index held at the close of bar t
    trades: list                        # (dir, open_bar, close_bar, ret) round trips
    n_fills: int
    state_log: list = field(default_factory=list)
    state_spans: dict = field(default_factory=dict)
    error: str | None = None            # exception class + message if the run aborted
    error_bar: int = -1

    @property
    def ok(self):
        return self.error is None


def _validate(out, t):
    if not isinstance(out, dict) or "target" not in out:
        raise StrategyOutputError(f"bar {t}: strategy must return a dict with 'target'")
    tgt = out["target"]
    try:
        tgt = float(tgt)
    except Exception as e:  # noqa: BLE001
        raise StrategyOutputError(f"bar {t}: target not a number: {tgt!r}") from e
    if not math.isfinite(tgt):
        raise StrategyOutputError(f"bar {t}: target is not finite")
    stop = out.get("stop")
    take = out.get("take")
    if stop is not None:
        stop = float(stop)
        if not math.isfinite(stop):
            stop = None
    if take is not None:
        take = float(take)
        if not math.isfinite(take):
            take = None
    return tgt, stop, take


def run_backtest(strategy, prices: dict, cfg: EngineConfig, track_state: bool = False,
                 end_bar: int | None = None) -> BacktestResult:
    feed = Feed(prices, cfg.interval_minutes)
    n = feed.n if end_bar is None else end_bar
    O, H, L, C = (feed._full[f] for f in ("open", "high", "low", "close"))
    ind = Indicators(feed)
    state = TrackedState(track=track_state, lenient=cfg.lenient_state)
    hist = feed.hist
    delta = cfg.quant_delta
    kmax = int(round(1.0 / delta))
    cost = (cfg.fee_bp) * 1e-4
    slip = cfg.slippage_bp * 1e-4

    equity = np.full(n, np.nan)
    target_q = np.zeros(n, dtype=np.int16)
    pos_q = np.zeros(n, dtype=np.int16)
    trades = []
    cash = cfg.initial_equity
    units = 0.0
    k_held = 0
    pend_k, pend_stop, pend_take = 0, None, None
    open_info = None        # (dir, open_bar, equity_before_open)
    n_fills = 0
    _hist.reset_hits()
    err, err_bar = None, -1

    def fill(new_units, price_mid, t):
        nonlocal cash, units, n_fills
        d = new_units - units
        if d == 0:
            return
        px = price_mid * (1 + slip) if d > 0 else price_mid * (1 - slip)
        cash -= d * px + abs(d) * px * cost
        units = new_units
        n_fills += 1

    def on_change(old_k, new_k, t, eq_before):
        nonlocal open_info
        if old_k != 0 and (new_k == 0 or (new_k > 0) != (old_k > 0)):
            d, ob, e0 = open_info
            trades.append((d, ob, t, eq_before / e0 - 1.0))
            open_info = None
        if new_k != 0 and (old_k == 0 or (new_k > 0) != (old_k > 0)):
            open_info = (1 if new_k > 0 else -1, t, eq_before)

    start_bar = min(cfg.warmup_bars, n)
    for t in range(start_bar):          # warm-up: history only, no decisions, flat
        feed.reveal(t)
        equity[t] = cash
    for t in range(start_bar, n):
        o, h, l, c = O[t], H[t], L[t], C[t]
        # ---- 1. execute pending decision at the open -----------------------
        if pend_k != k_held:
            eq_open = cash + units * o
            old = k_held
            closing_part = old != 0 and (pend_k == 0 or (pend_k > 0) != (old > 0))
            if closing_part:
                # flatten first so the round-trip return includes exit costs
                fill(0.0, o, t)
                on_change(old, 0, t, cash)
                old = 0
                eq_open = cash
            if pend_k != 0:
                eq_before = cash + units * o
                new_units = pend_k * delta * eq_open / o
                fill(new_units, o, t)
                if old == 0:
                    on_change(0, pend_k, t, eq_before)
            k_held = pend_k
        # ---- 2. intrabar stop / take on the held position -------------------
        if k_held != 0:
            px = None
            if k_held > 0:
                if pend_stop is not None and l <= pend_stop:
                    px = o if o <= pend_stop else pend_stop
                elif pend_take is not None and h >= pend_take:
                    px = o if o >= pend_take else pend_take
            else:
                if pend_stop is not None and h >= pend_stop:
                    px = o if o >= pend_stop else pend_stop
                elif pend_take is not None and l <= pend_take:
                    px = o if o <= pend_take else pend_take
            if px is not None:
                fill(0.0, px, t)
                on_change(k_held, 0, t, cash)
                k_held = 0
        # ---- 3. reveal bar t, call the strategy at the close ----------------
        feed.reveal(t)
        state._t = t
        pos = k_held * delta
        try:
            out = strategy(hist, state, pos, ind)
            if _hist.hits():
                raise LookaheadError("strategy accessed future data (exception was swallowed)")
            tgt, stop, take = _validate(out, t)
        except Exception as e:  # noqa: BLE001
            err, err_bar = f"{type(e).__name__}: {e}", t
            equity[t] = cash + units * c
            pos_q[t] = k_held
            break
        k = int(round(tgt / delta))
        k = max(-kmax, min(kmax, k))
        target_q[t] = k
        pos_q[t] = k_held
        equity[t] = cash + units * c
        pend_k, pend_stop, pend_take = k, stop, take

    # open trade at the end is closed at the last close (mark-to-market, no fee)
    if err is None and open_info is not None:
        d, ob, e0 = open_info
        trades.append((d, ob, n - 1, equity[n - 1] / e0 - 1.0))
    return BacktestResult(equity=equity, target_q=target_q, pos_q=pos_q, trades=trades,
                          n_fills=n_fills, state_log=state.log if track_state else [],
                          state_spans=state.spans if track_state else {},
                          error=err, error_bar=err_bar)


def perf_stats(equity: np.ndarray, initial: float, bars_per_year: int) -> dict:
    eq = equity[~np.isnan(equity)]
    if len(eq) < 2:
        return {"R": float("nan"), "sharpe": float("nan"), "maxdd": float("nan")}
    r = np.diff(eq) / eq[:-1]
    sd = r.std()
    sharpe = float(r.mean() / sd * math.sqrt(bars_per_year)) if sd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min())
    return {"R": float(eq[-1] / initial - 1.0), "sharpe": sharpe, "maxdd": maxdd}
