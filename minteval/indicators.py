"""History-only technical indicators, exposed to strategies as ``ind``.

Exact definitions (also given verbatim to the models under test):
  sma(x, n)       mean of the last n values; NaN for the first n-1 bars
  ema(x, n)       e[0]=x[0]; e[i]=a*x[i]+(1-a)*e[i-1], a=2/(n+1)
  rma(x, n)       Wilder smoothing: as ema but a=1/n
  atr(h, n)       rma(TR, n), TR[0]=high-low, TR[i]=max(h-l, |h-c[i-1]|, |l-c[i-1]|)
  rsi(x, n)       100-100/(1+rma(gain,n)/rma(loss,n)), gain/loss from diff (first = 0)
  macd(x,f,s,g)   line=ema(x,f)-ema(x,s); signal=ema(line,g); hist=line-signal
  highest(x, n)   max of the last n values incl. current; NaN for first n-1
  lowest(x, n)    min of the last n values incl. current; NaN for first n-1
Every function returns arrays aligned with its input (same length, last = current bar).

Every definition is causal (value i depends only on x[0..i]).  When the input is a
series of the feed (hist.close, hist.htf(60).high, ...), the full causal series
is computed once and revealed bar by bar through a NaN-initialised buffer, so
the arrays returned never contain future values.  Arbitrary arrays are computed
directly on what is passed in.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NAN = np.nan
_ndget = np.ndarray.__getitem__


def _sma(x, n):
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).mean(axis=1)
    return out


def _ewm(x, a):
    out = np.empty(len(x))
    if len(x) == 0:
        return out
    acc = x[0]
    out[0] = acc
    b = 1.0 - a
    for i in range(1, len(x)):
        acc = a * x[i] + b * acc
        out[i] = acc
    return out


try:  # optional speed-up; identical arithmetic
    from numba import njit
    _ewm = njit(cache=True)(_ewm)
except Exception:  # pragma: no cover
    pass


def _ema(x, n):
    return _ewm(np.ascontiguousarray(x, dtype=np.float64), 2.0 / (n + 1))


def _rma(x, n):
    return _ewm(np.ascontiguousarray(x, dtype=np.float64), 1.0 / n)


def _roll(x, n, fn):
    out = np.full(len(x), NAN)
    if len(x) >= n:
        out[n - 1:] = fn(sliding_window_view(x, n), axis=1)
    return out


def _highest(x, n):
    return _roll(x, n, np.max)


def _lowest(x, n):
    return _roll(x, n, np.min)


def _tr(h, l, c):
    tr = h - l
    if len(c) > 1:
        pc = c[:-1]
        tr = tr.copy()
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - pc), np.abs(l[1:] - pc)])
    return tr


def _atr(h, l, c, n):
    return _rma(_tr(h, l, c), n)


def _rsi(x, n):
    d = np.diff(x, prepend=x[0]) if len(x) else x
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag, al = _rma(gain, n), _rma(loss, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ag / al
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where(al == 0, np.where(ag == 0, 50.0, 100.0), out)
    return out


def _macd(x, f, s, g):
    line = _ema(x, f) - _ema(x, s)
    sig = _ema(line, g)
    return line, sig, line - sig


def _check_n(n):
    if int(n) != n or n < 1:
        raise ValueError("window must be a positive integer")
    return int(n)


class Indicators:
    """Bound to one backtest's Feed. Instances are what strategies receive as ``ind``."""

    __slots__ = ("_feed", "_cache")

    def __init__(self, feed=None):
        self._feed = feed
        self._cache = {}

    # -- cache machinery -------------------------------------------------------
    def _cached(self, key, minutes, compute_full, n_out=1):
        m = self._feed.visible_len(minutes)
        ent = self._cache.get(key)
        if ent is None:
            full = compute_full()
            if n_out == 1:
                full = (full,)
            bufs = tuple(np.full(len(f), NAN) for f in full)
            ros = []
            for bb in bufs:
                r = bb.view()
                r.flags.writeable = False
                ros.append(r)
            ent = [full, bufs, 0, tuple(ros), -1, None]
            self._cache[key] = ent
        elif ent[4] == m:
            return ent[5]
        lo = ent[2]
        if m == lo + 1:
            for f, bb in zip(ent[0], ent[1]):
                bb[lo] = f[lo]
            ent[2] = m
        elif m > lo:
            for f, bb in zip(ent[0], ent[1]):
                bb[lo:m] = f[lo:m]
            ent[2] = m
        if n_out == 1:
            out = _ndget(ent[3][0], slice(0, m))
        else:
            out = tuple(r[:m] for r in ent[3])
        ent[4], ent[5] = m, out
        return out

    def _series(self, name, x, params, fn, n_out=1):
        tag = self._feed.tag_of(x) if self._feed is not None else None
        if tag is None:
            arr = np.asarray(x, dtype=np.float64)
            return fn(arr, *(_check_n(p) for p in params))
        mins, field = tag
        key = (name, mins, field, params)
        if key not in self._cache:
            params = tuple(_check_n(p) for p in params)
        return self._cached(key, mins,
                            lambda: fn(self._feed.full_series(mins, field), *params), n_out)

    def _bars(self, name, h, params, fn):
        mins = self._feed.tag_of_hist(h) if self._feed is not None else None
        if mins is None:
            return fn(np.asarray(h.high, float), np.asarray(h.low, float),
                      np.asarray(h.close, float), *params)
        fs = self._feed.full_series
        params = tuple(_check_n(p) for p in params)
        return self._cached((name, mins, "hlc", params), mins,
                            lambda: fn(fs(mins, "high"), fs(mins, "low"), fs(mins, "close"), *params))

    # -- public API ------------------------------------------------------------
    def sma(self, x, n):
        return self._series("sma", x, (n,), _sma)

    def ema(self, x, n):
        return self._series("ema", x, (n,), _ema)

    def rma(self, x, n):
        return self._series("rma", x, (n,), _rma)

    def rsi(self, x, n=14):
        return self._series("rsi", x, (n,), _rsi)

    def highest(self, x, n):
        return self._series("highest", x, (n,), _highest)

    def lowest(self, x, n):
        return self._series("lowest", x, (n,), _lowest)

    def macd(self, x, fast=12, slow=26, signal=9):
        return self._series("macd", x, (fast, slow, signal), _macd, n_out=3)

    def atr(self, h, n=14):
        return self._bars("atr", h, (n,), _atr)
