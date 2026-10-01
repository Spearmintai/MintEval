"""History feed with physically enforced no-lookahead.

The engine reveals bar t by *copying* it into a NaN-initialised visible buffer
right before the strategy is called at the close of bar t.  Everything a
strategy can reach through ``hist`` (and through the arrays returned by
``ind``) is a view of such a buffer, so data from bars > t is simply not in
memory that the strategy can address without private (``_``-prefixed)
attributes, which the sandbox's AST check forbids.

Indexing a history array at position >= len (i.e. t+1 or later) raises
LookaheadError. The hit is also recorded in a module counter so that a strategy
that swallows the exception with ``except Exception`` is still flagged.
"""
from __future__ import annotations

import numpy as np

FIELDS = ("open", "high", "low", "close", "volume")


class LookaheadError(Exception):
    """Raised when a strategy touches data from the future."""


_HITS = [0]


def reset_hits():
    _HITS[0] = 0


def hits() -> int:
    return _HITS[0]


def _lookahead(msg):
    _HITS[0] += 1
    raise LookaheadError(msg)


class HistArray(np.ndarray):
    """1-D read-only view of bars [0..t]. Index >= len raises LookaheadError.

    Derived results (slices, arithmetic, numpy functions) are plain ndarrays.
    """

    def __getitem__(self, idx):
        n = self.shape[0]
        base = self.view(np.ndarray)
        if isinstance(idx, (int, np.integer)):
            if idx >= n:
                _lookahead(f"index {int(idx)} >= {n}: bar t+1 or later is in the future")
            return base[idx]
        if isinstance(idx, slice):
            if idx.stop is not None and isinstance(idx.stop, (int, np.integer)) and idx.stop > n:
                _lookahead(f"slice stop {int(idx.stop)} > {n}: future bars")
            if idx.start is not None and isinstance(idx.start, (int, np.integer)) and idx.start > n:
                _lookahead(f"slice start {int(idx.start)} > {n}: future bars")
            return base[idx]
        if isinstance(idx, (list, np.ndarray)):
            arr = np.asarray(idx)
            if arr.dtype != bool and arr.size and arr.max() >= n:
                _lookahead("fancy index into future bars")
            return base[idx]
        return base[idx]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        inputs = tuple(i.view(np.ndarray) if isinstance(i, HistArray) else i for i in inputs)
        if "out" in kwargs:
            kwargs["out"] = tuple(o.view(np.ndarray) if isinstance(o, HistArray) else o
                                  for o in kwargs["out"])
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        def strip(a):
            if isinstance(a, HistArray):
                return a.view(np.ndarray)
            if isinstance(a, (list, tuple)):
                return type(a)(strip(x) for x in a)
            return a
        return func(*strip(args), **{k: strip(v) for k, v in kwargs.items()})

    def __iter__(self):
        return iter(self.view(np.ndarray))

    def tolist(self):
        return self.view(np.ndarray).tolist()


def _ro_view(buf: np.ndarray, n: int, cls=None) -> np.ndarray:
    v = buf[:n]
    if cls is not None:
        v = v.view(cls)
    v.flags.writeable = False
    return v


_ndget = np.ndarray.__getitem__


class Hist:
    """What the strategy sees as ``hist``.

    Attributes: open, high, low, close, volume, time (bar open time, ms UTC) --
    each a HistArray over bars [0..t]. ``len(hist)`` == t+1.
    ``hist.htf(minutes)`` returns a Hist of *completed* higher-timeframe bars
    (aligned to UTC boundaries; the bar in progress is not included).
    """

    __slots__ = ("_bufs", "_n", "_views", "_vt", "_htf", "_minutes", "_feed")

    def __init__(self, bufs: dict, minutes: int, feed):
        # strategies only ever get slices of read-only aliases of the buffers
        ro = {}
        for f, b in bufs.items():
            a = b.view(HistArray)
            a.flags.writeable = False
            ro[f] = a
        self._bufs = ro
        self._n = 0
        self._views = {}
        self._vt = -1
        self._htf = {}
        self._minutes = minutes
        self._feed = feed

    def _view(self, f):
        if self._vt != self._n:
            self._views = {}
            self._vt = self._n
        v = self._views.get(f)
        if v is None:
            v = _ndget(self._bufs[f], slice(0, self._n))
            self._views[f] = v
            self._feed._tags[id(v)] = (self._minutes, f)
        return v

    open = property(lambda s: s._view("open"))
    high = property(lambda s: s._view("high"))
    low = property(lambda s: s._view("low"))
    close = property(lambda s: s._view("close"))
    volume = property(lambda s: s._view("volume"))
    time = property(lambda s: s._view("time"))

    def __len__(self):
        return self._n

    def htf(self, minutes: int):
        return self._feed.htf(int(minutes))


class Feed:
    """Owns the full data (private) and the progressively-revealed buffers."""

    def __init__(self, prices: dict, minutes: int):
        self._full = {f: np.ascontiguousarray(prices[f], dtype=np.float64) for f in FIELDS}
        self._full["time"] = np.asarray(prices["open_time"], dtype=np.float64)
        self.n = len(self._full["close"])
        self.minutes = minutes
        self._bufs = {f: np.full(self.n, np.nan) for f in self._full}
        self._pairs = [(self._bufs[f], self._full[f]) for f in self._full]
        self._tags = {}         # id(visible view) -> (minutes, field); views live in Hist._views
        self._addr = {b.ctypes.data: (minutes, f) for f, b in self._bufs.items()}
        self.hist = Hist(self._bufs, minutes, self)
        self._htf_full = {}     # minutes -> (full dict, m_of_t array)
        self._htf_bufs = {}
        self._htf_hist = {}
        self._htf_filled = {}
        self.t = -1

    def reveal(self, t: int):
        if len(self._tags) > 4096:
            self._tags = {}
        for b, full in self._pairs:
            b[t] = full[t]
        self.t = t
        self.hist._n = t + 1

    # ---- higher timeframes -------------------------------------------------
    def _build_htf(self, minutes):
        if minutes % self.minutes or minutes <= self.minutes:
            raise ValueError(f"htf minutes must be a multiple of {self.minutes} and larger")
        ms = minutes * 60_000
        tt = self._full["time"].astype(np.int64)
        bucket = tt // ms
        # bar t completes its bucket iff the next bar starts a new bucket (or t is the
        # last bar of the bucket by clock: close time is a multiple of ms)
        closes_bucket = ((tt + self.minutes * 60_000) % ms) == 0
        starts = np.r_[0, np.flatnonzero(np.diff(bucket)) + 1]
        ends = np.r_[starts[1:], len(tt)]
        o = self._full["open"][starts]
        c = self._full["close"][ends - 1]
        h = np.maximum.reduceat(self._full["high"], starts)
        lo = np.minimum.reduceat(self._full["low"], starts)
        v = np.add.reduceat(self._full["volume"], starts)
        tm = (bucket[starts] * ms).astype(np.float64)
        full = {"open": o, "high": h, "low": lo, "close": c, "volume": v, "time": tm}
        # m_of_t[t] = number of htf bars completed by the close of bar t
        bucket_idx = np.repeat(np.arange(len(starts)), ends - starts)
        # completion is only known from the clock: the 15m bar whose close lands on the
        # boundary completes the bucket. If that bar is missing (data gap) the bucket is
        # known complete once the first bar of the next bucket is revealed.
        m_of_t = np.where(closes_bucket, bucket_idx + 1, bucket_idx)
        self._htf_full[minutes] = (full, m_of_t)
        self._htf_bufs[minutes] = {f: np.full(len(o), np.nan) for f in full}
        for f, b in self._htf_bufs[minutes].items():
            self._addr[b.ctypes.data] = (minutes, f)
        self._htf_filled[minutes] = 0
        h = Hist(self._htf_bufs[minutes], minutes, self)
        self._htf_hist[minutes] = h

    def htf(self, minutes):
        if minutes not in self._htf_full:
            self._build_htf(minutes)
        full, m_of_t = self._htf_full[minutes]
        m = int(m_of_t[self.t])
        filled = self._htf_filled[minutes]
        if m > filled:
            for f, b in self._htf_bufs[minutes].items():
                b[filled:m] = full[f][filled:m]
            self._htf_filled[minutes] = m
        h = self._htf_hist[minutes]
        h._n = m
        return h

    # ---- tags used by the indicator cache ------------------------------------
    def tag_of(self, x):
        """Return (minutes, field, length) if x is a prefix view [0:length] of a visible feed series.

        Causal indicators on a prefix equal the prefix of the indicator on the full series, so
        such inputs (e.g. ``hist.high[:-1]``) can be served exactly from the cache."""
        tag = self._tags.get(id(x))
        if tag is not None:
            h = self.hist if tag[0] == self.minutes else self._htf_hist.get(tag[0])
            if h is not None and h._vt == h._n and h._views.get(tag[1]) is x:
                return (tag[0], tag[1], h._n)
        if not isinstance(x, np.ndarray) or x.ndim != 1 or x.dtype != np.float64 or x.strides != (8,):
            return None
        addr = x.__array_interface__["data"][0]
        hit = self._addr.get(addr)
        if hit is None:
            return None
        mins, f = hit
        vis = self.t + 1 if mins == self.minutes else self._htf_hist[mins]._n
        if len(x) > vis:
            return None
        return (mins, f, len(x))

    def tag_of_hist(self, h):
        if h is self.hist:
            return self.minutes
        for mins, hh in self._htf_hist.items():
            if hh is h:
                return mins
        return None

    def full_series(self, minutes, field):
        if minutes == self.minutes:
            return self._full[field]
        return self._htf_full[minutes][0][field]

    def visible_len(self, minutes):
        if minutes == self.minutes:
            return self.t + 1
        return self._htf_hist[minutes]._n
