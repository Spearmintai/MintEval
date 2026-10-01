"""TrackedState: a flat str -> float|int|bool mapping that logs every read and write.

Log entries are (t, key, op) with op in {"r", "w", "d"}. A read of a key that is
absent (e.g. ``"k" in state`` returning False, or ``state.get("k", 0)`` on a missing
key) is logged as op "m" (miss) and never contributes to tau.
"""
from __future__ import annotations

import math

import numpy as np

_SCALARS = (bool, int, float)


class StateTypeError(TypeError):
    pass


def _coerce(key, value):
    if type(key) is not str:
        raise StateTypeError(f"state keys must be str, got {type(key).__name__}")
    if isinstance(value, _SCALARS):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise StateTypeError(
        f"state[{key!r}] must be float | int | bool, got {type(value).__name__}")


class TrackedState:
    __slots__ = ("_d", "_t", "_track", "log", "_last_w", "spans")

    def __init__(self, track: bool = True):
        self._d: dict = {}
        self._t = -1
        self._track = track
        self.log: list[tuple[int, str, str]] = []
        self._last_w: dict[str, int] = {}
        self.spans: dict[str, list[int]] = {}   # key -> cross-bar read spans (t_read - t_last_write)

    # -- logging helpers -------------------------------------------------
    def _r(self, key):
        if not self._track:
            return
        if key in self._d:
            self.log.append((self._t, key, "r"))
            lw = self._last_w[key]
            if self._t > lw:
                self.spans.setdefault(key, []).append(self._t - lw)
        else:
            self.log.append((self._t, key, "m"))

    def _w(self, key):
        if self._track:
            self.log.append((self._t, key, "w"))
            self._last_w[key] = self._t

    # -- mapping API -----------------------------------------------------
    def __getitem__(self, key):
        self._r(key)
        return self._d[key]

    def get(self, key, default=None):
        self._r(key)
        return self._d.get(key, default)

    def __contains__(self, key):
        self._r(key)
        return key in self._d

    def __setitem__(self, key, value):
        value = _coerce(key, value)
        if isinstance(value, float) and math.isinf(value):
            pass  # inf/nan are legal floats
        self._d[key] = value
        self._w(key)

    def setdefault(self, key, default=None):
        if key in self._d:
            return self[key]
        self[key] = default
        return self._d[key]

    def update(self, other=(), **kw):
        for k, v in dict(other, **kw).items():
            self[k] = v

    def __delitem__(self, key):
        del self._d[key]
        if self._track:
            self.log.append((self._t, key, "d"))
            self._last_w.pop(key, None)

    def pop(self, key, *default):
        if key in self._d:
            v = self[key]
            del self[key]
            return v
        if self._track:
            self.log.append((self._t, key, "m"))
        if default:
            return default[0]
        raise KeyError(key)

    def clear(self):
        for k in list(self._d):
            del self[k]

    def keys(self):
        return list(self._d.keys())

    def values(self):
        return [self[k] for k in list(self._d)]

    def items(self):
        return [(k, self[k]) for k in list(self._d)]

    def __iter__(self):
        return iter(list(self._d))

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f"TrackedState({self._d!r})"

    def snapshot(self) -> dict:
        return dict(self._d)
