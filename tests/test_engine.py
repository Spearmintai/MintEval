"""Acceptance tests 2 (lookahead) and 3 (tau) plus engine-mechanics unit tests."""
import numpy as np
import pytest

from minteval.engine import EngineConfig, run_backtest
from minteval.hist import LookaheadError
from minteval.sandbox import SandboxViolation, check_source, run_sandboxed
from conftest import PRICE_FILE

FLAT = {"target": 0.0, "stop": None, "take": None}


def _slice(prices, n):
    return {k: v[:n] for k, v in prices.items()}


# ---------------------------------------------------------------- test 2: lookahead
def test_lookahead_index_raises(prices, cfg):
    def peek(hist, state, pos, ind):
        t = len(hist.close) - 1
        _ = hist.close[t + 1]
        return FLAT
    r = run_backtest(peek, _slice(prices, 500), cfg)
    assert r.error is not None and r.error.startswith("LookaheadError"), r.error
    assert r.error_bar == 0


def test_lookahead_slice_raises(prices, cfg):
    def peek(hist, state, pos, ind):
        t = len(hist) - 1
        _ = hist.high[t - 5:t + 2]
        return FLAT
    r = run_backtest(peek, _slice(prices, 500), cfg)
    assert r.error.startswith("LookaheadError")


def test_lookahead_swallowed_is_still_flagged(prices, cfg):
    def sneaky(hist, state, pos, ind):
        try:
            _ = hist.close[len(hist.close)]
        except Exception:
            pass
        return FLAT
    r = run_backtest(sneaky, _slice(prices, 500), cfg)
    assert r.error.startswith("LookaheadError")


def test_future_not_in_memory(prices, cfg):
    """Physical enforcement: buffers behind every reachable array hold NaN after t."""
    seen = []

    def probe(hist, state, pos, ind):
        t = len(hist) - 1
        if t == 100:
            base = np.asarray(hist.close).base
            while getattr(base, "base", None) is not None:
                base = base.base
            seen.append(np.isnan(base[t + 1:]).all())
            e = ind.ema(hist.close, 20)
            eb = e.base
            while getattr(eb, "base", None) is not None:
                eb = eb.base
            seen.append(np.isnan(eb[t + 1:]).all())
            h4 = hist.htf(240)
            hb = np.asarray(h4.close).base
            while getattr(hb, "base", None) is not None:
                hb = hb.base
            seen.append(np.isnan(hb[len(h4.close):]).all())
        return FLAT
    run_backtest(probe, _slice(prices, 300), cfg)
    assert seen == [True, True, True]


def test_lookahead_in_sandbox():
    code = ("def strategy(hist, state, pos, ind):\n"
            "    x = hist.close[len(hist.close)]\n"
            "    return {'target': 0.0, 'stop': None, 'take': None}\n")
    r = run_sandboxed(code, PRICE_FILE, EngineConfig().__dict__)
    assert r["error"].startswith("LookaheadError"), r["error"]


def test_indicators_match_truncated_recompute(prices, cfg):
    """Cached (precomputed) indicator values equal recomputation on hist[0..t] only."""
    from minteval import indicators as I
    rng = np.random.default_rng(0)
    checks = set(rng.integers(50, 3000, 25).tolist())
    bad = []

    def chk(hist, state, pos, ind):
        t = len(hist) - 1
        if t in checks:
            c = np.asarray(hist.close).copy()
            h, l = np.asarray(hist.high).copy(), np.asarray(hist.low).copy()
            pairs = [(ind.sma(hist.close, 20)[-1], I._sma(c, 20)[-1]),
                     (ind.ema(hist.close, 50)[-1], I._ema(c, 50)[-1]),
                     (ind.atr(hist, 14)[-1], I._atr(h, l, c, 14)[-1]),
                     (ind.rsi(hist.close, 14)[-1], I._rsi(c, 14)[-1]),
                     (ind.highest(hist.high, 20)[-1], I._highest(h, 20)[-1]),
                     (ind.macd(hist.close)[1][-1], I._macd(c, 12, 26, 9)[1][-1])]
            for a, b in pairs:
                if not (a == b or (np.isnan(a) and np.isnan(b))):
                    bad.append((t, a, b))
        return FLAT
    run_backtest(chk, _slice(prices, 3001), cfg)
    assert not bad, bad[:5]


def test_htf_only_completed_bars(prices, cfg):
    got = {}

    def chk(hist, state, pos, ind):
        t = len(hist) - 1
        if t in (2, 3, 4, 7):
            h = hist.htf(60)
            got[t] = (len(h), float(h.close[-1]) if len(h) else None)
        return FLAT
    run_backtest(chk, _slice(prices, 10), cfg)
    c = prices["close"]
    # data starts at 00:00 UTC: 1h bar 0 completes at the close of 15m bar 3
    assert got[2][0] == 0 and got[3] == (1, c[3]) and got[4] == (1, c[3]) and got[7] == (2, c[7])


# ---------------------------------------------------------------- test 3: tau
def test_tau_exact_three(prices, cfg):
    def s(hist, state, pos, ind):
        t = len(hist) - 1
        if t == 3:
            state["x"] = 1.0
        if t == 6:
            _ = state["x"]
        return FLAT
    r = run_backtest(s, _slice(prices, 20), cfg, track_state=True)
    from minteval.complexity import state_complexity
    sc = state_complexity(r.state_spans)
    print("spans:", r.state_spans, "->", sc)
    assert r.state_spans == {"x": [3]}
    assert sc["tau_max"] == 3 and sc["tau_p90"] == 3 and sc["n_registers"] == 1
    assert (6, "x", "r") in r.state_log and (3, "x", "w") in r.state_log


def test_tau_same_bar_and_rewrite(prices, cfg):
    def s(hist, state, pos, ind):
        t = len(hist) - 1
        if t in (2, 5):
            state["y"] = float(t)
        if t in (2, 4, 9):
            _ = state["y"]          # t=2 same bar (ignored), t=4 -> 2, t=9 -> 4
        if t == 9:
            _ = "z" in state         # miss, ignored
        return FLAT
    r = run_backtest(s, _slice(prices, 20), cfg, track_state=True)
    assert r.state_spans == {"y": [2, 4]}


# ---------------------------------------------------------------- engine mechanics
def _synthetic(o, h, l, c):
    n = len(o)
    return {"open": np.array(o, float), "high": np.array(h, float), "low": np.array(l, float),
            "close": np.array(c, float), "volume": np.ones(n),
            "open_time": np.arange(n, dtype=np.int64) * 900_000}


def test_fill_next_open_and_stop_priority():
    cfg = EngineConfig(fee_bp=0.0, slippage_bp=0.0, warmup_bars=0)
    # bar0 decide long with stop 95 & take 105; bar1 opens 100, low 90, high 110 -> stop first
    P = _synthetic([100, 100, 100], [100, 110, 100], [100, 90, 100], [100, 100, 100])

    def s(hist, state, pos, ind):
        if len(hist) == 1:
            return {"target": 1.0, "stop": 95.0, "take": 105.0}
        return {"target": 0.0, "stop": None, "take": None}
    r = run_backtest(s, P, cfg)
    assert r.trades[0][:3] == (1, 1, 1)
    assert abs(r.trades[0][3] - (-0.05)) < 1e-12
    assert list(r.pos_q) == [0, 0, 0]


def test_gap_through_stop_fills_at_open():
    cfg = EngineConfig(fee_bp=0.0, slippage_bp=0.0, warmup_bars=0)
    P = _synthetic([100, 100, 90, 90], [100, 100, 91, 90], [100, 99, 85, 90], [100, 100, 90, 90])

    def s(hist, state, pos, ind):
        return {"target": 1.0, "stop": 95.0, "take": None} if len(hist) == 1 else \
               {"target": pos, "stop": 95.0, "take": None}
    r = run_backtest(s, P, cfg)
    assert r.trades[0][:3] == (1, 1, 2)
    assert abs(r.trades[0][3] - (-0.10)) < 1e-12


def test_costs_charged():
    cfg = EngineConfig(fee_bp=5.0, slippage_bp=1.0, warmup_bars=0)
    P = _synthetic([100] * 4, [100] * 4, [100] * 4, [100] * 4)

    def s(hist, state, pos, ind):
        return {"target": 1.0 if len(hist) == 1 else 0.0, "stop": None, "take": None}
    r = run_backtest(s, P, cfg)
    # units sized on the open (100) with equity 1; buy at 100.01 + 5bp fee, sell at 99.99 - 5bp
    exp = 0.9999 * (1 - 5e-4) - 1.0001 * (1 + 5e-4)
    assert abs(r.trades[0][3] - exp) < 1e-9


def test_flip_creates_two_round_trips():
    cfg = EngineConfig(fee_bp=0.0, slippage_bp=0.0, warmup_bars=0)
    P = _synthetic([100] * 6, [100] * 6, [100] * 6, [100] * 6)

    def s(hist, state, pos, ind):
        t = len(hist) - 1
        return {"target": {0: 1.0, 1: 1.0, 2: -0.5}.get(t, 0.0), "stop": None, "take": None}
    r = run_backtest(s, P, cfg)
    assert [tr[:3] for tr in r.trades] == [(1, 1, 3), (-1, 3, 4)]


# ---------------------------------------------------------------- sandbox rules
@pytest.mark.parametrize("bad", [
    "import os\ndef strategy(hist, state, pos, ind):\n    return {}\n",
    "X = []\ndef strategy(hist, state, pos, ind):\n    return {}\n",
    "def strategy(hist, state, pos, ind, cache={}):\n    return {}\n",
    "def strategy(hist, state, pos, ind):\n    global Y\n    return {}\n",
    "def strategy(hist, state, pos, ind):\n    def f(c=[]):\n        return 1\n    return {}\n",
    "def strategy(hist, state, pos, ind):\n    x = 0\n    def f():\n        nonlocal x\n    return {}\n",
    "def strategy(hist, state, pos, ind):\n    hist.foo = 1\n    return {}\n",
    "def strategy(hist, state, pos, ind):\n    return open('x')\n",
    "def strategy(hist, state, pos, ind):\n    return hist._feed\n",
    "def strategy(hist, state, pos, ind):\n    return ().__class__\n",
])
def test_static_rejects(bad):
    with pytest.raises(SandboxViolation):
        check_source(bad)


def test_state_must_be_flat(prices, cfg):
    def s(hist, state, pos, ind):
        state["x"] = [1, 2]
        return FLAT
    r = run_backtest(s, _slice(prices, 10), cfg)
    assert r.error.startswith("StateTypeError")


def test_runtime_file_access_blocked():
    code = ("import numpy as np\n"
            "def strategy(hist, state, pos, ind):\n"
            "    np.load('data/prices/BTCUSDT_15m_2022_2023.npz')\n"
            "    return {'target': 0.0, 'stop': None, 'take': None}\n")
    r = run_sandboxed(code, PRICE_FILE, EngineConfig().__dict__)
    assert r["error"] and "PermissionError" in r["error"], r["error"]


def test_timeout():
    code = ("def strategy(hist, state, pos, ind):\n"
            "    while True:\n"
            "        pass\n")
    r = run_sandboxed(code, PRICE_FILE, EngineConfig().__dict__, timeout_s=2)
    assert r["error"].startswith("Timeout"), r["error"]


def test_prefix_slice_indicators_exact(prices, cfg):
    """ind.f(hist.x[:-k]) is served from the cache; must equal recomputation on the slice."""
    from minteval import indicators as I
    bad, seen = [], []

    def chk(hist, state, pos, ind):
        t = len(hist) - 1
        if t in (300, 1234, 2999):
            hs = hist.high[:-1]
            a = ind.highest(hs, 48)
            seen.append(len(a) == t)
            b = I._highest(np.asarray(hs).copy(), 48)
            if not np.array_equal(a, b, equal_nan=True):
                bad.append(("highest", t))
            e = ind.ema(hist.close[:-3], 20)
            if not np.array_equal(e, I._ema(np.asarray(hist.close[:-3]).copy(), 20), equal_nan=True):
                bad.append(("ema", t))
            w = ind.ema(hist.close[-200:], 20)          # non-prefix slice: direct computation
            if not np.array_equal(w, I._ema(np.asarray(hist.close[-200:]).copy(), 20)):
                bad.append(("window", t))
            h4 = hist.htf(240)
            x = ind.sma(h4.close[:-1], 5)
            if not np.array_equal(x, I._sma(np.asarray(h4.close[:-1]).copy(), 5), equal_nan=True):
                bad.append(("htf", t))
        return FLAT
    run_backtest(chk, _slice(prices, 3001), cfg)
    assert not bad and seen == [True] * 3, (bad, seen)
