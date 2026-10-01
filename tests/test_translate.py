import json
from conftest import ROOT
from minteval.translate import validate, load_jargon

SPEC = {"signal": {"id": "ma_cross", "params": {"kind": "sma", "fast": 5, "slow": 50}}, "direction": "long_short",
        "filter": {"id": "no_filter", "params": {}}, "sizing": {"id": "atr_risk", "params": {"r": 0.0025, "k": 2.0}},
        "risk": [{"id": "trailing_hwm", "params": {"b": 4.0, "after_be": False}}]}
BASE = ("Trade the 5 and 50 sma crossover both ways when flat, go long or get short, trail it 4 ATR behind "
        "the best price, and an opposite crossover gets you flat at the next open, nothing fancy here. ")


def test_vol_sizing_cue_accepts_decimal_percent():
    v = validate(BASE + "Size: risk 0.25% of equity per 2 ATR, capped at 1.0, floored at 0.05.", SPEC, load_jargon())
    assert not any("volatility sizing" in p for p in v["problems"]), v["problems"]


def test_vol_sizing_cue_rejects_ambiguous():
    v = validate(BASE + "Size in at 0.25% equity over 2 ATR, capped at 1.0, floored at 0.05.", SPEC, load_jargon())
    assert any("volatility sizing" in p for p in v["problems"])
