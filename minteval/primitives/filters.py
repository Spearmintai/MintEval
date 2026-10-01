from . import primitive


@primitive(id="no_filter", family="filter", params={},
           spec_template="No trend filter.")
def _none():
    return {}


@primitive(id="htf_trend", family="filter",
           params={"minutes": [60, 240], "n": [20, 50]},
           spec_template=("Trend filter on completed {minutes}-minute candles: only take long entries "
                          "when the last completed {minutes}-minute close is strictly above the "
                          "{n}-period EMA of those candles' closes, and only take short entries when it "
                          "is strictly below. The filter only blocks entries, never exits."))
def _htf():
    return {"gate": """
htf = hist.htf({minutes})
htf_ema = ind.ema(htf.close, {n})
if sig > 0 and not htf.close[-1] > htf_ema[-1]:
    ok = False
if sig < 0 and not htf.close[-1] < htf_ema[-1]:
    ok = False
"""}
