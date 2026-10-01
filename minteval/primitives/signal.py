from . import primitive

CROSS = """
sig = 0
if {a}[-2] <= {b}[-2] and {a}[-1] > {b}[-1]:
    sig = 1
elif {a}[-2] >= {b}[-2] and {a}[-1] < {b}[-1]:
    sig = -1
"""


@primitive(id="ma_cross", family="signal",
           params={"kind": ["sma", "ema"], "fast": [5, 10, 20], "slow": [50, 100, 200]},
           spec_template=("Entry signal: {kind} crossover. Bullish when the {fast}-bar {kind} of the "
                          "close crosses above the {slow}-bar {kind} (it was at or below it on the "
                          "previous bar and is strictly above it now); bearish on the mirror-image "
                          "cross below."))
def _ma_cross():
    return {"signal": """
fast = ind.{kind}(close, {fast})
slow = ind.{kind}(close, {slow})
""" + CROSS.format(a="fast", b="slow")}


@primitive(id="macd_cross", family="signal",
           params={"fsg": [(12, 26, 9), (8, 17, 9), (5, 35, 5)]},
           derive=lambda v: {"f": v["fsg"][0], "s": v["fsg"][1], "g": v["fsg"][2],
                             "fsg": "{},{},{}".format(*v["fsg"])},
           spec_template=("Entry signal: MACD({fsg}) line crossing its signal line. Bullish when the "
                          "MACD line crosses above the signal line (at or below on the previous bar, "
                          "strictly above now); bearish on the cross below."))
def _macd_cross():
    return {"signal": """
macd_line, macd_signal, macd_hist = ind.macd(close, {f}, {s}, {g})
""" + CROSS.format(a="macd_line", b="macd_signal")}


@primitive(id="donchian_break", family="signal",
           params={"n": [20, 48, 96, 192]},
           spec_template=("Entry signal: Donchian channel breakout. Bullish whenever the close is "
                          "strictly above the highest high of the {n} bars before the current bar; "
                          "bearish whenever the close is strictly below the lowest low of those {n} "
                          "bars. This is a level condition, true on every bar it holds."))
def _donchian():
    return {"signal": """
upper = ind.highest(high, {n})[-2]
lower = ind.lowest(low, {n})[-2]
sig = 0
if price > upper:
    sig = 1
elif price < lower:
    sig = -1
"""}


@primitive(id="rsi_revert", family="signal",
           params={"n": [7, 14, 21], "lo": [20, 25, 30]},
           spec_template=("Entry signal: RSI({n}) mean reversion. Bullish when RSI crosses back up "
                          "through {lo} (below {lo} on the previous bar, at or above {lo} now); bearish "
                          "when it crosses back down through 100-{lo} (above it on the previous bar, at "
                          "or below it now)."))
def _rsi():
    return {"signal": """
rsi = ind.rsi(close, {n})
sig = 0
if rsi[-2] < {lo} and rsi[-1] >= {lo}:
    sig = 1
elif rsi[-2] > 100 - {lo} and rsi[-1] <= 100 - {lo}:
    sig = -1
"""}


@primitive(id="box_breakout", family="signal",
           params={"m": [16, 32, 64], "w": [3.0, 4.0, 6.0]},
           spec_template=("Entry signal: range-box breakout. The box is the highest high and lowest "
                          "low of the {m} bars before the current bar. It only counts as a box if its "
                          "height (top minus bottom) is at most {w} times the current ATR(14). Bullish "
                          "when the close is strictly above the top of a valid box; bearish when the "
                          "close is strictly below its bottom."))
def _box():
    return {"signal": """
box_top = ind.highest(high, {m})[-2]
box_bot = ind.lowest(low, {m})[-2]
sig = 0
if box_top - box_bot <= {w} * atr:
    if price > box_top:
        sig = 1
    elif price < box_bot:
        sig = -1
"""}


@primitive(id="breakout_pullback", family="signal",
           params={"n": [48, 96], "wait": [8, 16, 32, 64]},
           spec_template=("Entry signal: breakout then retest. When the close is strictly above the "
                          "highest high of the {n} bars before the current bar, remember that high as "
                          "the breakout level (a newer breakout replaces the old one and restarts the "
                          "clock). The setup stays armed for {wait} bars after the breakout bar. While "
                          "armed, the first bar whose low trades at or below the level is the bullish "
                          "signal and disarms the setup; if more than {wait} bars pass first, the setup "
                          "simply expires. The bearish version is the mirror image (close below the "
                          "{n}-bar lowest low, then a high at or above that level). Arming, expiry and "
                          "triggering are checked on every bar, in or out of a position; a newer "
                          "breakout in either direction replaces the armed setup."))
def _bp():
    return {"signal": """
upper = ind.highest(high, {n})[-2]
lower = ind.lowest(low, {n})[-2]
sig = 0
if "arm_dir" in state:
    if t > state["arm_until"]:
        del state["arm_dir"]
        del state["arm_lvl"]
        del state["arm_until"]
    elif state["arm_dir"] > 0 and low[-1] <= state["arm_lvl"]:
        sig = 1
    elif state["arm_dir"] < 0 and high[-1] >= state["arm_lvl"]:
        sig = -1
    if sig != 0:
        del state["arm_dir"]
        del state["arm_lvl"]
        del state["arm_until"]
if price > upper:
    state["arm_dir"] = 1
    state["arm_lvl"] = upper
    state["arm_until"] = t + {wait}
elif price < lower:
    state["arm_dir"] = -1
    state["arm_lvl"] = lower
    state["arm_until"] = t + {wait}
"""}
