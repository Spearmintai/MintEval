from . import primitive


@primitive(id="fixed_frac", family="sizing",
           params={"f": [0.25, 0.5, 0.75, 1.0]},
           spec_template="Position size: a fixed {f} of equity per trade.")
def _ff():
    return {"size": "{f}"}


@primitive(id="atr_risk", family="sizing",
           params={"r": [0.0025, 0.005, 0.01], "k": [1.0, 2.0, 3.0]},
           spec_template=("Position size: volatility-scaled. Fraction of equity = {r} x close / "
                          "({k} x ATR(14)) on the signal bar, capped at 1.0 and floored at 0.05."))
def _atr():
    return {"size": "min(1.0, max(0.05, {r} * price / ({k} * atr)))"}


@primitive(id="pyramid", family="sizing",
           params={"f0": [0.25, 0.5], "step": [1.0, 2.0], "adds": [1, 2]},
           excludes=("partial_take",),
           spec_template=("Position size: start with {f0} of equity, then pyramid. While in the trade, "
                          "each time the close is at least {step} x the entry ATR beyond the price of "
                          "the last add (initially the entry price) in the trade's favour, add another "
                          "{f0} (total capped at 1.0), at most {adds} add(s) per trade; the add price "
                          "becomes that bar's close."),
           trade_keys=("n_adds", "last_add"))
def _pyr():
    return {"size": "{f0}",
            "on_entry": """
state["n_adds"] = 0
state["last_add"] = price
""",
            "hold": """
if state["n_adds"] < {adds} and side * (price - state["last_add"]) >= {step} * state["ref_atr"]:
    state["n_adds"] = state["n_adds"] + 1
    state["last_add"] = price
    target = side * min(1.0, abs(pos) + {f0})
"""}
