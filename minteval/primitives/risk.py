from . import primitive


@primitive(id="fixed_stop", family="risk", params={"k": [1.0, 1.5, 2.0, 3.0]},
           excludes=("box_shift",),
           spec_template="Protective stop {k} x entry ATR away from the entry price (below for longs, above for shorts).")
def _fs():
    return {"orders": """
stop = tighter_stop(stop, ref - s * {k} * ra, s)
"""}


@primitive(id="breakeven", family="risk", params={"a": [0.5, 1.0, 1.5, 2.0]},
           trade_keys=("be",),
           spec_template=("Breakeven: once a bar closes at least {a} x entry ATR in profit, move the "
                          "stop to the entry price for the rest of the trade (it never moves back)."))
def _be():
    return {"hold": """
if not state.get("be", False) and side * (price - state["ref"]) >= {a} * state["ref_atr"]:
    state["be"] = True
""", "orders": """
if state.get("be", False):
    stop = tighter_stop(stop, ref, s)
"""}


@primitive(id="trailing_hwm", family="risk",
           params={"b": [1.5, 2.0, 3.0, 4.0], "after_be": [False, True]},
           trade_keys=("ext",), requires_any=("breakeven",),
           derive=lambda v: {
               "gate_open": 'if state.get("be", False):\n    ' if v["after_be"] else "",
               "after_be_txt": (" The trailing stop only becomes active once the breakeven rule "
                                "has triggered." if v["after_be"] else "")},
           spec_template=("Trailing stop: track the best price of the trade (highest high for longs, "
                          "lowest low for shorts, starting from the entry price and updated with each "
                          "bar's high/low while the position is held); the stop sits {b} x entry ATR "
                          "behind that extreme.{after_be_txt}"))
def _tr():
    return {"on_entry": """
state["ext"] = price
""", "hold": """
if side > 0 and high[-1] > state["ext"]:
    state["ext"] = high[-1]
elif side < 0 and low[-1] < state["ext"]:
    state["ext"] = low[-1]
""", "orders": """
{gate_open}stop = tighter_stop(stop, state["ext"] - s * {b} * ra, s)
"""}


@primitive(id="time_stop", family="risk", params={"n": [16, 32, 64, 128, 256]},
           trade_keys=("bar0",),
           spec_template=("Time stop: if the trade is still open {n} bars after the signal bar, exit at "
                          "the next open (decide on the bar where bars-since-signal reaches {n})."))
def _ts():
    return {"on_entry": """
state["bar0"] = t
""", "exit": """
if t - state["bar0"] >= {n}:
    target = 0.0
"""}


@primitive(id="take_profit", family="risk", params={"c": [2.0, 3.0, 4.0, 6.0]},
           excludes=("box_shift",),
           spec_template="Take-profit order {c} x entry ATR beyond the entry price.")
def _tp():
    return {"orders": """
take = nearer_take(take, ref + s * {c} * ra, s)
"""}


@primitive(id="partial_take", family="risk", params={"c": [1.5, 2.0, 3.0]},
           trade_keys=("half",), excludes=("pyramid",),
           spec_template=("Scale out: the first time a bar closes at least {c} x entry ATR in profit, "
                          "cut the position to half its current size (once per trade); the rest stays on."))
def _pt():
    return {"hold": """
if not state.get("half", False) and side * (price - state["ref"]) >= {c} * state["ref_atr"]:
    state["half"] = True
    target = pos * 0.5
"""}


@primitive(id="box_shift", family="risk",
           params={"lo": [1.0, 2.0], "hi": [2.0, 3.0], "step": [1.0, 2.0]},
           trade_keys=("shift",), excludes=("fixed_stop", "take_profit"),
           spec_template=("Moving box: the trade lives in a box anchored at the entry price, with the "
                          "stop {lo} x entry ATR behind the anchor and the target {hi} x entry ATR beyond "
                          "it. Every time a bar closes at least another {step} x entry ATR in profit "
                          "beyond the current anchor, shift the whole box (anchor, stop and target) "
                          "{step} x entry ATR in the trade's favour (repeat if the close cleared several "
                          "steps). The box never shifts back."))
def _bx():
    return {"on_entry": """
state["shift"] = 0
""", "hold": """
while side * (price - state["ref"]) >= (state["shift"] + 1) * {step} * state["ref_atr"]:
    state["shift"] = state["shift"] + 1
""", "orders": """
anchor = ref + s * state["shift"] * {step} * ra
stop = tighter_stop(stop, anchor - s * {lo} * ra, s)
take = nearer_take(take, anchor + s * {hi} * ra, s)
"""}


@primitive(id="cooldown", family="risk", params={"n": [4, 16, 48, 96]},
           spec_template=("Cooldown: after any trade ends, take no new entry for {n} bars, counted from "
                          "the bar on which you first see yourself flat (entries allowed again from "
                          "that bar + {n})."))
def _cd():
    return {"on_exit": """
state["cool_until"] = t + {n}
""", "gate": """
if "cool_until" in state:
    if t < state["cool_until"]:
        ok = False
    else:
        del state["cool_until"]
"""}


@primitive(id="daily_cap", family="risk", params={"k": [1, 2, 3]},
           derive=lambda v: {"ies": "y" if v["k"] == 1 else "ies"},
           spec_template=("Daily cap: at most {k} new entr{ies} per UTC calendar day (by the signal "
                          "bar's open time); further signals that day are ignored."))
def _dc():
    return {"gate": """
day = int(hist.time[-1] // 86400000)
if state.get("day", -1) != day:
    state["day"] = day
    state["n_today"] = 0
if state["n_today"] >= {k}:
    ok = False
""", "on_entry": """
state["n_today"] = state["n_today"] + 1
"""}
