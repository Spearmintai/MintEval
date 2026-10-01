"""Independent second implementation of R007: explicit LEAN-like objects (insight with generated/close
time, HoldingsState with a running high) iterated over the realised position path."""
import numpy as np


def decide(P, pos_held, warmup):
    c = P["close"]
    now_arr = P["time"].astype(np.int64) + 15 * 60000
    period = 31 * 86400000
    n = len(c)
    tgt = np.zeros(n)
    last_gen = None
    insight_active = False
    hold_state = None             # running high of the holding value per unit
    for t in range(warmup, n):
        now = now_arr[t]
        invested = pos_held[t] > 0
        if not invested:
            hold_state = None
        if last_gen is None or now - last_gen >= period:     # ConstantAlphaModel.should_emit_insight
            last_gen = now
            insight_active = True
        desired = 1.0 if insight_active else 0.0
        if not invested:
            tgt[t] = desired
            if desired:
                hold_state = c[t]
            continue
        if hold_state is None:
            hold_state = c[t]
        if hold_state < c[t]:
            hold_state = c[t]
            tgt[t] = pos_held[t]
            continue
        dd = abs((hold_state - c[t]) / hold_state)
        if dd > 0.01:
            insight_active = False
            hold_state = None
            tgt[t] = 0.0
        else:
            tgt[t] = pos_held[t]
    return {"target": tgt}
