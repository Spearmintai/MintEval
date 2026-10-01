"""Independent second implementation of R009 (OCO regression algorithm): explicit order-ticket list
like LEAN (status per ticket), driven by the realised position path."""
import numpy as np


def decide(P, pos_held, warmup):
    c = P["close"]
    n = len(c)
    tgt = np.zeros(n)
    stop = np.full(n, np.nan)
    take = np.full(n, np.nan)
    tickets = None            # list of [kind, price, status]
    for t in range(warmup, n):
        if tickets is not None and pos_held[t] == 0:
            for tk in tickets:
                tk[2] = "closed"
        if tickets is not None:
            if any(tk[2] != "submitted" for tk in tickets):
                tickets = None
                continue          # target 0, no levels
            tgt[t] = pos_held[t]
            take[t] = tickets[0][1]
            stop[t] = tickets[1][1]
            continue
        tgt[t] = 1.0
        tickets = [["take", c[t] * 1.003, "submitted"], ["stop", c[t] * 0.997, "submitted"]]
        take[t], stop[t] = tickets[0][1], tickets[1][1]
    return {"target": tgt, "stop": stop, "take": take}
