"""FRL Fig 2: three silent failures, each a three-layer panel on a shared bar axis.
Top: position held (reference solid/filled, model hatched). Middle: 15m candles with the level that matters
for the failure (stop, box top or take). Bottom: cumulative return rebased at the window start.
Cases were chosen from results/frl/fig2_candidates.csv (scripts/frl_examples.py) after reading the code;
the cause of each divergence is stated in CASES and was verified against the traces (asserts below)."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scripts.frl_examples import traced, tasks, GEN, P

CASES = [
    dict(key=("S0798", "gpt-5.4-mini"), level="stop", tag="(a)", leg=("upper right", "lower left"),
         title="GPT-5.4-mini, S0798: breakeven stop never armed",
         cause="entry price never stored, so the trigger compares the close with itself"),
    dict(key=("S0264", "gpt-5.4-mini"), level="box", tag="(b)", leg=("lower right", "upper left"),
         title="GPT-5.4-mini, S0264: breakout box includes the current bar",
         cause="close can never exceed a high that includes its own bar, so no entry is taken"),
    dict(key=("S0697", "claude-opus-5.5"), level="signal", tag="(c)", leg=("upper center", None),
         title="Claude Opus 5.5, S0697: trend filter also gates the exit",
         cause="opposite RSI signal ignored while the 4h trend disagrees, so the position is held"),
]
PRE, LEN = 40, 300
H, L, O, Cl = P["high"], P["low"], P["open"], P["close"]
T0 = pd.to_datetime(P["open_time"], unit="ms")


def highest(x, n, include_current):
    s = pd.Series(x).rolling(n).max().to_numpy()
    return s if include_current else np.r_[np.nan, s[:-1]]


def candles(ax, a, b):
    w = 0.6
    for t in range(a, b):
        up = Cl[t] >= O[t]
        ax.vlines(t, L[t], H[t], color="0.35", lw=0.4, zorder=1)
        ax.add_patch(Rectangle((t - w / 2, min(O[t], Cl[t])), w, max(abs(Cl[t] - O[t]), 1e-9),
                               facecolor="white" if up else "0.35", edgecolor="0.35", lw=0.35, zorder=2))


def lane(ax, y, pos, a, b, hatch):
    """Draw held position as a bar: height ~ |position|, long above / short below the lane centre."""
    x = np.arange(a, b)
    p = pos[a:b] * 0.05
    ax.fill_between(x, y, y + 0.4 * p, step="post", facecolor="none" if hatch else "black",
                    edgecolor="black", hatch="////" if hatch else None, lw=0.6)
    ax.axhline(y, color="0.7", lw=0.4)


plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "hatch.linewidth": 0.5})
fig = plt.figure(figsize=(7.2, 9.6))
outer = fig.add_gridspec(3, 1, hspace=0.42)
info = []
for i, cs in enumerate(CASES):
    sid, m = cs["key"]
    rr, lr = traced(tasks[sid]["source"])
    rg, lg = traced(GEN[cs["key"]], check=True)
    assert rg.error is None
    t0 = int(np.nonzero(rr.pos_q != rg.pos_q)[0][0])
    # window starts before the first difference in the level the case is about (if any) and before t0
    if cs["level"] == "stop":
        d = np.nonzero(np.isfinite(lr["stop"][:t0]) != np.isfinite(lg["stop"][:t0]))[0]
        a = (int(d[0]) if len(d) else t0) - PRE
        assert np.isnan(lg["stop"]).all()          # (a) the model never returns a stop at all
    else:
        a = t0 - PRE
    b = min(a + LEN, len(Cl))
    assert (rr.pos_q[:t0] == rg.pos_q[:t0]).all()
    if cs["level"] == "box":
        assert (rg.pos_q == 0).all()               # (b) the model never trades
    gs = outer[i].subgridspec(3, 1, height_ratios=[0.8, 2.2, 1.2], hspace=0.08)
    a0, a1, a2 = (fig.add_subplot(gs[j]) for j in range(3))
    for ax in (a1, a2):
        ax.sharex(a0)
    # top: positions
    lane(a0, 1.0, rr.pos_q, a, b, hatch=False)
    lane(a0, 0.0, rg.pos_q, a, b, hatch=True)
    a0.set_yticks([0, 1], ["model", "reference"])
    a0.text(1.0, 1.25, "long above the line, short below", transform=a0.transAxes, ha="right", fontsize=6, color="0.4")
    a0.set_ylim(-0.5, 1.5)
    a0.spines["left"].set_visible(False); a0.tick_params(axis="y", length=0)
    a0.set_title(f"{cs['tag']} {cs['title']}", loc="left", fontsize=8, fontweight="bold")
    # middle: candles + level
    candles(a1, a, b)
    x = np.arange(a, b)
    if cs["level"] == "stop":
        a1.step(x, lr["stop"][a:b], where="post", color="black", lw=1.1, label="stop, reference")
        a1.plot([], [], color="black", ls="--", lw=1.1, label="stop, model (never set)")
    elif cs["level"] == "box":
        a1.plot(x, highest(H, 64, False)[a:b], color="black", lw=1.1, label="box top, reference (prior 64 bars)")
        a1.plot(x, highest(H, 64, True)[a:b], color="black", ls="--", lw=1.1, label="box top, model (incl. current bar)")
    else:
        # the opposite (RSI 25 up-cross) signal on which the reference covers its short at the next open
        assert rr.pos_q[t0 - 1] < 0 and rr.pos_q[t0] == 0 and rg.pos_q[t0] < 0
        a1.plot([t0 - 1], [L[t0 - 1] * 0.996], marker="^", color="black", ms=6, ls="none",
                label="opposite signal (reference covers;\nmodel requires 4h downtrend to agree)")
    a1.set_ylabel("BTCUSDT")
    a1.legend(frameon=False, fontsize=6.5, loc=cs["leg"][0])
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    # bottom: cumulative return rebased at a
    for r, ls, lab in ((rr, "-", "reference"), (rg, "--", "model")):
        a2.plot(x, (r.equity[a:b] / r.equity[a] - 1) * 100, color="black", ls=ls, lw=1.0, label=lab)
    a2.set_ylabel("return (%)")
    if cs["leg"][1]:
        a2.legend(frameon=False, fontsize=6.5, loc=cs["leg"][1], ncol=2)
    gap = ((rr.equity[b - 1] / rr.equity[a]) - (rg.equity[b - 1] / rg.equity[a])) * 1e4
    for ax in (a0, a1, a2):
        ax.axvline(t0, color="black", ls=":", lw=0.9)
    a1.annotate("first divergence", (t0, a1.get_ylim()[1]), xytext=(3, -8), textcoords="offset points",
                fontsize=6.5, va="top")
    for ax in (a0, a1):
        plt.setp(ax.get_xticklabels(), visible=False)
    ticks = np.linspace(a, b - 1, 5).astype(int)
    a2.set_xticks(ticks, [T0[t].strftime("%Y-%m-%d\n%H:%M") for t in ticks])
    a2.set_xlim(a - 1, b)
    info.append(dict(case=cs["tag"], sid=sid, model=m, t0=t0, date=str(T0[t0]), window=(a, b),
                     gap_bp=round(float(gap)), cause=cs["cause"]))
fig.savefig("paper_frl/figs/fig2_examples.pdf", bbox_inches="tight")
fig.savefig("paper_frl/figs/fig2_examples.png", dpi=170, bbox_inches="tight")
json.dump(info, open("paper_frl/fig2_cases.json", "w"), indent=1)
print(json.dumps(info, indent=1))
