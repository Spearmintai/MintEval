"""FRL Fig 1: where LLM-written code enters the trading pipeline, and where ES sits relative to the
implementation-shortfall components of Perold (1988). Greyscale; pure matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.sans-serif": ["DejaVu Sans"], "font.size": 8, "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(figsize=(6.6, 6.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="0.9", ls="-", fs=8, ha="center", weight="normal", family=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec="black", lw=0.9, ls=ls))
    tx = x if ha == "center" else x - w / 2 + 2
    ax.text(tx, y, text, ha=ha, va="center", fontsize=fs, weight=weight, linespacing=1.4, family=family)


def arrow(x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                 color="black", ls=ls, shrinkA=0, shrinkB=0))


# intent
box(50, 93, 82, 8, "Trader's intent\n\"long on breakout, stop to breakeven after 1 ATR, trail from the high ...\"",
    fc="white", fs=7.2)
# two paths
arrow(34, 88.6, 26, 78.5); arrow(66, 88.6, 74, 78.5, ls="--")
ax.text(28, 84.5, "conventional path", ha="right", fontsize=7.5, style="italic")
ax.text(72, 84.5, "LLM path (this paper)", ha="left", fontsize=7.5, style="italic")
box(26, 71, 32, 11, "Reference program\nbehaves as intended\n(known bar by bar)")
box(74, 69, 32, 15, "LLM-written program\n✓ compiles\n✓ backtests\n✓ places trades\n✓ passes code review",
    fc="white", ls="--", family="sans-serif")
# common engine
arrow(26, 65.1, 40, 55.5); arrow(74, 61.1, 60, 55.5, ls="--")
box(50, 51, 62, 6.5, "same data  ·  same frictions  ·  same execution engine", fc="0.8")
arrow(42, 47.4, 30, 39.5); arrow(58, 47.4, 70, 39.5, ls="--")
box(28, 36, 22, 5.5, r"$R^{\mathrm{ref}}$  (intended)", fc="white")
box(72, 36, 22, 5.5, r"$R^{\mathrm{llm}}$  (executed)", fc="white", ls="--")
# ES bracket
ax.plot([28, 28, 72, 72], [32.8, 30, 30, 32.8], color="black", lw=0.9)
ax.text(50, 27, r"$ES = R^{\mathrm{ref}} - R^{\mathrm{llm}}$:  silent implementation shortfall",
        ha="center", va="top", fontsize=8.5, weight="bold")

# decomposition band
yb = 11
ax.plot([4, 96], [yb, yb], color="black", lw=0.9)
ax.add_patch(FancyArrowPatch((93, yb), (97.5, yb), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="black"))
for x, lab in ((8, "intended\n(paper) strategy\nreturn"), (50, "executed\nstrategy\nreturn"), (92, "realised\nportfolio\nreturn")):
    ax.plot(x, yb, "o", color="black", ms=5, mfc="white", mew=1.0, zorder=3)
    ax.text(x, yb - 2.5, lab, ha="center", va="top", fontsize=7)


def brace(x0, x1, y, text, fc):
    ax.add_patch(FancyBboxPatch((x0 + 1.5, y), x1 - x0 - 3, 4.5, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc=fc, ec="black", lw=0.7))
    ax.text((x0 + x1) / 2, y + 2.25, text, ha="center", va="center", fontsize=7.2, linespacing=1.3)


brace(8, 50, yb + 2, "ES (this paper): intent ≠ program", fc="white")
brace(50, 92, yb + 2, "Perold (1988): impact, timing, fees", fc="0.85")
ax.text(4, yb + 9.8, "Decomposition of the gap between intended and realised return", fontsize=7.5,
        style="italic", va="bottom")
ax.plot([3, 97], [23.5, 23.5], color="0.6", lw=0.5, ls=":")
fig.savefig("paper_frl/figs/fig1_pipeline.pdf", bbox_inches="tight")
fig.savefig("paper_frl/figs/fig1_pipeline.png", dpi=200, bbox_inches="tight")
