"""Per-quintile 95% bootstrap CIs on the 200-task subset + slope of ActionMatch on log10 tau with CI,
to separate 'flat' from 'too noisy to tell'. Writes paper/figs/fig4_tau_frontier.{pdf,png} + json."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fr = pd.read_csv("results/minteval_v0_frontier.csv"); full = pd.read_csv("results/minteval_v0.csv")
D = pd.concat([fr, full[(full.setting == "open") & full.strategy_id.isin(set(fr.strategy_id))]])
D = D[D.compile_fail == 0].copy(); D["ltau"] = np.log10(D.tau_max)
rng = np.random.default_rng(0); out = {}
def boot(x, n=5000):
    b = rng.choice(x, (n, len(x))).mean(1); return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
models = [m for m in ["claude-opus-5.5", "qwen3.8-max", "deepseek-v4-pro", "gpt-5.4-mini", "claude-haiku"] if m in set(D.model)]
for m in models:
    g = D[D.model == m]
    bins = [boot(g[g.tau_bin == b].action_match.values) for b in range(5)]
    r = smf.ols("action_match ~ ltau", g).fit(cov_type="HC1")
    lo, hi = r.conf_int().loc["ltau"]
    # also exact-match share by bin (Opus' median is 1.0, so the mean hides the shape)
    ex = [float((g[g.tau_bin == b].action_match == 1).mean()) for b in range(5)]
    out[m] = {"bins": bins, "exact_by_bin": ex, "slope": float(r.params["ltau"]), "slope_ci": [float(lo), float(hi)],
              "slope_p": float(r.pvalues["ltau"]), "n": len(g)}
    print(f"{m:16s} n={len(g)} slope/decade {r.params['ltau']:+.3f} 95%CI [{lo:+.3f},{hi:+.3f}] p={r.pvalues['ltau']:.2g}")
    print("   bins:", [f"{b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}]" for b in bins]); print("   exact:", [round(e, 2) for e in ex])
json.dump(out, open("results/tau_ci_frontier.json", "w"), indent=1)
COL = {"claude-opus-5.5": "#4a3aa7", "qwen3.8-max": "#1baf7a", "deepseek-v4-pro": "#e87ba4", "gpt-5.4-mini": "#2a78d6", "claude-haiku": "#eb6834"}
MK = {"claude-opus-5.5": "D", "qwen3.8-max": "^", "deepseek-v4-pro": "v", "gpt-5.4-mini": "o", "claude-haiku": "s"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "font.family": "serif", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for i, m in enumerate(models):
    b = np.array(out[m]["bins"]); x = np.arange(5) + (i - (len(models) - 1) / 2) * 0.1
    ax.errorbar(x, b[:, 0], yerr=[b[:, 0] - b[:, 1], b[:, 2] - b[:, 0]], color=COL[m], marker=MK[m], ms=4.5, capsize=2,
                lw=1.5, label=m, markeredgecolor="white", markeredgewidth=0.7)
ax.set_xticks(range(5)); ax.set_xticklabels([str(i + 1) for i in range(5)]); ax.set_ylim(0, 1)
ax.set_xlabel(r"$\tau_{\max}$ quintile (40 tasks each)"); ax.set_ylabel("ActionMatch (compiled runs)")
fig.legend(frameon=False, fontsize=6.5, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig("paper/figs/fig4_tau_frontier.pdf"); fig.savefig("paper/figs/fig4_tau_frontier.png", dpi=200)
