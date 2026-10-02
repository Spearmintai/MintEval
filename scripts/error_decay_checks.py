"""Robustness for the diversification result: horizon aggregation (weekly/monthly), keeping zero-error desks,
market beta and worst-10%-day concentration of the equal-weight portfolio error."""
import sys, json, numpy as np, pandas as pd, yaml
sys.path.insert(0, ".")
from minteval.data import load_prices
asset = sys.argv[1]
cfg = yaml.safe_load(open("configs/base.yaml")); A = cfg["multiasset"]["assets"][asset]
Z = np.load(f"results/decay/{asset}_daily.npz", allow_pickle=True)
dates = pd.to_datetime(Z["dates"])
P = load_prices(A["price_file"]); t = pd.to_datetime(P["open_time"], unit="ms", utc=True)
if A["kind"] == "equity": t = t.tz_convert("America/New_York")
mkt = pd.Series(P["close"], index=t.strftime("%Y-%m-%d")).groupby(level=0).last().pct_change()
mkt = mkt.reindex(Z["dates"]).values
by = {}
for k in Z.files:
    if k == "dates": continue
    m, sid = k.split("|"); by.setdefault(m, []).append(Z[k])
out = {}
def rho_of(E):
    v = E.var(0, ddof=1); E = E[:, v > 0]; C = np.corrcoef(E, rowvar=False); iu = np.triu_indices(E.shape[1], 1); return float(np.nanmean(C[iu]))
for m, cols in by.items():
    E = np.column_stack(cols); n = E.shape[1]
    df = pd.DataFrame(E, index=dates)
    W = df.resample("W").sum().values; Mo = df.resample("ME").sum().values
    # portfolio incl. zero-error desks
    vbar = E.var(0, ddof=1).mean(); port = E.mean(1)
    ratio_all = float(port.var(ddof=1) / vbar)
    # market exposure of the portfolio error and of the average desk
    ok = np.isfinite(mkt)
    beta = float(np.polyfit(mkt[ok], port[ok], 1)[0])
    corr_pm = float(np.corrcoef(mkt[ok], port[ok])[0, 1])
    worst = mkt <= np.nanpercentile(mkt, 10)
    share_worst = float(np.sum(np.abs(port[worst])) / np.sum(np.abs(port[ok])))   # base rate ~0.10
    mean_worst = float(port[worst].mean() * 1e4); mean_other = float(port[ok & ~worst].mean() * 1e4)
    down = ok & (mkt < 0); up = ok & (mkt > 0)
    b_down = float(np.polyfit(mkt[down], port[down], 1)[0]); b_up = float(np.polyfit(mkt[up], port[up], 1)[0])
    out[m] = {"n_desks_incl_zero": n, "rho_daily": rho_of(E), "rho_weekly": rho_of(W), "rho_monthly": rho_of(Mo),
              "port_var_ratio_all_incl_zero": ratio_all, "indep_benchmark": 1.0 / n,
              "beta_port_err_to_mkt": beta, "corr_port_err_mkt": corr_pm, "beta_down": b_down, "beta_up": b_up,
              "abs_err_share_on_worst10pct_days": share_worst, "port_err_bp_worst10": mean_worst, "port_err_bp_other": mean_other}
    print(f"{m:18s} n={n:4d} rho d/w/m {out[m]['rho_daily']:+.3f}/{out[m]['rho_weekly']:+.3f}/{out[m]['rho_monthly']:+.3f} "
          f"varratio(all incl 0) {ratio_all:.4f} vs 1/n {1/n:.4f} | beta {beta:+.4f} (down {b_down:+.4f}, up {b_up:+.4f}) corr {corr_pm:+.3f} "
          f"| |err| share worst10% days {share_worst:.3f} | mean err bp worst {mean_worst:+.2f} other {mean_other:+.2f}")
R = json.load(open("results/decay/decay_checks.json")) if __import__("os").path.exists("results/decay/decay_checks.json") else {}
R[asset] = out; json.dump(R, open("results/decay/decay_checks.json", "w"), indent=1)
