# minteval_multiasset.csv
Existing reference programs (800) and generated programs (open setting: 4 models x 800; frontier subset:
3 models x 200) re-run unchanged on four instruments. One row per (asset, model, strategy_id).
- asset: BTCUSDT_2022_2023 (Binance), ETHUSDT_2018_2023 (Binance), SPY_CFD_2018_2023 and NDX_CFD_2018_2023
  (Dukascopy CFD proxies for SPY / Nasdaq-100, 15m regular-session bars, US exchange holidays removed)
- pass_filter: reference passes the benchmark filter ON THIS ASSET (>= 20 round trips, not flat, no early bust);
  analyses should use pass_filter == True
- tau_max, tau_p90, n_registers, R_bench, sharpe_bench, maxdd_bench, n_trades_bench: re-measured on this asset
- realized_vol: annualised vol over bars the reference holds a position; vix_mean_held / frac_held_days_high_vix:
  CBOE VIX over the reference's holding days (high = VIX > 25)
- fee_bp / slippage_bp: frictions used (crypto 5 + 1 bp; equity 0.1 + 0.5 bp); timeout_s: sandbox budget
  (10 s per 70k bars); subset: full800 or frontier200
- metrics: as in results/minteval_v0.csv (compile_fail, action_match, trade_f1, abs_es, ...)
Equity handling: positions may be held overnight; a stop gapped through overnight fills at the next open.
Raw/derived equity price files are not redistributed; rebuild with minteval/dukascopy.py + scripts/finish_equities.sh.
