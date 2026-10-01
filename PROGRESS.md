# MintEval v0 — progress log

Every step: what was done, problems hit, parameters changed. Numbers quoted here are
copied from pipeline outputs (files named in each entry).

## Environment
- Local box: 16 CPU, 60 GB RAM, **no GPU**, no LaTeX. Python 3.12 venv (`uv`), numba for EMA/RMA precompute.
- Remote GPU box (2x RTX PRO 6000 Blackwell 96 GB) supplied by the user for local vLLM models.

## Step 1 — engine, tracking, indicators, sandbox (commit "Step 1")
- Data: data.binance.vision BTCUSDT spot 15m, 2022-01-01..2023-12-31, 24 monthly zips, all
  SHA256 checksums OK. 70,075 bars of 70,080 expected; one gap of 5 bars
  (2023-03-24 12:30 -> 14:00 UTC, Binance outage). Logged in `data/prices/*.gaps.json`, not interpolated.
- No-lookahead is *physical*: bar t is copied into NaN-initialised buffers right before the
  strategy call; `hist.*` and every `ind.*` result are views of such buffers. Index >= t+1 raises
  `LookaheadError` (also counted globally, so `except Exception` cannot hide it). Indicators on feed
  series are precomputed causally and revealed bar by bar (test: identical to recomputing on hist[0..t]).
- Sandbox: AST whitelist (imports math/numpy only; no `_` attributes, no globals/nonlocal/nested
  defs/classes/mutable defaults/attribute stores/banned builtins) + subprocess with audit hook
  (file open, sockets, subprocess, ctypes... blocked), rlimits, 10 s in-child timer.
- Problem: the spec's 200 ms per backtest. Measured (70,075 bars, 3.12+numba, best of 3):
  empty strategy 54 ms; 2-EMA + ATR example 298 ms -> 256 ms after optimisations
  (read-only HistArray buffers, scalar reveal, id->tag map). Variants tried: Python 3.13 (335 ms),
  3.14 (316 ms) — no gain. Typical generated references run 300-500 ms. The remaining cost is
  Python per-call overhead of arbitrary strategy code; it does not limit throughput (multiprocess:
  800 references in ~8 s wall). **Target not met; documented.**
- Test bug fixed: cost-test expectation assumed costs come out of notional; engine (documented)
  sizes on the open and deducts costs from cash.

## Step 2 — primitives + complexity
- 21 primitives (spec asked 8; extended per review to break the tau <-> primitive confound and add a
  hard tier): signal {ma_cross, macd_cross, donchian_break, rsi_revert, box_breakout,
  breakout_pullback (stateful)}, filter {no_filter, htf_trend (multi-timeframe)}, sizing {fixed_frac,
  atr_risk, pyramid}, risk {fixed_stop, breakeven, trailing_hwm (+after_be cross-state variant),
  time_stop, take_profit, partial_take, box_shift (moving box), cooldown, daily_cap}; 1-3 risk layers.
- Within-primitive tau variation: time_stop n, cooldown n, breakout_pullback wait, trailing b.
- Desk conventions (go into the interface doc): entry price = close of the signal bar, entry ATR =
  ATR(14) on that bar (frozen), entries only when flat, opposite signal closes the trade.
- tau convention: only *value* reads count (`state[k]`, `state.get(k)` on a live key); membership
  tests and reads of absent keys are logged but never contribute. Reference programs delete per-trade
  keys when a trade ends, so a flat period does not create spurious long spans.
- Engine: added `warmup_bars: 960` (10 days): strategy first called at bar 960, with full history.

## Step 3 — assembly, filtering, stratified selection (`results/tasks/selection_report.json`)
- First pool of 3,000 (spec): 2,885 passed filters, but R_bench was strongly confounded with tau
  (bin medians -0.75 / -0.49 / -0.42 / -0.32 / -0.10; low-tau programs trade ~10x more -> fee drag).
- Changed: pool raised to **15,000** candidates (spec deviation) and selection made 2-D: within each
  tau_max quintile bin, 32 tasks from each of 5 equal-width R_bench bands over [-0.85, 0.15].
  One cell (bin 4, band 0) had 26 < 32 -> 6 tasks spilled from band 1 (logged as alarm).
  Result: 14,393 pass filters; 800 tasks, 160/bin; R_bench median per bin -0.368/-0.371/-0.367/
  -0.336/-0.338. Remaining confound: trades per task still fall with tau (median 712 -> 255).
- Acceptance test 6: per-bin counts [160 x 5]; pool bin counts all > 2,600: no alarm.
- Acceptance test 1 (all 800 references through the sandbox path): ActionMatch = 1, TradeF1 = 1,
  |ES| = 0 for 800/800, 0 compile failures. Acceptance test 4: two scoring runs, identical CSV
  SHA256 (`results/selftest/summary.json`). Pool generation determinism: pytest `test_pool_deterministic`.
- Mutation sanity (one risk parameter shifted one grid step, 8 tasks): ActionMatch 0.24-0.96, all < 1.
