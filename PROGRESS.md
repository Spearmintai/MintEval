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

## Step 4 — back-translation (`minteval/translate.py`, `results/tasks/prompts.jsonl`)
- Jargon dictionary `data/jargon.txt`: 83 expressions, **all AI-drafted and marked PENDING human
  review** (rule 6). Nothing scraped.
- Translator: `google/gemini-3.5-flash-lite` (T=0.7, seeded), 40-120 words, >=3 slang hits, every
  parameter value present (with accepted renderings: 16 bars = 4 hours, 0.5 = 50%/half, ...), no code-like
  tokens; up to 6 attempts with the failed checks fed back.
- Pilot iterations (6 tasks, all outputs kept in the LLM cache):
  1. v1 prompt: valid 4/4 but every message opened with "Yo"; boundary looseness ("under 3 ATR" for <=).
  2. v2: 8 rotating voices + "preserve boundaries exactly" -> text became spec-like ("strictly above"
     everywhere), defeating the realism goal (review point 4).
  3. v3: boundary words made a *desk convention* (above/below/breaks/crosses = strict; at least/at most/
     within = inclusive), stated in both the translator prompt and the model-facing interface doc; "never
     write 'strictly'". Natural text, but 2/6 had fidelity errors ("2 steps of 2 ATR", "multiplier 1").
  4. -> enabled the round-trip reader as a fidelity gate (spec says it may be off in v0; turned on because
     of (3)). On a mismatch the translator gets the slot-level diff and rewrites.
- Round-trip reader bugs/variants (each logged):
  a. deepseek-v4-flash, max_tokens 3000: most calls truncated in reasoning (finish=length) and were
     scored as "reader failed" -> spurious rewrites. **Bug**: fixed by (i) not gating on reader failures
     (recovery = NaN, passes ungated, counted), (ii) raising max_tokens to 16000.
  b. flash @16000: 3/6 still truncated. Variant sweep on the two failing tasks (no cache):
     flash effort=low 7-13k tok, 1.0/1.0; flash no-reasoning 3 s but 1.0/0.75; flash reasoning cap 2k
     (ignored) 1.0/1.0; **deepseek-v4-pro effort=low 3-5k tok, 1.0/1.0 (chosen)**; qwen3.7-flash 3k tok
     1.0/1.0 (rejected: Qwen is a subject family -> gate would favour it); kimi-k2.5 8-10k tok 1.0/1.0.
- Budget: account had $0.07 left at the start of this step; the user topped it up.

## Smoke tests (6 pilot tasks; `results/smoke/`)
- DeepSeek-flash as a throw-away subject: 5/6 compile failures were `finish=length` at max_tokens 8000
  (**harness bug, not model failure**). Fixed: max_tokens 32768 for API models; truncation and API errors
  get their own error types (`Truncated`, `APIError`). Closed setting: SpecMatch 1.0 on 6/6, i.e. the menu
  makes the reading step easy (supports review point 1: open setting is primary).
- Local Qwen2.5-Coder 7B/32B (vLLM, bf16, seed 0, VLLM_BATCH_INVARIANT=1, `--generation-config vllm`,
  max-model-len 32768, so max_tokens 8192 for these two). Low scores were audited by reading the code:
  32B open S0003 LookaheadError is genuine (`htf.close[t // 16]` indexes the 4h series with a 15m index);
  32B closed S0790 ActionMatch 0 with SpecMatch 1 is genuine (`highest(high,32)[-1]` includes the current
  bar, so `close > box_top` can never hold; stop re-anchored to the current close each bar).

## Step 5 — calibration (acceptance test 5), gpt-5.4-mini, open setting, 50 tasks (every 16th)
Attempt log (all runs kept under `results/archive/`):
1. v1: compile_fail 19/50 (Timeout 15, SandboxViolation 4); AM median 0.456, 52% in [0.3,0.95].
   Audit: timeouts were **my bug** — `ind.highest(high[:-1], n)` (natural "prior n bars" idiom) missed the
   indicator cache and recomputed over full history each bar (O(n^2)); est. 8-16 s vs 10 s budget.
   Fix: prefix views of feed buffers are served exactly from the cache (causal => prefix of full series);
   test `test_prefix_slice_indicators_exact`. Timeouts 15 -> 0. Self-match re-run: PASS.
2. The 4 SandboxViolations were stateless nested helper functions. Rule 4's intent (no state outside
   `state`) cannot be broken by a per-call helper when nonlocal/global/mutable defaults are banned, so they
   are now allowed and flagged in a `strict_violation` column (strict reading reportable). The model-facing
   doc still says "no nested functions" (unchanged, to keep prompts stable).
3. After 1-2: compile_fail 0/50, AM median 0.423, 48% in [0.3,0.95], 8% == 1, 38% < 0.3.
   Audit of the 19 low (<0.3) cases: no one-bar timing artefact (shifting +-1 bar never helps). Two
   **instruction ambiguities** (translation sufficiency, not model error) found:
   a. vol sizing "1% equity over 1 ATR" read literally as 0.01/ATR (reference: 0.01*close/ATR) — S0336
      has identical trade timing (96.5% same sign) but AM 0.028.
   b. Donchian/box "breaks the N-bar high" read as a one-time cross; reference is a level condition.
   The round-trip gate could not catch these because the reader sees the block menu (which contains the
   formulas): it verifies identification, not sufficiency. Fix: two desk conventions (model doc +
   translator), a deterministic cue check for vol sizing, full re-translation (v2). Genuine model errors
   also present in the low set (e.g. S0736 compares close to a highest() that includes the current bar).
4. Re-translation v2 (conventions + vol-sizing cue): first pass only 639/800 valid -> **my bug**: the cue
   regex's gap class `[^.;]` excluded '.', so "risk 0.25% ..." never matched. Fixed (`[^;\n]`), regression
   tests in `tests/test_translate.py`, re-run (first attempts replayed from cache): **796/800 valid,
   780 round-trip verified, 20 unverifiable (reader failure), 0 failed**. The 4 invalid ones stay in the
   set with `prompt_valid = 0`. v1 prompts archived in `results/archive/`.
5. Calibration v2 (same 50 tasks, `results/calib/`): compile_fail 4/50 (all audited, genuine: SyntaxError,
   None stored in state, htf(15) on 15m data, numpy TypeError); AM median 0.526, 60.9% in [0.3,0.95],
   6.5% == 1, 26.1% < 0.3. Ambiguity cases fixed (S0336 0.028 -> 0.965, S0704 0.011 -> 0.939); genuine
   model bug S0736 unchanged (0.037). **Test 5 passes; quant_delta stays 0.05** (no delta change made).

## Step 6 — full evaluation (launched)
- Cost per task measured on calibration: gpt-5.4-mini ~$0.0056 (open), claude-haiku ~$0.0082 (open).
  Balance after step 5: ~$44. Order: Qwen 7B/32B (local, free) both settings; gpt-5.4-mini both settings;
  claude-haiku open; claude-haiku closed only if budget remains.
- All 6,400 generations done (4 models x 2 settings x 800). Spend on the OpenRouter key: $34.58 total
  (translation ~$4, gpt-5.4-mini ~$9.8, claude-haiku ~$19.5, pilots/smoke the rest).
- Harness issues found and fixed during step 6 (each audited before attribution to models):
  a. 3 Qwen-32B closed requests failed with ReadTimeout (client timeout 300 s < long batch-invariant
     generations). Fixed (vLLM client timeout 3600 s), re-run: 0 API errors remain.
  b. Audit of compile-failure categories: StateTypeError = storing None/str (explicitly forbidden);
     LookaheadError (Qwen-32B open: 101) = indexing HTF series with the 15m index (`htf.close[t//4]`,
     reads the in-progress candle) — genuine look-ahead; Qwen-7B TypeError = `hist[-1]` interface misuse;
     "cannot delete conditional expression" = invalid Python in the raw model text (not extraction).
  c. Sensitivity (`scripts/sensitivity_state.py`, engine `lenient_state`): allowing None/str in state
     raises CompileOK substantially for Qwen but leaves ActionMatch on compiled runs ~unchanged.
- Acceptance test 4 on the final CSV: two scoring runs, SHA256 identical
  (`results/logs/final_hashes.txt`).

## Step 7 — figures/tables (`scripts/analyze.py` -> `paper/`)
- Table 1 (`table1.tex/csv`), Fig 1 (ActionMatch by tau quintile, 95% bootstrap CI), Fig 2 (closed:
  ActionMatch by SpecMatch level, box plots), Fig 3 (tau_max histogram), regressions with block fixed
  effects and task-clustered SEs (`analysis.json`). Palette validated (dataviz validator; contrast WARN
  on two hues -> distinct marker shapes + legend). Figures were rendered and inspected; legend overlap and
  Fig 2 overplotting fixed.
- Bug fixed: macro-name collision (7b/32b both -> "QwenCoderB") and digits in macro names (TradeF1);
  analyze.py now asserts both.

## Step 8 — paper (`paper/main.tex`, `paper/build.sh`)
- Every result number is a macro from `numbers.tex`, generated from `results/minteval_v0.csv`.
  arXiv variant `[sigconf,nonacm]` and submission variant `[sigconf,anonymous,review]`, 4 pages each.
- Bibliography: 16 entries, each verified (arXiv API / Crossref / SEC site). SR 11-7 dropped (URL not
  verifiable).
- Prose claims audited against data; two corrected (7B failure mix; an unverified "flags reset" claim
  removed; the "HTF wrong candle" silent pattern verified: 10 compiled runs, AM ~ 0).
- Remaining TODO for humans: author list/order, affiliation.

## Real subset (started; see the user's "diagnostic benchmark + real subset" plan)
- Survey/catalog: `data/real/SURVEY.md`, `catalog.csv` (155 candidates, licences with evidence;
  TradingView not scraped — ToS).
- Pilot: 10 permissive-licence ports in `data/real/tasks/R001..R010`, all PENDING_HUMAN_VERIFICATION;
  intents are placeholders pending human rewrite. Pipeline demo only (`results/real_scores.csv`).
