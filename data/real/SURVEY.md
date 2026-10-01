# MintEval "Real" subset — source survey and candidate catalog

Status: SURVEY ONLY (no ports yet). Date: 2026-10-01.
Companion file: `catalog.csv` (one row per candidate; columns
`source,id,url_or_path,license,timeframe,summary,intent_text_available,state_features,portability,notes`).
Raw clones live in `data/real/sources/` (git-ignored via `sources/.gitignore` = `*` / `!.gitignore`).

## 1. Portability scale used in the catalog

Target engine (see `minteval/prompts.py::INTERFACE_DOC`): single symbol BTCUSDT, 15m bars, decision at bar
close, fill at next open, target in [-1,1] on a 0.05 grid, one stop + one take level per bar (whole position),
`hist.htf(m)` for completed same-symbol higher-TF candles, flat scalar `state`, 10 s budget for ~70k bars.

* **direct** — every rule can be expressed exactly at bar granularity. Indicators the engine lacks
  (Bollinger, ADX/DI, Stoch, CCI, MFI, PSAR, Supertrend, TEMA, OBV, Heikin-Ashi, KAMA ...) must be
  computed by hand, incrementally in `state` where they are recursive (ADX, PSAR, Supertrend, TEMA, HA),
  or with numpy over a short trailing window (BB, CCI, MFI, CMF). This is extra porting work, not an
  approximation. Universal engine differences that apply to every source are NOT counted as approximation:
  (a) fills at next open instead of the source's own convention, (b) trailing/ROI/stop evaluated with the
  previous bars' information (freqtrade backtesting updates a trailing stop with the current candle's high
  before checking its low), (c) the original timeframe (often 5m/1h) is replaced by 15m or by `hist.htf(..)`.
  The ported intent text must restate the rule in our conventions.
* **needs-approximation** — something cannot be reproduced exactly: partial exits by order (take half),
  multiple simultaneous resting orders (grids, TP ladders), limit entries away from price, other symbols
  (informative pairs other than the traded one), intrabar trailing (exchange products), resampling with
  time-interpolation, TA-Lib candle patterns / Hilbert transforms, very long recursive indicators that risk the
  time budget, or code/doc contradictions that force a choice.
* **not-portable** — no meaningful single-asset price rule (skeletons, execution algos/TWAP, cross-sectional
  universes, options, lookahead-bias teaching examples, hyperopt artifacts with absolute altcoin price
  thresholds or empty intervals).

## 2. Sources, licenses (with evidence), counts

| source | repo / page | commit / date read | license (evidence) | rows | direct | needs-approx | not-portable |
|---|---|---|---|---|---|---|---|
| freqtrade | github.com/freqtrade/freqtrade-strategies | f3340ce1 (2026-09-08) | GPL-3.0 (LICENSE file) | 68 | 39 | 17 | 12 |
| jesse | github.com/jesse-ai/example-strategies | 7c91e0a3 (2024-03-21) | MIT (LICENSE file; upstream gabrielweich/jesse-strategies MIT per GitHub API) | 11 | 6 | 5 | 0 |
| jesse-gpt | github.com/jesse-ai/gpt-instructions (instructions.md examples) | 64873bfd (2025-04-16) | MIT (LICENSE file) | 7 | 2 | 5 | 0 |
| ysdede | github.com/ysdede/jesse_strategies (listed in jesse-ai/awesome-jesse) | ade9f4ba (2022-06-06) | CC0-1.0 (LICENSE file + GitHub API) | 8 | 1 | 7 | 0 |
| lean | github.com/QuantConnect/Lean Algorithm.Python | 41c6e603 (2026-09-30) | Apache-2.0 (LICENSE + per-file headers) | 43 | 21 | 20 | 2 |
| binance | binance.com support FAQ, developers.binance.com, binance-spot-api-docs (GitHub) | read 2026-10-01 | (c) Binance; spec paraphrase + short quotes only | 9 | 2 | 7 | 0 |
| okx | okx.com/docs-v5, okx.com/help | read 2026-10-01 | (c) OKX; spec paraphrase + short quotes only | 9 | 1 | 7 | 1 |
| **total** | | | | **155** | **72** | **68** | **15** |

Intent text present (`intent_text_available` starts with `y`): freqtrade 27/68, jesse 9/11, jesse-gpt 2/7,
ysdede 4/8, lean 43/43, binance 9/9, okx 9/9. Rows that are portable (direct or needs-approximation) AND have
real intent text: **96** (lean 41, freqtrade 23, jesse 9, binance 9, okx 8, ysdede 4, jesse-gpt 2);
direct + intent: 48. Reaching 150-300 tasks will need parameter/direction variants of these (e.g. the
long-only vs long/short pairs, Bandtastic's toggleable guards, grid N/mode variants) and/or a manual
TradingView batch (section 2.5).

### 2.1 freqtrade-strategies (GPL-3.0)
* Repo https://github.com/freqtrade/freqtrade-strategies, shallow clone at
  `f3340ce11f5bdf62f598522e64d1f5638eaa13f5` (2026-09-08).
* License evidence: `LICENSE` = "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007". README: "Strategies
  from this repo are free to use, though they are provided as-is and without any warranty."
  One file (`SwingHighToSky.py`) additionally carries a header `license = "MIT"`, `copyright = "Free For Use"`.
* 68 strategy files enumerated (all catalogued) + 1 hyperopt file (`user_data/hyperopts/GodStraHo.py`,
  excluded: it is a hyperopt space definition, not a strategy).
  Sub-folders: root (30), `berlinguyinca/` (30, mostly ports of Mynt / Slack strategies by Gert Wohlgemuth,
  2018), `futures/` (7, long/short), `lookahead_bias/` (4, intentionally broken — readme: "Strategies in this
  folder do have a lookahead bias").
* Intent text: freqtrade strategies rarely have prose. 26/68 have a usable docstring or comments
  (e.g. ASDTSRockwellTrading, MACDStrategy(_crossed), TDSequentialStrategy, Supertrend, hlhb, Quickie, Simple,
  FixedRiskRewardLoss, VolatilitySystem, TrendRiderStrategy). The rest would need intent text written by us
  from the code (not "real" intent).
* Common state features: ROI tables (time-decaying take-profit, keys in minutes since entry -> convert to
  15m bars), static % stoploss, trailing stops (`trailing_stop_positive` + offset, `trailing_only_offset_is_reached`),
  `exit_profit_only`, custom_stoploss (PSAR trail, ATR R:R with breakeven), position adjustment (pyramiding),
  long/short (`can_short`).
* Engine-feature gaps found: informative pairs of OTHER symbols (InformativeSample, multi_tf, TrendRider's
  BTC/USDT feed — harmless when the traded pair is BTCUSDT), resample-with-interpolation helpers
  (CCIStrategy, ReinforcedQuickie), TA-Lib candle patterns (Strategy002, PatternRecognition), sliced execution
  (TWAPStrategy, AlmgrenChrissStrategy), `ta` library mega-features (GodStra*, Zeus). No order-book use anywhere.
* Many hyperopt artifacts are degenerate on BTC: price-vs-volume crosses (Diamond), `close > close[-2]**3.85`
  (PowerTower), absolute altcoin price thresholds (GodStra), empty exit intervals (mabStra, UniversalMACD,
  HourBasedStrategy exit window).

### 2.2 Jesse
* `jesse-ai/example-strategies` (MIT, `LICENSE`: "MIT License / Copyright (c) 2020 jesse-ai"), commit
  `7c91e0a3` (2024-03-21). 11 strategies at top level; `example-strategies-master/` is an older duplicate
  snapshot of 8 of them (not catalogued separately). Several files credit `gabrielweich/jesse-strategies`
  (GitHub API: MIT) and `matty5690/example-strategies`. README: "All submitted strategies must be heavily
  documented" — so most have real docstrings (Donchian, IFR2, KDJ, MACD_EMA, RSI2, SimpleBollinger,
  SMACrossover, TradingView_RSI, TurtleRules).
* `jesse-ai/gpt-instructions` (MIT, "Copyright (c) 2025 Jesse"), commit `64873bfd`: the LLM instruction
  prompt contains 7 additional complete single-symbol strategies (TrendFollowingAI, AlligatorAi, GoldenCross,
  TrendSwingTrader, IchimokuCloud, TurtleAI, K1) plus 2 pairs-trading ones (multi-route -> excluded).
  `jesse-ai/llm-strategy-instructions` (MIT) repeats GoldenCross/TrendSwingTrader (not duplicated).
* `jesse-ai/awesome-jesse` (CC0) lists one more strategy collection: `ysdede/jesse_strategies`
  (GitHub API + LICENSE file: CC0-1.0), commit `ade9f4ba` (2022-06-06). ~25 strategy folders, mostly GA/Optuna
  variants of OTT-on-KAMA; 8 distinct ones catalogued (SimplEma, ewoexit2708, Ott2butKAMA1, KAMA1ShortOnly,
  OttBands1min, OB5F_LSv2, EthMaximalist, fractional); the `playground/` and `*-400Ds/*Re*` folders are
  re-optimisations of the same logic and were not catalogued.
* Other jesse-ai repos checked via GitHub API (`/orgs/jesse-ai/repos`): `project-template` contains only an
  empty `ExampleStrategy`; `pairs-trading-strategy` is multi-asset (excluded); the rest are tooling.
* Jesse semantics that matter for ports: market orders fill at the current candle close in backtests (close to
  our "entry price = signal close" convention); `ta.donchian(self.candles)` includes the current candle
  (`candles[:-1]` used in Donchian example excludes it); `utils.anchor_timeframe('15m') == '2h'` (verified in
  jesse/utils.py); `ta.ma(matype=11)` = linear-regression MA (verified in jesse/indicators/ma.py);
  KDJ defaults 9/3/3 SMA (verified in jesse/indicators/kdj.py).

### 2.3 QuantConnect LEAN (Apache-2.0)
* Repo https://github.com/QuantConnect/Lean, sparse clone of `Algorithm.Python` (+ `Algorithm.Framework/Alphas`,
  `Algorithm.Framework/Risk` to read model logic) at `41c6e603` (2026-09-30).
* License evidence: `LICENSE` line 1 "Apache License Version 2.0, January 2004"; every algorithm file header
  "Lean Algorithmic Trading Engine v2.0. Copyright 2014 QuantConnect Corporation. Licensed under the Apache
  License, Version 2.0".
* 469 .py files; 43 candidates catalogued, 426 excluded (options 109, universe/fundamental selection 93,
  pure API regression tests 87, other demos without a single-asset indicator rule 88, futures mechanics 23,
  custom-data plumbing 20, benchmarks 6 — automatic regex buckets, approximate).
* Only 8 crypto files exist and they are thin EMA-cross / buy-and-hold demos. The useful LEAN content is
  classic single-asset rules on equities/FX (MACD trend, EMA cross, Dual Thrust, displaced MA ribbon, RSI
  hysteresis alpha, intraday reversal) and risk/order-type demos (trailing stop risk model, OCO bracket,
  max drawdown/profit per security) that map onto our stop/take mechanics. Intent text: LEAN files almost
  always have a class docstring or header comment describing the algorithm.

### 2.4 Exchange product specifications (Binance, OKX)
Full notes with verbatim quotes, URLs and per-product mapping are in Appendix A (copied from the research
notes). Summary:

| product | venue / URL | documented behaviour (key quote) | params | engine mapping | verdict |
|---|---|---|---|---|---|
| Futures trailing stop | Binance `TRAILING_STOP_MARKET`, https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api ; rule quoted by Binance staff at https://dev.binance.vision/t/conditions-for-trailing-stop-market/12644 | "SELL: the highest price after order placed >= activationPrice, and the latest price <= the highest price * (1 - callbackRate)" | callbackRate (API: 0.1-10 %; FAQ 360042299292 says 0.01-20 %), activationPrice, workingType (CONTRACT/MARK) | state peak since activation; stop = peak*(1-cb) each bar | approx (one-bar lag, activation at close, mark price not reproducible) |
| Spot trailing stop | Binance `trailingDelta` (BIPS), https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/faqs/trailing-stop-faq.md | "Sell orders ... will trigger after a price _decrease_ of the supplied delta, relative to the _highest_ trade price since submission." stopPrice optional = activation | trailingDelta (100 = 1 %), stopPrice | as above; peak initialised at stopPrice on activation; trailing TAKE_PROFIT variant | approx |
| Trailing stop | OKX `move_order_stop`, https://www.okx.com/docs-v5/en/ ; https://www.okx.com/en-us/help/xi-strategy-order-types | "highest price* (1 - trail variance)"; "If not provided, the trailing stop is activated immediately upon order placement." | callbackRatio or callbackSpread, activePx, closeFraction | as above; closeFraction<1 not representable | approx |
| Spot grid (arith/geom) | Binance FAQ 688ff6ff..., d5f441e8..., trailing-up 3d987afd... | "Price Difference (d) = (Grid Upper Limit - Grid Lower Limit) / Number of Grids"; "Price Ratio (r) = (Grid Upper Limit / Grid Lower Limit) ^ (1 / Number of Grids)" | lower, upper, N, mode, trigger price, SL/TP, trailing up, sell-all-base-on-stop | target = (#levels above close)/N rounded to 0.05 (exact steps only for N in {1,2,4,5,10,20}); SL/TP -> stop/take; trailing-up shifts range in state | approx (intrabar round trips and limit fills lost) |
| Spot grid | OKX `grid` algo, docs-v5 + help "whats-the-spot-grid-bot" | levels "(1, 1+r, 1+2r ...)" / "(1, 1*r, 1*r^2 ...)"; TP/SL "sell all base assets at the market price"; RSI start trigger (15m allowed) | minPx, maxPx, gridNum, runType, tp/slTriggerPx, triggerParams | as Binance; RSI(14) 15m start trigger is bar-close native | approx |
| Futures/contract grid long/short/neutral | Binance FAQ f4c453ba..., trailing up/down 7a7bb224...; OKX `contract_grid` | neutral: "short orders when the price is above the reference point, and long orders when the price is below" | N 2-1000, direction, lever, basePos, tp/slRatio | signed target = level index / (N/2); leverage capped at 1 | approx |
| Recurring buy / DCA | Binance Auto-Invest -> Convert Recurring (Auto-Invest "scheduled to discontinue after 31 March 2025"); OKX recurring buy API | fixed amount at fixed intervals (hourly 1/4/8/12, daily, weekly, monthly); "Recurring prices may be different from market prices on Binance Spot." | amt, period, time, (OKX) minPx/maxPx band | deploy a fixed cash pile over K periods, target steps of 0.05 (about 20 buys), schedule snapped to 15m bar | approx |
| DCA (martingale) bot | Binance Spot DCA FAQ 797f6e46...; OKX `spot_dca`/`contract_dca` | TP = "target profit percentage based on the average price"; safety orders at price deviation with step and size multipliers | initOrdAmt, safetyOrdAmt, maxSafetyOrds, pxSteps, pxStepsMult, volMult, tpPct, slPct, cooldown | safety buys at next open after close/low crosses level; TP at avg*(1+tp) exact | approx |
| OCO / STOP_MARKET / TAKE_PROFIT_MARKET / OKX conditional TP/SL | Binance spot rest-api.md, futures New Order; OKX docs-v5 | "activation of one order immediately cancels the other" | stop, take, closePosition | engine stop + take each bar (stop-wins tie rule is conservative) | direct (market-type, whole position) |
| TWAP | OKX `twap`, Binance algo TWAP | execution algo | - | - | excluded (out of scope) |

Gaps (reported by the research pass): Binance spot-grid grid-count range not found (futures: 2-1000);
Binance card "Recurring Buy" not read in detail; developers.binance.com futures page only readable through a
summariser (verbatim rule taken from Binance staff post on the official dev forum).

### 2.5 TradingView (policy only, no scraping)
Policy pages only; no script pages were opened. Details and quotes in Appendix A, Part B.
* **Default license:** Terms of Use section 22 (https://www.tradingview.com/policies/): "You can publish the script under
  any license. If you do not include the license in the comment section of a script, you agree that your
  script is licensed under the Mozilla Public License 2.0." Pine manual (writing/publishing): "All open-source
  scripts on TradingView use the Mozilla Public License 2.0 by default. Authors wanting to use alternative
  licenses can specify them in the source code." So MPL-2.0 is the default, but each script's header must be
  checked (CC BY-NC-SA is common; protected/invite-only code is not visible and invite-only has a
  non-commercial clause, section 23).
* **Reuse norms:** House Rules (support/solutions/43000591638): "Don't copy or use someone else's content without
  asking for permission and crediting them." Script publishing rules (43000590599) require crediting the
  original author and treat TradingView built-ins/docs, published libraries and standard classic indicators
  as "public domain".
* **Automated collection:** the ToS has no literal "scraping" word, but section 3 limits content to "exclusive
  display-only use" and prohibits "any machine-driven processes that do not involve the direct, human-readable
  display" and "any processing of TradingView's content"; robots.txt disallows ClaudeBot/GPTBot/PerplexityBot
  etc. on /script/* and /scripts/*. Treat automated collection as prohibited.
* **Manual protocol needed:** human browser only; open-source scripts only; record license header verbatim
  (MPL-2.0 -> port kept in its own MPL-2.0 file with notices; CC-NC -> flag/exclude for commercial use;
  all-rights-reserved -> exclude); record URL, title, author, publish/update date, Pine version, access date,
  collector; keep original source; paraphrase descriptions; port with explicit handling of Pine semantics
  (default next-open fills match ours; `process_orders_on_close` = same-bar close; `calc_on_every_tick` not
  reproducible; broker emulator O-H-L-C path heuristic vs our stop-wins rule; `strategy.exit` trail in ticks;
  `qty_percent` partial exits and limit/stop entries need approximation; pyramiding -> target steps;
  `request.security` must be no-lookahead).
* **Already-present TradingView-derived ideas:** freqtrade `VolatilitySystem` (based on TV script 3hhs0XbR) and
  jesse `TradingView_RSI` (based on TV script Ru7qOVtp) re-implement TV ideas inside MIT/GPL repos; the original
  Pine code was not inspected, so only the re-implementations' licenses apply.

## 3. Recommended first 30 candidates for porting

Selection criteria, in order: (1) real intent text exists in the source (docstring/README/official spec),
(2) portability `direct` (or a single, well-defined approximation), (3) permissive license first
(MIT/Apache/CC0) — GPL items are included where the intent text is unusually good, see risk R1,
(4) coverage of state features the synthetic subset under-represents (ROI tables, trailing with activation
offset, breakeven, pyramiding, setup counters, martingale sizing, cooldowns, hysteresis, grids),
(5) crypto relevance and BTC-meaningful thresholds.

| # | source | id | license | why |
|---|---|---|---|---|
| 1 | jesse | RSI2 | MIT | Connors RSI(2) mean reversion, long/short with SMA200 regime and SMA5 exit; docstring cites Connors/StockCharts; built-in indicators only. |
| 2 | jesse | TurtleRules | MIT | Original Turtle S1 with ATR unit sizing and 4-level pyramiding (maps to target fractions), 2N stop raised on adds; rich quoted rules. |
| 3 | jesse | Donchian | MIT | Donchian(20) breakout with SMA200 filter, exit on lower band; clean docstring. |
| 4 | jesse | SimpleBollinger | MIT | BB(hl2) upper-band breakout with Ichimoku-cloud filter, exit at middle band; docstring explains rationale. |
| 5 | jesse | TradingView_RSI | MIT | RSI crosses 35 entry, 5 % stop / 10 % take, RSI 75 and 10 cross-down exits; docstring lists rules. |
| 6 | jesse | MACD_EMA | MIT | MACD vs signal with EMA100 filter; docstring-vs-code 'cross' ambiguity is a useful wording test. |
| 7 | jesse-gpt | TurtleAI | MIT | Donchian breakout + 4h SMA200 + ADX/CHOP filters, ATR stop ratcheted every bar, 1-bar cooldown. |
| 8 | jesse-gpt | TrendSwingTrader | MIT | EMA21/50/100 stack + ADX; half take-profit at 3 ATR then breakeven then ATR trail (partial exit approximated). |
| 9 | ysdede | SimplEma | CC0-1.0 | EMA cross with pump/dump filter, fixed % stop/take, martingale-like size after losses (loss-streak sizing state); README states intent. |
| 10 | lean | VIXDualThrustAlpha | Apache-2.0 | Classic Dual Thrust on 30m (htf(30)) with 3 % drawdown stop; long/short; intent comments + references. |
| 11 | lean | MACDTrendAlgorithm | Apache-2.0 | MACD(12,26,9) with normalized ±0.25 % band; canonical LEAN example. |
| 12 | lean | MovingAverageCrossAlgorithm | Apache-2.0 | 15/30 EMA cross with tolerance, long or flat; clear docstring. |
| 13 | lean | RsiAlphaModelFrameworkRegressionAlgorithm | Apache-2.0 | RSI 30/70 trip with 35/65 reset hysteresis state machine (collapse universe to one asset). |
| 14 | lean | TrailingStopRiskFrameworkRegressionAlgorithm | Apache-2.0 | Always-long with 1 % trailing stop from peak (TrailingStopRiskManagementModel) - trailing reference behaviour. |
| 15 | lean | IntradayReversalCurrencyMarketsAlpha | Apache-2.0 | Hourly SMA(5) reversal only inside a time window, flat at window end (time-of-day state; map NY hours to UTC). |
| 16 | lean | DisplacedMovingAverageRibbon | Apache-2.0 | Ribbon of SMA15 displaced 5..30 bars; ordering rule; tests lagged indicator handling. |
| 17 | binance | binance_futures_trailing_stop | docs (spec) | Official activation price + callback-rate trailing rule; directly tests activation flag + peak state. |
| 18 | binance | binance_spot_grid_arith (+ _geom variant) | docs (spec) | Grid -> stepwise target fractions; tests range/levels state, SL/TP, optional trailing-up. |
| 19 | okx | okx_spot_dca_martingale (+ binance_spot_dca_bot) | docs (spec) | Safety orders with step/size multipliers and TP on average price - average-price state, cycle restart. |
| 20 | binance | binance_spot_oco | docs (spec) | Market-type stop + take bracket closing the whole position - exact mapping; good easy control. |
| 21 | freqtrade | ASDTSRockwellTrading | GPL-3.0 | MACD>0 and >signal entry, MACD<signal exit, 4-step ROI table; docstring states the uptrend/downtrend/sell definitions. |
| 22 | freqtrade | MACDStrategy_crossed | GPL-3.0 | MACD cross with CCI filters; docstring lists buy/sell rules verbatim; ROI table. |
| 23 | freqtrade | TDSequentialStrategy | GPL-3.0 | TD Sequential setup counts with perfection check; docstring describes triggers in words; counter state. |
| 24 | freqtrade | Supertrend | GPL-3.0 | Triple Supertrend entry/exit, ROI table, 5 % trailing after +14.4 % offset; module docstring by author. |
| 25 | freqtrade | hlhb | GPL-3.0 | HLHB system (RSI10 50-cross + EMA5/10 cross + ADX), trailing only after offset, ROI table; docstring + babypips reference. |
| 26 | freqtrade | FixedRiskRewardLoss | GPL-3.0 | ATR initial stop, breakeven at 1R, stop locked at 3.5R - breakeven/profit-lock state; docstring explains. |
| 27 | freqtrade | VolatilitySystem | GPL-3.0 | 3h (htf(180)) close-change > 2*ATR breakout, stop-and-reverse, half-size entry then add on repeat signal (pyramiding). |
| 28 | freqtrade | Quickie | GPL-3.0 | ADX/TEMA/BB momentum with ROI table lacking a 0-minute key (no take before 10 min) - ROI timing edge case; docstring intent. |
| 29 | freqtrade | EMASkipPump | GPL-3.0 | Dip-buy at 12-bar low below lower BB with pump filter (volume < 20x mean); symmetric exit; docstring. |
| 30 | freqtrade | CustomStoplossWithPSAR | GPL-3.0 | PSAR-based trailing stop via custom_stoploss (indicator trail, never loosens); pair with its (acknowledged nonsensical) PSAR-turn entry or a stated entry. |

Mix: 20 permissive (MIT/Apache/CC0/exchange spec) + 10 GPL. 24 direct, 6 needs-approximation (TurtleRules,
TrendSwingTrader, RsiAlphaModel, Binance trailing-stop spec, spot grid, DCA bot). Alternates if any fail the
signal-count check: freqtrade Simple, MACDStrategy, Scalp, CMCWinner (native 15m), Bandtastic (native 15m,
ROI + trailing, no prose), FSupertrendStrategy (long/short), Strategy001_custom_exit; jesse SMACrossover,
IchimokuCloud; lean ParameterizedAlgorithm, OneCancelsOtherOrderRegressionAlgorithm, DataConsolidationAlgorithm;
okx_trailing_stop_move_order_stop, binance_futures_grid_long_short_neutral.
Before porting each: (1) quick signal-count on BTCUSDT 15m 2018-2025 (reject < ~20 trades), (2) pin indicator
definitions, (3) write the intent text in our desk conventions, quoting the source rule where licensing allows.

## 4. Risks

* **R1 License — GPL-3.0 (freqtrade).** A faithful Python port of a GPL strategy is plausibly a derivative
  work; distributing the reference programs (and maybe verbatim docstrings as intent text) would carry GPL-3
  obligations. Options: (a) publish the Real subset's freqtrade-derived reference programs under GPL-3 in a
  separate directory with attribution; (b) restrict quoting to short factual rule statements and re-author
  code from the rule (rules/ideas are not copyrightable, code is); (c) prefer MIT/Apache/CC0 sources for the
  first batch. Needs a decision before release. MIT/Apache/CC0 require attribution + license notice only.
* **R2 Exchange docs are copyrighted text.** Use them as specifications: paraphrase behaviour, quote only
  short key sentences with URL; do not redistribute pages.
* **R3 TradingView.** Automated collection is prohibited by the ToS; open-source scripts are
  author-licensed (MPL-2.0 default header) and House Rules restrict reuse; only a manual, per-script
  protocol is viable (section 2.5).
* **R4 "Real intent" is scarce.** Only ~40% of freqtrade strategies carry prose; many docstrings disagree
  with code (Low_BB mentions a trailing stop that is not configured; MACDStrategy docstring CCI -50/100 vs
  hyperopt values -48/687; KDJ `J > (K and D)` Python bug; MultiRSI "must be bearish" comment vs code;
  MACD_EMA docstring "crosses" vs state test). The port must choose and the task text must be explicit;
  record the choice per task.
* **R5 Degenerate hyperopt artifacts.** Several "real" strategies never trade or never exit on BTC (sec 2.1).
  Run a quick signal-count check on our BTCUSDT data before accepting a candidate.
* **R6 Indicator definition drift.** TA-Lib vs qtpylib vs jesse vs LEAN differ in seeding (MACD/EMA SMA-seed,
  Wilder RSI/ATR/ADX warm-up), Bollinger std ddof, typical-price vs close sources, Supertrend ATR flavour,
  Ichimoku displacement. Reference ports must pin one definition and the intent text must name it when it
  matters (e.g. "Bollinger on typical price").
* **R7 Time budget.** Strategies with dozens of recursive indicators (MultiMa TEMA up to 748, BinHV27,
  SmoothOperator, FSupertrend with 6 Supertrends) must keep incremental state; otherwise >10 s.
* **R8 Timeframe change.** Most sources run on 1m/5m/1h/4h. Porting to 15m (or `htf`) changes behaviour;
  ROI tables keyed in minutes do not align with 15m bars (e.g. 20m, 69m). Convention needed: round up to the
  next bar boundary, stated in the task text.
* **R9 Semantics of exits.** freqtrade `exit_profit_only`, `ignore_roi_if_entry_signal`, trailing stops that
  update on the same candle, and jesse partial exits (take half) are not native — each must be specified
  explicitly in the intent text.

## 5. Attempt log

Main survey (this agent):
| # | action | tool | outcome |
|---|---|---|---|
| 1 | read `minteval/prompts.py` INTERFACE_DOC, repo `.gitignore` | bash | OK |
| 2 | create `data/real/sources/.gitignore` (`*`, `!.gitignore`) | bash | OK (repo is a git work tree; clones ignored) |
| 3 | `git clone --depth 1 freqtrade/freqtrade-strategies` | git | OK, commit f3340ce1 |
| 4 | `git clone --depth 1 jesse-ai/example-strategies` | git | OK, commit 7c91e0a3 |
| 5 | grep-based dump of strategy attributes | bash | output too large (33 KB, persisted) -> replaced by an AST dumper (class docstring, ROI/SL/trailing attrs, method bodies without docstrings) |
| 6 | AST dump in 6 batches, read all 67 files | python | OK |
| 7 | coverage check vs file list | python | found TrendRiderStrategy.py missed -> dumped and catalogued; 68/68 covered; GodStraHo.py (hyperopt) excluded |
| 8 | read lookahead_bias/readme.md | bash | OK (confirms the 4 files are intentional lookahead examples) |
| 9 | GitHub API `/orgs/jesse-ai/repos` | curl | OK (24 repos + licenses) |
| 10 | GitHub API `gabrielweich/jesse-strategies` | curl | OK: MIT |
| 11 | clone jesse-ai awesome-jesse, gpt-instructions, llm-strategy-instructions, project-template | git | OK; gpt-instructions contains 7 single-symbol example strategies; project-template only an empty example |
| 12 | GitHub API `ysdede/jesse_strategies` + recursive tree | curl | OK: CC0-1.0, 186 md/init files |
| 13 | `git clone --depth 1 ysdede/jesse_strategies` | git | slow (>120 s, moved to background; finished later, exit 0, commit ade9f4ba) |
| 14 | meanwhile fetch 15 key files via raw.githubusercontent.com | curl | OK 15/15 |
| 15 | verify jesse semantics from jesse-ai/jesse master (ma.py matype 11, kdj.py defaults, donchian.py, utils.anchor_timeframe) | curl raw | OK (matype 11 = linearreg; 15m anchor = 2h) |
| 16 | `sleep 60` to wait for clone | bash | blocked by harness; waited for completion notifications instead |
| 17 | spot-check LEAN VIXDualThrustAlpha (k1=k2=0.63, 20-bar range, 3 % drawdown) | grep | OK, matches sub-agent row |
| 18 | assemble catalog | python | exchange_rows.csv line for okx_spot_dca_martingale had an unquoted comma (11 fields) -> fixed quoting, re-validated 10 fields per row |

LEAN sub-survey (delegated):
| # | action | outcome |
|---|---|---|
| 1 | inspect `sources/` | OK |
| 2 | `git clone --depth 1 --filter=blob:none --sparse QuantConnect/Lean` | OK first try |
| 3 | `git sparse-checkout set Algorithm.Python` | OK, 469 .py |
| 4 | read LICENSE + headers | Apache-2.0 confirmed |
| 5 | grep crypto terms | 25 hits reviewed |
| 6 | indicator/order-type heuristic scan | first regex matched everything (`str(`), refined and rerun OK |
| 7 | `sparse-checkout add Algorithm.Framework/Alphas Algorithm.Framework/Risk` | OK (model logic read) |
| 8 | read every candidate + borderline file | 43 candidates, 426 excluded |
No fallback (API/raw/codeload) was needed.

Exchange + TradingView sub-survey (delegated; 43 attempts, full table in Appendix A):
failures and how they were recovered -
* developers.binance.com futures New Order: curl HTTP 202 empty (WAF); WebFetch summary only -> verbatim rule
  recovered from Binance staff post on dev.binance.vision via Discourse `.json` (WebFetch of the HTML had failed
  with socket closed).
* OKX docs-v5 anchor via WebFetch: page too large -> curl of the 5.2 MB page + grep: OK.
* binance-docs.github.io legacy: redirect stub, not needed.
* Pine publishing doc via curl: compressed body -> WebFetch: OK.
* Binance create-spot-grid FAQ: partial only; supplemented by parameters/trailing-up FAQs and search snippets.
Open gaps: Binance spot-grid count range; Binance card Recurring Buy; Pine default header wording not
re-verified from a TV page.
No TradingView script page was fetched (policy pages + robots.txt only).

## Appendix A. Exchange product specs and TradingView licensing: research notes (verbatim from research pass)

Compiled 2026-10-01. Quotes are verbatim from the cited pages unless marked *(paraphrase via fetch summariser)*. Exchange docs are © the exchanges; we store the spec as a paraphrase and cite the URL.

Target engine (for reference): BTCUSDT 15m bars. `strategy()` runs at each bar close and returns a target position fraction in [-1,1] on a 0.05 grid, filled at the NEXT bar open. It can also return one stop and one take level. Each is active for the next bar only. A stop triggers on low<=stop for a long and fills at the level, or at the open if price gaps through. A triggered order closes the whole position, and the stop wins if both trigger. State is a flat dict.

General mapping rules used below:
- **Exact** only when the exchange product reduces to "one market-type stop/take level, fixed for the whole bar, that closes the entire position", or to a target-fraction change at a bar boundary.
- **Approximate** when the product updates its trigger tick-by-tick inside a bar (trailing), places several resting limit orders that can fill and refill within one bar (grid, DCA safety orders), or sizes in quote/base units that must be quantised to 0.05 of equity.
- **Intrabar ordering** cannot be known from OHLC. Pine's broker emulator has the same problem and uses an O→H→L→C / O→L→H→C heuristic (see Part B).

---

### PART A — Exchange products

#### A1. Trailing stop orders

##### A1a. Binance USDⓈ-M Futures `TRAILING_STOP_MARKET`
- API doc: https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api (New Order). The page is JS/WAF-protected: curl returned HTTP 202 with an empty body. WebFetch worked, but only through a summariser.
- Verbatim trigger rule, as a Binance staff member ("Chai") quoted it from the futures docs on the official dev forum (https://dev.binance.vision/t/conditions-for-trailing-stop-market/12644, fetched through the Discourse `.json` endpoint):
  > "TRAILING_STOP_MARKET:
  > BUY: the lowest price after order placed <= activationPrice, and the latest price >= the lowest price * (1 + callbackRate)
  > SELL: the highest price after order placed >= activationPrice, and the latest price <= the highest price * (1 - callbackRate)
  > For TRAILING_STOP_MARKET, if you got such error code. {"code": -2021, "msg": "Order would immediately trigger."} means that the parameters you send do not meet the following requirements: BUY: activationPrice should be smaller than latest price. SELL: activationPrice should be larger than latest price."
- From the API page (via WebFetch): `callbackRate` "min: 0.1 · max: 10" (percent units, so 0.1% to 10%). `activationPrice` defaults to "the latest price(supporting different `workingType`)". `workingType` is `MARK_PRICE` or `CONTRACT_PRICE`, default `CONTRACT_PRICE`. `closePosition` means "Close all current long position (if `SELL`) or current short position (if `BUY`)" and cannot be combined with `quantity`. `priceProtect` applies to STOP_MARKET/TAKE_PROFIT_MARKET.
- Support FAQ: https://www.binance.com/en/support/faq/what-is-a-trailing-stop-order-360042299292 (via WebFetch):
  > "A trailing stop order helps traders limit their losses and protect their gains when the market swings."
  > Callback rate: "The percentage of movement in the opposite direction that you are willing to tolerate" — **this FAQ gives the range as 0.01% to 20%, which conflicts with the API doc's 0.1–10. Record both values; the product range may have been widened.**
  > Activation Price: "Your desired price level that triggers the trailing stop order. If no activation price is set, the activation price will be the market price by default."
  > Sell trailing stop conditions: "Activation Price ≤ Highest Price" and "Rebound Rate ≥ Callback Rate"; buy: "Activation Price ≥ Lowest Price" and "Rebound Rate ≥ Callback Rate".
  > Example *(paraphrase)*: price 10,000 USDT, callback 5%, activation 10,500 USDT. The price peaks at 11,000, so the trailing trigger becomes 10,450, and "a sell order will be executed at market price to close the position."
- **Parameters:** callbackRate (%), activationPrice (optional), workingType (last/contract price or mark price), side, quantity or closePosition, reduceOnly.
- **Portability: needs-approximation.** Mapping:
  - `state.peak` = max(high) since activation (for a long). `state.active` is set when a completed bar's high reaches activationPrice; with no activation price it is set at entry.
  - Each bar, return `stop = state.peak * (1 - cb)`.
  - Differences from the exchange:
    1. Inside one bar the exchange can raise the peak and then trigger. The engine's stop for bar t+1 is fixed from data up to bar t, so it lags by up to one bar and is looser in a rally.
    2. Activation is detected at bar close, not at the moment of touch.
    3. Trigger price: the exchange uses last or mark price. Engine bars are last-trade OHLC, so the mark-price option is not reproducible.
    4. The exchange fills a market order with slippage. The engine fills at the level or at the open on a gap, which is close enough.
  - Result: exact only when no bar both makes a new peak and hits the trail. For a conservative variant, update the peak with the bar's high only after checking that bar's low against the old stop. The engine already does this by construction.

##### A1b. Binance Spot trailing stop (`trailingDelta`, BIPS)
- Docs: https://developers.binance.com/docs/binance-spot-api-docs/faqs/trailing-stop-faq. Canonical markdown: https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/faqs/trailing-stop-faq.md (curl 200).
  > "Trailing stop is a type of contingent order with a dynamic trigger price influenced by price changes in the market. For the SPOT API, the change required to trigger order entry is specified in the `trailingDelta` parameter, and is defined in BIPS."
  > "Buy orders: _low_ prices are good. Unlimited price _decreases_ are allowed but the order will trigger after a price _increase_ of the supplied delta, relative to the _lowest_ trade price since submission."
  > "Sell orders: _high_ prices are good. Unlimited price _increases_ are allowed but the order will trigger after a price _decrease_ of the supplied delta, relative to the _highest_ trade price since submission."
  > "For example, a `STOP_LOSS` `SELL` order with a `trailingDelta` of 100 is a trailing stop order which will be triggered after a price decrease of 1% from the highest price after placing the order."
  > "Trailing stop orders are supported for contingent orders such as `STOP_LOSS`, `STOP_LOSS_LIMIT`, `TAKE_PROFIT`, and `TAKE_PROFIT_LIMIT`. OCO orders also support trailing stop orders in the contingent leg. In this scenario if the trailing stop condition is triggered, the limit leg of the OCO order will be canceled."
  > "Unlike regular contingent orders, the `stopPrice` parameter is optional for trailing stop orders. If it is provided then the order will only start tracking price changes after the `stopPrice` condition is met. If the `stopPrice` parameter is omitted then the order starts tracking price changes from the next trade."
  > The scenario text also says: "When a trade is equal to, or surpasses, the `stopPrice` the order starts tracking price changes immediately; the first trade that meets this condition sets the 'lowest price'."
- BIPS: 1 = 0.01%, 100 = 1%, 1000 = 10%. The delta must lie inside the symbol's `TRAILING_DELTA` filter (`minTrailingAboveDelta`/`maxTrailingAboveDelta`, `minTrailingBelowDelta`/`maxTrailingBelowDelta`).
- Trigger-type table (verbatim rows): `TAKE_PROFIT` SELL: "market price >= stop price | *decrease* from maximum". `STOP_LOSS` SELL: "market price <= stop price | *decrease* from maximum".
  - Note the twist: a trailing **TAKE_PROFIT** sell activates once price is *above* stopPrice, then trails down from the max. This is a "trailing take-profit".
- Support page for the UI: https://www.binance.com/en/support/faq/how-to-use-spot-trailing-stop-order-339635f6260d43c5aefa4c3c921728ec. Found by search, not fetched.
- **Portability: needs-approximation.** The mapping is the same as A1a, using trailingDelta/10000 as the callback. Activation by stopPrice uses the first trade that crosses, so the peak starts at stopPrice and not at the bar's high. Reproduce this by initialising `state.peak = stopPrice` on the activation bar, then updating with later bars' highs.
  - The `_LIMIT` variants place a limit order after the trigger, and it may not fill. The engine assumes a fill at the stop level or at the open, which is an optimistic approximation for the limit variants.
  - Spot is long-only, so only the SELL-side trailing applies to exits from a long. The BUY-side trailing is a trailing *entry*. The engine can approximate it as: when close >= trough*(1+delta), set target>0 at the next open. This is an entry at the next open rather than at the trigger, so it is a coarser approximation.

##### A1c. OKX trailing stop (`ordType = move_order_stop`)
- API: https://www.okx.com/docs-v5/en/#order-book-trading-algo-trading-post-place-algo-order. The page is about 5 MB. WebFetch summarisation missed the section, so the page was downloaded with curl and grepped.
  > `ordType`: "conditional: One-way stop order / oco: One-cancels-the-other order / chase: chase order, only applicable to FUTURES and SWAP / trigger: Trigger order / move_order_stop: Trailing order / twap: TWAP order / smart_iceberg: Iceberg order"
  > `callbackRatio`: "Callback ratio, e.g. 0.05 represents 5%. Either callbackRatio or callbackSpread is required. Only one can be passed. Only applicable when ordType = move_order_stop"
  > `callbackSpread`: "Callback spread (price distance). Either callbackRatio or callbackSpread is required. Only one can be passed. Only applicable when ordType = move_order_stop"
  > `activePx`: "Activation price. The trailing stop is activated when the market price reaches the activation price. After activation, the system starts calculating the actual trigger price. If not provided, the trailing stop is activated immediately upon order placement. Only applicable when ordType = move_order_stop"
  > Also in the same param table: `closeFraction` — "Fraction of position to be closed ..." (the conditional/trailing order can close a fraction of the position).
- Help centre, "Strategy order types": https://www.okx.com/en-us/help/xi-strategy-order-types (via WebFetch):
  > Trigger price — Sell: "[Var.] highest price - trail variance; [Percentage] highest price* (1 - trail variance)"; Buy: "[Var.] lowest price + trail variance; [Percentage] lowest price* (1 + trail variance)"
  > "When the market price reaches or exceeds the activate price, the trailing order is activated."
  > "The trail variance is a custom percentage, along with the recorded highest/lowest prices, used to calculate the actual trigger price."
- Help centre, "How do I place a trailing stop order?": https://www.okx.com/en-us/help/how-to-use-trailing-stop:
  > "A trailing stop s[ic] a type of take-profit/stop-loss (TP/SL) order that adjusts automatically based on market price movements." "If you enable an Activation price, the trailing stop will only activate once the latest traded price reaches or exceeds the activation price." "If you do not set an activation price, the trailing stop will activate immediately after the order is placed."
- **Portability: needs-approximation.** The mapping is the same as A1a, with `stop = peak*(1-ratio)` or `stop = peak - spread`. A `closeFraction` < 1 (partial exit) is **not portable** via the stop, because a stop closes the whole position in the engine.

#### A2. Grid trading bots

##### A2a. Binance Spot Grid
- "What Is Spot Grid Trading and How Does It Work?": https://www.binance.com/en/support/faq/what-is-spot-grid-trading-and-how-does-it-work-d5f441e8ab544a5b98241e00efb3a4ab (via WebFetch):
  > "Grid trading is a type of strategic tool utilizing advanced orders to automate trading. This trading bot automates the buying and selling on spot trading."
  > "Orders are placed above and below a set price, creating a grid of orders at incrementally increasing and decreasing prices."
  > The page says the bot stops executing orders when price falls below the Lower Price or exceeds the Upper Price. *(Summariser's wording: the strategy is suspended outside the range and resumes when price returns.)* When a buy fills, "a new sell order will be placed at the next grid level above" *(summariser quote)*.
- "Binance Spot Grid Trading Parameters": https://www.binance.com/en/support/faq/binance-spot-grid-trading-parameters-688ff6ff08734848915de76a07b953dd:
  > Arithmetic: "Price Difference (d) = (Grid Upper Limit - Grid Lower Limit) / Number of Grids"
  > Geometric: "Price Ratio (r) = (Grid Upper Limit / Grid Lower Limit) ^ (1 / Number of Grids)"
  > Profit per grid: a "range" of values for arithmetic, "a fixed value" for geometric.
  - Search snippet for the same FAQ family: "In Arithmetic mode, each grid has an equal price difference. In Geometric mode, each grid has an equal price difference ratio."
- Trailing Up: https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-function-in-spot-grid-trading-3d987afd7906495cb4d997eccb8515bf:
  > "If the price increases above the upper limit price and the price difference between the grid levels ($45,000 + $4,000 = $49,000), the bot will adjust the grid upwards." "It will cancel the lowest buy order and place a new buy order at the previous upper limit price ($45,000)." "The Stop Loss price will rise by the same price difference as the change in price range caused by the Trailing Up mechanism."
  - The spot-grid page does not mention Trailing Down. The futures-grid page does (A2c).
- Create-strategy FAQ: https://www.binance.com/en/support/faq/how-to-create-a-spot-grid-trading-strategy-on-binance-95078b6293184bd79b56108092f337c1. It says you can "set the order's trigger price and the stop-loss/take-profit price", and that parameters are set in "Arithmetic" or "Geometric" mode.
  - "Sell All Base Coins on Stop" (from a search snippet of the same FAQ family): when enabled, "the strategy will automatically sell all base coins at market price when the grid is stopped"; it is "enabled by default".
- Spot grid count range: not stated on the pages fetched. The futures grid page gives 2..1000 (USDⓈ-M).
- **Portability: needs-approximation.** Mapping:
  - Precompute levels L_0 < … < L_N, arithmetic or geometric. Base inventory is implied by how many buy levels above the current price have filled.
  - Discretised policy: at each bar close, target = k/N, where k = number of grid levels strictly above close (clipped to [0, N]; 1 below the range, 0 above it). Round to 0.05.
  - Steps are exact only when 1/N is a multiple of 0.05, i.e. N ∈ {1, 2, 4, 5, 10, 20}. Larger N is quantised to 0.05 steps, so N ≥ 20 becomes 20 effective levels.
  - What is lost:
    1. Intrabar round trips: a bar whose range spans several levels would fill a buy and the matching sell in the same bar, and the engine sees only the close.
    2. Fill price: the exchange fills at the level (limit), the engine at the next open.
    3. Inventory: the exchange holds fixed base quantity per grid, while the engine target is a fraction of equity. Rebalancing to a fraction every bar differs slightly from fixed-quantity inventory.
    4. Maker fees vs the engine's fee model.
  - Grid stop-loss and take-profit (sell all base) map directly to the engine's `stop`/`take` fields, because they close everything.
  - Trailing up: shift the levels in `state` when close > upper + d (state has only scalars, so store lower/upper/N/mode).
  - Trigger price (start condition): use a state flag.

##### A2b. OKX Spot Grid
- API: same doc, "Place grid algo order" `POST /api/v5/tradingBot/grid/order-algo` (curl+grep):
  > `algoOrdType`: "grid: Spot grid / contract_grid: Contract grid"; `maxPx` "Upper price of price range"; `minPx` "Lower price of price range"; `gridNum` "Grid quantity"; `runType` "Grid type 1: Arithmetic, 2: Geometric Default is Arithmetic"; `tpTriggerPx` "TP tigger[sic] price Applicable to Spot grid / Contract grid"; `slTriggerPx` "SL tigger price".
  > `triggerParams`: triggerAction "start / stop"; triggerStrategy "instant / price / rsi"; for rsi, `timeframe` "3m, 5m, 15m, 30m, 1H, 4H, 1D", `thold` "integer between 1 to 100", `triggerCond` "cross_up / cross_down / above / below / cross", `timePeriod` "14"; `stopType` "Spot grid 1: Sell base currency 2: Keep base currency; Contract grid 1: Market Close All positions 2: Keep positions".
  > Spot: `quoteSz`/`baseSz` investment.
- Help: https://www.okx.com/en-us/help/whats-the-spot-grid-bot-and-how-to-use-it (via WebFetch):
  > "A spot grid bot is a trading bot that you can customise and deploy to automatically place buy and sell orders at set intervals and within a specified price range."
  > Arithmetic: "maintains a consistent difference between each grid level so grid levels become (1, 1+r, 1+2r, 1+3r...)." Geometric: "maintains a consistent ratio between each grid level so grid levels become (1, 1*r, 1*r^2, 1+r^3...)." (The "1+r^3" typo is in the original.)
  > "The bot will not place any new orders if the BTC/USDC moves outside this range."
  > Take Profit: "an optional price target. If the price reaches this level, the bot will stop trading and sell all base assets at the market price." Stop Loss: "If the price drops to this level, the bot will stop trading and sell all base assets at the market price."
  > Trailing up: "follow the price if it continues rising beyond the initial upper limit". Trailing down: "extend downward if the price continues to fall below the grid's lower limit" (requires additional capital).
- **Portability: needs-approximation.** Same as A2a.
  - The RSI start trigger on 15m is natively bar-close based. Note that OKX evaluates it on its own candles, and the engine can compute RSI(14) on 15m at bar close. That part is near-exact, with the fill at the next open.
  - TP/SL "sell all base at market" is a direct stop/take.

##### A2c. Futures / contract grid (long / short / neutral)
- Binance "What Is Futures Grid Trading?": https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3 (via WebFetch):
  > "Grid trading is a trading bot that automates the buying and selling of Futures contracts. It is designed to place orders in the market at preset intervals within a configured price range."
  > Arithmetic: "Each grid has an equal price difference." Geometric: "Each grid has an equal price difference ratio."
  > Grid count: min 2, max 1,000 (USDⓈ-M), 169 (COIN-M). Stop triggers by price, PNL or ROI% *(paraphrase)*.
  > Modes *(paraphrase)*: a long grid opens an initial long position, a short grid an initial short. A neutral grid opens no initial position: it shorts above the reference price and goes long below it.
  - Search snippet: "In a neutral grid strategy, the system will execute short orders when the price is above the reference point, and long orders when the price is below the reference point."
- Binance futures-grid Trailing Up/Down: https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-and-trailing-down-functions-in-usd%E2%93%A2-m-futures-grid-trading-7a7bb22420404385991dee3a0930207d:
  > Trailing Down "moves the trading range downwards to align with a downward-trend market". It places "a new sell order at the old lower limit price"; there are user "Trailing Up limit price" / "Trailing Down limit price" settings.
- OKX contract grid (API, same section):
  > `direction` "Contract grid type long, short, neutral"; `lever` "Leverage"; `basePos` "Whether or not open a position when the strategy activates Default is false Neutral contract grid should omit the parameter"; `tpRatio` "Take profit ratio, 0.1 represents 10%"; `slRatio` "Stop loss ratio".
- **Portability: needs-approximation.** Targets: long grid k/N ∈ [0,1]; short grid −k'/N; neutral grid (centre − price-level index)/(N/2) ∈ [−1,1].
  - Leverage above 1 is not representable, because |target| ≤ 1. Cap it, or scale exposure down and note the change.
  - The intrabar caveats are the same as for the spot grid.
  - PNL/ROI% stop triggers can be computed from state at bar close, i.e. evaluated at the close rather than intrabar.

#### A3. DCA / recurring buy, and DCA (martingale) bots

##### A3a. Binance Auto-Invest → Convert Recurring
- https://www.binance.com/en/support/faq/what-is-auto-invest-and-how-to-use-it-3dd41bc1d4ea4879863ffbf2211a17fe:
  > "Binance Auto-Invest has transitioned to Convert Recurring. The Auto-Invest service is scheduled to discontinue after 31 March 2025, with the exception of the Index-Linked Plan, which will remain operational."
- Convert Recurring FAQ: https://www.binance.com/en/support/faq/detail/5026705ac87942069b1bdcf5a99ffc48 (via WebFetch):
  > "Binance Convert 'Recurring' allows you to automate investments and grow your asset holdings with Zero Fees"
  > Frequencies: hourly (1/4/8/12), daily, weekly, bi-weekly and monthly *(paraphrase)*. For hourly plans, the first trade is 1 hour after creation; other plans run at the scheduled date and time, with the minute equal to the plan-creation minute *(paraphrase)*.
  > "Recurring prices may be different from market prices on Binance Spot." Pricing follows the Convert quote methodology.
- Note: Binance "Recurring Buy" (card/fiat) is a separate product. Search snippets say Recurring Buy supports fiat with Credit/Debit Card; not fetched in detail.
- **Portability: needs-approximation.** Mapping:
  - `state` holds a cash budget B, a per-period amount a, base units held q, and quote spent.
  - On scheduled bars (index % (period/15m) == 0), q += a/price_at_next_open, and target = q*P/(equity).
  - Problems:
    1. 0.05 quantisation: small periodic buys of about 1% of equity each round to zero. Either accumulate them until a 0.05 step is crossed, or choose a/B so that each step equals 0.05 (i.e. 20 purchases).
    2. The engine is fraction-of-equity with fixed capital and no external deposits. Model DCA as "deploy a fixed cash pile over K periods".
    3. The Convert price is not the spot open.
    4. A schedule minute that is not on the :00/:15/:30/:45 grid has to be snapped to the next bar open.

##### A3b. OKX Recurring Buy
- API `POST /api/v5/tradingBot/recurring/order-algo` (curl+grep):
  > `recurringList` > `ccy`, `ratio` "Proportion of recurring currency assets, e.g. "0.2" representing 20%", `minPx`/`maxPx` "Minimum/Maximum price of recurring currency. "" means no limit"; `period` "monthly / weekly / daily / hourly"; `recurringHour` "Recurring buy by hourly 1/4/8/12"; `recurringTime` "[0,23]"; `timeZone` "[-12,14]"; `amt` "Quantity invested per cycle"; `investmentCcy` "can only be USDT / USDC"; `tdMode` "cross / cash".
- Help pages (search snippets): https://www.okx.com/en-us/help/how-do-i-set-up-a-recurring-buy-plan, https://www.okx.com/en-us/help/vii-recurring-buy, https://www.okx.com/en-us/help/how-do-i-use-recurring-buy-trading-bot:
  > "a dollar-cost averaging (DCA) strategy for investing a fixed amount in crypto at your defined fixed intervals"; "up to 20 different digital currencies"; "OKX only supports plans fixed at the cash amount for recurring buy at this time."
- **Portability: needs-approximation.** Same as A3a. The price band (minPx/maxPx: skip the buy outside the band) can be checked at bar close. The engine is single-asset, so a multi-currency ratio list reduces to BTC only.

##### A3c. Binance Spot DCA bot
- "Binance Spot DCA Parameters": https://www.binance.com/en/support/faq/binance-spot-dca-parameters-797f6e465186474fac2add2f7ce0474a (via WebFetch):
  > Price Deviation: "the price difference percentage that triggers DCA orders. You can set the price difference from 0.1% to 15%."
  > Take Profit: "the target profit percentage based on the average price (before the trading bot ends)." [Fix] / [Trailing] modes. Search snippet: "Once the position increases by the take-profit percentage, the TP Order will close the position, and the bot will end this round."
  > DCA Order: "the subsequent orders that will be filled when the price reaches the designated price deviation." Max DCA Order: "The maximum number of DCA orders that will be placed per round."
  > Trigger Price: "The price of the token that will trigger the trading bot to start. If left blank, the last price will be used as the trigger price."
  > Price Deviation Multiplier: "Multiply the price difference in the percentage at which DCA orders will be placed starting from the second entry." DCA Order Size Multiplier: "Multiply the subsequent DCA Order investment amount(s)."
  > Cooldown: "The rest period between each round. By default, the cooldown between rounds is 60 seconds." Stop-Loss: "A stop-loss percentage that triggers the trading bot to stop the round."
- Also see: https://www.binance.com/en/support/faq/what-is-spot-dca-and-how-does-it-work-27713d3ddb3c406da52f36b9aaaa1360 (search only).

##### A3d. OKX Spot DCA (Martingale) bot / contract DCA
- API `POST /api/v5/tradingBot/dca/create` (curl+grep):
  > `algoOrdType` "contract_dca: Contract DCA order / spot_dca: Spot DCA order"; `initOrdAmt` "Initial order amount"; `safetyOrdAmt` "Safety order amount"; `maxSafetyOrds` "Max number of safety orders"; `pxSteps` "Safety order price step"; `pxStepsMult` "Price step multiplier"; `volMult` "Safety order amount multiplier"; `tpPct` "Take-profit target per cycle 0.05 represents 5%"; `slPct` "Stop-loss target"; `slMode` "limit / market"; `direction` "long / short" (contract only); `allowReinvest`; triggerParams with start "instant / price / rsi" (rsi on 3m…1D).
  > The amend endpoint describes `pxSteps` as "Price step ratio (price gap to trigger the first safety order)".
- Help: https://www.okx.com/en-eu/help/whats-the-spot-dca-bot-and-how-to-use-it (via WebFetch):
  > "Spot DCA Bot buys an asset at your chosen starting price, then automatically places extra buy orders ("safety orders") if the price falls."
  > Amount multiplier: "a multiplier of 1.5 will make each safety order 50% larger than the previous one." Price step multiplier: "used to multiply the price steps of the last safety order to calculate the deviation percentage of the new safety order."
  > Cycle: "the initial order to buy, optional filling of one or many safety orders and the final execution of take profit order." "When the bot achieves this target price, it ends the current cycle, completes the order, and starts a new cycle as specified."
- **Portability (A3c and A3d): needs-approximation.** Mapping:
  - Total budget = initOrd + Σ_{i=1..M} safety·volMult^(i−1). Each order's fraction of equity is its share of that budget, with the cumulative fraction rounded to 0.05. The result is usable when M is small, e.g. 1+1+2+4 = 8 units → 0.125 steps → quantised.
  - Safety level i sits at entry·(1 − Σ step·pxStepsMult^(j−1)).
  - At bar close, if low ≤ next safety level (or close below it, for a conservative rule), raise the target at the next open. The exchange fills at the limit level intrabar, so the engine price is worse on dips that rebound within the bar and different otherwise.
  - TP is a take level at avg·(1+tpPct), returned each bar, with `avg` computed in state. **This part is exact** apart from the take fill rule.
  - SL is a stop level. Its base differs by venue: Binance uses % from average price, OKX "stop-loss target" %, so read the docs carefully per venue.
  - Binance "Trailing" TP needs the trailing approximation (A1).
  - Cooldown: next bar, or after N bars.
  - Short contract DCA uses the mirror image.

#### A4. Other conditional orders (brief)
- **Binance Spot OCO** (https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md, `POST /api/v3/orderList/oco`):
  > "Send in an one-cancels-the-other (OCO) pair, where activation of one order immediately cancels the other." "One of the orders must be a `LIMIT_MAKER/TAKE_PROFIT/TAKE_PROFIT_LIMIT` order and the other must be `STOP_LOSS` or `STOP_LOSS_LIMIT` order."
  - OTOCO: "An OTOCO (One-Triggers-One-Cancels-the-Other) is an order list comprised of 3 orders." The pending OCO pair is placed only when the working order is "fully filled".
  - "For `STOP_LOSS`, `STOP_LOSS_LIMIT`, `TAKE_PROFIT_LIMIT` and `TAKE_PROFIT` orders, `trailingDelta` can be combined with `stopPrice`."
  - **Portability:** an OCO of market-type STOP_LOSS + TAKE_PROFIT that covers the whole position maps **directly** to the engine's stop+take. The engine's stop-wins tie rule is a conservative choice where the exchange would follow the true tick order. The LIMIT/LIMIT_MAKER legs are approximate (non-fill risk). OTOCO = entry at next open (the working leg as a market order) + stop/take, which is approximate if the working leg is a limit.
- **Binance futures STOP / STOP_MARKET / TAKE_PROFIT / TAKE_PROFIT_MARKET** (API page via WebFetch; triggers on latest MARK_PRICE or CONTRACT_PRICE; `closePosition`; `priceProtect`).
  - **Portability:** STOP_MARKET/TAKE_PROFIT_MARKET with closePosition, re-placed every bar, is **direct**, except when workingType=MARK_PRICE.
- **OKX** `conditional` (one-way TP/SL), `oco`, `trigger`, `chase`, `smart_iceberg`, `twap`; attached TP/SL `attachAlgoOrds` with `tpTriggerPx`/`slTriggerPx`; split TPs with "Cost-price SL ... slTriggerPx will move to avgPx when the first TP order is triggered".
  - **Portability:** single-level TP/SL closing the whole position is direct. Split TPs and closeFraction are not portable via orders: they have to be approximated with target reductions at the next open. The move-SL-to-breakeven logic is portable through state.
  - Help: "Stop order is an algorithmic trading strategy with which you can set a trigger price and an order price to limit losses and reduce risks"; OCO: "When either one is triggered, the other order is automatically canceled."
- **TWAP** (OKX `twap`; Binance algo TWAP): **excluded from scope** as requested. It is an execution algorithm, not a signal.
- **Chase / iceberg:** execution tactics, excluded.

---

### PART B — TradingView licence situation (policy pages only; no scripts read)

#### B1. Default licence for open-source scripts
- **Terms of Use §22 "Scripts"** (https://www.tradingview.com/policies/, curl 200, text extracted):
  > "By using the publish script feature, you grant us a world-wide, irrevocable, perpetual, royalty-free license to: publish this script as well as the username of the author; disclose the source code of the script for open scripts; perform, display, use and make available your open or protected script for any user or an invite-only script for any user that received an invite. We may remove your script without notice if we reasonably believe that you are in violation of these Terms of Use. **You can publish the script under any license. If you do not include the license in the comment section of a script, you agree that your script is licensed under the Mozilla Public License 2.0.**"
  > Also: "You represent and warrant that you have all intellectual property rights ... in and to your script. If you use third-party materials, you represent and warrant that you have the right to distribute third-party material in the script."
- **Pine Script User Manual, "Publishing scripts"** (https://www.tradingview.com/pine-script-docs/writing/publishing/, via WebFetch; curl returned compressed bytes):
  > "All open-source scripts on TradingView use the Mozilla Public License 2.0 by default. Authors wanting to use alternative licenses can specify them in the source code."
  > Open: "A script published with the 'Open' setting is _open-source_, meaning anyone who views the publication or uses the script can access its Pine Script code."
  > Protected: "...has _closed-source_ code, meaning the code is protected and not viewable to any user except the author."
  > Invite-only: "...has closed-source code. No user except the author can view the code. Additionally, unlike a protected script, only users _invited_ by the author can add the script to their charts and use it."
- **Conclusion:** MPL-2.0 is the **default**, applied when there is no licence comment. The Pine editor's new-script template header, "This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/", is the usual form, but this research did not re-verify it from a TradingView page. Authors **may choose any other licence** by stating it in the source, e.g. CC BY-NC-SA 4.0, which is common. So the header has to be checked per script; MPL-2.0 cannot be assumed if the header says otherwise.
  - MPL-2.0 is file-level copyleft: you may port, modify and redistribute, but a modified covered file must stay MPL-2.0, keep its notices, and make its source available.
  - A Python port is arguably a derivative "Modification" of the covered file. Keep ports in separate files under MPL-2.0 with attribution, and do not mix them into differently licensed files.
- **Invite-only commercial-use ban (ToS §23):** "Invite-only scripts published on TradingView may be used for non-commercial purposes only. Commercial use of any invite-only script is prohibited." Protected and invite-only code is not visible anyway, so exclude it.

#### B2. Reuse rules (House Rules + Script publishing rules)
- **House Rules** (https://www.tradingview.com/support/solutions/43000591638/):
  > "Don't plagiarize. Please, please, please create and share content that's unique to you and you alone. Don't copy or use someone else's content without asking for permission and crediting them. Make sure you've got the legal rights to reuse content."
- **Script publishing rules** (https://www.tradingview.com/support/solutions/43000590599-script-publishing-rules/):
  > "► Reuse open-source code responsibly. If your script reuses open-source code from another author: You must credit the original author in your publication's description, or in the release notes if the reused code is part of a script update. We also recommend crediting the author in your open-source code comments. The reused code should occupy only a portion of your script, and we expect you to add significant improvements. Minor modifications, such as style revisions, input edits, identifier changes, code rearrangements, and Pine version changes, do not constitute significant improvements. You must create an open-source publication. The only exception is if you can prove that you received explicit permission from the author to reuse their code in a closed-source script, or if the reused code is considered public domain and is only a portion of your entire script. Code that we consider public domain includes the following: All code from TradingView's built-in scripts and documentation. Imported code from publicly published library scripts. Ported code for standard classic indicators that are widely available on other platforms, such as a standard RSI. This excludes code for classic indicators with author-specific variations, and ports of indicators recently published elsewhere."
  > "► Don't imitate other authors. ... clear attempts to emulate or mimic scripts from other authors are not allowed."
  > "Open-source and protected scripts are, by definition, free."
- **Interpretation:** these rules govern *publishing on TradingView*. Off-platform reuse is governed by the script's licence plus copyright law. Still, the community norm is clear: credit is mandatory; built-in scripts and docs, published libraries, and standard classic indicators count as "public domain" in TV's view.

#### B3. ToS clauses relevant to automated collection
- The ToS has **no explicit "scraping/crawler/robot" clause**: a grep of the full ToS text for scrap/robot/spider/crawl/harvest found nothing. The relevant clause is **§3 "Ownership of information; license to use TradingView; redistribution of data; non-display usage"**:
  > "The content and market data provided on the TradingView platform, including but not limited to charts, alerts, webhooks, and any other forms of information, are licensed for exclusive display-only use. This license is strictly limited to personal or internal business purposes and explicitly prohibits any form of non-display usage. Such prohibited uses include, but are not limited to, any form of automated trading, automated order generation, price referencing, order verification, algorithmic decision-making, algorithmic trading, smart order routing, using data in operations control or risk management programs, or any machine-driven processes that do not involve the direct, human-readable display of such data. Such prohibited cases also include creating products or services based on TradingView content, any processing of TradingView's content, or any other use cases that undermine the restrictions in place by the Data Providers."
  > "Unless otherwise noted, all rights, titles, and interests in TradingView, and all information made available through TradingView ... are the exclusive property of TradingView, our affiliates or our Data Providers"
- AI features (§26): "You may use AI Features only for your own lawful, personal, non-automated use of the platform."
- **robots.txt** (https://www.tradingview.com/robots.txt) explicitly blocks AI crawlers. `User-agent: ClaudeBot`, `GPTBot`-family, `Google-Extended`, `PerplexityBot` and others get `Disallow: /scripts/*`, `/script/*`, `/ideas/*`, `/chart/*`, `/u/*`; all agents get `Disallow: /scripts/search/`.
- **Conclusion:** automated collection of scripts is **not permitted in practice**. §3 bans "machine-driven processes" and "any processing of TradingView's content", and robots.txt disallows AI agents on /script/*. Script *source code* is user content under the author's licence (§22 "unless otherwise noted"), so a human-curated, manual collection of open-source scripts under their licence is defensible. Pine code is portable logic, not TV market data. Nonetheless, keep it small and manual, and do not use TV data or automation.

#### B4. Manual collection protocol (recommended)
1. **Human browsing only.** No crawler, no headless browser, no API; do not use AI agents to fetch /script/ pages (robots.txt). Use a normal account and a browser.
2. **Eligibility:** visibility must be "Open"/open-source. Skip protected and invite-only scripts, and skip anything behind a paywall or Paid Space.
3. **Licence check per script:** read the header comment and record the licence verbatim.
   - MPL-2.0 (explicit or by default): OK to port. Keep the port in its own file under MPL-2.0 with notices, and make the source available.
   - CC BY-NC-SA / CC BY-NC: non-commercial only, share-alike. Flag it, and include only if the benchmark is non-commercial and ShareAlike-compatible.
   - "All rights reserved" or no-reuse statements: exclude.
   - No header at all: MPL-2.0 by ToS §22.
4. **Record metadata:** script URL, title, author handle, publish/update date, Pine version, visibility, licence text, date accessed, and who collected it. Save a copy of the original source with its header intact (MPL requires notices to be kept).
5. **Attribution:** credit the author and URL in the ported file header and in the benchmark catalogue. State "ported from Pine to MintEval interface; modified".
6. **Content scope:** port the strategy logic only. Do not copy TV market data, charts, or descriptions beyond short factual summaries; descriptions are author copyright, so paraphrase. Built-in TV strategies and docs examples are "public domain" per TV's publishing rules, but even so, cite TV.
7. **Porting to `strategy(hist,state,pos,ind)`, with Pine semantics to watch** (Pine User Manual "Strategies", https://www.tradingview.com/pine-script-docs/concepts/strategies/):
   - Default fills: "The earliest point at which the broker emulator fills an order is on the next available tick". Market orders placed at bar close fill at the next bar's open, which matches the engine.
   - `process_orders_on_close=true` fills on the *same* bar close. The engine fills at the next open, so expect a one-bar shift.
   - `calc_on_every_tick` / `calc_on_order_fills` recompute intrabar on realtime bars or after fills. The engine recomputes only at bar close, so these are not reproducible.
   - Broker-emulator intrabar path: "If the opening price of a bar is closer to the high than the low, the emulator assumes the market price moved: open → high → low → close" (and the reverse when open is closer to the low). `use_bar_magnifier` uses lower-timeframe data. The engine uses stop-wins instead, so results differ when stop and take both lie inside one bar.
   - `strategy.exit(trail_price | trail_points, trail_offset)`: activation level plus trailing offset in ticks, updated intrabar. Map it to state-tracked peak plus a per-bar stop (A1 approximation). `trail_points`/`trail_offset` are in *ticks* (syminfo.mintick), so convert to price units.
   - `strategy.exit(qty_percent=…)`, multiple exits, and partial TP ladders are not portable as orders; use target reductions at the next open.
   - `pyramiding` (default 1) and `strategy.entry` sizing (`default_qty_type` = percent_of_equity / fixed / cash) map to the target fraction, rounded to 0.05. Pyramiding with N entries becomes target steps.
   - `strategy.order` vs `strategy.entry` (entry reverses positions): the engine is a target-position model, so reversal = target sign flip.
   - Limit and stop *entry* orders (`limit=`/`stop=` on entry) are not supported; approximate them as entry at the next open after close crosses.
   - `request.security` to higher timeframes: beware lookahead. Use `lookahead_off` semantics, i.e. only completed HTF bars.
   - `var`/`varip` persistent variables map to the flat `state` dict. Arrays and maps need flattening.
   - Commission/slippage settings are Pine's own; use the engine's fee model instead.
   - Repainting: `barstate.isrealtime` logic and `security` lookahead have to be removed.

---

### Attempt log

| # | Target | URL | Tool | Outcome |
|---|---|---|---|---|
| 1 | Binance futures New Order | https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api | WebFetch | OK (summarised; partial verbatim; trailing bullets truncated) |
| 2 | Binance spot trailing FAQ | https://developers.binance.com/docs/binance-spot-api-docs/faqs/trailing-stop-faq | WebFetch | OK (summary) |
| 3 | OKX API docs (algo) | https://www.okx.com/docs-v5/en/#order-book-trading-algo-trading-post-place-algo-order | WebFetch | FAIL (page too large; section not in summary) |
| 4 | OKX API docs | https://www.okx.com/docs-v5/en/ | curl + python grep | OK (5.2 MB; move_order_stop, grid, DCA, recurring sections extracted verbatim) |
| 5 | Binance futures New Order | same as #1 | curl | FAIL (HTTP 202, 0 bytes — WAF/JS challenge) |
| 6 | Binance spot trailing FAQ (raw md) | https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/faqs/trailing-stop-faq.md | curl | OK (full verbatim) |
| 7 | binance-docs.github.io legacy | https://binance-docs.github.io/apidocs/futures/en/ | curl | redirect stub (325 bytes); not pursued |
| 8 | Binance futures New Order | same as #1 | WebFetch (verbatim prompt) | Partial (BUY/SELL bullets still truncated by summariser; -2021 note, closePosition OK) |
| 9 | Search Binance trailing FAQ | WebSearch | WebSearch | OK |
| 10 | Binance trailing FAQ | https://www.binance.com/en/support/faq/what-is-a-trailing-stop-order-360042299292 | WebFetch | OK |
| 11 | Binance dev forum thread | https://dev.binance.vision/t/conditions-for-trailing-stop-market/12644 | WebFetch | FAIL (socket closed) |
| 12 | Same thread | https://dev.binance.vision/t/conditions-for-trailing-stop-market/12644.json | curl (Discourse JSON) | OK — full verbatim TRAILING_STOP_MARKET rule |
| 13 | Search Binance spot grid | WebSearch | WebSearch | OK |
| 14 | Search Binance Spot DCA | WebSearch | WebSearch | OK |
| 15 | Search Binance Auto-Invest | WebSearch | WebSearch | OK |
| 16 | Binance spot grid parameters | https://www.binance.com/en/support/faq/binance-spot-grid-trading-parameters-688ff6ff08734848915de76a07b953dd | WebFetch | OK (formulas; no TP/SL/trailing on that page) |
| 17 | Binance what is spot grid | https://www.binance.com/en/support/faq/what-is-spot-grid-trading-and-how-does-it-work-d5f441e8ab544a5b98241e00efb3a4ab | WebFetch | OK |
| 18 | Binance spot DCA parameters | https://www.binance.com/en/support/faq/binance-spot-dca-parameters-797f6e465186474fac2add2f7ce0474a | WebFetch | OK |
| 19 | Binance Auto-Invest | https://www.binance.com/en/support/faq/what-is-auto-invest-and-how-to-use-it-3dd41bc1d4ea4879863ffbf2211a17fe | WebFetch ×2 | OK (transition to Convert Recurring) |
| 20 | Search Convert Recurring | WebSearch | WebSearch | OK |
| 21 | Binance trailing up (spot grid) | https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-function-in-spot-grid-trading-3d987afd7906495cb4d997eccb8515bf | WebFetch | OK |
| 22 | Binance Convert Recurring FAQ | https://www.binance.com/en/support/faq/detail/5026705ac87942069b1bdcf5a99ffc48 | WebFetch | OK |
| 23 | Search Binance futures grid | WebSearch | WebSearch | OK |
| 24 | Binance create spot grid | https://www.binance.com/en/support/faq/how-to-create-a-spot-grid-trading-strategy-on-binance-95078b6293184bd79b56108092f337c1 | WebFetch | Partial (mentions TP/SL/trigger, no detail) |
| 25 | Search spot grid trailing down / sell all base | WebSearch | WebSearch | OK (sell-all-base snippet; futures trailing-down page found) |
| 26 | Binance futures grid FAQ | https://www.binance.com/en/support/faq/what-is-futures-grid-trading-f4c453bab89648beb722aa26634120c3 | WebFetch | OK |
| 27 | Binance futures grid trailing up/down | https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-and-trailing-down-functions-in-usd%E2%93%A2-m-futures-grid-trading-7a7bb22420404385991dee3a0930207d | WebFetch | OK |
| 28 | Search OKX spot grid | WebSearch | WebSearch | OK |
| 29 | Search OKX trailing stop | WebSearch | WebSearch | OK |
| 30 | OKX how to use trailing stop | https://www.okx.com/en-us/help/how-to-use-trailing-stop | WebFetch | OK (no formula) |
| 31 | OKX spot grid help | https://www.okx.com/en-us/help/whats-the-spot-grid-bot-and-how-to-use-it | WebFetch | OK |
| 32 | OKX spot DCA help | https://www.okx.com/en-eu/help/whats-the-spot-dca-bot-and-how-to-use-it | WebFetch | OK |
| 33 | Search OKX recurring buy | WebSearch | WebSearch | OK (snippets) |
| 34 | OKX strategy order types | https://www.okx.com/en-us/help/xi-strategy-order-types | WebFetch | OK (trailing formula) |
| 35 | TV House Rules | https://www.tradingview.com/support/solutions/43000591638/ | WebFetch, then curl | OK |
| 36 | Search TV script publishing rules | WebSearch | WebSearch | OK |
| 37 | TV Terms of Use | https://www.tradingview.com/policies/ | WebFetch, then curl + grep | OK (§3, §22, §23, §26) |
| 38 | TV robots.txt | https://www.tradingview.com/robots.txt | curl | OK |
| 39 | TV Script publishing rules | https://www.tradingview.com/support/solutions/43000590599-script-publishing-rules/ | curl + grep | OK |
| 40 | Pine docs publishing | https://www.tradingview.com/pine-script-docs/writing/publishing/ | curl | FAIL (compressed/binary body) |
| 41 | Pine docs publishing | same | WebFetch | OK |
| 42 | Pine docs strategies | https://www.tradingview.com/pine-script-docs/concepts/strategies/ | WebFetch | OK (summary + key verbatim lines) |
| 43 | Binance spot REST API (OCO/OTOCO) | https://raw.githubusercontent.com/binance/binance-spot-api-docs/master/rest-api.md | curl + grep | OK |

Not done or gaps:
- The spot-grid grid-count range on Binance was not found on the pages fetched (futures: 2–1000).
- Binance "Recurring Buy" (card) was not fetched in detail.
- The Pine new-script default header text was not re-verified from a TV page.
- The Binance futures trailing BUY/SELL bullets come from the official dev forum quoting the docs; the developers.binance.com page itself returned only a summarised version.
