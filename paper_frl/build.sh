#!/usr/bin/env bash
# SPY/QQQ bars (once): POLYGON_API_KEY=... .venv/bin/python scripts/download_polygon.py -> data/prices/*_rth_2024_2026.csv
# FRL paper: regenerate numbers and figures from results/frl/minteval_v0_frl.csv, then compile.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/frl_analysis.py > /dev/null      # Tables 1-2, robustness, Fig 3 -> paper_frl/frl_analysis.json
.venv/bin/python scripts/frl_examples.py > /dev/null      # Fig 2 candidate search -> results/frl/fig2_candidates.csv
.venv/bin/python scripts/frl_equity_examples.py > /dev/null  # SPY/QQQ re-run -> results/frl/equity_candidates.csv
.venv/bin/python scripts/frl_fig2.py > /dev/null          # Fig 2 -> paper_frl/fig2_cases.json
.venv/bin/python scripts/frl_fig1.py                      # Fig 1
cd paper_frl
${TECTONIC:-~/.local/bin/tectonic} -X compile main.tex
