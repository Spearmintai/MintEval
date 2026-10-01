#!/usr/bin/env bash
# Builds both variants. numbers.tex / table1.tex / figs are regenerated from the CSV first.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/analyze.py results/minteval_v0.csv paper > /dev/null
cd paper
TECTONIC=${TECTONIC:-~/.local/bin/tectonic}
$TECTONIC -X compile main.tex                                         # arXiv: [sigconf,nonacm]
sed 's/\\documentclass\[sigconf,nonacm\]{acmart}/\\documentclass[sigconf,anonymous,review]{acmart}/' main.tex > main_submission.tex
$TECTONIC -X compile main_submission.tex                              # submission: [sigconf,anonymous,review]
