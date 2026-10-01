# MintEval v0

Behavioural-equivalence benchmark: can an LLM turn a trader's colloquial instruction into a
strategy program that *acts* like the reference, bar by bar, on identical data and frictions?

## Pipeline (all parameters in `configs/base.yaml`; progress/decision log in `PROGRESS.md`)
```bash
uv venv --python 3.12 .venv && uv pip install numpy pandas pyyaml radon matplotlib scipy statsmodels pytest requests numba pymupdf
python -m pytest -q tests                      # acceptance tests 2, 3, 6 + engine/sandbox units
python scripts/build_pool.py                   # 15,000 candidate programs -> results/pool/
python scripts/select_tasks.py                 # 800 tasks, tau quintile x R band -> results/tasks/
python scripts/self_match.py                   # acceptance tests 1 + 4 (all 800 references)
python scripts/translate_tasks.py all results/tasks/prompts.jsonl   # needs OPENROUTER_API_KEY
python scripts/run_models.py results/tasks/prompts.jsonl all open,closed
python scripts/score.py results/generations results/tasks/prompts.jsonl results/minteval_v0.csv
python scripts/sensitivity_state.py            # lenient-state sensitivity
paper/build.sh                                 # figures, tables, numbers.tex, both PDFs
```
Local models: vLLM on a remote GPU box, reached through an SSH port forward (127.0.0.1:18007/18032).

## Layout
`minteval/` engine (physical no-lookahead), tracked state, indicators, sandbox, primitives,
assembler, complexity, translate, prompts, runner, evaluate, metrics. `data/jargon.txt` is
AI-drafted and **pending human review**. `data/real/` holds the real-subset survey and a 10-task
pilot (**pending human verification**). `data/human_rewrite/` holds the 60-task sheet for traders.
