# Human-rewrite subset (60 tasks, 12 per tau bin; seed 7)
Give `to_rewrite.csv` to 2+ practising traders. Each writes, in their own words, the chat message they would
send a quant developer for the strategy in `spec`. Do not show them the LLM-written version. Desk conventions
(entry = signal-bar close, ATR = 14, above/below strict, at least/at most inclusive, opposite signal closes)
may be assumed. Every number must survive. Then run:
  python scripts/run_models.py data/human_rewrite/prompts_human.jsonl all open all results/generations_human
to compare model scores on human vs LLM instructions for the same 60 programs.
