"""Replication of the QuantCode-Bench LLM judge (Khoroshilov et al., arXiv:2604.15151).

Prompt and rating parser copied verbatim from
https://github.com/LimexAILab/QuantCode-Bench/blob/main/quantcode_bench/judge.py (MIT License,
Copyright (c) the QuantCode-Bench authors). Settings as in their code: temperature 0, max_tokens 8000,
thinking disabled. task_description = our colloquial instruction q_s; the Backtrader-specific leniency
clauses are kept unchanged so the judge is exactly theirs.
"""
from __future__ import annotations

import re


def evaluation_prompt(task_description: str, generated_code: str) -> str:
    return f"""You are an expert in evaluating trading strategies. Your task is to determine whether the generated strategy code meets the requirements of the assignment.

**STRATEGY ASSIGNMENT:**
{task_description}

**GENERATED CODE:**
```python
{generated_code}
```

**EVALUATION CRITERIA:**

1. **Indicator compliance**: Does the code use the indicators or equivalent logic from the assignment? IMPORTANT: Accept reasonable simplifications if the assignment requires complex statistical methods (e.g. Johansen test) that are impossible in Backtrader. Simplifying to SMA/StdDev for Z-score is ACCEPTABLE.

2. **Problem solving**: Does the code implement the MAIN IDEA of the task? If the task is about mean reversion and the code uses Z-score and trades on deviations — this is COMPLIANT, even if details are simplified.

3. **Relevance**: Is the code related to the task description? The code must explicitly ignore ALL conditions of the assignment to receive 0.

**INSTRUCTIONS:**

Analyze the code on each criterion (1-2 sentences per criterion).

Then provide a final rating in the format "Rating: [[X]]", where:
- X = 1 if the code implements the MAIN LOGIC of the assignment (even with simplifications)
- X = 0 if the code COMPLETELY DOES NOT COMPLY (ignores the main idea)

IMPORTANT: Be LENIENT with simplifications. Penalize ONLY if the code clearly ignores the main task or uses completely different indicators/logic WITHOUT reason. If the assignment is technically impossible to implement exactly in Backtrader — accept reasonable simplifications.

You MUST end your response with the line "Rating: [[X]]"."""


def parse_rating(text: str):
    """Their regex cascade (keyword fallback omitted: returns None instead so we can count parse failures)."""
    low = (text or "").lower().strip()
    for pat in (r'rating:\s*\[\[(\d+)\]\]', r'rating\s*\[\[(\d+)\]\]', r'\[\[(\d+)\]\]', r'rating:\s*(\d+)'):
        m = re.findall(pat, low)
        if m:
            try:
                return int(m[-1]) == 1
            except ValueError:
                continue
    return None


def judge(client, q: str, code: str, tag: str) -> dict:
    out = client.chat([{"role": "user", "content": evaluation_prompt(q, code)}], temperature=0.0,
                      max_tokens=8000, seed=None, tag=tag)
    return {"judge_pass": parse_rating(out.get("text")), "judge_text": out.get("text"),
            "judge_error": out.get("error"), "cost": (out.get("usage") or {}).get("cost")}
