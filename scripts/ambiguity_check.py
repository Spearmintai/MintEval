"""Breakout-retest tasks: does the instruction say the breakout is on the CLOSE? (Opus silent failures)"""
import json, re, pandas as pd
T = {json.loads(l)["strategy_id"]: json.loads(l)["spec"] for l in open("results/tasks/tasks.jsonl")}
P = {json.loads(l)["strategy_id"]: json.loads(l)["q"] for l in open("results/tasks/prompts.jsonl")}
fr = pd.read_csv("results/minteval_v0_frontier.csv")
out = {}
for m, g in fr[fr.compile_fail == 0].groupby("model"):
    rows = []
    for r in g.itertuples():
        if T[r.strategy_id]["signal"]["id"] != "breakout_pullback": continue
        q = P[r.strategy_id].lower()
        says = bool(re.search(r"clos\w*[^.]{0,40}(above|over|beyond|past|below|under|through)|(above|below)[^.]{0,30}clos", q))
        rows.append({"says_close": says, "silent": r.action_match < 0.9})
    R = pd.DataFrame(rows)
    if R.empty: continue
    out[m] = {f"{'close' if k else 'noclose'}": {"n": int(len(v)), "silent": int(v.silent.sum())} for k, v in R.groupby("says_close")}
json.dump(out, open("results/ambiguity_check.json", "w"), indent=1); print(json.dumps(out, indent=1))
