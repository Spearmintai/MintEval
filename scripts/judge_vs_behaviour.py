"""QuantCode-Bench judge vs behavioural equivalence on the frontier-200 subset (open setting).
Usage: judge_vs_behaviour.py <scores.csv> <out_prefix> [model,...]"""
import sys, json, os
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from minteval.llm import ChatClient
from minteval.qcb_judge import judge
JUDGE = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-4")
cl = ChatClient("https://openrouter.ai/api/v1", JUDGE, os.environ["OPENROUTER_API_KEY"],
                extra={"reasoning": {"enabled": False}})
scores = pd.read_csv(sys.argv[1]); prefix = sys.argv[2]
models = sys.argv[3].split(",") if len(sys.argv) > 3 else sorted(scores.model.unique())
ids = [json.loads(l)["strategy_id"] for l in open("results/tasks/frontier200.jsonl")]
Q = {json.loads(l)["strategy_id"]: json.loads(l)["q"] for l in open("results/tasks/prompts.jsonl")}
gens = {}
for m in models:
    for d in ("results/generations", "results/generations_frontier"):
        f = f"{d}/{m}__open.jsonl"
        if os.path.exists(f):
            for l in open(f):
                r = json.loads(l); gens[(r["strategy_id"], m)] = r["code"]
rows = scores[(scores.setting == "open") & scores.model.isin(models) & scores.strategy_id.isin(ids) & (scores.compile_fail == 0)]
jobs = [(r.strategy_id, r.model) for r in rows.itertuples()]
def f(k):
    return k, judge(cl, Q[k[0]], gens[k], tag=f"qcbjudge-{JUDGE}-{k[1]}-{k[0]}")
with ThreadPoolExecutor(4) as ex:
    res = dict(ex.map(f, jobs))
J = pd.DataFrame([{"strategy_id": s, "model": m, **v} for (s, m), v in res.items()])
D = rows.merge(J, on=["strategy_id", "model"])
D.to_csv(f"{prefix}_rows.csv", index=False)
D["behaviour_ok"] = D.action_match >= 0.9
tab = {}
for m, g in D.groupby("model"):
    g = g[g.judge_pass.notna()]
    tab[m] = {"n": len(g), "judge_pass": float(g.judge_pass.mean()),
              "pass_and_ok": int((g.judge_pass & g.behaviour_ok).sum()), "pass_and_bad": int((g.judge_pass & ~g.behaviour_ok).sum()),
              "fail_and_ok": int((~g.judge_pass.astype(bool) & g.behaviour_ok).sum()), "fail_and_bad": int((~g.judge_pass.astype(bool) & ~g.behaviour_ok).sum()),
              "share_silent_among_pass": float((~g[g.judge_pass.astype(bool)].behaviour_ok).mean()),
              "AM_mean_among_pass": float(g[g.judge_pass.astype(bool)].action_match.mean())}
out = {"judge_model": JUDGE, "parse_failures": int(D.judge_pass.isna().sum()), "api_errors": int(D.judge_error.notna().sum()),
       "cost": float(D.cost.fillna(0).sum()), "by_model": tab}
json.dump(out, open(f"{prefix}.json", "w"), indent=1); print(json.dumps(out, indent=1))
