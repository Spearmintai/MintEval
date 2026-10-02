"""Run the QuantCode-Bench judge (verbatim prompt) with a given judge model over
(a) compiled open-setting model runs on the 200-task subset and (b) the graded mutants.
Usage: judge_runs.py <judge_model_id> <out_tag> [extra_json]"""
import sys, json, os
sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from minteval.llm import ChatClient
from minteval.qcb_judge import judge
JM, TAG = sys.argv[1], sys.argv[2]
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"reasoning": {"enabled": False}}
cl = ChatClient("https://openrouter.ai/api/v1", JM, os.environ["OPENROUTER_API_KEY"], extra=extra)
Q = {json.loads(l)["strategy_id"]: json.loads(l)["q"] for l in open("results/tasks/prompts.jsonl")}
ids = {json.loads(l)["strategy_id"] for l in open("results/tasks/frontier200.jsonl")}
S = pd.concat([pd.read_csv("results/minteval_v0.csv"), pd.read_csv("results/minteval_v0_frontier.csv")])
S = S[(S.setting == "open") & S.strategy_id.isin(ids) & (S.compile_fail == 0)]
code = {}
for d in ("results/generations", "results/generations_frontier"):
    for f in os.listdir(d):
        if f.endswith("__open.jsonl"):
            for l in open(f"{d}/{f}"):
                r = json.loads(l); code[(r["strategy_id"], r["model"])] = r["code"]
jobs = [("run", r.strategy_id, r.model, Q[r.strategy_id], code[(r.strategy_id, r.model)], r.action_match) for r in S.itertuples()]
MS = pd.read_csv("results/mutants/mutant_scores.csv").set_index(["strategy_id", "type"])
for l in open("results/mutants/mutants.jsonl"):
    m = json.loads(l)
    jobs.append(("mutant", m["strategy_id"], m["type"], m["q"], m["code"], MS.loc[(m["strategy_id"], m["type"]), "action_match"]))
def f(j):
    kind, sid, who, q, c, am = j
    r = judge(cl, q, c, tag=f"qcbjudge-{JM}-{who}-{sid}")
    return {"kind": kind, "strategy_id": sid, "who": who, "action_match": am, **{k: r[k] for k in ("judge_pass", "judge_error", "cost")}}
with ThreadPoolExecutor(16) as ex:
    R = pd.DataFrame(list(ex.map(f, jobs)))
R["judge_model"] = JM
os.makedirs("results/judges", exist_ok=True); R.to_csv(f"results/judges/{TAG}.csv", index=False)
R["bad"] = R.action_match < 0.9
print(JM, "rows", len(R), "api_err", R.judge_error.notna().sum(), "parse_fail", (R.judge_pass.isna() & R.judge_error.isna()).sum(), "cost", round(R.cost.fillna(0).sum(), 2))
print(R[R.judge_pass.notna()].groupby(["kind", "who"]).apply(lambda g: pd.Series({
    "n": len(g), "judge_pass": g.judge_pass.astype(bool).mean(), "pass&bad": (g.judge_pass.astype(bool) & g.bad).sum(),
    "fail&bad": (~g.judge_pass.astype(bool) & g.bad).sum(), "pass&ok": (g.judge_pass.astype(bool) & ~g.bad).sum(),
    "fail&ok": (~g.judge_pass.astype(bool) & ~g.bad).sum()}), include_groups=False).round(3).to_string())
