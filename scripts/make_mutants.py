"""Graded negative control: inject one known error into reference programs (~30 per type).
Each mutation must change the source text exactly once (asserted). Output: results/mutants/mutants.jsonl"""
import sys, json, copy, re
sys.path.insert(0, ".")
import numpy as np
from minteval.assembler import render
from minteval.primitives import REGISTRY
from minteval.sandbox import check_source
rng = np.random.default_rng(20261003)
tasks = [json.loads(l) for l in open("results/tasks/tasks.jsonl")]
Q = {json.loads(l)["strategy_id"]: json.loads(l)["q"] for l in open("results/tasks/prompts.jsonl")}
def once(src, old, new):
    assert src.count(old) >= 1, old
    i = src.index(old); return src[:i] + new + src[i + len(old):]
def m_stop_anchor(t):
    if not any(r["id"] == "fixed_stop" for r in t["spec"]["risk"]): return None
    k = next(r["params"]["k"] for r in t["spec"]["risk"] if r["id"] == "fixed_stop")
    old = f"stop = tighter_stop(stop, ref - s * {k!r} * ra, s)"
    return once(t["source"], old, f"stop = tighter_stop(stop, price - s * {k!r} * ra, s)") if old in t["source"] else None
def m_window_incl(t):
    if t["spec"]["signal"]["id"] not in ("donchian_break", "box_breakout", "breakout_pullback"): return None
    s = t["source"]; m = re.search(r"ind\.highest\(high, \d+\)\[-2\]", s)
    if not m: return None
    s = once(s, m.group(0), m.group(0).replace("[-2]", "[-1]"))
    m2 = re.search(r"ind\.lowest\(low, \d+\)\[-2\]", s)
    return once(s, m2.group(0), m2.group(0).replace("[-2]", "[-1]"))   # same window error on both sides
def m_cooldown_reset(t):
    if not any(r["id"] == "cooldown" for r in t["spec"]["risk"]): return None
    old = '        if t < state["cool_until"]:'
    s = t["source"]
    if old not in s: return None
    return once(s, '    if "side" in state and state["side"] != side:',
                '    if "cool_until" in state:\n        del state["cool_until"]\n    if "side" in state and state["side"] != side:')
def m_param_step(t):
    sp = copy.deepcopy(t["spec"]); slots = [sp["signal"], sp["sizing"]] + sp["risk"]
    cands = [(sl, n) for sl in slots for n, g in REGISTRY[sl["id"]].params.items() if len(g) > 1 and not isinstance(g[0], bool)]
    sl, n = cands[int(rng.integers(len(cands)))]; g = REGISTRY[sl["id"]].params[n]
    cur = tuple(sl["params"][n]) if isinstance(sl["params"][n], list) else sl["params"][n]
    i = [tuple(x) if isinstance(x, tuple) else x for x in g].index(cur)
    nv = g[i + 1] if i + 1 < len(g) else g[i - 1]
    sl["params"][n] = list(nv) if isinstance(nv, tuple) else nv
    out = render(sp); return out if out != t["source"] else None
def m_cmp_reverse(t):
    # only comparisons inside `if` conditions (a reversed `while` bound would loop forever: not a realistic bug)
    s = t["source"]
    for line in s.splitlines():
        if line.lstrip().startswith("if ") and ") >= " in line and ("state[\"ref\"]" in line or "state[\"last_add\"]" in line):
            return once(s, line, line.replace(") >= ", ") <= ", 1))
    return None
TYPES = {"stop_anchor": m_stop_anchor, "window_incl": m_window_incl, "cooldown_reset": m_cooldown_reset,
         "param_step": m_param_step, "cmp_reverse": m_cmp_reverse}
out = []
for name, fn in TYPES.items():
    order = rng.permutation(len(tasks)); got = 0
    for i in order:
        t = tasks[i]; src = fn(t)
        if src is None or src == t["source"]: continue
        check_source(src)
        out.append({"mutant_id": f"{name}-{t['strategy_id']}", "type": name, "strategy_id": t["strategy_id"],
                    "q": Q[t["strategy_id"]], "code": src}); got += 1
        if got == 30: break
    print(name, got)
import os; os.makedirs("results/mutants", exist_ok=True)
open("results/mutants/mutants.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in out))
