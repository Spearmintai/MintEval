"""Complexity measures.

Execution-side (measured by instrumenting the reference program's backtest):
  tau_max / tau_p90  over all cross-bar reads of live keys (t_read - t_last_write > 0),
                     pooled across keys
  n_registers (R)    number of keys with at least one cross-bar read
Static (on source): nesting_depth (max nested if/for/while/try/with depth inside
functions), McCabe (radon, max over functions + sum), Halstead volume (radon).
Generation-side: K_bits (see assembler.k_bits).
"""
from __future__ import annotations

import ast

import numpy as np


def state_complexity(spans: dict) -> dict:
    allv = [s for v in spans.values() for s in v]
    if not allv:
        return {"tau_max": 0, "tau_p90": 0.0, "n_registers": 0}
    a = np.asarray(allv)
    return {"tau_max": int(a.max()), "tau_p90": float(np.percentile(a, 90)),
            "n_registers": int(sum(1 for v in spans.values() if v))}


_NEST = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.IfExp)


def nesting_depth(src: str) -> int:
    tree = ast.parse(src)

    def depth(node, d):
        best = d
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, _NEST):
                # elif chains are represented as nested If in orelse: do not count them
                best = max(best, depth(ch, d + 1))
            else:
                best = max(best, depth(ch, d))
        return best

    def depth_if_aware(node, d):
        best = d
        for field, value in ast.iter_fields(node):
            items = value if isinstance(value, list) else [value]
            for ch in items:
                if not isinstance(ch, ast.AST):
                    continue
                is_elif = (isinstance(node, ast.If) and field == "orelse" and isinstance(ch, ast.If)
                           and len(node.orelse) == 1)
                if isinstance(ch, _NEST) and not is_elif:
                    best = max(best, depth_if_aware(ch, d + 1))
                else:
                    best = max(best, depth_if_aware(ch, d))
        return best

    return max((depth_if_aware(f, 0) for f in tree.body if isinstance(f, ast.FunctionDef)), default=0)


def static_complexity(src: str) -> dict:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit
    blocks = cc_visit(src)
    mccabe = max((b.complexity for b in blocks), default=1)
    hv = h_visit(src).total.volume
    return {"mccabe": int(mccabe), "halstead": float(hv), "nesting_depth": nesting_depth(src)}
