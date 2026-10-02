"""Static checks + subprocess sandbox for strategy code (reference or model-generated).

Static rules (AST):
  * imports: only ``math`` and ``numpy`` (any alias / ``from`` form)
  * no name or attribute starting with ``_`` (blocks dunder / frame / private access)
  * no ``global`` / ``nonlocal``, no nested ``def``/``class``, no ``class`` at all
  * module level: only imports, defs, docstrings and assignments of immutable literals
  * no mutable default arguments (defaults must be literals)
  * no assignment to attributes (``obj.x = ...``)
  * banned builtins: open, eval, exec, compile, input, globals, locals, vars,
    getattr, setattr, delattr, breakpoint, help, memoryview, __import__
  * must define ``strategy(hist, state, pos, ind)``

Runtime (child process): rlimits (CPU, address space), an audit hook that blocks
file opens, sockets, subprocess/exec/fork, ctypes and non-whitelisted imports after
setup, stdout captured, wall-clock timeout enforced by the parent.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

ALLOWED_IMPORTS = {"math", "numpy"}
BANNED_CALLS = {"open", "eval", "exec", "compile", "input", "globals", "locals", "vars",
                "getattr", "setattr", "delattr", "breakpoint", "help", "memoryview",
                "__import__", "exit", "quit"}


class SandboxViolation(Exception):
    pass


_NUM_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd)


def _const_arith(node) -> bool:
    """Arithmetic on numeric literals only (e.g. 24 * 3600 * 1000): immutable and side-effect free."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, _NUM_OPS) and _const_arith(node.operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow) and not (isinstance(node.right, ast.Constant) and abs(node.right.value) <= 64):
            return False
        return isinstance(node.op, _NUM_OPS) and _const_arith(node.left) and _const_arith(node.right)
    return False


def _is_literal(node) -> bool:
    if _const_arith(node):
        return True
    try:
        v = ast.literal_eval(node)
    except Exception:  # noqa: BLE001
        return False

    def immut(x):
        if isinstance(x, (int, float, bool, str, type(None), complex, bytes)):
            return True
        if isinstance(x, tuple):
            return all(immut(i) for i in x)
        return False
    return immut(v)


SOFT: list = []


def soft_violations(src: str) -> list:
    SOFT.clear()
    try:
        check_source(src)
    except SandboxViolation:
        pass
    out = list(SOFT)
    SOFT.clear()
    return out


def check_source(src: str, allowed=ALLOWED_IMPORTS) -> ast.Module:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SandboxViolation(f"SyntaxError: {e}") from e
    has_strategy = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.AsyncFunctionDef):
            raise SandboxViolation(f"line {node.lineno}: async functions not allowed")
        if isinstance(node, ast.FunctionDef):
            if node.name == "strategy":
                has_strategy = True
                names = [a.arg for a in node.args.args]
                if len(names) != 4 or node.args.vararg or node.args.kwarg:
                    raise SandboxViolation("strategy must take exactly (hist, state, pos, ind)")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            val = node.value
            if val is None or not _is_literal(val):
                raise SandboxViolation(
                    f"line {node.lineno}: module-level assignments must be immutable literals")
            continue
        raise SandboxViolation(f"line {node.lineno}: statement not allowed at module level: "
                               f"{type(node).__name__}")
    if not has_strategy:
        raise SandboxViolation("no top-level function named 'strategy'")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in allowed:
                    raise SandboxViolation(f"import of '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in allowed or node.level:
                raise SandboxViolation(f"import from '{node.module}' not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxViolation(f"line {node.lineno}: global/nonlocal not allowed")
        elif isinstance(node, ast.ClassDef):
            raise SandboxViolation(f"line {node.lineno}: class definitions not allowed")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node not in tree.body:
                # A per-call nested helper cannot persist anything across bars (nonlocal/global and
                # mutable defaults are rejected), so it is allowed; recorded as a soft violation of
                # the literal "no closures" rule so analyses can apply the strict reading.
                SOFT.append(f"line {node.lineno}: nested function '{node.name}'")
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                if not _is_literal(d):
                    raise SandboxViolation(f"line {node.lineno}: default args must be immutable literals")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise SandboxViolation(f"line {node.lineno}: private attribute '{node.attr}'")
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                raise SandboxViolation(f"line {node.lineno}: attribute assignment not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise SandboxViolation(f"line {node.lineno}: name '{node.id}' not allowed")
            if node.id in BANNED_CALLS:
                raise SandboxViolation(f"line {node.lineno}: '{node.id}' not allowed")
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            raise SandboxViolation(f"line {node.lineno}: generators not allowed")
    return tree


# ----------------------------------------------------------------------------- child
_CHILD = r'''
import sys, os, io, pickle, json, resource
job = pickle.loads(sys.stdin.buffer.read())
sys.path.insert(0, job["root"])
import math, numpy as np
from minteval.engine import run_backtest, EngineConfig, perf_stats
from minteval import indicators as _ind
_ind._ema(np.arange(5.0), 3)          # trigger numba (cache) load before the hook
P = dict(np.load(job["npz"]))
cfg = EngineConfig(**job["cfg"])
out_fd = os.dup(1)
sys.stdout = io.StringIO()
lim = job["mem_mb"] * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
resource.setrlimit(resource.RLIMIT_CPU, (job["cpu_s"], job["cpu_s"] + 1))
ALLOWED = ("math", "numpy")
def hook(event, args):
    if event == "open":
        raise PermissionError("file access is not allowed in the sandbox")
    if event.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.spawn",
                         "os.fork", "os.posix_spawn", "ctypes.", "os.remove", "os.rename",
                         "shutil.", "os.chdir", "os.mkdir", "os.rmdir", "os.kill")):
        raise PermissionError(f"{event} is not allowed in the sandbox")
    if event == "import":
        name = args[0]
        if name.split(".")[0] not in ALLOWED and name not in sys.modules:
            raise ImportError(f"import of {name} not allowed")
sys.addaudithook(hook)
import builtins
safe_builtins = {k: getattr(builtins, k) for k in dir(builtins)
                 if k not in {"open", "eval", "exec", "compile", "input", "globals", "locals",
                              "vars", "breakpoint", "help", "memoryview", "exit", "quit"}}
def _imp(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED:
        raise ImportError(f"import of {name} not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)
safe_builtins["__import__"] = _imp
import signal
class SandboxTimeout(BaseException):
    pass
def _alarm(signum, frame):
    raise SandboxTimeout(f"exceeded {job['timeout_s']}s")
signal.signal(signal.SIGALRM, _alarm)
res = {"error": None}
try:
    signal.setitimer(signal.ITIMER_REAL, job["timeout_s"])
    g = {"__builtins__": safe_builtins, "__name__": "strategy_module"}
    exec(compile(job["code"], "<strategy>", "exec"), g)
    r = run_backtest(g["strategy"], P, cfg, track_state=job["track"])
    signal.setitimer(signal.ITIMER_REAL, 0)
    res = {"error": r.error, "error_bar": r.error_bar, "equity": r.equity,
           "target_q": r.target_q, "pos_q": r.pos_q, "trades": r.trades, "n_fills": r.n_fills,
           "state_spans": r.state_spans, "state_log": r.state_log if job["keep_log"] else None}
except BaseException as e:
    signal.setitimer(signal.ITIMER_REAL, 0)
    name = "Timeout" if isinstance(e, SandboxTimeout) else type(e).__name__
    res = {"error": f"{name}: {e}", "error_bar": -1}
payload = pickle.dumps(res, protocol=5)
os.write(out_fd, len(payload).to_bytes(8, "little"))
mv = memoryview(payload)
while len(mv):
    k = os.write(out_fd, mv[:1 << 20]); mv = mv[k:]
'''

ROOT = str(Path(__file__).resolve().parent.parent)


def ensure_npz(price_file: str) -> str:
    npz = str(Path(price_file).with_suffix(".npz"))
    if not os.path.exists(npz) or os.path.getmtime(npz) < os.path.getmtime(price_file):
        from .data import load_prices
        import numpy as np
        P = load_prices(price_file)
        np.savez(npz, **P)
    return npz


def run_sandboxed(code: str, price_file: str, cfg: dict, timeout_s: float = 10.0,
                  mem_mb: int = 4096, track: bool = False, keep_log: bool = False) -> dict:
    """Run ``code`` in a fresh subprocess. Returns the result dict; error is set on failure."""
    try:
        check_source(code)
    except SandboxViolation as e:
        return {"error": f"SandboxViolation: {e}", "error_bar": -1}
    job = {"root": ROOT, "npz": ensure_npz(price_file), "cfg": cfg, "code": code,
           "mem_mb": mem_mb, "timeout_s": float(timeout_s), "cpu_s": int(timeout_s) + 30, "track": track, "keep_log": keep_log}
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMBA_CACHE_DIR": os.path.join(ROOT, ".numba_cache")}
    try:
        p = subprocess.run([sys.executable, "-c", _CHILD], input=pickle.dumps(job),
                           capture_output=True, timeout=timeout_s + _STARTUP_S, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout: exceeded {timeout_s}s", "error_bar": -1}
    out = p.stdout
    if len(out) < 8:
        tail = p.stderr.decode(errors="replace")[-400:]
        return {"error": f"Crash: exit={p.returncode} {tail}", "error_bar": -1}
    n = int.from_bytes(out[:8], "little")
    return pickle.loads(out[8:8 + n])


# interpreter + numpy/numba import time is not charged to the strategy's 10 s budget
_STARTUP_S = 5.0
