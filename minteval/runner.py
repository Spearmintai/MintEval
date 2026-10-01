"""Query the models under test (open + closed settings). All responses are cached on disk."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import client_from_cfg
from .prompts import STEP2_TASK, closed_prompt_step1, extract_code, extract_json, open_prompt


def generate_one(client, q: str, setting: str, gcfg: dict, sid: str) -> dict:
    kw = dict(temperature=gcfg["temperature"], max_tokens=gcfg["max_tokens"])
    if setting == "open":
        msgs = [{"role": "user", "content": open_prompt(q)}]
        out = client.chat(msgs, tag=f"open-{sid}", **kw)
        return {"code": _code(out), "spec": None, "raw": [out]}
    m1 = [{"role": "user", "content": closed_prompt_step1(q)}]
    o1 = client.chat(m1, tag=f"closed1-{sid}", **kw)
    spec = extract_json(o1.get("text"))
    m2 = m1 + [{"role": "assistant", "content": o1.get("text") or ""},
               {"role": "user", "content": STEP2_TASK}]
    o2 = client.chat(m2, tag=f"closed2-{sid}", **kw)
    return {"code": _code(o2), "spec": spec, "raw": [o1, o2]}


def _code(out):
    """None -> scored as compile failure; truncation and API errors are labelled distinctly."""
    code = extract_code(out.get("text"))
    if out.get("error"):
        return "# API_ERROR\n" if code is None else code
    if code is None and out.get("finish_reason") == "length":
        return "# TRUNCATED\n"
    return code


def run_model(mcfg: dict, prompts: dict, setting: str, gcfg: dict, out_dir="results/generations"):
    client = client_from_cfg(mcfg)
    workers = gcfg["workers"].get(mcfg.get("kind", "openrouter"), 8)
    sids = sorted(prompts)

    def f(sid):
        return sid, generate_one(client, prompts[sid], setting, gcfg, sid)
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(f, sids))
    p = Path(out_dir) / f"{mcfg['name']}__{setting}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for sid in sids:
            r = res[sid]
            fh.write(json.dumps({"strategy_id": sid, "model": mcfg["name"], "setting": setting,
                                 "code": r["code"], "spec": r["spec"],
                                 "api_error": next((o.get("error") for o in r["raw"] if o.get("error")), None),
                                 "finish": [o.get("finish_reason") for o in r["raw"]],
                                 "usage": [o.get("usage") for o in r["raw"]]}) + "\n")
    return res
