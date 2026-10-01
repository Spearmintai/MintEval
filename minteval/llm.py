"""Minimal OpenAI-compatible chat client (OpenRouter / vLLM) with an on-disk cache.

Cache key = sha256(endpoint-kind, model, messages, sampling params). Re-running the
pipeline with the same config therefore replays identical generations (rule 2).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

CACHE = Path(os.environ.get("MINTEVAL_LLM_CACHE", "results/cache/llm"))


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, kind: str = "openrouter",
                 extra: dict | None = None, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.kind = kind
        self.extra = extra or {}
        self.timeout = timeout

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096,
             seed: int | None = 0, use_cache: bool = True, retries: int = 6, tag: str = "") -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, **self.extra}
        if seed is not None:
            payload["seed"] = seed
        key = _key({"kind": self.kind, "tag": tag, **payload})
        path = CACHE / key[:2] / f"{key}.json"
        if use_cache and path.exists():
            return json.loads(path.read_text())
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last = None
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers,
                                  data=json.dumps(payload), timeout=self.timeout)
                if r.status_code == 200:
                    d = r.json()
                    if "choices" in d and d["choices"]:
                        msg = d["choices"][0]["message"]
                        out = {"text": msg.get("content") or "",
                               "reasoning": msg.get("reasoning"),
                               "finish_reason": d["choices"][0].get("finish_reason"),
                               "usage": d.get("usage"), "model_returned": d.get("model"),
                               "provider": d.get("provider"), "time": time.time()}
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(out))
                        return out
                    last = f"bad body: {str(d)[:300]}"
                else:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code in (400, 401, 403, 404):
                        if r.status_code != 400 or attempt >= 1:
                            break
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
            time.sleep(min(60, 2 ** attempt))
        return {"text": None, "error": last}


def client_from_cfg(m: dict) -> ChatClient:
    kind = m.get("kind", "openrouter")
    if kind == "openrouter":
        return ChatClient("https://openrouter.ai/api/v1", m["model"], os.environ.get("OPENROUTER_API_KEY"),
                          kind, m.get("extra"))
    # local vLLM in batch-invariant mode can take >300 s for an 8k-token answer under load
    return ChatClient(m["base_url"], m["model"], m.get("api_key", "EMPTY"), kind, m.get("extra"), timeout=3600)
