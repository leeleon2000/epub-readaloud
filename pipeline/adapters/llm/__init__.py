"""LLM adapters. Every call takes chat messages and a JSON schema and returns a
parsed JSON object. Failures are typed so stages can decide what to do:

  LLMTruncated   output hit max_tokens twice (a runaway loop or too much to say)
  LLMInvalid     output was not valid JSON after retries
  LLMUnavailable connection / timeout / 5xx after backoff retries (service down?)
  LLMError       anything else (e.g. HTTP 4xx: a bad request; do not retry)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLMTruncated(LLMError):
    """The model hit max_tokens: the JSON is incomplete."""


class LLMInvalid(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMAdapter(Protocol):
    name: str
    obs: object | None

    def complete_json(self, messages: list[dict], schema: dict, task: str = "",
                      payload: dict | None = None, max_tokens: int | None = None,
                      truncation_retries: int = 1) -> dict: ...


class OpenAICompatLLM:
    """OpenAI-compatible chat endpoint (vLLM) with structured outputs (json_schema)."""
    name = "openai_compat"

    def __init__(self, cfg: dict, retries: int = 3):
        self.endpoint = cfg.get("endpoint", "http://localhost:8000/v1").rstrip("/")
        self.model = cfg.get("model", "qwen3.6")
        self.thinking = bool(cfg.get("thinking", False))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.seed = cfg.get("seed")          # fixed seed for measurement runs (docs/DESIGN.md)
        self.max_tokens = int(cfg.get("max_tokens", 8192))
        self.timeout = int(cfg.get("timeout", 600))
        self.retries = int(cfg.get("retries", retries))
        self.backoff = float(cfg.get("backoff_seconds", 2.0))
        self.obs = None                  # Observer, set by the stage
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "truncated": 0}

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.endpoint + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _record(self, task, body, resp, t0, finish, attempt, result, error=""):
        self.stats["calls"] += 1
        self.stats["seconds"] += time.time() - t0
        u = (resp or {}).get("usage") or {}
        self.stats["prompt_tokens"] += u.get("prompt_tokens") or 0
        self.stats["completion_tokens"] += u.get("completion_tokens") or 0
        if self.obs is not None:
            from ...memory.prompts import PROMPT_VERSION
            self.obs.llm_call(task, PROMPT_VERSION, self.model, body, resp, time.time() - t0, finish,
                              attempt, result, error)

    def complete_json(self, messages: list[dict], schema: dict, task: str = "",
                      payload: dict | None = None, max_tokens: int | None = None,
                      truncation_retries: int = 1) -> dict:
        """truncation_retries: retries (with a repetition penalty) after hitting max_tokens before raising
        LLMTruncated. 0 for callers that split the input instead (a retry can "succeed" by closing the
        JSON early and silently dropping most of the answer)."""
        limit = min(max_tokens or self.max_tokens, self.max_tokens)
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": limit,
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": task or "result", "schema": schema, "strict": True}},
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.seed is not None:
            body["seed"] = int(self.seed)
        down_tries = invalid = truncated = attempt = 0
        while True:
            t0 = time.time()
            try:
                resp = self._post(body)
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode(errors="replace")
                if e.code < 500:
                    self._record(task, body, None, t0, None, attempt, "error", f"HTTP {e.code}: {detail}")
                    raise LLMError(f"{task}: HTTP {e.code}: {detail}") from e
                err = f"HTTP {e.code}: {detail}"
            except (OSError, TimeoutError) as e:      # URLError, connection refused, timeout
                err = f"{type(e).__name__}: {e}"
            else:
                choice = resp["choices"][0]
                finish = choice.get("finish_reason")
                content = choice["message"].get("content") or ""
                if finish == "length":
                    truncated += 1
                    self.stats["truncated"] += 1
                    self._record(task, body, resp, t0, finish, attempt, "truncated")
                    if truncated > truncation_retries:
                        raise LLMTruncated(f"{task}: output hit max_tokens={limit}"
                                           + (" twice" if truncated > 1 else ""))
                    body.update(repetition_penalty=1.1, seed=1000 + attempt)   # break a repetition loop
                    attempt += 1
                    continue
                try:
                    out = json.loads(content)
                except json.JSONDecodeError as e:
                    invalid += 1
                    self._record(task, body, resp, t0, finish, attempt, "invalid", str(e))
                    if invalid >= 3:
                        raise LLMInvalid(f"{task}: invalid JSON 3 times: {e}") from e
                    attempt += 1
                    continue
                self._record(task, body, resp, t0, finish, attempt, "ok")
                return out
            # connection / timeout / 5xx: exponential backoff
            self._record(task, body, None, t0, None, attempt, "error", err)
            down_tries += 1
            if down_tries > self.retries:
                raise LLMUnavailable(f"{task}: {self.endpoint} unavailable after {self.retries} retries: {err}")
            time.sleep(min(self.backoff * 2 ** (down_tries - 1), 60))
            attempt += 1


def get_llm(cfg: dict, monitor: dict | None = None) -> LLMAdapter:
    name = cfg.get("adapter", "openai_compat")
    if name == "mock":
        from .mock import MockLLM
        return MockLLM(cfg)
    if name == "openai_compat":
        return OpenAICompatLLM(cfg, retries=int((monitor or {}).get("llm_retries", 3)))
    raise ValueError(f"unknown LLM adapter: {name}")
