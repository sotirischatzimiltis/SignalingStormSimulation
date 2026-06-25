"""
Pluggable LLM backends for the Reasoner seam (agents/base/reason.py).

Three providers, all via plain httpx calls (no extra SDKs required, httpx is
already a dependency):
  - ollama:    local daemon, http://localhost:11434 by default. No API key.
  - anthropic: https://api.anthropic.com/v1/messages. Needs ANTHROPIC_API_KEY.
  - openai:    https://api.openai.com/v1/chat/completions. Needs OPENAI_API_KEY.

LLMReasoner wraps a provider call with a prompt builder and a JSON-output
parser; on ANY failure (provider unreachable, timeout, malformed JSON), it
falls back to the same deterministic stub decision_fn the system used before
LLMs were wired in -- a research demo of *resilience* orchestration should
not itself crash because a local model wasn't running.

Cold-start note: loading a model into Ollama for the first time can take
~15-20s (observed with llama3.2:latest on this machine); subsequent calls
are ~1-4s. Call `warmup()` once at agent startup (before serving) so that
one-time cost isn't charged against the first real decision during an
episode.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx

from .reason import Reasoner

Messages = list  # list[dict] of {"role": str, "content": str}


@dataclass
class ProviderResponse:
    text: str
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


ProviderCall = Callable[[Messages], Awaitable[ProviderResponse]]


@dataclass
class LLMConfig:
    provider: str  # "stub" | "ollama" | "anthropic" | "openai"
    model: str = ""
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""
    timeout_s: float = 60.0


async def _ollama_call(cfg: LLMConfig, messages: Messages) -> ProviderResponse:
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        resp = await client.post(
            f"{cfg.base_url}/api/chat",
            json={"model": cfg.model, "messages": messages, "stream": False, "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return ProviderResponse(
            text=data["message"]["content"],
            tokens_in=data.get("prompt_eval_count"),
            tokens_out=data.get("eval_count"),
        )


async def _anthropic_call(cfg: LLMConfig, messages: Messages) -> ProviderResponse:
    api_key = os.environ[cfg.api_key_env or "ANTHROPIC_API_KEY"]
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_messages = [m for m in messages if m["role"] != "system"]
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={"model": cfg.model, "max_tokens": 256, "system": system, "messages": user_messages},
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return ProviderResponse(text=text, tokens_in=usage.get("input_tokens"), tokens_out=usage.get("output_tokens"))


async def _openai_call(cfg: LLMConfig, messages: Messages) -> ProviderResponse:
    api_key = os.environ[cfg.api_key_env or "OPENAI_API_KEY"]
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
            json={
                "model": cfg.model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return ProviderResponse(
            text=data["choices"][0]["message"]["content"],
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
        )


_PROVIDERS = {"ollama": _ollama_call, "anthropic": _anthropic_call, "openai": _openai_call}


def _extract_json(text: str) -> Optional[dict]:
    """Models occasionally wrap JSON in prose or code fences despite
    instructions; take the largest {...} span as a best-effort fallback."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class LLMReasoner(Reasoner):
    def __init__(
        self,
        cfg: LLMConfig,
        prompt_fn: Callable[[dict], str],
        fallback_fn: Callable[[dict], dict],
        system_prompt: str = "",
        recorder=None,  # InstrumentedRecorder, optional -- logs full prompt/response/tokens if given
    ):
        self.cfg = cfg
        self.prompt_fn = prompt_fn
        self.fallback_fn = fallback_fn
        self.system_prompt = system_prompt
        self.recorder = recorder
        self._call = _PROVIDERS[cfg.provider]

    def _build_messages(self, user_content: str) -> Messages:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    async def warmup(self) -> None:
        try:
            await self._call(self.cfg, self._build_messages("Reply with only: {\"ok\": true}"))
        except Exception:  # noqa: BLE001 - best-effort; real failures surface on first real call
            pass

    async def _log(self, user_prompt: str, response_text, decision, tokens_in, tokens_out,
                    latency_s, source, t_start, error=None) -> None:
        print(
            f"[llm] {self.cfg.provider}:{self.cfg.model} tokens_in={tokens_in} "
            f"tokens_out={tokens_out} latency={latency_s:.2f}s source={source}"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        if self.recorder is not None:
            await self.recorder.record_llm(
                provider=self.cfg.provider,
                model=self.cfg.model,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                response_text=response_text,
                decision=decision,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_s=latency_s,
                source=source,
                t_start=t_start,
                error=error,
            )

    async def reason(self, context: dict) -> dict:
        t_start = time.monotonic()
        user_prompt = self.prompt_fn(context)
        try:
            resp = await self._call(self.cfg, self._build_messages(user_prompt))
            latency_s = time.monotonic() - t_start
            parsed = _extract_json(resp.text)
            if isinstance(parsed, dict):
                source = f"llm:{self.cfg.provider}:{self.cfg.model}"
                parsed["_source"] = source
                await self._log(user_prompt, resp.text, parsed, resp.tokens_in, resp.tokens_out,
                                 latency_s, source, t_start)
                return parsed
            decision = self.fallback_fn(context)
            decision["_source"] = "fallback"
            decision["_error"] = "unparseable_llm_output"
            await self._log(user_prompt, resp.text, decision, resp.tokens_in, resp.tokens_out,
                             latency_s, "fallback", t_start, error="unparseable_llm_output")
            return decision
        except Exception as exc:  # noqa: BLE001 - provider down/timeout/bad output -> fall back
            latency_s = time.monotonic() - t_start
            decision = self.fallback_fn(context)
            decision["_source"] = "fallback"
            decision["_error"] = str(exc)
            await self._log(user_prompt, None, decision, None, None, latency_s, "fallback", t_start, error=str(exc))
            return decision


def build_reasoner(
    llm_provider: str,
    llm_model: str,
    llm_base_url: str,
    prompt_fn: Callable[[dict], str],
    fallback_fn: Callable[[dict], dict],
    system_prompt: str = "",
    stub_latency_range_s=(0.1, 0.5),
    seed: Optional[int] = None,
    recorder=None,  # InstrumentedRecorder, optional -- see LLMReasoner
) -> Reasoner:
    """Factory used by every agent's __main__.py. provider="stub" preserves
    the original randomized-latency/deterministic-decision behavior for fast
    testing without any model running (and has nothing to log -- no real
    provider call is ever made)."""
    if llm_provider == "stub":
        from .reason import StubReasoner

        return StubReasoner(fallback_fn, latency_range_s=stub_latency_range_s, seed=seed)
    cfg = LLMConfig(provider=llm_provider, model=llm_model, base_url=llm_base_url)
    return LLMReasoner(cfg, prompt_fn, fallback_fn, system_prompt=system_prompt, recorder=recorder)
