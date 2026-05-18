"""Provider abstraction for the eval agent.

One small interface, `ChatProvider.complete(messages, tools) -> ChatReply`.
Adapters:

* `OpenAICompatibleProvider` — OpenAI Chat Completions wire format. Covers
  local **vLLM** (matches Training's rollout path) and **OpenRouter**
  (the Phase-0 test target) by base_url alone.
* `BedrockProvider` — AWS Bedrock Converse. Stubbed with a precise
  NotImplementedError so the wiring exists before the dependency does.

Selection is pure config (`ProviderConfig`); no provider-specific code
leaks into the agent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

ProviderName = Literal["openai", "vllm", "openrouter", "bedrock"]

# Convenience presets; base_url/api_key_env still overridable in config.
_PRESETS: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "vllm": {
        "base_url": "http://localhost:8100/v1",
        "api_key_env": "VLLM_API_KEY",  # vLLM ignores the value
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
}


@dataclass
class ProviderConfig:
    provider: ProviderName = "openrouter"
    model: str = "anthropic/claude-3.5-sonnet"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 120.0
    vision: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        preset = _PRESETS.get(self.provider)
        if not preset:
            raise ValueError(
                f"no base_url and no preset for provider {self.provider!r}"
            )
        return preset["base_url"]

    def resolved_api_key(self) -> str:
        env = self.api_key_env or _PRESETS.get(self.provider, {}).get("api_key_env")
        if not env:
            raise ValueError(f"no api_key_env for provider {self.provider!r}")
        key = os.environ.get(env, "")
        if not key and self.provider != "vllm":
            raise RuntimeError(
                f"{env} not set — required for provider {self.provider!r}"
            )
        return key or "not-needed"


@dataclass
class ChatReply:
    """Normalized model reply."""

    text: str
    tool_calls: list[dict]  # [{"name": str, "arguments": dict}]
    reasoning: str = ""  # chain-of-thought, when the model/provider emits it
    raw: dict = field(default_factory=dict)


class ChatProvider:
    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        raise NotImplementedError


class OpenAICompatibleProvider(ChatProvider):
    """OpenAI /chat/completions with `tools`. vLLM + OpenRouter + OpenAI."""

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        cfg = self.cfg
        headers = {
            "Authorization": f"Bearer {cfg.resolved_api_key()}",
            "Content-Type": "application/json",
            **cfg.extra_headers,
        }
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": self._wire_messages(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        resp = self._client.post(
            f"{cfg.resolved_base_url()}/chat/completions",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        return self._reply_from_data(resp.json())

    # Keys the OpenAI Chat Completions wire format accepts per message.
    # `history` carries extra playback-only keys (notably "reasoning");
    # those must never be posted back or strict servers (vLLM) 400.
    _WIRE_KEYS = frozenset(
        {"role", "content", "name", "tool_calls", "tool_call_id"}
    )

    @staticmethod
    def _wire_messages(messages: list[dict]) -> list[dict]:
        """Pure: project each message onto OpenAI-legal keys only."""
        return [
            {k: v for k, v in m.items() if k in OpenAICompatibleProvider._WIRE_KEYS}
            for m in messages
        ]

    @staticmethod
    def _reply_from_data(data: dict) -> ChatReply:
        """Pure parse of a Chat Completions response, including the
        provider-specific reasoning channel (vLLM/DeepSeek emit
        `reasoning_content`; OpenRouter/others a flat `reasoning`)."""
        msg = data["choices"][0]["message"]
        calls: list[dict] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        rc = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if isinstance(rc, list):  # some providers chunk it
            rc = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in rc
            )
        return ChatReply(
            text=msg.get("content") or "",
            tool_calls=calls,
            reasoning=str(rc),
            raw=data,
        )

    def close(self) -> None:
        self._client.close()


class BedrockProvider(ChatProvider):
    """AWS Bedrock Converse. Wired but not yet implemented."""

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        raise NotImplementedError(
            "BedrockProvider not implemented yet. Use provider='openrouter' "
            "or 'vllm' for Phase 0; Bedrock Converse adapter is a tracked "
            "follow-up (needs boto3 + message/tool shape translation)."
        )


def make_provider(cfg: ProviderConfig) -> ChatProvider:
    if cfg.provider == "bedrock":
        return BedrockProvider(cfg)
    if cfg.provider in ("openai", "vllm", "openrouter"):
        return OpenAICompatibleProvider(cfg)
    raise ValueError(f"unknown provider {cfg.provider!r}")
