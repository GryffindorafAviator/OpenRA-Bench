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

ProviderName = Literal["openai", "vllm", "openrouter", "bedrock", "together"]

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
    # together.ai — OpenAI-compatible API. Models are namespaced like
    # `Qwen/Qwen3.6-Plus`, `meta-llama/Llama-3.3-70B-Instruct-Turbo`,
    # etc. Native tool-calling supported on most chat models; check
    # https://docs.together.ai/docs/function-calling for per-model
    # gating before enabling tool use on a new model.
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
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
    # Spatial channel: "vision" = PNG minimap; "structured" = NO image,
    # a text "Unexplored regions" block instead (text-vs-vision A/B;
    # pair structured runs with the easy/medium level of the setup).
    fog_mode: str = "vision"
    # Minimap unit colours: "auto" = per-type palette on hard, constant
    # own/enemy colours on easy/medium; or force "per_type"/"constant".
    minimap_color_mode: str = "auto"
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Merged into the request JSON body — e.g. OpenRouter provider
    # routing to avoid the rate-limited free pool:
    #   extra_body={"provider": {"sort": "throughput",
    #                            "allow_fallbacks": True}}
    # (premium/paid routing also needs account credits).
    extra_body: dict = field(default_factory=dict)
    # Streaming: some models (notably together.ai's Qwen3.6-Plus and
    # other newer ones) refuse non-streaming requests with
    # `streaming_required` 400. Set True to send `"stream": true` and
    # accumulate the SSE chunks into a single ChatReply. The non-
    # streaming path is the default for backward compatibility.
    stream: bool = False
    # Resilience (real OpenRouter runs): bounded retry, throttle, price.
    max_retries: int = 5
    retry_base_s: float = 1.0
    retry_cap_s: float = 30.0
    qps: float = 0.0  # 0 = unthrottled; shared limiter set by evaluate
    max_history_turns: int = 16  # sliding wire-history window (0=unbounded)
    price_in_per_m: float = 0.0   # USD / 1M prompt tokens
    price_out_per_m: float = 0.0  # USD / 1M completion tokens

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
    usage: dict = field(default_factory=dict)  # prompt/completion tokens
    raw: dict = field(default_factory=dict)


class ChatProvider:
    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        raise NotImplementedError


class OpenAICompatibleProvider(ChatProvider):
    """OpenAI /chat/completions with `tools`. vLLM + OpenRouter + OpenAI."""

    def __init__(self, cfg: ProviderConfig, *, rate_limiter=None,
                 cost_meter=None):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)
        from .resilience import CostMeter, RateLimiter, RetryPolicy

        self._rl = rate_limiter or RateLimiter(cfg.qps)
        self._cost = cost_meter or CostMeter(
            cfg.price_in_per_m, cfg.price_out_per_m
        )
        self._policy = RetryPolicy(
            max_attempts=max(1, cfg.max_retries),
            base=cfg.retry_base_s,
            cap=cfg.retry_cap_s,
        )

    @property
    def cost_meter(self):
        return self._cost

    def _post_once(self, url, headers, body):
        from .resilience import FatalProviderError

        try:
            resp = self._client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as e:
            e.transient = True  # type: ignore[attr-defined]
            e.retry_after = None  # type: ignore[attr-defined]
            raise
        except httpx.TransportError as e:
            e.transient = True  # type: ignore[attr-defined]
            e.retry_after = None  # type: ignore[attr-defined]
            raise
        if resp.status_code >= 400:
            ra = resp.headers.get("retry-after")
            try:
                retry_after = float(ra) if ra is not None else None
            except ValueError:
                retry_after = None
            transient = self._policy.is_transient_status(resp.status_code)
            cls = RuntimeError if transient else FatalProviderError
            exc = cls(
                f"{resp.status_code} from provider: {resp.text[:800]}"
            )
            exc.transient = transient  # type: ignore[attr-defined]
            exc.retry_after = retry_after  # type: ignore[attr-defined]
            raise exc
        return resp

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        from .resilience import retry_call

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
        if cfg.extra_body:
            # e.g. OpenRouter {"provider": {...}} routing — premium/
            # paid endpoints instead of the rate-limited free pool.
            body.update(cfg.extra_body)
        url = f"{cfg.resolved_base_url()}/chat/completions"

        self._rl.acquire()
        if cfg.stream:
            body["stream"] = True
            # together.ai gates usage emission on this flag (otherwise
            # usage is null in streaming mode → cost-meter sees zero).
            body.setdefault(
                "stream_options", {"include_usage": True}
            )
            reply = retry_call(
                lambda: self._stream_once(url, headers, body), self._policy
            )
        else:
            resp = retry_call(
                lambda: self._post_once(url, headers, body), self._policy
            )
            reply = self._reply_from_data(resp.json())
        u = reply.usage or {}
        self._cost.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        self._cost.check()  # raises BudgetExceeded → evaluate finalizes
        return reply

    def _stream_once(self, url, headers, body) -> ChatReply:
        """Streaming POST: accumulate SSE chunks into a single ChatReply.

        OpenAI's stream format yields chunks with `choices[0].delta`
        containing partial `content`, partial `tool_calls`, and
        eventually `finish_reason`. Tool-call `function.arguments`
        arrives as a stream of JSON-string fragments that must be
        concatenated per `index`. The final chunk (with
        `stream_options: include_usage`) carries the usage dict.
        """
        from .resilience import FatalProviderError

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        # tool_calls accumulator keyed by delta.tool_calls[i].index
        # — providers stream calls in any interleaving; we re-assemble
        # by index, then materialise to a list in index order.
        tcs_acc: dict[int, dict[str, Any]] = {}
        usage: dict[str, int] | None = None
        finish_reason: str | None = None

        try:
            with self._client.stream(
                "POST", url, headers=headers, json=body
            ) as resp:
                if resp.status_code >= 400:
                    body_text = resp.read().decode("utf-8", errors="replace")
                    ra = resp.headers.get("retry-after")
                    try:
                        retry_after = float(ra) if ra is not None else None
                    except ValueError:
                        retry_after = None
                    transient = self._policy.is_transient_status(
                        resp.status_code
                    )
                    cls = RuntimeError if transient else FatalProviderError
                    exc = cls(
                        f"{resp.status_code} from provider: "
                        f"{body_text[:800]}"
                    )
                    exc.transient = transient  # type: ignore[attr-defined]
                    exc.retry_after = retry_after  # type: ignore[attr-defined]
                    raise exc
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].lstrip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content"):
                            content_parts.append(d["content"])
                        # vLLM / DeepSeek-style reasoning channel
                        rc = d.get("reasoning_content") or d.get("reasoning")
                        if rc:
                            reasoning_parts.append(
                                rc if isinstance(rc, str) else str(rc)
                            )
                        for tc in d.get("tool_calls") or []:
                            idx = tc.get("index", 0)
                            slot = tcs_acc.setdefault(
                                idx, {"id": "", "type": "function",
                                      "function": {"name": "",
                                                   "arguments": ""}}
                            )
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments") is not None:
                                slot["function"]["arguments"] += fn[
                                    "arguments"
                                ]
                        if ch.get("finish_reason"):
                            finish_reason = ch["finish_reason"]
        except httpx.TimeoutException as e:
            e.transient = True  # type: ignore[attr-defined]
            e.retry_after = None  # type: ignore[attr-defined]
            raise
        except httpx.TransportError as e:
            e.transient = True  # type: ignore[attr-defined]
            e.retry_after = None  # type: ignore[attr-defined]
            raise

        # Re-pack into the non-streaming response shape and re-use
        # the existing _reply_from_data parser (which already handles
        # tool_call argument JSON-decoding).
        tool_calls_list = [tcs_acc[i] for i in sorted(tcs_acc)]
        message: dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls_list:
            message["tool_calls"] = tool_calls_list
        if reasoning_parts:
            message["reasoning"] = "".join(reasoning_parts)
        data = {
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage,
        }
        return self._reply_from_data(data)

    # Keys the OpenAI Chat Completions wire format accepts per message.
    # `history` carries extra playback-only keys (notably "reasoning");
    # those must never be posted back or strict servers (vLLM) 400.
    _WIRE_KEYS = frozenset(
        {"role", "content", "name", "tool_calls", "tool_call_id"}
    )

    @staticmethod
    def _wire_messages(messages: list[dict]) -> list[dict]:
        """Pure: project each message onto OpenAI-legal keys only, and
        coerce `tool_calls[].function.arguments` to a JSON **string**
        (the wire spec requires a string; history keeps the dict for
        readable playback). Pure — inputs are not mutated."""
        out: list[dict] = []
        for m in messages:
            wm = {
                k: v for k, v in m.items()
                if k in OpenAICompatibleProvider._WIRE_KEYS
            }
            tcs = wm.get("tool_calls")
            if tcs:
                fixed = []
                for tc in tcs:
                    fn = dict(tc.get("function", {}))
                    args = fn.get("arguments", {})
                    if not isinstance(args, str):
                        fn["arguments"] = json.dumps(args)
                    fixed.append({**tc, "function": fn})
                wm["tool_calls"] = fixed
            out.append(wm)
        return out

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
        usage = data.get("usage") or {}
        return ChatReply(
            text=msg.get("content") or "",
            tool_calls=calls,
            reasoning=str(rc),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
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


def make_provider(cfg: ProviderConfig, *, rate_limiter=None,
                  cost_meter=None) -> ChatProvider:
    if cfg.provider == "bedrock":
        return BedrockProvider(cfg)
    if cfg.provider in ("openai", "vllm", "openrouter", "together"):
        # together.ai's newer Qwen3.x and Llama-3.x families gate on
        # streaming (`streaming_required` 400 in non-stream mode); flip
        # the default on for that provider so the SSE-accumulating path
        # in OpenAICompatibleProvider takes over. Users can still
        # force-disable via cfg.stream=False at construction.
        if cfg.provider == "together" and cfg.stream is False:
            cfg.stream = True
        return OpenAICompatibleProvider(
            cfg, rate_limiter=rate_limiter, cost_meter=cost_meter
        )
    raise ValueError(f"unknown provider {cfg.provider!r}")
