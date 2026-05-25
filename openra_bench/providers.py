"""Provider abstraction for the eval agent.

One small interface, `ChatProvider.complete(messages, tools) -> ChatReply`.
Adapters:

* `OpenAICompatibleProvider` — OpenAI Chat Completions wire format. Covers
  local **vLLM** (matches Training's rollout path) and **OpenRouter**
  (the Phase-0 test target) by base_url alone.
* `BedrockProvider` — AWS Bedrock Converse. Translates the agent's
  OpenAI-shape messages + tool schemas to Bedrock Converse and back to
  the same `ChatReply` the OpenAI path returns, so the agent stays
  provider-agnostic. Auth comes from the AWS credential chain (env /
  shared config / role) — never hardcoded.

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
    # AWS Bedrock — auth via the boto3 credential chain (env, shared
    # config, instance/role). `base_url` is unused (the SDK derives the
    # endpoint from the region). `api_key_env` is unused (left for
    # interface parity); `bedrock_region` on ProviderConfig wins.
    "bedrock": {
        "base_url": "",
        "api_key_env": "",
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
    # AWS Bedrock: inference region. Sonnet 4.6 is exposed via the
    # `us.anthropic.claude-sonnet-4-6` cross-region inference profile,
    # which routes from `us-west-2` (the on-demand model id returns
    # ValidationException — only the inference profile is callable).
    bedrock_region: str = "us-west-2"

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
        # Audit hook: when set (a list), every successful complete()
        # appends a dict {"request": <body>, "response": <raw>} so the
        # FullPlayback recorder can capture the literal wire payloads.
        # Drained by the caller after each turn. None disables capture.
        self.request_log: list[dict] | None = None

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
        # OpenAI's newer "reasoning" / GPT-5+ model families require
        # `max_completion_tokens` instead of `max_tokens` and reject the
        # legacy parameter outright. Sniff the model id for the known
        # forward-dated families (gpt-5, gpt-5-mini, gpt-5-nano,
        # gpt-5.x-... — every model with a `gpt-5` prefix today
        # requires the new param). Same code path is used by openrouter
        # but those models don't currently include the gpt-5 family
        # under their own slug, so the prefix check is safe.
        _model_l = (cfg.model or "").lower()
        _needs_completion_tokens = (
            _model_l.startswith("gpt-5")
            or _model_l.startswith("o1")
            or _model_l.startswith("o3")
            or _model_l.startswith("o4")
        )
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": self._wire_messages(messages),
            "temperature": cfg.temperature,
        }
        if _needs_completion_tokens:
            body["max_completion_tokens"] = cfg.max_tokens
        else:
            body["max_tokens"] = cfg.max_tokens
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
        if self.request_log is not None:
            # Audit capture: redact the bearer header (the body is the
            # interesting part) and store the literal request + raw
            # response side-by-side. FullPlayback drains after each turn.
            try:
                self.request_log.append(
                    {
                        "request": {
                            "url": url,
                            "body": body,
                        },
                        "response": {
                            "raw": reply.raw,
                            "text": reply.text,
                            "tool_calls": reply.tool_calls,
                            "reasoning": reply.reasoning,
                            "usage": dict(reply.usage or {}),
                            "finish_reason": (
                                (reply.raw.get("choices") or [{}])[0]
                                .get("finish_reason")
                                if isinstance(reply.raw, dict)
                                else None
                            ),
                        },
                    }
                )
            except Exception:  # noqa: BLE001 — audit must never break a run
                pass
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
            elif "tool_calls" in wm:
                # Strict endpoints (Together's Qwen3.6-Plus, some vLLM
                # builds) reject an assistant message that carries
                # `tool_calls: []` — "Empty tool_calls is not supported
                # in message." A plain-text assistant turn must omit
                # the key entirely.
                wm.pop("tool_calls", None)
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
    """AWS Bedrock Converse adapter.

    Translates between the agent's OpenAI-shape messages + tool
    schemas and the Bedrock Converse wire format, and translates the
    response back to a `ChatReply` so the agent and FullPlayback see
    the SAME shape they get from the OpenAI-compatible path. Auth
    flows through boto3's standard credential chain — env vars, the
    shared config file, IAM role, etc. The model id is the inference
    profile id (`us.anthropic.claude-sonnet-4-6`), not the on-demand
    model id (which returns ValidationException).

    Wire-shape mapping:
      * OpenAI `system` messages         → top-level `system: [{text}]`
      * OpenAI text user/assistant       → `content: [{text}]`
      * OpenAI multimodal user content   → `content: [{text}, {image}]`
      * OpenAI assistant `tool_calls`    → `content: [{toolUse}]`
      * OpenAI `tool` reply              → user `[{toolResult}]`
      * OpenAI `tools` (JSON-Schema)     → `toolConfig: {tools: [{toolSpec}]}`
      * Bedrock `output.message.content` → ChatReply.text + tool_calls
      * Bedrock `usage.{input,output}Tokens` → usage.{prompt,completion}_tokens

    Tool-call ids: Bedrock requires a `toolUseId` on every assistant
    `toolUse` and the matching user `toolResult`. The bench agent
    canonicalises these as `c0/c1/...` per turn, so the translation
    passes them straight through.
    """

    def __init__(self, cfg: ProviderConfig, *, rate_limiter=None,
                 cost_meter=None, client=None):
        self.cfg = cfg
        self.model_id = cfg.model
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
        # Lazy import: keep boto3 a soft dep — only providers='bedrock'
        # forces the dependency, never the OpenRouter / vLLM paths.
        if client is not None:
            self._client = client
        else:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover — env-dep
                raise RuntimeError(
                    "BedrockProvider needs boto3. Install with "
                    "`pip install boto3`."
                ) from e
            self._client = boto3.client(
                "bedrock-runtime", region_name=cfg.bedrock_region
            )
        # Audit hook (parallels OpenAICompatibleProvider): when set to
        # a list, every successful complete() appends a record so
        # FullPlayback can capture literal request + raw response.
        self.request_log: list[dict] | None = None

    @property
    def cost_meter(self):
        return self._cost

    # ── Wire translation: OpenAI → Bedrock ──────────────────────────

    @staticmethod
    def _to_bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Pure: split OpenAI messages into (system, conversation).

        System messages are concatenated into a list of `{text}` blocks
        for Bedrock's top-level `system` parameter. Tool replies
        (`role=tool`) become user-role `toolResult` content blocks; an
        assistant message with `tool_calls` becomes Bedrock `toolUse`
        content blocks (text content, if any, is preserved alongside).
        Adjacent same-role messages are merged because Bedrock REQUIRES
        strictly alternating user/assistant turns — a `tool` reply
        followed by another user briefing must collapse into ONE
        Bedrock user message with multiple content blocks.
        """
        sys_blocks: list[dict] = []
        out: list[dict] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                txt = m.get("content")
                if isinstance(txt, list):
                    txt = "\n".join(
                        p.get("text", "") for p in txt
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                if txt:
                    sys_blocks.append({"text": str(txt)})
                continue
            blocks = BedrockProvider._content_to_blocks(m)
            if not blocks:
                continue
            br_role = "user" if role in ("user", "tool") else "assistant"
            if out and out[-1]["role"] == br_role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": br_role, "content": blocks})
        return sys_blocks, out

    @staticmethod
    def _content_to_blocks(msg: dict) -> list[dict]:
        """Pure: OpenAI message → list of Bedrock content blocks."""
        role = msg.get("role")
        # Tool-result reply → toolResult block.
        if role == "tool":
            tcid = msg.get("tool_call_id") or ""
            content = msg.get("content")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return [{
                "toolResult": {
                    "toolUseId": str(tcid),
                    "content": [{"text": str(content) if content else "ok"}],
                }
            }]
        blocks: list[dict] = []
        c = msg.get("content")
        if isinstance(c, str):
            if c:
                blocks.append({"text": c})
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                t = part.get("type")
                if t == "text":
                    txt = part.get("text", "")
                    if txt:
                        blocks.append({"text": txt})
                elif t == "image_url":
                    iu = part.get("image_url") or {}
                    url = iu.get("url", "") if isinstance(iu, dict) else ""
                    img = BedrockProvider._image_block_from_data_url(url)
                    if img is not None:
                        blocks.append(img)
        # Assistant tool_calls → toolUse blocks (after any text).
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            blocks.append({
                "toolUse": {
                    "toolUseId": str(tc.get("id") or ""),
                    "name": fn.get("name", ""),
                    "input": args,
                }
            })
        return blocks

    @staticmethod
    def _image_block_from_data_url(url: str) -> dict | None:
        """Pure: turn a `data:image/png;base64,...` URL into a Bedrock
        `{image: {format, source: {bytes}}}` block. Bedrock accepts
        png / jpeg / gif / webp; the bench only emits png minimaps."""
        import base64

        if not url.startswith("data:"):
            return None
        try:
            header, b64 = url.split(",", 1)
        except ValueError:
            return None
        fmt = "png"
        if "image/" in header:
            mt = header.split("image/", 1)[1].split(";", 1)[0].lower()
            if mt in ("png", "jpeg", "jpg", "gif", "webp"):
                fmt = "jpeg" if mt == "jpg" else mt
        try:
            raw = base64.b64decode(b64)
        except (ValueError, TypeError):
            return None
        return {"image": {"format": fmt, "source": {"bytes": raw}}}

    @staticmethod
    def _to_bedrock_tools(tools: list[dict]) -> dict | None:
        """Pure: OpenAI tool list → Bedrock `toolConfig`. The OpenAI
        schema is `{type: "function", function: {name, description,
        parameters}}`; Bedrock wants `{toolSpec: {name, description,
        inputSchema: {json: <parameters>}}}`. Bedrock additionally
        requires `inputSchema.json.type` (some agents emit empty
        params) — we backfill an empty object schema."""
        if not tools:
            return None
        specs = []
        for t in tools:
            fn = t.get("function") or {}
            params = fn.get("parameters") or {"type": "object", "properties": {}}
            if "type" not in params:
                params = {"type": "object", **params}
            specs.append({
                "toolSpec": {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "inputSchema": {"json": params},
                }
            })
        return {"tools": specs}

    # ── Wire translation: Bedrock → ChatReply ───────────────────────

    @staticmethod
    def _reply_from_bedrock(resp: dict) -> ChatReply:
        """Pure: parse a Bedrock Converse response into a ChatReply.

        Bedrock emits one assistant message; its content blocks are
        either `{text}` (plain reply) or `{toolUse}` (a function call).
        We concatenate text blocks and lift toolUse blocks into the
        same `[{name, arguments}]` list the OpenAI parser produces."""
        msg = (resp.get("output") or {}).get("message") or {}
        content_blocks = msg.get("content") or []
        text_parts: list[str] = []
        calls: list[dict] = []
        reasoning_parts: list[str] = []
        for blk in content_blocks:
            if not isinstance(blk, dict):
                continue
            if "text" in blk:
                text_parts.append(blk["text"])
            elif "toolUse" in blk:
                tu = blk["toolUse"]
                calls.append({
                    "name": tu.get("name", ""),
                    "arguments": tu.get("input") or {},
                })
            elif "reasoningContent" in blk:
                # Bedrock surfaces extended thinking under
                # reasoningContent.{reasoningText: {text}} — preserve
                # it on the reply for FullPlayback.
                rc = blk["reasoningContent"] or {}
                rt = rc.get("reasoningText") or {}
                t = rt.get("text") if isinstance(rt, dict) else None
                if t:
                    reasoning_parts.append(str(t))
        usage = resp.get("usage") or {}
        return ChatReply(
            text="".join(text_parts),
            tool_calls=calls,
            reasoning="".join(reasoning_parts),
            usage={
                "prompt_tokens": int(usage.get("inputTokens", 0) or 0),
                "completion_tokens": int(usage.get("outputTokens", 0) or 0),
            },
            raw=resp,
        )

    # ── Public API ──────────────────────────────────────────────────

    def _converse_once(self, system_blocks, br_messages, tool_config,
                       inference_cfg) -> dict:
        from .resilience import FatalProviderError
        try:
            kwargs = {
                "modelId": self.model_id,
                "messages": br_messages,
                "inferenceConfig": inference_cfg,
            }
            if system_blocks:
                kwargs["system"] = system_blocks
            if tool_config:
                kwargs["toolConfig"] = tool_config
            return self._client.converse(**kwargs)
        except Exception as e:  # noqa: BLE001
            # Boto raises ClientError with a `response[Error][Code]`.
            code = ""
            status = 0
            try:
                err = getattr(e, "response", {}) or {}
                meta = err.get("ResponseMetadata") or {}
                status = int(meta.get("HTTPStatusCode", 0) or 0)
                code = (err.get("Error") or {}).get("Code", "")
            except Exception:  # noqa: BLE001
                pass
            transient = status in (408, 425, 429, 500, 502, 503, 504) or code in (
                "ThrottlingException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
                "InternalServerException",
                "ModelStreamErrorException",
            )
            cls = RuntimeError if transient else FatalProviderError
            new = cls(f"bedrock {code or status or 'error'}: {e}")
            new.transient = transient  # type: ignore[attr-defined]
            new.retry_after = None  # type: ignore[attr-defined]
            raise new from e

    def complete(self, messages: list[dict], tools: list[dict]) -> ChatReply:
        from .resilience import retry_call

        cfg = self.cfg
        sys_blocks, br_messages = self._to_bedrock_messages(messages)
        tool_config = self._to_bedrock_tools(tools)
        inference_cfg = {
            "temperature": cfg.temperature,
            "maxTokens": cfg.max_tokens,
        }

        self._rl.acquire()
        resp = retry_call(
            lambda: self._converse_once(
                sys_blocks, br_messages, tool_config, inference_cfg,
            ),
            self._policy,
        )
        reply = self._reply_from_bedrock(resp)
        u = reply.usage or {}
        self._cost.add(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        self._cost.check()
        if self.request_log is not None:
            try:
                # Redact image bytes from the request log (they're
                # huge, and duplicated per turn). Replace with a
                # short placeholder; the rest of the body is small.
                def _redact(b):
                    if isinstance(b, dict):
                        return {k: _redact(v) for k, v in b.items()}
                    if isinstance(b, list):
                        return [_redact(x) for x in b]
                    if isinstance(b, (bytes, bytearray)):
                        return f"<bytes:{len(b)}>"
                    return b

                self.request_log.append({
                    "request": {
                        "model": self.model_id,
                        "system": _redact(sys_blocks),
                        "messages": _redact(br_messages),
                        "toolConfig": tool_config,
                        "inferenceConfig": inference_cfg,
                    },
                    "response": {
                        "raw": _redact(reply.raw),
                        "text": reply.text,
                        "tool_calls": reply.tool_calls,
                        "reasoning": reply.reasoning,
                        "usage": dict(reply.usage or {}),
                        "finish_reason": resp.get("stopReason"),
                    },
                })
            except Exception:  # noqa: BLE001 — audit must never break a run
                pass
        return reply

    def close(self) -> None:  # noqa: D401 — interface parity
        # boto3 clients don't need explicit close; provided for
        # symmetry with OpenAICompatibleProvider.
        pass


def make_provider(cfg: ProviderConfig, *, rate_limiter=None,
                  cost_meter=None) -> ChatProvider:
    if cfg.provider == "bedrock":
        return BedrockProvider(
            cfg, rate_limiter=rate_limiter, cost_meter=cost_meter,
        )
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
