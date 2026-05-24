"""Bedrock Converse adapter — wire-shape translation in both directions.

These tests exercise the pure translation helpers (no AWS, no
network). The end-to-end smoke test against a real `us-west-2`
inference profile lives in `docs/BEDROCK_SMOKE.md` — runnable but
not part of CI (avoids non-zero AWS charges in the default suite).
"""

from __future__ import annotations

import base64

from openra_bench.providers import (
    BedrockProvider,
    ChatReply,
    ProviderConfig,
)


def _png_data_url() -> str:
    """A 1x1 transparent PNG, sufficient to exercise the data-url path."""
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAj"
        "CB0C8AAAAASUVORK5CYII="
    )
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ── Outbound: OpenAI messages → Bedrock Converse ───────────────────────


def test_system_messages_lift_to_top_level():
    sys, conv = BedrockProvider._to_bedrock_messages([
        {"role": "system", "content": "you are a commander"},
        {"role": "user", "content": "hi"},
    ])
    assert sys == [{"text": "you are a commander"}]
    assert conv == [{"role": "user", "content": [{"text": "hi"}]}]


def test_multimodal_user_message_lifts_image_block():
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "MAP TURN 1"},
            {"type": "image_url",
             "image_url": {"url": _png_data_url()}},
        ],
    }
    _, conv = BedrockProvider._to_bedrock_messages([msg])
    assert len(conv) == 1
    blocks = conv[0]["content"]
    assert blocks[0] == {"text": "MAP TURN 1"}
    assert "image" in blocks[1]
    assert blocks[1]["image"]["format"] == "png"
    assert isinstance(blocks[1]["image"]["source"]["bytes"], (bytes, bytearray))


def test_assistant_tool_calls_become_toolUse_blocks():
    msg = {
        "role": "assistant",
        "content": "moving",
        "tool_calls": [{
            "id": "c0", "type": "function",
            "function": {
                "name": "move_units",
                "arguments": {"unit_ids": [1004], "target_x": 50, "target_y": 50},
            },
        }],
    }
    _, conv = BedrockProvider._to_bedrock_messages([msg])
    blocks = conv[0]["content"]
    assert conv[0]["role"] == "assistant"
    assert blocks[0] == {"text": "moving"}
    tu = blocks[1]["toolUse"]
    assert tu["toolUseId"] == "c0"
    assert tu["name"] == "move_units"
    assert tu["input"] == {"unit_ids": [1004], "target_x": 50, "target_y": 50}


def test_assistant_tool_calls_with_string_arguments_decoded():
    """The OpenAI wire spec stores `arguments` as a JSON STRING; the
    Bedrock toolUse `input` MUST be a dict."""
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "c0", "type": "function",
            "function": {"name": "observe", "arguments": '{}'},
        }],
    }
    _, conv = BedrockProvider._to_bedrock_messages([msg])
    tu = conv[0]["content"][0]["toolUse"]
    assert tu["input"] == {}


def test_tool_reply_becomes_user_toolResult_block():
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{
             "id": "c0", "type": "function",
             "function": {"name": "observe", "arguments": {}},
         }]},
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
    ]
    _, conv = BedrockProvider._to_bedrock_messages(msgs)
    # 3 turns: user, assistant, user(toolResult).
    assert [m["role"] for m in conv] == ["user", "assistant", "user"]
    tr = conv[2]["content"][0]["toolResult"]
    assert tr["toolUseId"] == "c0"
    assert tr["content"] == [{"text": "ok"}]


def test_consecutive_user_messages_merge_to_satisfy_alternation():
    """Bedrock REQUIRES alternating user/assistant turns. After a
    tool reply (which becomes a user message) the next briefing is
    also a user message — they must collapse into one user turn."""
    msgs = [
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
        {"role": "user", "content": "next briefing"},
    ]
    _, conv = BedrockProvider._to_bedrock_messages(msgs)
    assert len(conv) == 1
    assert conv[0]["role"] == "user"
    # toolResult + text under one user message
    assert "toolResult" in conv[0]["content"][0]
    assert conv[0]["content"][1] == {"text": "next briefing"}


# ── Outbound: tool schemas → toolConfig ────────────────────────────────


def test_tool_schema_translation():
    tools = [{
        "type": "function",
        "function": {
            "name": "move_units",
            "description": "move units to a cell",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array",
                                 "items": {"type": "integer"}},
                    "target_x": {"type": "integer"},
                    "target_y": {"type": "integer"},
                },
                "required": ["unit_ids", "target_x", "target_y"],
            },
        },
    }]
    cfg = BedrockProvider._to_bedrock_tools(tools)
    assert cfg is not None
    spec = cfg["tools"][0]["toolSpec"]
    assert spec["name"] == "move_units"
    assert spec["description"] == "move units to a cell"
    assert spec["inputSchema"]["json"]["type"] == "object"
    assert "unit_ids" in spec["inputSchema"]["json"]["properties"]


def test_tool_schema_backfills_object_type_for_paramless_tool():
    tools = [{"type": "function", "function": {
        "name": "observe", "description": "noop",
        "parameters": {"properties": {}},
    }}]
    cfg = BedrockProvider._to_bedrock_tools(tools)
    spec = cfg["tools"][0]["toolSpec"]
    assert spec["inputSchema"]["json"]["type"] == "object"


def test_no_tools_returns_none():
    assert BedrockProvider._to_bedrock_tools([]) is None


# ── Inbound: Bedrock response → ChatReply ──────────────────────────────


def test_reply_from_bedrock_text_only():
    resp = {
        "output": {"message": {"role": "assistant",
                                "content": [{"text": "hello there"}]}},
        "usage": {"inputTokens": 11, "outputTokens": 3, "totalTokens": 14},
        "stopReason": "end_turn",
    }
    reply = BedrockProvider._reply_from_bedrock(resp)
    assert isinstance(reply, ChatReply)
    assert reply.text == "hello there"
    assert reply.tool_calls == []
    assert reply.usage == {"prompt_tokens": 11, "completion_tokens": 3}


def test_reply_from_bedrock_tool_use():
    resp = {
        "output": {"message": {"role": "assistant", "content": [
            {"text": "I'll scout."},
            {"toolUse": {
                "toolUseId": "tooluse_abc",
                "name": "move_units",
                "input": {"unit_ids": [1004], "target_x": 60, "target_y": 60},
            }},
        ]}},
        "usage": {"inputTokens": 200, "outputTokens": 25},
        "stopReason": "tool_use",
    }
    reply = BedrockProvider._reply_from_bedrock(resp)
    assert reply.text == "I'll scout."
    assert reply.tool_calls == [{
        "name": "move_units",
        "arguments": {"unit_ids": [1004], "target_x": 60, "target_y": 60},
    }]
    assert reply.usage["prompt_tokens"] == 200


def test_reply_from_bedrock_reasoning_block():
    resp = {
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {
                "reasoningText": {"text": "thinking…", "signature": "s"},
            }},
            {"text": "answer"},
        ]}},
        "usage": {"inputTokens": 5, "outputTokens": 2},
    }
    reply = BedrockProvider._reply_from_bedrock(resp)
    assert reply.text == "answer"
    assert reply.reasoning == "thinking…"


# ── Plumbing: make_provider routes bedrock through BedrockProvider ─────


def test_make_provider_routes_to_bedrock():
    from openra_bench.providers import make_provider

    cfg = ProviderConfig(
        provider="bedrock",
        model="us.anthropic.claude-sonnet-4-6",
    )

    class _StubClient:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"role": "assistant",
                                        "content": [{"text": "ack"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
                "stopReason": "end_turn",
            }

    # Bypass make_provider's lazy boto3 import by constructing directly
    # with a stub client; smoke that complete() round-trips.
    p = BedrockProvider(cfg, client=_StubClient())
    reply = p.complete(
        [{"role": "system", "content": "hi"},
         {"role": "user", "content": "ping"}],
        tools=[],
    )
    assert reply.text == "ack"
    assert reply.usage == {"prompt_tokens": 1, "completion_tokens": 1}
