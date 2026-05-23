"""OpenAI-compatible message wire formatting — the strict-endpoint
edge cases the bench has hit in production."""

from __future__ import annotations

from openra_bench.providers import OpenAICompatibleProvider


def test_wire_strips_playback_only_keys():
    """`reasoning` is playback-only and would 400 vLLM if posted back."""
    out = OpenAICompatibleProvider._wire_messages([
        {"role": "assistant", "content": "x", "reasoning": "thoughts",
         "tool_calls": [{"id": "c0", "type": "function",
                         "function": {"name": "observe", "arguments": {}}}]},
    ])
    assert "reasoning" not in out[0]
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_wire_strips_empty_tool_calls_from_assistant_text_turns():
    """Together's Qwen3.6-Plus rejects an assistant message that
    carries `tool_calls: []` ("Empty tool_calls is not supported in
    message"). Plain-text assistant turns must omit the key."""
    out = OpenAICompatibleProvider._wire_messages([
        {"role": "assistant", "content": "thinking out loud",
         "tool_calls": []},
    ])
    assert "tool_calls" not in out[0]
    assert out[0]["content"] == "thinking out loud"


def test_wire_keeps_non_empty_tool_calls_and_stringifies_args():
    out = OpenAICompatibleProvider._wire_messages([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c0", "type": "function",
            "function": {"name": "move_units",
                         "arguments": {"unit_ids": [1004]}},
        }]},
    ])
    tcs = out[0]["tool_calls"]
    assert len(tcs) == 1
    # args coerced to JSON string per the wire spec
    assert isinstance(tcs[0]["function"]["arguments"], str)
    assert "1004" in tcs[0]["function"]["arguments"]
