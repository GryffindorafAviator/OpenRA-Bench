"""Agent + provider tests.

Offline by default: a FakeProvider returns scripted tool calls so the
briefing build, tool-schema filtering, tool-call parsing, and the full
ModelAgent -> eval_core -> live-Rust loop are all asserted without a
network. The OpenRouter live test is opt-in (skips without an API key).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ot = pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.agent import ModelAgent, build_briefing, _to_commands, _tool_schemas
from openra_bench.providers import ChatProvider, ChatReply, ProviderConfig

TRAIN = Path("/Users/berta/Projects/OpenRA-RL-Training")
PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


class FakeProvider(ChatProvider):
    """Returns a fixed sequence of tool-call sets, then observe()."""

    def __init__(self, script: list[list[dict]]):
        self.script = script
        self.i = 0
        self.seen_messages: list[list[dict]] = []

    def complete(self, messages, tools):
        self.seen_messages.append(messages)
        calls = self.script[self.i] if self.i < len(self.script) else [
            {"name": "observe", "arguments": {}}
        ]
        self.i += 1
        return ChatReply(text="", tool_calls=calls)


def test_tool_schema_filtering():
    only_move = _tool_schemas(["move_units"])
    names = {t["function"]["name"] for t in only_move}
    assert "move_units" in names
    assert "attack_unit" not in names
    assert "observe" in names, "a no-op must always be offered"
    # Unset → DEFAULT_CORE (movement/combat), not every tool.
    core = {t["function"]["name"] for t in _tool_schemas(None)}
    assert {"move_units", "attack_unit", "observe"} <= core
    assert "build" not in core and "surrender" not in core
    # Wildcard exposes the full set (economy/structure/etc).
    allp = {t["function"]["name"] for t in _tool_schemas(["*"])}
    assert {"build", "harvest", "place_building", "stop", "deploy"} <= allp


def test_build_briefing_format():
    rs = {
        "game_tick": 120,
        "explored_percent": 12.5,
        "units_summary": [
            {"id": "1001", "type": "jeep", "cell_x": 5, "cell_y": 6, "activity": "idle"}
        ],
        "enemy_summary": [],
    }
    b = build_briefing(rs, objective="find the base")
    assert "OBJECTIVE: find the base" in b
    assert "1001 jeep @(5,6)" in b
    assert "tick=120" in b and "explored=12.5%" in b
    assert "none (scout the fog)" in b


def test_tool_call_parsing_and_aliases():
    cmds = _to_commands(
        [
            {"name": "move_units", "arguments": {"unit_ids": [1, 2], "target_x": 9, "target_y": 4}},
            {"name": "attack_target", "arguments": {"unit_ids": [3], "target_id": 77}},  # alias
            {"name": "observe", "arguments": {}},
            {"name": "garbage", "arguments": {}},  # dropped
            {"name": "move_units", "arguments": {"unit_ids": [1]}},  # malformed -> dropped
        ],
        ot.Command,
    )
    # 3 valid (move, attack-alias, observe); 2 dropped
    assert len(cmds) == 3


def test_model_agent_drives_live_rust_with_fake_provider():
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    pack = load_pack(PACKS / "perception-frontier-reading.yaml")
    compiled = compile_level(pack, "easy")

    # Scripted "scout east" behaviour, then observe forever.
    fake = FakeProvider(
        [[{"name": "move_units", "arguments": {"unit_ids": [1001], "target_x": 100, "target_y": 20}}]]
        * 8
    )
    agent = ModelAgent(
        ProviderConfig(vision=False),
        allowed_tools=compiled.scenario.tools,
        objective=compiled.scenario.description,
        provider=fake,
    )
    res = run_level(compiled, agent.agent_fn, seed=1)

    assert res.outcome in {"win", "draw", "loss"}
    assert res.turns >= 1 and len(res.trace) == res.turns
    assert agent.stats["turns"] == res.turns
    # Provider actually saw a system prompt + a user briefing.
    first = fake.seen_messages[0]
    assert first[0]["role"] == "system"
    assert any(m["role"] == "user" for m in first)


def test_history_strips_stale_images():
    hist = [
        {"role": "user", "content": [{"type": "text", "text": "t1"}, {"type": "image_url", "image_url": {}}]},
        {"role": "user", "content": [{"type": "text", "text": "t2"}, {"type": "image_url", "image_url": {}}]},
    ]
    ModelAgent._strip_old_images(hist)
    assert isinstance(hist[0]["content"], str) and hist[0]["content"] == "t1"
    assert isinstance(hist[1]["content"], list)  # newest image kept


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="set OPENROUTER_API_KEY to run the live provider test",
)
def test_openrouter_live_smoke():
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    pack = load_pack(PACKS / "perception-frontier-reading.yaml")
    compiled = compile_level(pack, "easy")
    agent = ModelAgent(
        ProviderConfig(
            provider="openrouter",
            model=os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet"),
            vision=False,
            max_tokens=512,
        ),
        allowed_tools=compiled.scenario.tools,
        objective=compiled.scenario.description,
    )
    res = run_level(compiled, agent.agent_fn, seed=1)
    assert res.outcome in {"win", "draw", "loss"}
    assert agent.stats["tool_calls"] >= 1, "model issued no usable tool calls"
