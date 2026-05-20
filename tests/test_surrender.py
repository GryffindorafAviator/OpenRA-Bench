"""S7: surrender — agent concedes → scenario ends as a LOSS; tool
schema stays 1:1 congruent with the engine command set."""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios.schema import ScenarioPack

_META = {
    "id": "surrender-selftest",
    "title": "Surrender Self Test",
    "capability": "reasoning",
    "real_world_meaning": "concede when the position is unrecoverable",
    "robotics_analogue": "abort mission on unrecoverable state",
    "author": "ci",
}
_BASE = {
    "agent": {"faction": "allies"},
    "enemy": {"faction": "soviet"},
    "tools": ["move_units", "stop", "surrender"],
    "actors": [
        {"type": "jeep", "owner": "agent", "position": [6, 6], "count": 2},
        {"type": "e1", "owner": "enemy", "position": [110, 30], "stance": 2},
    ],
    "termination": {"max_ticks": 40000},
}


def _pack():
    lvl = {
        "description": "unreachable win so outcome is decided by play",
        "win_condition": {"explored_pct_gte": 999},
        "max_turns": 25,
    }
    return ScenarioPack(
        meta=_META, base_map="rush-hour-arena", base=_BASE,
        levels={"easy": lvl, "medium": lvl, "hard": lvl},
    )


def test_surrender_yields_loss_and_ends_fast():
    c = _pack().compile("easy", map_supported=True)
    res = run_level(c, lambda rs, C: [C.surrender()], seed=1)
    assert res.outcome == "loss", f"surrender must be a loss, got {res.outcome}"
    assert res.turns == 1, "surrender ends the episode immediately"
    assert res.signals.outcome == 0.0


def test_no_surrender_is_not_a_loss():
    c = _pack().compile("easy", map_supported=True)
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    # Unreachable win + idle on a no-progress run → draw (not loss).
    assert res.outcome == "draw"


def test_tool_schema_still_1to1_with_engine():
    from openra_bench.agent import _TOOL_SCHEMAS
    import openra_train as ot

    tools = set(_TOOL_SCHEMAS)
    cmds = {m for m in dir(ot.Command) if not m.startswith("_")}
    assert "surrender" in tools and "surrender" in cmds
    assert tools == cmds, f"tool/command mismatch: {tools ^ cmds}"
