"""Test that the power_down action flows through the game API and UI path.

Covers:
1. HumanAction(mode="power_down") produces the correct tool call.
2. power_down with no units returns None (dropped).
3. The engine honours a power_down command (power_provided changes).
"""

from __future__ import annotations

import pytest

from openra_bench.human_labeling import HumanAction


def test_power_down_human_action_to_tool_call():
    action = HumanAction(mode="power_down", units=["42"])
    tc = action.to_tool_call()
    assert tc is not None
    assert tc["name"] == "power_down"
    assert tc["arguments"]["unit_ids"] == ["42"]


def test_power_down_human_action_no_units_returns_none():
    action = HumanAction(mode="power_down", units=[])
    assert action.to_tool_call() is None


pytest.importorskip("openra_train", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: power-down-ui-test
      title: 'Power down UI test'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'power management toggle'
    base_map: rush-hour-arena
    starting_cash: 5000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, power_down]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 1000}
      actors:
        - {type: powr, owner: agent, position: [20, 20]}
        - {type: proc, owner: agent, position: [24, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'power down test scenario'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      medium:
        description: 'power down test scenario'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      hard:
        description: 'power down test scenario'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
    """
)


def _make_pack():
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="power_down_test_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_power_down_command_changes_power_stats():
    """Sending power_down on a powr building reduces power_provided."""
    pack_path = _make_pack()
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        scen_path = _scenario_to_tmp_yaml(compiled)
    except Exception:
        pack_path.unlink(missing_ok=True)
        raise
    pool = RustEnvPool(size=1, scenario_path=scen_path)
    env = pool.acquire()
    try:
        from openra_train import Command

        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        rs0 = ad.render_state()
        p0 = rs0.get("power_provided", 0)
        assert p0 > 0, "Pre-placed powr should provide power"

        powr = next(b for b in rs0["own_buildings"] if b["type"] == "powr")
        obs, _, _, _ = env.step([Command.power_down([powr["id"]])])
        ad.observe(obs)
        rs1 = ad.render_state()
        p1 = rs1.get("power_provided", 0)
        assert p1 < p0, (
            f"power_provided should decrease after power_down: {p0} -> {p1}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
