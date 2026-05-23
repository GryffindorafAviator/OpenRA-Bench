"""End-to-end guardrail: `Command.capture_actor` lets an engineer (e6)
walk to an enemy building (proc) and capture it. On arrival the
target's owner flips to the agent player AND the engineer is consumed.

Mirror of the Rust engine acceptance suite
(`OpenRA-Rust/openra-sim/tests/test_capture.rs`) — exercises the
same loop via the Python `OpenRAEnv` boundary so the bench-side
`Command.capture_actor` shim is pinned.

NOTE: this test depends on the wheel having been rebuilt after the
engineer-capture engine commit (the new `Command.capture_actor`
staticmethod was added in that commit). Until the wheel is rebuilt
the test is SKIPPED, not failed, so a stale environment doesn't
mask other test results.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: engineer-capture-test
      title: 'Engineer captures enemy proc (engine smoke)'
      capability: action
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'agent reassigns ownership of a target asset'
    base_map: rush-hour-arena
    starting_cash: 0
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, capture_actor]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 6000}
      actors:
        - {type: e6, owner: agent, position: [16, 20]}
        - {type: proc, owner: enemy, position: [22, 20]}
    levels:
      easy:
        description: 'engineer walks ~6 cells east and captures the proc'
        starting_cash: 0
        win_condition: {has_building: proc}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
      medium:
        description: 'placeholder'
        starting_cash: 0
        win_condition: {has_building: proc}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
      hard:
        description: 'placeholder'
        starting_cash: 0
        win_condition: {has_building: proc}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="engineer_capture_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_engineer_captures_enemy_proc_via_command_capture_actor():
    from openra_train import Command

    if not hasattr(Command, "capture_actor"):
        pytest.skip(
            "Command.capture_actor not available — rebuild the wheel "
            "(maturin develop --release) after the engineer-capture "
            "engine commit."
        )

    from openra_bench.rust_adapter import RustObsAdapter

    pack_path = _pack_tmp()
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
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        render0 = ad.render_state()

        # Locate the engineer (own_units) and the enemy proc (enemy_buildings or
        # enemy actors surfaced in the snapshot).
        own_units = render0.get("own_units", []) or []
        e6s = [u for u in own_units if str(u.get("type", "")).lower() == "e6"]
        assert len(e6s) == 1, f"expected exactly one e6 in own_units, got {own_units}"
        engineer_id = str(e6s[0]["id"])

        # The enemy proc surfaces via `enemy_buildings` / actor enumeration. We
        # accept either presentation — locate it by type==proc and enemy ownership.
        enemy_buildings = (
            render0.get("enemy_buildings")
            or render0.get("enemies", [])
            or []
        )
        proc_id = None
        for b in enemy_buildings:
            if str(b.get("type", "")).lower() == "proc":
                proc_id = str(b.get("id"))
                break
        if proc_id is None:
            # Fall back to the full actors dump if available.
            for a in render0.get("actors", []) or []:
                if (
                    str(a.get("type", "")).lower() == "proc"
                    and str(a.get("owner", "")) not in ("agent", str(render0.get("agent_player_id", "")))
                ):
                    proc_id = str(a.get("id"))
                    break
        assert proc_id is not None, (
            "could not find enemy proc id in initial observation "
            f"(render0 keys={list(render0.keys())})"
        )

        # Issue capture and step until the win predicate (has_building: proc)
        # flips — i.e. the proc now belongs to the agent.
        for _ in range(40):
            obs, _r, done, _i = env.step(
                [Command.capture_actor([engineer_id], proc_id)]
            )
            ad.observe(obs)
            rs = ad.render_state()
            own_b = {
                str(b.get("type", "")).lower()
                for b in (rs.get("own_buildings", []) or [])
            }
            if "proc" in own_b or done:
                break

        own_b_final = {
            str(b.get("type", "")).lower()
            for b in (ad.render_state().get("own_buildings", []) or [])
        }
        assert "proc" in own_b_final, (
            "Command.capture_actor should have transferred the enemy proc "
            f"to the agent within the timeout (own_buildings={own_b_final})"
        )

        # Engineer must be consumed (no e6 left in own_units).
        own_units_final = ad.render_state().get("own_units", []) or []
        live_e6 = [
            u for u in own_units_final if str(u.get("type", "")).lower() == "e6"
        ]
        assert not live_e6, (
            "engineer should be consumed after a successful capture; still "
            f"alive: {live_e6}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
