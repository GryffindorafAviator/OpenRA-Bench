"""End-to-end guardrail for the per-actor `health:` scenario field.

A scenario may pre-place a damaged actor with `health: N` (an HP
percentage, 1-100). Before the engine fix the Rust scenario parser
(`oramap.rs::RawScenarioActor` / `ScenarioActor`) parsed only
`actor_type / owner / position / count / spawn_point / stance` — the
`health:` line was silently dropped, so a `proc` placed with
`health: 40` spawned at full HP.

This test loads a tiny scenario with a `proc` at `health: 40`,
resets the env, and asserts the building surfaces at ~40% HP in the
Python observation. Mirror of the Rust parsing test
`OpenRA-Rust/openra-data/tests/test_actor_health.rs`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: actor-health-test
      title: 'Actor health field smoke'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'sensor surfaces pre-damaged actor state'
    base_map: rush-hour-arena
    starting_cash: 1000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 1000}
      actors:
        - {type: proc, owner: agent, position: [20, 20], health: 40}
        - {type: fact, owner: agent, position: [24, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'pre-placed proc at 40% HP, fact at full'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      medium:
        description: 'pre-placed proc at 40% HP, fact at full'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      hard:
        description: 'pre-placed proc at 40% HP, fact at full'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="actor_health_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_pre_placed_proc_health_40_surfaces_at_40_percent():
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
        own = {
            str(b.get("type", "")).lower(): b
            for b in render0.get("own_buildings", []) or []
        }
        assert "proc" in own, f"proc must surface in own_buildings: {own}"
        assert "fact" in own, f"fact must surface in own_buildings: {own}"

        proc_hp = float(own["proc"].get("hp", 1.0) or 0.0)
        fact_hp = float(own["fact"].get("hp", 1.0) or 0.0)

        # `health: 40` ⇒ ~40% HP. Allow a small tolerance for the
        # max_hp * 40 / 100 integer division.
        assert 0.30 <= proc_hp <= 0.50, (
            f"pre-placed proc with health:40 must spawn at ~40% HP; "
            f"got hp_pct={proc_hp}. (Engine bug: `health:` dropped by "
            "the scenario parser ⇒ actor spawns at 100%.)"
        )
        # The sibling fact omits `health:` ⇒ full HP.
        assert fact_hp >= 0.95, (
            f"fact without health: must spawn at full HP; got {fact_hp}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
