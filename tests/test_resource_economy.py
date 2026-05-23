"""End-to-end guardrail for the resource (ore) layer + harvest loop.

The engine wave adds three pieces glued together:

1. A scenario-YAML top-level `ore_patches:` list — each entry is
   `{x, y, amount, radius}` and is materialised by the env at
   world-build time into a disk of harvestable ore on the terrain.
2. A `World::auto_route_idle_harvesters` per-tick pass that installs a
   `Harvest` activity on any owned, idle harvester whose owner owns a
   refinery (`proc`). Scenario-placed `harv` actors are otherwise
   `activity: None` and would sit inert.
3. An `ore_cells: [{cell_x, cell_y, amount}, ...]` field on the
   Python observation dict, surfaced flat (no fog gating) so a
   structured-only agent can still see where the patches are.

This test loads a minimal scenario pack with one ore patch, a fact,
a powr, and an idle harv. Under a STALL policy (no orders) we
expect zero income — the harv sits idle because there is no proc.
Under a SCRIPTED policy that issues `build('proc')` +
`place_building(proc)` adjacent to the patch, we expect cash to grow
above the starting cash within a few decision turns — the proof that
the harvest loop closes end-to-end.

Mirror of the Rust integration test
`OpenRA-Rust/openra-sim/tests/test_resource_layer.rs`.
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
      id: resource-economy-test
      title: 'Resource layer + harvest loop smoke'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'collector + processor harvest loop'
    base_map: rush-hour-arena
    starting_cash: 2000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, build, place_building, move_units, stop]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 4000}
      actors:
        - {type: fact, owner: agent, position: [15, 20]}
        - {type: powr, owner: agent, position: [15, 18]}
        - {type: harv, owner: agent, position: [18, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    ore_patches:
      - {x: 24, y: 20, amount: 2000, radius: 2}
    levels:
      easy:
        description: 'one ore patch, idle harv, no pre-placed proc'
        starting_cash: 2000
        win_condition: {cash_gte: 99999}
        fail_condition: {after_ticks: 9999}
        max_turns: 30
      medium:
        description: 'one ore patch, idle harv, no pre-placed proc'
        starting_cash: 2000
        win_condition: {cash_gte: 99999}
        fail_condition: {after_ticks: 9999}
        max_turns: 30
      hard:
        description: 'one ore patch, idle harv, no pre-placed proc'
        starting_cash: 2000
        win_condition: {cash_gte: 99999}
        fail_condition: {after_ticks: 9999}
        max_turns: 30
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="resource_economy_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def _setup_env():
    pack_path = _pack_tmp()
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        scen_path = _scenario_to_tmp_yaml(compiled)
    except Exception:
        pack_path.unlink(missing_ok=True)
        raise
    pool = RustEnvPool(size=1, scenario_path=scen_path)
    return pool, pool.acquire(), Path(scen_path), pack_path


def test_observation_surfaces_ore_cells():
    """`reset()`'s observation must expose `ore_cells` populated with
    the disk seeded from the YAML's `ore_patches:` block."""
    pool, env, scen_path, pack_path = _setup_env()
    try:
        obs = env.reset(seed=1)
        ore = obs.get("ore_cells") if isinstance(obs, dict) else None
        assert ore is not None, (
            "observation must include `ore_cells` (engine resource wave). "
            f"keys: {sorted(obs.keys()) if isinstance(obs, dict) else type(obs)}"
        )
        assert len(ore) > 0, (
            "ore_patches: [...] in YAML must materialise into at least "
            "one ore cell on the terrain; got empty list"
        )
        # Every entry should have positive `amount` and coordinates
        # inside the disk (radius 2 around (24, 20)).
        for cell in ore:
            assert int(cell["amount"]) > 0
            assert abs(int(cell["cell_x"]) - 24) <= 3
            assert abs(int(cell["cell_y"]) - 20) <= 3
    finally:
        pool.release(env)
        pool.shutdown()
        scen_path.unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)


def test_stall_policy_yields_zero_income():
    """A stall (only `observe`) policy must leave cash unchanged —
    no proc → harv is idle → no harvest income. This is the
    "lose by doing nothing" lower bound."""
    import openra_train

    pool, env, scen_path, pack_path = _setup_env()
    try:
        obs = env.reset(seed=1)
        starting_cash = int(obs["economy"]["cash"])
        for _ in range(15):  # 15 decision turns × ~90 ticks ≈ 1350 ticks
            _ = env.step([openra_train.Command.observe()])
        # observation() after step
        final_obs = env.last_observation()
        final_cash = int(final_obs["economy"]["cash"])
        assert final_cash <= starting_cash, (
            f"stall must not grow cash (no proc ⇒ harv idle); "
            f"start={starting_cash} final={final_cash}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        scen_path.unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)


def test_intended_policy_grows_cash_via_harvest():
    """The intended policy: build proc, place adjacent to the ore
    patch. The harv auto-routes the moment the proc lands; within
    a few decision turns cash + stored should rise above starting
    cash."""
    import openra_train

    pool, env, scen_path, pack_path = _setup_env()
    try:
        obs = env.reset(seed=1)
        starting_cash = int(obs["economy"]["cash"])
        starting_total = starting_cash + int(obs["economy"].get("resources", 0))

        # Turn 1: queue the refinery.
        _ = env.step([openra_train.Command.build("proc")])
        # Spin a few turns to let production complete (proc is 1400
        # cost, build time ~ a few turns).
        for _ in range(6):
            _ = env.step([openra_train.Command.observe()])

        # Turn N: place proc adjacent to the ore patch centre (24, 20).
        # `place_building` does not enforce build-adjacency, so we drop
        # it directly next to the ore.
        _ = env.step([openra_train.Command.place_building("proc", 21, 20)])

        # Spin out the remaining decision budget.
        grew = False
        for _ in range(20):
            _ = env.step([openra_train.Command.observe()])
            o = env.last_observation()
            total = int(o["economy"]["cash"]) + int(
                o["economy"].get("resources", 0)
            )
            if total > starting_total:
                grew = True
                break

        assert grew, (
            "intended policy (build proc + place adjacent to ore) must "
            f"grow cash+resources above starting total {starting_total}; "
            f"last total stayed flat — the harvest loop is not closing."
        )
    finally:
        pool.release(env)
        pool.shutdown()
        scen_path.unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
