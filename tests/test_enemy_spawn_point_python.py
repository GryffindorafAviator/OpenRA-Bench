"""End-to-end coverage for the Wave-9 enemy `spawn_point:` filter.

The Rust engine's `expand_scenario_actors` (in
`openra-data/src/oramap.rs`) now applies the `spawn_point:` filter
PER OWNER: when any enemy actor declares `spawn_point`, only enemies
whose `spawn_point` matches the chosen value place. This Python test
rigs a self-contained scenario whose enemy composition differs
between `spawn_point=0` and `spawn_point=1`, then round-trips it
through `OpenRAEnv` to verify that:

  * loading with `spawn_point=0` surfaces ONLY the spawn_point=0
    enemy composition (e1 swarm);
  * loading with `spawn_point=1` surfaces ONLY the spawn_point=1
    enemy composition (3tnk column);
  * the env's seed→spawn_point round-robin falls back to the
    enemy-side spawn points when the agent declares none (the
    contract `adv-rps-counter-pick.yaml` relies on);
  * a scenario where NO enemy declares `spawn_point` still places
    every enemy on every spawn (back-compat).

Validation is scripted (no model / no network); the test owns its
scenarios so it isn't coupled to any bench pack.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv  # type: ignore[import]

# Absolute path to the bundled rush-hour terrain — older templates
# wrote `base_map: {base_map}` and relied on the engine's
# HOME-dir fallback to `~/Projects/OpenRA-RL-Training/...`, which
# does not exist on CI runners. Same fix as commit 00d01ad4.
_BUNDLED_MAP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "maps"
    / "rush-hour-arena.oramap"
)


def _enemy_actor_types(obs: dict) -> list[str]:
    """Every visible enemy actor_type at the current observation. The
    rush-hour-arena map is small enough that two starter jeeps and the
    composition cluster are all within visible/shroud range after a
    few steps; we step a handful of frames before counting to let the
    shroud reveal complete."""
    out: list[str] = []
    for p in obs.get("enemy_positions", []) or []:
        t = p.get("actor_type")
        if t:
            out.append(str(t))
    return out


SCENARIO_TPL = textwrap.dedent(
    """\
    name: enemy-spawn-roundtrip
    description: per-owner spawn_point filter round-trip
    base_map: {base_map}
    agent:
      faction: allies
    enemy:
      faction: soviet
    spawn_mcvs: false
    actors:
    - type: jeep
      owner: agent
      position:
      - 16
      - 20
      stance: 0
    - type: jeep
      owner: agent
      position:
      - 16
      - 22
      stance: 0
    # Composition cluster placed within JEEP sight (radius ~7c) so
    # the shroud-filtered `enemy_positions` actually surfaces the
    # loaded archetype. The archetype rotates per `spawn_point`.
    # ── spawn_point=0 enemies: e1 rifle swarm.
    - type: e1
      owner: enemy
      position:
      - 20
      - 20
      stance: 0
      spawn_point: 0
    - type: e1
      owner: enemy
      position:
      - 21
      - 20
      stance: 0
      spawn_point: 0
    - type: e1
      owner: enemy
      position:
      - 22
      - 20
      stance: 0
      spawn_point: 0
    # ── spawn_point=1 enemies: 3tnk heavy column at the same cluster.
    - type: 3tnk
      owner: enemy
      position:
      - 20
      - 21
      stance: 0
      spawn_point: 1
    - type: 3tnk
      owner: enemy
      position:
      - 22
      - 21
      stance: 0
      spawn_point: 1
    # ── Persistent far-east enemy marker, duplicated across every
    # spawn group (mirroring the agent-side idiom — actors of an
    # owner whose filter is active must declare a matching
    # spawn_point to pass).
    - type: fact
      owner: enemy
      position:
      - 124
      - 20
      spawn_point: 0
    - type: fact
      owner: enemy
      position:
      - 124
      - 20
      spawn_point: 1
    """
)

NO_ENEMY_SP_TPL = textwrap.dedent(
    """\
    name: enemy-no-spawn-point
    description: back-compat — no enemy declares spawn_point
    base_map: {base_map}
    agent:
      faction: allies
    enemy:
      faction: soviet
    spawn_mcvs: false
    actors:
    - type: jeep
      owner: agent
      position:
      - 16
      - 20
      stance: 0
    # Place cluster within sight of the starting jeep so the obs's
    # `enemy_positions` (shroud-filtered) actually surfaces them.
    - type: e1
      owner: enemy
      position:
      - 20
      - 20
      stance: 0
    - type: e1
      owner: enemy
      position:
      - 21
      - 20
      stance: 0
    - type: 3tnk
      owner: enemy
      position:
      - 22
      - 20
      stance: 0
    - type: fact
      owner: enemy
      position:
      - 124
      - 20
    """
)


def _step_for_reveal(env: OpenRAEnv, obs: dict, frames: int = 5) -> dict:
    """Step a handful of frames so the shroud reveals the static
    enemy cluster (units have RevealsShroud). Returns the latest obs.
    """
    for _ in range(frames):
        obs, _r, _done, _info = env.step([])
    return obs


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "enemy-sp-roundtrip.yaml"
    p.write_text(SCENARIO_TPL.format(base_map=str(_BUNDLED_MAP)))
    return p


@pytest.fixture
def no_enemy_sp_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "enemy-no-sp.yaml"
    p.write_text(NO_ENEMY_SP_TPL.format(base_map=str(_BUNDLED_MAP)))
    return p


def _has_visible_enemy(env: OpenRAEnv, obs: dict, max_frames: int = 60) -> dict:
    """Step until at least one enemy actor surfaces in
    `enemy_positions` (the cluster is at x=70 — out of starting
    sight; the agent's jeeps move 0 cells per `[]` step, so visibility
    only kicks in once shroud over time recomputes. For obs schemas
    that don't auto-reveal static far-east clusters, we still rely on
    the loaded actor count being correct — the round-trip assertion
    targets the COMPOSITION not just visibility)."""
    for _ in range(max_frames):
        if (obs.get("enemy_positions") or []):
            return obs
        obs, _r, _done, _info = env.step([])
    return obs


def test_spawn_point_0_loads_only_e1_swarm(scenario_yaml: Path):
    """`spawn_point=0` keeps the e1 swarm enemies and FILTERS OUT
    the 3tnk column. The cluster is placed within starting-jeep
    sight so the obs's shroud-filtered `enemy_positions` actually
    surfaces it."""
    env = OpenRAEnv(
        str(scenario_yaml), seed=1, ticks_per_step=30, max_ticks=2000,
        spawn_point=0,
    )
    obs = env.reset()
    for _ in range(10):
        obs, _r, _done, _info = env.step([])

    types = _enemy_actor_types(obs)
    # e1 must appear (positive), 3tnk must NOT (filtered).
    assert "e1" in types, (
        f"spawn 0 must place e1 swarm in obs sight range, got {types}"
    )
    assert "3tnk" not in types, (
        f"spawn 0 must filter out 3tnk (spawn_point=1) enemies, got {types}"
    )


def test_spawn_point_1_loads_only_3tnk_column(scenario_yaml: Path):
    """`spawn_point=1` keeps the 3tnk column and FILTERS OUT the
    e1 swarm at the cluster."""
    env = OpenRAEnv(
        str(scenario_yaml), seed=1, ticks_per_step=30, max_ticks=2000,
        spawn_point=1,
    )
    obs = env.reset()
    for _ in range(10):
        obs, _r, _done, _info = env.step([])

    types = _enemy_actor_types(obs)
    # 3tnk must appear (positive), e1 must NOT (filtered).
    assert "3tnk" in types, (
        f"spawn 1 must place 3tnk column in obs sight range, got {types}"
    )
    assert "e1" not in types, (
        f"spawn 1 must filter out e1 (spawn_point=0) cluster enemies, got {types}"
    )


def test_seed_round_robins_over_enemy_spawn_points_when_agent_has_none(
    scenario_yaml: Path,
):
    """Wave-9 env contract: when the AGENT side declares no
    spawn_points (only enemies do), the env falls back to the
    enemy-side `distinct_enemy_spawn_points` for the seed→spawn
    round-robin. seed=0 → enemy_sps[0]=0 (e1 swarm); seed=1 →
    enemy_sps[1]=1 (3tnk column).

    The contract is observable through the loaded enemy composition.
    """
    # seed=0 → spawn_point=0 (e1 swarm; no 3tnk)
    env0 = OpenRAEnv(str(scenario_yaml), seed=0, ticks_per_step=30, max_ticks=2000)
    obs0 = env0.reset()
    for _ in range(10):
        obs0, _r, _done, _info = env0.step([])
    types0 = _enemy_actor_types(obs0)
    assert "e1" in types0 and "3tnk" not in types0, (
        f"seed=0 → enemy sp=0 (e1 swarm); got {types0}"
    )

    # seed=1 → spawn_point=1 (3tnk; no e1 cluster)
    env1 = OpenRAEnv(str(scenario_yaml), seed=1, ticks_per_step=30, max_ticks=2000)
    obs1 = env1.reset()
    for _ in range(10):
        obs1, _r, _done, _info = env1.step([])
    types1 = _enemy_actor_types(obs1)
    assert "3tnk" in types1 and "e1" not in types1, (
        f"seed=1 → enemy sp=1 (3tnk column); got {types1}"
    )


def test_back_compat_no_enemy_spawn_point_all_pass(no_enemy_sp_yaml: Path):
    """A scenario where NO enemy declares `spawn_point` still places
    every enemy on every spawn (pre-Wave-9 contract). The static
    cluster types reported via the obs schema must include BOTH e1
    and 3tnk (or at least one if the obs schema undercounts; the
    strict assertion is that NO enemy was dropped — both types must
    NOT be simultaneously absent across all the seeds we probe)."""
    union: set[str] = set()
    for seed in (0, 1, 2, 3):
        env = OpenRAEnv(
            str(no_enemy_sp_yaml),
            seed=seed,
            ticks_per_step=30,
            max_ticks=2000,
        )
        obs = env.reset()
        for _ in range(40):
            obs, _r, _done, _info = env.step([])
        union.update(_enemy_actor_types(obs))
    # Back-compat: both e1 and 3tnk must surface (otherwise the
    # owner filter dropped no-spawn_point enemies, which would
    # regress the pre-Wave-9 contract).
    assert "e1" in union, f"back-compat regression: e1 missing across seeds, union={union}"
    assert "3tnk" in union, f"back-compat regression: 3tnk missing across seeds, union={union}"
