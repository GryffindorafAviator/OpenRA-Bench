"""End-to-end coverage for the Wave-9 `scheduled_events:` block.

The Rust engine parses scenario YAML, holds a `Vec<ScheduledEvent>`,
and fires each event on the world-tick boundary. This test rigs a
scenario whose ONLY enemy presence comes from a `spawn_actors` event
at tick 200 — we verify the agent sees zero enemies before tick 200
and `>= 2` after the event fires.

Validation is scripted (no model / no network); the test owns the
scenario YAML so it isn't coupled to any bench pack.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import textwrap

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]


# Base map fallback used by `openra_data::oramap::load_rush_hour_map_*`.
# We pin a base_map ref the loader resolves through its HOME fallback
# chain so the test runs without copying fixtures.
SCENARIO = textwrap.dedent(
    """\
    name: test-scheduled-spawn
    description: scheduled-event spawn at tick 200
    base_map: rush-hour-arena.oramap
    agent:
      faction: allies
    enemy:
      faction: soviet
    spawn_mcvs: false
    actors:
    - type: jeep
      owner: agent
      position:
      - 10
      - 10
      stance: 0
    # Persistent enemy marker that holds the episode alive past
    # scheduled-event firing (the env auto-`done`s on enemy elim).
    - type: fact
      owner: enemy
      position:
      - 80
      - 40
    scheduled_events:
    - tick: 200
      type: spawn_actors
      actors:
      - type: e1
        owner: enemy
        position:
        - 12
        - 12
        stance: 0
        count: 3
    """
)


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "sched.yaml"
    p.write_text(SCENARIO)
    return p


def _agent_visible_enemy_count(obs: dict) -> int:
    """Count visible enemy units in the observation. Different obs
    schemas surface enemies under `enemy_positions` / `enemy_units`;
    pick whichever exists."""
    for key in ("enemy_positions", "enemy_units"):
        if key in obs and obs[key]:
            return len(obs[key])
    return 0


def _near_count(obs: dict, cx: int, cy: int, r: int = 5) -> int:
    def _xy(p: dict) -> tuple[int, int]:
        # The obs may surface enemy positions under either
        # `cell_x`/`cell_y` (current Rust env) or `x`/`y` (legacy).
        x = p.get("cell_x", p.get("x", 9999))
        y = p.get("cell_y", p.get("y", 9999))
        return x, y

    out = 0
    for p in obs.get("enemy_positions", []) or []:
        x, y = _xy(p)
        if abs(x - cx) <= r and abs(y - cy) <= r:
            out += 1
    return out


def test_spawn_actors_fires_at_scheduled_tick(scenario_yaml: Path):
    """Wave-9 acceptance: zero enemy COMBAT units near the agent
    before tick 200, ≥2 after the scheduled spawn fires.

    Note: `World::process_frame` advances by `NetFrameInterval=3` ticks,
    and `Env::step` calls it `ticks_per_step` times. We use the
    minimum `ticks_per_step=1` to get a fine-grained probe (each
    `step()` ≈ 3 world-ticks).
    """
    env = OpenRAEnv(str(scenario_yaml), seed=1, ticks_per_step=1, max_ticks=2000)
    obs = env.reset()

    # Before the event: only the agent jeep + the far-east enemy fact
    # (out of vision range). Spawned-combat-unit count near the agent
    # must be 0.
    pre_visible = _agent_visible_enemy_count(obs)
    assert _near_count(obs, 12, 12) == 0

    # Step until JUST before tick 200 (each step ≈ 3 world ticks).
    while obs.get("game_tick", 0) < 195:
        obs, _r, _done, _info = env.step([])
    mid_tick = obs.get("game_tick", 0)
    assert mid_tick < 200, f"expected tick<200 pre-fire, got {mid_tick}"
    assert _near_count(obs, 12, 12) == 0, (
        "no enemy units expected pre-event near (12,12), "
        f"found {_near_count(obs, 12, 12)}"
    )

    # Step a few more times to cross the trigger tick. The fire-on-frame
    # may also need one additional step before the spawned units show
    # up in the shroud-filtered observation.
    while obs.get("game_tick", 0) < 220:
        obs, _r, _done, _info = env.step([])
    post_tick = obs.get("game_tick", 0)
    assert post_tick >= 200, f"expected tick>=200 post-fire, got {post_tick}"

    assert _near_count(obs, 12, 12) >= 2, (
        f"expected >=2 spawned enemies near (12,12) after tick 200, "
        f"found {_near_count(obs, 12, 12)}. pre_visible={pre_visible}, "
        f"all_enemy_positions={obs.get('enemy_positions')}"
    )


def test_no_scheduled_events_is_back_compat(scenario_yaml: Path):
    """A scenario without `scheduled_events:` continues to load and
    step normally (the parser tolerates the absence). Sanity check
    that the engine wiring is purely additive."""
    plain = SCENARIO.split("scheduled_events:")[0]
    p = scenario_yaml.parent / "plain.yaml"
    p.write_text(plain)
    env = OpenRAEnv(str(p), seed=1, ticks_per_step=30, max_ticks=500)
    obs = env.reset()
    for _ in range(5):
        obs, _r, _done, _info = env.step([])
    assert obs.get("game_tick", 0) >= 100
