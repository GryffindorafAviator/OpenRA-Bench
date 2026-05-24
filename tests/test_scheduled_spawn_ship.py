"""F11 engine-risk pin: `scheduled_events.spawn_actors` injects
Ships (`dd`) correctly.

Mirrors `test_scheduled_spawn_aircraft.py` for the ship axis. The
F11 combined-arms idiom (`f11-vertical-strike-naval` and similar)
may use a scheduled naval reinforcement; this test pins that path
through the engine's `apply_spawn_actors` → `build_scenario_actor`
chain for ActorKind::Ship.

Ship-specific gotcha: the injected dd must spawn on a water cell.
If the scheduled actor's position is over land the engine still
inserts the actor (no terrain check at injection time), so the
ship is observable but stuck on land. For F11 pack authoring this
means the scheduled-event position MUST be in the water_rect — the
test pins both injection and water placement.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]


SCENARIO_TEXT = """\
name: scheduled-spawn-ship
description: F11 engine-risk - mid-episode dd injection via scheduled_events
base_map: rush-hour-arena
starting_cash: 0
spawn_mcvs: false
agent:
  faction: allies
  cash: 0
enemy:
  faction: soviet
  cash: 0
tools:
- observe
planning: true
termination:
  max_ticks: 8000
# Water column at x=20..21 spanning the playable y-range; the
# scheduled dd is positioned inside this column.
water_rect:
- 20
- 2
- 2
- 36
actors:
# Agent scout positioned to see the scheduled ship.
- type: jeep
  owner: agent
  position:
  - 18
  - 20
  stance: 0
# Persistent enemy marker.
- type: fact
  owner: enemy
  position:
  - 80
  - 80
scheduled_events:
- tick: 150
  type: spawn_actors
  actors:
  - type: dd
    owner: enemy
    position:
    - 20
    - 20
    stance: 0
"""


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "sched_dd.yaml"
    p.write_text(SCENARIO_TEXT)
    return p


def _dd_entries(obs: dict) -> list[dict]:
    out = []
    for e in obs.get("enemy_positions", []) or []:
        if not isinstance(e, dict):
            continue
        if str(e.get("actor_type", "")).lower() == "dd":
            out.append(e)
    return out


def test_scheduled_spawn_actors_injects_ship(scenario_yaml: Path):
    """Scheduled spawn of a `dd` at tick 150 must surface in the
    agent's enemy_positions after the event fires, at the declared
    water cell."""
    env = OpenRAEnv(
        str(scenario_yaml), seed=1, ticks_per_step=1, max_ticks=2000,
    )
    obs = env.reset()

    pre_dd = _dd_entries(obs)
    assert not pre_dd, f"no dd expected pre-event, found {pre_dd}"

    # Step past the scheduled tick + a few extra frames for shroud
    # reveal of the new ship.
    while obs.get("game_tick", 0) < 180:
        obs, _r, _d, _i = env.step([])
    post_tick = obs.get("game_tick", 0)
    assert post_tick >= 150, f"expected post-event tick>=150, got {post_tick}"

    dds = _dd_entries(obs)
    assert len(dds) >= 1, (
        f"expected ≥1 dd in enemy_positions after tick 150, found "
        f"{dds}; all enemy_positions={obs.get('enemy_positions')}"
    )
    dd = dds[0]
    assert int(dd.get("cell_x", 0)) == 20
    assert int(dd.get("cell_y", 0)) == 20
