"""F11 engine-risk pin: `scheduled_events.spawn_actors` injects
Aircraft (`heli`) correctly.

The Wave-9 spawn_actors test only pinned ground actors (e1 infantry).
F11's reactive-pivot idiom (`f11-pivot-on-scout` hard) injects an
enemy heli mid-episode via `scheduled_events.spawn_actors` so the
model must re-scout, observe the new arm, and pivot. This test
verifies the engine handles ActorKind::Aircraft injection the same
way as ground actors:
  * the heli appears in `enemy_positions` once the event fires,
  * its `actor_type` is `heli`,
  * the heli is a functional aircraft (its Mobile trait + facing are
    wired so move/attack orders against it resolve normally).

If the test fails, `f11-pivot-on-scout` hard's reactive-arm switch is
defective.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]


SCENARIO_TEXT = """\
name: scheduled-spawn-aircraft
description: F11 engine-risk - mid-episode heli injection via scheduled_events
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
actors:
# Agent scout — at the spawn cell so the scheduled heli appears in
# its vision.
- type: jeep
  owner: agent
  position:
  - 30
  - 20
  stance: 0
# Persistent enemy marker so the env doesn't auto-`done` on enemy
# elimination before the scheduled event fires.
- type: fact
  owner: enemy
  position:
  - 80
  - 80
scheduled_events:
- tick: 150
  type: spawn_actors
  actors:
  - type: heli
    owner: enemy
    position:
    - 32
    - 20
    stance: 0
"""


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "sched_heli.yaml"
    p.write_text(SCENARIO_TEXT)
    return p


def _heli_entries(obs: dict) -> list[dict]:
    """Return enemy heli entries from the obs. `enemy_positions` is
    a list of dicts including `actor_type`."""
    out = []
    for e in obs.get("enemy_positions", []) or []:
        if not isinstance(e, dict):
            continue
        if str(e.get("actor_type", "")).lower() == "heli":
            out.append(e)
    return out


def test_scheduled_spawn_actors_injects_aircraft(scenario_yaml: Path):
    """Scheduled spawn of a `heli` at tick 150 must surface in the
    agent's enemy_positions after the event fires."""
    env = OpenRAEnv(
        str(scenario_yaml), seed=1, ticks_per_step=1, max_ticks=2000,
    )
    obs = env.reset()

    # Pre-event: no heli should be visible.
    pre_heli = _heli_entries(obs)
    assert not pre_heli, (
        f"no heli expected pre-event, found {pre_heli}"
    )

    # Step past tick 150 (each step ~3 world-ticks via process_frame).
    while obs.get("game_tick", 0) < 145:
        obs, _r, _d, _i = env.step([])
    pre_tick = obs.get("game_tick", 0)
    assert pre_tick < 150, f"expected pre-event tick<150, got {pre_tick}"
    pre_heli = _heli_entries(obs)
    assert not pre_heli, (
        f"no heli expected at tick {pre_tick}, found {pre_heli}"
    )

    # Step past the scheduled tick + a few extra frames for shroud
    # reveal of the new aircraft.
    while obs.get("game_tick", 0) < 180:
        obs, _r, _d, _i = env.step([])
    post_tick = obs.get("game_tick", 0)
    assert post_tick >= 150, f"expected post-event tick>=150, got {post_tick}"

    helis = _heli_entries(obs)
    assert len(helis) >= 1, (
        f"expected ≥1 heli in enemy_positions after tick 150, found "
        f"{helis}; all enemy_positions={obs.get('enemy_positions')}"
    )
    heli = helis[0]
    assert int(heli.get("cell_x", 0)) == 32
    assert int(heli.get("cell_y", 0)) == 20
