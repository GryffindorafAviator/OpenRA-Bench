"""F11 engine-risk pin: `termination.agent_units_killed: false` plus
`place_building` after a mid-episode wipe.

The `f11-rebuild-after-attrition` pack chains:
  1. Agent starts with fact + powr + proc + harv + a weap (the
     production building under attrition pressure).
  2. A `scheduled_events.destroy_actors` at tick T_attrition wipes
     the agent's weap.
  3. The agent must `Command.build('weap')` + `place_building` to
     rebuild AFTER the wipe and continue producing.

For this chain to work the engine must:
  * Not terminate the episode when the agent's combat units are
    destroyed (the agent only has buildings + a harv at the point
    of attrition). This is what `termination.agent_units_killed:
    false` controls.
  * Allow `place_building` past the wipe — the build-radius is
    anchored on the fact (still standing) so this should 'just
    work', but it's the load-bearing risk to pin.

This test exercises both: a destroy_actors event removes the weap
at tick 50, then the agent rebuilds and places a new weap. The
episode must still be running at the placement step and the new
weap must surface in own_buildings.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]


SCENARIO_TEXT = """\
name: rebuild-after-attrition
description: F11 engine-risk - rebuild after scheduled destroy_actors
base_map: rush-hour-arena
starting_cash: 5000
spawn_mcvs: false
agent:
  faction: allies
  cash: 5000
enemy:
  faction: soviet
  cash: 0
tools:
- observe
- build
- place_building
planning: true
termination:
  max_ticks: 12000
  agent_units_killed: false
  enemy_units_killed: false
actors:
# Agent base: tech tree for weap pre-built so we can scope the test
# to the attrition + rebuild path.
- type: fact
  owner: agent
  position:
  - 10
  - 10
- type: powr
  owner: agent
  position:
  - 14
  - 10
- type: proc
  owner: agent
  position:
  - 10
  - 14
- type: weap
  owner: agent
  position:
  - 14
  - 14
# Persistent enemy marker.
- type: fact
  owner: enemy
  position:
  - 80
  - 80
scheduled_events:
- tick: 50
  type: destroy_actors
  filter:
    owner: agent
    region:
      x: 14
      y: 14
      radius: 2
"""


@pytest.fixture
def scenario_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "rebuild.yaml"
    p.write_text(SCENARIO_TEXT)
    return p


def _own_buildings(obs: dict) -> list[dict]:
    return obs.get("own_buildings", []) or []


def _has_done_in_prod(obs: dict, item: str) -> bool:
    for p in obs.get("production", []) or []:
        if isinstance(p, dict):
            if str(p.get("item", "")).lower() == item and bool(p.get("done", False)):
                return True
    return False


def test_termination_flag_keeps_episode_alive_and_rebuild_works(scenario_yaml: Path):
    """Verify the full chain: weap exists → destroyed at tick 50 →
    agent rebuilds + places a new weap → episode still running and
    new weap is in own_buildings."""
    env = OpenRAEnv(
        str(scenario_yaml), seed=1, ticks_per_step=3, max_ticks=12000,
    )
    obs = env.reset()

    # Initial state: weap is present.
    types = {b["type"] for b in _own_buildings(obs)}
    assert "weap" in types, (
        f"pre-attrition weap missing; own_buildings types={types}"
    )

    # Step past tick 50 (destroy event tick). With ticks_per_step=3
    # each step ~ 3 world-ticks; 25 steps = ~75 world-ticks.
    for _ in range(40):
        obs, _r, done, _info = env.step([Command.observe()])
        if obs.get("game_tick", 0) >= 80:
            break

    # Weap must be gone after the destroy event.
    types_post = {b["type"] for b in _own_buildings(obs)}
    assert "weap" not in types_post, (
        f"weap should be destroyed after tick 50, "
        f"types={types_post}, tick={obs.get('game_tick')}"
    )
    # Episode must NOT have terminated (agent_units_killed: false).
    # The agent still has its non-combat buildings (fact/powr/proc),
    # so we shouldn't be done.
    assert not done, "episode should not terminate after weap wipe"

    # Now rebuild the weap. Queue + wait for done + place. weap costs
    # 2000, build time ~ cost/sec, so ~2000 ticks at 3/step = 666
    # steps worst-case. Use a generous bound.
    obs, _r, _d, _i = env.step([Command.build("weap")])
    for _ in range(800):
        obs, _r, _d, _i = env.step([Command.observe()])
        if _has_done_in_prod(obs, "weap"):
            break
    assert _has_done_in_prod(obs, "weap"), (
        f"rebuilt weap never completed in production queue; "
        f"production={obs.get('production')}, "
        f"cash={obs.get('cash') if 'cash' in obs else 'n/a'}"
    )

    # Place at a free cell adjacent to the fact.
    obs, _r, _d, _i = env.step([Command.place_building("weap", 14, 14)])
    for _ in range(3):
        obs, _r, _d, _i = env.step([Command.observe()])

    new_weaps = [
        b for b in _own_buildings(obs)
        if str(b.get("type", "")).lower() == "weap"
    ]
    assert len(new_weaps) >= 1, (
        f"rebuilt weap not in own_buildings after place; "
        f"types={[b.get('type') for b in _own_buildings(obs)]}"
    )
