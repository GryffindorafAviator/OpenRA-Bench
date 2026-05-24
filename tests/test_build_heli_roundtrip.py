"""F11 engine-risk pin: `build('hpad')` + `place_building` + `build('heli')`
end-to-end.

The pre-placed-heli path is wired (`combat-heli-flank`). This test pins
the BUILT path - the chain a Family-11 (full-game) pack will exercise:
  1. Agent starts with `fact + powr + proc + tent + dome` so all the
     prerequisites for `hpad` (and `heli`) are satisfied.
  2. Agent calls `Command.build('hpad')`, waits for the production queue
     to report `done=True`, then `Command.place_building('hpad', x, y)`.
  3. A new `hpad` must surface in `own_buildings`.
  4. Agent calls `Command.build('heli')`, steps until the heli surfaces
     in `units_summary` (it should auto-spawn near the hpad).
  5. Verify the heli is functional - issue `Command.move_units` to a
     far target and observe the heli moving.

If any step fails the corresponding F11 pack (`f11-vertical-strike-
ground-air`, `f11-pivot-on-scout`, `f11-full-combined-arms`) is blocked.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_build_heli.yaml", delete=False,
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _make_scenario() -> dict:
    """Agent base seed: every prereq for building a heli, pre-placed so
    the test exercises the BUILD path (not the tech chain). In the
    vendored RA YAML, `heli` requires `~hpad, atek, ~techlevel.high`,
    and `atek` itself requires `weap, dome, ~structures.allies`. So the
    seed is fact + powr + proc + weap + dome + atek (all already
    standing). `hpad` (prereq: dome) is the one the agent BUILDS in
    this test, and then `heli` is queued from it.

    Cash is high enough to cover hpad ($500) + heli ($2000). A far
    enemy `fact` keeps the engine from auto-`done`ing on enemy
    elimination before the test completes.
    """
    return {
        "name": "build-heli-roundtrip",
        "description": "F11 engine-risk: build(hpad)+build(heli) end-to-end",
        "base_map": "rush-hour-arena",
        "starting_cash": 5000,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 5000},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "build", "place_building", "move_units"],
        "planning": True,
        "termination": {"max_ticks": 12000},
        "actors": [
            {"type": "fact", "owner": "agent", "position": [10, 10]},
            {"type": "powr", "owner": "agent", "position": [14, 10]},
            {"type": "powr", "owner": "agent", "position": [17, 10]},
            {"type": "proc", "owner": "agent", "position": [10, 14]},
            {"type": "weap", "owner": "agent", "position": [14, 14]},
            {"type": "dome", "owner": "agent", "position": [10, 18]},
            {"type": "atek", "owner": "agent", "position": [14, 18]},
            # Far enemy marker (out of agent sight; gates engine auto-done).
            {"type": "fact", "owner": "enemy", "position": [90, 80]},
        ],
    }


def _own_buildings(obs: dict) -> list[dict]:
    return obs.get("own_buildings", []) or []


def _units(obs: dict) -> list[dict]:
    """Surface own units as a list. The raw env emits `unit_positions`
    as a dict keyed by actor id; the `type` comes from the actor type
    inferred at spawn. We synthesize a list shape matching the bench
    adapter's `units_summary` (with `id`, `cell_x`, `cell_y`, `type`).
    """
    raw = obs.get("unit_positions", {}) or {}
    if not isinstance(raw, dict):
        return []
    out = []
    for aid, pos in raw.items():
        if not isinstance(pos, dict):
            continue
        out.append({
            "id": str(aid),
            "cell_x": int(pos.get("cell_x", 0)),
            "cell_y": int(pos.get("cell_y", 0)),
            # Engine emits `actor_type`; bench adapter remaps to `type`.
            "type": pos.get("actor_type", pos.get("type", "")),
        })
    return out


def _has_done_in_prod(obs: dict, item: str) -> bool:
    raw_prod = obs.get("production", []) or []
    for p in raw_prod:
        if isinstance(p, dict):
            if str(p.get("item", "")).lower() == item and bool(p.get("done", False)):
                return True
    return False


def _step_until(env, predicate, *, max_steps: int = 400, cmd=None):
    """Step the env with `cmd` (default observe) until `predicate(obs)`
    is truthy or `max_steps` consumed. Returns the last obs."""
    obs = None
    for _ in range(max_steps):
        c = cmd if cmd is not None else [Command.observe()]
        obs, _r, _done, _info = env.step(c)
        if predicate(obs):
            return obs
    return obs


def test_build_hpad_then_build_heli_end_to_end():
    """The headline F11 build-roundtrip: hpad then heli, end-to-end."""
    path = _scenario_path(_make_scenario())
    try:
        # Default ticks_per_step (30) gives each step a full second of
        # game time so production progresses visibly across steps.
        env = OpenRAEnv(path, seed=1)
        obs = env.reset()

        # Sanity: pre-placed base must be present.
        own_b0 = _own_buildings(obs)
        types0 = {b["type"] for b in own_b0}
        assert {"fact", "powr", "proc", "weap", "dome", "atek"} <= types0, (
            f"pre-placed base missing buildings; types={types0}"
        )

        # 1) build(hpad)
        obs, _r, _d, _i = env.step([Command.build("hpad")])
        # Wait for hpad to complete in the production queue.
        obs = _step_until(
            env,
            lambda o: _has_done_in_prod(o, "hpad"),
            max_steps=120,
        )
        assert _has_done_in_prod(obs, "hpad"), (
            f"hpad never finished in production queue; "
            f"production={obs.get('production')}, "
            f"own_buildings={[b.get('type') for b in _own_buildings(obs)]}"
        )

        # 2) place_building(hpad) - pick a free spot adjacent to dome.
        hpad_x, hpad_y = 13, 18
        obs, _r, _d, _i = env.step([
            Command.place_building("hpad", hpad_x, hpad_y),
        ])
        # Step a couple of frames so the world processes the place.
        for _ in range(3):
            obs, _r, _d, _i = env.step([Command.observe()])

        own_b = _own_buildings(obs)
        hpads = [b for b in own_b if str(b.get("type", "")).lower() == "hpad"]
        assert len(hpads) >= 1, (
            f"hpad missing from own_buildings after place; "
            f"own_buildings={[(b.get('type'), b.get('cell_x'), b.get('cell_y')) for b in own_b]}"
        )

        # 3) build(heli) - record pre-existing aircraft so we know
        # the new one is genuinely new.
        pre_units = _units(obs)
        pre_heli_ids = {
            int(u["id"]) for u in pre_units
            if str(u.get("type", "")).lower() == "heli"
        }
        obs, _r, _d, _i = env.step([Command.build("heli")])

        # The heli should be spawned as a UNIT (not placed). Poll until
        # a new heli appears in units_summary.
        obs = _step_until(
            env,
            lambda o: any(
                str(u.get("type", "")).lower() == "heli"
                and int(u["id"]) not in pre_heli_ids
                for u in _units(o)
            ),
            max_steps=200,
        )
        units = _units(obs)
        new_helis = [
            u for u in units
            if str(u.get("type", "")).lower() == "heli"
            and int(u["id"]) not in pre_heli_ids
        ]
        assert len(new_helis) >= 1, (
            f"heli never spawned after build('heli'); "
            f"units_summary types={[u.get('type') for u in units]}, "
            f"production={obs.get('production')}"
        )
        heli = new_helis[0]
        heli_id = int(heli["id"])
        hx0, hy0 = int(heli["cell_x"]), int(heli["cell_y"])

        # 4) Verify the heli is functional - order a move to a far cell
        # and observe motion.
        target_x, target_y = hx0 + 20, hy0 + 10
        obs, _r, _d, _i = env.step([
            Command.move_units([str(heli_id)], target_x, target_y),
        ])
        obs = _step_until(
            env,
            lambda o: any(
                int(u["id"]) == heli_id
                and (int(u["cell_x"]) != hx0 or int(u["cell_y"]) != hy0)
                for u in _units(o)
            ),
            max_steps=120,
        )
        new_pos = next(
            (u for u in _units(obs) if int(u["id"]) == heli_id),
            None,
        )
        assert new_pos is not None, "heli vanished from units after move"
        assert (int(new_pos["cell_x"]) != hx0
                or int(new_pos["cell_y"]) != hy0), (
            f"heli did not move from ({hx0},{hy0}); "
            f"now at ({new_pos['cell_x']},{new_pos['cell_y']})"
        )
    finally:
        Path(path).unlink(missing_ok=True)
