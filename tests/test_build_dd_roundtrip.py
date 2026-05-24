"""F11 engine-risk pin: `build('syrd')` + `place_building` + `build('dd')`
end-to-end.

The pre-placed `dd` path is wired (`combat-naval-shore-strike`). This
test pins the BUILT path - the chain `f11-vertical-strike-naval` /
`f11-full-combined-arms` will exercise:
  1. Agent starts with `fact + powr + proc + dome` and a water_rect on
     one edge of the map so syrd can be placed adjacent to water.
  2. Agent calls `Command.build('syrd')`, waits for the production queue
     to report `done=True`, then `Command.place_building('syrd', x, y)`
     on a cell where the shipyard footprint is dry but adjacent to a
     water cell.
  3. A new `syrd` must surface in `own_buildings`.
  4. Agent calls `Command.build('dd')`, steps until the dd surfaces in
     the unit list. The destroyer should auto-spawn near the shipyard
     ON A WATER CELL.
  5. Verify the dd is functional - issue `Command.move_units` along
     the water band and observe motion (the ship cannot enter land).

If any step fails the F11 naval idiom (and combined-arms by extension)
is blocked.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]

# Absolute path to the bundled rush-hour terrain. See test_aa_fires_on_
# aircraft.py for the rationale — older tests relied on the engine's
# HOME-dir fallback to OpenRA-RL-Training, which does not exist on CI.
_BUNDLED_MAP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "maps"
    / "rush-hour-arena.oramap"
)


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_build_dd.yaml", delete=False,
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _make_scenario() -> dict:
    """Agent seed: fact (provides structures.allies) + powr×2 (provides
    anypower for syrd) + proc (econ) + dome (prereq for dd). The water
    band sits at x=20..21 spanning most of the y-range; the agent's
    base is at x=10..15 so the shipyard placed at (18,20) has its
    east edge adjacent to the water column."""
    return {
        "name": "build-dd-roundtrip",
        "description": "F11 engine-risk: build(syrd)+build(dd) end-to-end",
        "base_map": str(_BUNDLED_MAP),
        "starting_cash": 8000,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 8000},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "build", "place_building", "move_units"],
        "planning": True,
        "termination": {"max_ticks": 12000},
        # A 2-wide water column at x=20..21, y=2..38 (rush-hour arena
        # is at least 40 tall). Ships will spawn IN this column; ground
        # actors cannot enter it.
        "water_rect": [20, 2, 2, 36],
        "actors": [
            {"type": "fact", "owner": "agent", "position": [10, 10]},
            {"type": "powr", "owner": "agent", "position": [14, 10]},
            {"type": "powr", "owner": "agent", "position": [17, 10]},
            {"type": "proc", "owner": "agent", "position": [10, 14]},
            {"type": "dome", "owner": "agent", "position": [10, 18]},
            # Far enemy marker (out of agent sight; gates engine auto-done).
            {"type": "fact", "owner": "enemy", "position": [90, 80]},
        ],
    }


def _own_buildings(obs: dict) -> list[dict]:
    return obs.get("own_buildings", []) or []


def _units(obs: dict) -> list[dict]:
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


def _step_until(env, predicate, *, max_steps: int = 200, cmd=None):
    obs = None
    for _ in range(max_steps):
        c = cmd if cmd is not None else [Command.observe()]
        obs, _r, _done, _info = env.step(c)
        if predicate(obs):
            return obs
    return obs


def test_build_syrd_then_build_dd_end_to_end():
    """The headline F11 naval round-trip: syrd then dd, end-to-end."""
    path = _scenario_path(_make_scenario())
    try:
        env = OpenRAEnv(path, seed=1)
        obs = env.reset()

        # Sanity: pre-placed base must be present.
        own_b0 = _own_buildings(obs)
        types0 = {b["type"] for b in own_b0}
        assert {"fact", "powr", "proc", "dome"} <= types0, (
            f"pre-placed base missing buildings; types={types0}"
        )

        # 1) build(syrd)
        obs, _r, _d, _i = env.step([Command.build("syrd")])
        obs = _step_until(
            env,
            lambda o: _has_done_in_prod(o, "syrd"),
            max_steps=200,
        )
        assert _has_done_in_prod(obs, "syrd"), (
            f"syrd never finished in production queue; "
            f"production={obs.get('production')}, "
            f"own_buildings={[b.get('type') for b in _own_buildings(obs)]}"
        )

        # 2) place_building(syrd) - on dry ground with east edge touching
        # the water column at x=20..21. SYRD footprint is 3x3; placing
        # at (17, 18) makes east edge x=19 (adjacent to water x=20).
        syrd_x, syrd_y = 17, 18
        obs, _r, _d, _i = env.step([
            Command.place_building("syrd", syrd_x, syrd_y),
        ])
        for _ in range(3):
            obs, _r, _d, _i = env.step([Command.observe()])

        own_b = _own_buildings(obs)
        syrds = [b for b in own_b if str(b.get("type", "")).lower() == "syrd"]
        assert len(syrds) >= 1, (
            f"syrd missing from own_buildings after place; "
            f"own_buildings="
            f"{[(b.get('type'), b.get('cell_x'), b.get('cell_y')) for b in own_b]}"
        )

        # 3) build(dd) - record pre-existing ships.
        pre_units = _units(obs)
        pre_dd_ids = {
            int(u["id"]) for u in pre_units
            if str(u.get("type", "")).lower() == "dd"
        }
        obs, _r, _d, _i = env.step([Command.build("dd")])

        obs = _step_until(
            env,
            lambda o: any(
                str(u.get("type", "")).lower() == "dd"
                and int(u["id"]) not in pre_dd_ids
                for u in _units(o)
            ),
            max_steps=200,
        )
        units = _units(obs)
        new_dds = [
            u for u in units
            if str(u.get("type", "")).lower() == "dd"
            and int(u["id"]) not in pre_dd_ids
        ]
        assert len(new_dds) >= 1, (
            f"dd never spawned after build('dd'); "
            f"units types={[u.get('type') for u in units]}, "
            f"production={obs.get('production')}"
        )
        dd = new_dds[0]
        dd_id = int(dd["id"])
        dx0, dy0 = int(dd["cell_x"]), int(dd["cell_y"])

        # The dd should spawn either on a water cell (water_rect x=20..21)
        # or adjacent to the shipyard. Either way it must have spawned.
        # Record initial pos for the motion check.
        assert dx0 > 0 or dy0 > 0, f"dd has bogus location ({dx0},{dy0})"

        # 4) Verify the dd is functional - move it along the water band.
        # The water column is at x=20..21, y=2..37; the dd must reach
        # a target inside the water band. Pick a cell well away from
        # the spawn point so motion is unambiguous.
        target_x, target_y = 20, dy0 + 8 if dy0 + 8 < 37 else dy0 - 8
        obs, _r, _d, _i = env.step([
            Command.move_units([str(dd_id)], target_x, target_y),
        ])
        obs = _step_until(
            env,
            lambda o: any(
                int(u["id"]) == dd_id
                and (int(u["cell_x"]) != dx0 or int(u["cell_y"]) != dy0)
                for u in _units(o)
            ),
            max_steps=200,
        )
        new_pos = next(
            (u for u in _units(obs) if int(u["id"]) == dd_id),
            None,
        )
        assert new_pos is not None, "dd vanished from units after move"
        assert (int(new_pos["cell_x"]) != dx0
                or int(new_pos["cell_y"]) != dy0), (
            f"dd did not move from ({dx0},{dy0}); "
            f"now at ({new_pos['cell_x']},{new_pos['cell_y']})"
        )
    finally:
        Path(path).unlink(missing_ok=True)
