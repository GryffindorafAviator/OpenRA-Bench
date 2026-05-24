"""F11 engine-risk pin: an enemy AA defense (`agun`/`sam`) auto-fires
on the agent's heli and damages it.

This is the bench-side mirror of
`OpenRA-Rust/openra-sim/tests/test_aa_fires_on_aircraft.rs`. Exercised
via the Python `OpenRAEnv` boundary so the bench's scenario-loader,
shroud, and obs adapter all see the engagement.

Load-bearing for the Family-11 "all-air LOSES vs AA" wrong-arm trap.
Without AA-firing-on-heli, a model that ignores the wrong-arm trap and
goes all-heli would win for free.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_train import OpenRAEnv, Command  # type: ignore[import]

# Absolute path to the bundled rush-hour terrain. Older versions of this
# test referenced `base_map: "rush-hour-arena"` and relied on the Rust
# engine's HOME-dir fallback to `~/Projects/OpenRA-RL-Training/scenarios/
# maps/rush-hour-arena.oramap`. That path does not exist on CI runners
# (or any machine without the legacy OpenRA-RL-Training checkout), so the
# test now points at the bench's own bundled .oramap.
_BUNDLED_MAP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "maps"
    / "rush-hour-arena.oramap"
)


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_aa.yaml", delete=False,
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _make_scenario() -> dict:
    """Place an agent heli within range of an enemy agun. The heli is
    on HoldFire (stance:0) so its return-fire never overrides the AA
    damage measurement. A far enemy `fact` keeps the engine from
    auto-`done`ing while we measure HP."""
    return {
        "name": "aa-fires-on-aircraft",
        "description": "F11 engine-risk: enemy AA defense damages heli",
        "base_map": str(_BUNDLED_MAP),
        "starting_cash": 0,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 0},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe"],
        "planning": True,
        "termination": {"max_ticks": 12000},
        "actors": [
            # Agent heli — HoldFire so its own armament doesn't kill
            # the AA before the AA damages it.
            {"type": "heli", "owner": "agent", "position": [12, 12], "stance": 0},
            # Enemy agun (Allied AA gun) at (10, 10). Range 8c covers
            # (12, 12) with Chebyshev=2.
            {"type": "agun", "owner": "enemy", "position": [10, 10]},
            # Persistent enemy marker (engine auto-done guard).
            {"type": "fact", "owner": "enemy", "position": [80, 80]},
        ],
    }


def _heli_hp(obs: dict, heli_id: str | None = None) -> float | None:
    """Return the HP fraction of the (first) agent heli in the obs.
    The raw obs surfaces `unit_hp` as a dict keyed by id, paired with
    `unit_positions` carrying the actor_type."""
    units = obs.get("unit_positions", {}) or {}
    hps = obs.get("unit_hp", {}) or {}
    for aid, pos in units.items():
        if not isinstance(pos, dict):
            continue
        if heli_id is not None and str(aid) != heli_id:
            continue
        if pos.get("actor_type") == "heli":
            return hps.get(aid)
    return None


def _heli_id(obs: dict) -> str | None:
    units = obs.get("unit_positions", {}) or {}
    for aid, pos in units.items():
        if isinstance(pos, dict) and pos.get("actor_type") == "heli":
            return str(aid)
    return None


def test_enemy_agun_damages_agent_heli():
    """An agent heli sitting within range of an enemy AA gun must take
    damage. Fails the F11 wrong-arm trap if the heli HP stays at 1.0."""
    path = _scenario_path(_make_scenario())
    try:
        env = OpenRAEnv(path, seed=1)
        obs = env.reset()

        heli_id = _heli_id(obs)
        assert heli_id is not None, (
            f"agent heli missing from unit_positions; got "
            f"{obs.get('unit_positions')}"
        )
        initial_hp = _heli_hp(obs, heli_id)
        assert initial_hp is not None, "heli HP missing from obs"
        # The env's reset path runs a few warmup frames so the AA may
        # already have landed a burst before we observe — that's fine,
        # we just want the FINAL HP to be strictly lower than initial.

        # Step long enough for the auto-engage scan + several AA bursts.
        # AAStub: damage 2000, burst 2, reload 12 ticks. Heli HP=12000;
        # 3 bursts (~96 ticks) is enough to draw HP well below 1.0.
        for _ in range(30):
            obs, _r, _done, _info = env.step([Command.observe()])

        final_hp = _heli_hp(obs, heli_id)
        # The heli may have died (hp=None, removed from obs) — that's
        # still a passing outcome for the "AA damages aircraft" pin.
        if final_hp is None:
            return  # heli destroyed; AA definitely fired
        assert final_hp < initial_hp - 0.01, (
            f"heli should have taken damage from the enemy agun; "
            f"initial_hp={initial_hp}, final_hp={final_hp}"
        )
    finally:
        Path(path).unlink(missing_ok=True)
