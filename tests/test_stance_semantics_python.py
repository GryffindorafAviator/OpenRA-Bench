"""Stance semantics — Python harness mirror of
`OpenRA-Rust/openra-sim/tests/test_stance_semantics.rs`.

Pins the three behaviourally-distinct stances at the wheel /
RustEnvPool boundary so packs that key on a stance flip
(`combat-stance-mgmt-attack`, `def-stance-mgmt-hold-then-attack`)
remain solvable:

  • stance:0 (HoldFire)       — never auto-engages.
  • stance:1 (ReturnFire)     — fires ONLY at an attacker that has
                                 recently damaged this unit.
  • stance:3 (AttackAnything) — fires at in-range enemies AND
                                 ADVANCES toward visible enemies
                                 beyond weapon range (hunt).

Background: the engine originally collapsed stance:1 and stance:3
into stance:2 (auto-fire on in-range enemies, never advance). The
fix in `world.rs::tick_actors`:
  - tracks `recently_received_fire: HashMap<actor_id, world_tick>`
    via the damage-apply loops; stance:1 auto-engagement is gated
    on a fresh hit within a 60-tick window.
  - adds a hunt path: stance:3 idle units with no in-range enemy
    but a visible-but-out-of-range enemy issue `order_move` toward
    that enemy; the next-tick scan promotes the now-in-range
    encounter into an Attack.

See `OpenRA-Bench/CLAUDE.md` "engine blockers" section.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")


def _scenario(actors: list[dict], max_ticks: int = 3000) -> dict:
    """Minimal rush-hour-arena scenario with explicit actor placement."""
    return {
        "name": "stance-semantics-test",
        "description": "stance semantics fixture",
        "base_map": "rush-hour-arena",
        "starting_cash": 0,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 0},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": [
            "observe",
            "move_units",
            "attack_unit",
            "attack_move",
            "stop",
            "set_stance",
        ],
        "planning": True,
        "termination": {"max_ticks": max_ticks},
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix="_stance.yaml", delete=False)
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _enemy_units(rs: dict) -> list[dict]:
    """Visible enemy units (excludes buildings)."""
    return [
        u
        for u in (rs.get("enemy_summary") or [])
        if not u.get("is_building")
    ]


def _own_unit_by_type(rs: dict, atype: str) -> dict | None:
    for u in rs.get("units_summary") or []:
        if str(u.get("type", "")).lower() == atype.lower():
            return u
    return None


def _enemy_hp_fraction(rs: dict, atype: str) -> float | None:
    """HP fraction (0..1) for first visible enemy of type `atype`,
    or None if no such enemy is visible (dead or out of sight)."""
    for u in _enemy_units(rs):
        if str(u.get("type") or "").lower() == atype.lower():
            return float(u.get("hp") or 0.0)
    return None


def _own_unit_pos(rs: dict, atype: str) -> tuple[int, int] | None:
    u = _own_unit_by_type(rs, atype)
    if u is None:
        return None
    return (int(u.get("cell_x", 0)), int(u.get("cell_y", 0)))


def _adapter_run(scenario: dict, n_steps: int, *, seed: int = 1):
    """Run the scenario for n_steps observe() calls and yield the
    final RustObsAdapter (with last render_state) plus the env."""
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    path = _scenario_path(scenario)
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    ad = RustObsAdapter()
    try:
        obs = env.reset(seed=seed)
        ad.observe(obs)
        for _ in range(n_steps):
            obs, _r, done, _i = env.step([env.Command.observe()])
            ad.observe(obs, done=done)
            if done:
                break
        return ad, ad.render_state()
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


# ── stance:0 contract ────────────────────────────────────────────


def test_stance_0_holds_fire_python():
    """Agent 2tnk on stance:0 must NOT engage an attacking enemy
    even after 100+ ticks (HoldFire blocks all auto-engagement).
    The proof is `units_killed == 0` — the tank never landed a
    blow."""
    scen = _scenario(
        [
            # Persistent enemy fact so engine doesn't auto-`done`.
            {"type": "fact", "owner": "enemy", "position": [120, 20]},
            {"type": "2tnk", "owner": "agent", "position": [40, 20], "stance": 0},
            {"type": "e1", "owner": "enemy", "position": [43, 20], "stance": 3},
        ]
    )
    ad, _rs = _adapter_run(scen, n_steps=8)
    assert ad.signals.units_killed == 0, (
        f"stance:0 tank scored kills despite HoldFire; "
        f"units_killed={ad.signals.units_killed}"
    )


# ── stance:1 contract ────────────────────────────────────────────


def test_stance_1_holds_fire_against_passive_enemy_python():
    """Agent 2tnk on stance:1 paired with an enemy e1 on stance:0
    (never fires first). The tank must NOT engage — ReturnFire
    requires having received fire. units_killed == 0 proves it."""
    scen = _scenario(
        [
            {"type": "fact", "owner": "enemy", "position": [120, 20]},
            {"type": "2tnk", "owner": "agent", "position": [40, 20], "stance": 1},
            {"type": "e1", "owner": "enemy", "position": [43, 20], "stance": 0},
        ]
    )
    ad, _rs = _adapter_run(scen, n_steps=8)
    assert ad.signals.units_killed == 0, (
        f"stance:1 tank fired on a passive (never-attacking) enemy; "
        f"units_killed={ad.signals.units_killed} — ReturnFire must "
        f"require an actual attack to unlock"
    )


def test_stance_1_returns_fire_on_attacker_python():
    """Agent 2tnk on stance:1 paired with an aggressive enemy e1
    (stance:3) in range must return fire and eventually kill the
    attacker — proof via units_killed >= 1."""
    scen = _scenario(
        [
            {"type": "fact", "owner": "enemy", "position": [120, 20]},
            {"type": "2tnk", "owner": "agent", "position": [40, 20], "stance": 1},
            {"type": "e1", "owner": "enemy", "position": [43, 20], "stance": 3},
        ]
    )
    ad, _rs = _adapter_run(scen, n_steps=15)
    assert ad.signals.units_killed >= 1, (
        f"stance:1 tank failed to return fire on its attacker; "
        f"units_killed={ad.signals.units_killed} after 15 turns"
    )


# ── stance:3 contract ────────────────────────────────────────────


def test_stance_3_hunts_visible_enemy_python():
    """Agent 2tnk on stance:3 paired with an enemy 15 cells away
    (out of cannon range ≈5 cells) must ADVANCE and eventually
    kill the enemy ("hunt" semantics)."""
    scen = _scenario(
        [
            {"type": "fact", "owner": "enemy", "position": [120, 20]},
            # Friendly scouts strung along the corridor so the
            # target is in fact-shared sight from t=0 (the hunter's
            # own RevealsShroud may be ~7 cells, less than the 15-
            # cell gap; with scouts visible the player can "see"
            # the target).
            {"type": "2tnk", "owner": "agent", "position": [10, 20], "stance": 3},
            {"type": "e1", "owner": "agent", "position": [15, 20], "stance": 0},
            {"type": "e1", "owner": "agent", "position": [20, 20], "stance": 0},
            {"type": "e1", "owner": "enemy", "position": [25, 20], "stance": 0},
        ],
        max_ticks=4000,
    )
    start_pos = (10, 20)
    ad, rs = _adapter_run(scen, n_steps=25)
    end_pos = _own_unit_pos(rs, "2tnk")
    assert ad.signals.units_killed >= 1, (
        f"stance:3 tank failed to HUNT a visible-but-out-of-range "
        f"enemy; units_killed={ad.signals.units_killed} after 25 "
        f"turns. Tank start={start_pos} end={end_pos}"
    )
    if end_pos is not None:
        assert end_pos[0] > start_pos[0], (
            f"stance:3 tank killed the enemy without moving east "
            f"(start={start_pos}, end={end_pos}); something else "
            f"scored the kill"
        )
