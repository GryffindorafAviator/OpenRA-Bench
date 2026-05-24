"""Armor-class damage multipliers — Python harness mirror of
`OpenRA-Rust/openra-sim/tests/test_armor_class_damage.rs`.

Pins the armor-class combat model at the wheel / RustEnvPool
boundary: each weapon scales its hit by a per-armor-class `Versus`
multiplier, AND a multi-armament actor picks the weapon best suited
to the target.

Background: the engine had the `Versus` multiplier wired but always
fired `weapons[0]`. e3 (rocket soldier) lists RedEye (anti-air) as
its PRIMARY armament and Dragon (anti-ground, anti-armor) as
SECONDARY, so it shot its anti-air missile at tanks and never used
its anti-armor Dragon. The fix adds target-aware weapon selection in
`world.rs` (`GameRules::best_weapon_against`) so e3 fires Dragon at
heavy armor.

Consequences pinned here:
  • e3 vs 2tnk — the rocket squad genuinely destroys heavy tanks
    (anti-armor counter works).
  • e1 vs 2tnk — the rifle's 10%-vs-heavy multiplier means a
    cost-equal rifle squad cannot break a tank (small-arms-vs-armor
    is feeble).

See `OpenRA-Bench/CLAUDE.md` "engine blockers" section and
`OpenRA-Rust/openra-sim/tests/test_armor_class_damage.rs`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

# Absolute path to the bundled rush-hour terrain. See test_aa_fires_on_
# aircraft.py for the rationale — older tests relied on the engine's
# HOME-dir fallback to OpenRA-RL-Training, which does not exist on CI.
_BUNDLED_MAP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "maps"
    / "rush-hour-arena.oramap"
)


def _scenario(actors: list[dict], max_ticks: int = 6000) -> dict:
    """Minimal arena scenario with explicit actor placement."""
    return {
        "name": "armor-class-test",
        "description": "armor-class damage fixture",
        "base_map": str(_BUNDLED_MAP),
        "starting_cash": 0,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 0},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "move_units", "attack_unit", "attack_move", "stop"],
        "planning": True,
        "termination": {"max_ticks": max_ticks},
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix="_armor.yaml", delete=False)
    # sort_keys=False is load-bearing — the scenario YAML loader is
    # key-order sensitive.
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _adapter_run(scenario: dict, n_steps: int, *, seed: int = 1):
    """Run for n_steps observe() calls; return (adapter, render_state)."""
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    path = _scenario_path(scenario)
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    ad = RustObsAdapter()
    try:
        obs = env.reset(seed=seed)
        ad.observe(obs)
        rs = ad.render_state()
        for _ in range(n_steps):
            obs, _r, done, _i = env.step([env.Command.observe()])
            ad.observe(obs, done=done)
            rs = ad.render_state()
            if done:
                break
        return ad, rs
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_e3_rockets_destroy_heavy_tanks_python():
    """A cost-fair rocket-soldier squad (anti-armor) must destroy a
    heavy-tank line. e3 fires its Dragon (anti-ground) — armor-class
    weapon selection routes the anti-armor warhead at the tanks.

    9× e3 (cost 2700) vs 3× 2tnk (cost 2550) — cost parity. Both
    sides are stance:3 (auto-hunt), so the squad engages without
    agent micro. Proof: `units_killed >= 3` — the rocket squad
    destroyed every tank."""
    actors: list[dict] = []
    for i in range(9):
        actors.append(
            {
                "type": "e3",
                "owner": "agent",
                "position": [40 + i % 3, 16 + i],
                "stance": 3,
            }
        )
    for i in range(3):
        actors.append(
            {
                "type": "2tnk",
                "owner": "enemy",
                "position": [44, 18 + i],
                "stance": 3,
            }
        )
    scen = _scenario(actors)

    # ~70 observe steps × ~90 ticks/step ≈ 6000 ticks — a generous
    # but finite engagement window.
    ad, _rs = _adapter_run(scen, n_steps=70)
    assert ad.signals.units_killed >= 3, (
        f"anti-armor e3 squad failed to destroy all 3 heavy tanks: "
        f"only {ad.signals.units_killed} kills credited"
    )


def test_rifles_cannot_break_heavy_armor_python():
    """A cost-equal rifle squad must NOT destroy a heavy tank inside
    a real engagement window — the M1Carbine's 10%-vs-heavy `Versus`
    multiplier makes small-arms feeble against armor.

    8× e1 (cost 800) vs 1× 2tnk (cost 800) — cost parity. The agent
    rifles are stance:3 and pile onto the tank; the tank (stance:0,
    HoldFire) does not fight back, isolating pure rifle-vs-armor
    damage. Proof: `units_killed == 0` — no rifle scores a kill, the
    tank survives."""
    actors: list[dict] = []
    for i in range(8):
        actors.append(
            {
                "type": "e1",
                "owner": "agent",
                "position": [40 + i % 4, 16 + i // 4],
                "stance": 3,
            }
        )
    # Tank on HoldFire so the measurement is pure rifle-vs-armor: no
    # tank kills muddy the `units_killed == 0` proof.
    actors.append(
        {
            "type": "2tnk",
            "owner": "enemy",
            "position": [44, 19],
            "stance": 0,
        }
    )
    # A realistic single-engagement window (~8 decision turns at
    # ~90 ticks/turn). The rifle's 10%-vs-heavy multiplier means a
    # cost-equal rifle squad cannot break a heavy tank inside a real
    # fight — by contrast `test_e3_rockets_destroy_heavy_tanks_python`
    # shows anti-armor rockets shred tanks. That asymmetry IS the
    # armor-class combat model.
    scen = _scenario(actors, max_ticks=2000)
    ad, _rs = _adapter_run(scen, n_steps=8)
    assert ad.signals.units_killed == 0, (
        f"a cost-equal rifle squad broke a heavy tank in a real "
        f"engagement window — small-arms must be feeble vs armor "
        f"(units_killed={ad.signals.units_killed})"
    )
