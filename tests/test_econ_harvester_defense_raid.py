"""econ-harvester-defense-raid — ACTION capability validation.

Live extraction operation under recurring asset-targeted harass. The
agent owns a refinery, a single auto-routing harvester, and a small
escort parked at the depot on stance:0 (HoldFire). Recurring raider
2tnks spawn via scheduled_events and beeline at the harvester; the
only east→west crossing on the chokepoint map is a 4-cell corridor
at y=18..21. The escort must FLIP from hold-fire to active (set_stance
2/3 or explicit attack_unit / attack_move) AND be planted on the
corridor mouth so the raiders die on approach.

Bar (CLAUDE.md "no defect, no cheat"):
  - stall LOSES every tier / seed (escort never engages → raider
    crosses corridor + kills harv → unit_type_count_gte:{harv,1} fails).
  - intended (set_stance(escort, 2) + attack_unit raider when in vision
    else attack_move to corridor mouth (50,20)) WINS every tier /
    every hard seed.
  - hard tier defines ≥2 agent spawn_point groups (NORTH y=8 / SOUTH
    y=32) round-robined by seed.

The map (chokepoint-arena 96×40) forces every raid wave through the
single corridor — set_stance + position-on-corridor is the load-
bearing capability the geometry exposes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "econ-harvester-defense-raid.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """Pure observe — escort stays stance:0 and the raider crosses
    the corridor unmolested + kills the harv → LOSS."""
    return [Command.observe()]


def _intended(rs, Command):
    """Flip the escort active AND plant it on the corridor mouth at
    (50,20) so every inbound raider dies inside the corridor's
    weapon envelope. When a raider is in vision, attack_unit it
    directly (explicit order overrides stance:0)."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tanks = [u for u in units if u.get("type") == "2tnk"]
    raiders = [
        e for e in enemies
        if e.get("type") == "2tnk" and not e.get("is_building", False)
    ]
    cmds = []
    tank_ids = [str(t["id"]) for t in tanks]
    if tank_ids:
        # Stance:2 (Defend) — auto-fire in range, no auto-hunt.
        # stance:3 (AttackAnything) would walk the escort east and
        # destroy the persistent far enemy fact at (90,20), ending
        # the episode via enemy-elimination auto-`done`.
        cmds.append(Command.set_stance(tank_ids, 2))
    if raiders:
        for t in tanks:
            cmds.append(Command.attack_unit([str(t["id"])], str(raiders[0]["id"])))
    else:
        # Pre-position at corridor mouth — (50,20) sits INSIDE the
        # corridor where the escort's weapon envelope covers both
        # corridor entrances.
        for t in tanks:
            cmds.append(Command.attack_move([str(t["id"])], 50, 20))
    return cmds or [Command.observe()]


def _passive_at_base(rs, Command):
    """Defensive-but-passive: explicit stop on every escort, no
    stance flip, no attack order. Behaviourally identical to stall —
    the harv dies → LOSS."""
    units = rs.get("units_summary", []) or []
    tanks = [u for u in units if u.get("type") == "2tnk"]
    cmds = [Command.stop([str(t["id"])]) for t in tanks]
    return cmds or [Command.observe()]


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "chokepoint-arena map must materialize"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-harvester-defense-raid"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = pack.meta.benchmark_anchor
    joined = " ".join(anchors).lower()
    assert "sc2" in joined or "stance" in joined
    assert "chokepoint" in joined or "corridor" in joined or "route" in joined


def test_uses_raider_bot():
    """Raider bot's harv_foes() picks the agent's harvester each
    tick — bot-driven target binding is what makes set_stance the
    load-bearing verb."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "raider", f"expected raider bot, got {bot!r}"


def test_scheduled_events_present_per_tier():
    """Each tier MUST schedule at least one raid wave (the test
    premise — the recurring harass)."""
    pack = load_pack(PACK)
    expected = {"easy": 1, "medium": 2, "hard": 3}
    for lvl, n in expected.items():
        c = compile_level(pack, lvl)
        events = [
            e for e in c.scheduled_events
            if e.get("type") == "spawn_actors"
        ]
        assert len(events) == n, (
            f"{lvl}: expected {n} spawn_actors event(s), got {len(events)}"
        )


def test_escort_starts_stance_0_hold_fire():
    """Pre-placed escort 2tnks MUST start on stance:0 (HoldFire) —
    a stance:2/3 escort would auto-engage from spawn and trivialise
    the "flip stance" capability under test."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        escorts = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert escorts, f"{lvl}: pack has no escort 2tnks"
        for e in escorts:
            stance = getattr(e, "stance", None)
            if stance is None:
                raw = getattr(e, "model_extra", None) or {}
                stance = raw.get("stance")
            assert stance == 0, (
                f"{lvl}: escort at {e.position} must be stance:0, "
                f"got stance={stance!r}"
            )


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_all_tiers_have_reachable_deadlines():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        wt = next(
            int(c["within_ticks"])
            for c in L.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch"
        )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, cash=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (then:[after_ticks AND cash_gte]) AND harv alive
    AND proc alive AND in-time; fail fires on timeout OR harv dead
    OR proc destroyed."""
    c = compile_level(load_pack(PACK), "easy")
    one_harv = [{"cell_x": 12, "cell_y": 12, "type": "harv"}]
    proc = [("proc", 10, 12)]

    # Intended: past the latch tick AND cash bar met, harv alive,
    # proc alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=one_harv, tick=1800, cash=1200, own_buildings=proc),
    )
    # Harv dead → real fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=1800, cash=2000, own_buildings=proc),
    )
    # Proc destroyed → real fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=one_harv, tick=1800, cash=2000, own_buildings=[]),
    )
    # Past deadline → timeout fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=one_harv, tick=3002, cash=0, own_buildings=proc),
    )


# ── engine-driven (the no-cheat bar) ────────────────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    """Pure observe → escort never engages → raider kills the harv
    → unit_type_count_gte:{harv,1} fail clause fires → LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"cash={r.signals.cash} units_lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_passive_at_base_loses(level, seed):
    """Explicit stop on the escort, no stance flip — behaviourally
    identical to stall (stance:0 plus explicit stop) → LOSS."""
    _, r = _run(level, _passive_at_base, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: passive-at-base must LOSE; got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_corridor_defense_wins(level, seed):
    """set_stance(escort,2) + attack_unit raider OR attack_move to
    corridor mouth (50,20) — escort kills every raider on approach,
    harv survives, cash accumulates → WIN."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended corridor defense should WIN; "
        f"got {r.outcome} cash={r.signals.cash} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost} "
        f"t={r.turns}"
    )
