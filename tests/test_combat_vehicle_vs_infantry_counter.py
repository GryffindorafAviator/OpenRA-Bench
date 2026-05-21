"""combat-vehicle-vs-infantry-counter — rock-paper-scissors hard-counter
selection.

The pack tests CAPABILITY counter-selection: cash $2550 funds EITHER
3× 2tnk (medium tanks @ $850) OR 8× e3 (anti-tank rocket soldiers @
$300) OR up to 25× e1 (rifle infantry @ $100). The enemy is a pure-
infantry cluster (e1 mass). Tanks are the dominant hard counter
(heavy armour vs small-arms); rockets are the wrong counter (anti-
tank ordnance against soft targets — cost-per-effect waste + the
rocket squad's short stand-off + low HP gets out-DPSed by the rifle
mass); matching with own rifles is a 1:1 attrition that loses.

The win predicate is `unit_type_count_gte 2tnk:3 AND units_killed_gte
K AND has_building fact` — the 2tnk:3 clause is the load-bearing
anti-cheat: only a policy that ACTUALLY BUILDS the 3-tank fist can
clear the bar. (The armour-class engine fix on OpenRA-Rust main made
pre-placed agent combat units auto-fire effectively, so the starter
jeep is `stance: 0` HoldFire — it scouts, it cannot rack up kills on
its own.)

The bar (per the spec):
  • stall (only observe)            → LOSS (no 2tnk, no kills; the
    idle HoldFire jeep is hunted down → force-wipe / after_ticks)
  • build-only-e1 (match 1:1)       → LOSS (never builds 2tnk → the
    2tnk:3 clause is structurally unmet)
  • build-only-e3 (wrong counter)   → LOSS (never builds 2tnk → the
    2tnk:3 clause is structurally unmet)
  • intended build-2tnk             → WIN (3 medium tanks walk
    through the e1 mass; 2tnk:3 + kill bar both latch)

Validation is scripted (no model / network) — every policy is
exercised against the live engine on every level and every hard
seed 1..4.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-vehicle-vs-infantry-counter.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-vehicle-vs-infantry-counter"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the doctrines the brief calls out (per the Wave-7
    # spec): SC2 hard-counter + military RPS + capability-based defense.
    assert "sc2" in joined and "hard-counter" in joined
    assert "military" in joined and "rps" in joined
    assert "capability" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None
        # Cash MUST be $2550 (the budget is the load-bearing scarcity
        # — exactly 3× 2tnk = $2550 = the right counter; 8× e3 = $2400
        # = the wrong counter; 25× e1 = $2500 = the 1:1 match).
        assert c.starting_cash == 2550, (
            f"{lvl}: starting_cash must be 2550 (right-counter budget); "
            f"got {c.starting_cash}"
        )


def _ctx(*, tanks=0, tick=1000, kills=0, lost=0, has_fact=True, units=None):
    """Synthesize a WinContext for predicate-level checks.

    `tanks` synthesizes that many 2tnk units in `units_summary`;
    pass `units` explicitly to model a different composition (or an
    empty force).
    """
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types=(
            {"fact", "powr", "tent", "weap", "fix"}
            if has_fact
            else {"powr", "tent", "weap", "fix"}
        ),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    if units is None:
        units = [
            {"cell_x": 30, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)}
            for i in range(tanks)
        ]
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 3 tanks fielded, 6 kills, fact still up, in time → WIN
    assert evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=6))
    # Only 2 tanks fielded → 2tnk:3 clause unmet → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=2, tick=2000, kills=6))
    # Kill bar unmet (only 5 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=5))
    # Wrong counter: 8 e3 fielded, kill bar met, but 0 tanks → not a win
    e3s = [
        {"cell_x": 30, "cell_y": 20, "type": "e3", "id": str(2000 + i)}
        for i in range(8)
    ]
    assert not evaluate(c.win_condition, _ctx(units=e3s, tick=2000, kills=6))
    # Force wipe (all units dead) → fail via not own_units_gte:1
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=6, lost=4))
    # Fact destroyed → fail via not has_building:fact
    assert evaluate(
        c.fail_condition,
        _ctx(tanks=3, tick=2000, kills=6, has_fact=False),
    )
    # Timeout with bar unmet → fail (after_ticks 5401 reachable)
    assert evaluate(c.fail_condition, _ctx(tanks=3, tick=5402, kills=5))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 3 tanks, 8 kills, fact still up → WIN
    assert evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=8))
    # Only 2 tanks → 2tnk:3 clause unmet → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=2, tick=2000, kills=8))
    # Bar unmet (only 7 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=7))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=8, lost=4))
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(tanks=3, tick=2000, kills=8, has_fact=False),
    )
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(tanks=3, tick=5402, kills=7))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 3 tanks, 8 kills, fact up → WIN
    assert evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=8))
    # Only 2 tanks → 2tnk:3 clause unmet → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=2, tick=2000, kills=8))
    # Bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=7))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=8, lost=4))
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(tanks=3, tick=2000, kills=8, has_fact=False),
    )
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(tanks=3, tick=5402, kills=7))


def test_win_requires_three_medium_tanks():
    """The load-bearing anti-cheat: every level's win predicate must
    require `unit_type_count_gte 2tnk:3` — a stall / wrong-counter
    policy that never builds the medium-tank fist can never win
    regardless of how many kills the entrenched enemy concedes."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # 0 tanks, kill bar trivially exceeded, fact up, in time → NOT a win.
        assert not evaluate(c.win_condition, _ctx(tanks=0, tick=2000, kills=99)), (
            f"{lvl}: win must require 3 fielded 2tnk — a 0-tank policy "
            f"with the kill bar met must NOT win (anti-cheat clause)"
        )
        # 3 tanks + kill bar met + fact up + in time → WIN.
        assert evaluate(c.win_condition, _ctx(tanks=3, tick=2000, kills=99)), (
            f"{lvl}: 3 fielded 2tnk + kill bar met must WIN"
        )


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 5401 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 5401 <= max_tick, (
            f"{lvl}: after_ticks 5401 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )
        assert 5400 <= max_tick, (
            f"{lvl}: within_ticks 5400 > max reachable tick {max_tick}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the base latitude (NORTH y=12 /
    SOUTH y=28). Engine-roundtrip is asserted by
    tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_enemy_is_pure_infantry_no_anti_armour():
    """The whole pack premise is `enemy is anti-armour-WEAK`: every
    level's live enemy force must be PURE rifle infantry (e1) with NO
    rocket soldiers (e3) and NO tanks. The far persistent `fact`
    marker is allowed (engine auto-done mitigation)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        enemy_units = [a for a in c.scenario.actors if a.owner == "enemy"]
        types = [a.type for a in enemy_units]
        # No anti-armour units (e3 / 2tnk / 3tnk / etc.)
        for t in types:
            assert t not in ("e3", "2tnk", "3tnk", "4tnk", "ftrk", "v2rl"), (
                f"{lvl}: enemy must be PURE infantry (no anti-armour); "
                f"found {t}"
            )
        # Persistent far enemy marker (engine auto-done mitigation).
        assert "fact" in types, f"{lvl}: needs a persistent enemy fact"
        # ≥6 rifle infantry to satisfy the kill bar headroom.
        n_e1 = sum(1 for t in types if t == "e1")
        assert n_e1 >= 6, f"{lvl}: needs ≥6 e1 in the enemy cluster; got {n_e1}"


def test_starter_jeep_is_hold_fire():
    """The armour-class engine fix made pre-placed agent combat units
    auto-fire effectively. The starter jeep must be `stance: 0`
    (HoldFire) on every spawn group of every level so a pure-observe
    stall policy cannot rack up kills with it for free."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        jeeps = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "jeep"
        ]
        assert jeeps, f"{lvl}: needs a starter jeep"
        for j in jeeps:
            assert j.stance == 0, (
                f"{lvl}: starter jeep must be stance:0 (HoldFire) so a "
                f"stall policy cannot score free kills; got stance={j.stance}"
            )


def test_agent_base_can_build_both_counters():
    """The composition decision is COMPOSITION, not tech-up. Each
    spawn group on every level must have the buildings that make BOTH
    counters producible from turn 1: tent (e1/e3), weap+powr+fix
    (2tnk — the war-factory vehicle queue needs power online AND a
    service depot for the medium tank to clear its prerequisites).
    The starter jeep must also be present."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Per spawn group, the agent must have the full base + jeep.
        # On non-hard levels there is exactly one (default) spawn
        # group (spawn_point None → 0); on hard there are two.
        groups: dict[int, list] = {}
        for a in c.scenario.actors:
            if a.owner != "agent":
                continue
            g = a.spawn_point if a.spawn_point is not None else 0
            groups.setdefault(g, []).append(a.type)
        assert groups, f"{lvl}: no agent actors found"
        for g, ts in groups.items():
            for need in ("fact", "powr", "tent", "weap", "fix", "jeep"):
                assert need in ts, (
                    f"{lvl}: spawn group {g} missing {need}; got {ts}"
                )


def test_starting_cash_funds_exactly_one_pure_composition():
    """The budget ($2550) is the load-bearing scarcity. The intended
    counter (3× 2tnk @ $850) costs exactly $2550. The wrong-counter
    plays (8× e3 @ $300 = $2400; 25× e1 @ $100 = $2500) also fit but
    are dominated on combat outcome — that's the discrimination."""
    pack = load_pack(PACK_PATH)
    assert pack.starting_cash == 2550, (
        f"pack starting_cash must be 2550 (intended-counter exact budget); "
        f"got {pack.starting_cash}"
    )
    # 3× 2tnk = 2550 (the right counter, full commit, no waste)
    assert 3 * 850 == 2550
    # 8× e3 = 2400 (the wrong counter — anti-tank rockets on soft
    # infantry; cost-per-effect waste)
    assert 8 * 300 == 2400
    # 25× e1 = 2500 (1:1 attrition match — no positional advantage)
    assert 25 * 100 == 2500


# ── engine-driven scripted policies ─────────────────────────────────
#
# The full RPS-counter bar (stall LOSES / build-e3 LOSES / build-e1
# LOSES / build-2tnk WINS) is exercised against the live engine on
# every level and every hard seed 1..4.


def _own_units(rs, *, type_filter=None):
    out = []
    for u in (rs.get("units_summary", []) or []):
        if type_filter and (u.get("type") or "").lower() not in type_filter:
            continue
        out.append(u)
    return out


def _enemy_infantry(rs):
    return [
        e for e in (rs.get("enemy_summary") or [])
        if (e.get("type") or "").lower() == "e1" and not e.get("is_building")
    ]


def _stall(rs, Command):
    """Pure observe — no production. The HoldFire jeep never fires →
    0 kills, 0 tanks; the 2tnk:3 win clause never latches → LOSS
    (force-wipe when the e1 swarm hunts the jeep, or after_ticks)."""
    return [Command.observe()]


def _make_build_policy(unit_type, cost):
    """Queue `unit_type` every turn the budget allows and send each
    produced unit at the enemy infantry cluster."""

    def policy(rs, Command):
        cmds = []
        if rs.get("cash", 0) >= cost:
            cmds.append(Command.build(unit_type))
        fighters = _own_units(rs, type_filter={unit_type})
        targets = _enemy_infantry(rs)
        for u in fighters:
            if targets:
                cmds.append(
                    Command.attack_unit([str(u["id"])], str(targets[0]["id"]))
                )
            else:
                cmds.append(Command.attack_move([str(u["id"])], 70, 20))
        return cmds if cmds else [Command.observe()]

    return policy


_build_e3 = _make_build_policy("e3", 300)
_build_e1 = _make_build_policy("e1", 100)
_build_2tnk = _make_build_policy("2tnk", 850)


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    """Stall must be a real LOSS on every level and every hard seed
    (no draw): the win predicate requires `unit_type_count_gte
    2tnk:3` which a pure-observe policy can never satisfy, and the
    idle HoldFire jeep is hunted down → force-wipe / after_ticks."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _stall, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: stall must be a real LOSS (no 2tnk → "
            f"win clause unmet); got {r.outcome} after {r.turns} turns "
            f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_e3_wrong_counter_loses(level):
    """Mass anti-tank rockets are the WRONG counter — and crucially
    the policy never builds 2tnk, so the `unit_type_count_gte 2tnk:3`
    win clause is structurally unmet → real LOSS on every level and
    every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _build_e3, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: build-e3 wrong-counter must LOSE (no "
            f"2tnk → win clause unmet); got {r.outcome} "
            f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_e1_wrong_counter_loses(level):
    """Matching the enemy 1:1 with own rifles never builds 2tnk, so
    the `unit_type_count_gte 2tnk:3` win clause is structurally unmet
    → real LOSS on every level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _build_e1, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: build-e1 wrong-counter must LOSE (no "
            f"2tnk → win clause unmet); got {r.outcome} "
            f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_build_2tnk_wins(level):
    """The RPS counter pick: build 3× 2tnk (medium tanks) and engage.
    Heavy armour walks through the e1 mass — the `2tnk:3` clause and
    the kill bar both latch. Wins on every level and every hard
    seed 1..4."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _build_2tnk, seed=s)
        assert r.outcome == "win", (
            f"{level} seed={s}: intended build-2tnk must WIN; got "
            f"{r.outcome} (kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )
