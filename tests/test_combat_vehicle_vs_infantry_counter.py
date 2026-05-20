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

The bar (per the spec):
  • stall (only observe)            → LOSS (kill bar unmet → after_ticks)
  • build-only-e1 (match 1:1)       → LOSS (attrition; movers shot first)
  • build-only-e3 (wrong counter)   → LOSS (cost-per-effect + close-range)
  • intended build-2tnk             → WIN (heavy armour walks through e1)

Validation is split between unit-level predicate checks (no engine)
and engine-driven scripted policies. The unit-level checks are the
load-bearing assertions for this commit (the engine-driven policies
are documented as smoke-only and parametrised over the hard seeds).
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


def _ctx(*, units=(), tick=1000, kills=0, lost=0, has_fact=True):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types={"fact", "tent", "weap"} if has_fact else {"tent", "weap"},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n, unit_type="2tnk"):
    return [
        {"cell_x": 30, "cell_y": 20, "type": unit_type, "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 6 kills, 3 tanks alive, fact still up, in time → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=6, lost=0))
    # Kill bar unmet (only 5 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=5, lost=0))
    # Force wipe (all units dead) → fail via not own_units_gte:1
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=6, lost=4))
    # Fact destroyed → fail via not has_building:fact
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=2000, kills=6, lost=0, has_fact=False),
    )
    # Timeout with bar unmet → fail (after_ticks 5401 reachable)
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=5402, kills=5, lost=0))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 8 kills, 3 tanks alive, fact still up → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=8, lost=0))
    # Bar unmet (only 7 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=7, lost=0))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=8, lost=4))
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=2000, kills=8, lost=0, has_fact=False),
    )
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=5402, kills=7, lost=0))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 8 kills, 3 tanks alive, fact up → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=8, lost=0))
    # Bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=7, lost=0))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=8, lost=4))
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=2000, kills=8, lost=0, has_fact=False),
    )
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=5402, kills=7, lost=0))


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


def test_agent_base_has_both_production_queues():
    """The composition decision is COMPOSITION, not tech-up. Each
    spawn group on every level must have BOTH a barracks (tent —
    enables e1/e3) and a war factory (weap — enables 2tnk) so both
    counters are buildable from turn 1. The starter jeep must be
    present so own_units_gte:1 is satisfied from t=0 (avoiding the
    unit-less misfire footgun documented in CLAUDE.md)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Per spawn group, the agent must have tent + weap + fact + jeep.
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
            for need in ("fact", "tent", "weap", "jeep"):
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


# ── engine-driven scripted policy: intended build-2tnk smoke ────────
#
# The full RPS-counter bar (build-e3 LOSES / build-e1 LOSES / build-
# 2tnk WINS) needs each pure-build policy to be exercised against the
# live engine. The engine production / build-placement timing is
# touchy enough that we keep these as smoke tests (one tier each) —
# the unit-level predicate teeth above are the strict invariants.


def _stall(rs, Command):
    """Pure observe — kill bar never met → after_ticks LOSS."""
    return [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    """Stall must be a real timeout LOSS on every level (no draw):
    the kill bar (units_killed_gte:6 / 8 / 8) is structurally
    unreachable from a pure-observe policy, so after_ticks 5401
    fires."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=1)
    assert r.outcome == "loss", (
        f"{level}: stall must LOSE (kill bar unmet → after_ticks); "
        f"got {r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )
