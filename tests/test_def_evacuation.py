"""def-evacuation — evacuate the doomed base east to the safe zone.

Wave-8 REASONING pack. The base WILL fall regardless of the agent's
play; the test is whether the agent abandons the sunk-cost defence and
routes the mobile force EAST to the pre-designated safe zone before
attrition / the deadline burns it down.

Bar (the doomed-base discrimination):
  • stall (only observe)            → LOSS (tanks die at the base)
  • hold-the-base (defend the fact) → LOSS (out-traded by heavies)
  • intended EVAC east              → WIN (≥3 tanks at safe zone)

The fact + proc ARE EXPECTED TO DIE — losing the base is NOT a
fail clause. Only attrition (`own_units_gte:N`) and the deadline
(`after_ticks`) are fail triggers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "def-evacuation.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(units_xy=(), tick=1000, killed=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # ≥3 tanks at the east NORTH-corridor safe zone (90,12) r=6,
    # ≥3 alive, in time → WIN (easy keeps the (90,12) corridor)
    safe3 = [(90, 11), (90, 12), (90, 13)]
    assert evaluate(c.win_condition, _ctx(safe3, tick=2000, killed=0, lost=2))
    # Same tanks but only 2 alive (lost 3) → survival cap fails
    assert not evaluate(c.win_condition, _ctx(safe3[:2], tick=2000, killed=0, lost=3))
    # Tanks still at the doomed base (10,12..14) — outside the safe zone
    home = [(10, 11), (10, 12), (10, 13)]
    assert not evaluate(c.win_condition, _ctx(home, tick=2000, killed=0, lost=0))
    # Full force-wipe → fail clause own_units_gte:1 fires
    assert evaluate(c.fail_condition, _ctx([], tick=1500, killed=0, lost=5))
    # Past deadline → real LOSS reachable within max_turns (interrupt
    # mode ~3000-3700 ticks across 50 turns; 2701 is reachable).
    assert evaluate(c.fail_condition, _ctx(safe3, tick=2800, killed=0, lost=0))
    # Sanity on the FIXED-step budget (also reachable in interrupt mode).
    assert 2701 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_medium_tighter_clock():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # medium pulls the corridor to the far-NORTH safe zone (90,6) r=6
    safe3 = [(90, 5), (90, 6), (90, 7)]
    # Intended: ≥3 in zone, ≥3 alive, before tick 2400 → WIN
    assert evaluate(c.win_condition, _ctx(safe3, tick=2000, killed=0, lost=2))
    # In zone but past tick 2400 → win clause fails on within_ticks
    assert not evaluate(c.win_condition, _ctx(safe3, tick=2500, killed=0, lost=2))
    # Past the deadline → real LOSS reachable
    assert evaluate(c.fail_condition, _ctx(safe3, tick=2500, killed=0, lost=0))
    assert 2401 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_two_safe_zones():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # FAR-NORTH safe zone (90,6) satisfies the any_of geometry
    safe_north = [(90, 5), (90, 6), (90, 7)]
    assert evaluate(c.win_condition, _ctx(safe_north, tick=2000, killed=0, lost=2))
    # FAR-SOUTH safe zone (90,34) also satisfies the any_of geometry
    safe_south = [(90, 33), (90, 34), (90, 35)]
    assert evaluate(c.win_condition, _ctx(safe_south, tick=2000, killed=0, lost=2))
    # Tanks at the WRONG centre y (90,20) — outside BOTH zones at r=6
    # ((90,20)-(90,6)=14>6 and (90,20)-(90,34)=14>6) → fails geometry
    assert not evaluate(
        c.win_condition,
        _ctx([(90, 19), (90, 20), (90, 21)], tick=2000, killed=0, lost=2),
    )
    # Past tighter deadline → real LOSS reachable
    assert evaluate(c.fail_condition, _ctx(safe_north, tick=2500, killed=0, lost=0))
    assert 2401 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 agent spawn_point groups so the
    seed flips the safe-zone corridor (NORTH vs SOUTH)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_losing_the_base_is_not_a_fail_clause():
    """The fact + proc ARE EXPECTED to die. The fail_condition must NOT
    fire on losing the base — only on full force-wipe or the clock.

    This is the load-bearing invariant of the doomed-base idiom: if
    `not has_building:fact` were in fail_condition, the intended EVAC
    play would lose the moment the base fell (which the spec
    designs to be inevitable), inverting the discrimination."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Serialize the fail clause and assert the doomed-base
        # negation idioms are NOT present.
        s = str(c.fail_condition.model_dump())
        assert "has_building" not in s, (
            f"{lvl}: fail_condition must not negate has_building "
            f"(the base is EXPECTED to die); got {s}"
        )


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "def-evacuation"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the BCP / emergency-management EVAC doctrines the
    # brief calls out.
    assert "bcp" in joined or "evacuation" in joined
    assert "emergency" in joined or "evac" in joined or "retreat" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: each level's `after_ticks` deadline fits
    inside max_turns.

    Interrupt mode (this pack uses enemy_unit_spotted +
    own_unit_destroyed) cuts each step short to ~60..75 ticks, so the
    effective interrupt-mode budget at max_turns 50 is ~3000..3700
    ticks (verified). All after_ticks deadlines must fit inside that
    interrupt-mode budget — not just the fixed-step 93+90·(N-1)."""
    pack = load_pack(PACK_PATH)
    deadlines = {"easy": 2701, "medium": 2401, "hard": 2401}
    # Conservative interrupt-mode budget: ~60 ticks/turn (lower bound).
    interrupt_budget = lambda mt: 60 * mt
    for lvl, dl in deadlines.items():
        c = compile_level(pack, lvl)
        # Fixed-step sanity (loose upper bound).
        assert dl <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks {dl} not reachable in fixed-step budget"
        )
        # Interrupt-mode budget (the actual one for this pack).
        assert dl <= interrupt_budget(c.max_turns), (
            f"{lvl}: after_ticks {dl} not reachable in interrupt-mode "
            f"budget ~{interrupt_budget(c.max_turns)} at max_turns {c.max_turns}"
        )


def test_persistent_far_east_enemy_fact_present():
    """Anti-DRAW marker (CLAUDE.md footgun #5): a persistent enemy
    actor must remain so the engine doesn't auto-done on enemy-elim
    before the win/fail evaluator runs on this retreat scenario."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        far_east = [
            a
            for a in c.scenario.actors
            if a.owner == "enemy"
            and a.type == "fact"
            and a.position[0] >= 100
        ]
        assert far_east, (
            f"{lvl}: missing persistent far-east enemy fact (anti-DRAW)"
        )


# ── engine-driven scripted policies ──────────────────────────────────
#
# The three-policy bar. All engine-driven tests guard on the Rust env
# wheel; predicate-level tests above run without it.


def _stall_policy(rs, Command):
    """Stall: only observe. The hunt assault wipes the immobile tanks
    at the doomed base inside the engagement window → own_units_gte:1
    fails → LOSS."""
    return [Command.observe()]


def _hold_the_base_policy(rs, Command):
    """Hold-the-base: attack_move ALL tanks INTO the assault at the
    fact. The 3× 3tnk + e3 swarm hard-counters 5× 2tnk at close range
    — the column dies before clearing the assault → own_units_gte:1
    fails → LOSS. This is the sunk-cost defence that the EVAC idiom
    correctly abandons."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    # The enemy assault sits adjacent to the fact at (10,20) — order
    # all tanks to attack_move INTO the assault cells.
    return [
        Command.attack_move([str(u["id"])], target_x=17, target_y=20)
        for u in units
    ]


def _make_intended_evac_east_policy():
    """Intended EVAC-east policy: on the FIRST observation latch the
    starting tank column's own latitude (median cell_y) and every turn
    order ALL surviving tanks east to (90, that-latitude). Evacuating
    along the column's OWN corridor lands the survivors in whatever
    safe zone sits at that latitude — easy's NORTH (90,12), medium's
    far-NORTH (90,6), or hard's seed-chosen far-NORTH (90,6) /
    far-SOUTH (90,34) — without hardcoding the zone. The fact + proc
    are abandoned — they are sunk cost in the doomed-base idiom."""
    state = {"ty": None}

    def pol(rs, Command):
        units = rs.get("units_summary", []) or []
        if not units:
            return [Command.observe()]
        if state["ty"] is None:
            ys = sorted(u["cell_y"] for u in units)
            state["ty"] = ys[len(ys) // 2]
        ty = state["ty"]
        return [
            Command.move_units([str(u["id"])], target_x=90, target_y=ty)
            for u in units
        ]

    return pol


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level — the hunt assault wipes the
    immobile tanks at the base → own_units_gte:1 fails (force-wipe)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome} "
            f"k={res.signals.units_killed} l={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_hold_the_base_policy_loses(level):
    """Hold-the-base must LOSE on every level — the heavy + rocket
    assault out-trades 5× 2tnk in melee → own_units_gte fails."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _hold_the_base_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: hold-the-base must LOSE; got {res.outcome} "
            f"k={res.signals.units_killed} l={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_evac_east_wins(level):
    """Intended EVAC-east must WIN on every level and every hard seed
    (1..4): from turn 1 route the surviving tanks toward the safe
    zone, end with ≥3 tanks inside the radius and ≥3 still alive."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    for s in seeds:
        pol = _make_intended_evac_east_policy()
        res = run_level(c, pol, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended EVAC-east must WIN; "
            f"got {res.outcome} k={res.signals.units_killed} "
            f"l={res.signals.units_lost}"
        )
