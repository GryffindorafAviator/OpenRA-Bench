"""build-sequence-tech-most-resilient pack — full no-cheat validation.

Wave-11 REASONING — robust build-order planning. The agent must REACH
a tech capability (a powered war factory) AND KEEP it through a
mid-episode strike. A scheduled `destroy_actors` event razes the one
exposed power plant at tick 1500. A build order that provisions only
ONE `powr` is a single point of failure — when the strike lands the
grid goes negative, the war factory drops to 50% production speed
(engine low-power slowdown) and `building_count_gte:{powr,1}` is
FALSE for the rest of the episode. The resilient build order
pre-builds a SECOND, redundant `powr` in the safe west base BEFORE
the strike: one plant survives, the grid stays in surplus, the army
finishes on the clock.

Bar (CLAUDE.md "no defect, no cheat"):
  - stall (observe only)            ⇒ LOSS on every (level, seed)
  - single-powr (no redundant powr) ⇒ LOSS on every (level, seed)
  - intended resilient (redundant   ⇒ WIN  on every (level, seed)
    2nd powr, then weap, then 3×2tnk)
Real LOSS not DRAW: `fail after_ticks:T+1` reachable inside
max_turns; the persistent far enemy `fact` blocks the engine
auto-done path.

Scenario shape:
  - rush-hour-arena, allies vs soviet (bot disabled).
  - easy:   within_ticks 5400, max_turns 60 — generous.
  - medium: within_ticks 4500, max_turns 50 — tighter clock.
  - hard:   within_ticks 4500, max_turns 50 — +2 spawn_point groups
            (NORTH base y=12 / SOUTH base y=26, round-robined).

Measured (seed 1, scripted policies): the intended resilient policy
WINS at ~tick 3243 on every level; stall and single-powr LOSE on the
deadline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-sequence-tech-most-resilient.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _single_powr_policy():
    """Single-point-of-failure play: build the war factory and spam
    `2tnk`, but NEVER build a redundant power plant. The lone exposed
    `powr` is razed at tick 1500 → the grid goes negative → 50%
    production AND `building_count_gte:{powr,1}` is false → LOSS."""
    ms = {"weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        base = [b for b in ob if b["type"] == "fact"]
        cmds = []
        if "weap" in own:
            ms["weap"] = True
        if not ms["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 5, base[0]["cell_y"]
                ))
        else:
            if "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))
        return cmds or [Cmd.observe()]
    return pol


def _intended_policy():
    """Resilient N+1 build order: build a redundant `powr` in the safe
    west base (placed relative to the actual Construction Yard so it
    generalises across the hard-tier spawn variation), then the war
    factory, then 3× `2tnk`. Must WIN on every (level, seed)."""
    ms = {"powr2": False, "weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        base = [b for b in ob if b["type"] == "fact"]
        # The redundant powr lives in the safe west base (x<30);
        # the exposed inherited powr sits forward at x=40.
        safe_powr = [
            b for b in ob if b["type"] == "powr" and b["cell_x"] < 30
        ]
        weap_b = [b for b in ob if b["type"] == "weap"]
        cmds = []
        if safe_powr:
            ms["powr2"] = True
        if weap_b:
            ms["weap"] = True
        if not ms["powr2"]:
            if "powr" not in prod:
                cmds.append(Cmd.build("powr"))
            if base:
                cmds.append(Cmd.place_building(
                    "powr", base[0]["cell_x"] + 3, base[0]["cell_y"] + 4
                ))
        elif not ms["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 6, base[0]["cell_y"]
                ))
        else:
            if "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))
        return cmds or [Cmd.observe()]
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-sequence-tech-most-resilient"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """meta.benchmark_anchor must cite PlanBench robust planning,
    N+1 resilient design and redundancy (the seed taxonomy)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("N+1" in a or "resilient" in a for a in anchors), anchors
    assert any("redundancy" in a.lower() for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_then_composite_used_in_win():
    """The win must wire the powr→weap happened-before chain — the
    'reach the tech capability in order' clause."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        then = next((cl["then"] for cl in inner if "then" in cl), None)
        assert then is not None, f"{lvl} win missing then-chain: {win}"
        clauses = then.get("clauses") or []
        assert len(clauses) == 2, (
            f"{lvl} then-chain must be powr→weap (2 clauses); got {clauses}"
        )
        assert clauses[0].get("has_building") == "powr"
        assert clauses[1].get("has_building") == "weap"


def test_win_requires_surviving_powr_three_tanks_and_fact():
    """Structural: the win clause must require a LIVE Power Plant
    (`building_count_gte:{powr,1}` — the redundancy teeth that toggle
    FALSE when the exposed powr is razed), three medium tanks
    (`unit_type_count_gte:{2tnk,3}`), a live Construction Yard, and a
    `within_ticks` deadline. `building_count_gte` (live-list) — NOT
    `has_building` (accumulating set) — is mandatory for the powr
    clause so it toggles false on the strike."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        all_of = c.win_condition.model_dump(exclude_none=True).get("all_of", [])
        powr = next(
            (x["building_count_gte"] for x in all_of
             if "building_count_gte" in x
             and (x["building_count_gte"] or {}).get("type") == "powr"),
            None,
        )
        assert powr is not None and int(powr.get("n", 0)) >= 1, (
            f"{lvl}: win must require building_count_gte powr>=1 "
            f"(a live power plant survives the strike)"
        )
        tanks = next(
            (x["unit_type_count_gte"] for x in all_of
             if "unit_type_count_gte" in x
             and (x["unit_type_count_gte"] or {}).get("type") == "2tnk"),
            None,
        )
        assert tanks is not None and int(tanks.get("n", 0)) >= 3, (
            f"{lvl}: win must require unit_type_count_gte 2tnk>=3"
        )
        fact = next(
            (x["building_count_gte"] for x in all_of
             if "building_count_gte" in x
             and (x["building_count_gte"] or {}).get("type") == "fact"),
            None,
        )
        assert fact is not None and int(fact.get("n", 0)) >= 1, (
            f"{lvl}: win must require building_count_gte fact>=1"
        )
        assert any("within_ticks" in x for x in all_of), (
            f"{lvl}: win must include a within_ticks deadline"
        )


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns and the fail
    `after_ticks` must equal within_ticks+1 (real LOSS, no draw, no
    overlap). Engine advances ~90 ticks/turn → reachable = 93 +
    90·(max_turns-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        reachable = 93 + 90 * (c.max_turns - 1)
        all_of = c.win_condition.model_dump(exclude_none=True).get("all_of", [])
        wt = next(int(x["within_ticks"]) for x in all_of if "within_ticks" in x)
        assert wt <= reachable, (
            f"{lvl}: within_ticks={wt} > reachable={reachable} "
            f"(max_turns={c.max_turns}) — deadline never bites"
        )
        fail = c.fail_condition.model_dump(exclude_none=True)
        after = next(
            int(x["after_ticks"]) for x in fail["any_of"] if "after_ticks" in x
        )
        assert after <= reachable, (
            f"{lvl}: fail after_ticks {after} unreachable within "
            f"{c.max_turns} turns (max {reachable}) — draw degeneracy"
        )
        assert after == wt + 1, (
            f"{lvl}: after_ticks {after} must equal within_ticks+1 ({wt+1})"
        )


def test_exactly_one_exposed_powr_pre_placed():
    """The single-point-of-failure premise: each tier pre-places
    EXACTLY ONE agent `powr` (the exposed forward plant). The
    redundant second power plant must be BUILT by the agent — it is
    not given. Hard duplicates the base across two spawn groups, so
    each spawn group still ships exactly one exposed powr."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        powrs = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "powr"
        ]
        if lvl == "hard":
            per_spawn = {}
            for a in powrs:
                sp = a.spawn_point if a.spawn_point is not None else 0
                per_spawn[sp] = per_spawn.get(sp, 0) + 1
            assert per_spawn and all(v == 1 for v in per_spawn.values()), (
                f"hard: each spawn group must pre-place exactly one "
                f"exposed powr; got {per_spawn}"
            )
        else:
            assert len(powrs) == 1, (
                f"{lvl}: must pre-place exactly one exposed agent powr; "
                f"got {len(powrs)}"
            )


def test_scheduled_destroy_event_razes_the_exposed_powr():
    """Each tier must declare a `scheduled_events: destroy_actors`
    that fires mid-episode (before the deadline) on the agent, with a
    region tight around the exposed forward powr (x≈40) so it can
    never catch a redundant powr placed in the safe west base."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        evs = c.scheduled_events or []
        destroys = [e for e in evs if e.get("type") == "destroy_actors"]
        assert destroys, f"{lvl}: needs a destroy_actors scheduled event"
        for e in destroys:
            assert e["filter"]["owner"] == "agent"
            reg = e["filter"]["region"]
            assert reg["x"] == 40, (
                f"{lvl}: strike region must be centred on the exposed "
                f"forward powr at x=40; got {reg}"
            )
            assert e["tick"] < 4500, (
                f"{lvl}: strike must fire mid-episode (before the "
                f"deadline); got tick {e['tick']}"
            )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define >=2 agent spawn_point groups so the seed
    varies the start base (tests/test_hard_tier.py::UPGRADED)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs >=2 spawn groups, got {sp}"


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_resilient_policy_wins(level, seed):
    """The intended resilient play (redundant 2nd powr → weap → 3×
    2tnk) must WIN on every (level, seed). The load-bearing test that
    the pack is solvable inside the budget by the advertised
    robust-planning capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(), seed=seed)
    assert res.outcome == "win", (
        f"intended resilient must WIN on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"buildings={sorted(res.signals.own_building_types)})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_policy_loses(level, seed):
    """A stall (observe-only) builds nothing — the exposed powr is
    razed, no weap, no tanks → must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_single_powr_policy_loses(level, seed):
    """The single-point-of-failure play — build the war factory and
    produce tanks but NEVER a redundant power plant — must LOSE on
    every (level, seed): the strike razes the lone powr, so
    `building_count_gte:{powr,1}` is false at the deadline."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _single_powr_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"single-powr (no redundancy) must LOSE on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick})"
    )
