"""build-sequence-tech-most-resilient pack — full no-cheat validation.

REASONING — robust build-order planning under spatial attrition. The
two-base strategic-retreat template (task #81 redesign):

  * HOME — a complete safe base in the deep west (fact + proc + powr +
    fix + harv + ore patch).
  * FORWARD — a partial doomed base in the east (fact + proc + powr).
  * Threat — a `scheduled_events: destroy_actors` rotation razes the
    forward zone at ticks 1200/2400/3600. Each wave kills anything
    placed in the forward strike radius since the previous wave.

Win predicate is STATE-BASED at the deadline (NOT a sequence latch):
  * a War Factory alive (any base)
  * 3 medium tanks alive
  * a Power Plant alive (any base)
  * a Construction Yard alive (any base)
  * within the deadline

Bar (CLAUDE.md "no defect, no cheat"):
  - stall (observe only)            ⇒ LOSS on every (level, seed)
  - build-at-FORWARD only           ⇒ LOSS on every (level, seed)
  - build-at-HOME (intended)        ⇒ WIN  on every (level, seed)
  - build-at-BOTH (redundancy)      ⇒ WIN  on every (level, seed)

Real LOSS not DRAW: `fail after_ticks:T+1` is reachable inside
max_turns; the persistent far enemy `fact` blocks the engine
auto-done path.
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


def _build_at_forward_policy():
    """Build at the FORWARD doomed base. Cache the forward placement
    cell at T0 (the strike rotation razes the forward fact mid-
    episode, so the policy must REMEMBER the forward cell to keep
    re-targeting it even after the forward fact is destroyed and
    rebuilt). The forward strike razes the rebuilt war factory on
    the next wave → LOSS on every (level, seed)."""
    ms = {"weap": False, "forward_cell": None}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        if ms["forward_cell"] is None:
            facts = [b for b in ob if b["type"] == "fact"]
            if facts:
                forward_fact = max(facts, key=lambda b: b["cell_x"])
                ms["forward_cell"] = (
                    forward_fact["cell_x"] + 2, forward_fact["cell_y"]
                )
        if ms["forward_cell"] is None:
            return [Cmd.observe()]
        # Cached forward x is >= 30; if a weap exists there, latch.
        weap_b = [
            b for b in ob if b["type"] == "weap" and b["cell_x"] >= 30
        ]
        if weap_b:
            ms["weap"] = True
        cmds = []
        if not ms["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building(
                "weap", ms["forward_cell"][0], ms["forward_cell"][1]
            ))
        else:
            if "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))
        return cmds or [Cmd.observe()]
    return pol


def _build_at_home_policy():
    """Intended resilient play: build the war factory at the safe HOME
    base (the westernmost fact), then produce 3× 2tnk. Must WIN on
    every (level, seed)."""
    ms = {"weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        facts = [b for b in ob if b["type"] == "fact"]
        if not facts:
            return [Cmd.observe()]
        # HOME fact is the WESTERNMOST agent fact (x≈8).
        home_fact = min(facts, key=lambda b: b["cell_x"])
        weap_b = [
            b for b in ob if b["type"] == "weap" and b["cell_x"] < 30
        ]
        if weap_b:
            ms["weap"] = True
        cmds = []
        if not ms["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building(
                "weap", home_fact["cell_x"] + 6, home_fact["cell_y"]
            ))
        else:
            if "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))
        return cmds or [Cmd.observe()]
    return pol


def _build_at_both_policy():
    """Redundancy play: build a war factory at BOTH bases. HOME copy
    survives the strike rotation, so the win clauses are satisfied
    even if the forward copy is razed — must WIN on every (level,
    seed). The pack must credit preemptive spatial redundancy.

    Order: HOME weap first → start tank production → ATTEMPT forward
    weap once (will eventually be razed but that's the point). The
    HOME weap keeps the win clause live."""
    ms = {
        "weap_home": False,
        "forward_attempted": False,
        "home_cell": None,
        "forward_cell": None,
    }

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        # Cache the two bases' build-target cells at first sight (the
        # forward fact gets destroyed mid-episode, so we can't look
        # it up every turn).
        if ms["home_cell"] is None or ms["forward_cell"] is None:
            facts = [b for b in ob if b["type"] == "fact"]
            if facts:
                home_fact = min(facts, key=lambda b: b["cell_x"])
                forward_fact = max(facts, key=lambda b: b["cell_x"])
                ms["home_cell"] = (
                    home_fact["cell_x"] + 6, home_fact["cell_y"]
                )
                ms["forward_cell"] = (
                    forward_fact["cell_x"] + 2, forward_fact["cell_y"]
                )
        if ms["home_cell"] is None:
            return [Cmd.observe()]
        home_weap = [
            b for b in ob if b["type"] == "weap" and b["cell_x"] < 30
        ]
        if home_weap:
            ms["weap_home"] = True
        cmds = []
        if not ms["weap_home"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building(
                "weap", ms["home_cell"][0], ms["home_cell"][1]
            ))
        elif not ms["forward_attempted"]:
            # Try the forward weap ONCE (the "redundancy" attempt).
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building(
                "weap", ms["forward_cell"][0], ms["forward_cell"][1]
            ))
            # Latch after one tick of placement attempts (the build
            # may take many turns to complete — but once we've queued
            # the placement orders alongside production, we move on
            # to tanks so the HOME weap is exercised).
            ms["forward_attempted"] = True
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
    """meta.benchmark_anchor must cite the robust-planning + spatial-
    redundancy taxonomy the redesigned pack belongs to."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a or "robust" in a.lower() for a in anchors), anchors
    assert any("redundancy" in a.lower() or "hazard" in a.lower() for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_win_is_state_based_no_sequence_latch():
    """The redesigned win is STATE-BASED — gates on what is alive at
    the deadline (weap + 2tnk + powr + fact), not on a happened-before
    sequence. A `then:` chain on the build sequence would invert the
    redundancy play (since the preemptive copy never observes a
    'destroyed' frame). The fix asserts NO `then:` in the win."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        for clause in inner:
            assert "then" not in clause, (
                f"{lvl} win must be state-based; "
                f"found a then-chain clause: {clause}"
            )


def test_win_requires_weap_three_tanks_powr_and_fact():
    """Structural: the win clause must require a LIVE War Factory,
    3 medium tanks, a live Power Plant, a live Construction Yard,
    and a within_ticks deadline. `building_count_gte` (live-list)
    is mandatory so the clause toggles false when the asset is
    razed."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        all_of = c.win_condition.model_dump(exclude_none=True).get("all_of", [])
        weap = next(
            (x["building_count_gte"] for x in all_of
             if "building_count_gte" in x
             and (x["building_count_gte"] or {}).get("type") == "weap"),
            None,
        )
        assert weap is not None and int(weap.get("n", 0)) >= 1, (
            f"{lvl}: win must require building_count_gte weap>=1"
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
        powr = next(
            (x["building_count_gte"] for x in all_of
             if "building_count_gte" in x
             and (x["building_count_gte"] or {}).get("type") == "powr"),
            None,
        )
        assert powr is not None and int(powr.get("n", 0)) >= 1, (
            f"{lvl}: win must require building_count_gte powr>=1"
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


def test_two_base_layout_home_and_forward():
    """Pack must instantiate a HOME (safe, deep west) AND a FORWARD
    (doomed, east) base — each with its own `fact`. The HOME fact
    sits at x<25; the FORWARD fact sits at x>25 inside the strike
    region. Hard duplicates this pair across both spawn groups."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        agent_facts = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "fact"
        ]
        if lvl == "hard":
            per_spawn: dict[int, list] = {}
            for a in agent_facts:
                sp = a.spawn_point if a.spawn_point is not None else 0
                per_spawn.setdefault(sp, []).append(a)
            assert per_spawn, "hard: agent facts must declare spawn_point"
            for sp, fs in per_spawn.items():
                xs = [a.position[0] for a in fs]
                assert any(x < 25 for x in xs), (
                    f"hard spawn {sp}: missing HOME fact (x<25); xs={xs}"
                )
                assert any(x > 25 for x in xs), (
                    f"hard spawn {sp}: missing FORWARD fact (x>25); xs={xs}"
                )
        else:
            xs = [a.position[0] for a in agent_facts]
            assert any(x < 25 for x in xs), (
                f"{lvl}: missing HOME fact (x<25); xs={xs}"
            )
            assert any(x > 25 for x in xs), (
                f"{lvl}: missing FORWARD fact (x>25); xs={xs}"
            )


def test_scheduled_destroy_events_target_forward_zone_repeatedly():
    """Each tier must declare ≥2 `scheduled_events: destroy_actors`
    waves on the FORWARD zone (x≈40), each firing mid-episode before
    the deadline. Repeated waves are load-bearing: a single-wave
    pack would let the agent rebuild after the first strike and win
    at FORWARD, defeating the placement-decision intent."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        evs = c.scheduled_events or []
        destroys = [e for e in evs if e.get("type") == "destroy_actors"]
        assert len(destroys) >= 2, (
            f"{lvl}: needs ≥2 destroy_actors waves (placement-decision teeth)"
        )
        for e in destroys:
            assert e["filter"]["owner"] == "agent"
            reg = e["filter"]["region"]
            assert reg["x"] >= 35, (
                f"{lvl}: strike region must be in the FORWARD east zone "
                f"(x≥35); got {reg}"
            )
            assert e["tick"] < 4500, (
                f"{lvl}: strike must fire mid-episode; got tick {e['tick']}"
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
def test_build_at_home_wins(level, seed):
    """The intended resilient play (build the war factory and tank
    army at the safe HOME base) must WIN on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_at_home_policy(), seed=seed)
    assert res.outcome == "win", (
        f"build-at-home must WIN on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"buildings={sorted(res.signals.own_building_types)})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_build_at_both_wins(level, seed):
    """The redundancy play (build the war factory at BOTH bases) must
    WIN — the pack must credit preemptive spatial redundancy. The
    HOME copy survives the strike even if the FORWARD copy is razed."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_at_both_policy(), seed=seed)
    assert res.outcome == "win", (
        f"build-at-both (redundancy) must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_policy_loses(level, seed):
    """A stall (observe-only) builds nothing — no weap, no tanks →
    must LOSE on every (level, seed) on the reachable timeout."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_build_at_forward_policy_loses(level, seed):
    """Building the war factory and tanks at the FORWARD doomed base
    must LOSE on every (level, seed): the strike rotation razes the
    forward weap so `building_count_gte:{weap,1}` is false at the
    deadline. This is the spatial-placement teeth."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_at_forward_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"build-at-forward must LOSE on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick})"
    )
