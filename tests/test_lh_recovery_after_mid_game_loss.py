"""No-cheat + solvency proof for `lh-recovery-after-mid-game-loss`
(Wave-11 / Wave-13 REASONING: two-base strategic-retreat template;
PlanBench replanning under exogenous loss, disaster recovery, SC2
comeback).

REDESIGN (task #81): the pack now follows the two-base strategic-
retreat template (def-retreat-and-rebuild lineage). The agent
starts with a FORWARD base (Construction Yard + Refinery + Power
Plant + Ore Truck + 2 defenders) at mid-map AND a HOME base
(parked Mobile Construction Vehicle + Power Plant + lone garrison)
at the deep-west safe zone. An in-world rusher band closes from
the east; scheduled `spawn_actors` reinforcement waves keep the
forward region hostile through the run. The win is STATE-BASED:
a `fact` AND a `proc` are alive INSIDE the deep-west HOME radius
at the deadline (with at least one own unit alive). The fail is
in-game: total fact extinction, or after_ticks.

The bar (every level × every hard seed):
  • STALL (observe only) — LOSS. Home MCV never deployed; the
    rush razes the forward base; total fact count drops to 0 →
    fact-loss fail clause fires.
  • BUILD-AT-FORWARD (try to rebuild proc at the forward latitude
    after the loss, never touch HOME MCV) — LOSS. The rebuilt
    forward proc is razed by the next wave; no home fact ever
    exists; home-radius win clauses are never satisfied →
    deadline LOSS (and/or fact-loss).
  • BUILD-AT-HOME (deploy HOME MCV + build a home-radius proc,
    accept the forward loss) — WIN.
  • BUILD-AT-BOTH (deploy HOME + home proc AND attempt a forward
    rebuild on the side) — WIN. The state-based predicate fully
    credits the home rebuild regardless of the forward attempt;
    preemptive redundancy is rewarded.

This is the two-base strategic-retreat template (def-retreat-and-
rebuild, def-evacuation, mid-concede-vs-hold lineage). Validation
is scripted (no model / no network) — uses
`openra_bench.eval_core.run_level`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs" / (
    "lh-recovery-after-mid-game-loss.yaml"
)
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _find_mcv(rs):
    for u in (rs.get("units_summary") or []):
        if str(u.get("type", "")).lower() == "mcv":
            return u
    return None


def _has_home_fact(rs, sy):
    """Is there an agent `fact` inside the radius-8 HOME disc?"""
    for b in (rs.get("own_buildings") or []):
        if str(b.get("type", "")).lower() != "fact":
            continue
        dx = int(b.get("cell_x", 0)) - 10
        dy = int(b.get("cell_y", 0)) - sy
        if dx * dx + dy * dy <= 64:
            return True
    return False


def _has_home_proc(rs, sy):
    """Is there an agent `proc` inside the radius-8 HOME disc?"""
    for b in (rs.get("own_buildings") or []):
        if str(b.get("type", "")).lower() != "proc":
            continue
        dx = int(b.get("cell_x", 0)) - 10
        dy = int(b.get("cell_y", 0)) - sy
        if dx * dx + dy * dy <= 64:
            return True
    return False


def _prod_str(rs):
    return " ".join(str(p).lower() for p in (rs.get("production") or []))


def _stall(rs, Command):
    """Give up after the loss: only observe. Home MCV never deployed,
    no home rebuild → fact-loss + deadline LOSS."""
    return [Command.observe()]


def _build_at_forward(rs, Command):
    """Try to rebuild the proc at the FORWARD latitude after the
    loss; never touch the HOME MCV. The rebuild is razed by the next
    reinforcement wave; no home fact ever exists → LOSS."""
    bldgs = rs.get("own_buildings") or []
    has_forward_proc = any(
        str(b.get("type", "")).lower() == "proc"
        and int(b.get("cell_x", 0)) >= 30
        for b in bldgs
    )
    fact = next(
        (b for b in bldgs if str(b.get("type", "")).lower() == "fact"
         and int(b.get("cell_x", 0)) >= 30),
        None,
    )
    ps = _prod_str(rs)
    cmds = []
    # If we still have a forward fact, queue + place a new proc at
    # the forward latitude (replacing one the rush may have razed).
    if not has_forward_proc and fact is not None:
        if "proc" not in ps:
            cmds.append(Command.build("proc"))
        else:
            # Place adjacent to the surviving forward fact.
            cmds.append(
                Command.place_building(
                    "proc",
                    target_x=int(fact["cell_x"]) + 4,
                    target_y=int(fact["cell_y"]),
                )
            )
    return cmds or [Command.observe()]


def _build_at_home_policy():
    """Intended recovery: deploy the HOME MCV (creates a HOME fact
    inside the safe disc), then build('proc') + place_building inside
    the HOME radius. WINs every level × every seed inside the
    deadline."""
    state = {"deployed": False, "safe_y": None}

    def policy(rs, Command):
        if not state["deployed"]:
            mcv = _find_mcv(rs)
            if mcv is not None:
                state["safe_y"] = int(mcv["cell_y"])
                state["deployed"] = True
                return [Command.deploy([str(mcv["id"])])]
        sy = state["safe_y"] or 20
        if _has_home_proc(rs, sy):
            return [Command.observe()]
        ps = _prod_str(rs)
        cmds = []
        if "proc" not in ps:
            cmds.append(Command.build("proc"))
        # Place at (12, sy) — adjacent to the post-deploy home fact
        # (~(9, sy-1)) AND inside the radius-8 home disc around
        # (10, sy).
        cmds.append(Command.place_building("proc", 12, sy))
        return cmds or [Command.observe()]

    return policy


def _build_at_both_policy():
    """Preemptive redundancy: deploy the HOME MCV + build the home
    proc AND ALSO queue a forward proc rebuild on the side. The
    home rebuild satisfies the win clause; the forward attempt is
    wasted cash but not fatal → still WIN."""
    state = {"deployed": False, "safe_y": None,
             "home_placed": False, "forward_placed": False}

    def policy(rs, Command):
        cmds = []
        if not state["deployed"]:
            mcv = _find_mcv(rs)
            if mcv is not None:
                state["safe_y"] = int(mcv["cell_y"])
                state["deployed"] = True
                return [Command.deploy([str(mcv["id"])])]
        sy = state["safe_y"] or 20
        bldgs = rs.get("own_buildings") or []
        ps = _prod_str(rs)
        # PRIORITY 1: stand up the home proc.
        if not _has_home_proc(rs, sy):
            if "proc" not in ps:
                cmds.append(Command.build("proc"))
            else:
                cmds.append(Command.place_building("proc", 12, sy))
            return cmds
        # PRIORITY 2 (after the home proc latches): also try a
        # forward proc rebuild — preemptive redundancy.
        if not state["forward_placed"]:
            fact = next(
                (b for b in bldgs if str(b.get("type", "")).lower() == "fact"
                 and int(b.get("cell_x", 0)) >= 30),
                None,
            )
            if fact is not None:
                if "proc" not in ps:
                    cmds.append(Command.build("proc"))
                else:
                    cmds.append(
                        Command.place_building(
                            "proc",
                            target_x=int(fact["cell_x"]) + 4,
                            target_y=int(fact["cell_y"]),
                        )
                    )
                    state["forward_placed"] = True
        return cmds or [Command.observe()]

    return policy


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "lh-recovery-after-mid-game-loss-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-recovery-after-mid-game-loss"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue


def test_benchmark_anchor_lists_planbench_disaster_and_comeback():
    """meta.benchmark_anchor must name PlanBench replanning, disaster
    recovery, and the SC2 comeback (Wave-11 spec anchors)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "planbench" in blob, anchors
    assert "disaster recovery" in blob, anchors
    assert "comeback" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_termination_keeps_run_alive_past_enemy_wipe(level):
    """The pack must preserve `termination.enemy_units_killed: false`
    (commit d9ecc30c) so a state-based win clause evaluates at the
    deadline regardless of in-world enemy elimination."""
    c = compile_level(load_pack(PACK), level)
    term = c.scenario.termination
    assert term is not None, f"{level}: termination block must exist"
    assert term.enemy_units_killed is False, (
        f"{level}: termination.enemy_units_killed must be false; "
        f"got {term.enemy_units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_threat_is_in_world_or_scheduled_spawn_actors(level):
    """The threat that destroys the forward base must be in-world
    (the `rusher` bot driving the on-map enemy band) AND/OR
    `scheduled_events: spawn_actors` reinforcement waves. The
    pre-redesign pack used `destroy_actors` to externally delete
    agent assets; that punished preemptive redundancy and is
    forbidden by the task #81 template. Assert no `destroy_actors`
    appears in `scheduled_events`."""
    c = compile_level(load_pack(PACK), level)
    sched = c.scheduled_events or []
    for e in sched:
        assert e.get("type") != "destroy_actors", (
            f"{level}: scheduled_events must not include destroy_actors "
            "(task #81 two-base template: threat must be in-world or "
            "spawn_actors so a preemptive home rebuild is fully credited)"
        )
    # And the enemy bot must be `rusher` (the in-world threat driver).
    bot = (c.scenario.enemy.bot_type or "").lower()
    assert bot == "rusher", (
        f"{level}: the in-world threat must be driven by the rusher bot; "
        f"got {bot!r}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_requires_home_fact_and_home_proc(level):
    """Structural: the win must require a `fact` AND a `proc` inside
    the deep-west HOME radius (state-based two-base template). On
    easy/medium that is a direct radius-8 disc around (10,20); on
    hard it is an `any_of` over (10,14) and (10,26)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])

    def _flatten(node):
        """Recursively yield every dict leaf from a nested all_of /
        any_of tree."""
        if isinstance(node, dict):
            if "all_of" in node:
                for sub in node["all_of"]:
                    yield from _flatten(sub)
            elif "any_of" in node:
                for sub in node["any_of"]:
                    yield from _flatten(sub)
            else:
                yield node
        elif isinstance(node, list):
            for sub in node:
                yield from _flatten(sub)

    leaves = list(_flatten(all_of))
    # Look for `building_in_region` clauses on fact + proc with
    # x≈10 (the deep-west HOME column).
    home_fact = [
        L for L in leaves
        if "building_in_region" in L
        and (L["building_in_region"] or {}).get("type") == "fact"
        and (L["building_in_region"] or {}).get("x") == 10
    ]
    home_proc = [
        L for L in leaves
        if "building_in_region" in L
        and (L["building_in_region"] or {}).get("type") == "proc"
        and (L["building_in_region"] or {}).get("x") == 10
    ]
    assert home_fact, f"{level}: win must require a HOME-radius fact clause"
    assert home_proc, f"{level}: win must require a HOME-radius proc clause"
    # The hard tier must offer BOTH north (y=14) AND south (y=26).
    if level == "hard":
        ys = {(L["building_in_region"] or {}).get("y") for L in home_fact}
        assert 14 in ys and 26 in ys, (
            f"hard: HOME fact clauses must cover both spawn latitudes; got {ys}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_fail_clause_includes_total_fact_loss_and_reachable_deadline(level):
    """Non-win must be a real LOSS, never a DRAW: fail must trigger
    on total fact extinction OR a reachable after_ticks. The fail
    after_ticks must fit inside max_turns (tick ≈ 93 + 90·(turns-1))
    and equal within_ticks + 1 so a non-finisher LOSES on the very
    next tick after the win window."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fail = c.fail_condition.model_dump()
    fail_after = next(
        (int(x["after_ticks"]) for x in fail.get("any_of", [])
         if "after_ticks" in x),
        None,
    )
    assert fail_after is not None, f"{level}: fail must include after_ticks"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert fail_after <= reachable, (
        f"{level}: fail after_ticks {fail_after} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within if "within_ticks" in x)
    assert fail_after == wt + 1, (
        f"{level}: fail after_ticks {fail_after} must equal within_ticks+1"
    )
    # Total fact extinction is also a fail (defends against the
    # silent DRAW path on a wiped-base run).
    fact_loss = [
        x for x in fail.get("any_of", [])
        if isinstance(x, dict)
        and "not" in x
        and isinstance(x["not"], dict)
        and (x["not"].get("building_count_gte") or {}).get("type") == "fact"
    ]
    assert fact_loss, f"{level}: fail must include total fact-loss clause"


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups so the engine round-robins the
    HOME + FORWARD latitudes by seed."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended / preemptive WIN ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_at_home_wins(level, seed):
    """The intended recovery — deploy the HOME MCV and stand up a
    home-radius refinery — WINS every level × every seed inside the
    deadline."""
    c, r = _run(level, _build_at_home_policy(), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: build-at-home should WIN, got {r.outcome}; "
        f"tick={getattr(r.signals, 'game_tick', '?')}, "
        f"buildings={getattr(r.signals, 'own_buildings', '?')}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_at_both_also_wins(level, seed):
    """Preemptive redundancy (build HOME and also attempt a forward
    rebuild) must ALSO WIN — the state-based predicate fully credits
    the home rebuild regardless of the forward attempt. The pre-
    redesign pack punished this play; the redesign must reward it."""
    c, r = _run(level, _build_at_both_policy(), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: build-at-both (preemptive redundancy) "
        f"should WIN, got {r.outcome}; "
        f"tick={getattr(r.signals, 'game_tick', '?')}, "
        f"buildings={getattr(r.signals, 'own_buildings', '?')}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall / give-up-after-loss must LOSE every level × every seed
    — home MCV never deployed, rush razes the forward base, total
    fact count drops to 0 → fact-loss + deadline LOSS."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE; got {r.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_at_forward_loses(level, seed):
    """Build-at-forward (rebuild the proc at the forward latitude
    after the loss, never touch the HOME MCV) must LOSE — the
    forward rebuild is razed by the next wave; the HOME radius
    never has a fact → win never latches → LOSS."""
    c, r = _run(level, _build_at_forward, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed}: build-at-forward must LOSE "
        f"(no home rebuild + forward razed by reinforcement waves); "
        f"got {r.outcome}"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _build_at_home_policy(), seed=2)
    b = run_level(c, _build_at_home_policy(), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
