"""No-cheat + solvency proof for `build-engineer-rebuild-after-loss`
(Wave-8 REASONING: power-grid continuity through an in-world enemy
strike on a forward base — PlanBench replanning / disaster recovery /
two-site business continuity anchor).

REDESIGNED (v2): the previous version used `scheduled_events:
destroy_actors` + a `then:[lost powr, rebuilt powr]` sequence latch.
That version's failure mode was predicate-gated (an externally
scripted destruction event the model never sees in-world). The
redesigned pack places loss modes IN-WORLD per the user's design
intent and audits/EDIT_PRINCIPLES.md §9.5 (no-solution-leak):

Two bases:
  - HOME base (west, safe): fact + harv + small garrison, NO powr.
  - FORWARD base (east, doomed): fact + powr + proc, with a
    pre-placed enemy 4tnk strike force adjacent.

The strike force destroys the forward buildings inside the first
few turns of in-world simulation; it also lingers to destroy any
rebuild placed at the forward base. It is too few / too slow to
reach the home base before the deadline.

State-based win: `building_count_gte:{powr,1}` checked AFTER a
gating tick (`after_ticks:800` — long enough for the strike to
have landed). At least one Power Plant alive ANYWHERE on the map
at the gating tick → WIN. Reactive (rebuild at home after loss)
and preemptive (build redundancy at home turn 1) both satisfy
this naturally.

For every level + every hard seed (1-4):
  * INTENDED-REACTIVE (rebuild at home after loss) WINS;
  * INTENDED-PREEMPTIVE (build redundancy at home immediately) WINS;
  * STALL (only `observe`) LOSES — forward powr dies in-world, no
    rebuild, live count 0 at the gating tick → loss via deadline
    or via the home fact eventually falling to stragglers;
  * FORWARD-ONLY rebuild LOSES — the lingering strike force
    destroys the rebuilt powr in place; live count 0 at the
    gating tick → loss.

The 2 lazy plays + 2 intended × 3 levels × 4 seeds gives the full
no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-engineer-rebuild-after-loss.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Idle: never builds anything → forward powr dies in-world, no
    replacement → live powr count is 0 at the gating tick → LOSS via
    the deadline (or earlier via the home fact eventually falling to
    enemy stragglers that walk west)."""
    return [Command.observe()]


def _forward_only(rs, Command):
    """Rebuild the powr AT the forward base — wrong choice. The
    lingering 4tnk strike force destroys the rebuilt powr the moment
    it lands (the forward base footprint is in their kill envelope).
    Live powr count never recovers → LOSS."""
    bldgs = rs.get("own_buildings", []) or []
    facts = [b for b in bldgs if b["type"] == "fact"]
    if not facts:
        return [Command.observe()]
    # FORWARD = east-most fact (highest cell_x).
    fwd = max(facts, key=lambda f: f["cell_x"])
    prod = rs.get("production", []) or []
    cmds = []
    if "powr" not in prod:
        cmds.append(Command.build("powr"))
    cmds.append(Command.place_building("powr", fwd["cell_x"] + 2, fwd["cell_y"]))
    return cmds


def _intended_reactive(rs, Command):
    """REACTIVE rebuild at the safe HOME base after observing the
    forward powr loss. WINS every level × every seed: the home base
    is geographically out of the strike force's reach inside the
    deadline, so a powr placed there survives to the gating tick."""
    bldgs = rs.get("own_buildings", []) or []
    facts = [b for b in bldgs if b["type"] == "fact"]
    if not facts:
        return [Command.observe()]
    # HOME = west-most fact (lowest cell_x).
    home = min(facts, key=lambda f: f["cell_x"])
    powrs = [b for b in bldgs if b["type"] == "powr"]
    # Wait until the forward powr is gone, then rebuild at home.
    if powrs:
        return [Command.observe()]
    prod = rs.get("production", []) or []
    cmds = []
    if "powr" not in prod:
        cmds.append(Command.build("powr"))
    # Place on the WEST side of the home fact, out of any enemy reach.
    cmds.append(Command.place_building("powr", home["cell_x"] - 2, home["cell_y"]))
    return cmds


class _PreemptState:
    """Per-episode one-shot latch — the preemptive policy issues the
    redundant build on the FIRST decision turn, before the strike
    has landed."""

    queued_build: bool = False


def _intended_preemptive(rs, Command):
    """PREEMPTIVE redundancy at the HOME base on turn 1 — the
    rational operator who sees the doomed forward position and
    insures against it immediately. The new home powr lands well
    before the strike completes; even after the forward powr
    falls, the home powr keeps live count ≥ 1 → WIN."""
    bldgs = rs.get("own_buildings", []) or []
    facts = [b for b in bldgs if b["type"] == "fact"]
    if not facts:
        return [Command.observe()]
    home = min(facts, key=lambda f: f["cell_x"])
    prod = rs.get("production", []) or []
    cmds = []
    if not _PreemptState.queued_build and "powr" not in prod:
        cmds.append(Command.build("powr"))
        _PreemptState.queued_build = True
    # Always (re)issue the place order until the rebuild has landed —
    # idempotent against an in-flight production queue.
    cmds.append(Command.place_building("powr", home["cell_x"] - 2, home["cell_y"]))
    return cmds or [Command.observe()]


def _reset_preempt():
    _PreemptState.queued_build = False


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-engineer-rebuild-after-loss"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue


def test_benchmark_anchor_lists_planbench_and_disaster_recovery():
    """meta.benchmark_anchor must name PlanBench replanning, disaster
    recovery, and exogenous loss (Wave-8 spec anchors)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "planbench" in blob, anchors
    assert "disaster recovery" in blob, anchors
    assert "exogenous loss" in blob, anchors


def test_tools_match_build_engineer_palette():
    """The advertised toolset is the build-engineer kit: observe +
    build + place_building + harvest + move_units + attack_unit +
    stop. No `attack_move` or `deploy` (this is not an offensive
    pack and there is no MCV)."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []))
    for t in ("observe", "build", "place_building", "harvest",
              "move_units", "attack_unit", "stop"):
        assert t in tools, f"missing tool {t!r} in {sorted(tools)}"
    for forbidden in ("deploy",):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in this palette"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns (tick ≈ 93 +
    90·(max_turns − 1)); within_ticks + 1 == after_ticks so a
    non-finisher LOSES on the very next tick after the win window."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fail = c.fail_condition.model_dump()
    after_ticks = next(
        int(x["after_ticks"]) for x in fail["any_of"] if "after_ticks" in x
    )
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within if "within_ticks" in x)
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt+1})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_is_state_based_powr_alive_after_strike_window(level):
    """Structural: the redesigned win predicate is state-based — at
    the gating tick (`after_ticks:N` for some N well past the
    in-world strike completion) the agent must own at least one
    Power Plant ANYWHERE on the map AND keep a Construction Yard
    alive AND finish before the deadline. No `then:` latch — the
    strike is in-world, not predicate-gated; the agent's task is
    geographic (place the rebuild where the strike force cannot
    reach) not temporal (sequence loss-then-rebuild)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    # Power continuity clause — anywhere on the map.
    powr_clause = next(
        (x["building_count_gte"] for x in all_of
         if "building_count_gte" in x
         and (x["building_count_gte"] or {}).get("type") == "powr"),
        None,
    )
    assert powr_clause is not None and int(powr_clause.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte powr≥1"
    )
    # Fact survival clause.
    fact_clause = next(
        (x["building_count_gte"] for x in all_of
         if "building_count_gte" in x
         and (x["building_count_gte"] or {}).get("type") == "fact"),
        None,
    )
    assert fact_clause is not None and int(fact_clause.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte fact≥1"
    )
    # End-state gate: an `after_ticks` clause that delays the win
    # check until the in-world strike has had time to land.
    after_gate = next(
        (int(x["after_ticks"]) for x in all_of if "after_ticks" in x), None
    )
    assert after_gate is not None and after_gate >= 200, (
        f"{level}: win must include an `after_ticks` end-state gate "
        f"(≥200 ticks) so the powr check fires AFTER the in-world "
        f"forward strike has had time to land; got {after_gate}"
    )
    # And no legacy `then:` latch (the v1 footgun).
    def has_then(node) -> bool:
        if isinstance(node, dict):
            if "then" in node:
                return True
            return any(has_then(v) for v in node.values())
        if isinstance(node, list):
            return any(has_then(x) for x in node)
        return False
    assert not has_then(win), (
        f"{level}: win must not include a `then:` happened-before "
        f"latch — the redesigned predicate is state-based, not "
        f"sequence-based; got {win}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_pack_has_two_bases_geographically_separated(level):
    """The redesign premise: each tier must pre-place agent facts at
    TWO well-separated x-coords — a HOME base (west, safe) and a
    FORWARD base (east, doomed). The strike force destroys the
    forward base by geometry; the home base is the safe rebuild
    site. A pack with both facts at the same x has collapsed the
    two-base template and is defective."""
    c = compile_level(load_pack(PACK), level)
    agent_facts = [
        a for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    ]
    # On hard the duplicate spawn_point groups can double the count;
    # collapse by spawn_point first.
    by_spawn: dict[int | None, list] = {}
    for a in agent_facts:
        by_spawn.setdefault(a.spawn_point, []).append(a)
    # Each spawn group should have at least 2 facts (home + forward).
    for sp, group in by_spawn.items():
        assert len(group) >= 2, (
            f"{level}: spawn group {sp} has only {len(group)} agent "
            f"fact(s); the two-base template requires home + forward"
        )
        xs = sorted(a.position[0] for a in group)
        assert xs[-1] - xs[0] >= 20, (
            f"{level}: spawn group {sp} facts at xs {xs} are too "
            f"close — home and forward must be ≥20 cells apart so "
            f"the forward strike geometry doesn't reach home"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_pre_placed_strike_force_present(level):
    """Each tier must pre-place ≥2 enemy `4tnk` at stance:3
    (AttackAnything) so the forward strike fires on contact at tick
    0 and lands the in-world forward-base demolition before the
    agent's first few decision turns."""
    c = compile_level(load_pack(PACK), level)
    strikers = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.type == "4tnk" and a.stance == 3
    ]
    assert len(strikers) >= 2, (
        f"{level}: must pre-place ≥2 enemy `4tnk` at stance:3 to "
        f"land the in-world forward-base demolition; got {len(strikers)}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_forward_base_carries_a_powr(level):
    """The forward (east) base must pre-place a powr so the in-world
    loss event is visible to the agent (the powr count drops from
    ≥1 to 0 as the strike force chews through the forward base)."""
    c = compile_level(load_pack(PACK), level)
    agent_powrs = [
        a for a in c.scenario.actors
        if a.owner == "agent" and a.type == "powr"
    ]
    assert agent_powrs, (
        f"{level}: must pre-place ≥1 agent `powr` (the in-world "
        f"loss event needs a powr to lose); got 0"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_no_scheduled_destroy_actors(level):
    """The redesign explicitly drops `scheduled_events: destroy_actors`
    — failure modes are in-game (enemy strike), not predicate-gated.
    A `scheduled_events: spawn_actors` reinforcement wave would be
    OK in principle (the model sees the new units arrive); but a
    `destroy_actors` block silently teleports asset deletion and
    breaks the design contract."""
    c = compile_level(load_pack(PACK), level)
    events = getattr(c.scenario, "scheduled_events", None) or []
    for ev in events:
        kind = getattr(ev, "kind", None) or (
            ev.get("kind") if isinstance(ev, dict) else None
        )
        assert kind != "destroy_actors", (
            f"{level}: scheduled_events must not include "
            f"`destroy_actors` (the redesign uses in-game enemy "
            f"strike for the loss event); got {ev}"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups so the engine round-robins
    start by seed. NORTH y≈14 vs SOUTH y≈26; the full base (home
    fact + harv + garrison + forward fact + powr + proc) is
    duplicated across both groups per CLAUDE.md `spawn_point`
    filter rules."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WINs ──────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_reactive_rebuild_wins(level, seed):
    """REACTIVE path: wait for the in-world forward powr loss,
    `build('powr')` + `place_building` at the safe HOME base on the
    west side. WINS every level × every hard seed: the home base is
    geographically out of the strike force's reach inside the
    deadline, so the rebuilt powr survives to the gating tick."""
    c, r = _run(level, _intended_reactive, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended reactive rebuild should WIN, got "
        f"{r.outcome}; types={r.signals.own_building_types}, "
        f"cash={r.signals.cash}, tick={r.signals.game_tick}"
    )
    types = set(r.signals.own_building_types)
    assert "powr" in types, types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_preemptive_redundancy_wins(level, seed):
    """PREEMPTIVE path: `build('powr') + place_building` at the HOME
    base on the FIRST decision turn — the rational operator who sees
    the doomed forward position and insures against it immediately.
    WINS every level × every hard seed."""
    _reset_preempt()
    c, r = _run(level, _intended_preemptive, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended preemptive redundancy should WIN, "
        f"got {r.outcome}; types={r.signals.own_building_types}, "
        f"cash={r.signals.cash}, tick={r.signals.game_tick}"
    )
    types = set(r.signals.own_building_types)
    assert "powr" in types, types


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds
    anything, so when the in-world strike destroys the forward powr
    the live count drops to 0 and never recovers → LOSS at the
    deadline (or earlier when stragglers reach the home fact)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_forward_only_rebuild_loses(level, seed):
    """FORWARD-ONLY rebuild must LOSE every level × every seed — the
    lingering 4tnk strike force destroys the rebuilt powr the moment
    it lands at the forward base (the forward footprint is in their
    kill envelope). Live powr count never recovers → LOSS. This is
    the canonical 'rebuilt where the threat still is' failure mode
    the two-base redesign is meant to teach."""
    c, r = _run(level, _forward_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} forward-only rebuild must LOSE; got "
        f"{r.outcome} (types={r.signals.own_building_types}, "
        f"tick={r.signals.game_tick})"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared
    spawn_point groups (NORTH y=14 / SOUTH y=26) so a memorised
    opening cannot generalise. The base buildings are duplicated
    across both spawn groups per CLAUDE.md spawn_point filter rules,
    so the units_summary cell coords flip per spawn."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended_reactive, seed=2)
    b = run_level(c, _intended_reactive, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
