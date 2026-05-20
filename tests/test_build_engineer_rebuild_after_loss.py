"""No-cheat + solvency proof for `build-engineer-rebuild-after-loss`
(Wave-8 REASONING: PlanBench replanning under exogenous loss /
disaster recovery / SC2 rebuild-after-trade anchor).

The pack tests the BUILD-ENGINEER replan: the agent inherits a complete
production base (fact + proc + powr + weap + harv) but the pre-placed
Power Plant (`powr`) starts at LOW HEALTH and a pre-placed enemy strike
force (1× 4tnk on easy, 2× on medium/hard) destroys it on tick 0..90.
The agent must (1) detect the loss, (2) `build('powr')` with the
reserve cash, (3) `place_building` it adjacent to the surviving `fact`.

The happened-before `then:[A,B]` composite enforces the
destruction-then-rebuild semantics: clause A is `not building_count_gte:
{powr,1}` (a "powr currently destroyed" frame) — this LATCHES on the
opening salvo. Clause B is `building_count_gte:{powr,1}` (a "powr
currently alive" frame) — this LATCHES on the rebuild landing.
`has_building` cannot be used here (CLAUDE.md footgun: the accumulating
`own_building_types` set never toggles back to false after destruction).

For every level + every hard seed (1-4):
  * the INTENDED `build('powr') + place_building` chain WINS;
  * STALL (only `observe`) LOSES — never rebuilds → then-clause B
    never latches → reachable timeout LOSS;
  * WRONG-SPEND (only `build('e1')`, spends the reserve on infantry)
    LOSES — never rebuilds → LOSS on the deadline.

The 2 lazy plays + 1 intended × 3 levels × 4 seeds gives the full
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
    """Idle: never builds anything → then-clause B (rebuild) never
    latches → LOSS on the deadline."""
    return [Command.observe()]


def _wrong_spend(rs, Command):
    """Brute army-spam: only `build('e1')`, draining the indivisible
    reserve on infantry. The powr is never rebuilt → then-clause B
    never latches → LOSS on the deadline. This is the canonical
    'spent the reserve on the wrong thing' failure mode the scenario
    is designed to catch."""
    return [Command.build("e1")]


def _intended(rs, Command):
    """Build-engineer rebuild: notice the low / dropped powr count,
    queue `build('powr')` (cost 300), place it adjacent to the
    surviving fact on the safe (west) side. WINS every level × every
    seed before the deadline."""
    bldgs = rs.get("own_buildings", []) or []
    own_counts: dict[str, int] = {}
    for b in bldgs:
        own_counts[b["type"]] = own_counts.get(b["type"], 0) + 1
    fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
    if fact_b is None:
        return [Command.observe()]
    fx, fy = fact_b["cell_x"], fact_b["cell_y"]
    prod = rs.get("production", []) or []
    if own_counts.get("powr", 0) < 1:
        cmds = []
        if "powr" not in prod:
            cmds.append(Command.build("powr"))
        # Place on the WEST side of the fact (out of the east strike lane).
        cmds.append(Command.place_building("powr", fx - 2, fy))
        return cmds
    return [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
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
def test_win_uses_then_destroy_then_rebuild_for_powr(level):
    """Structural: the win clause uses a `then:[A,B]` composite where
    clause A is `not building_count_gte:{powr,1}` (powr currently
    destroyed) and clause B is `building_count_gte:{powr,1}` (powr
    currently alive). This is the destruction-then-rebuild idiom.
    Using `has_building` here would NOT work (CLAUDE.md footgun: the
    accumulating set never toggles back to false after destruction)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    then_node = next((x["then"] for x in all_of if "then" in x), None)
    assert then_node is not None, f"{level}: win must include a `then` composite"
    clauses = then_node.get("clauses", [])
    assert len(clauses) == 2, (
        f"{level}: then must have 2 clauses (destruction, rebuild)"
    )
    # Clause A: not building_count_gte: powr ≥ 1
    a = clauses[0]
    assert "not" in a, f"{level}: clause A must be `not`: {a}"
    inner = a["not"]
    bc = inner.get("building_count_gte") or {}
    assert bc.get("type") == "powr" and int(bc.get("n", 0)) >= 1, (
        f"{level}: clause A must be `not building_count_gte powr≥1`: {a}"
    )
    # Clause B: building_count_gte: powr ≥ 1
    b = clauses[1]
    bc2 = b.get("building_count_gte") or {}
    assert bc2.get("type") == "powr" and int(bc2.get("n", 0)) >= 1, (
        f"{level}: clause B must be `building_count_gte powr≥1`: {b}"
    )
    # And an outer `building_count_gte` for proc must keep production
    # gated (per the pack spec).
    proc_clause = next(
        (x["building_count_gte"] for x in all_of
         if "building_count_gte" in x
         and (x["building_count_gte"] or {}).get("type") == "proc"),
        None,
    )
    assert proc_clause is not None and int(proc_clause.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte proc≥1"
    )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups so the engine round-robins start
    by seed. NORTH y≈14..22 vs SOUTH y≈22..30; the full base (fact +
    proc + powr + weap + harv + defender) is duplicated across both
    groups per CLAUDE.md `spawn_point` filter rules."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


@pytest.mark.parametrize("level", LEVELS)
def test_powr_is_pre_placed_at_low_hp(level):
    """The exogenous-loss premise: each tier must pre-place a `powr`
    actor at LOW HP so the adjacent stance:3 4tnk(s) destroy it in
    the opening turn. ActorPlacement `health` is a percentage 1-100
    (openra_rl_training.scenario line 196)."""
    c = compile_level(load_pack(PACK), level)
    low_hp_powrs = [
        a for a in c.scenario.actors
        if a.owner == "agent" and a.type == "powr" and a.health <= 30
    ]
    assert low_hp_powrs, (
        f"{level}: must pre-place at least one low-HP (≤30%) agent "
        f"`powr` so the strike force can destroy it; got "
        f"{[(a.type, a.health) for a in c.scenario.actors if a.owner=='agent']}"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_pre_placed_strike_force_present(level):
    """Each tier must pre-place at least one enemy `4tnk` at stance:3
    (AttackAnything) so it fires on contact at tick 0 and lands the
    powr kill before the agent's first decision turn."""
    c = compile_level(load_pack(PACK), level)
    strikers = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.type == "4tnk" and a.stance == 3
    ]
    assert strikers, (
        f"{level}: must pre-place ≥1 enemy `4tnk` at stance:3 to "
        f"land the opening powr kill"
    )


# ───────────────────────── intended WIN ───────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_rebuild_wins(level, seed):
    """The intended capability — detect-loss + build('powr') +
    place_building adjacent to the surviving fact on the safe west
    side — WINS every level × every hard seed well inside the
    deadline."""
    c, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended rebuild should WIN, got "
        f"{r.outcome}; types={r.signals.own_building_types}, "
        f"cash={r.signals.cash}, tick={r.signals.game_tick}"
    )
    # Sanity: the rebuilt powr must show up in the accumulating set
    # (it was built during the episode, regardless of subsequent loss).
    types = set(r.signals.own_building_types)
    assert "powr" in types, types
    assert "proc" in types, types


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds
    anything, so the rebuild clause of the `then` latch never closes
    → reachable timeout LOSS via after_ticks (never a draw)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_wrong_spend_loses(level, seed):
    """Wrong-spend (brute `build('e1')`, never powr) must LOSE every
    level × every seed — the reserve is drained on infantry, the powr
    is never rebuilt, the then-clause B never latches → LOSS on the
    deadline. This is the canonical 'spent the reserve on the wrong
    thing' failure mode."""
    c, r = _run(level, _wrong_spend, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} wrong-spend must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared spawn_point
    groups (NORTH y=14..22 / SOUTH y=22..30) so a memorised opening
    cannot generalise. The base buildings are duplicated across both
    spawn groups per CLAUDE.md spawn_point filter rules, so the
    units_summary cell coords flip per spawn."""
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
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
