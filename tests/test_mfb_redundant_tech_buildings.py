"""No-cheat + solvency proof for `mfb-redundant-tech-buildings`
(Wave-10 REASONING: production resilience — N+1 redundant capacity
planning / robust capacity planning / SC2 multi-production anchor).

The pack tests PROACTIVE redundancy planning: the agent inherits a
complete base whose only unit-producing structure — a single War
Factory (`weap`) — is a known single point of failure. The exposed
`weap` is pre-placed far east at low HP, ringed by six enemy mammoth
tanks (`4tnk`) at `stance:2` (Defend) that raze it inside the first
~3 decision turns (~tick 273), long before any vehicle could roll
out. The agent must pre-build a SECOND `weap` in the safe western
base BEFORE the exposed one falls, then produce ≥3 medium tanks
(`2tnk`) from the surviving redundant factory.

The win predicate makes the N+1 capability load-bearing:
  * `building_count_gte:{weap,1}` reads the LIVE `own_buildings` list
    per frame — FALSE once the exposed `weap` is razed unless a
    redundant `weap` exists (`has_building` would NOT work — its
    accumulating type-set never toggles back to false).
  * `unit_type_count_gte:{2tnk,3}` is unreachable from the exposed
    factory alone (razed at ~tick 273; first `2tnk` needs ~540 ticks
    of production).

For every level + every hard seed (1-4):
  * the INTENDED N+1 plan (pre-build a redundant `weap` in the safe
    western base, then produce ≥3 `2tnk`) WINS;
  * STALL (only `observe`) LOSES — builds nothing → reachable
    timeout LOSS;
  * BUILD-NO-BACKUP (spam `build('2tnk')` from the inherited exposed
    `weap`, never build a 2nd factory) LOSES — the single factory is
    razed before any tank completes → production halts → 2tnk count
    stays 0 → LOSS on the deadline. This is the canonical single-
    point-of-failure inversion the scenario is built to catch.

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

PACK = PACKS_DIR / "mfb-redundant-tech-buildings.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Idle: never builds anything → no redundant factory, no tanks →
    LOSS on the deadline."""
    return [Command.observe()]


def _build_no_backup(rs, Command):
    """Single-point-of-failure play: spam `build('2tnk')` from the
    inherited exposed `weap`, never build a redundant second factory.
    The exposed factory is razed at ~tick 273 before a single tank
    completes; production halts permanently → the `2tnk` count never
    reaches 3 and the `weap` count stays 0 → LOSS on the deadline.
    This is the canonical inversion the scenario is designed to
    catch."""
    prod = rs.get("production", []) or []
    if "2tnk" in prod:
        return [Command.observe()]
    return [Command.build("2tnk")]


def _intended(rs, Command):
    """N+1 redundancy plan: pre-build a SECOND `weap`, place it in the
    safe western base next to the surviving Construction Yard (out of
    the eastern strikers' range), then produce ≥3 `2tnk` from the
    surviving redundant factory. WINS every level × every seed inside
    the deadline."""
    bldgs = rs.get("own_buildings", []) or []
    fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
    if fact_b is None:
        return [Command.observe()]
    fx, fy = fact_b["cell_x"], fact_b["cell_y"]
    weaps = [b for b in bldgs if b["type"] == "weap"]
    prod = rs.get("production", []) or []
    # A "safe" (redundant) weap is one near the western Construction
    # Yard — the exposed inherited weap sits far east (x≈60).
    safe = [b for b in weaps if abs(b["cell_x"] - fx) < 25]
    if not safe:
        cmds = []
        if "weap" not in prod:
            cmds.append(Command.build("weap"))
        cmds.append(Command.place_building("weap", fx + 12, fy + 3))
        return cmds
    if "2tnk" not in prod:
        return [Command.build("2tnk")]
    return [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "mfb-redundant-tech-buildings"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue


def test_benchmark_anchor_lists_redundancy_and_capacity_planning():
    """meta.benchmark_anchor must name N+1 redundancy planning, robust
    capacity planning, and SC2 multi-production (Wave-10 spec
    anchors)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "redundancy" in blob and "n+1" in blob, anchors
    assert "robust capacity planning" in blob, anchors
    assert "multi-production" in blob, anchors


def test_tools_match_capacity_planning_palette():
    """The advertised toolset is the capacity-planning kit: observe +
    build + place_building + harvest + move_units + stop. No offensive
    verbs (`attack_unit` / `attack_move`) — this is a planning pack,
    not a combat pack."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []))
    for t in ("observe", "build", "place_building", "harvest",
              "move_units", "stop"):
        assert t in tools, f"missing tool {t!r} in {sorted(tools)}"
    for forbidden in ("attack_unit", "attack_move", "deploy"):
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
def test_win_requires_surviving_weap_three_tanks_and_fact(level):
    """Structural: the win clause must require a LIVE War Factory
    (`building_count_gte:{weap,1}` — the N+1-redundancy teeth), three
    medium tanks (`unit_type_count_gte:{2tnk,3}`), a live Construction
    Yard (`building_count_gte:{fact,1}`), and a `within_ticks`
    deadline. `building_count_gte` (live-list) — NOT `has_building`
    (accumulating set) — is mandatory so the clause toggles false when
    the exposed weap is razed."""
    c = compile_level(load_pack(PACK), level)
    all_of = c.win_condition.model_dump().get("all_of", [])
    weap = next(
        (x["building_count_gte"] for x in all_of
         if "building_count_gte" in x
         and (x["building_count_gte"] or {}).get("type") == "weap"),
        None,
    )
    assert weap is not None and int(weap.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte weap≥1 (live "
        f"factory survives the loss)"
    )
    tanks = next(
        (x["unit_type_count_gte"] for x in all_of
         if "unit_type_count_gte" in x
         and (x["unit_type_count_gte"] or {}).get("type") == "2tnk"),
        None,
    )
    assert tanks is not None and int(tanks.get("n", 0)) >= 3, (
        f"{level}: win must require unit_type_count_gte 2tnk≥3"
    )
    fact = next(
        (x["building_count_gte"] for x in all_of
         if "building_count_gte" in x
         and (x["building_count_gte"] or {}).get("type") == "fact"),
        None,
    )
    assert fact is not None and int(fact.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte fact≥1"
    )
    assert any("within_ticks" in x for x in all_of), (
        f"{level}: win must include a within_ticks deadline"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_exactly_one_exposed_weap_is_pre_placed(level):
    """The single-point-of-failure premise: each tier pre-places
    EXACTLY ONE agent `weap` (the exposed factory). The redundant
    second factory must be BUILT by the agent — it is not given."""
    c = compile_level(load_pack(PACK), level)
    weaps = [
        a for a in c.scenario.actors
        if a.owner == "agent" and a.type == "weap"
    ]
    # Hard duplicates the base across two spawn groups, so each spawn
    # group still ships exactly one exposed weap.
    if level == "hard":
        per_spawn = {}
        for a in weaps:
            sp = a.spawn_point if a.spawn_point is not None else 0
            per_spawn[sp] = per_spawn.get(sp, 0) + 1
        assert per_spawn and all(v == 1 for v in per_spawn.values()), (
            f"hard: each spawn group must pre-place exactly one exposed "
            f"weap; got {per_spawn}"
        )
    else:
        assert len(weaps) == 1, (
            f"{level}: must pre-place exactly one exposed agent weap; "
            f"got {len(weaps)}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_exposed_weap_is_ringed_by_defend_stance_strikers(level):
    """The exposed `weap` must be ringed by enemy `4tnk` at `stance:2`
    (Defend) — they auto-fire on the in-range building and raze it but
    NEVER advance, so the strike does its single job and the agent's
    `fact` always survives (the only LOSS path is the deadline)."""
    c = compile_level(load_pack(PACK), level)
    strikers = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.type == "4tnk" and a.stance == 2
    ]
    # Need enough mammoths to raze the war factory quickly (smoke-
    # verified: 6 per latitude → razed ~tick 273).
    assert len(strikers) >= 6, (
        f"{level}: must pre-place ≥6 enemy 4tnk at stance:2 to raze the "
        f"exposed weap before any tank can roll out; got {len(strikers)}"
    )
    for s in strikers:
        assert s.stance == 2, (
            f"{level}: strikers must be stance:2 (Defend) so they raze "
            f"the exposed weap then sit inert — not stance:3 (which would "
            f"march on the western base and turn this into a combat pack)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups so the engine round-robins start
    by seed. The full base (fact + powr + proc + fix + exposed weap +
    harv) is duplicated across both groups per CLAUDE.md `spawn_point`
    filter rules."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN ───────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_n_plus_one_wins(level, seed):
    """The intended capability — pre-build a redundant `weap` in the
    safe western base, then produce ≥3 `2tnk` from the surviving
    factory — WINS every level × every hard seed inside the
    deadline."""
    c, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended N+1 plan should WIN, got "
        f"{r.outcome}; types={r.signals.own_building_types}, "
        f"cash={r.signals.cash}, tick={r.signals.game_tick}"
    )
    types = set(r.signals.own_building_types)
    assert "weap" in types, types
    assert "fact" in types, types


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds a
    redundant factory or any tank → reachable timeout LOSS via
    after_ticks (never a draw)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_no_backup_loses(level, seed):
    """Build-no-backup (spam `build('2tnk')` from the inherited exposed
    `weap`, never build a redundant factory) must LOSE every level ×
    every seed — the single factory is razed at ~tick 273 before a
    tank completes; production halts; the 2tnk count never reaches 3
    and the weap count stays 0 → LOSS on the deadline. This is the
    canonical single-point-of-failure inversion the scenario is built
    to catch."""
    c, r = _run(level, _build_no_backup, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} build-no-backup must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared spawn_point
    groups (NORTH y=14 / SOUTH y=26) so a memorised opening cannot
    generalise. The base buildings are duplicated across both spawn
    groups per CLAUDE.md spawn_point filter rules, so the building
    cell coords flip per spawn."""
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
            bs = ad.render_state().get("own_buildings", []) or []
            if bs:
                starts.add(
                    tuple(sorted((b["cell_x"], b["cell_y"]) for b in bs))
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
