"""No-cheat + solvency proof for `power-budget-online` (Wave-10 Group I
REASONING: grid bring-up under a live power budget — keep the grid
POSITIVE while expanding production).

The pack tests an ONGOING capacity-planning discipline: from a standing
base whose single Power Plant only just covers its installed load
(provided 100, drained 80, surplus +20), the agent must expand into
proc + weap + fix + 3x 2tnk while bringing additional Power Plants
ONLINE AHEAD of each new production load. Building any production
building (`proc` already drains -30) on the single starter plant drives
the grid NEGATIVE; the engine then throttles ALL production queues to
50% speed (world.rs:3140-3146), and the build-out overruns the deadline.

For every level + every hard seed (1-4):
  * the INTENDED power-first interleave (build the extra Power Plant(s)
    first, then proc -> weap -> fix -> 3x 2tnk) WINS;
  * STALL (only `observe`) LOSES — never builds anything, deadline bites
    as a real timeout LOSS;
  * PRODUCTION-FIRST (build proc/weap/fix/tanks, defer the extra Power
    Plants to the end) LOSES — the grid sits negative through the whole
    post-proc phase, every queue runs at 50%, the build-out overruns
    `within_ticks`; even though it ends with provided power >= the floor
    it has run out the clock;
  * NO-POWER (build proc/weap/fix/tanks, NEVER build an extra Power
    Plant) LOSES — fails the `power_provided_gte` floor outright.

The 3 lazy plays + 1 intended x 3 levels x 4 seeds gives the full
no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "power-budget-online.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Total Power Plants (including the pre-placed starter) the intended
# policy must own per level — derived from the power_provided_gte floor
# (each powr is +100, the starter is +100).
_TARGET_POWR = {"easy": 2, "medium": 3, "hard": 3}


# ───────────────────────── scripted policies ──────────────────────────────


def _fact(rs):
    return next(
        (b for b in (rs.get("own_buildings", []) or []) if b["type"] == "fact"),
        None,
    )


def _types(rs):
    return [b["type"] for b in (rs.get("own_buildings", []) or [])]


def _ntanks(rs):
    return sum(
        1
        for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "2tnk"
    )


def _stall(rs, Command):
    """Idle: never builds anything → every win clause fails → reachable
    timeout LOSS (not a draw)."""
    return [Command.observe()]


def _intended(target):
    """Intended power-first interleave: build the extra Power Plant(s)
    FIRST so provided power leads the load, then proc → weap → fix →
    3x 2tnk. The grid never goes negative → full-speed production →
    WIN inside the window."""

    def policy(rs, Command):
        f = _fact(rs)
        if f is None:
            return [Command.observe()]
        fx, fy = f["cell_x"], f["cell_y"]
        types = _types(rs)
        prod = list(rs.get("production", []) or [])
        n_powr = types.count("powr")
        slots = [(fx + 5, fy), (fx + 5, fy + 4)]
        if n_powr < target:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", *slots[n_powr - 1]))
            return cmds
        if "proc" not in types:
            cmds = []
            if "proc" not in prod:
                cmds.append(Command.build("proc"))
            cmds.append(Command.place_building("proc", fx, fy + 9))
            return cmds
        if "weap" not in types:
            cmds = []
            if "weap" not in prod:
                cmds.append(Command.build("weap"))
            cmds.append(Command.place_building("weap", fx + 9, fy))
            return cmds
        if "fix" not in types:
            cmds = []
            if "fix" not in prod:
                cmds.append(Command.build("fix"))
            cmds.append(Command.place_building("fix", fx + 9, fy + 5))
            return cmds
        if _ntanks(rs) < 3 and "2tnk" not in prod:
            return [Command.build("2tnk")]
        return [Command.observe()]

    return policy


def _production_first(rs, Command):
    """Wrong-order: build proc → weap → fix → 3x 2tnk and only THEN
    build the extra Power Plants. The grid sits NEGATIVE from the moment
    proc is placed through the whole war-factory / depot / tank phase;
    every queue runs at 50% speed; the build-out overruns the deadline
    → real timeout LOSS even though it eventually reaches the power
    floor."""
    f = _fact(rs)
    if f is None:
        return [Command.observe()]
    fx, fy = f["cell_x"], f["cell_y"]
    types = _types(rs)
    prod = list(rs.get("production", []) or [])
    if "proc" not in types:
        cmds = []
        if "proc" not in prod:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", fx, fy + 9))
        return cmds
    if "weap" not in types:
        cmds = []
        if "weap" not in prod:
            cmds.append(Command.build("weap"))
        cmds.append(Command.place_building("weap", fx + 9, fy))
        return cmds
    if "fix" not in types:
        cmds = []
        if "fix" not in prod:
            cmds.append(Command.build("fix"))
        cmds.append(Command.place_building("fix", fx + 9, fy + 5))
        return cmds
    if _ntanks(rs) < 3 and "2tnk" not in prod:
        return [Command.build("2tnk")]
    n_powr = types.count("powr")
    if n_powr < 3:
        cmds = []
        slots = [(fx + 5, fy), (fx + 5, fy + 4)]
        if "powr" not in prod:
            cmds.append(Command.build("powr"))
        cmds.append(Command.place_building("powr", *slots[n_powr - 1]))
        return cmds
    return [Command.observe()]


def _no_power(rs, Command):
    """Wrong-order: build proc → weap → fix → 3x 2tnk and NEVER build a
    single extra Power Plant. Gross provided power stays at 100 → fails
    the `power_provided_gte` floor outright → real LOSS."""
    f = _fact(rs)
    if f is None:
        return [Command.observe()]
    fx, fy = f["cell_x"], f["cell_y"]
    types = _types(rs)
    prod = list(rs.get("production", []) or [])
    if "proc" not in types:
        cmds = []
        if "proc" not in prod:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", fx, fy + 9))
        return cmds
    if "weap" not in types:
        cmds = []
        if "weap" not in prod:
            cmds.append(Command.build("weap"))
        cmds.append(Command.place_building("weap", fx + 9, fy))
        return cmds
    if "fix" not in types:
        cmds = []
        if "fix" not in prod:
            cmds.append(Command.build("fix"))
        cmds.append(Command.place_building("fix", fx + 9, fy + 5))
        return cmds
    if _ntanks(rs) < 3 and "2tnk" not in prod:
        return [Command.build("2tnk")]
    return [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "power-budget-online"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Pure grid-budget test — only observe + build + place + stop.
    for t in ("observe", "build", "place_building", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"
    for forbidden in ("move_units", "attack_unit", "deploy", "harvest"):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in the build-only palette"
        )


def test_benchmark_anchor_lists_sc2_and_grid_bringup():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "sc2" in blob and "power" in blob, anchors
    assert "electrical" in blob and "grid" in blob, anchors
    assert "capacity planning" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns (tick ≈ 93 +
    90·(max_turns − 1)), and equal within_ticks + 1."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within_clauses = c.win_condition.model_dump().get("all_of", [])
    wt = next(
        int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x
    )
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt+1})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_uses_power_provided_floor_and_production_clauses(level):
    """Structural: the win clause is an all_of with the load-bearing
    `power_provided_gte` floor (200 easy / 300 medium+hard), the
    proc + weap building counts and the 3x 2tnk unit count."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    floor = next((x["power_provided_gte"] for x in all_of if "power_provided_gte" in x), None)
    assert floor is not None, f"{level}: win must include power_provided_gte"
    expected = 200 if level == "easy" else 300
    assert int(floor) == expected, f"{level}: power floor {floor} != {expected}"
    bc_types = {
        x["building_count_gte"]["type"]
        for x in all_of
        if "building_count_gte" in x
    }
    assert {"proc", "weap"} <= bc_types, f"{level}: missing proc/weap clauses"
    utc = next(
        (x["unit_type_count_gte"] for x in all_of if "unit_type_count_gte" in x),
        None,
    )
    assert utc is not None and utc["type"] == "2tnk" and int(utc["n"]) == 3, (
        f"{level}: win must require >=3 2tnk, got {utc}"
    )


def test_hard_has_two_spawn_groups_for_the_base():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups (NORTH y≈12 / SOUTH y≈28)."""
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
def test_intended_power_first_wins(level, seed):
    """The intended power-first interleave WINS every level × every seed
    before the deadline — the grid stays positive so production runs at
    full speed and the build-out finishes in budget."""
    c, r = _run(level, _intended(_TARGET_POWR[level]), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended power-first interleave should WIN, "
        f"got {r.outcome}; types={sorted(set(r.signals.own_building_types))}, "
        f"provided={r.signals.power_provided}, tick={r.signals.game_tick}"
    )
    expected = 200 if level == "easy" else 300
    assert r.signals.power_provided >= expected, r.signals.power_provided
    types = set(r.signals.own_building_types)
    assert {"proc", "weap", "fix"} <= types, types


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds anything
    → win clauses fail → reachable timeout LOSS (not a draw)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(types={sorted(set(r.signals.own_building_types))}, "
        f"tick={r.signals.game_tick})"
    )
    assert "proc" not in r.signals.own_building_types
    assert "weap" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_production_first_wrong_order_loses(level, seed):
    """The headline wrong-order discriminator: build proc/weap/fix/tanks
    first and defer the Power Plants. The grid runs negative through the
    whole post-proc phase, every queue throttles to 50%, and the
    build-out overruns `within_ticks` → real timeout LOSS — even though
    it ends with provided power at or above the floor."""
    c, r = _run(level, _production_first, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} production-first must LOSE; got {r.outcome} "
        f"(provided={r.signals.power_provided}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_no_power_floor_violation_loses(level, seed):
    """A policy that builds the production chain but NEVER an extra Power
    Plant must LOSE — gross provided power stays at 100, below the
    `power_provided_gte` floor → real LOSS."""
    c, r = _run(level, _no_power, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} no-power must LOSE; got {r.outcome} "
        f"(provided={r.signals.power_provided}, tick={r.signals.game_tick})"
    )
    assert r.signals.power_provided < (200 if level == "easy" else 300), (
        f"{level} seed{seed}: no-power policy should stay below the floor, "
        f"got provided={r.signals.power_provided}"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared spawn_point
    groups (NORTH y≈12 / SOUTH y≈28) so a memorised opening cannot
    generalise. The inert spawn-witness `e1` per group surfaces the
    variation via units_summary."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in SEEDS:
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
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, _intended(_TARGET_POWR["easy"]), seed=2)
    b = run_level(c, _intended(_TARGET_POWR["easy"]), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
