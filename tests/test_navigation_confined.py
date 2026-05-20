"""navigation-confined-hard-only — confined-aisle egress under the clock.

The hard tier places the agent on a `confined-aisle-64x40` arena
partitioned by three vertical silo walls (x=18, 30, 42) with a single
6-cell gap each, alternating TOP/BOTTOM/TOP. The agent's squad of 4
infantry must reach the egress zone at the far SE corner (centred on
(55,30), radius 6) before the deadline (within_ticks 2700), with
zero losses. Two `spawn_point` groups (NORTH y=8..14, SOUTH y=25..31)
vary the staging row by seed.

Bar (binding):
    Stall / shortest-line-into-a-wall / reckless-squeeze must LOSE on
    every level + every hard seed (1..4); the intended detour-
    navigation policy must WIN.

Easy/medium share the worked custom-map-no-enemy idiom on an open
map (singles-maginot) — only hard adds the confined-aisle obstacle
axis, so attribution stays clean.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "navigation-confined-hard-only.yaml"

# ── policies (scripted; the bar is enforced by scripted_explore_agent-
# style closures, no model required). ────────────────────────────────


def _stall(rs, C):
    """Do nothing — only observe. Must LOSE on every level via
    the reachable deadline (no draw degeneracy)."""
    return [C.observe()]


def _intended_move_to_goal(goal_x, goal_y):
    """Commit a direct route to the goal every turn. The engine's
    pathfinder threads the static-wall serpentine optimally; on hard
    this is the lowest-cost route that exists, so it WINS inside the
    tick budget."""
    def policy(rs, C):
        us = rs.get("units_summary", []) or []
        ids = [str(u["id"]) for u in us]
        if not ids:
            return [C.observe()]
        return [C.move_units(ids, target_x=goal_x, target_y=goal_y)]
    return policy


def _shortest_line_into_wall(rs, C):
    """The "shortest manhattan-line" failure mode: target a cell INSIDE
    wall B (at column x=30) — the silo blocks the cell, the pathfinder
    parks the squad against the wall instead of routing around it via
    the (30, 4..9) gap, and no unit ever reaches the egress zone. The
    deadline bites at tick 2701. Must LOSE."""
    us = rs.get("units_summary", []) or []
    ids = [str(u["id"]) for u in us]
    if not ids:
        return [C.observe()]
    # Target a cell that is squarely inside wall B (one of the silo
    # cells at x=30, y=20). The pathfinder cannot occupy that cell.
    return [C.move_units(ids, target_x=30, target_y=20)]


def _reckless_squeeze(rs, C):
    """The "bash through the wall" failure mode: every turn, target
    one of the wall-B silo cells directly — same as shortest-line-
    into-wall but expressed as a continually-re-issued bash command.
    Units park against the wall and the deadline overruns. Must LOSE.
    (Hard intentionally omits `attack_unit` from the tool whitelist,
    so we can't even ask units to attack the silos — the only way
    to "reckless squeeze" via the move API is to target wall cells.)"""
    us = rs.get("units_summary", []) or []
    ids = [str(u["id"]) for u in us]
    if not ids:
        return [C.observe()]
    # Cycle the target through wall-B cells so the units keep
    # pushing against the wall instead of detouring.
    tick = rs.get("game_tick", 0) or 0
    y = 10 + (tick // 90) % 20  # walks across wall B's blocked span
    return [C.move_units(ids, target_x=30, target_y=y)]


# ── tests ────────────────────────────────────────────────────────────


def test_pack_loads_and_compiles_all_levels():
    p = load_pack(PACK)
    assert p.meta.id == "navigation-confined-hard-only"
    assert p.meta.capability == "perception"
    # Required anchor list (suite-enforced elsewhere too).
    assert "ERQA spatial commit" in p.meta.benchmark_anchor
    assert "AI2-THOR navigation" in p.meta.benchmark_anchor
    assert "Habitat 3.0 confined-space" in p.meta.benchmark_anchor
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        # Every level has a fail condition (no loss==draw degeneracy).
        assert c.fail_condition is not None, f"{lv}: missing fail_condition"
        # Deadline is reachable inside max_turns (anti-stall teeth).
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        wt = [cl["within_ticks"] for cl in wc["all_of"]
              if "within_ticks" in cl][0]
        tick_max = 93 + 90 * (c.max_turns - 1)
        assert wt < tick_max, (
            f"{lv}: within_ticks {wt} not reachable in max_turns "
            f"{c.max_turns} (tick_max={tick_max})"
        )
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        at = [cl["after_ticks"] for cl in fc["any_of"]
              if "after_ticks" in cl][0]
        assert at <= tick_max, (
            f"{lv}: after_ticks {at} unreachable in max_turns — "
            f"staller would draw, not lose (tick_max={tick_max})"
        )


def test_hard_uses_arena_generator_and_static_silo_walls():
    """Hard switches to the confined-aisle arena (mapgen) and places
    silo-wall obstacles statically (no spawn_point on enemy actors —
    the engine's spawn_point filter only applies to AGENT actors)."""
    c = compile_level(load_pack(PACK), "hard")
    assert c.map_supported, "confined-aisle map must materialise"
    assert "confined-aisle" in c.scenario.base_map, (
        f"hard must use the generated arena, got {c.scenario.base_map}"
    )
    # No spawn_point on enemy/neutral silos (engine ignores it anyway,
    # but we want it absent from the YAML for clarity).
    silos = [a for a in c.scenario.actors if a.type == "silo"]
    assert len(silos) >= 60, f"hard needs many silos, got {len(silos)}"
    assert all(s.spawn_point is None for s in silos), (
        "silo walls must be static (no spawn_point) — the engine ignores "
        "spawn_point on non-agent actors so a spawn_point silo would be "
        "always-on anyway and silently double-walls the map"
    )
    # Exactly two agent spawn_point groups, each with 4 e1 units.
    from collections import Counter
    spawns = Counter()
    for a in c.scenario.actors:
        if a.owner == "agent":
            spawns[a.spawn_point if a.spawn_point is not None else 0] += 1
    assert set(spawns) == {0, 1}, f"hard needs ≥2 spawn groups, got {dict(spawns)}"
    assert spawns[0] == 4 and spawns[1] == 4, (
        f"each spawn group needs 4 units, got {dict(spawns)}"
    )


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(lv):
    """Bar: stall MUST LOSE on every level (reachable deadline)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _stall, seed=1)
    assert res.outcome == "loss", (
        f"{lv}: stall must LOSE on the reachable deadline, got "
        f"{res.outcome} at tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_on_every_hard_seed(seed):
    """Bar: stall MUST LOSE on every hard seed (1..4)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"hard seed={seed}: stall must LOSE, got "
        f"{res.outcome} at tick={res.signals.game_tick}"
    )


def test_intended_detour_wins_easy():
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "easy")
    res = run_level(c, _intended_move_to_goal(55, 16), seed=1)
    assert res.outcome == "win", (
        f"easy: committed navigation should WIN, got {res.outcome}"
    )


def test_intended_detour_wins_medium():
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "medium")
    res = run_level(c, _intended_move_to_goal(72, 16), seed=1)
    assert res.outcome == "win", (
        f"medium: committed rendezvous should WIN, got {res.outcome}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_detour_wins_on_every_hard_seed(seed):
    """Bar: the intended detour-navigation policy MUST WIN on every
    hard seed (1..4). The engine's pathfinder threads the serpentine
    optimally from a direct goal command; both stages clear the
    2700-tick budget (~1983 N / ~2343 S)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _intended_move_to_goal(55, 30), seed=seed)
    assert res.outcome == "win", (
        f"hard seed={seed}: intended detour must WIN, got "
        f"{res.outcome} at tick={res.signals.game_tick} "
        f"(turns={res.turns}, lost={res.signals.units_lost})"
    )
    assert res.signals.units_lost == 0, (
        f"hard seed={seed}: intended detour must not lose units, "
        f"got units_lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_shortest_line_into_wall_loses_on_every_hard_seed(seed):
    """Bar: a "shortest manhattan-line" policy that targets a cell
    INSIDE the wall (not the egress zone) parks the squad against
    the silo and never reaches the goal — must LOSE on every hard
    seed via the reachable deadline."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _shortest_line_into_wall, seed=seed)
    assert res.outcome == "loss", (
        f"hard seed={seed}: shortest-line-into-wall must LOSE, got "
        f"{res.outcome} at tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_reckless_squeeze_loses_on_every_hard_seed(seed):
    """Bar: a "reckless squeeze" policy that keeps re-issuing moves
    into wall cells (push-through-the-racking play) wastes the budget
    and the deadline expires — must LOSE on every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _reckless_squeeze, seed=seed)
    assert res.outcome == "loss", (
        f"hard seed={seed}: reckless squeeze must LOSE, got "
        f"{res.outcome} at tick={res.signals.game_tick}"
    )


def test_hard_two_distinct_spawn_groups_in_practice():
    """Hard tier contract (see tests/test_hard_tier.py): the two
    spawn_point groups must produce visibly different agent start
    positions across seeds 1..4."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    starts = set()
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
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
        f"hard seeds 1..4 produced identical starts {starts}; "
        f"spawn_point round-robin not taking effect"
    )
