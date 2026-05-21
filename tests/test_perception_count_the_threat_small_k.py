"""perception-count-the-threat-small-k: exact-K enemy discovery under fog.

Structural tests assert the predicate shape — each level discriminates
by the exact K (enemies_discovered_gte) and has a real fail_condition.

The engine-driven bar (per CLAUDE.md): the intended split-scout policy
WINS on every level and every hard seed (1-4); STALL (only observe →
discovers 0), a single-axis WRONG-PATH sweep (finds < K clusters), and
a BRUTE ram-everything-at-mid-map (finds < K, bleeds attrition) all
LOSE on every level and every seed. Non-win is a real reachable
timeout LOSS.

Recalibrated after the engine balance fixes (armor-class weapon
selection, stance semantics, parallel production, pbox-fires):
  * The scout column is pinned to stance:0 in the pack YAML. The
    default stance:3 (AttackAnything) now actively HUNTS — left at the
    default the column self-delivered across the map and a do-nothing
    stall WON the perception task for free.
  * Each hidden enemy squad is `count: 1` (one actor per cluster) so
    `enemies_discovered_gte: K` counts CLUSTERS, not bodies — a
    `count: 2` squad made a K=3 win satisfiable by finding only 2 of 3
    clusters.
  * The `dog` pockets are stance:0 — a stance:2 dog now chases a
    passing scout and runs it down off-route, making the careful
    split-scout unsolvable inside the loss cap.

Module-level importorskip because openra_bench.scenarios imports the
Rust adapter at import time, so even pure-Python structural tests need
the wheel to load the package.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "perception-count-the-threat-small-k.yaml"
)

EXPECTED_K = {"easy": 2, "medium": 3, "hard": 4}

# Hidden enemy cluster centres per level — the intended split-scout
# routes one scout toward each, the wrong-path sweeps a single lane.
CLUSTERS = {
    "easy": [(55, 15), (105, 18)],
    "medium": [(105, 8), (105, 32), (55, 4)],
    "hard": [(105, 8), (105, 32), (115, 4), (60, 28)],
}


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


# ── structural / predicate-shape checks (no engine) ──────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_predicates_isolate_exact_count(level):
    pack = load_pack(PACK)
    assert pack.meta.capability == "perception"
    c = compile_level(pack, level)

    win = _win_clauses(c)
    # exact K target for this level
    ks = [cl["enemies_discovered_gte"] for cl in win if "enemies_discovered_gte" in cl]
    assert ks == [EXPECTED_K[level]], (
        f"{level}: expected enemies_discovered_gte={EXPECTED_K[level]}, got {ks}"
    )
    # tight clock binds (< max_turns * 90)
    wt = [cl["within_ticks"] for cl in win if "within_ticks" in cl][0]
    assert wt < c.max_turns * 90, f"{level}: within_ticks {wt} doesn't bind"
    # attrition cap present
    assert any("units_lost_lte" in cl for cl in win), (
        f"{level}: missing units_lost_lte attrition cap"
    )

    # every level can LOSE (timeout-fail at min)
    fail = _fail_clauses(c)
    assert any("after_ticks" in cl for cl in fail), (
        f"{level}: missing timeout in fail_condition"
    )


def test_k_increases_monotonically_with_difficulty():
    pack = load_pack(PACK)
    ks = []
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        ks.append([cl["enemies_discovered_gte"] for cl in _win_clauses(c) if "enemies_discovered_gte" in cl][0])
    assert ks == sorted(ks) and ks[0] < ks[-1], (
        f"K must scale with difficulty; got {ks}"
    )


def test_hidden_actor_count_matches_K():
    """K hidden enemy CLUSTERS must actually exist on the map — and
    each cluster must be a single actor (count:1) so discovering one
    cluster is worth exactly 1 toward K (a count:2 squad would let a
    K-cluster win trigger on K/2 clusters)."""
    pack = load_pack(PACK)
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        enemy_actors = [a for a in c.scenario.actors if a.owner == "enemy"]
        positions = {(a.position[0], a.position[1]) for a in enemy_actors}
        assert len(positions) >= EXPECTED_K[level], (
            f"{level}: only {len(positions)} enemy positions on map, "
            f"win needs {EXPECTED_K[level]}"
        )
        # Each hidden squad is exactly one actor: count is 1 (or unset).
        for a in enemy_actors:
            assert (a.count or 1) == 1, (
                f"{level}: enemy squad at {a.position} has count "
                f"{a.count}; each cluster must be count:1 so "
                f"enemies_discovered_gte counts clusters, not bodies"
            )


def test_scout_column_is_hold_fire():
    """The agent scout column must be stance:0 (HoldFire) on every
    level. The default stance:3 (AttackAnything) actively hunts after
    the engine stance-semantics fix — a stance:3 column self-delivers
    across the map and a do-nothing stall wins the perception task for
    free."""
    pack = load_pack(PACK)
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        agent = [a for a in c.scenario.actors if a.owner == "agent"]
        assert agent, f"{level}: no agent actors"
        for a in agent:
            assert a.stance == 0, (
                f"{level}: agent scout {a.type} at {a.position} has "
                f"stance {a.stance}; must be stance:0 so it does not "
                f"auto-hunt and hand a staller a free win"
            )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall(rs, C):
    """Only observe — never move. Discovers 0 enemies → the
    enemies_discovered_gte bar is never met → after_ticks LOSS."""
    return [C.observe()]


def _east_sweep(rs, C):
    """Wrong-path: drive the whole column straight east along one lane
    (y=10). A single-axis sweep reveals at most the clusters on that
    lane — fewer than K — and the off-lane pockets stay hidden →
    enemies_discovered_gte unmet → LOSS."""
    units = rs.get("units_summary", []) or []
    return [
        C.move_units([str(u["id"])], target_x=125, target_y=10) for u in units
    ] or [C.observe()]


def _brute(rs, C):
    """Brute: ram the whole column straight at mid-map (110,20) with no
    inference about where the K-th squad hides. Finds < K clusters and
    bleeds attrition charging the armed squads → LOSS."""
    units = rs.get("units_summary", []) or []
    return [
        C.move_units([str(u["id"])], target_x=110, target_y=20) for u in units
    ] or [C.observe()]


def _make_intended(level):
    """Intended split-scout: route one scout toward each hidden cluster
    in parallel; once all K are discovered, retreat the column home.
    Discovers exactly K clusters inside the clock → WIN."""
    clusters = CLUSTERS[level]
    target_k = EXPECTED_K[level]

    def _policy(rs, C):
        units = rs.get("units_summary", []) or []
        seen = rs.get("enemies_discovered", 0)
        cmds = []
        for i, u in enumerate(units):
            if seen >= target_k:
                cmds.append(C.move_units([str(u["id"])], target_x=4, target_y=20))
            else:
                cx, cy = clusters[i % len(clusters)]
                cmds.append(C.move_units([str(u["id"])], target_x=cx, target_y=cy))
        return cmds or [C.observe()]

    return _policy


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_split_scout_wins(level, seed):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    r = run_level(c, _make_intended(level), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended split-scout must WIN, got "
        f"{r.outcome} after {r.turns} turns "
        f"(discovered={len(r.signals.enemies_seen_ids)}, "
        f"lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE on the timeout "
        f"(discovers 0), got {r.outcome} "
        f"(discovered={len(r.signals.enemies_seen_ids)})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_wrong_path_single_axis_sweep_loses(level, seed):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _east_sweep, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-axis east sweep must LOSE "
        f"(finds < K clusters), got {r.outcome} "
        f"(discovered={len(r.signals.enemies_seen_ids)}, "
        f"K={EXPECTED_K[level]})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_ram_loses(level, seed):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _brute, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute ram-at-mid-map must LOSE "
        f"(finds < K, bleeds attrition), got {r.outcome} "
        f"(discovered={len(r.signals.enemies_seen_ids)}, "
        f"lost={r.signals.units_lost})"
    )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks fail must fit inside
    max_turns on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        at = [cl["after_ticks"] for cl in _fail_clauses(c) if "after_ticks" in cl][0]
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert at <= max_tick, (
            f"{lvl}: after_ticks {at} > max reachable tick {max_tick}"
        )
