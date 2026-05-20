"""perception-count-the-threat: exact-K enemy discovery under fog.

Structural tests assert the predicate shape — each level discriminates
by the exact K (enemies_discovered_gte) and has a real fail_condition.
Run-smoke test confirms the pack loads and a do-nothing agent loses on
the timeout (proves fail_condition is wired through compile()).

Module-level importorskip because openra_bench.scenarios imports the
Rust adapter at import time (schema.py:15), so even pure-Python
structural tests need the wheel to load the package. Same pattern as
test_building_planning.py.
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


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


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
    """K hidden enemies must actually exist on the map — otherwise the
    win is unsatisfiable."""
    pack = load_pack(PACK)
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        enemy_actors = [a for a in c.scenario.actors if a.owner == "enemy"]
        # count distinct enemy *positions* (each group is a discoverable squad)
        positions = {(a.position[0], a.position[1]) for a in enemy_actors}
        assert len(positions) >= EXPECTED_K[level], (
            f"{level}: only {len(positions)} enemy positions on map, "
            f"win needs {EXPECTED_K[level]}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_pack_runs_and_donothing_loses(level):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    # do-nothing must lose on the timeout (proves fail_condition is live)
    assert res.outcome == "loss", (
        f"{level}: do-nothing should LOSE on timeout, got {res.outcome}"
    )
