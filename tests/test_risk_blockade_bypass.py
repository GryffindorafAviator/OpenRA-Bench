"""risk-blockade-bypass: route-risk decision with two visible routes
(heavy corridor vs longer detour).

Structural tests verify the win predicate combines a positional
objective (reach_region) with a binding attrition cap and clock —
the three pieces that make this a risk decision, not a navigation
or combat test. Hard adds a tempo gate (after_ticks) tightening the
risk calculus (can't speedrun).
"""
from __future__ import annotations

from pathlib import Path

import pytest

# openra_bench.scenarios eagerly imports the Rust adapter at module
# load (schema.py:15), so collection fails without the wheel. Skip the
# whole module if the env is missing — matches test_building_planning.
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "risk-blockade-bypass.yaml"
)

# (deadline, loss_cap, after_ticks_gate_or_None)
# Tempo gate dropped from hard to keep the difficulty axis on route-
# risk only (CLAUDE.md C.10: one new controlled variable per tier).
EXPECTED = {
    "easy":   (3500, 4, None),
    "medium": (2800, 2, None),
    "hard":   (2400, 1, None),
}


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_encodes_reach_with_attrition_and_clock(level):
    pack = load_pack(PACK)
    assert pack.meta.capability == "reasoning"
    c = compile_level(pack, level)

    win = _win_clauses(c)
    reach = [cl["reach_region"] for cl in win if "reach_region" in cl]
    deadline = [cl["within_ticks"] for cl in win if "within_ticks" in cl]
    loss_cap = [cl["units_lost_lte"] for cl in win if "units_lost_lte" in cl]
    after = [cl["after_ticks"] for cl in win if "after_ticks" in cl]

    assert len(reach) == 1, f"{level}: must reach exactly one region"
    assert reach[0]["x"] == 110, f"{level}: objective should be at the east marker"
    assert len(deadline) == 1 and len(loss_cap) == 1, (
        f"{level}: must have both within_ticks and units_lost_lte"
    )

    exp_deadline, exp_loss, exp_after = EXPECTED[level]
    assert deadline[0] == exp_deadline
    assert loss_cap[0] == exp_loss
    if exp_after is None:
        assert after == [], f"{level}: should not have an after_ticks gate"
    else:
        assert after == [exp_after], f"{level}: expected after_ticks={exp_after}, got {after}"


def test_attrition_cap_tightens_with_difficulty():
    caps = [EXPECTED[lv][1] for lv in ("easy", "medium", "hard")]
    assert caps == [4, 2, 1] and caps == sorted(caps, reverse=True), (
        f"attrition cap must tighten 4->2->1: {caps}"
    )


def test_deadline_tightens_with_difficulty():
    deadlines = [EXPECTED[lv][0] for lv in ("easy", "medium", "hard")]
    assert deadlines == sorted(deadlines, reverse=True), (
        f"deadline must tighten: {deadlines}"
    )


def test_both_routes_visibly_present_on_map():
    """The map must contain BOTH the heavy-corridor garrison (mid-east)
    and the light-detour picket (north) — otherwise it isn't a route
    choice scenario."""
    pack = load_pack(PACK)
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        enemy = [a for a in c.scenario.actors if a.owner == "enemy"]
        # heavy corridor garrison at y~20 (mid-latitude)
        corridor = [a for a in enemy if 15 <= a.position[1] <= 25 and 40 <= a.position[0] <= 70]
        # light detour picket at y~5 (top)
        detour = [a for a in enemy if a.position[1] <= 10 and 40 <= a.position[0] <= 70]
        assert corridor, f"{level}: missing heavy corridor garrison"
        assert detour, f"{level}: missing light detour picket"


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_pack_runs_and_donothing_loses(level):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    assert res.outcome == "loss", (
        f"{level}: do-nothing should LOSE on timeout, got {res.outcome}"
    )
