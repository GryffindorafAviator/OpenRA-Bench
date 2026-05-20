"""tempo-strike-window: forced lull then kill quota.

Structural tests verify every level encodes (a) an after_ticks lull
gate, (b) a kill quota, (c) a within_ticks deadline, (d) an attrition
cap, AND (e) a "premature engagement" fail clause keyed off
units_killed_gte inside the lull window.
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
    / "tempo-strike-window.yaml"
)

# (lull_until, kill_target, deadline, loss_cap)
EXPECTED = {
    "easy":   (2000, 4, 4200, 3),
    "medium": (1800, 5, 3600, 2),
    "hard":   (1500, 6, 3000, 1),
}


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_encodes_lull_kill_deadline_attrition(level):
    pack = load_pack(PACK)
    assert pack.meta.capability == "reasoning"
    c = compile_level(pack, level)

    win = _win_clauses(c)
    after = [cl["after_ticks"] for cl in win if "after_ticks" in cl]
    killed = [cl["units_killed_gte"] for cl in win if "units_killed_gte" in cl]
    within = [cl["within_ticks"] for cl in win if "within_ticks" in cl]
    lost = [cl["units_lost_lte"] for cl in win if "units_lost_lte" in cl]

    assert len(after) == 1 and len(killed) == 1 and len(within) == 1 and len(lost) == 1, (
        f"{level}: must have exactly one of each (after_ticks/units_killed_gte/"
        f"within_ticks/units_lost_lte)"
    )
    lull_until, kill_target, deadline, loss_cap = EXPECTED[level]
    assert after[0] == lull_until, f"{level}: lull_until expected {lull_until}, got {after[0]}"
    assert killed[0] == kill_target, f"{level}: kill_target expected {kill_target}, got {killed[0]}"
    assert within[0] == deadline, f"{level}: deadline expected {deadline}, got {within[0]}"
    assert lost[0] == loss_cap, f"{level}: loss_cap expected {loss_cap}, got {lost[0]}"
    assert after[0] < within[0], f"{level}: lull must end before deadline"


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_fail_condition_includes_premature_engagement(level):
    """The 'premature engagement' fail = units_killed_gte:N AND
    within_ticks:lull-1 — i.e. having too many kills before the lull
    elapses immediately loses."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    fail = _fail_clauses(c)
    lull_until = EXPECTED[level][0]

    has_premature = False
    for cl in fail:
        if "all_of" in cl:
            inner = cl["all_of"]
            has_killed = any("units_killed_gte" in x for x in inner)
            within_cap = [x["within_ticks"] for x in inner if "within_ticks" in x]
            if has_killed and within_cap and within_cap[0] < lull_until:
                has_premature = True
                break
    assert has_premature, (
        f"{level}: missing premature-engagement fail "
        f"(units_killed_gte AND within_ticks < {lull_until})"
    )


def test_difficulty_tightens_lull_kill_deadline_attrition():
    """easy -> medium -> hard: shorter lull, higher kill bar, shorter
    deadline, tighter attrition cap."""
    lulls = [EXPECTED[lv][0] for lv in ("easy", "medium", "hard")]
    kills = [EXPECTED[lv][1] for lv in ("easy", "medium", "hard")]
    deadlines = [EXPECTED[lv][2] for lv in ("easy", "medium", "hard")]
    losses = [EXPECTED[lv][3] for lv in ("easy", "medium", "hard")]

    assert lulls == sorted(lulls, reverse=True), f"lull should shrink: {lulls}"
    assert kills == sorted(kills), f"kill target should grow: {kills}"
    assert deadlines == sorted(deadlines, reverse=True), f"deadline should shrink: {deadlines}"
    assert losses == sorted(losses, reverse=True), f"loss cap should shrink: {losses}"


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_pack_runs_and_donothing_loses(level):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    assert res.outcome == "loss", (
        f"{level}: do-nothing should LOSE on timeout, got {res.outcome}"
    )
