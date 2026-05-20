"""proc-instruction-following-edge-case: IFBench / BFCL V4 negative
instruction adherence (selective-action discipline).

The brief carries an EXPLICIT CONDITIONAL RULE — "move the JEEP units
to (90,20); do NOT move the TANK units." The bench measures whether
the agent honours BOTH halves of the instruction. The win predicate
checks:

  1. ≥3 jeeps in the goal region around (90,20).
  2. ≥3 tanks STILL in the start region around (8,20).

The fail predicate mirrors clause 2 as a `not` clause so a single tank
dragged out of the start cluster trips an IMMEDIATE loss (no need to
wait for the deadline) — that is the IFBench instruction-precision
teeth.

The bar (per CLAUDE.md) must hold on every level × every hard seed:

  - STALL          -> LOSS (clock)
  - MOVE-ALL       -> WIN on easy (no negative teeth), LOSS on
                       medium/hard (touching the tanks fails fast)
  - JEEPS-ONLY     -> WIN  (the intended capability play)
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "proc-instruction-following-edge-case.yaml"
)


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


# ── A. STRUCTURAL: predicate / clauses / deadlines wired correctly ───

def test_easy_has_positive_clause_only():
    """Easy is the calibration tier — only the positive instruction
    (jeeps reach (90,20)) is in the win predicate. Moving the tanks
    is harmless on easy (the negative teeth come in on medium)."""
    c = compile_level(load_pack(PACK), "easy")
    win = _win_clauses(c)
    type_clauses = [cl for cl in win if "units_of_type_in_region_gte" in cl]
    types = [cl["units_of_type_in_region_gte"]["type"] for cl in type_clauses]
    assert types == ["jeep"], f"easy: expected only the jeep clause, got {types}"


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_medium_and_hard_enforce_both_halves_of_the_instruction(level):
    """Medium and hard MUST encode BOTH:
      - positive: ≥3 jeeps at goal (90,20) and
      - negative: ≥3 tanks still at start (8,20).
    Anything less and the bench fails to enforce the carve-out."""
    c = compile_level(load_pack(PACK), level)
    win = _win_clauses(c)
    type_clauses = [cl for cl in win if "units_of_type_in_region_gte" in cl]
    by_type = {cl["units_of_type_in_region_gte"]["type"]: cl["units_of_type_in_region_gte"]
               for cl in type_clauses}
    assert "jeep" in by_type, f"{level}: missing positive jeep clause"
    assert by_type["jeep"]["x"] == 90 and by_type["jeep"]["y"] == 20, (
        f"{level}: jeep goal must be (90,20), got "
        f"({by_type['jeep']['x']},{by_type['jeep']['y']})"
    )
    assert by_type["jeep"]["n"] >= 3, f"{level}: jeep clause needs n>=3"
    assert "2tnk" in by_type, f"{level}: missing negative tanks-at-start clause"
    assert by_type["2tnk"]["x"] == 8 and by_type["2tnk"]["y"] == 20, (
        f"{level}: tanks-at-start must be (8,20), got "
        f"({by_type['2tnk']['x']},{by_type['2tnk']['y']})"
    )
    assert by_type["2tnk"]["n"] >= 3, f"{level}: tanks clause needs n>=3"


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_medium_and_hard_have_immediate_violation_fail_clause(level):
    """The fail predicate must include a `not units_of_type_in_region
    _gte 2tnk` clause so moving a tank trips fail IMMEDIATELY (no
    wait for deadline)."""
    c = compile_level(load_pack(PACK), level)
    fails = _fail_clauses(c)
    not_clauses = [cl["not"] for cl in fails if "not" in cl]
    has_immediate = any(
        "units_of_type_in_region_gte" in nc
        and nc["units_of_type_in_region_gte"]["type"] == "2tnk"
        for nc in not_clauses
    )
    assert has_immediate, (
        f"{level}: fail_condition must include `not units_of_type_in"
        f"_region_gte 2tnk` so moving a tank trips IMMEDIATE fail"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_has_binding_deadline_and_reachable_loss(level):
    c = compile_level(load_pack(PACK), level)
    win = _win_clauses(c)
    wt = [cl["within_ticks"] for cl in win if "within_ticks" in cl]
    assert wt, f"{level}: missing within_ticks deadline"
    # The deadline must bite within max_turns (engine ~90 ticks/turn).
    assert wt[0] < 93 + 90 * (c.max_turns - 1), (
        f"{level}: within_ticks {wt[0]} unreachable inside max_turns "
        f"{c.max_turns} (would draw on timeout)"
    )
    fail = _fail_clauses(c)
    at = [cl["after_ticks"] for cl in fail if "after_ticks" in cl]
    assert at, f"{level}: missing after_ticks in fail_condition"
    assert at[0] <= 93 + 90 * (c.max_turns - 1), (
        f"{level}: after_ticks {at[0]} unreachable inside max_turns "
        f"{c.max_turns} (would draw on timeout)"
    )


def test_hard_has_two_spawn_groups_for_jeeps():
    """Hard tier must have ≥2 jeep spawn_point groups (the
    test_hard_tier.py::UPGRADED contract). Tanks are anchored at
    (8..10, 20) across BOTH groups so the negative-instruction
    predicate is identical and well-defined across seeds."""
    c = compile_level(load_pack(PACK), "hard")
    agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
    sp = {(a.spawn_point if a.spawn_point is not None else 0) for a in agent_actors}
    assert len(sp) >= 2, (
        f"hard: must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )
    # Tanks must be duplicated in EVERY spawn group at the same coords
    # so the negative-instruction predicate is identical across seeds.
    tank_cells_per_group = {}
    for a in agent_actors:
        if a.type == "2tnk":
            g = a.spawn_point if a.spawn_point is not None else 0
            tank_cells_per_group.setdefault(g, set()).add(tuple(a.position))
    assert len(tank_cells_per_group) >= 2, (
        "hard: tanks must be present in BOTH spawn groups"
    )
    cell_sets = list(tank_cells_per_group.values())
    assert all(s == cell_sets[0] for s in cell_sets), (
        f"hard: tanks must occupy the SAME cells across spawn groups; "
        f"got {tank_cells_per_group}"
    )


# ── B. PREDICATE UNIT TEST: tank-at-start negative clause semantics ──

class _Sig:
    def __init__(self):
        self.game_tick = 100
        self.then_progress: dict = {}


def _ctx(units):
    return WinContext(signals=_Sig(), render_state={"units_summary": units})


def test_tanks_still_at_start_is_satisfied_by_unmoved_tanks():
    spec = {"units_of_type_in_region_gte":
        {"type": "2tnk", "x": 8, "y": 20, "radius": 6, "n": 3}}
    tanks = [
        {"type": "2tnk", "cell_x": 8, "cell_y": 20, "id": "1"},
        {"type": "2tnk", "cell_x": 9, "cell_y": 20, "id": "2"},
        {"type": "2tnk", "cell_x": 10, "cell_y": 20, "id": "3"},
    ]
    assert evaluate(spec, _ctx(tanks)) is True


def test_tanks_still_at_start_breaks_the_moment_a_tank_leaves_cluster():
    """The negative-instruction teeth: as soon as <3 tanks remain in
    the start region, the clause becomes False — so the fail-side
    `not` clause fires immediately."""
    spec = {"units_of_type_in_region_gte":
        {"type": "2tnk", "x": 8, "y": 20, "radius": 6, "n": 3}}
    # Two tanks still at start, one dragged east — predicate is False.
    moved = [
        {"type": "2tnk", "cell_x": 8, "cell_y": 20, "id": "1"},
        {"type": "2tnk", "cell_x": 9, "cell_y": 20, "id": "2"},
        {"type": "2tnk", "cell_x": 40, "cell_y": 20, "id": "3"},
    ]
    assert evaluate(spec, _ctx(moved)) is False
    # And the corresponding `not` (which IS the fail clause) is True.
    not_spec = {"not": spec}
    assert evaluate(not_spec, _ctx(moved)) is True


def test_jeeps_do_not_satisfy_the_tanks_at_start_clause():
    """Type-filter teeth: a jeep parked at the tanks' start does NOT
    count toward the 2tnk-at-start clause (the carve-out is on UNIT
    TYPE, not on cell occupancy)."""
    spec = {"units_of_type_in_region_gte":
        {"type": "2tnk", "x": 8, "y": 20, "radius": 6, "n": 3}}
    jeeps = [
        {"type": "jeep", "cell_x": 8, "cell_y": 20, "id": "j1"},
        {"type": "jeep", "cell_x": 9, "cell_y": 20, "id": "j2"},
        {"type": "jeep", "cell_x": 10, "cell_y": 20, "id": "j3"},
    ]
    assert evaluate(spec, _ctx(jeeps)) is False


# ── C. SOLVENCY / NO-CHEAT: scripted policies on the live engine ─────

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level  # noqa: E402


def _split(rs):
    us = rs.get("units_summary", []) or []
    jeeps = [str(u["id"]) for u in us if u.get("type") == "jeep"]
    tanks = [str(u["id"]) for u in us if u.get("type") == "2tnk"]
    return jeeps, tanks


def _stall(rs, C):
    return [C.observe()]


def _move_all(rs, C):
    """The classic IFBench failure mode: model treats the brief as
    'move everything east' and ignores the carve-out. On medium and
    hard the negative-instruction teeth fire the moment a tank
    leaves the start cluster."""
    jeeps, tanks = _split(rs)
    cmds = []
    if jeeps:
        cmds.append(C.move_units(jeeps, 90, 20))
    if tanks:
        cmds.append(C.move_units(tanks, 90, 20))
    return cmds or [C.observe()]


def _jeeps_only(rs, C):
    """The intended capability play: move ONLY the jeeps to the goal,
    issue NO order to the tanks. The tanks stay at their start and
    both halves of the instruction are honoured."""
    jeeps, _ = _split(rs)
    if not jeeps:
        return [C.observe()]
    return [C.move_units(jeeps, 90, 20)]


# Seeds 1..4 = held-out seed contract from CLAUDE.md.

@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE on timeout, got {res.outcome}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_move_all_loses_when_negative_teeth_present(level, seed):
    """On medium/hard the brief forbids moving the tanks; spraying
    move-east across both squads must LOSE the moment a tank leaves
    the start cluster."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _move_all, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed}: move-all (touched the tanks) must LOSE, "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_jeeps_only_wins(level, seed):
    """The intended selective-action policy must WIN on every level
    × every seed."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _jeeps_only, seed=seed)
    assert res.outcome == "win", (
        f"{level} seed{seed}: jeeps-only (intended play) must WIN, "
        f"got {res.outcome}"
    )
