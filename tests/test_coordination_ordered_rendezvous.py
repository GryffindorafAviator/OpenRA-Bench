"""coordination-ordered-rendezvous: ordered multi-waypoint delivery.

Structural tests verify the win predicate uses waypoint_sequence (not
parallel units_in_region clauses) so order is enforced — and that the
overall deadline binds. A pure-predicate unit test on waypoint_sequence
also confirms the order-violation case (later waypoint visited first
does NOT satisfy the sequence).
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
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "coordination-ordered-rendezvous.yaml"
)

EXPECTED_WAYPOINTS = {"easy": 2, "medium": 3, "hard": 4}


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


def _seq_value(c):
    for cl in _win_clauses(c):
        if "waypoint_sequence" in cl:
            return cl["waypoint_sequence"]
    return None


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_uses_waypoint_sequence_not_simultaneous_regions(level):
    pack = load_pack(PACK)
    assert pack.meta.capability == "action"
    c = compile_level(pack, level)

    win = _win_clauses(c)
    seq = _seq_value(c)
    assert seq is not None, f"{level}: must use waypoint_sequence for order"
    assert seq["n"] == 2, f"{level}: each leg requires >=2 units"
    assert len(seq["points"]) == EXPECTED_WAYPOINTS[level], (
        f"{level}: expected {EXPECTED_WAYPOINTS[level]} waypoints, got {len(seq['points'])}"
    )
    # critically NOT using simultaneous units_in_region (the
    # action-multiunit-coordination shape) — that doesn't enforce order
    assert not any("units_in_region_gte" in cl for cl in win), (
        f"{level}: should use waypoint_sequence, not units_in_region_gte"
    )

    # deadline binds
    wt = [cl["within_ticks"] for cl in win if "within_ticks" in cl][0]
    assert wt < c.max_turns * 90, f"{level}: within_ticks {wt} doesn't bind"

    # every level can LOSE
    fail = _fail_clauses(c)
    assert any("after_ticks" in cl for cl in fail), (
        f"{level}: missing timeout in fail_condition"
    )


def test_waypoint_count_scales_with_difficulty():
    pack = load_pack(PACK)
    counts = []
    for level in ("easy", "medium", "hard"):
        c = compile_level(pack, level)
        counts.append(len(_seq_value(c)["points"]))
    assert counts == [2, 3, 4], f"waypoint count should ladder 2->3->4, got {counts}"


# ---- pure-Python predicate unit tests on waypoint_sequence ordering ----


class _FakeSignals:
    def __init__(self):
        self.game_tick = 100
        self.seq_progress: dict = {}


def _ctx(units):
    return WinContext(signals=_FakeSignals(), render_state={"units_summary": units})


def test_waypoint_sequence_advances_only_in_order():
    sig = _FakeSignals()
    pts = [{"x": 110, "y": 8}, {"x": 110, "y": 32}]
    spec = {"waypoint_sequence": {"id": "t", "n": 2, "radius": 7, "points": pts}}

    # Visit W2 first — must NOT satisfy
    ctx = WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": 110, "cell_y": 32}, {"cell_x": 111, "cell_y": 32},
        ]},
    )
    assert evaluate(spec, ctx) is False
    # idx must still be 0 (no premature advance from W2-only presence)
    assert sig.seq_progress.get("t", 0) == 0

    # Now visit W1
    ctx = WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": 110, "cell_y": 8}, {"cell_x": 111, "cell_y": 8},
        ]},
    )
    assert evaluate(spec, ctx) is False  # W1 advances; W2 not yet
    assert sig.seq_progress["t"] == 1

    # Then visit W2 — sequence completes
    ctx = WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": 110, "cell_y": 32}, {"cell_x": 111, "cell_y": 32},
        ]},
    )
    assert evaluate(spec, ctx) is True
    assert sig.seq_progress["t"] == 2


def test_waypoint_sequence_requires_min_n_units():
    sig = _FakeSignals()
    pts = [{"x": 110, "y": 8}, {"x": 110, "y": 32}]
    spec = {"waypoint_sequence": {"id": "t2", "n": 2, "radius": 7, "points": pts}}

    # Only 1 unit at W1 — does not advance
    ctx = WinContext(
        signals=sig,
        render_state={"units_summary": [{"cell_x": 110, "cell_y": 8}]},
    )
    assert evaluate(spec, ctx) is False
    assert sig.seq_progress.get("t2", 0) == 0


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_pack_runs_and_donothing_loses(level):
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    assert res.outcome == "loss", (
        f"{level}: do-nothing should LOSE on timeout, got {res.outcome}"
    )
