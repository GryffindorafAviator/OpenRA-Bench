"""waypoint_sequence: stateful ordered-visit latch.

Enforces W1→W2→…→Wk IN ORDER on a per-episode signals scratch, so
'sequenced execution' scenarios actually test the sequence (skip a
waypoint or arrive out of order ⇒ never satisfied) instead of being
beaten by a beeline-to-final.
"""
from __future__ import annotations

import types

import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.game_knowledge import objective_brief
from openra_bench.scenarios.win_conditions import WinContext, evaluate


def _sig():
    # SimpleNamespace so the predicate can latch onto signals.seq_progress.
    return types.SimpleNamespace(game_tick=100, seq_progress={})


def _ctx(sig, units):
    return WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": x, "cell_y": y} for x, y in units
        ]},
    )


SEQ = {
    "id": "A", "radius": 5,
    "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 90, "y": 10}],
}


def test_in_order_completes_only_after_last():
    s = _sig()
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(10, 10)])) is False
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(50, 10)])) is False
    # latched W1,W2; now reach W3 → satisfied
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(90, 10)])) is True


def test_skipping_a_waypoint_never_satisfies():
    s = _sig()
    # Jump straight to the final point, never touching W1/W2.
    for _ in range(6):
        assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(90, 10)])) is False
    assert s.seq_progress["A"] == 0  # stuck at W1; sequence unbeatable


def test_out_of_order_does_not_advance():
    s = _sig()
    # At W2 before W1 → no progress.
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(50, 10)])) is False
    assert s.seq_progress["A"] == 0
    # Now W1 → advance to 1; W3 (skipping W2) → stays at 1.
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(10, 10)])) is False
    assert evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(90, 10)])) is False
    assert s.seq_progress["A"] == 1


def test_latch_is_monotonic_across_turns():
    s = _sig()
    evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(10, 10)]))   # W1
    # Unit wanders far away — progress must NOT regress.
    evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(0, 0)]))
    assert s.seq_progress["A"] == 1
    evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(50, 10)]))   # W2
    evaluate({"waypoint_sequence": SEQ}, _ctx(s, [(0, 0)]))
    assert s.seq_progress["A"] == 2


def test_two_independent_sequences_both_required():
    s = _sig()
    A = SEQ
    B = {"id": "B", "radius": 5,
         "points": [{"x": 10, "y": 30}, {"x": 90, "y": 30}]}
    cond = {"all_of": [{"waypoint_sequence": A}, {"waypoint_sequence": B}]}
    # Column 1 finishes A; column 2 only partway through B → not yet.
    evaluate(cond, _ctx(s, [(10, 10), (10, 30)]))
    evaluate(cond, _ctx(s, [(50, 10), (10, 30)]))
    assert evaluate(cond, _ctx(s, [(90, 10), (10, 30)])) is False  # A done, B at W1
    assert evaluate(cond, _ctx(s, [(90, 10), (90, 30)])) is True   # both done
    assert s.seq_progress == {"A": 3, "B": 2}


def test_n_units_required_per_waypoint():
    s = _sig()
    seq = {"id": "C", "radius": 5, "n": 2,
           "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}]}
    # one unit at W1 is not enough (need 2)
    assert evaluate({"waypoint_sequence": seq}, _ctx(s, [(10, 10)])) is False
    assert s.seq_progress["C"] == 0
    assert evaluate({"waypoint_sequence": seq},
                    _ctx(s, [(10, 10), (11, 10)])) is False  # W1 ok (2 units)
    assert s.seq_progress["C"] == 1
    assert evaluate({"waypoint_sequence": seq},
                    _ctx(s, [(50, 10), (51, 10)])) is True


def test_objective_brief_renders_ordered_and_relative():
    wc = {"all_of": [{"waypoint_sequence": {
        "id": "A", "radius": 5,
        "points": [
            {"x": 10, "y": 10, "label": "the far NORTH-WEST"},
            {"x": 90, "y": 10, "label": "the far NORTH-EAST"},
        ]}}]}
    exact = objective_brief("", wc, None, 30, "exact")
    rel = objective_brief("", wc, None, 30, "relative")
    assert "IN ORDER" in exact and "(10,10)" in exact
    assert "the far NORTH-WEST" in rel and "(10,10)" not in rel
    assert "10" not in rel  # no coordinate leak in relative mode


def test_hard_base_map_override_takes_effect():
    """base_map must be set INSIDE overrides (a Level-level base_map is
    silently ignored) — every tier on the audit-tight pack overrides
    to a dedicated procedural arena, not the pack-default
    rush-hour-arena."""
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import PACKS_DIR, compile_level
    p = load_pack(PACKS_DIR / "action-sequenced-execution.yaml")
    assert compile_level(p, "easy").scenario.base_map == "action-sequenced-execution-arena-easy"
    assert compile_level(p, "medium").scenario.base_map == "action-sequenced-execution-arena-medium"
    hard = compile_level(p, "hard")
    assert hard.scenario.base_map == "action-sequenced-execution-arena-hard"
    assert hard.map_supported, "hard arena must resolve to a real .oramap"
