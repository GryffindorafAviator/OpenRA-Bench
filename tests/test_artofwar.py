"""Art-of-War long-horizon family: delayed terminal credit.

Asserts the *ordering* property that makes these long-horizon (not the
agent behaviour, which is the model's job): arriving at the objective
early must NOT win when a prerequisite hold (after_ticks) is unmet —
the enabling phase is unrewarded, credit lands only at the end. Plus
all 12 levels compile and run on the live engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
FAMILY = [
    "artofwar-decoy-sacrifice",
    "artofwar-indirect-approach",
    "artofwar-lure-the-tiger",
    "artofwar-sequenced-citadel",
]


def _ctx(units_xy, tick, lost=0):
    return WinContext(
        signals=type("S", (), {"game_tick": tick, "units_lost": lost})(),
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_sequenced_citadel_credits_only_after_prerequisite_hold():
    c = compile_level(load_pack(PACKS / "artofwar-sequenced-citadel.yaml"),
                       "easy")
    at_c = [(44, 20), (44, 20), (44, 20)]
    # arrived at the citadel EARLY (tick 1000 < after_ticks 3000):
    # the enabling phase isn't done → no terminal credit yet.
    assert evaluate(c.win_condition, _ctx(at_c, 1000)) is False
    # same position, prerequisite hold satisfied, inside deadline → win.
    assert evaluate(c.win_condition, _ctx(at_c, 8000)) is True
    # too late (past within_ticks) → no win.
    assert evaluate(c.win_condition, _ctx(at_c, 999999)) is False


def test_indirect_hard_requires_zero_loss_whole_force_arrival():
    c = compile_level(load_pack(PACKS / "artofwar-indirect-approach.yaml"),
                       "hard")
    # Redesigned: far-east objective (112,20); the WHOLE surviving force
    # (every unit, ≥3) must arrive with ZERO losses, in budget.
    at_obj = [(112, 20)] * 3
    assert evaluate(c.win_condition, _ctx(at_obj, 4000, lost=0)) is True
    # charging the lethal short lane (any loss) fails the hard rung.
    assert evaluate(c.win_condition, _ctx(at_obj, 4000, lost=1)) is False
    # a stale unit left behind (not all in region) fails all_units_in_region.
    assert evaluate(
        c.win_condition, _ctx([(112, 20), (112, 20), (40, 20)], 4000, lost=0)
    ) is False
    # past the deadline (within_ticks 5000) → no win.
    assert evaluate(c.win_condition, _ctx(at_obj, 999999, lost=0)) is False


def test_indirect_easy_short_lane_loss_fails_and_timeout_loses():
    """Easy: loss cap 1; the timeout fail must be reachable in max_turns
    (no draw degeneracy) — i.e. after_ticks <= 93 + 90*(max_turns-1)."""
    c = compile_level(load_pack(PACKS / "artofwar-indirect-approach.yaml"),
                       "easy")
    arrived = [(112, 20)] * 3
    assert evaluate(c.win_condition, _ctx(arrived, 1500, lost=0)) is True
    assert evaluate(c.win_condition, _ctx(arrived, 1500, lost=1)) is True
    # losing >1 (head-on charge) fails the win and trips the fail clause.
    assert evaluate(c.win_condition, _ctx(arrived, 1500, lost=2)) is False
    assert evaluate(c.fail_condition, _ctx(arrived, 1500, lost=2)) is True
    # timeout is a real LOSS, and reachable within max_turns.
    assert evaluate(c.fail_condition, _ctx([(6, 20)] * 3, 4001, lost=0)) is True
    assert 4001 <= 93 + 90 * (c.max_turns - 1)


def test_decoy_hard_loss_cap_allows_bait_not_army():
    c = compile_level(load_pack(PACKS / "artofwar-decoy-sacrifice.yaml"),
                       "hard")
    # Redesigned: far-east objective (112,20); hard caps losses at 2
    # (only the two bait jeeps may be spent — burning a tank fails).
    at_obj = [(112, 20)] * 3
    assert evaluate(c.win_condition, _ctx(at_obj, 5000, lost=2)) is True
    assert evaluate(c.win_condition, _ctx(at_obj, 5000, lost=3)) is False


@pytest.mark.parametrize("pid", FAMILY)
def test_artofwar_pack_compiles_and_runs(pid):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    pack = load_pack(PACKS / f"{pid}.yaml")
    for lvl in ("easy", "medium", "hard"):
        cc = compile_level(pack, lvl)
        assert cc.meta.capability == "reasoning" and cc.map_supported
        assert len(cc.meta.real_world_meaning) > 10
    res = run_level(compile_level(pack, "easy"),
                    lambda rs, C: [C.observe()], seed=1)
    assert res.outcome in {"win", "draw", "loss"} and res.turns >= 1
