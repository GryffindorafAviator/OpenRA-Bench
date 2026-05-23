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

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
FAMILY = [
    # artofwar-decoy-sacrifice was archived (see
    # openra_bench/scenarios/packs/_archive/); the sister packs cover
    # the decoy/sacrifice idiom in the active set.
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


def _seq_ctx(sig, units_xy):
    """A ctx sharing one persistent signals (so waypoint_sequence's
    latch carries across calls, like a real episode)."""
    return WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": x, "cell_y": y} for x, y in units_xy
        ]},
    )


def test_sequenced_citadel_enforces_ordered_chain_then_timed_strike():
    """Redesigned: win = waypoint_sequence(A→B→C, ordered/latched) AND
    a prerequisite hold (after_ticks) AND a reachable deadline. A
    beeline to C that skips A,B can NEVER satisfy the ordered latch;
    arriving in order but before the hold doesn't count; the timeout
    is a real, reachable LOSS (no draw degeneracy)."""
    import types
    c = compile_level(load_pack(PACKS / "artofwar-sequenced-citadel.yaml"),
                       "easy")
    A, B, Cc = (35, 20), (70, 12), (110, 20)

    # Beeline straight to C, skipping A and B — the ordered latch never
    # advances, so it never wins no matter the tick.
    sig = types.SimpleNamespace(game_tick=1500, units_lost=0, seq_progress={})
    for _ in range(4):
        assert evaluate(c.win_condition, _seq_ctx(sig, [Cc, Cc, Cc])) is False

    # Visit A → B in order, then arrive at C. Before the hold
    # (after_ticks): not yet. Inside the band: win. Past within_ticks:
    # no win (too late).
    sig = types.SimpleNamespace(game_tick=300, units_lost=0, seq_progress={})
    evaluate(c.win_condition, _seq_ctx(sig, [A, A, A]))      # latch A
    evaluate(c.win_condition, _seq_ctx(sig, [B, B, B]))      # latch B
    sig.game_tick = 800     # chain done but before the hold → no credit
    assert evaluate(c.win_condition, _seq_ctx(sig, [Cc, Cc, Cc])) is False
    sig.game_tick = 1800    # in [after_ticks, within_ticks] → win
    assert evaluate(c.win_condition, _seq_ctx(sig, [Cc, Cc, Cc])) is True
    sig.game_tick = 999999  # past the deadline → no win
    assert evaluate(c.win_condition, _seq_ctx(sig, [Cc, Cc, Cc])) is False

    # Timeout is a real LOSS and reachable within max_turns
    # (≈ 93 + 90·(max_turns-1), ~90 ticks/turn).
    lose = types.SimpleNamespace(game_tick=2301, units_lost=0,
                                 seq_progress={})
    assert evaluate(c.fail_condition, _seq_ctx(lose, [(6, 20)] * 3)) is True
    assert 2301 <= 93 + 90 * (c.max_turns - 1)


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


# test_decoy_hard_loss_cap_allows_bait_not_army was removed when
# artofwar-decoy-sacrifice was archived (file moved to _archive/ and
# marked status: quarantine).


def test_lure_the_tiger_win_requires_preserved_main_body_in_budget():
    """Redesigned (no-cheat, #4-idiom): far-east objective (112,20),
    win = THREE units in region (the main body, not one touring unit)
    AND a loss cap (only the two bait jeeps may be spent — a survivable
    head-on grab that burns armour cannot win) AND the deadline. The
    lure is physically required (the guard wall is lethal head-on) and
    the predicate refuses every lazy/greedy substitute."""
    for lvl, cap, wt in (("easy", 2, 5800), ("medium", 2, 6600),
                         ("hard", 2, 7400)):
        c = compile_level(load_pack(PACKS / "artofwar-lure-the-tiger.yaml"),
                          lvl)
        at_obj = [(112, 20)] * 3
        # intended: 3 tanks landed, only the 2 bait jeeps spent → win.
        assert evaluate(c.win_condition, _ctx(at_obj, wt - 100, lost=cap))
        # one touring unit reaching the region does NOT win (n=3 split,
        # not reach_region ≥1 — the old cheat).
        assert evaluate(
            c.win_condition, _ctx([(112, 20)], wt - 100, lost=0)
        ) is False
        # a survivable head-on grab that burned armour (lost > cap)
        # cannot win AND trips the fail clause.
        assert evaluate(c.win_condition, _ctx(at_obj, wt - 100,
                                              lost=cap + 1)) is False
        assert evaluate(c.fail_condition, _ctx(at_obj, wt - 100,
                                               lost=cap + 1)) is True
        # past the deadline → no win; the timeout is a real LOSS and is
        # reachable within max_turns (no draw degeneracy).
        assert evaluate(c.win_condition, _ctx(at_obj, 10 ** 7,
                                              lost=0)) is False
        assert evaluate(c.fail_condition, _ctx([(6, 20)] * 5, wt + 1,
                                               lost=0)) is True
        assert (wt + 1) <= 93 + 90 * (c.max_turns - 1)


def test_lure_the_tiger_hard_seed_varied_two_spawn_groups():
    """Hard-tier contract: ≥2 spawn_point groups → seed-varied start
    (the lure line can't be memorised). In-bounds (rush-hour y 2..38;
    the old pack had off-map y=40/32 → engine panic — fixed)."""
    c = compile_level(load_pack(PACKS / "artofwar-lure-the-tiger.yaml"),
                       "hard")
    groups = {a.spawn_point for a in c.scenario.actors
              if a.spawn_point is not None}
    assert groups == {0, 1}
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


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
