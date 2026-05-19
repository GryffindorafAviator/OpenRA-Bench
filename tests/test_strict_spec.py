"""Strict-action-API family: spec fidelity as the objective.

The new unit-type-count predicates are the no-overproduction teeth;
asserted exhaustively as pure logic, plus the two packs compile and run
on the live engine and the overproduction fail-tree evaluates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def _ctx(types: list[str]):
    return WinContext(
        signals=type("S", (), {"game_tick": 100})(),
        render_state={"units_summary": [{"type": t} for t in types]},
    )


def test_unit_type_count_eq_is_exact_no_overproduction():
    c = _ctx(["e1", "e1", "e1"])
    assert evaluate({"unit_type_count_eq": {"type": "e1", "n": 3}}, c)
    # a 4th e1 breaks the exact spec (the strict property)
    c4 = _ctx(["e1", "e1", "e1", "e1"])
    assert not evaluate({"unit_type_count_eq": {"type": "e1", "n": 3}}, c4)
    # case-insensitive, ignores other types
    cm = _ctx(["E1", "e1", "e1", "e3", "e3"])
    assert evaluate({"unit_type_count_eq": {"type": "e1", "n": 3}}, cm)
    assert evaluate({"unit_type_count_eq": {"type": "e3", "n": 2}}, cm)


def test_unit_type_count_gte_as_fail_predicate():
    # the packs use _gte as the *fail* condition: ≥4 e1 ⇒ violation
    over = _ctx(["e1"] * 4)
    assert evaluate({"unit_type_count_gte": {"type": "e1", "n": 4}}, over)
    assert not evaluate(
        {"unit_type_count_gte": {"type": "e1", "n": 4}}, _ctx(["e1"] * 3)
    )


def test_bom_fail_tree_triggers_on_overproduction():
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    c = compile_level(load_pack(PACKS / "strict-production-bom.yaml"),
                       "medium")
    # exactly-spec state → win true, fail false
    good = _ctx(["e1", "e1", "e1", "e3", "e3"])
    assert evaluate(c.win_condition, good) is True
    assert evaluate(c.fail_condition, good) is False
    # overproduced e3 → fail tree (any_of) fires, win false
    bad = _ctx(["e1", "e1", "e1", "e3", "e3", "e3"])
    assert evaluate(c.fail_condition, bad) is True
    assert evaluate(c.win_condition, bad) is False


@pytest.mark.parametrize("pid", ["strict-production-bom", "strict-sequence"])
def test_strict_pack_compiles_and_runs(pid):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    pack = load_pack(PACKS / f"{pid}.yaml")
    for lvl in ("easy", "medium", "hard"):
        cc = compile_level(pack, lvl)
        assert cc.meta.capability == "action" and cc.map_supported
    res = run_level(compile_level(pack, "easy"),
                    lambda rs, C: [C.observe()], seed=1)
    assert res.outcome in {"win", "draw", "loss"} and res.turns >= 1


def test_scenario_tool_allowlist_is_restrictive():
    # the strict contract: the model only gets the declared tools.
    from openra_bench.scenarios import load_pack

    p = load_pack(PACKS / "strict-sequence.yaml")
    assert set(p.base.get("tools", [])) == {"move_units", "stop"}
    bom = load_pack(PACKS / "strict-production-bom.yaml")
    assert "attack_unit" not in bom.base.get("tools", [])


def test_strict_sequence_enforces_strict_order_no_cheat():
    """No-cheat invariant for strict-sequence: every level's win is a
    latched `waypoint_sequence` (so out-of-order / skip / beeline can
    never satisfy it) + a reachable within_ticks deadline, and the
    fail tree carries an after_ticks timeout that is genuinely
    reachable inside max_turns (tick ≈ 93 + 90·(turn-1)) so a staller
    LOSES, never draws. No force-loss clause (no-enemy map).
    """
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    pack = load_pack(PACKS / "strict-sequence.yaml")
    for lvl in ("easy", "medium", "hard"):
        cc = compile_level(pack, lvl)
        win = dict(cc.win_condition.__pydantic_extra__ or {})
        clauses = win["all_of"]
        wp = next(c["waypoint_sequence"] for c in clauses if "waypoint_sequence" in c)
        assert len(wp["points"]) >= 3, (lvl, "needs a real ordered route")
        within = next(c["within_ticks"] for c in clauses if "within_ticks" in c)
        fail = dict(cc.fail_condition.__pydantic_extra__ or {})
        after = next(
            c["after_ticks"] for c in fail["any_of"] if "after_ticks" in c
        )
        max_tick = 93 + 90 * (cc.max_turns - 1)
        # deadline + timeout are reachable inside the turn budget …
        assert within <= max_tick, (lvl, within, max_tick)
        assert after <= max_tick, (lvl, after, max_tick)
        # … and the timeout is strictly past the win deadline so a
        # non-completing run is a real LOSS, never a draw.
        assert after == within + 1, (lvl, after, within)
        # no force-loss clause: _NO_ENEMY classification stays valid.
        keys = {k for c in fail["any_of"] for k in c}
        assert "units_lost_lte" not in keys and "units_killed_gte" not in keys
