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
