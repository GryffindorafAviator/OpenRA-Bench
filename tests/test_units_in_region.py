"""units_in_region_gte predicate + action-multiunit-coordination
solvency/stability after the fix."""
from __future__ import annotations
from pathlib import Path
import pytest
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def _ctx(units):
    return WinContext(signals=type("S", (), {"game_tick": 100})(),
                      render_state={"units_summary": units})


def test_units_in_region_gte_counts_units_in_radius():
    u = [{"cell_x": 110, "cell_y": 6}, {"cell_x": 111, "cell_y": 7},
         {"cell_x": 5, "cell_y": 5}]
    v = {"x": 110, "y": 6, "radius": 8, "n": 2}
    assert evaluate({"units_in_region_gte": v}, _ctx(u)) is True
    v3 = {"x": 110, "y": 6, "radius": 8, "n": 3}
    assert evaluate({"units_in_region_gte": v3}, _ctx(u)) is False  # only 2
    # one touring unit no longer satisfies a >=2 split clause
    assert evaluate({"units_in_region_gte": v},
                    _ctx([{"cell_x": 110, "cell_y": 6}])) is False


def test_pack_enforces_split_and_has_real_fail():
    p = load_pack(PACKS / "action-multiunit-coordination.yaml")
    assert p.base["tools"] == ["move_units", "attack_unit", "stop"]
    for lv in ("easy", "medium", "hard"):
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        clauses = wc["all_of"]
        # split enforced via units_in_region_gte (n>=2), not reach_region
        regs = [cl for cl in clauses if "units_in_region_gte" in cl]
        assert len(regs) >= 2
        assert all(cl["units_in_region_gte"]["n"] >= 2 for cl in regs)
        assert not any("reach_region" in cl for cl in clauses)
        # binding deadline below the max_turns*~90 ceiling
        wt = [cl["within_ticks"] for cl in clauses if "within_ticks" in cl][0]
        assert wt < c.max_turns * 90, f"{lv}: within_ticks {wt} doesn't bind"
        # every level can emit a LOSS (deadline passed or force wiped)
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        assert "any_of" in fc
        assert any("after_ticks" in x for x in fc["any_of"])


@pytest.mark.parametrize("lv", ["easy", "medium", "hard"])
def test_compiles_and_runs(lv):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(
        load_pack(PACKS / "action-multiunit-coordination.yaml"), lv)
    assert c.map_supported
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    # do-nothing now LOSES on the deadline (not the old loss==draw)
    assert res.outcome == "loss"


def test_split_dash_policy_can_win_easy():
    """Solvency: a parallel split policy reaches >=2 units in BOTH
    regions before the deadline (the scenario IS winnable)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(
        load_pack(PACKS / "action-multiunit-coordination.yaml"), "easy")

    def split(rs, C):
        us = rs.get("units_summary", []) or []
        ids = [str(u["id"]) for u in us]
        if len(ids) < 2:
            return [C.observe()]
        half = len(ids) // 2
        return [C.move_units(ids[:half], 110, 6),
                C.move_units(ids[half:], 110, 33)]

    res = run_level(c, split, seed=1)
    assert res.outcome == "win", f"split policy should win easy, got {res.outcome}"
