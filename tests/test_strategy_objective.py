"""Strategy packs: faithful "destroy the enemy's key economic
buildings" objective (training design — fact+proc are MustBeDestroyed;
the enemy is deliberately strong so brute force loses).

Covers the new adapter destruction signal, the two predicates, the
non-gameability (discovering a building no longer wins), and that all
three packs compile + run on the live engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
STRAT = ["strategy-dilemma", "strategy-gauntlet", "strategy-twobody"]


# ── adapter: destruction detection (absent + cell explored ⇒ killed) ──


def test_adapter_counts_destroyed_only_when_unit_is_present():
    a = RustObsAdapter()
    # t1: scout sees fact+proc from afar (unit at 5,5) — nothing killed.
    a.observe({
        "unit_positions": {"1": {"cell_x": 5, "cell_y": 5}},
        "enemy_buildings_summary": [
            {"id": 900, "type": "fact", "cell_x": 80, "cell_y": 35},
            {"id": 901, "type": "proc", "cell_x": 80, "cell_y": 32},
        ],
    })
    assert a.signals.enemy_buildings_destroyed == 0
    # t2: a unit is on the proc (79,33) and proc is gone ⇒ killed;
    # fact still standing.
    a.observe({
        "unit_positions": {"1": {"cell_x": 79, "cell_y": 33}},
        "enemy_buildings_summary": [
            {"id": 900, "type": "fact", "cell_x": 80, "cell_y": 35},
        ],
    })
    assert a.signals.enemy_buildings_destroyed == 1
    assert a.signals.enemy_buildings_destroyed_types == {"proc": 1}
    # t3: fact also absent but the force RETREATED far away ⇒ that's
    # fog, not a kill — must NOT be counted.
    a.observe({
        "unit_positions": {"1": {"cell_x": 5, "cell_y": 5}},
        "enemy_buildings_summary": [],
    })
    assert a.signals.enemy_buildings_destroyed == 1  # unchanged (fog)
    # t4: a unit is back on the fact cell and it's gone ⇒ now a kill.
    a.observe({
        "unit_positions": {"1": {"cell_x": 79, "cell_y": 36}},
        "enemy_buildings_summary": [],
    })
    assert a.signals.enemy_buildings_destroyed == 2
    assert a.signals.enemy_buildings_destroyed_types == {"proc": 1, "fact": 1}


# ── predicates ──


class _Sig:
    def __init__(self, destroyed_types=None, seen=0):
        self.enemy_buildings_destroyed_types = destroyed_types or {}
        self.enemy_buildings_destroyed = sum(
            (destroyed_types or {}).values()
        )
        self.enemy_buildings_seen_ids = set(range(seen))
        self.game_tick = 100


def _ctx(sig):
    return WinContext(signals=sig, render_state={})


def test_enemy_key_buildings_predicate_requires_all_types():
    assert evaluate(
        {"enemy_key_buildings_destroyed": {"types": ["fact", "proc"]}},
        _ctx(_Sig({"fact": 1, "proc": 1})),
    )
    # only one of the two ⇒ not satisfied
    assert not evaluate(
        {"enemy_key_buildings_destroyed": {"types": ["fact", "proc"]}},
        _ctx(_Sig({"fact": 1})),
    )
    assert evaluate(
        {"enemy_buildings_destroyed_gte": 2},
        _ctx(_Sig({"fact": 1, "proc": 1})),
    )


def test_objective_is_not_gameable_by_mere_discovery():
    c = compile_level(load_pack(PACKS / "strategy-dilemma.yaml"), "easy")
    # Saw the whole enemy base but destroyed nothing → NOT a win
    # (the old buildings_discovered_gte:1 bug would have passed here).
    seen_only = _Sig(destroyed_types={}, seen=5)
    assert evaluate(c.win_condition, _ctx(seen_only)) is False
    # fact+proc down, in time → win
    won = _Sig({"fact": 1, "proc": 1})
    won.game_tick = 1000
    assert evaluate(c.win_condition, _ctx(won)) is True


@pytest.mark.parametrize("pid", STRAT)
def test_strategy_pack_compiles_runs_and_has_faithful_objective(pid):
    pack = load_pack(PACKS / f"{pid}.yaml")
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        node = dict(c.win_condition.__pydantic_extra__ or {})
        clauses = node.get("all_of", [])
        assert any("enemy_key_buildings_destroyed" in cl for cl in clauses), (
            f"{pid}:{lvl} win must require destroying fact+proc"
        )
        assert c.fail_condition is not None  # loss reachable (brute force)
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    res = run_level(compile_level(pack, "easy"),
                    lambda rs, C: [C.observe()], seed=1)
    assert res.outcome in {"win", "draw", "loss"} and res.turns >= 1
