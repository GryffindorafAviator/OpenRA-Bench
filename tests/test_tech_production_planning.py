"""tech-production-planning pack — full loop on Rust.

Canonical Family 2 (Sequential Dependency Planning) from the Benchmark
Design Proposal: easy = 2-hop chain (weap -> fix -> 3tnk), medium =
parallel branches under power_surplus_gte:0, hard = adaptive transition.

A scripted "tech-planner" agent walks the easy chain (build War Factory
-> Service Depot -> train one Heavy Tank) and proves the declarative
win_condition fires; medium/hard run deterministically with the same
agent (they need richer play than a scripted depth-first chain to win,
so we assert determinism + budget bound, mirroring test_building_planning).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "tech-production-planning.yaml"


def tech_planner_agent(render_state, Command):
    """Walk the 3tnk prereq chain depth-first.

    Pre-placed: fact, powr, proc. Build weap -> fix, then train one 3tnk.
    `build` queues the item; `place_building` no-ops until the building
    is ready to be placed.
    """
    btypes = [b["type"] for b in render_state.get("own_buildings", [])]
    producing = render_state.get("production", []) or []
    units = [u["type"] for u in render_state.get("units_summary", [])]

    if "weap" not in btypes:
        cmds = []
        if "weap" not in producing:
            cmds.append(Command.build("weap"))
        cmds.append(Command.place_building("weap", 18, 22))
        return cmds
    if "fix" not in btypes:
        cmds = []
        if "fix" not in producing:
            cmds.append(Command.build("fix"))
        cmds.append(Command.place_building("fix", 22, 22))
        return cmds
    if "3tnk" not in units:
        if "3tnk" not in producing:
            return [Command.build("3tnk")]
        return [Command.observe()]
    return [Command.observe()]


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "tech-production-planning"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_easy_tech_chain_wins_and_scores():
    pack = load_pack(PACK)
    c = compile_level(pack, "easy")
    assert c.map_supported
    assert c.starting_cash == 5000

    res = run_level(c, tech_planner_agent, seed=1)

    # The chain went through both weap AND fix (the prereq for 3tnk).
    btypes = res.signals.own_building_types
    assert "weap" in btypes and "fix" in btypes, (
        f"chain incomplete; own_building_types={btypes}"
    )
    # Production debited the budget.
    assert res.signals.cash < 5000, "construction must debit the budget"
    # A win implies a Heavy Tank was trained: the win condition
    # includes `unit_type_count_gte {type: 3tnk, n: 1}`, so outcome ==
    # "win" is itself proof the 3tnk was produced (and observed — this
    # also exercises the own-unit actor_type fix).
    assert res.outcome == "win", f"easy should be winnable, got {res.outcome}"

    sc = score_episode(c, res)
    assert sc.outcome == "win" and 0.0 <= sc.composite <= 1.0


def test_easy_run_is_deterministic():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, tech_planner_agent, seed=7)
    b = run_level(c, tech_planner_agent, seed=7)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must be deterministic"


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_medium_hard_run_and_score_within_constraints(level):
    """Harder levels need richer play than the scripted depth-first chain
    (medium wants parallel queues + 2nd powr; hard wants scouting +
    pivot to anti-armor). Assert they execute within declared constraints
    and score deterministically — the contributor loop must be robust
    even on a loss."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    a = run_level(c, tech_planner_agent, seed=3)
    b = run_level(c, tech_planner_agent, seed=3)
    assert a.outcome in {"win", "draw", "loss"}
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must yield identical outcome"
    sc = score_episode(c, a)
    assert 0.0 <= sc.composite <= 1.0
    # Budget constraint enforced by the engine.
    assert a.signals.cash <= c.starting_cash


def test_hard_fail_predicate_set():
    """Hard declares overproducing infantry past 10 = fail. Confirm the
    fail_condition is wired through compile (the engine actually checks
    it). Schema-level check, no episode run required."""
    c = compile_level(load_pack(PACK), "hard")
    assert c.fail_condition is not None
    # Hard loses on EITHER the timeout (deadline aligned to the tick
    # reachable at max_turns — no draw degeneracy) OR overproducing
    # infantry past 10.
    fc = c.fail_condition.model_dump(exclude_none=True)
    clauses = fc.get("any_of", [fc])
    assert {"unit_type_count_gte": {"type": "e1", "n": 10}} in clauses, fc
    assert any("after_ticks" in cl for cl in clauses), fc
