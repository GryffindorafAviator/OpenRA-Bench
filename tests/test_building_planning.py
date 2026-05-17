"""Building & Planning scenario family, full loop on Rust:

a scripted "architect" respects the tech-tree (barracks needs power),
builds + places structures, and the declarative win_condition scores it.
Proves placement + dependency + economy + region predicates end-to-end.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "building-and-planning.yaml"


def architect_agent(render_state, Command):
    """Plan a legal build order: power first, then the power-dependent
    barracks (tent). Issues build + place; place no-ops until ready."""
    btypes = [b["type"] for b in render_state.get("own_buildings", [])]
    producing = render_state.get("production", []) or []
    powr_n = sum(t == "powr" for t in btypes)
    if powr_n < 2:
        cmds = []
        if "powr" not in producing:
            cmds.append(Command.build("powr"))
        cmds.append(Command.place_building("powr", 18, 18))
        return cmds
    if "tent" not in btypes:
        cmds = []
        if "tent" not in producing:
            cmds.append(Command.build("tent"))
        cmds.append(Command.place_building("tent", 12, 22))
        return cmds
    return [Command.observe()]


def test_easy_tech_dependent_build_order_wins_and_scores():
    pack = load_pack(PACK)
    c = compile_level(pack, "easy")
    assert c.map_supported and c.starting_cash == 6000

    res = run_level(c, architect_agent, seed=1)

    # Tech-dependent barracks got built (needs a power plant first).
    assert "tent" in res.signals.own_building_types, (
        f"barracks (tech-dependent) not built; buildings="
        f"{res.signals.own_building_types}"
    )
    assert len(res.signals.own_buildings) >= 4, res.signals.own_buildings
    assert res.signals.cash < 6000, "construction must debit the budget"
    assert res.outcome == "win", f"easy should be winnable, got {res.outcome}"

    sc = score_episode(c, res)
    assert sc.outcome == "win" and 0.0 <= sc.composite <= 1.0


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_medium_hard_run_and_score_within_constraints(level):
    """Harder levels need richer planning than the scripted architect;
    assert they execute within their declared constraints and score
    deterministically (the contributor loop is robust even on a loss)."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    a = run_level(c, architect_agent, seed=3)
    b = run_level(c, architect_agent, seed=3)
    assert a.outcome in {"win", "draw", "loss"}
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must be deterministic"
    sc = score_episode(c, a)
    assert 0.0 <= sc.composite <= 1.0
    # Budget constraint is enforced by the engine.
    assert a.signals.cash <= c.starting_cash
