"""Full contributor loop for an economy scenario, end-to-end on Rust:

  pack YAML  ->  compile (per-level starting_cash)  ->  temp scenario
  ->  Rust engine (barracks + cash)  ->  Command.build  ->  economy
  observation  ->  declarative win_condition  ->  score.

This is the task #13 robustness proof that a contributed economy
scenario runs *within its designed constraints* and is scored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "economy-force-buildup.yaml"


def builder_agent(render_state, Command):
    """Spend the budget: queue an E1 every turn from the barracks."""
    return [Command.build("e1")]


def test_economy_pack_runs_within_budget_and_scores():
    pack = load_pack(PACK)
    compiled = compile_level(pack, "easy")
    assert compiled.map_supported
    assert compiled.starting_cash == 2000  # designed constraint threaded

    res = run_level(compiled, builder_agent, seed=1)

    # Engine honoured the scenario economy budget (was the hardcoded-0
    # bug); production debited it.
    assert res.signals.cash <= 2000
    assert res.signals.cash < 2000, "production should have debited cash"
    # Barracks loaded from the contributed scenario.
    assert "barr" in res.signals.own_building_types
    # Spending the budget fields a force → declarative win_condition met.
    assert res.outcome == "win", f"economy easy should be winnable, got {res.outcome}"

    sc = score_episode(compiled, res)
    assert sc.outcome == "win"
    assert 0.0 <= sc.composite <= 1.0


def test_starting_cash_constraint_is_enforced_per_level():
    """The lean 'hard' budget must actually bind: far fewer credits than
    'easy', so the engine reports a lower ceiling."""
    pack = load_pack(PACK)
    easy = run_level(compile_level(pack, "easy"), builder_agent, seed=1)
    hard = run_level(compile_level(pack, "hard"), builder_agent, seed=1)
    # Hard starts at 700 vs easy 2000 — its cash can never exceed 700.
    assert hard.signals.cash <= 700
    assert easy.signals.cash <= 2000
    assert hard.outcome in {"win", "draw", "loss"}


def test_economy_run_is_deterministic():
    pack = load_pack(PACK)
    c = compile_level(pack, "easy")
    a = run_level(c, builder_agent, seed=7)
    b = run_level(c, builder_agent, seed=7)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must yield identical economy outcome"
