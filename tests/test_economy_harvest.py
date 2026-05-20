"""Catalog C13 — harvest economy, full loop on Rust (the user's
economy families): earn economic value by harvesting, scored by
economy_value; storage capacity comes from refineries/silos.
Closes task #14 end-to-end on the bench side."""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

TIMEBOX = PACKS_DIR / "economy-harvest-timebox.yaml"
INVEST = PACKS_DIR / "economy-harvest-investment.yaml"


def harvester(render_state, Command):
    ids = [str(u["id"]) for u in render_state.get("units_summary", [])]
    return [Command.harvest(ids, 22, 18)] if ids else [Command.observe()]


def test_timebox_earns_economy_value_and_scores():
    c = compile_level(load_pack(TIMEBOX), "easy")
    assert c.map_supported
    res = run_level(c, harvester, seed=1)

    # Harvest income actually accrued (start cash 200).
    ev = res.signals.cash + res.signals.resources
    assert ev > 200, f"no harvest income (economy_value={ev})"
    assert res.signals.resource_capacity == 2000, "proc storage cap"
    assert res.outcome == "win", f"easy harvest target should be met: {res.outcome}"

    sc = score_episode(c, res)
    assert sc.outcome == "win" and 0.0 <= sc.composite <= 1.0


def test_timebox_is_deterministic():
    c = compile_level(load_pack(TIMEBOX), "easy")
    a = run_level(c, harvester, seed=4)
    b = run_level(c, harvester, seed=4)
    assert (a.outcome, a.turns, a.signals.cash, a.signals.resources) == (
        b.outcome,
        b.turns,
        b.signals.cash,
        b.signals.resources,
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_investment_runs_and_scores_within_constraints(level):
    """Investment/storage decision needs richer play than a scripted
    harvester; assert it executes within its declared constraints and
    scores deterministically (contributor loop robust even on a loss)."""
    c = compile_level(load_pack(INVEST), level)
    a = run_level(c, harvester, seed=3)
    b = run_level(c, harvester, seed=3)
    assert a.outcome in {"win", "draw", "loss"}
    assert (a.outcome, a.turns) == (b.outcome, b.turns), "non-deterministic"
    assert a.signals.resource_capacity >= 2000  # proc present from the start
    sc = score_episode(c, a)
    assert 0.0 <= sc.composite <= 1.0
