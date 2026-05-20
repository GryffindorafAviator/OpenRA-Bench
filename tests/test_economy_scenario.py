"""Full contributor loop for the economy-force-buildup scenario on Rust:

  pack YAML  ->  compile (per-level starting_cash)  ->  temp scenario
  ->  Rust engine (fact+powr+tent + cash)  ->  build/place_building  ->
  economy/building observation  ->  declarative win/fail  ->  score.

This is the no-cheat + solvency proof: the lazy "click-build the
cheapest unit" / do-nothing / single-axis policies must LOSE every
level (not draw), and only the intended allocation (build+place a 2nd
power plant, THEN train the e3 force) WINS within the binding budget
and the tick-aligned deadline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "economy-force-buildup.yaml"


def spam_cheapest(render_state, Command):
    """Click-build the cheapest unit (e1) every turn — the degenerate
    'just spend' policy the old scenario rewarded."""
    return [Command.build("e1")]


def do_nothing(render_state, Command):
    return [Command.observe()]


def spam_quality(render_state, Command):
    """Train the required e3 but never make the structural investment."""
    return [Command.build("e3")]


def greedy_structures(render_state, Command):
    """Only queue power plants; never field a force."""
    return [Command.build("powr")]


def _make_allocator():
    """Intended allocation: queue + place a 2nd power plant, then
    convert the rest of the budget into the e3 force."""
    st = {"t": 0, "placed": False}

    def allocator(render_state, Command):
        st["t"] += 1
        own = render_state.get("own_buildings") or []
        n_powr = sum(
            1 for b in own if str(b.get("type", "")).lower() == "powr"
        )
        if st["t"] == 1:
            return [Command.build("powr")]
        if not st["placed"] and n_powr < 2:
            return [Command.place_building("powr", 17, 18)]
        st["placed"] = True
        return [Command.build("e3")]

    return allocator


LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy",
    [spam_cheapest, do_nothing, spam_quality, greedy_structures],
    ids=["spam_e1", "do_nothing", "spam_e3", "greedy_powr"],
)
def test_lazy_policies_lose_every_level_and_seed(level, policy):
    """No cheat: no lazy/greedy/spam/stall policy may win OR draw."""
    pack = load_pack(PACK)
    compiled = compile_level(pack, level)
    for seed in SEEDS:
        res = run_level(compiled, policy, seed=seed)
        assert res.outcome == "loss", (
            f"{level} seed={seed}: lazy policy must LOSE (no win, no "
            f"draw); got {res.outcome}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_intended_allocation_wins_every_level_and_seed(level):
    """Solvency: the correct allocation wins, within budget + clock."""
    pack = load_pack(PACK)
    compiled = compile_level(pack, level)
    for seed in SEEDS:
        res = run_level(compiled, _make_allocator(), seed=seed)
        assert res.outcome == "win", (
            f"{level} seed={seed}: intended allocation must WIN; "
            f"got {res.outcome} (tick={res.signals.game_tick}, "
            f"cash={res.signals.cash})"
        )
        sc = score_episode(compiled, res)
        assert sc.outcome == "win"
        assert 0.0 <= sc.composite <= 1.0


def test_starting_cash_constraint_binds_per_level():
    """The per-level budgets actually bind: cash never exceeds the
    designed ceiling, and the leaner levels start lower."""
    pack = load_pack(PACK)
    e = run_level(compile_level(pack, "easy"), spam_cheapest, seed=1)
    m = run_level(compile_level(pack, "medium"), spam_cheapest, seed=1)
    assert e.signals.cash <= 2400
    assert m.signals.cash <= 1900
    assert compile_level(pack, "easy").starting_cash == 2400
    assert compile_level(pack, "hard").starting_cash == 2150


def test_economy_run_is_deterministic():
    pack = load_pack(PACK)
    c = compile_level(pack, "easy")
    a = run_level(c, _make_allocator(), seed=7)
    b = run_level(c, _make_allocator(), seed=7)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must yield identical economy outcome"
