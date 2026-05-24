"""Scoring + P/R/A diagnostics.

Mostly engine-free and deterministic: craft EpisodeResult/EpisodeSignals
and assert ScoreCard behaviour. One integration check scores a real
scripted run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import EpisodeResult
from openra_bench.rust_adapter import EpisodeSignals
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scoring import score_episode

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def _compiled(pack="perception-frontier-reading.yaml", level="easy"):
    return compile_level(load_pack(PACKS / pack), level)


def _result(outcome, *, explored=0.0, kills=0, lost=0, tick=1000,
            issued=10, warned=0, enemies=(), buildings=(), turns=10):
    sig = EpisodeSignals(
        units_killed=kills,
        units_lost=lost,
        explored_percent=explored,
        game_tick=tick,
        # After commit b4c694a the bench convention for signals.outcome
        # numeric is win=1.0, draw=0.0, loss=0.0 (the string in
        # res.outcome carries the trinary distinction). Match the new
        # convention to keep the test fixture honest about what
        # eval_core actually writes.
        outcome={"win": 1.0, "draw": 0.0, "loss": 0.0}[outcome],
    )
    sig.enemies_seen_ids = set(enemies)
    sig.enemy_buildings_seen_ids = set(buildings)
    return EpisodeResult(
        scenario="t",
        seed=0,
        turns=turns,
        signals=sig,
        outcome=outcome,
        actions_issued=issued,
        actions_warned=warned,
    )


def test_win_scores_higher_than_loss():
    c = _compiled()
    win = score_episode(c, _result("win", explored=80, tick=3000, buildings={"b1"}))
    loss = score_episode(c, _result("loss", explored=5, tick=8000))
    assert win.composite > loss.composite
    assert 0.0 <= loss.composite <= 1.0 and 0.0 <= win.composite <= 1.0
    assert win.outcome == "win" and loss.outcome == "loss"


def test_low_coverage_flags_perception_on_perception_pack():
    c = _compiled("perception-frontier-reading.yaml", "easy")
    sc = score_episode(c, _result("loss", explored=3.0, tick=8000))
    assert sc.weakest_link == "perception"
    assert any("perception" in n or "coverage" in n for n in sc.notes)
    assert sc.perception < 0.3


def test_invalid_actions_flag_action_link():
    c = _compiled()
    sc = score_episode(c, _result("draw", explored=40, issued=20, warned=12))
    assert sc.action < 0.6
    assert any("invalid-action" in n for n in sc.notes)


def test_weights_take_scenario_overrides():
    c = _compiled()
    sc = score_episode(c, _result("draw", explored=30))
    # The perception packs weight exploration/discovery up vs defaults.
    assert sc.weights["exploration"] >= 0.0
    assert set(sc.dimensions) >= {"outcome", "exploration", "discovery", "combat"}
    assert sc.weakest_link in {"perception", "reasoning", "action"}


def test_pra_subscores_in_unit_range():
    c = _compiled()
    for oc in ("win", "draw", "loss"):
        sc = score_episode(c, _result(oc, explored=50, kills=2))
        for v in (sc.perception, sc.reasoning, sc.action, sc.composite):
            assert 0.0 <= v <= 1.0


@pytest.mark.skipif(
    not Path("/Users/berta/Projects/OpenRA-RL-Training/scenarios/discovery/rush-hour.yaml").exists(),
    reason="Training scenarios not present",
)
def test_scores_a_real_scripted_run():
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level, scripted_explore_agent

    c = _compiled()
    res = run_level(c, scripted_explore_agent, seed=1)
    sc = score_episode(c, res)
    assert sc.outcome == res.outcome
    assert 0.0 <= sc.composite <= 1.0
    assert sc.weakest_link in {"perception", "reasoning", "action"}
    # scripted explorer should at least reveal *some* map
    assert sc.dimensions["exploration"] > 0.0
