"""Goal tracker wired into scoring + leaderboard.

Partial objective credit must move the composite (a near-miss loss
beats a no-effort loss), the `objective` dimension/weight must exist,
and the aggregate path (run_eval._agg → ingest_run) must carry the
objective mean and the cumulative reward-vector signature.
"""

from __future__ import annotations

import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import EpisodeResult
from openra_bench.leaderboard import _capability_breakdown, ingest_run
from openra_bench.rust_adapter import EpisodeSignals
from openra_bench.run_eval import _agg
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scoring import score_episode

PACKS = __import__("pathlib").Path(__file__).parent.parent / (
    "openra_bench/scenarios/packs"
)


def _compiled():
    return compile_level(
        load_pack(PACKS / "perception-frontier-reading.yaml"), "easy"
    )


def _loss(progress: float) -> EpisodeResult:
    sig = EpisodeSignals(units_killed=0, units_lost=0, explored_percent=5.0,
                         game_tick=8000, outcome=0.0)
    return EpisodeResult(scenario="t", seed=0, turns=10, signals=sig,
                         outcome="loss", actions_issued=10, actions_warned=0,
                         objective_progress=progress,
                         reward_vector={"economy": 0.0, "military": 0.0,
                                        "territory": 0.05, "scouting": 0.0,
                                        "objective": progress})


def test_partial_objective_credit_beats_no_effort_loss():
    c = _compiled()
    near = score_episode(c, _loss(0.9))
    none = score_episode(c, _loss(0.0))
    assert near.composite > none.composite, "partial progress must score higher"
    assert "objective" in near.dimensions
    assert "objective" in near.weights and near.weights["objective"] > 0
    assert near.dimensions["objective"] == 0.9
    # and it lifts the reasoning proxy (planning moved the goal forward)
    assert near.reasoning > none.reasoning


def test_scenario_can_override_objective_weight():
    c = _compiled()
    c.scenario.reward = {"objective": 0.0}
    sc = score_episode(c, _loss(0.9))
    assert sc.weights["objective"] == 0.0  # override honored


def test_agg_and_ingest_carry_objective_and_reward_vector(tmp_path):
    c = _compiled()
    cards = [score_episode(c, _loss(0.8)), score_episode(c, _loss(0.4))]
    agg = _agg(cards)
    assert "objective_mean" in agg
    assert agg["objective_mean"] == round((0.8 + 0.4) / 2, 4)

    stats = {
        "episodes": [
            {"capability": "perception", "composite": 0.3,
             "outcome": "loss", "objective_progress": 0.8},
            {"capability": "perception", "composite": 0.1,
             "outcome": "loss", "objective_progress": 0.2},
        ],
        "overall": {"n": 2, "win_rate": 0.0, "composite_mean": 0.2,
                    "objective_mean": 0.5},
        "reward_vector_mean": {"economy": 0.1, "objective": 0.5},
        "summary": {"perception-frontier-reading:easy": {}},
    }
    rec = ingest_run(stats, "m1", store=tmp_path / "lb.jsonl")
    assert rec["objective"] == 0.5
    assert rec["reward_vector"] == {"economy": 0.1, "objective": 0.5}
    cap = _capability_breakdown(stats["episodes"])
    assert cap["perception"]["objective"] == round((0.8 + 0.2) / 2, 4)


def _win(tick: int, turns: int = 12) -> EpisodeResult:
    sig = EpisodeSignals(units_killed=1, units_lost=0, explored_percent=20.0,
                         game_tick=tick, outcome=1.0)
    return EpisodeResult(scenario="t", seed=0, turns=turns, signals=sig,
                         outcome="win", actions_issued=turns, actions_warned=0,
                         objective_progress=1.0,
                         reward_vector={"objective": 1.0})


def test_win_speed_bonus_orders_fast_above_slow_but_below_correctness():
    c = _compiled()  # perception-frontier-reading easy
    from openra_bench.scoring import _win_budget, SPEED_BONUS
    budget = _win_budget(c)
    fast = score_episode(c, _win(tick=budget // 5))
    slow = score_episode(c, _win(tick=int(budget * 0.95)))
    # recorded fields present on every win
    assert fast.win_tick == budget // 5 and fast.win_turns == 12
    assert fast.win_budget == budget and 0.0 < fast.speed <= 1.0
    # faster win ranks above slower win...
    assert fast.composite > slow.composite
    # ...by no more than the capped bonus (correctness still dominates)
    assert (fast.composite - fast.composite_base) <= SPEED_BONUS + 1e-9
    assert abs(slow.composite - slow.composite_base) <= SPEED_BONUS + 1e-9
    # a slow win still beats any loss (speed never rescues a loss)
    assert slow.composite > score_episode(c, _loss(0.9)).composite


def test_speed_zero_and_no_bonus_on_non_win():
    c = _compiled()
    lo = score_episode(c, _loss(0.5))
    assert lo.speed == 0.0 and lo.win_tick == 0 and lo.win_turns == 0
    assert lo.composite == lo.composite_base  # no bonus applied


def test_win_budget_prefers_tightest_within_ticks():
    from openra_bench.scoring import _win_budget
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level as _cl
    p = load_pack(PACKS / "action-sequenced-execution.yaml")
    # easy win has within_ticks: 2400
    assert _win_budget(_cl(p, "easy")) == 2400
    # medium within_ticks: 3000
    assert _win_budget(_cl(p, "medium")) == 3000
