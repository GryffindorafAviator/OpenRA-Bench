"""Pairwise adversarial Elo: pure determinism/correctness + one
engine-backed comparative run (the user's 'pairwise conditions')."""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.pairwise import pairwise_elo, run_pairwise

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def test_pairwise_elo_ranks_dominant_model_first():
    scores = {
        "strong": {"c1:easy": 0.9, "c2:easy": 0.8, "c3:easy": 0.7},
        "weak": {"c1:easy": 0.2, "c2:easy": 0.3, "c3:easy": 0.1},
    }
    r = pairwise_elo(scores)
    assert r["rank"]["strong"] == 1 and r["rank"]["weak"] == 2
    assert r["elo"]["strong"] > r["elo"]["weak"]
    # strong beat weak on all 3 shared cells → match score 1.0
    assert r["matrix"]["strong"]["weak"] == 1.0
    assert r["matrix"]["weak"]["strong"] == 0.0
    assert r["shared_cells"]["strong|weak"] == 3


def test_pairwise_elo_is_deterministic_and_order_independent():
    s1 = {"a": {"x": 0.5}, "b": {"x": 0.6}, "c": {"x": 0.4}}
    s2 = {"c": {"x": 0.4}, "b": {"x": 0.6}, "a": {"x": 0.5}}  # reordered
    assert pairwise_elo(s1) == pairwise_elo(s2)


def test_ties_give_half_and_no_shared_cells_skipped():
    r = pairwise_elo({"a": {"x": 0.5}, "b": {"x": 0.5}, "lonely": {"y": 0.9}})
    assert r["matrix"]["a"]["b"] == 0.5
    assert r["shared_cells"]["a|lonely"] == 0
    assert r["elo"]["a"] == r["elo"]["b"] == 1000.0  # tie → no rating change


def test_needs_two_models():
    with pytest.raises(ValueError):
        run_pairwise([], [], [], {"only": lambda c: None})


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("openra_train"),
    reason="Rust env wheel not installed",
)
def test_run_pairwise_competent_beats_idle_on_engine():
    from openra_bench.eval_core import scripted_explore_agent

    agents = {
        "explorer": lambda c: scripted_explore_agent,
        "idle": lambda c: (lambda rs, Cmd: [Cmd.observe()]),
    }
    out = run_pairwise(
        [PACKS / "perception-frontier-reading.yaml"],
        ["easy"],
        [1, 2],
        agents,
    )
    pw = out["pairwise"]
    # The exploring agent should not rank below the idle one on a
    # perception/exploration scenario.
    assert pw["rank"]["explorer"] <= pw["rank"]["idle"]
    assert set(out["cell_scores"]) == {"explorer", "idle"}
    assert pw["elo"]["explorer"] >= pw["elo"]["idle"]
