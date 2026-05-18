"""Adversarial 1v1 spotlight: ladder rating + family integ.

The ladder metric is pure logic (fast, exhaustive); the 3 packs are
also compiled and smoke-run on the live Rust engine to prove the new
`adversarial` capability flows end to end (schema → engine → score →
leaderboard breakdown).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.adversarial import (
    RUNGS,
    adversarial_summary,
    ladder_rating,
    ladder_ratings,
)
from openra_bench.leaderboard import _capability_breakdown, ingest_run

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
ADV = ["adversarial-duel", "adversarial-skirmish", "adversarial-siege"]


def test_ladder_rating_is_contiguous_from_easy():
    assert ladder_rating({}) == 0
    assert ladder_rating({"easy": "loss"}) == 0
    assert ladder_rating({"easy": "win"}) == 1
    assert ladder_rating({"easy": "win", "medium": "win"}) == 2
    assert ladder_rating(dict.fromkeys(RUNGS, "win")) == 3
    # non-contiguous: hard won but medium lost → still 1
    assert ladder_rating(
        {"easy": "win", "medium": "loss", "hard": "win"}
    ) == 1
    # draw does not clear a rung
    assert ladder_rating({"easy": "draw"}) == 0


def test_ladder_ratings_need_all_seeds_won():
    stats = {
        "episodes": [
            {"cell": "adversarial-duel:easy", "capability": "adversarial",
             "split": "public", "outcome": "win"},
            {"cell": "adversarial-duel:easy", "capability": "adversarial",
             "split": "public", "outcome": "loss"},  # one seed lost
            {"cell": "adversarial-duel:medium", "capability": "adversarial",
             "split": "public", "outcome": "win"},
            # non-adversarial + held-out ignored
            {"cell": "rush-hour:easy", "capability": "action",
             "split": "public", "outcome": "win"},
            {"cell": "adversarial-duel:hard", "capability": "adversarial",
             "split": "held_out", "outcome": "win"},
        ]
    }
    # easy not all-won → rating 0 (medium can't count, non-contiguous)
    assert ladder_ratings(stats) == {"adversarial-duel": 0}
    s = adversarial_summary(stats)
    assert s["packs"] == ["adversarial-duel"]
    assert s["mean_ladder_rating"] == 0.0 and s["max_rung"] == 3


def test_summary_and_leaderboard_carry_adversarial(tmp_path):
    stats = {
        "episodes": [
            {"cell": "adversarial-duel:easy", "capability": "adversarial",
             "split": "public", "outcome": "win", "composite": 0.7},
            {"cell": "adversarial-duel:medium", "capability": "adversarial",
             "split": "public", "outcome": "win", "composite": 0.6},
            {"cell": "adversarial-duel:hard", "capability": "adversarial",
             "split": "public", "outcome": "loss", "composite": 0.2},
        ],
        "overall": {"n": 3, "win_rate": 0.66, "composite_mean": 0.5},
        "adversarial": None,
    }
    stats["adversarial"] = adversarial_summary(stats)
    assert stats["adversarial"]["ladder_ratings"] == {"adversarial-duel": 2}
    rec = ingest_run(stats, "m1", store=tmp_path / "lb.jsonl")
    assert rec["adversarial_rating"] == 2 / 1  # mean over 1 pack
    assert rec["adversarial_ladders"] == {"adversarial-duel": 2}
    cap = _capability_breakdown(stats["episodes"])
    assert "adversarial" in cap and cap["adversarial"]["n"] == 3


@pytest.mark.parametrize("pid", ADV)
def test_adversarial_pack_compiles_and_runs(pid):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import compile_level

    pack = load_pack(PACKS / f"{pid}.yaml")
    for lvl in RUNGS:
        c = compile_level(pack, lvl)
        assert c.meta.capability == "adversarial"
        assert c.map_supported, f"{pid}:{lvl} map must be Rust-loadable"
    c = compile_level(pack, "easy")
    res = run_level(c, lambda rs, C: [C.observe()], seed=1)
    assert res.outcome in {"win", "draw", "loss"}
    assert res.turns >= 1
