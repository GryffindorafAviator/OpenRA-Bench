"""run_eval aggregation + report (scripted baseline, live Rust)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.run_eval import evaluate, write_report

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def test_evaluate_aggregates_and_reports(tmp_path):
    stats = evaluate(
        packs=[PACKS / "perception-frontier-reading.yaml"],
        levels=["easy"],
        seeds=[1, 2],
    )
    cell = "perception-frontier-reading:easy"
    assert cell in stats["summary"]
    s = stats["summary"][cell]
    assert s["n"] == 2
    assert 0.0 <= s["win_rate"] <= 1.0
    assert 0.0 <= s["composite_mean"] <= 1.0
    assert set(s["weakest_link_hist"]) <= {"perception", "reasoning", "action"}
    assert sum(s["weakest_link_hist"].values()) == 2

    o = stats["overall"]
    assert o["n"] == 2
    assert len(stats["episodes"]) == 2
    for ep in stats["episodes"]:
        assert ep["cell"] == cell
        assert ep["capability"] == "perception"
        assert ep["outcome"] in {"win", "draw", "loss"}
        assert 0.0 <= ep["composite"] <= 1.0

    out = tmp_path / "eval_stats.json"
    write_report(stats, out)
    loaded = json.loads(out.read_text())
    assert loaded["overall"]["n"] == 2


def test_held_out_split_reports_generalization_gap(tmp_path):
    stats = evaluate(
        packs=[PACKS / "perception-frontier-reading.yaml"],
        levels=["easy"],
        seeds=[1, 2],
        held_out_seeds=[7, 8],
    )
    assert "overall_held_out" in stats
    assert stats["overall_held_out"]["n"] == 2
    assert isinstance(stats["generalization_gap"], float)
    splits = {e["split"] for e in stats["episodes"]}
    assert splits == {"public", "held_out"}
    assert sum(e["split"] == "held_out" for e in stats["episodes"]) == 2
    # gap == public composite − held-out composite (sign can be ±).
    g = round(
        stats["overall"]["composite_mean"]
        - stats["overall_held_out"]["composite_mean"],
        4,
    )
    assert stats["generalization_gap"] == g

    # Leaderboard captures the gap.
    from openra_bench.leaderboard import build_table, ingest_run

    s = tmp_path / "lb.jsonl"
    ingest_run(stats, "m", s)
    row = build_table(s, min_episodes=1)[0]
    assert row["generalization_gap"] == stats["generalization_gap"]
    assert row["held_out_composite"] == stats["overall_held_out"]["composite_mean"]


def test_no_held_out_keeps_backward_compatible_shape():
    stats = evaluate(
        packs=[PACKS / "perception-frontier-reading.yaml"],
        levels=["easy"],
        seeds=[1],
    )
    assert "overall_held_out" not in stats and "generalization_gap" not in stats
    assert all(e["split"] == "public" for e in stats["episodes"])


def test_concurrency_is_deterministic_and_isolated():
    import json

    packs = [
        PACKS / "perception-frontier-reading.yaml",
        PACKS / "reasoning-frontier-commit.yaml",
    ]
    seq = evaluate(packs, ["easy"], [1, 2, 3], concurrency=1)
    par = evaluate(packs, ["easy"], [1, 2, 3], concurrency=4)
    # Same report regardless of worker scheduling (episodes sorted,
    # aggregates order-independent) — episodes ran in isolation.
    assert json.dumps(seq, sort_keys=True) == json.dumps(par, sort_keys=True)
    assert seq["overall"]["n"] == 6
    assert [e["seed"] for e in par["episodes"]] == sorted(
        e["seed"] for e in par["episodes"]
    ) or len({e["cell"] for e in par["episodes"]}) > 1


def test_unsupported_map_is_skipped_not_crashed(tmp_path):
    """A pack on a non-Rust map must be reported as skipped, not raise."""
    pack = (PACKS / "perception-frontier-reading.yaml").read_text()
    pack = pack.replace("base_map: rush-hour-arena", "base_map: some-future-map")
    f = tmp_path / "future.yaml"
    f.write_text(pack)
    stats = evaluate(packs=[f], levels=["easy"], seeds=[1])
    assert stats["overall"]["n"] == 0
    assert any("not Rust-loadable" in s for s in stats["skipped"])
