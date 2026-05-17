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


def test_unsupported_map_is_skipped_not_crashed(tmp_path):
    """A pack on a non-Rust map must be reported as skipped, not raise."""
    pack = (PACKS / "perception-frontier-reading.yaml").read_text()
    pack = pack.replace("base_map: rush-hour-arena", "base_map: some-future-map")
    f = tmp_path / "future.yaml"
    f.write_text(pack)
    stats = evaluate(packs=[f], levels=["easy"], seeds=[1])
    assert stats["overall"]["n"] == 0
    assert any("not Rust-loadable" in s for s in stats["skipped"])
