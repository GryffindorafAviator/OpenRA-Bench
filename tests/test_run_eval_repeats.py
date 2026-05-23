"""--repeats N: run each (cell, seed) N times so the report carries
the variance + pass^k a single inference per cell cannot give."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.run_eval import evaluate

PACK = (
    Path(__file__).parent.parent
    / "openra_bench" / "scenarios" / "packs"
    / "perception-count-the-threat.yaml"
)


def test_repeats_multiplies_the_task_count():
    base = evaluate([PACK], levels=["easy"], seeds=[1, 2],
                    repeats=1, dry_run=True)
    rep3 = evaluate([PACK], levels=["easy"], seeds=[1, 2],
                    repeats=3, dry_run=True)
    assert rep3["tasks"] == base["tasks"] * 3


def test_records_carry_repeat_index():
    stats = evaluate([PACK], levels=["easy"], seeds=[1], repeats=3)
    eps = stats.get("episodes", [])
    assert len(eps) == 3
    assert {e["repeat"] for e in eps} == {0, 1, 2}
    # all repeats stay at the same (cell, seed) — same key, different rep
    assert len({(e["cell"], e["seed"]) for e in eps}) == 1
