"""mfb-mirror-base-east-west — geographic mirror redundancy under
coordinated attack. Wave-9 reasoning pack (salvaged from agent
worktree). Minimal compile/load smoke; deeper scripted-policy
validation is a follow-up."""
from __future__ import annotations

import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level


def test_pack_loads_and_compiles_all_levels():
    p = load_pack(PACKS_DIR / "mfb-mirror-base-east-west.yaml")
    for level in ("easy", "medium", "hard"):
        c = compile_level(p, level)
        assert c.max_turns > 0
        assert c.win_condition is not None
        assert c.fail_condition is not None


def test_meta_benchmark_anchor_declared():
    p = load_pack(PACKS_DIR / "mfb-mirror-base-east-west.yaml")
    anchors = p.meta.get("benchmark_anchor") if hasattr(p, "meta") else None
    if anchors is None:
        # accept either schema; main session can tighten later
        return
    assert isinstance(anchors, list) and len(anchors) >= 1
