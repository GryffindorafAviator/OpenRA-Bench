"""Catalog integrity: the research-grounded 200-level suite is complete,
categorized, capability-tagged, and difficulty-monotone (the controlled
-ladder requirement from Procgen/SMACv2/SmartPlay). Pure/schema only —
the engine-run gate lives in test_robustness."""

from __future__ import annotations

import re
from collections import Counter

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR

CATEGORIES = [f"c{i}" for i in range(1, 13)]


def _catalog_files():
    return sorted(PACKS_DIR.glob("cat-*.yaml"))


def test_twelve_categories_and_about_200_levels():
    files = _catalog_files()
    cats = Counter(re.match(r"cat-(c\d+)-", f.name).group(1) for f in files)
    assert set(cats) == set(CATEGORIES), f"missing/extra categories: {sorted(cats)}"
    for c in CATEGORIES:
        assert cats[c] >= 5, f"category {c} has too few packs: {cats[c]}"
    total_levels = len(files) * 3
    assert 195 <= total_levels <= 215, f"expected ~200 levels, got {total_levels}"


def test_every_catalog_pack_tagged_and_meaningful():
    for f in _catalog_files():
        pk = load_pack(f)
        assert pk.meta.capability in {"perception", "reasoning", "action"}
        # Research rule: every scenario must carry a genuine
        # real-world / robotics meaning tying it to a capability.
        assert len(pk.meta.real_world_meaning) >= 20
        assert len(pk.meta.robotics_analogue) >= 10
        assert set(pk.levels) == {"easy", "medium", "hard"}


def _deadline(level) -> int | None:
    """Pull the within_ticks bound from an all_of win condition."""
    node = dict(level.win_condition.__pydantic_extra__ or {})
    for clause in node.get("all_of", []):
        if "within_ticks" in clause:
            return int(clause["within_ticks"])
    return None


def test_difficulty_is_monotone():
    """easy→medium→hard must get *harder*: the time budget must not
    increase (decision-hardness ladder, not raw scaling)."""
    for f in _catalog_files():
        pk = load_pack(f)
        ds = [_deadline(pk.levels[l]) for l in ("easy", "medium", "hard")]
        if all(d is not None for d in ds):
            assert ds[0] >= ds[1] >= ds[2], f"{f.name}: non-monotone clock {ds}"


def test_capability_coverage_spans_all_three_links():
    caps = {load_pack(f).meta.capability for f in _catalog_files()}
    assert caps == {"perception", "reasoning", "action"}, caps
