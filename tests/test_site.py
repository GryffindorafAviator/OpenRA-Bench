"""Tests for the static mission-player site pipeline (site/generate.py).

Validates: data generation, scenario coverage invariants, bilingual
instruction coverage, schema validity, and the key coverage invariant
from the command file.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))

from generate import (
    _annotator_hints_en,
    _annotator_hints_zh,
    _translate_objective_zh,
    build_scenario,
    generate,
)


class TestBilingualGeneration:
    """Deterministic bilingual description generation."""

    def test_translate_win_when(self):
        zh = _translate_objective_zh("WIN WHEN: explored at least 50% of the map")
        assert "胜利条件" in zh
        assert "探索" in zh

    def test_translate_lose_if(self):
        zh = _translate_objective_zh("YOU LOSE IF: lost no more than 2 units")
        assert "失败条件" in zh

    def test_translate_turns(self):
        zh = _translate_objective_zh("You have at most 20 decision turns")
        assert "决策回合" in zh

    def test_empty_string(self):
        assert _translate_objective_zh("") == ""

    def test_annotator_hints_en_has_capability(self):
        from openra_bench.scenarios import discover_packs
        pack = next(p for p in discover_packs() if p.meta.status == "active")
        hints = _annotator_hints_en(pack)
        assert any("capability" in h for h in hints)

    def test_annotator_hints_zh_has_capability(self):
        from openra_bench.scenarios import discover_packs
        pack = next(p for p in discover_packs() if p.meta.status == "active")
        hints = _annotator_hints_zh(pack)
        assert any("能力" in h for h in hints)


class TestBuildScenario:
    """Single scenario model generation."""

    def test_schema_has_required_fields(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        pack = next(p for p in discover_packs() if p.meta.status == "active")
        s = build_scenario(pack, objective_brief)

        assert "scenarioId" in s
        assert "title" in s
        assert "capability" in s
        assert "map" in s
        assert "humanReadable" in s
        assert "levels" in s

    def test_has_bilingual_instructions(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        pack = next(p for p in discover_packs() if p.meta.status == "active")
        s = build_scenario(pack, objective_brief)

        hr = s["humanReadable"]
        assert len(hr["playerInstructions"]["en"]) > 0
        assert len(hr["playerInstructions"]["zh"]) > 0
        assert len(hr["annotatorHints"]["en"]) > 0
        assert len(hr["annotatorHints"]["zh"]) > 0

    def test_has_three_levels(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        pack = next(p for p in discover_packs() if p.meta.status == "active")
        s = build_scenario(pack, objective_brief)
        assert "easy" in s["levels"]
        assert "medium" in s["levels"]
        assert "hard" in s["levels"]

    def test_level_has_objective(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        pack = next(p for p in discover_packs() if p.meta.status == "active")
        s = build_scenario(pack, objective_brief)
        for lv in ("easy", "medium", "hard"):
            assert "objective" in s["levels"][lv]
            assert "en" in s["levels"][lv]["objective"]
            assert "zh" in s["levels"][lv]["objective"]


class TestGeneratePipeline:
    """Full pipeline dry-run."""

    def test_dry_run_returns_stats(self):
        stats = generate(dry_run=True)
        assert stats["total"] > 100
        assert stats["english_instructions"] == stats["total"]
        assert stats["chinese_instructions"] == stats["total"]


class TestCoverageInvariant:
    """The key coverage invariant from the command file:
    hf_scenarios_with_maps == scenarios_visible_in_ui
    == scenarios_with_clear_bilingual_instructions
    == scenarios_covered_by_tests
    """

    def test_all_active_packs_generate_successfully(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        packs = [p for p in discover_packs() if p.meta.status == "active"]
        generated = []
        for p in packs:
            s = build_scenario(p, objective_brief)
            generated.append(s)

        assert len(generated) == len(packs)

    def test_all_have_en_instructions(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        packs = [p for p in discover_packs() if p.meta.status == "active"]
        missing = []
        for p in packs:
            s = build_scenario(p, objective_brief)
            if not s["humanReadable"]["playerInstructions"]["en"]:
                missing.append(p.meta.id)
        assert missing == [], f"Missing EN instructions: {missing}"

    def test_all_have_zh_instructions(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        packs = [p for p in discover_packs() if p.meta.status == "active"]
        missing = []
        for p in packs:
            s = build_scenario(p, objective_brief)
            if not s["humanReadable"]["playerInstructions"]["zh"]:
                missing.append(p.meta.id)
        assert missing == [], f"Missing ZH instructions: {missing}"

    def test_no_stale_hardcoded_counts(self):
        """Scenario count must come from live data, not hardcoded."""
        from openra_bench.scenarios import discover_packs
        packs = [p for p in discover_packs() if p.meta.status == "active"]
        assert len(packs) >= 100, "Expected many active packs from live data"

    def test_generated_json_schema_valid(self):
        from openra_bench.game_knowledge import objective_brief
        from openra_bench.scenarios import discover_packs

        pack = next(p for p in discover_packs() if p.meta.status == "active")
        s = build_scenario(pack, objective_brief)

        required = ["scenarioId", "title", "capability", "map", "levels",
                     "humanReadable", "raw", "capabilityLabel"]
        for k in required:
            assert k in s, f"Missing top-level key: {k}"

        assert s["map"]["type"] == "image"
        assert "src" in s["map"]
        assert "summary" in s["humanReadable"]
        assert "playerInstructions" in s["humanReadable"]
        assert "annotatorHints" in s["humanReadable"]


class TestSiteIndexExists:
    """Verify the static site HTML exists."""

    def test_index_html_present(self):
        p = ROOT / "site" / "index.html"
        assert p.is_file(), "site/index.html missing"

    def test_index_html_has_mission_player(self):
        html = (ROOT / "site" / "index.html").read_text()
        assert "OpenRA-Bench Missions" in html
        assert "scenario-card" in html
        assert "annotation" in html.lower()
        assert "Chinese" in html or "zh" in html
