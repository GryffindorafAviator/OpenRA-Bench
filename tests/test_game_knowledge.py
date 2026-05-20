"""Deterministic scenario-scoped game knowledge + explicit objective.

The benchmark must judge reasoning, not Red Alert trivia or
goal-divination: every model gets the same glossary (only the codes in
THIS scenario), the fixed game model, and a plain-language translation
of the exact machine win/fail condition.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.game_knowledge import (
    ACTOR_GLOSSARY,
    objective_brief,
    scenario_primer,
)
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import LEAF_KEYS, WinCondition

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def _active():
    for f in sorted(glob.glob(str(PACKS / "*.yaml"))):
        b = os.path.basename(f)
        if b.startswith(("_", "TEMPLATE")):
            continue
        p = load_pack(f)
        if p.meta.status == "active":
            yield p


def test_primer_is_scenario_scoped_and_covers_present_codes():
    c = compile_level(load_pack(PACKS / "strategy-dilemma.yaml"), "easy")
    prim = scenario_primer(c)
    # codes actually in the scenario are explained...
    assert "fact = construction yard" in prim
    assert "proc = ore refinery" in prim or "proc =" in prim
    assert "Stances: 0=HoldFire" in prim
    # ...and irrelevant ones are NOT dumped in (scoped, not bloated)
    assert "mcv =" not in prim  # no MCV in strategy-dilemma
    # tech note only when production is allowed
    bom = compile_level(
        load_pack(PACKS / "strict-production-bom.yaml"), "easy"
    )
    assert "TECH TREE" in scenario_primer(bom)  # has build/place_building
    assert "TECH TREE" not in scenario_primer(c)  # move/attack only


def test_every_active_pack_code_has_a_glossary_entry():
    missing = set()
    for p in _active():
        c = compile_level(p, "easy")
        prim = scenario_primer(c)
        for line in prim.splitlines():
            if line.strip().endswith("= unknown actor"):
                missing.add((p.meta.id, line.strip()))
    assert not missing, f"actor codes with no glossary entry: {missing}"


def test_objective_brief_translates_win_and_fail_to_plain_language():
    c = compile_level(load_pack(PACKS / "strategy-dilemma.yaml"), "hard")
    ob = objective_brief(
        c.scenario.description, c.win_condition, c.fail_condition,
        c.max_turns,
    )
    assert "WIN WHEN:" in ob and "YOU LOSE IF:" in ob
    assert "destroy the enemy fact+proc" in ob
    # Hard tier: redesigned to make the timeout/attrition fail conditions
    # actually reachable within max_turns (the old within_ticks: 9000 was
    # unreachable in 80 turns → inert deadline, stallers drew rather than
    # losing). New budget: 2700 ticks, attrition cap 5.
    assert "before game tick 2700" in ob
    assert "lose ≤5 of your own units" in ob
    assert "decision turns" in ob  # turn budget surfaced


def test_objective_brief_handles_composites_and_regions():
    wc = WinCondition(**{"any_of": [
        {"reach_region": {"x": 10, "y": 20, "radius": 4}},
        {"all_of": [{"units_killed_gte": 3}, {"within_ticks": 5000}]},
    ]})
    txt = objective_brief("scout it", wc, None, 30)
    assert "scout it" in txt
    assert "OR" in txt and "AND" in txt
    assert "region (10,20)" in txt and "destroy ≥3 enemy units" in txt
    assert "YOU LOSE IF" not in txt  # no fail condition given


def test_all_predicate_keys_have_a_translation():
    # Guard: no leaf predicate silently renders as "key=value".
    from openra_bench.game_knowledge import _leaf_phrase

    sample = {
        "reach_region": {"x": 1, "y": 2, "radius": 3},
        "all_units_in_region": {"x": 1, "y": 2, "radius": 3},
        "building_count_gte": {"type": "powr", "n": 2},
        "building_in_region": {"x": 1, "y": 2, "count": 1},
        "unit_type_count_eq": {"type": "e1", "n": 3},
        "unit_type_count_gte": {"type": "e3", "n": 2},
        "enemy_key_buildings_destroyed": {"types": ["fact", "proc"]},
        "enemy_key_buildings_destroyed_in_region": {
            "x": 24, "y": 21, "radius": 9, "types": ["fact", "proc"],
        },
    }
    for k in LEAF_KEYS:
        v = sample.get(k, 5)
        phrase = _leaf_phrase(k, v)
        assert phrase and not phrase.startswith(f"{k}="), (
            f"predicate {k!r} has no plain-language translation"
        )


def test_model_agent_system_is_vendored_v2_with_objective_and_codex():
    """System prompt = vendored training system_v2 (objective lives
    HERE, not per-turn) + the scenario unit codex."""
    from openra_bench.agent import ModelAgent
    from openra_bench.prompt_v2 import unit_codex
    from openra_bench.providers import ProviderConfig

    a = ModelAgent(
        ProviderConfig(vision=False),
        allowed_tools=["observe"],
        objective="WIN WHEN: destroy 3 enemy units.",
        provider=type("P", (), {"complete": lambda *a, **k: None})(),
        unit_codex=unit_codex({"e1", "2tnk"}),
    )
    sys = a.history[0]
    assert a.history[0]["role"] == "system"
    sys = sys["content"]
    # vendored system_v2 header + combat model
    assert sys.startswith("You are playing Command & Conquer: Red Alert.")
    assert "Combat model" in sys
    # objective substituted into the system prompt; no leftover token
    assert "OBJECTIVE (this scenario): WIN WHEN: destroy 3 enemy units." in sys
    assert "{objective}" not in sys
    # scenario unit codex with numbers appended
    assert "UNIT CODEX" in sys and "e1" in sys and "hp" in sys


def test_objective_coords_relative_hides_numbers():
    """objective_coords:relative discloses the authored compass label
    only — no (x,y) leaks into the briefing the model sees."""
    import re
    from openra_bench.game_knowledge import objective_brief
    wc = {"all_of": [
        {"units_in_region_gte": {"x": 115, "y": 6, "radius": 7, "n": 2,
                                 "label": "the NORTH-EAST corner"}},
        {"within_ticks": 2800},
    ]}
    exact = objective_brief("", wc, None, 36, "exact")
    rel = objective_brief("", wc, None, 36, "relative")
    assert "(115,6)" in exact
    assert "the NORTH-EAST corner" in rel
    assert "115" not in rel and "(115" not in rel
    # win predicate still drives off the real coords regardless.
    from openra_bench.scenarios.win_conditions import WinCondition
    WinCondition(**wc)  # validates with the extra label key present


def test_objective_coords_default_is_exact():
    from openra_bench.game_knowledge import objective_brief
    wc = {"reach_region": {"x": 9, "y": 9, "radius": 3, "label": "the dot"}}
    assert "(9,9)" in objective_brief("", wc, None, 5)  # default exact
