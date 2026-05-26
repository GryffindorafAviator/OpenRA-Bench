"""prompt_v2: training-faithful system/briefing/minimap.

Key contract: the briefing carries NO ASCII minimap (the PNG is the
spatial channel; training strips the grid). System prompt holds the
objective + codex; briefing is the vendored v2 one-unit-per-line form.
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench import prompt_v2 as P


def _render_state():
    return {
        "game_tick": 540,
        "cash": 5000, "resources": 0, "harvesters": 0,
        "power_provided": 0, "power_drained": 0,
        "explored_percent": 15.1,
        "minimap": "#" * 128 + "\n" + "." * 128,  # MUST NOT leak in
        "units_summary": [
            {"id": "1001", "type": "2tnk", "cell_x": 6, "cell_y": 8,
             "idle": False, "can_attack": True, "activity": "moving",
             "target_x": 30, "target_y": 8},
            {"id": "1002", "type": "jeep", "cell_x": 6, "cell_y": 11,
             "idle": True, "can_attack": True, "activity": "idle"},
        ],
        "enemy_summary": [
            {"id": "9001", "type": "e1", "cell_x": 34, "cell_y": 22,
             "is_building": False},
        ],
        "enemy_buildings_summary": [],
        "own_buildings": [],
        "production": [],
        "map_width": 128, "map_height": 40, "bounds": (0, 0, 128, 40),
        "_raw": {"unit_positions": {"1001": {"cell_x": 6, "cell_y": 8}},
                 "enemy_positions": [], "explored_cells": [[6, 8]]},
    }


def test_briefing_has_no_ascii_minimap():
    b = P.briefing(_render_state())
    # no long fog/explored runs from the ascii grid
    assert "########" not in b and "........" not in b
    assert "#" * 40 not in b
    # but it IS the v2 briefing with the real content
    assert b.startswith("--- TURN BRIEFING (tick 540")
    assert "1001 2tnk" in b and "moving to (30,8)" in b
    assert "Idle: 1002" in b
    assert "Funds: $5000" in b


def test_state_from_render_strips_minimap():
    st = P.state_from_render(_render_state())
    assert st["minimap"] == ""               # explicit: no ascii grid
    assert st["tick"] == 540
    assert st["units_summary"][0]["target_x"] == 30


def test_system_prompt_has_objective_and_codex_no_placeholder():
    s = P.system_prompt("WIN WHEN: destroy 3 enemy units.",
                        P.unit_codex({"e1", "2tnk", "jeep"}))
    assert s.startswith("You are playing Command & Conquer: Red Alert.")
    assert "OBJECTIVE (this scenario): WIN WHEN: destroy 3 enemy units." in s
    assert "{objective}" not in s
    assert "UNIT CODEX" in s and "2tnk" in s and "hp" in s


def test_system_prompt_default_appends_full_codex_and_tech_tree():
    """Paper baseline: every model call (no scenario-scoped override)
    sees the FULL RA codex + tech tree — equal information for every
    model, equivalent to a human reading the RA manual."""
    s = P.system_prompt("WIN WHEN: build 2 destroyers.", "")
    # Full codex header + all major categories present
    assert "UNIT CODEX (full RA reference" in s
    for section in ("Infantry:", "Vehicles:", "Aircraft:", "Ships:",
                    "Structures:", "Defenses:"):
        assert section in s, f"missing section {section!r} in default codex"
    # Naval units must be visible — the load-bearing fix vs the old
    # scenario-scoped filter (a syrd pack used to hide ship stats).
    for ship in ("dd ", "ca ", "ss ", "pt ", "lst "):
        assert ship in s, f"ship code {ship!r} not in default codex"
    # Tech tree section + at least one prereq line
    assert "TECH TREE" in s
    assert "Ships (built at syrd / spen)" in s
    # Faction + prereq present for a Ship row (dd is Allied + needs syrd)
    assert "Allied" in s and "syrd" in s


def test_full_codex_text_constants_populated_from_vendor():
    """The vendor-derived constants must be non-empty whenever the
    in-repo vendor YAML is reachable (which it is in the bench
    checkout — that's how the engine wheel is built)."""
    assert P.FULL_CODEX_TEXT.startswith("UNIT CODEX (full RA reference")
    assert P.TECH_TREE_TEXT.startswith("TECH TREE")
    # Spot-check one row: dd has cost 1000 and hp400 in vendor YAML.
    dd_row = [l for l in P.FULL_CODEX_TEXT.splitlines() if l.startswith("  dd ")][0]
    assert "$1000" in dd_row and "hp400" in dd_row and "naval" in dd_row


def test_default_codex_size_reasonable():
    """Sanity: the default system prompt must fit comfortably under a
    25 KB budget (well within any 128 K-context LLM)."""
    s = P.system_prompt("any objective", "")
    assert len(s) < 25_000, (
        f"system prompt grew too large: {len(s)} bytes"
    )


def test_minimap_b64_is_valid_png_with_terrain():
    pytest.importorskip("PIL")
    from PIL import Image

    from openra_bench.minimap import terrain_png_for

    t = terrain_png_for("rush-hour-arena")
    if not t:
        pytest.skip("rush-hour-arena.oramap not resolvable")
    b64 = P.minimap_b64(_render_state(), t, set())
    assert b64
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    im.verify()
    assert im.format == "PNG"
    # None without terrain (graceful text-only) and without _raw
    assert P.minimap_b64(_render_state(), None, set()) is None
    assert P.minimap_b64({"_raw": {}}, t, set()) is None


# ── structured-fog text mode + codex descriptions + premium routing ────────


def test_structured_fog_regions_and_text():
    from openra_bench.structured_fog import (
        compute_unexplored_regions,
        format_structured_fog,
    )
    bounds = (0, 0, 10, 10)
    # explored a 3x3 NW block → one big unexplored component
    explored = [(x, y) for x in range(3) for y in range(3)]
    regs = compute_unexplored_regions(explored, bounds)
    assert regs and regs[0]["cells"] == 100 - 9
    txt = format_structured_fog({"explored_cells": explored}, bounds)
    assert txt.startswith("Unexplored regions (largest first")
    assert "x ∈ [" in txt and "cells)" in txt
    # nothing explored → whole map
    assert "entire playable map" in format_structured_fog(
        {"explored_cells": []}, bounds
    )


def _rs_with_raw():
    return {
        "game_tick": 100, "cash": 0, "resources": 0, "harvesters": 0,
        "power_provided": 0, "power_drained": 0, "explored_percent": 5.0,
        "units_summary": [{"id": "1", "type": "2tnk", "cell_x": 6,
                           "cell_y": 8, "idle": True, "can_attack": True}],
        "enemy_summary": [], "enemy_buildings_summary": [],
        "own_buildings": [], "production": [],
        "map_width": 20, "map_height": 12, "bounds": (0, 0, 20, 12),
        "_raw": {"unit_positions": {"1": {"cell_x": 6, "cell_y": 8}},
                 "enemy_positions": [], "explored_cells": [[6, 8], [6, 9]]},
    }


def test_structured_fog_mode_sends_text_no_image():
    from openra_bench.agent import ModelAgent
    from openra_bench.providers import ProviderConfig

    a = ModelAgent(
        ProviderConfig(vision=True, fog_mode="structured"),
        allowed_tools=["observe"],
        provider=type("P", (), {"complete": lambda *x, **k: None})(),
        base_map="rush-hour-arena", level="easy",
    )
    msg = a._user_message(_rs_with_raw())
    assert isinstance(msg["content"], str)              # NO image
    assert "Unexplored regions" in msg["content"]
    assert "TURN BRIEFING" in msg["content"]


def test_minimap_constant_vs_per_type_differ():
    from openra_bench.minimap import terrain_png_for
    from openra_bench.prompt_v2 import minimap_b64

    t = terrain_png_for("rush-hour-arena")
    if not t:
        import pytest as _p
        _p.skip("oramap absent")
    rs = _rs_with_raw()
    rs["units_summary"] = [
        {"id": "1", "type": "2tnk", "cell_x": 6, "cell_y": 8},
        {"id": "2", "type": "jeep", "cell_x": 7, "cell_y": 8},
    ]
    rs["_raw"]["unit_positions"] = {
        "1": {"cell_x": 6, "cell_y": 8}, "2": {"cell_x": 7, "cell_y": 8}}
    per = minimap_b64(rs, t, set(), constant_colors=False)
    const = minimap_b64(rs, t, set(), constant_colors=True)
    assert per and const and per != const


def test_codex_leads_with_one_sentence_description():
    from openra_bench.prompt_v2 import unit_codex

    cx = unit_codex({"1tnk", "2tnk", "tsla"})
    assert "Medium tank" in cx and "Light tank" in cx
    assert "Tesla coil" in cx
    # description then stats on the same line
    line = [l for l in cx.splitlines() if l.strip().startswith("2tnk")][0]
    assert "tank" in line and "$850" in line and "hp400" in line


def test_extra_body_merged_into_request(monkeypatch):
    from openra_bench.providers import OpenAICompatibleProvider, ProviderConfig

    cfg = ProviderConfig(provider="openrouter",
                         extra_body={"provider": {"sort": "throughput"}})
    p = OpenAICompatibleProvider(cfg)
    captured = {}

    class _R:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "",
                     "tool_calls": []}}], "usage": {}}

    def fake_post(url, headers=None, json=None):
        captured.update(json or {})
        return _R()

    monkeypatch.setattr(p._client, "post", fake_post)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    p.complete([{"role": "user", "content": "hi"}], [])
    assert captured.get("provider") == {"sort": "throughput"}


def test_briefing_sanitizes_empty_buildings_line():
    """Unit-only scenarios must not show the vendored fallback's
    confusing 'Buildings: ? ()'."""
    b = P.briefing(_render_state())
    assert "Buildings: ? ()" not in b
    assert "Buildings: 0 ()" not in b
    line = [l for l in b.splitlines() if l.startswith("Buildings:")][0]
    assert "none" in line and "mobile units only" in line
