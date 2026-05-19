"""prompt_v2: training-faithful system/briefing/minimap.

Key contract: the briefing carries NO ASCII minimap (the PNG is the
spatial channel; training strips the grid). System prompt holds the
objective + codex; briefing is the vendored v2 one-unit-per-line form.
"""

from __future__ import annotations

import base64
import io

import pytest

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
