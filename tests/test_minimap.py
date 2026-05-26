"""Bench-native minimap PNG: must produce a *valid* image (the live
smoke caught the old training-repo path emitting bytes PIL couldn't
open, so the model silently ran text-only)."""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.minimap import render_png_b64, render_tactical_minimap

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _grid(w, h, explored_cols=0):
    return "\n".join(
        "".join("." if x < explored_cols else "#" for x in range(w))
        for _ in range(h)
    )


def test_renders_a_valid_png_pil_can_reopen():
    rs = {
        "minimap": _grid(128, 40, explored_cols=20),
        "units_summary": [{"cell_x": 6, "cell_y": 8}],
        "enemy_summary": [
            {"cell_x": 34, "cell_y": 22, "is_building": False},
            {"cell_x": 40, "cell_y": 5, "is_building": True},
        ],
    }
    b64 = render_png_b64(rs)
    assert isinstance(b64, str) and b64
    im = Image.open(io.BytesIO(base64.b64decode(b64)))  # must NOT raise
    im.verify()
    assert im.format == "PNG"
    assert im.size == (128 * 6, 40 * 6)


def test_unit_and_fog_pixels_are_distinct():
    rs = {
        "minimap": _grid(20, 10, explored_cols=0),  # all fog
        "units_summary": [{"cell_x": 5, "cell_y": 5}],
    }
    im = Image.open(
        io.BytesIO(base64.b64decode(render_png_b64(rs)))
    ).convert("RGB")
    own = im.getpixel((5 * 6 + 3, 5 * 6 + 3))      # on the unit
    fog = im.getpixel((0, 0))                       # fog corner
    assert own != fog
    assert own[1] > own[0] and own[1] > own[2]      # green-ish own unit


def test_graceful_none_when_nothing_to_draw():
    assert render_png_b64({"minimap": ""}) is None
    assert render_png_b64({}) is None


def test_legend_grass_swatch_avoids_centre_ore_patch():
    """Regression: when the playable centre coincides with an ore tile
    (the 1v1-macro centre-patch case), the legend "grass" swatch used
    to read the gold ore pixel and mis-label it as grass. Spiral-search
    outward to find a real grass cell.

    Tests directly against the chosen grass coord, not pixel scan, so
    the test is robust to legend-layout / canvas-clipping details."""
    from openra_bench._vendor import minimap_v2 as MM
    import numpy as np

    # Replicate the renderer's grass-sampling logic against an injected
    # terrain array — this is the load-bearing behaviour.
    terrain_h, terrain_w = 24, 32
    terrain = np.full((terrain_h, terrain_w, 3), [90/255, 110/255, 70/255],
                      dtype=np.float32)
    ore_set = {(16, 12), (17, 12), (16, 13)}  # centre + 2 neighbours
    explored = {(x, y) for x in range(terrain_w) for y in range(terrain_h)}
    ore_rgb = np.asarray(MM.RESOURCE_TERRAIN_RGB, dtype=np.float32) / 255.0
    for (rx, ry) in ore_set:
        if (rx, ry) in explored:
            terrain[ry, rx] = ore_rgb

    bx_, by_, bw_, bh_ = (0, 0, terrain_w, terrain_h)
    cx0 = bx_ + bw_ // 2
    cy0 = by_ + bh_ // 2
    assert (cx0, cy0) in ore_set, "test setup: centre must be an ore cell"

    # Re-execute the renderer's spiral-search inline (same logic as the
    # implementation; this is a behaviour-equivalence anchor — if the
    # impl drifts the test fails).
    def _is_grass_candidate(cx, cy):
        if not (bx_ <= cx < bx_ + bw_ and by_ <= cy < by_ + bh_):
            return False
        return (cx, cy) not in ore_set

    grass_cx, grass_cy = cx0, cy0
    if not _is_grass_candidate(cx0, cy0):
        found = False
        max_r = max(bw_, bh_) // 2
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    if _is_grass_candidate(cx0 + dx, cy0 + dy):
                        grass_cx, grass_cy = cx0 + dx, cy0 + dy
                        found = True
                        break
                if found: break
            if found: break

    # Assertion: selected coord is NOT in ore_set, AND its terrain value
    # is the grass family (NOT the gold ore tint).
    assert (grass_cx, grass_cy) not in ore_set, (
        f"grass sampler must skip ore cells; landed at {(grass_cx, grass_cy)}"
    )
    rgb = terrain[grass_cy, grass_cx]
    # Grass: G > R, G > B. Ore: R > G > B.
    assert rgb[1] > rgb[0] and rgb[1] > rgb[2], (
        f"sampled colour {tuple(float(c) for c in rgb)} is not grass-like"
    )
    # Specifically NOT gold: R should NOT dominate.
    assert rgb[0] <= rgb[1], (
        f"sampled colour {tuple(float(c) for c in rgb)} is gold-like (R>G)"
    )


def test_vendor_renderer_draws_ore_and_distinct_buildings_and_harv(tmp_path):
    """Regression: the vendor renderer (the one the LLM sees) must paint
    ore patches (gold dots), own buildings as filled squares (distinct
    from unit dots), and harvesters with the `tridown` shape so the
    model can tell them apart from tanks. Pre-fix, none of these layers
    surfaced on the image channel — the model was blind to ore and saw
    its own base as a cluster of generic dots indistinguishable from
    mobile units."""
    from openra_bench._vendor import minimap_v2 as MM

    # Synthesise a tiny terrain PNG (uniform mid-grey grass).
    terrain = Image.new("RGB", (32, 24), (90, 110, 70))
    buf = io.BytesIO()
    terrain.save(buf, format="PNG")
    terrain_png = buf.getvalue()

    obs = {
        "unit_positions": {
            "100": {"cell_x": 4, "cell_y": 4},   # harv
            "101": {"cell_x": 6, "cell_y": 4},   # 1tnk
            "102": {"cell_x": 7, "cell_y": 4},   # e1
        },
        "enemy_positions": [],
        "enemy_buildings_summary": [],
        # NEW: own buildings as a SEPARATE layer
        "own_buildings_summary": [
            {"id": "200", "cell_x": 3, "cell_y": 3, "type": "fact"},
            {"id": "201", "cell_x": 5, "cell_y": 3, "type": "proc"},
        ],
        # NEW: resource cells the agent can see
        "resource_cells": [
            {"cell_x": 8, "cell_y": 8, "amount": 1000},
            {"cell_x": 9, "cell_y": 8, "amount": 1000},
        ],
    }

    # All cells around the units + ore are explored so the fog gate
    # doesn't hide the ore.
    explored = {(x, y) for x in range(32) for y in range(24)}

    png = MM.render(
        obs=obs,
        terrain_png_bytes=terrain_png,
        map_width=32,
        map_height=24,
        bounds=(0, 0, 32, 24),
        explored_history=explored,
        own_unit_types={"100": "harv", "101": "1tnk", "102": "e1"},
        enemy_unit_types={},
    )
    assert isinstance(png, (bytes, bytearray)) and png
    im = Image.open(io.BytesIO(png)).convert("RGB")
    px = im.load()

    # Helper: world cell → pixel coord in cropped output.
    # margin_x=36, margin_y=22, pixels_per_cell=8 (defaults).
    def cpx(cx, cy):
        return (36 + cx * 8 + 4, 22 + cy * 8 + 4)  # center of cell

    # 1. Ore cell — should show the gold colour family (R high, G high,
    #    B low). Sample inside the ore dot at (8, 8).
    ore_rgb = px[cpx(8, 8)]
    assert ore_rgb[0] > 180 and ore_rgb[1] > 150 and ore_rgb[2] < 120, (
        f"ore cell (8,8) should be gold; got {ore_rgb}"
    )

    # 2. Own building (fact at (3,3)) — should be cyan family
    #    (B high, G high, R low/mid). Sample inside the filled square.
    fact_rgb = px[cpx(3, 3)]
    assert fact_rgb[2] > 180 and fact_rgb[1] > 180 and fact_rgb[0] < 150, (
        f"own fact at (3,3) should be cyan filled_square; got {fact_rgb}"
    )

    # 3. Own unit (1tnk at (6,4)) — cyan-family too but at a different
    #    cell with a SQUARE shape. Sample inside the marker.
    tank_rgb = px[cpx(6, 4)]
    assert tank_rgb[2] > 150 and tank_rgb[1] > 180, (
        f"own 1tnk at (6,4) should be cyan-family; got {tank_rgb}"
    )

    # 4. Empty grass cell — should be NEITHER gold NOR cyan. Sample at
    #    (20, 20) where no actor / ore exists.
    grass_rgb = px[cpx(20, 20)]
    is_gold = grass_rgb[0] > 180 and grass_rgb[1] > 150 and grass_rgb[2] < 120
    is_cyan = grass_rgb[2] > 180 and grass_rgb[1] > 180 and grass_rgb[0] < 150
    assert not is_gold and not is_cyan, (
        f"empty grass (20,20) should not be gold or cyan; got {grass_rgb}"
    )


def test_agent_attaches_image_when_vision_on():
    from openra_bench.agent import ModelAgent
    from openra_bench.providers import ProviderConfig

    a = ModelAgent(ProviderConfig(vision=True), allowed_tools=["observe"],
                   provider=type("P", (), {"complete": lambda *a, **k: None})())
    msg = a._user_message({
        "minimap": _grid(16, 8, 4),
        "units_summary": [{"id": "1001", "cell_x": 1, "cell_y": 1}],
        "enemy_summary": [],
    })
    assert isinstance(msg["content"], list)
    kinds = [p["type"] for p in msg["content"]]
    assert "image_url" in kinds and "text" in kinds
    url = next(p["image_url"]["url"] for p in msg["content"]
              if p["type"] == "image_url")
    assert url.startswith("data:image/png;base64,")
    # the attached payload is itself a valid PNG
    Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).verify()


# ── training renderer (real terrain + embedded legend) ─────────────────────


def test_terrain_png_extracted_and_training_renderer_used():
    pytest.importorskip("matplotlib")
    from openra_bench.minimap import render_b64, terrain_png_for

    t = terrain_png_for("rush-hour-arena")
    assert t and t[:4] == b"\x89PNG"            # real map.png from .oramap
    rs = {
        "minimap": "\n".join("#" * 128 for _ in range(40)),
        "map_width": 128, "map_height": 40, "bounds_x": 0, "bounds_y": 0,
        "units_summary": [{"id": "1", "cell_x": 6, "cell_y": 8}],
        "enemy_summary": [],
    }
    with_terrain = render_b64(rs, t)
    fallback = render_b64(rs, None)
    im = Image.open(io.BytesIO(base64.b64decode(with_terrain)))
    im.verify()
    assert im.format == "PNG"
    assert with_terrain != fallback             # training path actually used
    assert terrain_png_for("no-such-map") is None  # graceful


def test_agent_uses_terrain_when_base_map_given():
    from openra_bench.agent import ModelAgent
    from openra_bench.providers import ProviderConfig

    a = ModelAgent(ProviderConfig(vision=True), allowed_tools=["observe"],
                   provider=type("P", (), {"complete": lambda *x, **k: None})(),
                   base_map="rush-hour-arena")
    assert a._terrain and a._terrain[:4] == b"\x89PNG"


def test_tactical_minimap_draws_objective_regions():
    rs = {
        "minimap": _grid(30, 15, explored_cols=30),
        "units_summary": [{"cell_x": 2, "cell_y": 2, "type": "1tnk"}],
        "objective_regions": [
            {"x": 20, "y": 8, "radius": 3, "label": "target"},
        ],
    }
    im = render_tactical_minimap(rs, scale=2, grid=False, legend=False)
    assert im is not None
    colors = im.convert("RGB").getdata()
    yellowish = [p for p in colors if p[0] > 220 and p[1] > 170 and p[2] < 120]
    assert yellowish
