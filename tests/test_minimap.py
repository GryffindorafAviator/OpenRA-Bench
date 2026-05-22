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
