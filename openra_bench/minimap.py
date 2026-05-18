"""Self-contained minimap PNG (no training-repo / terrain_png dep).

The bench engine emits an ASCII fog grid (`#`=unknown, `.`=explored)
plus unit/building cell lists — enough to draw a real, legible colour
minimap the model can actually see. Pure; returns base64 PNG, or None
if Pillow is missing or there's nothing to draw (graceful text-only
fallback).
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)

CELL = 6  # px per map cell (≈768×240 for a 128×40 map — legible)

# (R,G,B)
_BG_UNKNOWN = (18, 18, 22)      # '#'  fog / never seen
_BG_EXPLORED = (70, 74, 82)     # '.'  revealed terrain
_OWN = (60, 200, 90)            # your units
_OWN_BLD = (60, 130, 230)       # your buildings
_ENEMY = (225, 60, 55)          # enemy units
_ENEMY_BLD = (240, 160, 40)     # enemy buildings


def _draw_cell(px, w: int, h: int, cx: int, cy: int, rgb, r: int = 1) -> None:
    for yy in range(max(0, cy - r), min(h, cy + r + 1)):
        for xx in range(max(0, cx - r), min(w, cx + r + 1)):
            for sy in range(CELL):
                for sx in range(CELL):
                    px[xx * CELL + sx, yy * CELL + sy] = rgb


def render_png_b64(render_state: dict) -> str | None:
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001 — vision optional
        return None

    ascii_mm = render_state.get("minimap") or ""
    rows = [r for r in ascii_mm.split("\n") if r]
    if not rows:
        return None
    h = len(rows)
    w = max(len(r) for r in rows)
    if w == 0 or w * h > 200_000:  # sanity cap
        return None

    img = Image.new("RGB", (w * CELL, h * CELL), _BG_UNKNOWN)
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != "#":  # explored / visible
                for sy in range(CELL):
                    for sx in range(CELL):
                        px[x * CELL + sx, y * CELL + sy] = _BG_EXPLORED

    def _plot(items, rgb_unit, rgb_bld):
        for it in items or []:
            if not isinstance(it, dict):
                continue
            cx, cy = int(it.get("cell_x", 0)), int(it.get("cell_y", 0))
            if 0 <= cx < w and 0 <= cy < h:
                is_b = bool(it.get("is_building"))
                _draw_cell(px, w, h, cx, cy, rgb_bld if is_b else rgb_unit,
                           r=1 if is_b else 0)

    _plot(render_state.get("units_summary"), _OWN, _OWN_BLD)
    _plot(render_state.get("own_buildings"), _OWN_BLD, _OWN_BLD)
    _plot(render_state.get("enemy_summary"), _ENEMY, _ENEMY_BLD)

    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:  # noqa: BLE001
        logger.debug("minimap encode failed: %s", e)
        return None
