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
import zipfile
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def terrain_png_for(base_map: str) -> bytes | None:
    """Raw `map.png` bytes from the resolved `.oramap` (a zip), cached.
    None when the map can't be resolved — caller falls back."""
    if not base_map:
        return None
    try:
        from .scenarios.loader import resolve_map_path

        p = resolve_map_path(base_map)
        if not p:
            return None
        with zipfile.ZipFile(p, "r") as zf:
            if "map.png" in zf.namelist():
                return zf.read("map.png")
    except Exception as e:  # noqa: BLE001
        logger.debug("terrain extract failed for %s: %s", base_map, e)
    return None


def render_b64(render_state: dict, terrain_png: bytes | None = None) -> str | None:
    """Preferred minimap: the training renderer (real terrain + an
    embedded legend the model can read) when terrain is available;
    otherwise the self-contained bench fallback. Either way a *valid*
    base64 PNG, or None for graceful text-only."""
    if terrain_png:
        try:
            from openra_rl_training.training.minimap_renderer import (
                render_minimap,
            )

            b64 = render_minimap(
                terrain_png=terrain_png,
                map_width=int(render_state.get("map_width", 64) or 64),
                map_height=int(render_state.get("map_height", 64) or 64),
                bounds_x=int(render_state.get("bounds_x", 0) or 0),
                bounds_y=int(render_state.get("bounds_y", 0) or 0),
                own_units=render_state.get("units_summary", []) or [],
                enemy_units=render_state.get("enemy_summary", []) or [],
                ascii_minimap=render_state.get("minimap", "") or "",
            )
            if b64:
                return b64
        except Exception as e:  # noqa: BLE001 — fall back below
            logger.debug("training minimap failed, using fallback: %s", e)
    return render_png_b64(render_state)

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


# ── Tactical minimap: per-type shapes + overlap counts + legend ───────
# A richer renderer than render_png_b64 — distinguishes unit TYPES by
# shape, shows a COUNT when units stack on one cell (otherwise they
# render as a single dot), draws a coordinate GRID, and an embedded
# LEGEND. Used by the human Play tab; reusable for the model's view.

_INFANTRY_TYPES = {"e1", "e2", "e3", "e4", "e6", "e7", "medi", "mech",
                   "spy", "thf", "dog", "engineer"}


def _unit_category(actor_type: str, is_building: bool) -> str:
    """Coarse class for shape selection: infantry / harvester /
    building / vehicle."""
    t = (actor_type or "").strip().lower()
    if is_building:
        return "building"
    if t.startswith("harv"):
        return "harvester"
    if t in _INFANTRY_TYPES or (
        len(t) == 2 and t[0] == "e" and t[1].isdigit()
    ):
        return "infantry"
    return "vehicle"


def _minimap_font(size: int):
    """A legible TrueType font (falls back to a scaled default)."""
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        return ImageFont.load_default()


def _draw_unit_shape(draw, cx, cy, cp, category, color):
    """Draw `category`'s shape, filling ~70% of the cp-pixel cell at
    grid cell (cx, cy)."""
    m = cp * 0.16
    x0, y0 = cx * cp + m, cy * cp + m
    x1, y1 = (cx + 1) * cp - m, (cy + 1) * cp - m
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    outline = (15, 15, 18)
    if category == "infantry":
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=outline)
    elif category == "harvester":
        draw.polygon(
            [(mx, y0), (x0, y1), (x1, y1)], fill=color, outline=outline
        )
    elif category == "building":
        draw.polygon(
            [(mx, y0), (x1, my), (mx, y1), (x0, my)],
            fill=color, outline=outline,
        )
    else:  # vehicle
        draw.rectangle([x0, y0, x1, y1], fill=color, outline=outline)


def _draw_move_arrow(draw, fx, fy, tx, ty, cp, color):
    """Arrow from cell (fx,fy) centre to cell (tx,ty) centre — a unit's
    move/attack destination link."""
    import math

    x0, y0 = (fx + 0.5) * cp, (fy + 0.5) * cp
    x1, y1 = (tx + 0.5) * cp, (ty + 0.5) * cp
    width = max(2, cp // 8)
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    # Arrowhead at the destination.
    ang = math.atan2(y1 - y0, x1 - x0)
    head = cp * 0.7
    spread = math.radians(26)
    p1 = (x1 - head * math.cos(ang - spread),
          y1 - head * math.sin(ang - spread))
    p2 = (x1 - head * math.cos(ang + spread),
          y1 - head * math.sin(ang + spread))
    draw.polygon([(x1, y1), p1, p2], fill=color)


def render_tactical_minimap(
    render_state: dict,
    scale: int = 4,
    grid: bool = True,
    legend: bool = True,
    selected=None,
    arrows=None,
):
    """A legible tactical minimap as a PIL RGB image:

    * per-type SHAPES — ● infantry, ■ vehicle, ▲ harvester, ◆ building;
    * COUNT badge when >1 unit stacks on a cell (so overlapping units
      are not silently rendered as one dot);
    * colour by side — green = you, red = enemy;
    * a coordinate GRID with axis labels every 10 cells, and a LEGEND
      strip beneath the map.

    `scale` multiplies the 6px base cell. Returns None if Pillow is
    missing or there is nothing to draw."""
    try:
        from PIL import Image, ImageDraw
    except Exception:  # noqa: BLE001
        return None
    rows = [r for r in (render_state.get("minimap") or "").split("\n")
            if r]
    if not rows:
        return None
    h = len(rows)
    w = max(len(r) for r in rows)
    if w == 0 or w * h > 200_000:
        return None
    cp = max(1, CELL * scale)
    legend_h = cp * 2 if legend else 0
    img = Image.new("RGB", (w * cp, h * cp + legend_h), _BG_UNKNOWN)
    draw = ImageDraw.Draw(img)

    # Explored terrain.
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch != "#":
                draw.rectangle(
                    [x * cp, y * cp, (x + 1) * cp - 1, (y + 1) * cp - 1],
                    fill=_BG_EXPLORED,
                )

    # Collect every actor by cell so stacked units can be counted.
    by_cell: dict = {}

    def _collect(items, side, force_building):
        for it in items or []:
            if not isinstance(it, dict):
                continue
            cx = int(it.get("cell_x", -99))
            cy = int(it.get("cell_y", -99))
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            is_b = force_building or bool(it.get("is_building"))
            cat = _unit_category(
                it.get("actor_type") or it.get("type") or "", is_b
            )
            by_cell.setdefault((cx, cy), []).append((side, cat))

    _collect(render_state.get("units_summary"), "own", False)
    _collect(render_state.get("own_buildings"), "own", True)
    _collect(render_state.get("enemy_summary"), "enemy", False)
    _collect(
        render_state.get("enemy_buildings_summary")
        or render_state.get("enemy_buildings"),
        "enemy", True,
    )

    def _color(side, cat):
        if side == "own":
            return _OWN_BLD if cat == "building" else _OWN
        return _ENEMY_BLD if cat == "building" else _ENEMY

    badge_font = _minimap_font(max(9, int(cp * 0.62)))
    for (cx, cy), occ in by_cell.items():
        # Dominant occupant decides the shape; prefer a building.
        side, cat = next(
            (o for o in occ if o[1] == "building"), occ[0]
        )
        _draw_unit_shape(draw, cx, cy, cp, cat, _color(side, cat))
        if len(occ) > 1:
            tx, ty = (cx + 1) * cp - cp * 0.42, cy * cp + 1
            draw.text(
                (tx, ty), str(len(occ)), fill=(255, 255, 255),
                font=badge_font, stroke_width=max(2, cp // 12),
                stroke_fill=(0, 0, 0),
            )

    # Movement arrows — drawn under the selection boxes so a selected
    # unit's boundary stays on top. queued = yellow, en-route = cyan.
    for ar in (arrows or []):
        try:
            fx, fy, tx, ty, kind = ar
        except (ValueError, TypeError):
            continue
        col = (255, 230, 90) if kind == "queued" else (90, 220, 245)
        _draw_move_arrow(draw, fx, fy, tx, ty, cp, col)

    # Selection boundary — a bright white box around each selected
    # unit's cell.
    if selected:
        sel_ids = {str(s) for s in selected}
        for u in render_state.get("units_summary", []) or []:
            if not isinstance(u, dict):
                continue
            if str(u.get("id", "")) not in sel_ids:
                continue
            cx = int(u.get("cell_x", -99))
            cy = int(u.get("cell_y", -99))
            if 0 <= cx < w and 0 <= cy < h:
                inset = max(1, cp // 14)
                draw.rectangle(
                    [cx * cp + inset, cy * cp + inset,
                     (cx + 1) * cp - inset, (cy + 1) * cp - inset],
                    outline=(255, 255, 255), width=max(2, cp // 8),
                )

    # Coordinate grid + axis labels.
    if grid:
        gcol = (120, 123, 135)
        lcol = (255, 246, 120)
        gfont = _minimap_font(max(12, int(cp * 0.85)))
        map_h = h * cp
        for gx in range(0, w + 1, 10):
            x = min(w * cp - 1, gx * cp)
            draw.line([(x, 0), (x, map_h)], fill=gcol, width=2)
            if gx < w:
                draw.text(
                    (x + 3, 2), str(gx), fill=lcol, font=gfont,
                    stroke_width=3, stroke_fill=(0, 0, 0),
                )
        for gy in range(0, h + 1, 10):
            y = min(map_h - 1, gy * cp)
            draw.line([(0, y), (w * cp, y)], fill=gcol, width=2)
            if gy < h:
                draw.text(
                    (3, y + 2), str(gy), fill=lcol, font=gfont,
                    stroke_width=3, stroke_fill=(0, 0, 0),
                )

    # Legend strip.
    if legend:
        ly = h * cp
        draw.rectangle([0, ly, w * cp, ly + legend_h], fill=(24, 24, 30))
        lfont = _minimap_font(max(11, int(cp * 0.7)))
        sample = cp  # one-cell-sized sample swatch
        items = [
            ("infantry", "Infantry"),
            ("vehicle", "Vehicle"),
            ("harvester", "Harvester"),
            ("building", "Building"),
        ]
        x = int(cp * 0.4)
        row_y = ly + int(cp * 0.2)
        for cat, name in items:
            # Sample shape swatch drawn at pixel coords.
            m = sample * 0.18
            sx0, sy0 = x + m, row_y + m
            sx1, sy1 = x + sample - m, row_y + sample - m
            smx, smy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            if cat == "infantry":
                draw.ellipse([sx0, sy0, sx1, sy1], fill=_OWN)
            elif cat == "harvester":
                draw.polygon(
                    [(smx, sy0), (sx0, sy1), (sx1, sy1)], fill=_OWN
                )
            elif cat == "building":
                draw.polygon(
                    [(smx, sy0), (sx1, smy), (smx, sy1), (sx0, smy)],
                    fill=_OWN_BLD,
                )
            else:
                draw.rectangle([sx0, sy0, sx1, sy1], fill=_OWN)
            draw.text(
                (x + sample + 4, row_y + sample * 0.18), name,
                fill=(235, 235, 245), font=lfont,
            )
            x += sample + int(cp * 4.2)
        draw.text(
            (int(cp * 0.4), ly + int(cp * 1.05)),
            "green = your forces    red/orange = enemy    "
            "number = units stacked on that cell",
            fill=(200, 202, 212), font=lfont,
        )

    return img
