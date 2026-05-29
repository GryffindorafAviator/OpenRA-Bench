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
import struct
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


# Terrain tile codes — match the `mapgen.py` constants. The .oramap
# `map.bin` v2 layout is column-major u16 tile ids; we extract the full
# w×h grid so the minimap can paint water/wall cells as a TERRAIN
# UNDERLAY (visible from t=0, even in unexplored areas). Without this
# underlay the model has to learn map topology by walking units into
# the fog — the SHAPE of the map (water channels, bridges, walls)
# should not require fog-of-war discovery; only the contents (units,
# buildings, ore) should.
_TERRAIN_CLEAR = 255      # passable grass — drawn as the default bg
_TERRAIN_WATER = 1        # impassable water — drawn as blue underlay
# A small allowlist of tile ids that show up in the bench's procedural
# maps; anything outside this set is treated as clear. RA's real
# tilesets carry hundreds of distinct tile ids — the bench's generated
# maps only emit two — but we keep this defensive so a hand-authored
# .oramap with extra ids degrades gracefully (renders as clear, not as
# a misleading water cell).
_WATER_TILE_IDS = frozenset({_TERRAIN_WATER})


@lru_cache(maxsize=32)
def terrain_grid_for(base_map: str) -> tuple[tuple[int, ...], ...] | None:
    """Decode `map.bin` from the resolved `.oramap` into a `grid[y][x]`
    of tile ids (read-only tuple-of-tuples so it's safely cached).

    Returns None on any failure (missing map, unsupported `map.bin`
    version, parse error). Callers must tolerate None — the renderer
    falls back to the existing all-clear background.

    `map.bin` v2 layout (matches `mapgen._map_bin`):
        byte 0     : format version (2)
        bytes 1-2  : width (u16 LE)
        bytes 3-4  : height (u16 LE)
        bytes 5-8  : tiles offset (u32 LE)  — typically 17
        bytes 9-12 : height-layer offset (u32 LE, 0 if absent)
        bytes 13-16: resources offset (u32 LE)
        tiles      : column-major (x outer, y inner) — 3 bytes per cell:
                     u16 tile id + u8 tile index
    """
    if not base_map:
        return None
    try:
        from .scenarios.loader import resolve_map_path

        p = resolve_map_path(base_map)
        if not p:
            return None
        with zipfile.ZipFile(p, "r") as zf:
            if "map.bin" not in zf.namelist():
                return None
            data = zf.read("map.bin")
        if len(data) < 17 or data[0] != 2:
            return None
        w = struct.unpack_from("<H", data, 1)[0]
        h = struct.unpack_from("<H", data, 3)[0]
        tiles_off = struct.unpack_from("<I", data, 5)[0]
        # Sanity: tiles block must fit
        need = tiles_off + 3 * w * h
        if w <= 0 or h <= 0 or w * h > 1_000_000 or need > len(data):
            return None
        # Build grid[y][x] from the column-major payload
        rows = [[0] * w for _ in range(h)]
        for x in range(w):
            for y in range(h):
                idx = tiles_off + 3 * (x * h + y)
                rows[y][x] = struct.unpack_from("<H", data, idx)[0]
        return tuple(tuple(r) for r in rows)
    except Exception as e:  # noqa: BLE001
        logger.debug("terrain grid extract failed for %s: %s", base_map, e)
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

# (R,G,B) — three shroud states, RA semantics:
#   _BG_UNKNOWN  ('#') — never explored, full shroud (darkest)
#   _BG_FOGGED   ('.') — explored once but NO live vision NOW (mid)
#   _BG_VISIBLE  ('+') — within an agent actor's sight RIGHT NOW (bright)
# The fogged tint is intentionally distinguishable at a glance from the
# bright tint — without this, a unit that retreats from a region leaves
# the region looking like it still has live vision while the enemy
# markers vanish (the user-reported "the enemy disappeared" bug). For
# legacy `.` callers that don't write `+` cells, `.` keeps the old
# bright tint (back-compat: a 2-state grid renders identical to before).
#
# The 3-state delta was originally 44→70 (only 26 levels apart): too
# subtle for image-primary LLM inference at minimap resolution. The
# values below give a ≥2× brightness delta between fogged and visible
# (≈60 vs ≈140) — the model can now tell "I see this cell NOW" from
# "I explored this cell 10 turns ago" at a glance.
_BG_UNKNOWN = (18, 18, 22)      # '#'  unexplored          — very dark
_BG_FOGGED = (60, 62, 70)       # '.'  fogged-only         — mid-dark
_BG_VISIBLE = (140, 144, 152)   # '+'  currently visible   — bright
_BG_EXPLORED = _BG_VISIBLE      # legacy 2-state alias for '.'

# Terrain underlay — painted BEFORE the shroud so map structure (water
# channels, bridges, walls) is visible from t=0 even in unexplored
# regions. The shroud then brightens (visible) / dims (fogged) /
# obscures (unexplored) the terrain colour. Three terrain palettes,
# each with three shroud-state brightnesses:
#   * unexplored = dim ── the model sees "there is water HERE" but
#     can't see enemy units / buildings in that water yet.
#   * fogged     = mid
#   * visible    = bright
# The CLEAR palette intentionally reuses the existing _BG_UNKNOWN /
# _BG_FOGGED / _BG_VISIBLE so a map with no water (every cell CLEAR)
# renders byte-identical to the pre-terrain renderer (back-compat).
_BG_WATER_UNKNOWN  = (18, 30, 50)     # very dark blue
_BG_WATER_FOGGED   = (32, 60, 100)    # mid blue
_BG_WATER_VISIBLE  = (60, 110, 180)   # bright blue
_BG_CLEAR_UNKNOWN  = _BG_UNKNOWN      # alias — same defaults as 'no terrain'
_BG_CLEAR_FOGGED   = _BG_FOGGED
_BG_CLEAR_VISIBLE  = _BG_VISIBLE
_OWN = (60, 200, 90)            # your units
_OWN_BLD = (60, 200, 90)        # your buildings — SAME hue as units
                                # (per-type SHAPE distinguishes role,
                                # hue distinguishes side).
_ENEMY = (225, 60, 55)          # enemy units
_ENEMY_BLD = (225, 60, 55)      # enemy buildings — SAME hue as units
_OBJECTIVE = (255, 218, 70)     # objective / target region
_ORE = (185, 150, 70)           # visible ore/resource cells

# Per-side hue family for buildings — kept identical to the unit hue so
# "self vs enemy" reads from COLOUR alone and "role" reads from SHAPE
# alone. The previous renderer used blue=own-bld / orange=enemy-bld; the
# image-primary upgrade collapses both onto green=own / red=enemy so a
# model can read side and type independently (the task brief's key
# requirement).
_OWN_ACCENT = (190, 250, 210)   # lighter green — overlay marks on own
_ENEMY_ACCENT = (255, 200, 180) # lighter red — overlay marks on enemy


def _bg_for(ch: str, has_visible_mark: bool) -> tuple[int, int, int] | None:
    """Map an ASCII shroud char → RGB for the 3-state shroud, or None
    for unexplored (caller leaves the default unknown background).

    `has_visible_mark` is True iff the ASCII grid contains any '+'
    cells (the 3-state encoding). When False the grid is a legacy
    2-state map — we render '.' as bright (back-compat) so existing
    callers don't see their maps go dim. When True we render '.' as
    DIM and '+' as bright — the new 3-state shroud.
    """
    if ch == "#":
        return None
    if ch == "+":
        return _BG_VISIBLE
    # ch == '.' or any other non-'#' char
    return _BG_FOGGED if has_visible_mark else _BG_EXPLORED


def _terrain_bg(tile_id: int, ch: str, has_visible_mark: bool
                ) -> tuple[int, int, int]:
    """Resolve the (terrain × shroud) background for one cell.

    Returns the unexplored tint when `ch == '#'` (so unexplored water
    still reads as DIM BLUE — the model sees the map shape even in
    fog), the fogged tint for explored-not-visible cells, the visible
    tint for currently-in-sight cells. Per-terrain (clear vs water)
    palettes — each with three shroud brightnesses — give a
    perception channel that lifts map TOPOLOGY out of the fog while
    keeping CONTENTS (enemies) hidden until scouted.
    """
    is_water = tile_id in _WATER_TILE_IDS
    if ch == "#":
        return _BG_WATER_UNKNOWN if is_water else _BG_CLEAR_UNKNOWN
    if ch == "+":
        return _BG_WATER_VISIBLE if is_water else _BG_CLEAR_VISIBLE
    # ch == '.' or any other non-'#' char
    if has_visible_mark:
        return _BG_WATER_FOGGED if is_water else _BG_CLEAR_FOGGED
    # 2-state legacy: '.' renders as bright (back-compat).
    return _BG_WATER_VISIBLE if is_water else _BG_CLEAR_VISIBLE


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
    has_visible = any("+" in r for r in rows)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            bg = _bg_for(ch, has_visible)
            if bg is None:
                continue
            for sy in range(CELL):
                for sx in range(CELL):
                    px[x * CELL + sx, y * CELL + sy] = bg

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


# Distinct SHAPE per unit TYPE — so e.g. 1tnk and 2tnk are visually
# different on the minimap, not both "a vehicle". Types not listed fall
# back by category (infantry→circle, harvester→tridown, else→square).
_TYPE_ICON: dict[str, tuple[str, str | None]] = {
    # infantry / specialists
    "e1": ("circle", None),
    "e2": ("diamond", "cross"),
    "e3": ("triangle", "dot"),
    "e4": ("cross", None),
    "e6": ("pentagon", "cross"),
    "engineer": ("pentagon", "cross"),
    "e7": ("star", "halo"),
    "tanya": ("star", "halo"),
    "medi": ("plus", "dot"),
    "mech": ("plus", "cross"),
    "spy": ("kite", "ring"),
    "thf": ("kite", "dot"),
    "dog": ("diamond", "dot"),
    # ground vehicles
    "1tnk": ("square", None),
    "2tnk": ("diamond", None),
    "3tnk": ("hexagon", None),
    "4tnk": ("triangle", None),
    "harv": ("tridown", None),
    "jeep": ("pentagon", None),
    "apc": ("hexagon", "dot"),
    "mcv": ("trapezoid", "halo"),
    "arty": ("star", "crosshair"),
    "v2rl": ("star", "dot"),
    "ftrk": ("trapezoid", "cross"),
    "msam": ("hexagon", "antenna"),
    "stnk": ("kite", "ring"),
    "ftnk": ("trapezoid", "dot"),
    "ttnk": ("hexagon", "cross"),
    # naval / aircraft
    "dd": ("diamond", "wave"),
    "lst": ("square", "wave"),
    "heli": ("chevron", None),
    "hind": ("chevron", "dot"),
    "tran": ("chevron", "ring"),
    "mig": ("kite", "antenna"),
    "yak": ("kite", "cross"),
    "badr": ("kite", "halo"),
    "u2": ("kite", "dot2"),
}


# ── Per-building-type icons (image-primary readability) ────────────────
# Each building type maps to (SHAPE, ACCENT) where ACCENT is an extra
# glyph drawn ON TOP of the fill — a central dot, antenna line,
# crosshair, etc. — that disambiguates buildings that share a base
# shape. The model glances at the silhouette + accent and reads the
# role (production / power / tech / defence / wall / fact) without
# needing the text legend. Side (own / enemy) is encoded by COLOUR
# only — green / red — so shape and hue carry orthogonal information.
#
# Shapes referenced here MUST be supported in `_shape_points` /
# `_draw_building_icon`; accents are handled by `_draw_accent`.
_BUILDING_ICON: dict[str, tuple[str, str | None]] = {
    # base / loss-critical
    "fact":  ("pentagon", "halo"),     # large pentagon w/ inner halo
    # economy
    "proc":  ("trapezoid", None),      # refinery trapezoid
    "silo":  ("bar", None),            # narrow upright bar
    "mine":  ("circle", "ring"),       # neutral ore mine marker
    # power
    "powr":  ("diamond", "dot"),       # diamond w/ central dot
    "apwr":  ("diamond", "dot2"),      # diamond w/ double dot (advanced)
    # production
    "weap":  ("hexagon", None),
    "tent":  ("triangle", None),
    "barr":  ("triangle", "dot"),
    "hpad":  ("square", "x"),          # airpad — square w/ X
    "afld":  ("square", "x"),
    "syrd":  ("square", "wave"),       # naval — square w/ wave
    "spen":  ("square", "wave"),
    # support / tech / radar
    "dome":  ("circle", "ring"),       # radar dome — circle w/ ring
    "radar": ("circle", "antenna"),    # radar w/ antenna
    "fix":   ("square", "cross"),      # service depot — square w/ +
    "atek":  ("hex_tall", None),       # tech — tall hex
    "stek":  ("hex_tall", None),
    "kenn":  ("circle", "dot"),        # dog kennel — small marker
    # superweapons (unique silhouettes — large stars)
    "mslo":  ("star", None),
    "pdox":  ("star", None),
    "iron":  ("star", None),
    # defences (all share crosshair-square; cheap pillbox slightly
    # smaller via `r` shrink in _draw_building_icon)
    "pbox":  ("def_square", "crosshair"),
    "hbox":  ("def_square", "dot"),
    "gun":   ("def_square", "ring"),
    "agun":  ("def_square", "x"),
    "sam":   ("def_square", "antenna"),
    "ftur":  ("def_square", "cross"),
    "tsla":  ("star", "ring"),
    # walls — thin tile (drawn small/centred so a wall row reads as
    # a continuous strip rather than fat blocks)
    "brik":  ("wall_tile", None),
    "sbag":  ("wall_tile", None),
    "cycl":  ("wall_tile", None),
    "fenc":  ("wall_tile", None),
    "wall":  ("wall_tile", None),
}


def _building_icon(actor_type: str) -> tuple[str, str | None]:
    """(shape, accent) for a building type, with fallback."""
    t = (actor_type or "").strip().lower()
    return _BUILDING_ICON.get(t, ("building", None))


def _unit_shape(actor_type: str, is_building: bool) -> str:
    """Shape key for a unit/building — distinct per unit TYPE."""
    if is_building:
        shape, _accent = _building_icon(actor_type)
        return shape
    t = (actor_type or "").strip().lower()
    if t in _TYPE_ICON:
        shape, _accent = _TYPE_ICON[t]
        return shape
    cat = _unit_category(t, False)
    if cat == "infantry":
        return "circle"
    if cat == "harvester":
        return "tridown"
    return "square"


def _unit_accent(actor_type: str, is_building: bool) -> str | None:
    if is_building:
        _shape, accent = _building_icon(actor_type)
        return accent
    t = (actor_type or "").strip().lower()
    if t in _TYPE_ICON:
        _shape, accent = _TYPE_ICON[t]
        return accent
    return None


# The set of building-only shapes — used by `render_png_b64` (legacy
# fallback) to draw with the building hue / radius. The richer
# `render_tactical_minimap` carries an explicit is_building flag per
# occupant so it never relies on this set.
_BUILDING_ONLY_SHAPES = frozenset({
    "building", "pentagon", "trapezoid", "bar", "hex_tall",
    "def_square", "wall_tile",
})


def _shape_points(shape, x0, y0, x1, y1):
    """Polygon vertices for `shape` in the box; None for an ellipse."""
    import math

    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2, (y1 - y0) / 2
    if shape == "circle":
        return None
    if shape == "diamond":
        return [(mx, y0), (x1, my), (mx, y1), (x0, my)]
    if shape == "triangle":
        return [(mx, y0), (x1, y1), (x0, y1)]
    if shape == "tridown":
        return [(x0, y0), (x1, y0), (mx, y1)]
    if shape == "kite":
        return [(mx, y0), (x1, my + ry * 0.12), (mx, y1),
                (x0, my + ry * 0.12)]
    if shape == "chevron":
        # Aircraft / helicopter marker: a forward-pointing chevron.
        return [(mx, y0), (x1, y1), (mx, y1 - ry * 0.45),
                (x0, y1)]
    if shape == "plus":
        a = min(rx, ry) * 0.36
        return [(mx - a, y0), (mx + a, y0), (mx + a, my - a),
                (x1, my - a), (x1, my + a), (mx + a, my + a),
                (mx + a, y1), (mx - a, y1), (mx - a, my + a),
                (x0, my + a), (x0, my - a), (mx - a, my - a)]
    if shape == "cross":
        # Diagonal cross: flame/area-effect infantry reads differently
        # from rifle/rocket/specialist silhouettes.
        a = min(rx, ry) * 0.32
        return [(x0 + a, y0), (mx, my - a), (x1 - a, y0),
                (x1, y0 + a), (mx + a, my), (x1, y1 - a),
                (x1 - a, y1), (mx, my + a), (x0 + a, y1),
                (x0, y1 - a), (mx - a, my), (x0, y0 + a)]
    if shape == "trapezoid":
        # Refinery silhouette — wide base, narrow top.
        top_inset = (x1 - x0) * 0.22
        return [(x0 + top_inset, y0), (x1 - top_inset, y0),
                (x1, y1), (x0, y1)]
    if shape == "bar":
        # Narrow upright bar (silo).
        bw = (x1 - x0) * 0.32
        return [(mx - bw, y0), (mx + bw, y0),
                (mx + bw, y1), (mx - bw, y1)]
    if shape == "hex_tall":
        # Tall hexagon — tech-building silhouette.
        s = (y1 - y0) * 0.22
        return [(mx, y0), (x1, y0 + s), (x1, y1 - s),
                (mx, y1), (x0, y1 - s), (x0, y0 + s)]
    if shape == "def_square":
        # Defence square — full box (crosshair accent layered on).
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if shape == "wall_tile":
        # Thin centred tile — a wall row reads as a continuous strip.
        ix = (x1 - x0) * 0.22
        iy = (y1 - y0) * 0.22
        return [(x0 + ix, y0 + iy), (x1 - ix, y0 + iy),
                (x1 - ix, y1 - iy), (x0 + ix, y1 - iy)]
    if shape in ("hexagon", "pentagon", "star"):
        n = {"hexagon": 6, "pentagon": 5, "star": 5}[shape]
        pts = []
        if shape == "star":
            for i in range(2 * n):
                ang = -math.pi / 2 + i * math.pi / n
                r = 1.0 if i % 2 == 0 else 0.42
                pts.append((mx + r * rx * math.cos(ang),
                            my + r * ry * math.sin(ang)))
        else:
            for i in range(n):
                ang = -math.pi / 2 + i * 2 * math.pi / n
                pts.append((mx + rx * math.cos(ang),
                            my + ry * math.sin(ang)))
        return pts
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # square / building


def _draw_shape(draw, x0, y0, x1, y1, shape, color):
    """Draw `shape` filling the box (x0,y0)-(x1,y1) in `color`."""
    outline = (15, 15, 18)
    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=color, outline=outline)
    elif shape == "building":
        draw.rectangle([x0, y0, x1, y1], fill=color,
                       outline=outline, width=2)
    elif shape == "def_square":
        # Defence base — square with a thicker outline so the crosshair
        # accent reads on top of the fill.
        draw.rectangle([x0, y0, x1, y1], fill=color,
                       outline=outline, width=2)
    else:
        draw.polygon(_shape_points(shape, x0, y0, x1, y1),
                     fill=color, outline=outline)


def _draw_accent(draw, x0, y0, x1, y1, accent, side: str = "own"):
    """Overlay a per-building accent glyph (dot / cross / x / ring /
    antenna / wave / crosshair / halo) on the filled shape, so two
    buildings sharing a base silhouette (e.g. hpad vs syrd — both
    squares) read as distinct at a glance."""
    if not accent:
        return
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = max(1.0, (x1 - x0) / 2), max(1.0, (y1 - y0) / 2)
    col = (240, 240, 248) if side == "own" else (250, 230, 220)
    dark = (15, 15, 18)
    if accent == "dot":
        r = min(rx, ry) * 0.28
        draw.ellipse([mx - r, my - r, mx + r, my + r], fill=col,
                     outline=dark)
    elif accent == "dot2":
        # double-dot — distinguish apwr from powr
        r = min(rx, ry) * 0.22
        dx = rx * 0.36
        for cx in (mx - dx, mx + dx):
            draw.ellipse([cx - r, my - r, cx + r, my + r], fill=col,
                         outline=dark)
    elif accent == "ring":
        r1 = min(rx, ry) * 0.5
        r2 = min(rx, ry) * 0.22
        w = max(2, int(min(rx, ry) * 0.18))
        draw.ellipse([mx - r1, my - r1, mx + r1, my + r1], outline=col,
                     width=w)
        draw.ellipse([mx - r2, my - r2, mx + r2, my + r2], fill=col)
    elif accent == "antenna":
        # vertical line through centre, taller than the cell, plus a
        # bulb at the top — the "radar tower" silhouette
        w = max(2, int(min(rx, ry) * 0.18))
        draw.line([(mx, my - ry * 1.05), (mx, my + ry * 0.2)],
                  fill=col, width=w)
        r = min(rx, ry) * 0.18
        draw.ellipse([mx - r, my - ry * 1.05 - r,
                      mx + r, my - ry * 1.05 + r], fill=col)
    elif accent == "x":
        w = max(2, int(min(rx, ry) * 0.22))
        p = min(rx, ry) * 0.55
        draw.line([(mx - p, my - p), (mx + p, my + p)], fill=col,
                  width=w)
        draw.line([(mx + p, my - p), (mx - p, my + p)], fill=col,
                  width=w)
    elif accent == "cross":
        w = max(2, int(min(rx, ry) * 0.22))
        p = min(rx, ry) * 0.55
        draw.line([(mx - p, my), (mx + p, my)], fill=col, width=w)
        draw.line([(mx, my - p), (mx, my + p)], fill=col, width=w)
    elif accent == "wave":
        # sine-like dashes — "water / naval" hint
        w = max(2, int(min(rx, ry) * 0.18))
        p = rx * 0.6
        draw.line([(mx - p, my - ry * 0.18), (mx - p * 0.2, my - ry * 0.18)],
                  fill=col, width=w)
        draw.line([(mx + p * 0.2, my + ry * 0.18),
                   (mx + p, my + ry * 0.18)], fill=col, width=w)
    elif accent == "crosshair":
        w = max(2, int(min(rx, ry) * 0.22))
        p = min(rx, ry) * 0.55
        # central dot
        r = min(rx, ry) * 0.18
        draw.ellipse([mx - r, my - r, mx + r, my + r], fill=col,
                     outline=dark)
        # arms — stop at the dot
        draw.line([(mx - p, my), (mx - r, my)], fill=col, width=w)
        draw.line([(mx + r, my), (mx + p, my)], fill=col, width=w)
        draw.line([(mx, my - p), (mx, my - r)], fill=col, width=w)
        draw.line([(mx, my + r), (mx, my + p)], fill=col, width=w)
    elif accent == "halo":
        # inner halo ring — the loss-critical fact glyph
        r = min(rx, ry) * 0.42
        w = max(2, int(min(rx, ry) * 0.18))
        draw.ellipse([mx - r, my - r, mx + r, my + r], outline=col,
                     width=w)


def _draw_unit_shape(draw, cx, cy, cp, shape, color, accent: str | None = None,
                     side: str = "own"):
    """Draw `shape` filling ~70% of the cp-px cell at grid (cx, cy)."""
    m = cp * 0.16
    x0, y0 = cx * cp + m, cy * cp + m
    x1, y1 = (cx + 1) * cp - m, (cy + 1) * cp - m
    _draw_shape(draw, x0, y0, x1, y1, shape, color)
    if accent:
        _draw_accent(draw, x0, y0, x1, y1, accent, side)


# ── HP bar overlay ─────────────────────────────────────────────────────
# A small horizontal bar drawn ABOVE the icon, length and colour both
# encoding health. The bench surfaces HP as a 0..1 fraction on each
# unit / building entry (`render_state` schema: `hp` for units,
# `hp_pct` for the raw engine payload). Skip drawing at full HP so the
# minimap stays clean — only damaged actors flag for attention.

def _hp_color(frac: float) -> tuple[int, int, int]:
    if frac >= 0.75:
        return (90, 220, 110)        # green
    if frac >= 0.50:
        return (240, 230, 90)        # yellow
    if frac >= 0.25:
        return (245, 165, 70)        # orange
    return (235, 70, 70)             # red


def _draw_hp_bar(draw, cx: int, cy: int, cp: int, frac: float) -> None:
    """Draw a short HP bar above cell (cx, cy). `frac` ∈ [0,1]. The bar
    is omitted at full HP (≥ 0.99) so undamaged actors don't add
    visual noise."""
    try:
        f = max(0.0, min(1.0, float(frac)))
    except (TypeError, ValueError):
        return
    if f >= 0.99:
        return  # full HP — skip
    bar_w = int(cp * 0.86)
    bar_h = max(3, int(cp * 0.20))
    x0 = cx * cp + (cp - bar_w) // 2
    # Sit the bar just above the icon (icon margin = cp*0.16).
    y0 = cy * cp + max(1, int(cp * 0.02))
    # Bg track (so 0-fill is still visible)
    draw.rectangle([x0 - 1, y0 - 1, x0 + bar_w + 1, y0 + bar_h + 1],
                   fill=(20, 20, 24), outline=(255, 255, 255))
    fill_w = max(2, int(bar_w * f))
    draw.rectangle([x0, y0, x0 + fill_w, y0 + bar_h],
                   fill=_hp_color(f))


def _draw_objective_region(draw, region, cp, w, h, index):
    try:
        cx = int(region["x"])
        cy = int(region["y"])
        radius = float(region.get("radius", 3))
    except (KeyError, TypeError, ValueError):
        return
    if not (0 <= cx < w and 0 <= cy < h):
        return

    x0 = max(0, (cx - radius) * cp)
    y0 = max(0, (cy - radius) * cp)
    x1 = min(w * cp - 1, (cx + radius + 1) * cp - 1)
    y1 = min(h * cp - 1, (cy + radius + 1) * cp - 1)
    width = max(3, cp // 7)
    draw.ellipse([x0, y0, x1, y1], outline=_OBJECTIVE, width=width)

    mid_x, mid_y = (cx + 0.5) * cp, (cy + 0.5) * cp
    arm = max(cp * 0.45, width * 2)
    draw.line([(mid_x - arm, mid_y), (mid_x + arm, mid_y)],
              fill=_OBJECTIVE, width=width)
    draw.line([(mid_x, mid_y - arm), (mid_x, mid_y + arm)],
              fill=_OBJECTIVE, width=width)

    label = str(region.get("label") or f"OBJ {index}")
    font = _minimap_font(max(11, int(cp * 0.7)))
    draw.text(
        (min(w * cp - cp * 4, max(2, x0 + cp * 0.2)), max(2, y0 - cp)),
        label, fill=_OBJECTIVE, font=font,
        stroke_width=max(2, cp // 12), stroke_fill=(0, 0, 0),
    )


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


def _draw_resource_cell(draw, cell, cp, w, h):
    try:
        cx = int(cell["cell_x"])
        cy = int(cell["cell_y"])
    except (KeyError, TypeError, ValueError):
        return
    if not (0 <= cx < w and 0 <= cy < h):
        return
    pad = max(1, cp // 8)
    draw.rectangle(
        [cx * cp + pad, cy * cp + pad, (cx + 1) * cp - pad,
         (cy + 1) * cp - pad],
        fill=_ORE,
    )


def render_tactical_minimap(
    render_state: dict,
    scale: int = 4,
    grid: bool = True,
    legend: bool = True,
    selected=None,
    arrows=None,
    unit_labels=None,
    base_map: str = "",
    terrain_grid=None,
):
    """A legible tactical minimap as a PIL RGB image:

    * per-type SHAPES — ● infantry, ■ vehicle, ▲ harvester, ◆ building;
    * COUNT badge when >1 unit stacks on a cell (so overlapping units
      are not silently rendered as one dot);
    * colour by side — green = you, red = enemy;
    * a coordinate GRID with axis labels every 10 cells, and a LEGEND
      strip beneath the map.

    `scale` multiplies the 6px base cell. Returns None if Pillow is
    missing or there is nothing to draw.

    `unit_labels` (id-str → handle, e.g. `{"1004": "tank-1"}`) drives
    the image-primary perception channel: every actor is tagged with
    its legible handle so the model can identify and command units
    from the picture alone (the text briefing carries no positions).
    When set, the per-cell count badge is replaced by the labels.

    `base_map` (logical id, e.g. `"adversarial-1v1-macro-arena"`) or
    an explicit `terrain_grid` (list[list[int]] of tile ids, indexed
    [y][x]) enables the TERRAIN UNDERLAY: water cells render as blue
    even in unexplored regions, so the map's SHAPE (channels,
    bridges, walls) is legible from t=0 — only the CONTENTS (units,
    buildings, ore) remain gated by the fog. Without this, an
    image-primary model has to walk a unit into the fog just to
    learn that there's a river in the way. `base_map` is looked up
    via `terrain_grid_for()` (cached); when both are provided the
    explicit `terrain_grid` wins."""
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
    # Legend height grows to 3 cells (chip strip + HP/ore/objective row +
    # one row of help text). Without legend the image is just the map.
    legend_h = cp * 3 if legend else 0
    img = Image.new("RGB", (w * cp, h * cp + legend_h), _BG_UNKNOWN)
    draw = ImageDraw.Draw(img)

    # Terrain underlay grid — explicit `terrain_grid` wins, else
    # look up by `base_map`, else by `render_state["base_map"]`.
    # None ⇒ no underlay (the legacy all-clear background).
    if terrain_grid is None:
        bm = base_map or str(render_state.get("base_map") or "")
        if bm:
            terrain_grid = terrain_grid_for(bm)

    def _tile_at(x: int, y: int) -> int:
        # Defensive: a hand-written `terrain_grid` may not exactly
        # match the ASCII minimap dims (e.g. a `Bounds:` rectangle
        # inside a larger `MapSize:`); clamp to the grid extent and
        # treat OOB cells as clear so we never crash on edge pixels.
        if terrain_grid is None:
            return _TERRAIN_CLEAR
        try:
            row = terrain_grid[y]
            return row[x]
        except (IndexError, TypeError):
            return _TERRAIN_CLEAR

    # Shroud + terrain combined paint — 3-state shroud when the ASCII
    # grid contains '+' marks (visible), 2-state legacy bright fill
    # otherwise. Terrain underlay paints every cell (including '#'
    # unexplored ones) so map SHAPE is visible from t=0; the shroud
    # then dims/brightens that terrain to encode "do I have vision
    # here right now".
    has_visible = any("+" in r for r in rows)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if terrain_grid is not None:
                # Paint every cell — unexplored water/wall is visible
                # as a DIM blue/gray, not solid black.
                bg = _terrain_bg(_tile_at(x, y), ch, has_visible)
            else:
                # No terrain underlay: legacy path. Unexplored cells
                # leave the default _BG_UNKNOWN background unset so
                # we keep byte-identical output for back-compat with
                # the existing tests + 2-state callers.
                bg = _bg_for(ch, has_visible)
                if bg is None:
                    continue
            draw.rectangle(
                [x * cp, y * cp, (x + 1) * cp - 1, (y + 1) * cp - 1],
                fill=bg,
            )

    for i, region in enumerate(render_state.get("objective_regions") or [], 1):
        if isinstance(region, dict):
            _draw_objective_region(draw, region, cp, w, h, i)
    for cell in render_state.get("resource_cells") or []:
        if isinstance(cell, dict):
            _draw_resource_cell(draw, cell, cp, w, h)

    # Collect every actor by cell so stacked units can be counted.
    # Each occupant is a dict so we can carry shape + accent + side +
    # is_building + hp without growing a fragile tuple position.
    by_cell: dict[tuple[int, int], list[dict]] = {}
    # Distinct (type, shape, accent, is_building) of OWN actors seen —
    # drives the legend strip. Order preserved by insertion.
    own_types: dict[str, tuple[str, str | None, bool]] = {}
    # Same for enemy actors that the agent has spotted — so the legend
    # shows a RED swatch next to a representative enemy type.
    enemy_types: dict[str, tuple[str, str | None, bool]] = {}

    def _collect(items, side, force_building):
        for it in items or []:
            if not isinstance(it, dict):
                continue
            cx = int(it.get("cell_x", -99))
            cy = int(it.get("cell_y", -99))
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            is_b = force_building or bool(it.get("is_building"))
            atype = (it.get("actor_type") or it.get("type") or "?")
            shape = _unit_shape(atype, is_b)
            accent = _unit_accent(atype, is_b)
            # HP — units carry `hp` (0..1), buildings the same. Some
            # legacy callers omit the field; treat absent as full HP so
            # the bar is suppressed.
            hp_raw = it.get("hp")
            try:
                hp = float(hp_raw) if hp_raw is not None else 1.0
            except (TypeError, ValueError):
                hp = 1.0
            by_cell.setdefault((cx, cy), []).append({
                "side": side,
                "shape": shape,
                "accent": accent,
                "is_building": is_b,
                "id": it.get("id"),
                "hp": hp,
                "type": str(atype).lower() if atype != "?" else "?",
            })
            bucket = own_types if side == "own" else enemy_types
            if atype != "?":
                bucket.setdefault(str(atype).lower(),
                                  (shape, accent, is_b))

    _collect(render_state.get("units_summary"), "own", False)
    _collect(render_state.get("own_buildings"), "own", True)
    _collect(render_state.get("enemy_summary"), "enemy", False)
    _collect(
        render_state.get("enemy_buildings_summary")
        or render_state.get("enemy_buildings"),
        "enemy", True,
    )

    def _color(side, is_b):
        if side == "own":
            return _OWN_BLD if is_b else _OWN
        return _ENEMY_BLD if is_b else _ENEMY

    badge_font = _minimap_font(max(9, int(cp * 0.62)))
    for (cx, cy), occ in by_cell.items():
        # Dominant occupant decides the shape; prefer a building (the
        # static structure beats a transient unit passing through).
        bld = next((o for o in occ if o["is_building"]), None)
        dom = bld if bld else occ[0]
        _draw_unit_shape(
            draw, cx, cy, cp, dom["shape"],
            _color(dom["side"], dom["is_building"]),
            accent=dom["accent"], side=dom["side"],
        )
        # HP bar — uses the MIN hp across stacked actors so the model
        # sees "something at this cell is hurt"; a damaged garrison /
        # patched building both flag the right cell.
        min_hp = min((o["hp"] for o in occ), default=1.0)
        _draw_hp_bar(draw, cx, cy, cp, min_hp)
        # The count badge and per-unit labels are mutually exclusive —
        # `unit_labels` (image-primary) names each occupant individually,
        # which already disambiguates a stack.
        if len(occ) > 1 and not unit_labels:
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

    # Per-unit ID labels — the image-primary channel. Each actor's
    # legible handle (`tank-1`, `enemy-2`) is placed near its marker
    # with greedy collision avoidance (labels nudge clear of one
    # another into free space) and a leader line when a label drifts
    # off its cell — so the model can identify every unit from the
    # picture alone. Drawn last so nothing occludes the text.
    if unit_labels:
        lab_font = _minimap_font(max(15, int(cp * 0.95)))
        img_w, img_h = w * cp, h * cp
        # One entry per actor id — an actor can appear in both the unit
        # and building lists; dedup so its label is drawn once.
        seen: set = set()
        actors = []  # (cx, cy, label, side)
        for (cx, cy), occ in by_cell.items():
            for o in occ:
                aid = o["id"]
                key = str(aid)
                if aid is None or key in seen or key not in unit_labels:
                    continue
                seen.add(key)
                actors.append((cx, cy, unit_labels[key], o["side"]))
        placed: list = []  # occupied label rects (x0, y0, x1, y1)

        def _free(x0, y0, x1, y1):
            return all(
                x1 < r[0] or x0 > r[2] or y1 < r[1] or y0 > r[3]
                for r in placed
            )

        for cx, cy, lab, side in sorted(actors, key=lambda a: (a[1], a[0])):
            try:
                bb = draw.textbbox((0, 0), lab, font=lab_font, stroke_width=3)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:  # noqa: BLE001
                tw, th = len(lab) * 9, 16
            mx, my = cx * cp + cp // 2, cy * cp + cp // 2
            lx = min(max(0, mx + 4), img_w - tw - 1)
            ly = my - cp - th
            step = th + 3
            for _ in range(60):  # nudge upward into free space
                cand = min(max(1, ly), img_h - th - 1)
                if _free(lx, cand, lx + tw, cand + th):
                    ly = cand
                    break
                ly -= step
            else:
                ly = min(max(1, my + cp), img_h - th - 1)
            placed.append((lx, ly, lx + tw, ly + th))
            # Leader line when the label sits away from its marker.
            if abs(lx + tw // 2 - mx) > cp or abs(ly + th // 2 - my) > cp:
                draw.line(
                    [(mx, my), (lx + 2, ly + th // 2)],
                    fill=(165, 167, 178), width=1,
                )
            col = (175, 255, 175) if side == "own" else (255, 190, 172)
            draw.text(
                (lx, ly), lab, fill=col, font=lab_font,
                stroke_width=3, stroke_fill=(0, 0, 0),
            )

    # ── Legend strip — VISUAL chip key ─────────────────────────────────
    # The legend is the image-only model's only ground-truth mapping
    # from glyph to type, so every chip is drawn with the SAME shape +
    # accent + colour the map uses for that type. Two rows:
    #   row 1: own building chips + own unit chips (in that order so
    #          the bigger silhouettes anchor the left edge).
    #   row 2: enemy chips + HP indicator swatch + ore swatch +
    #          objective ring swatch + help text.
    # When the strip runs out of width the surplus types are dropped
    # silently (the image stays legible — the model just sees fewer
    # legend entries; the actors are still on the map).
    if legend:
        ly = h * cp
        draw.rectangle([0, ly, w * cp, ly + legend_h], fill=(24, 24, 30))
        lfont = _minimap_font(max(11, int(cp * 0.62)))
        sample = cp
        m = sample * 0.16

        def _chip(x: int, row_y: int, label: str,
                  draw_swatch_fn) -> int:
            """Draw a swatch + label, return new x cursor."""
            draw_swatch_fn(x + m, row_y + m,
                           x + sample - m, row_y + sample - m)
            tx = x + sample + int(cp * 0.14)
            try:
                draw.text((tx, row_y + sample * 0.22), label,
                          fill=(235, 235, 245), font=lfont)
                tw = draw.textlength(label, font=lfont)
            except Exception:  # noqa: BLE001
                tw = len(label) * cp * 0.45
            return int(tx + tw + cp * 0.55)

        # Split entries into building / unit chips so the row composes
        # in a predictable order — buildings first (bigger silhouettes
        # anchor the eye), then units.
        own_b = [(t, s, a) for t, (s, a, isb) in own_types.items() if isb]
        own_u = [(t, s, a) for t, (s, a, isb) in own_types.items() if not isb]
        enemy_any = [(t, s, a, isb) for t, (s, a, isb) in enemy_types.items()]
        own_b.sort()
        own_u.sort()
        enemy_any.sort()

        # Row 1 — own actors.
        x = int(cp * 0.4)
        row1_y = ly + int(cp * 0.06)
        max_x = w * cp - int(cp * 0.4)
        for tname, shape, accent in own_b + own_u:
            if x > max_x - sample * 4:
                break
            def _sw(x0, y0, x1, y1, _s=shape, _a=accent):
                _draw_shape(draw, x0, y0, x1, y1, _s, _OWN)
                if _a:
                    _draw_accent(draw, x0, y0, x1, y1, _a, "own")
            x = _chip(x, row1_y, tname, _sw)

        # Row 2 — enemy actors, then the universal swatches.
        x = int(cp * 0.4)
        row2_y = ly + int(cp * 1.10)
        for tname, shape, accent, isb in enemy_any:
            if x > max_x - sample * 4:
                break
            def _sw(x0, y0, x1, y1, _s=shape, _a=accent, _isb=isb):
                col = _ENEMY_BLD if _isb else _ENEMY
                _draw_shape(draw, x0, y0, x1, y1, _s, col)
                if _a:
                    _draw_accent(draw, x0, y0, x1, y1, _a, "enemy")
            x = _chip(x, row2_y, f"enemy {tname}", _sw)

        # HP indicator — a 50% bar above a generic icon swatch.
        def _hp_swatch(x0, y0, x1, y1):
            # background icon stays grey so the bar reads
            draw.rectangle([x0, y0 + (y1 - y0) * 0.4, x1, y1],
                           fill=(110, 110, 120),
                           outline=(15, 15, 18))
            bw = (x1 - x0) * 0.78
            bh = max(2, int((y1 - y0) * 0.18))
            bx0 = x0 + ((x1 - x0) - bw) / 2
            by0 = y0 + (y1 - y0) * 0.08
            draw.rectangle([bx0, by0, bx0 + bw, by0 + bh],
                           fill=(28, 28, 32),
                           outline=(10, 10, 12))
            draw.rectangle([bx0, by0, bx0 + bw * 0.5, by0 + bh],
                           fill=_hp_color(0.5))
        if x < max_x - sample * 3:
            x = _chip(x, row2_y, "HP bar (damaged)", _hp_swatch)

        # Ore swatch — brown filled square.
        def _ore_swatch(x0, y0, x1, y1):
            pad = (x1 - x0) * 0.12
            draw.rectangle([x0 + pad, y0 + pad, x1 - pad, y1 - pad],
                           fill=_ORE, outline=(15, 15, 18))
        if x < max_x - sample * 2.5:
            x = _chip(x, row2_y, "ore", _ore_swatch)

        # Objective ring — yellow circle outline.
        def _obj_swatch(x0, y0, x1, y1):
            w_ = max(2, int((x1 - x0) * 0.18))
            draw.ellipse([x0, y0, x1, y1], outline=_OBJECTIVE, width=w_)
        if x < max_x - sample * 2.5:
            x = _chip(x, row2_y, "objective", _obj_swatch)

        # Trailing help text (third row).
        help_txt = (
            "shape = type, hue = side (green=you, red=enemy)   "
            "HP bar shows damaged actors only   "
            "labels = unit ids (pass to tools)"
            if unit_labels else
            "shape = type, hue = side (green=you, red=enemy)   "
            "HP bar shows damaged actors only   "
            "number = stack, white box = selected, arrow = order"
        )
        try:
            draw.text(
                (int(cp * 0.4), ly + int(cp * 2.18)), help_txt,
                fill=(200, 202, 212), font=lfont,
            )
        except Exception:  # noqa: BLE001
            pass

    return img
