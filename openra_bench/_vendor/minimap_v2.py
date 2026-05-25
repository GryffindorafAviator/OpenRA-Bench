"""Clean minimap renderer for the Rust env probes.

Replaces the production renderer with something the user can actually read:

- Terrain: native-resolution bitmap (1 px per cell), upscaled with NEAREST so
  cell boundaries stay crisp.
- Three-tier fog: BRIGHT (currently visible), DIM (previously explored, no
  current vision), DARK (never seen).
- Unit markers drawn at cell centers with clear contrast.
- Grid lines + axis labels matching the cell coordinate system.

API:
    render(obs, scenario_terrain_png_bytes, map_width, map_height,
           bounds=(bx, by, bw, bh), explored_history, output_pixels_per_cell=8,
           sight_radius=8) -> bytes (PNG)

`explored_history` is mutated in place to accumulate seen cells.
"""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Brightness multipliers
BRIGHT = 1.00   # currently visible
DIM = 0.45      # previously explored
DARK = 0.10     # never seen

# Output styling
GRID_COLOR = (255, 255, 255, 90)
PLAYABLE_BORDER_COLOR = (255, 145, 0, 220)
LABEL_COLOR = (255, 200, 60, 255)
BG_COLOR = (12, 12, 18, 255)
OUTLINE_COLOR = (255, 255, 255, 230)

# Per-unit-type styling. shape ∈ {"circle", "square", "triangle", "diamond", "pentagon", "x"}.
# Color is the fill (RGBA). Own units use cyan family; enemies use red family.
UNIT_STYLE_OWN: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    # tanks
    "2tnk": ("square",   (40, 200, 230, 255)),     # medium tank — solid square cyan
    "1tnk": ("square",   (90, 230, 255, 255)),     # light tank — paler square
    "3tnk": ("square",   (20, 160, 200, 255)),     # heavy tank — darker square
    "4tnk": ("square",   (10, 120, 170, 255)),     # mammoth — even darker square
    # economy: tridown so the harv is immediately distinguishable by
    # SHAPE. Hue stays in the OWN cyan family — gold is reserved for
    # ore-patch tiles so own/enemy harv don't collide on colour.
    "harv": ("tridown",  (100, 220, 220, 255)),    # own harvester — cyan-teal tridown
    # support / scout
    "jeep": ("triangle", (180, 240, 255, 255)),    # scout — triangle
    "apc":  ("pentagon", (0, 160, 200, 255)),      # APC — pentagon, deeper cyan
    "arty": ("diamond",  (160, 220, 250, 255)),    # arty — diamond (cyan-blue)
    "mnly": ("diamond",  (180, 200, 220, 255)),    # mine layer
    # infantry — all cyan family (own-side contract)
    "e1":   ("circle",   (90, 200, 255, 255)),     # rifle
    "e2":   ("circle",   (60, 150, 200, 255)),     # grenadier
    "e3":   ("circle",   (130, 240, 255, 255)),    # rocket
    "e4":   ("circle",   (180, 220, 240, 255)),    # flamer
    "e6":   ("circle",   (160, 220, 200, 255)),    # engineer
    "spy":  ("x",        (200, 220, 255, 255)),    # spy
    "tanya":("x",        (220, 240, 255, 255)),    # tanya
    "thf":  ("x",        (180, 200, 220, 255)),    # thief
    "dog":  ("x",        (220, 240, 200, 255)),    # dog
    "medi": ("circle",   (220, 240, 240, 255)),    # medic
    "mech": ("circle",   (160, 220, 240, 255)),    # mechanic
    # fallback
    "?":    ("circle",   (0, 220, 255, 255)),
}

UNIT_STYLE_ENEMY: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    "2tnk": ("square",   (255, 60, 60, 255)),
    "1tnk": ("square",   (255, 110, 110, 255)),
    "3tnk": ("square",   (200, 30, 30, 255)),
    "4tnk": ("square",   (160, 20, 20, 255)),
    # Enemy harvester — RED tridown (not gold). Both sides keep the
    # side-colour contract: own=cyan, enemy=red. Harvester-ness is
    # signalled by SHAPE; side is signalled by COLOUR.
    "harv": ("tridown",  (255, 110, 110, 255)),    # enemy harv — red tridown
    "jeep": ("triangle", (255, 160, 160, 255)),
    "apc":  ("pentagon", (200, 30, 30, 255)),
    "arty": ("diamond",  (255, 120, 100, 255)),
    "mnly": ("diamond",  (140, 80, 80, 255)),
    "e1":   ("circle",   (255, 80, 80, 255)),
    "e2":   ("circle",   (220, 50, 50, 255)),
    "e3":   ("circle",   (255, 130, 130, 255)),
    "e4":   ("circle",   (240, 100, 100, 255)),
    "e6":   ("circle",   (200, 80, 100, 255)),
    "spy":  ("x",        (255, 100, 100, 255)),
    "tanya":("x",        (255, 60, 60, 255)),
    "thf":  ("x",        (220, 140, 140, 255)),
    "dog":  ("x",        (255, 160, 100, 255)),
    "?":    ("circle",   (255, 60, 60, 255)),
}

# Own buildings — cyan family (mirrors UNIT_STYLE_OWN's hue) so the model
# can tell them apart from enemy buildings (warm reds/yellows) at a
# glance. All use `filled_square` for the structure-vs-unit distinction.
BUILDING_STYLE_OWN: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    # production / key targets
    "fact": ("filled_square", (90, 230, 255, 255)),  # construction yard — bright cyan
    "proc": ("filled_square", (60, 200, 200, 255)),  # refinery — cyan-teal (NOT gold)
    "weap": ("filled_square", (40, 200, 230, 255)),  # war factory — solid cyan
    "tent": ("filled_square", (130, 220, 240, 255)), # barracks (allies) — lighter cyan
    "barr": ("filled_square", (130, 220, 240, 255)), # barracks (soviet)
    "hpad": ("filled_square", (160, 220, 230, 255)),
    "afld": ("filled_square", (160, 220, 230, 255)),
    "spen": ("filled_square", (100, 180, 220, 255)),
    "syrd": ("filled_square", (100, 180, 220, 255)),
    "atek": ("filled_square", (200, 240, 255, 255)),
    "stek": ("filled_square", (200, 240, 255, 255)),
    # scenery / power
    "powr": ("filled_square", (140, 200, 220, 255)),
    "apwr": ("filled_square", (120, 180, 200, 255)),
    "silo": ("filled_square", (180, 200, 200, 255)),
    "fix":  ("filled_square", (160, 200, 220, 255)),
    "dome": ("filled_square", (200, 230, 240, 255)),
    # defenses (own side — cyan-tinted)
    "gun":  ("filled_square", (100, 180, 230, 255)),
    "pbox": ("filled_square", (90, 170, 220, 255)),
    "hbox": ("filled_square", (90, 170, 220, 255)),
    "ftur": ("filled_square", (90, 170, 220, 255)),
    "agun": ("filled_square", (110, 190, 240, 255)),
    "sam":  ("filled_square", (110, 190, 240, 255)),
    "tsla": ("filled_square", (180, 130, 240, 255)),
    "?":    ("filled_square", (90, 200, 230, 255)),
}

# Static enemy buildings get a distinct shape (filled square outline) and
# warmer red so the model can tell defensive turrets / production
# structures apart from mobile enemy units at a glance.
BUILDING_STYLE_ENEMY: dict[str, tuple[str, tuple[int, int, int, int]]] = {
    # defenses (shoot back)
    "gun":  ("filled_square", (255, 90, 0, 255)),    # turret — orange-red
    "tsla": ("filled_square", (200, 100, 255, 255)), # tesla — purple
    "pbox": ("filled_square", (255, 90, 0, 255)),
    "hbox": ("filled_square", (255, 90, 0, 255)),
    "ftur": ("filled_square", (255, 90, 0, 255)),
    "agun": ("filled_square", (255, 130, 0, 255)),
    "sam":  ("filled_square", (255, 130, 0, 255)),
    # production / key targets (must-be-destroyed) — all red family,
    # NEVER gold (gold is reserved for ore tiles only). Refinery (proc)
    # used to render yellow, identical to own gold proc — broke the
    # side-colour contract.
    "fact": ("filled_square", (255, 80, 80, 255)),   # construction yard — bright red
    "proc": ("filled_square", (200, 50, 50, 255)),   # refinery — dark red
    "weap": ("filled_square", (220, 60, 60, 255)),   # war factory
    "tent": ("filled_square", (255, 120, 100, 255)), # barracks
    "barr": ("filled_square", (255, 120, 100, 255)),
    # scenery
    "powr": ("filled_square", (160, 160, 160, 255)),
    "apwr": ("filled_square", (140, 140, 140, 255)),
    "silo": ("filled_square", (180, 180, 180, 255)),
    "?":    ("filled_square", (255, 100, 100, 255)),
}

# Ore / resource cells — painted as a TERRAIN-STYLE TILE FILL (the
# whole cell goes gold) rather than as a small dot. Reads as map
# content (alongside water/grass) instead of as an actor — which is
# the operator's mental model: ore is part of the world, not a unit
# you click. Pre-fix this surfaced as a small filled circle that the
# model could easily mistake for a unit marker.
#
# RESOURCE_COLOR  — the legend-swatch + actor-style debug colour (RGBA).
# RESOURCE_TERRAIN_RGB — the per-tile fill applied at terrain-composite
# time (RGB only; brightness is multiplied separately for fog gating).
RESOURCE_COLOR = (220, 175, 50, 255)        # legend swatch — warm gold
RESOURCE_TERRAIN_RGB = (200, 160, 50)       # cell-tile gold fill


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common fonts; fall back to PIL default."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _compute_visible_mask(
    own_units: list[tuple[int, int]],
    map_width: int,
    map_height: int,
    sight_radius: int,
) -> np.ndarray:
    """Boolean mask (h, w): True where cells are within sight radius of any own unit."""
    mask = np.zeros((map_height, map_width), dtype=bool)
    r2 = sight_radius * sight_radius
    for cx, cy in own_units:
        x_lo = max(0, cx - sight_radius)
        x_hi = min(map_width, cx + sight_radius + 1)
        y_lo = max(0, cy - sight_radius)
        y_hi = min(map_height, cy + sight_radius + 1)
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    mask[y, x] = True
    return mask


def _draw_unit_marker(
    draw: ImageDraw.ImageDraw,
    cx_px: int,
    cy_px: int,
    radius: int,
    shape: str,
    fill: tuple[int, int, int, int],
) -> None:
    """Draw a single unit marker of the given shape centered at (cx_px, cy_px)."""
    r = radius
    if shape == "square":
        draw.rectangle(
            [(cx_px - r, cy_px - r), (cx_px + r, cy_px + r)],
            fill=fill, outline=OUTLINE_COLOR, width=1,
        )
    elif shape == "filled_square":
        # Buildings: bigger filled square with thick black outline so it
        # reads as a structure (not a unit) at all zoom levels.
        big_r = r + 2
        draw.rectangle(
            [(cx_px - big_r, cy_px - big_r), (cx_px + big_r, cy_px + big_r)],
            fill=fill, outline=(0, 0, 0, 255), width=2,
        )
    elif shape == "triangle":
        draw.polygon(
            [(cx_px, cy_px - r), (cx_px - r, cy_px + r), (cx_px + r, cy_px + r)],
            fill=fill, outline=OUTLINE_COLOR,
        )
    elif shape == "diamond":
        draw.polygon(
            [(cx_px, cy_px - r), (cx_px + r, cy_px),
             (cx_px, cy_px + r), (cx_px - r, cy_px)],
            fill=fill, outline=OUTLINE_COLOR,
        )
    elif shape == "pentagon":
        import math as _m
        pts = []
        for i in range(5):
            theta = _m.pi / 2 + i * 2 * _m.pi / 5
            pts.append((cx_px + r * _m.cos(theta), cy_px - r * _m.sin(theta)))
        draw.polygon(pts, fill=fill, outline=OUTLINE_COLOR)
    elif shape == "x":
        draw.line([(cx_px - r, cy_px - r), (cx_px + r, cy_px + r)], fill=fill, width=2)
        draw.line([(cx_px + r, cy_px - r), (cx_px - r, cy_px + r)], fill=fill, width=2)
    elif shape == "tridown":
        # Inverted triangle — apex down. Used for harvesters so they are
        # IMMEDIATELY distinguishable from tanks (square) and jeeps
        # (apex-up triangle) on the model's vision channel.
        draw.polygon(
            [(cx_px - r, cy_px - r), (cx_px + r, cy_px - r), (cx_px, cy_px + r)],
            fill=fill, outline=OUTLINE_COLOR,
        )
    else:  # circle (default)
        draw.ellipse(
            [(cx_px - r, cy_px - r), (cx_px + r, cy_px + r)],
            fill=fill, outline=OUTLINE_COLOR, width=1,
        )


def render(
    *,
    obs: dict[str, Any],
    terrain_png_bytes: bytes,
    map_width: int,
    map_height: int,
    bounds: tuple[int, int, int, int],
    explored_history: set[tuple[int, int]],
    sight_radius: int = 8,
    pixels_per_cell: int = 8,
    own_unit_types: dict[str, str] | None = None,
    enemy_unit_types: dict[str, str] | None = None,
    draw_grid: bool = True,
    draw_axis_labels: bool = True,
    draw_playable_border: bool = True,
    draw_legend: bool = True,
    crop_to_playable: bool = True,
) -> bytes:
    """Render a fog-of-war minimap and return PNG bytes.

    Args:
        obs: observation dict from openra_train env (unit_positions, enemy_positions).
        terrain_png_bytes: native-resolution terrain bitmap (e.g. 128x40 RGB).
        map_width, map_height: cell dimensions (e.g. 128, 40).
        bounds: (bx, by, bw, bh) playable area in cells.
        explored_history: set of (x, y) cell coords ever seen; mutated in place.
        sight_radius: cells within this radius of any own unit count as currently visible.
        pixels_per_cell: scale-up factor for output.
        draw_grid: if False, suppress the every-20-cells grid lines (use
            for transfer-learning training where the model should not
            rely on the grid as a coordinate-counting cue).
        draw_axis_labels: if False, suppress the 0/20/40/.../127 numeric
            labels on the x and y axes (same rationale).
        draw_playable_border: if False, suppress the orange-dashed
            playable-area rectangle.
        draw_legend: if False, suppress the bottom legend strip.
    """
    # ---- Load terrain bitmap and ensure size matches map cells ----
    terrain_img = Image.open(io.BytesIO(terrain_png_bytes)).convert("RGB")
    if terrain_img.size != (map_width, map_height):
        terrain_img = terrain_img.resize((map_width, map_height), Image.NEAREST)
    terrain = np.asarray(terrain_img, dtype=np.float32) / 255.0  # (h, w, 3) in [0,1]

    # ---- Build own/enemy lists in (x, y) cell coords ----
    unit_positions = obs.get("unit_positions") or {}
    if isinstance(unit_positions, list):
        unit_positions = {p[0]: p[1] for p in unit_positions}
    own_cells: list[tuple[int, int]] = []
    own_dots: list[tuple[int, int, str]] = []
    for uid_str, pos in unit_positions.items():
        cx = int(pos.get("cell_x") if isinstance(pos, dict) else pos[0])
        cy = int(pos.get("cell_y") if isinstance(pos, dict) else pos[1])
        own_cells.append((cx, cy))
        own_dots.append((cx, cy, uid_str))

    enemy_dots: list[tuple[int, int, str]] = []
    for ep in obs.get("enemy_positions") or []:
        ex = int(ep.get("cell_x", 0))
        ey = int(ep.get("cell_y", 0))
        enemy_dots.append((ex, ey, str(ep.get("id", "?"))))
    # Enemy buildings: same world-space, drawn with a distinct marker so
    # the model (and humans browsing the viewer) can tell at a glance
    # which dots are mobile units vs static structures (defenses + key
    # targets). Strategy scenarios place ONLY buildings on the enemy
    # side, so without this layer the minimap shows nothing where the
    # threat actually is.
    enemy_building_dots: list[tuple[int, int, str]] = []
    for eb in obs.get("enemy_buildings_summary") or []:
        ex = int(eb.get("cell_x", 0))
        ey = int(eb.get("cell_y", 0))
        enemy_building_dots.append((ex, ey, str(eb.get("id", "?"))))

    # Own buildings: mirror of enemy_buildings_summary so the model can
    # tell their own base apart from mobile units (filled square with a
    # cyan-family tint). Without this layer, own buildings are either
    # absent from the image (pre-57a9440e) or merged into unit_positions
    # and drawn as a single small unit dot — both bad for spatial
    # reasoning when the base footprint matters (e.g. "is the proc near
    # the ore?").
    own_building_dots: list[tuple[int, int, str]] = []
    for ob in obs.get("own_buildings_summary") or []:
        ox = int(ob.get("cell_x", 0))
        oy = int(ob.get("cell_y", 0))
        own_building_dots.append((ox, oy, str(ob.get("id", "?"))))

    # ---- Compute visible + explored masks at cell resolution ----
    visible = _compute_visible_mask(own_cells, map_width, map_height, sight_radius)
    # Prefer the engine's per-tick `explored_cells` (ground truth — captures
    # cells that units transited *between* briefings, which a snapshot-based
    # vision-mask reveal misses and produces "disconnected green blobs").
    # Fall back to the legacy briefing-snapshot history if the engine field
    # is absent (older envs).
    engine_explored = obs.get("explored_cells")
    if engine_explored:
        for cell in engine_explored:
            if isinstance(cell, (list, tuple)) and len(cell) == 2:
                explored_history.add((int(cell[0]), int(cell[1])))
    else:
        for y in range(map_height):
            for x in range(map_width):
                if visible[y, x]:
                    explored_history.add((x, y))
    # Build explored mask from history
    explored = np.zeros((map_height, map_width), dtype=bool)
    for (x, y) in explored_history:
        if 0 <= x < map_width and 0 <= y < map_height:
            explored[y, x] = True

    # ---- Inject ore-patch tint into the TERRAIN layer ----
    # Ore is a property of the WORLD, not an actor — so we paint ore
    # cells at the terrain composite stage and let the same brightness
    # multiply gate them through the three-tier fog. Units (drawn
    # later) overlay on top exactly as they do over grass / water.
    # Gated by `explored_history` (cumulative) so ore the agent has
    # never seen stays hidden in the dark band.
    #
    # Capture the set of ore cells so the legend grass-swatch sampler
    # downstream can avoid landing on an ore tile (which used to make
    # the legend label gold cells as "grass" when the centre patch sat
    # at the playable midpoint — the 1v1-macro centre patch case).
    ore_rgb = np.asarray(RESOURCE_TERRAIN_RGB, dtype=np.float32) / 255.0
    ore_set: set[tuple[int, int]] = set()
    for rc in obs.get("resource_cells") or []:
        if not isinstance(rc, dict):
            continue
        rx = int(rc.get("cell_x", -1))
        ry = int(rc.get("cell_y", -1))
        if not (0 <= rx < map_width and 0 <= ry < map_height):
            continue
        ore_set.add((rx, ry))
        if (rx, ry) not in explored_history:
            continue
        terrain[ry, rx] = ore_rgb

    # ---- Apply 3-tier brightness ----
    brightness = np.full((map_height, map_width), DARK, dtype=np.float32)
    brightness[explored] = DIM
    brightness[visible] = BRIGHT
    composite = terrain * brightness[..., np.newaxis]
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)

    # ---- Crop to playable bounds if requested ----
    bx, by, bw, bh = bounds
    if crop_to_playable:
        # Clip bounds to valid range first.
        cx0 = max(0, min(bx, map_width))
        cy0 = max(0, min(by, map_height))
        cx1 = max(cx0, min(bx + bw, map_width))
        cy1 = max(cy0, min(by + bh, map_height))
        composite_uint8 = composite_uint8[cy0:cy1, cx0:cx1, :]
        disp_w = cx1 - cx0
        disp_h = cy1 - cy0
        x_origin, y_origin = cx0, cy0  # world cell at left/top edge of crop
    else:
        disp_w, disp_h = map_width, map_height
        x_origin, y_origin = 0, 0

    # ---- Upscale (NEAREST keeps cell boundaries crisp) ----
    cell_img = Image.fromarray(composite_uint8, mode="RGB")
    out_w = disp_w * pixels_per_cell
    out_h = disp_h * pixels_per_cell
    cell_img = cell_img.resize((out_w, out_h), Image.NEAREST)

    # ---- Add a margin for axis labels ----
    margin_x = 36
    margin_y = 22
    # Legend height is decided once we know how many types appear in
    # this frame (own + enemy units + enemy buildings). Each row is 16
    # pixels; the swatch+terrain header is one row; unit / building
    # rows wrap at ~7 entries each. Worst case 4 rows.
    legend_h = 16 * 5  # set after counting, reserved upper bound
    canvas_w = out_w + margin_x + 8
    canvas_h = out_h + margin_y + legend_h + 8
    canvas = Image.new("RGBA", (canvas_w, canvas_h), BG_COLOR)
    canvas.paste(cell_img, (margin_x, margin_y))

    draw = ImageDraw.Draw(canvas, "RGBA")
    label_font = _load_font(11)
    legend_font = _load_font(11)

    # Helper: world cell x/y → pixel x/y in the output canvas.
    def _wx(cx: int) -> int:
        return margin_x + (cx - x_origin) * pixels_per_cell

    def _wy(cy: int) -> int:
        return margin_y + (cy - y_origin) * pixels_per_cell

    # x/y world-cell ranges currently displayed.
    disp_x_lo, disp_x_hi = x_origin, x_origin + disp_w
    disp_y_lo, disp_y_hi = y_origin, y_origin + disp_h

    # ---- Grid lines every 20 cells (x), 10 cells (y), in world coords ----
    if draw_grid:
        # Find first world x divisible by 20 within the displayed range.
        first_x = ((disp_x_lo + 19) // 20) * 20
        for x in range(first_x, disp_x_hi + 1, 20):
            px = _wx(x)
            draw.line([(px, margin_y), (px, margin_y + out_h)], fill=GRID_COLOR, width=1)
        first_y = ((disp_y_lo + 9) // 10) * 10
        for y in range(first_y, disp_y_hi + 1, 10):
            py = _wy(y)
            draw.line([(margin_x, py), (margin_x + out_w, py)], fill=GRID_COLOR, width=1)

    # ---- Playable bounds rectangle (orange dashed) ----
    # Skip when the entire visible region IS the playable area (crop_to_playable).
    if draw_playable_border and not crop_to_playable:
        pb_x0 = _wx(bx)
        pb_y0 = _wy(by)
        pb_x1 = _wx(bx + bw)
        pb_y1 = _wy(by + bh)
        # Dashed manually
        dash_len = 6
        gap = 4
        def _dashed_line(p0, p1):
            x0, y0 = p0
            x1, y1 = p1
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length == 0:
                return
            ux, uy = dx / length, dy / length
            d = 0.0
            while d < length:
                d2 = min(d + dash_len, length)
                draw.line(
                    [(x0 + ux * d, y0 + uy * d), (x0 + ux * d2, y0 + uy * d2)],
                    fill=PLAYABLE_BORDER_COLOR,
                    width=2,
                )
                d = d2 + gap
        _dashed_line((pb_x0, pb_y0), (pb_x1, pb_y0))
        _dashed_line((pb_x1, pb_y0), (pb_x1, pb_y1))
        _dashed_line((pb_x1, pb_y1), (pb_x0, pb_y1))
        _dashed_line((pb_x0, pb_y1), (pb_x0, pb_y0))

    # ---- Axis labels (world coords across the displayed range) ----
    if draw_axis_labels:
        # x-axis: every 20 cells starting from a multiple of 20 within
        # [disp_x_lo, disp_x_hi-1]. Skip if it would land within ~3 cells
        # of the far-right label (avoids "80" + "81" overlap).
        first_x = ((disp_x_lo + 19) // 20) * 20
        last_x = disp_x_hi - 1
        for x in range(first_x, disp_x_hi, 20):
            if abs(x - last_x) < 3:
                continue
            draw.text((_wx(x) - 6, 4), str(x), fill=LABEL_COLOR, font=label_font)
        draw.text((_wx(last_x) - 8, 4), str(last_x),
                  fill=LABEL_COLOR, font=label_font)

        first_y = ((disp_y_lo + 9) // 10) * 10
        last_y = disp_y_hi - 1
        for y in range(first_y, disp_y_hi, 10):
            if abs(y - last_y) < 3:
                continue
            draw.text((4, _wy(y) - 6), str(y), fill=LABEL_COLOR, font=label_font)
        draw.text((4, _wy(last_y) - 6), str(last_y),
                  fill=LABEL_COLOR, font=label_font)

    # ---- Unit markers ----
    # When N units share a cell, jitter each marker on a small ring around
    # the cell center so every unit is individually visible. Smaller marker
    # radius when stacked so the cluster fits within ~1.5 cells.
    own_unit_types = own_unit_types or {}
    enemy_unit_types = enemy_unit_types or {}

    def _cell_offsets(n: int, ring_radius: float) -> list[tuple[float, float]]:
        """Return N (dx, dy) offsets distributed at the cell center.
        n=1 → centered. n=2-6 → single ring. n>6 → ring + center.
        """
        if n <= 1:
            return [(0.0, 0.0)]
        if n <= 6:
            return [
                (ring_radius * math.cos(2 * math.pi * i / n - math.pi / 2),
                 ring_radius * math.sin(2 * math.pi * i / n - math.pi / 2))
                for i in range(n)
            ]
        # >6: place 1 in center + remaining on outer ring
        offsets = [(0.0, 0.0)]
        outer = n - 1
        for i in range(outer):
            offsets.append((
                ring_radius * math.cos(2 * math.pi * i / outer - math.pi / 2),
                ring_radius * math.sin(2 * math.pi * i / outer - math.pi / 2),
            ))
        return offsets

    # Group by cell to compute jitter offsets
    def _draw_group(dots, type_map, style_map):
        by_cell: dict[tuple[int, int], list[tuple[int, int, str]]] = {}
        for d in dots:
            by_cell.setdefault((d[0], d[1]), []).append(d)
        for (cx, cy), members in by_cell.items():
            n = len(members)
            base_r = max(2, pixels_per_cell // 2 - max(0, n - 1))  # shrink with stack
            ring = pixels_per_cell * (0.55 if n <= 6 else 0.75)
            offsets = _cell_offsets(n, ring)
            cx_px_center = _wx(cx) + pixels_per_cell // 2
            cy_px_center = _wy(cy) + pixels_per_cell // 2
            for (mem, off) in zip(members, offsets):
                _, _, uid = mem
                utype = type_map.get(uid, "?")
                shape, fill = style_map.get(utype, style_map["?"])
                px = int(cx_px_center + off[0])
                py = int(cy_px_center + off[1])
                _draw_unit_marker(draw, px, py, base_r, shape, fill)

    # NOTE: ore cells are painted into the TERRAIN composite above
    # (search for `RESOURCE_TERRAIN_RGB`), not drawn here. That way
    # units overlay on ore exactly as they do on grass — a harvester
    # sitting on its patch shows the harv marker on a gold tile, not
    # buried under a gold dot. Side-colour contract preserved: ore is
    # terrain (gold), own actors are cyan, enemy actors are red.

    # Static structures first (largest), then mobile enemies, then own
    # units on top. Building type is taken from enemy_buildings_summary
    # entries (kept by the engine in a separate `type` field).
    enemy_building_types = {
        str(eb.get("id", "?")): eb.get("type", "?")
        for eb in (obs.get("enemy_buildings_summary") or [])
    }
    own_building_types = {
        str(ob.get("id", "?")): ob.get("type", "?")
        for ob in (obs.get("own_buildings_summary") or [])
    }
    _draw_group(enemy_building_dots, enemy_building_types, BUILDING_STYLE_ENEMY)
    _draw_group(own_building_dots, own_building_types, BUILDING_STYLE_OWN)
    _draw_group(enemy_dots, enemy_unit_types, UNIT_STYLE_ENEMY)
    _draw_group(own_dots, own_unit_types, UNIT_STYLE_OWN)

    # ---- Legend strip ----
    if draw_legend:
        lx = margin_x
        ly = margin_y + out_h + 6
        draw.text((lx, ly), "BRIGHT visible | DIM explored | DARK unexplored",
                  fill=(220, 220, 220, 230), font=legend_font)

        # Terrain swatches — sample one grass-typical cell from the
        # interior of the playable rect and one water-typical cell from
        # the impassable border. After hue-rotation theming (applied to
        # the whole canvas), these swatches still match the in-image
        # terrain colors because they're drawn pre-theming with the same
        # source pixels.
        #
        # Footgun fixed (2026-05-25): the grass sampler used to read
        # `terrain[centre]` UNCONDITIONALLY. After the ore-tile injection
        # block (above) overwrote ore cells with gold, the legend
        # sampler at the playable midpoint landed on the 1v1-macro
        # CENTRE ORE PATCH — so the swatch showed gold labelled
        # "grass". Skip ore-tinted cells via a spiral search outward
        # from the centre, fall back to the original cell if nothing
        # non-ore can be found within the playable rect.
        terrain_x = lx + 290
        bx_, by_, bw_, bh_ = bounds
        cx0 = bx_ + bw_ // 2
        cy0 = by_ + bh_ // 2

        def _is_grass_candidate(cx: int, cy: int) -> bool:
            if not (bx_ <= cx < bx_ + bw_ and by_ <= cy < by_ + bh_):
                return False
            return (cx, cy) not in ore_set

        grass_cx, grass_cy = cx0, cy0
        if not _is_grass_candidate(cx0, cy0):
            # outward ring search — bounded to half-playable to keep this O(1)
            found = False
            max_r = max(bw_, bh_) // 2
            for r in range(1, max_r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if abs(dx) != r and abs(dy) != r:
                            continue  # ring boundary only
                        if _is_grass_candidate(cx0 + dx, cy0 + dy):
                            grass_cx, grass_cy = cx0 + dx, cy0 + dy
                            found = True
                            break
                    if found: break
                if found: break
        grass_color = tuple(int(c) for c in terrain[grass_cy, grass_cx] * 255) + (255,)
        # Water: a corner just inside the impassable border (rush-hour
        # has water at y=0..2 and y=37..39)
        water_cy = max(0, by_ - 1)
        water_cx = bx_ + bw_ // 2
        water_color = tuple(int(c) for c in terrain[water_cy, water_cx] * 255) + (255,)
        # Draw small color swatches with labels
        sw_w, sw_h = 12, 10
        for label, color in [("grass", grass_color), ("water", water_color)]:
            draw.rectangle(
                [(terrain_x, ly + 1), (terrain_x + sw_w, ly + sw_h + 1)],
                fill=color, outline=OUTLINE_COLOR,
            )
            draw.text((terrain_x + sw_w + 4, ly), label,
                      fill=(220, 220, 220, 230), font=legend_font)
            terrain_x += sw_w + 38

        # ── Dynamic legend ──
        # Walk the actual dots present in this frame and emit one entry
        # per distinct type, using its real marker shape + colour. This
        # makes the legend exhaustive: every coloured square / dot on
        # the minimap can be decoded right here. Three rows are stacked:
        #   row 1: terrain swatches + visibility key (already drawn)
        #   row 2: agent units (cyan family)
        #   row 3: enemy units (red family)  +  enemy buildings (filled
        #          squares of various colours)
        #
        # Wraps at ~7 entries per row and grows downward; the canvas
        # height was reserved generously above.

        def _emit_row(items, y_offset, prefix=None):
            x = margin_x
            if prefix:
                draw.text((x, y_offset), prefix, fill=(180, 180, 180, 230), font=legend_font)
                x += 32
            for label, shape, fill in items:
                _draw_unit_marker(draw, x, y_offset + 7, 5, shape, fill)
                draw.text((x + 8, y_offset), label,
                          fill=(220, 220, 220, 230), font=legend_font)
                x += 50  # wider spacing for short type codes
                if x + 50 > canvas_w:
                    return  # silently truncate (very rare with ≤8 types/row)

        own_types_in_frame = sorted({own_unit_types.get(uid, "?")
                                     for _, _, uid in own_dots} - {"?"})
        own_bld_types_in_frame = sorted({own_building_types.get(uid, "?")
                                         for _, _, uid in own_building_dots} - {"?"})
        enemy_unit_types_in_frame = sorted({enemy_unit_types.get(uid, "?")
                                            for _, _, uid in enemy_dots} - {"?"})
        enemy_bld_types_in_frame = sorted({enemy_building_types.get(uid, "?")
                                           for _, _, uid in enemy_building_dots} - {"?"})

        # Own row: units first, then buildings (filled squares) so the
        # model can see the two visual languages side by side.
        if own_types_in_frame or own_bld_types_in_frame:
            row_items = [(t, *UNIT_STYLE_OWN.get(t, UNIT_STYLE_OWN["?"]))
                         for t in own_types_in_frame]
            for t in own_bld_types_in_frame:
                row_items.append((t, *BUILDING_STYLE_OWN.get(t, BUILDING_STYLE_OWN["?"])))
            _emit_row(row_items, ly + 16, prefix="own:")

        # Enemies + buildings on the same line if they fit; else wrap.
        enemy_row_items = []
        for t in enemy_unit_types_in_frame:
            enemy_row_items.append((t, *UNIT_STYLE_ENEMY.get(t, UNIT_STYLE_ENEMY["?"])))
        for t in enemy_bld_types_in_frame:
            enemy_row_items.append((t, *BUILDING_STYLE_ENEMY.get(t, BUILDING_STYLE_ENEMY["?"])))
        if enemy_row_items:
            _emit_row(enemy_row_items, ly + 32, prefix="enemy:")

        # Ore swatch — appended to the terrain row so the model has a
        # decoded reference colour for the gold dots scattered across
        # the map. Placed on the terrain row (row 1) to keep the legend
        # compact when no ore is visible (`resource_cells` empty); the
        # swatch is drawn unconditionally because the model needs to
        # know the colour ahead of first ore reveal.
        ore_x = lx + 290 + (12 + 38) * 2  # past grass + water swatches
        draw.ellipse(
            [(ore_x, ly + 2), (ore_x + 10, ly + 12)],
            fill=RESOURCE_COLOR, outline=(60, 50, 20, 255), width=1,
        )
        draw.text((ore_x + 14, ly), "ore",
                  fill=(220, 220, 220, 230), font=legend_font)

    # ---- Encode PNG ----
    buf = io.BytesIO()
    canvas.convert("RGB").save(buf, format="PNG", optimize=True)
    return buf.getvalue()
