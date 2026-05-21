"""Render a vision minimap for the planning phase.

Produces a small PNG image (~448x222, ~96 vision tokens) showing:
- Actual terrain from the base map (map.png)
- Visibility layers: visible (bright), fog of war (dimmed), unexplored (dark)
- Own units (cyan circles), enemy units (red circles), enemy buildings (red squares)
- Coordinate grid and compact legend

The image is returned as a base64-encoded PNG for injection into the
OpenAI-compatible vision API (SGLang/vLLM).
"""

from __future__ import annotations

import base64
import io
import logging

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
# Use Figure() OO API instead of pyplot — pyplot's global figure manager is
# NOT thread-safe, which prevents off-loading rendering from the event loop.
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Buildings from OpenRA game data
BUILDINGS = {
    "fact", "powr", "apwr", "tent", "barr", "proc", "weap", "dome",
    "fix", "hpad", "afld", "spen", "syrd", "pbox", "hbox", "gun",
    "ftur", "tsla", "agun", "sam", "gap", "iron", "mslo", "atek", "stek",
    "kenn", "silo",
}

# Visibility brightness multipliers
VIS_BRIGHT = 1.0    # Currently visible (unit line of sight)
VIS_FOG = 0.40       # Previously explored, no current vision
VIS_UNEXPLORED = 0.08  # Never seen

# Unit vision radius in cells
VISION_RADIUS = 10

# Supersampling: render at 2x, downsample with LANCZOS
RENDER_SCALE = 2
TARGET_WIDTH = 448


def _blur_2d(arr: np.ndarray, sigma: float = 1.5, size: int = 7) -> np.ndarray:
    """Simple separable gaussian blur without scipy dependency."""
    x = np.arange(size) - size // 2
    k = np.exp(-x**2 / (2 * sigma**2))
    k /= k.sum()
    r = np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), 1, arr)
    return np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, r)


def _parse_ascii_minimap(
    ascii_minimap: str, map_width: int, map_height: int
) -> np.ndarray:
    """Parse ASCII minimap to get explored mask at full map resolution.

    Characters: # = unexplored, everything else = explored.
    The ASCII grid is downsampled by scale = ceil(map_width / 28).

    Returns:
        Boolean array (map_height, map_width) — True = explored.
    """
    lines = [l for l in ascii_minimap.strip().split("\n") if l.strip()]
    # Skip header lines (e.g. "Map (28x14, 1cell=4x4):")
    grid_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and all(c in "#.@!X~$B " for c in stripped):
            grid_lines.append(stripped)

    if not grid_lines:
        return np.zeros((map_height, map_width), dtype=bool)

    grid_h = len(grid_lines)
    grid_w = max(len(l) for l in grid_lines)
    scale_x = max(1, map_width // grid_w) if grid_w > 0 else 1
    scale_y = max(1, map_height // grid_h) if grid_h > 0 else 1

    explored = np.zeros((map_height, map_width), dtype=bool)
    for gy, line in enumerate(grid_lines):
        for gx, ch in enumerate(line):
            if ch != "#":
                # Mark the corresponding map cells as explored
                y0 = gy * scale_y
                x0 = gx * scale_x
                y1 = min(y0 + scale_y, map_height)
                x1 = min(x0 + scale_x, map_width)
                explored[y0:y1, x0:x1] = True

    return explored


def _compute_visible_mask(
    own_units: list[dict], map_width: int, map_height: int
) -> np.ndarray:
    """Compute currently visible cells from own unit positions."""
    visible = np.zeros((map_height, map_width), dtype=bool)
    r = VISION_RADIUS
    for u in own_units:
        cx = u.get("cell_x", 0)
        cy = u.get("cell_y", 0)
        y_lo = max(0, cy - r)
        y_hi = min(map_height, cy + r + 1)
        x_lo = max(0, cx - r)
        x_hi = min(map_width, cx + r + 1)
        for y in range(y_lo, y_hi):
            for x in range(x_lo, x_hi):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    visible[y, x] = True
    return visible


def render_minimap(
    terrain_png: bytes,
    map_width: int,
    map_height: int,
    bounds_x: int,
    bounds_y: int,
    own_units: list[dict],
    enemy_units: list[dict],
    ascii_minimap: str,
    output_width: int = TARGET_WIDTH,
) -> str | None:
    """Render a vision minimap and return base64-encoded PNG.

    Args:
        terrain_png: Raw bytes of map.png from the .oramap file.
        map_width: Full map width in cells.
        map_height: Full map height in cells.
        bounds_x: Playable area X offset.
        bounds_y: Playable area Y offset.
        own_units: List of own unit dicts with cell_x, cell_y, type.
        enemy_units: List of visible enemy unit dicts with cell_x, cell_y, type.
        ascii_minimap: ASCII minimap string from game state.
        output_width: Target image width in pixels.

    Returns:
        Base64-encoded PNG string, or None on failure.
    """
    try:
        # Load terrain
        terrain_img = Image.open(io.BytesIO(terrain_png)).convert("RGB")
        pw, ph = terrain_img.size  # terrain image pixel dimensions
        terrain_arr = np.array(terrain_img).astype(float) / 255.0

        # Compute visibility masks in cell coordinates
        explored = _parse_ascii_minimap(ascii_minimap, map_width, map_height)
        visible = _compute_visible_mask(own_units, map_width, map_height)
        explored |= visible

        # Use full map (including borders) so terrain boundaries are visible
        playable_w = min(map_width - bounds_x, map_width)
        playable_h = min(map_height - bounds_y, map_height)
        explored_full = explored[:ph, :pw] if explored.shape[0] >= ph and explored.shape[1] >= pw else explored
        visible_full = visible[:ph, :pw] if visible.shape[0] >= ph and visible.shape[1] >= pw else visible

        # Resize visibility masks to match terrain image pixel dimensions
        if explored_full.shape != (ph, pw):
            explored_full = np.array(Image.fromarray(explored_full).resize((pw, ph), Image.NEAREST))
            visible_full = np.array(Image.fromarray(visible_full).resize((pw, ph), Image.NEAREST))

        # Smooth edges
        explored_s = np.clip(_blur_2d(explored_full.astype(float), sigma=1.5, size=7), 0, 1)
        visible_s = np.clip(_blur_2d(visible_full.astype(float), sigma=1.5, size=7), 0, 1)

        # Composite terrain with visibility (vectorized)
        brightness = VIS_UNEXPLORED * (1 - explored_s) + VIS_FOG * explored_s
        brightness = brightness * (1 - visible_s) + VIS_BRIGHT * visible_s
        # Ensure terrain borders (water/cliffs) are always visible — detect by
        # checking if the terrain pixel is distinctly different from grass.
        # Water/cliff pixels are blue-ish (high B, low G), grass is green-ish.
        _is_water = terrain_arr[..., 2] > terrain_arr[..., 1]  # blue > green
        brightness = np.where(_is_water, np.maximum(brightness, VIS_FOG), brightness)
        composite = terrain_arr * brightness[..., np.newaxis]

        # Render with matplotlib at 2x for supersampling
        render_dpi = 192 * RENDER_SCALE
        fig_w = 3.5
        fig_h = fig_w * ph / pw  # maintain aspect ratio
        # OO API (thread-safe, no global figure manager)
        fig = Figure(figsize=(fig_w, fig_h), dpi=render_dpi)
        ax = fig.add_subplot(1, 1, 1)
        bg = "#0a0a0f"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)

        ax.imshow(
            composite,
            extent=[0, pw, ph, 0],
            interpolation="bilinear",
            aspect="auto",
        )

        # Grid
        for x in range(0, map_width + 1, 20):
            ax.axvline(x, color="white", alpha=0.15, linewidth=0.4)
        for y in range(0, map_height + 1, 10):
            ax.axhline(y, color="white", alpha=0.15, linewidth=0.4)

        # Plot own units — cyan circles with glow.
        # Halted-unreachable units (unit halted on a bad target — pathfinding
        # failed repeatedly) get a YELLOW X overlay so the model can spot
        # them clearly on the minimap and understand they need a new target.
        for u in own_units:
            ux, uy = u.get("cell_x", 0), u.get("cell_y", 0)
            is_halted = bool(u.get("halted_unreachable"))
            ax.plot(ux, uy, "o", color="#00b8d4", markersize=8, alpha=0.3, zorder=9)
            ax.plot(
                ux, uy, "o", color="#00e5ff", markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=10,
            )
            if is_halted:
                # Yellow X overlay marking the unit as halted/unreachable.
                ax.plot(
                    ux, uy, marker="x", color="#ffe600", markersize=8,
                    markeredgewidth=1.5, zorder=11,
                )

        # Plot enemy units — red circles/squares with glow
        for u in enemy_units:
            ux, uy = u.get("cell_x", 0), u.get("cell_y", 0)
            utype = u.get("type", "").lower()
            is_bldg = utype in BUILDINGS
            marker = "s" if is_bldg else "o"
            ms = 6 if is_bldg else 5
            ax.plot(
                ux, uy, marker, color="#ff1744", markersize=ms + 3,
                alpha=0.25, zorder=9,
            )
            ax.plot(
                ux, uy, marker, color="#ff1744", markersize=ms,
                markeredgecolor="white", markeredgewidth=0.5, zorder=10,
            )

        # Show the FULL map including water/cliff borders — not just playable area.
        # This lets the model see terrain boundaries clearly.
        _x_max = map_width
        _y_max = map_height
        ax.set_xlim(0, _x_max)
        ax.set_ylim(_y_max, 0)
        # Ticks: evenly spaced within playable area + boundary values
        _xticks = [x for x in range(0, _x_max + 1, 20) if x <= _x_max]
        if _xticks[-1] != _x_max:
            _xticks.append(_x_max)
        _yticks = [y for y in range(0, _y_max + 1, 10) if y <= _y_max]
        if _yticks[-1] != _y_max:
            _yticks.append(_y_max)
        ax.set_xticks(_xticks)
        ax.set_yticks(_yticks)
        ax.tick_params(
            axis="both", colors="#8899aa", labelsize=6,
            length=2, width=0.4, pad=1,
        )
        for spine in ax.spines.values():
            spine.set_color("#2a3a50")
            spine.set_linewidth(0.5)

        # Compact legend — units + terrain
        legend_elements = [
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#00e5ff",
                markersize=5, label="Own", linestyle="None",
            ),
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="#ff1744",
                markersize=5, label="Enemy", linestyle="None",
            ),
            Line2D(
                [0], [0], marker="x", color="#ffe600", markersize=5,
                label="Halted", linestyle="None", markeredgewidth=1.5,
            ),
            Line2D(
                [0], [0], marker="s", color="w", markerfacecolor="#50a03c",
                markersize=5, label="Land", linestyle="None",
            ),
            Line2D(
                [0], [0], marker="s", color="w", markerfacecolor="#1e3c78",
                markersize=5, label="Water", linestyle="None",
            ),
            Line2D(
                [0], [0], marker="s", color="w", markerfacecolor="#6b5b3a",
                markersize=5, label="Cliff", linestyle="None",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", fontsize=5,
            framealpha=0.85, facecolor="#0a0a0f", edgecolor="#2a3a50",
            labelcolor="#ccddee", handletextpad=0.3, borderpad=0.3,
            columnspacing=0.6, ncol=6,
        )

        fig.tight_layout(pad=0.3)

        # Render to buffer
        buf = io.BytesIO()
        fig.savefig(
            buf, dpi=render_dpi, bbox_inches="tight",
            facecolor=bg, pad_inches=0.03, format="png",
        )
        # No plt.close needed — Figure is local, no global state to release

        # LANCZOS downsample to target size
        buf.seek(0)
        hi_res = Image.open(buf)
        scale = output_width / hi_res.width
        target_h = int(hi_res.height * scale)
        final = hi_res.resize((output_width, target_h), Image.LANCZOS)

        # Encode as base64
        out_buf = io.BytesIO()
        final.save(out_buf, format="PNG", optimize=True)
        b64 = base64.b64encode(out_buf.getvalue()).decode("ascii")

        logger.info(
            "Rendered minimap: %dx%d, %d bytes, ~%d vision tokens",
            final.width, final.height,
            len(out_buf.getvalue()),
            (final.width * final.height) // (32 * 32),
        )
        return b64

    except Exception as e:
        logger.warning("Minimap render failed: %s", e)
        return None
