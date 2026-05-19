"""Structured-fog text representation (text alternative to the PNG
minimap), for the text-vs-vision comparison.

`compute_unexplored_regions` + `format_structured_fog` are ported
**verbatim** from the training repo
(`scripts/run_one_game_rust_lmstudio.py`, ~L920–1010 — the v2-ablation
representation validated against Qwen3.5-9B, Finding 7 Test 4). Kept
byte-faithful so the text channel is identical to training's. Do not
hand-edit; re-port if upstream changes.
"""

from __future__ import annotations

from typing import Any


def compute_unexplored_regions(
    explored_cells, playable_bounds, max_regions: int = 8,
) -> list[dict[str, int]]:
    """4-connected flood-fill of the playable rect, treating cells in
    `explored_cells` as 'seen' and everything else as shroud. Returns a
    list of {x_lo, x_hi, y_lo, y_hi, cells} bounding boxes for each
    contiguous unexplored component, sorted by cell count descending and
    capped at `max_regions`."""
    bx, by, bw, bh = playable_bounds
    explored = set((int(c[0]), int(c[1])) for c in explored_cells)
    visited: set[tuple[int, int]] = set()
    components: list[dict[str, int]] = []
    for y0 in range(by, by + bh):
        for x0 in range(bx, bx + bw):
            if (x0, y0) in explored or (x0, y0) in visited:
                continue
            stack = [(x0, y0)]
            cells: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited or (cx, cy) in explored:
                    continue
                if cx < bx or cx >= bx + bw or cy < by or cy >= by + bh:
                    continue
                visited.add((cx, cy))
                cells.append((cx, cy))
                stack.extend([(cx + 1, cy), (cx - 1, cy),
                              (cx, cy + 1), (cx, cy - 1)])
            if cells:
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                components.append({
                    "x_lo": min(xs), "x_hi": max(xs),
                    "y_lo": min(ys), "y_hi": max(ys),
                    "cells": len(cells),
                })
    components.sort(key=lambda r: -r["cells"])
    return components[:max_regions]


def format_structured_fog(obs: dict[str, Any], playable_bounds) -> str:
    """Render the engine's `explored_cells` as a structured-text fog
    summary (the text channel that substitutes for the PNG minimap)."""
    ec = obs.get("explored_cells") or []
    if not ec:
        bx, by, bw, bh = playable_bounds
        return (
            "Unexplored regions (largest first, computed from engine fog state):\n"
            f"  - x ∈ [{bx}, {bx+bw-1}], y ∈ [{by}, {by+bh-1}]  "
            f"({bw*bh} cells, entire playable map)"
        )
    regions = compute_unexplored_regions(ec, playable_bounds, max_regions=6)
    if not regions:
        return "Unexplored regions: none — entire playable map is explored."
    lines = ["Unexplored regions (largest first, computed from engine fog state):"]
    for r in regions:
        lines.append(
            f"  - x ∈ [{r['x_lo']}, {r['x_hi']}], "
            f"y ∈ [{r['y_lo']}, {r['y_hi']}]  ({r['cells']} cells)"
        )
    return "\n".join(lines)
