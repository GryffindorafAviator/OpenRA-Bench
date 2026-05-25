"""Pin the 3-state shroud (unexplored / fogged / visible) all the way
from the Rust env to the rendered PNG.

The user-reported bug (zh): "vision region is still shown on the map
but it is actually already past visible region and only explored. The
map is not updated correctly." A tank moves away from an enemy; the
map cell stays BRIGHT (looks like live vision) but the enemy MARKER
disappears (because the engine drops the actor from `enemy_summary`
when it's no longer in agent sight). The visual contradiction reads
as "the enemy disappeared".

Root cause: the Rust engine maintains the proper RA 3-state shroud
(`world.rs::shroud`: 0=unexplored, 1=fogged, 2=visible), but the
PyO3 observation surfaces only `explored_cells` (the union of fogged
+ visible) — the fogged/visible split is lost at the bench boundary.
The bench therefore reconstructs `visible_cells` per turn from each
live agent actor's sight radius (the same shape the engine itself
uses to reveal shroud) and renders fogged-only cells with a dim
tint distinct from the bright "currently visible" tint.

These tests pin:
1. Fresh after reset, every cell within an agent unit's sight is
   marked visible AND explored.
2. After the units MOVE far away, the prior-vision area is in
   `fogged_cells` (explored but NOT in `visible_cells`).
3. The ASCII minimap uses the 3-state encoding ('+' visible, '.'
   fogged, '#' unexplored), and the rendered PNG paints visible vs
   fogged cells in DIFFERENT colors (so a human can SEE that they've
   lost vision of an area).
"""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from openra_bench.minimap import (  # noqa: E402
    _BG_FOGGED,
    _BG_UNKNOWN,
    _BG_VISIBLE,
    render_png_b64,
    render_tactical_minimap,
)
from openra_bench.rust_adapter import (  # noqa: E402
    _SIGHT_BY_TYPE,
    RustObsAdapter,
)


def _xy(cells):
    return {(int(c[0]), int(c[1])) for c in cells}


def _make_obs(
    unit_positions: dict,
    explored: list[tuple[int, int]],
    map_w: int = 32,
    map_h: int = 32,
    **extra,
):
    """Build a minimal obs dict mirroring the Rust env's shape."""
    obs = {
        "unit_positions": unit_positions,
        "unit_hp": {uid: 1.0 for uid in unit_positions},
        "enemy_positions": [],
        "enemy_hp": {},
        "enemy_buildings_summary": [],
        "own_buildings": [],
        "explored_cells": [list(c) for c in explored],
        "explored_percent": 0.0,
        "map_info": {"width": map_w, "height": map_h},
        "game_tick": 0,
        "units_killed": 0,
        "production": [],
        "spatial": [],
        "spatial_shape": (0, 0, 0),
    }
    obs.update(extra)
    return obs


def test_visible_cells_cover_unit_sight_radius_at_reset():
    """Every cell within sight of a live agent unit is marked visible
    AND explored — the union forms the bright disc around each unit."""
    # 1tnk sight=5, placed at (10, 10) on a 32×32 map.
    sight = _SIGHT_BY_TYPE["1tnk"]
    pos = {"1001": {"cell_x": 10, "cell_y": 10, "actor_type": "1tnk"}}
    # Engine reveals roughly the sight disc — for the test, seed
    # explored with the same disc so the adapter's "explored ∩ visible"
    # filter doesn't drop anything.
    explored = [
        (x, y)
        for x in range(10 - sight, 10 + sight + 1)
        for y in range(10 - sight, 10 + sight + 1)
    ]
    obs = _make_obs(pos, explored)
    ad = RustObsAdapter()
    ad.observe(obs)
    rs = ad.render_state()
    vis = _xy(rs["visible_cells"])
    expl = _xy(rs["explored_cells"])
    # The unit's own cell is visible.
    assert (10, 10) in vis, "unit's cell must be in visible set"
    # The full sight disc is visible (Chebyshev radius).
    for dx in range(-sight, sight + 1):
        for dy in range(-sight, sight + 1):
            cell = (10 + dx, 10 + dy)
            assert cell in vis, (
                f"cell {cell} (within sight={sight} of unit) missing "
                f"from visible_cells"
            )
            assert cell in expl, f"cell {cell} must be in explored_cells too"
    # No cell outside the disc is visible.
    assert (10, 10 + sight + 1) not in vis
    # visible ⊆ explored.
    assert vis.issubset(expl)


def test_visible_drops_when_unit_moves_away_leaving_fogged_region():
    """The load-bearing assertion for the user-reported bug. After
    the unit moves far away, the previously-visible area must
    transition from `visible_cells` to `fogged_cells` (explored but
    not currently visible) — without this transition the map stays
    bright behind a retreating unit while enemy markers vanish."""
    sight = _SIGHT_BY_TYPE["1tnk"]
    ad = RustObsAdapter()

    # Turn 1: unit at (10, 10) — visible disc around it.
    pos1 = {"1001": {"cell_x": 10, "cell_y": 10, "actor_type": "1tnk"}}
    explored1 = [
        (x, y)
        for x in range(10 - sight, 10 + sight + 1)
        for y in range(10 - sight, 10 + sight + 1)
    ]
    ad.observe(_make_obs(pos1, explored1))
    rs1 = ad.render_state()
    vis1 = _xy(rs1["visible_cells"])
    assert (10, 10) in vis1
    assert (12, 10) in vis1  # within sight=5

    # Turn 2: same unit moves to (25, 25) — far outside the old disc.
    # The engine accumulates explored_cells (union of all reveals).
    pos2 = {"1001": {"cell_x": 25, "cell_y": 25, "actor_type": "1tnk"}}
    explored2 = explored1 + [
        (x, y)
        for x in range(25 - sight, 25 + sight + 1)
        for y in range(25 - sight, 25 + sight + 1)
    ]
    ad.observe(_make_obs(pos2, explored2))
    rs2 = ad.render_state()
    vis2 = _xy(rs2["visible_cells"])
    fog2 = _xy(rs2["fogged_cells"])
    expl2 = _xy(rs2["explored_cells"])

    # New disc IS visible.
    assert (25, 25) in vis2
    # Old cell (10, 10) — was visible at t=1 — is now FOGGED (explored
    # but not visible). This is the assertion that fails before the
    # fix: vis2 used to also contain (10, 10) (since the bench
    # collapsed the 3-state shroud into the 2-state explored mask).
    assert (10, 10) not in vis2, (
        "cell (10,10) must NOT be visible after the unit moved away "
        "— the bug is that this cell used to stay 'visible'"
    )
    assert (10, 10) in expl2, "cell (10,10) must remain in explored"
    assert (10, 10) in fog2, (
        "cell (10,10) must be in fogged_cells (explored − visible)"
    )
    # visible ⊆ explored, and visible ∩ fogged = ∅.
    assert vis2.issubset(expl2)
    assert not (vis2 & fog2)
    # Union equals explored.
    assert vis2 | fog2 == expl2


def test_ascii_minimap_uses_3state_encoding():
    """ASCII grid uses '+' = visible, '.' = fogged, '#' = unexplored.
    The minimap renderer (`minimap.py::_bg_for`) keys off the '+'
    marks to pick the dim-vs-bright tint."""
    sight = _SIGHT_BY_TYPE["1tnk"]
    ad = RustObsAdapter()

    # t=1: unit at (5, 5) — small map.
    pos1 = {"1001": {"cell_x": 5, "cell_y": 5, "actor_type": "1tnk"}}
    explored1 = [
        (x, y)
        for x in range(max(0, 5 - sight), 5 + sight + 1)
        for y in range(max(0, 5 - sight), 5 + sight + 1)
    ]
    ad.observe(_make_obs(pos1, explored1, map_w=20, map_h=20))
    # t=2: unit at (15, 15).
    pos2 = {"1001": {"cell_x": 15, "cell_y": 15, "actor_type": "1tnk"}}
    explored2 = explored1 + [
        (x, y)
        for x in range(15 - sight, min(20, 15 + sight + 1))
        for y in range(15 - sight, min(20, 15 + sight + 1))
    ]
    ad.observe(_make_obs(pos2, explored2, map_w=20, map_h=20))

    ascii_mm = ad.ascii_minimap()
    rows = ascii_mm.split("\n")

    # All three chars present.
    assert any("+" in r for r in rows), "ASCII map must contain '+' (visible)"
    assert any("." in r for r in rows), "ASCII map must contain '.' (fogged)"
    assert any("#" in r for r in rows), "ASCII map must contain '#' (unexplored)"

    # (15, 15) is visible (current unit cell) → '+'.
    assert rows[15][15] == "+"
    # (5, 5) is fogged (explored at t=1, not currently visible) → '.'.
    assert rows[5][5] == "."
    # (19, 0) is unexplored → '#' (far from both unit positions).
    assert rows[0][19] == "#"


def test_png_paints_visible_and_fogged_in_distinct_colors():
    """The PNG (the actual artifact the human sees) must draw visible
    cells with the bright tint and fogged cells with the dim tint —
    otherwise the user reports "the area looks like I still have
    vision but the enemy disappeared"."""
    sight = _SIGHT_BY_TYPE["1tnk"]
    ad = RustObsAdapter()
    pos1 = {"1001": {"cell_x": 5, "cell_y": 5, "actor_type": "1tnk"}}
    explored1 = [
        (x, y)
        for x in range(max(0, 5 - sight), 5 + sight + 1)
        for y in range(max(0, 5 - sight), 5 + sight + 1)
    ]
    ad.observe(_make_obs(pos1, explored1, map_w=20, map_h=20))
    pos2 = {"1001": {"cell_x": 15, "cell_y": 15, "actor_type": "1tnk"}}
    explored2 = explored1 + [
        (x, y)
        for x in range(15 - sight, min(20, 15 + sight + 1))
        for y in range(15 - sight, min(20, 15 + sight + 1))
    ]
    ad.observe(_make_obs(pos2, explored2, map_w=20, map_h=20))
    rs = ad.render_state()
    b64 = render_png_b64(rs)
    assert b64
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    from openra_bench.minimap import CELL

    def _bg_at(cx, cy):
        # Sample the centre of the cell; avoid the unit marker if
        # we sample at the unit's exact cell.
        return im.getpixel((cx * CELL + CELL // 2, cy * CELL + CELL // 2))

    # (15, 15) is the unit's cell — skip (unit marker overlays). Pick
    # an adjacent visible cell instead.
    assert _bg_at(13, 15) == _BG_VISIBLE, (
        "cell adjacent to live unit must be painted with VISIBLE tint"
    )
    # (5, 5) is fogged — must use the DIM tint, NOT the bright tint.
    assert _bg_at(5, 5) == _BG_FOGGED, (
        "previously-explored cell after unit moved away must use the "
        "FOGGED (dim) tint — this is the user-reported bug fix"
    )
    assert _BG_FOGGED != _BG_VISIBLE  # sanity
    # (19, 0) is unexplored — full shroud tint.
    assert _bg_at(19, 0) == _BG_UNKNOWN


def test_legacy_2state_grid_renders_as_bright_backcompat():
    """A 2-state ASCII grid (no '+' marks — older callers, tests with
    hand-built grids) must continue to render '.' as the bright tint
    so we don't silently dim every existing minimap."""
    rs = {
        "minimap": "\n".join(
            "".join("." if x < 8 else "#" for x in range(16))
            for _ in range(8)
        ),
        "units_summary": [{"cell_x": 2, "cell_y": 3}],
    }
    b64 = render_png_b64(rs)
    assert b64
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    from openra_bench.minimap import CELL, _BG_EXPLORED

    # Cell (5, 5) is '.' in a 2-state grid → bright (legacy alias).
    px = im.getpixel((5 * CELL + CELL // 2, 5 * CELL + CELL // 2))
    assert px == _BG_EXPLORED, (
        "2-state legacy grid must keep '.' bright (back-compat)"
    )


def test_end_to_end_rust_env_visible_shrinks_when_units_move_away():
    """End-to-end: real Rust env, move units east, then the old
    spawn-area cells transition from visible to fogged.

    This is the only test in the file that exercises the LIVE engine
    — the others use synthetic obs dicts. Skipped gracefully if the
    bench-side test harness (compile_level / scenarios) isn't
    available."""
    try:
        from openra_train import Command, OpenRAEnv

        from openra_bench.eval_core import _scenario_to_tmp_yaml
        from openra_bench.scenarios import load_pack
        from openra_bench.scenarios.loader import PACKS_DIR, compile_level
    except Exception:
        pytest.skip("Live engine harness unavailable")

    pack = load_pack(PACKS_DIR / "action-multiunit-coordination.yaml")
    compiled = compile_level(pack, "easy")
    tmp = _scenario_to_tmp_yaml(compiled)
    env = OpenRAEnv(scenario_path=tmp, seed=1)
    obs = env.reset()

    type_by_id = {
        uid: (p.get("actor_type") or "?").lower()
        for uid, p in obs["unit_positions"].items()
        if isinstance(p, dict)
    }
    ad = RustObsAdapter(type_by_id=type_by_id)
    ad.observe(obs)
    rs0 = ad.render_state()
    vis0 = _xy(rs0["visible_cells"])
    # Pick a cell at the spawn-area centre.
    spawn_cells = [
        (int(p["cell_x"]), int(p["cell_y"]))
        for p in obs["unit_positions"].values()
        if isinstance(p, dict)
    ]
    sx, sy = spawn_cells[0]
    assert (sx, sy) in vis0

    # Move all units far east.
    ids = list(obs["unit_positions"].keys())
    last_obs = obs
    for _ in range(20):
        last_obs, _r, done, _info = env.step(
            [Command.move_units(ids, 35, 8)]
        )
        if done:
            break
    ad.observe(last_obs)
    rs1 = ad.render_state()
    vis1 = _xy(rs1["visible_cells"])
    fog1 = _xy(rs1["fogged_cells"])
    expl1 = _xy(rs1["explored_cells"])

    # New unit locations are visible.
    new_cells = {
        (int(p["cell_x"]), int(p["cell_y"]))
        for p in last_obs["unit_positions"].values()
        if isinstance(p, dict)
    }
    assert new_cells.issubset(vis1)

    # If the units actually moved away from at least one original
    # spawn cell (sanity check — sometimes a unit blocks the path
    # and stays put), that original cell should now be FOGGED, not
    # VISIBLE.
    moved_away = [
        c for c in spawn_cells if c not in new_cells and c in expl1
    ]
    if not moved_away:
        pytest.skip("unit didn't move far enough to leave its spawn cell")
    for c in moved_away:
        # The load-bearing assertion: an explored cell with no live
        # unit nearby must be fogged.
        nearest = min(
            max(abs(c[0] - nc[0]), abs(c[1] - nc[1])) for nc in new_cells
        )
        # If the cell is OUT of sight of every current unit it must
        # be fogged, not visible. (We pick units with sight ≤ 7, so
        # any cell ≥ 8 away from every unit is definitely out of
        # sight.)
        if nearest >= 8:
            assert c in fog1, (
                f"cell {c} (≥{nearest} cells from every unit) must "
                f"be fogged, not visible"
            )
            assert c not in vis1
