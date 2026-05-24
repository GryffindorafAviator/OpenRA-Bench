"""YAML-referenceable map generator (mapgen).

A pack can declare `base_map: {generator: arena, ...}` and the bench
materializes a real, deterministic .oramap the loader/engine resolve
like any shipped map.
"""
from __future__ import annotations

import zipfile

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.mapgen import materialize, resolve_base_map, spec_id
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level, resolve_map_path


def test_arena_materializes_and_resolves():
    s = {"generator": "arena", "width": 160, "height": 72, "cordon": 4}
    mid = materialize(s)
    assert mid == spec_id(s) == materialize(s)        # deterministic
    p = resolve_map_path(mid)
    assert p is not None and p.is_file()
    y = zipfile.ZipFile(p).read("map.yaml").decode()
    assert "MapSize: 160,72" in y and "Bounds: 4,4,152,64" in y
    assert {"map.yaml", "map.bin", "map.png"} <= set(
        zipfile.ZipFile(p).namelist()
    )


def test_content_addressed_and_idempotent():
    a = {"generator": "arena", "width": 128, "height": 64}
    b = {"generator": "arena", "height": 64, "width": 128}  # key order
    assert spec_id(a) == spec_id(b)                    # order-independent
    p = resolve_map_path(materialize(a))
    before = p.read_bytes()
    materialize(b)                                     # rewrite same spec
    assert p.read_bytes() == before                    # byte-identical


def test_named_spec_gives_readable_id():
    s = {"generator": "arena", "name": "tiny-test-arena",
         "width": 64, "height": 48, "cordon": 2}
    assert materialize(s) == "tiny-test-arena"
    with pytest.raises(ValueError):
        spec_id({"generator": "arena", "name": "bad name!"})


def test_unknown_generator_and_bounds_rejected():
    with pytest.raises(ValueError):
        resolve_base_map({"generator": "does-not-exist"})
    with pytest.raises(ValueError):
        materialize({"generator": "arena", "width": 9999, "height": 40})
    with pytest.raises(ValueError):
        materialize({"generator": "arena", "width": 64, "height": 64,
                     "cordon": 40})


def test_string_base_map_passthrough():
    assert resolve_base_map("rush-hour-arena") == "rush-hour-arena"


def test_pack_with_generator_spec_compiles():
    # #2 hard dogfoods the YAML path: base_map is a generator spec.
    p = load_pack(PACKS_DIR / "action-sequenced-execution.yaml")
    hard = compile_level(p, "hard")
    assert hard.scenario.base_map == "action-sequenced-execution-arena-hard"
    assert hard.map_supported


def test_arena_obstacles_paint_interior_water():
    """Obstacles are WATER rectangles inside the playable area —
    distinguishable from the cordon by their position."""
    s = {"generator": "arena", "name": "test-obs-paint",
         "width": 40, "height": 20, "cordon": 2,
         "obstacles": [{"x": 10, "y": 8, "w": 4, "h": 2}]}
    mid = materialize(s)
    p = resolve_map_path(mid)
    # Parse map.bin and confirm the (10,8)..(13,9) block is water but
    # surrounding playable cells are clear.
    blob = zipfile.ZipFile(p).read("map.bin")
    w = int.from_bytes(blob[1:3], "little")
    h = int.from_bytes(blob[3:5], "little")
    tiles_off = int.from_bytes(blob[5:9], "little")
    def cell(x, y):
        idx = tiles_off + 3 * (x * h + y)
        return int.from_bytes(blob[idx:idx + 2], "little")
    assert (w, h) == (40, 20)
    assert cell(10, 8) == 1 and cell(13, 9) == 1, "obstacle painted as water"
    assert cell(20, 10) == 255, "interior outside obstacle stays clear"
    assert cell(0, 0) == 1, "cordon corner still water"


def test_bridges_arena_channel_with_gaps():
    """Horizontal water band with explicit bridge gaps."""
    s = {"generator": "bridges-arena", "name": "test-br-h",
         "width": 60, "height": 30, "cordon": 2,
         "channel_y": 14, "channel_width": 2,
         "bridges": [{"pos": 20, "width": 3}]}
    mid = materialize(s)
    p = resolve_map_path(mid)
    blob = zipfile.ZipFile(p).read("map.bin")
    w = int.from_bytes(blob[1:3], "little")
    h = int.from_bytes(blob[3:5], "little")
    tiles_off = int.from_bytes(blob[5:9], "little")
    def cell(x, y):
        idx = tiles_off + 3 * (x * h + y)
        return int.from_bytes(blob[idx:idx + 2], "little")
    # Channel row 14 is water on the non-bridge cells
    assert cell(40, 14) == 1, "off-bridge channel cell is water"
    # Bridge gap (20..22) is clear
    assert cell(20, 14) == 255 and cell(22, 14) == 255
    # Row 13 (above channel) is open
    assert cell(40, 13) == 255


def test_chokepoint_arena_corridor_is_only_route():
    """One narrow corridor across the wall; everywhere else along the
    wall x-band is water."""
    s = {"generator": "chokepoint-arena", "name": "test-chk",
         "width": 60, "height": 30, "cordon": 2,
         "pinch_x": 30, "pinch_width": 6, "corridor_width": 4,
         "corridor_y": 14}
    mid = materialize(s)
    p = resolve_map_path(mid)
    blob = zipfile.ZipFile(p).read("map.bin")
    h = int.from_bytes(blob[3:5], "little")
    tiles_off = int.from_bytes(blob[5:9], "little")
    def cell(x, y):
        idx = tiles_off + 3 * (x * h + y)
        return int.from_bytes(blob[idx:idx + 2], "little")
    # Corridor (y=12..15) clear at the pinch column
    assert cell(30, 12) == 255 and cell(30, 15) == 255
    # Above corridor (y=8) at the pinch is water
    assert cell(30, 8) == 1
    # Below corridor (y=22) at the pinch is water
    assert cell(30, 22) == 1
    # Open areas left/right of the wall still clear
    assert cell(10, 14) == 255 and cell(50, 14) == 255
