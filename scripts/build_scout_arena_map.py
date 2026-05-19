#!/usr/bin/env python3
"""Mint `scout-arena.oramap` — a larger open arena for the hard tier of
action-sequenced-execution (and any scenario wanting real fog scouting
across dispersed enemy bases).

rush-hour-arena is 128x40 (124x36 playable) — too cramped for multiple
dispersed enemy bases discovered by scouting. scout-arena is 176x80
(168x72 playable, ~3.3x the area) so several enemy bases can sit far
apart in fog and a coordinate-blind agent must genuinely scout to find
its ordered waypoints.

Byte-format identical to tools/build_rush_hour_map.py in the training
repo (map.bin v2 column-major, miniYAML map.yaml, raw map.png terrain)
so the Rust engine's generic .oramap loader (task #12) parses it like
any shipped map. Written into the training maps dir so the bench's
resolve_map_path() finds it by logical id, exactly like rush-hour-arena.

Usage:  python scripts/build_scout_arena_map.py
"""

import io
import os
import struct
import zipfile
from pathlib import Path

WIDTH = 176
HEIGHT = 80
BOUNDS_X = 4
BOUNDS_Y = 4
PLAY_W = WIDTH - 2 * BOUNDS_X    # 168
PLAY_H = HEIGHT - 2 * BOUNDS_Y   # 72

CLEAR = 255  # passable grass
WATER = 1    # impassable water

# Four corner spawn references (mpspawn) — gives scenarios up to 4
# dispersed staging corners and seed-driven spawn variation.
SPAWNS = [(8, 8), (WIDTH - 9, 8), (8, HEIGHT - 9), (WIDTH - 9, HEIGHT - 9)]

OUT_DIRS = [
    Path.home() / "Projects/OpenRA-RL-Training/scenarios/maps",
    Path.home() / "Projects/openra-rl/maps",
]


def build_terrain():
    """Open grass interior with a water cordon (matches rush-hour's
    BOUNDS_X/Y so the playable rect is exactly PLAY_W x PLAY_H)."""
    grid = [[CLEAR] * WIDTH for _ in range(HEIGHT)]
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if (
                x < BOUNDS_X or x >= WIDTH - BOUNDS_X
                or y < BOUNDS_Y or y >= HEIGHT - BOUNDS_Y
            ):
                grid[y][x] = WATER
    return grid


def create_map_bin(grid):
    height = len(grid)
    width = len(grid[0])
    buf = bytearray()
    buf.append(2)  # format version
    buf.extend(struct.pack("<H", width))
    buf.extend(struct.pack("<H", height))
    tiles_offset = 17
    heights_offset = 0
    resources_offset = tiles_offset + 3 * width * height
    buf.extend(struct.pack("<I", tiles_offset))
    buf.extend(struct.pack("<I", heights_offset))
    buf.extend(struct.pack("<I", resources_offset))
    # COLUMN-MAJOR: for x in width, for y in height
    for x in range(width):
        for y in range(height):
            buf.extend(struct.pack("<H", grid[y][x]))
            buf.append(0)  # tile index
    for _ in range(width * height):
        buf.extend(b"\x00\x00")  # resources: none
    return bytes(buf)


def create_map_yaml():
    spawn_actors = "".join(
        f"\tActor{i}: mpspawn\n\t\tOwner: Neutral\n\t\tLocation: {x},{y}\n"
        for i, (x, y) in enumerate(SPAWNS)
    )
    return (
        "MapFormat: 12\n"
        "RequiresMod: ra\n"
        "Title: Scout Arena\n"
        "Author: openra-bench\n"
        "Tileset: TEMPERAT\n"
        f"MapSize: {WIDTH},{HEIGHT}\n"
        f"Bounds: {BOUNDS_X},{BOUNDS_Y},{PLAY_W},{PLAY_H}\n"
        "Visibility: Lobby\n"
        "Categories: Conquest\n"
        "Players:\n"
        "\tPlayerReference@Neutral:\n\t\tName: Neutral\n"
        "\t\tOwnsWorld: True\n\t\tNonCombatant: True\n\t\tFaction: allies\n"
        "\tPlayerReference@Creeps:\n\t\tName: Creeps\n"
        "\t\tNonCombatant: True\n\t\tFaction: allies\n"
        "\t\tEnemies: Multi0, Multi1\n"
        "\tPlayerReference@Multi0:\n\t\tName: Multi0\n\t\tPlayable: True\n"
        "\t\tFaction: Random\n\t\tEnemies: Creeps\n"
        "\tPlayerReference@Multi1:\n\t\tName: Multi1\n\t\tPlayable: True\n"
        "\t\tFaction: Random\n\t\tEnemies: Creeps\n"
        "Actors:\n" + spawn_actors
    )


def create_map_png(grid):
    from PIL import Image

    img = Image.new("RGB", (WIDTH, HEIGHT))
    px = img.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            px[x, y] = (30, 60, 120) if grid[y][x] == WATER else (80, 160, 60)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_oramap(path: Path):
    grid = build_terrain()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("map.yaml", create_map_yaml())
        zf.writestr("map.bin", create_map_bin(grid))
        zf.writestr("map.png", create_map_png(grid))
    print(f"Created {path} ({os.path.getsize(path)} bytes) "
          f"{WIDTH}x{HEIGHT}, playable {PLAY_W}x{PLAY_H}")


def main():
    wrote = False
    for d in OUT_DIRS:
        if d.is_dir():
            build_oramap(d / "scout-arena.oramap")
            wrote = True
    if not wrote:
        raise SystemExit(f"no map dir found among {OUT_DIRS}")


if __name__ == "__main__":
    main()
