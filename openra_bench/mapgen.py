"""YAML-referenceable map generators.

A scenario pack can write a generator spec instead of a static map id:

    base_map: {generator: arena, width: 176, height: 80, cordon: 4}

`materialize(spec)` builds the `.oramap` once (deterministic, content-
addressed) into a maps dir the loader already scans and returns its
logical id, so everything downstream (`resolve_map_path`,
`terrain_png_for`, the Rust engine via the absolute-path pin) keeps
working unchanged. Same spec ⇒ same file ⇒ reproducible.

Byte-format is identical to the proven training builder
(`tools/build_rush_hour_map.py`): map.bin v2 column-major, miniYAML
map.yaml, raw map.png — so the engine's generic `.oramap` loader parses
it like any shipped map.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable

CLEAR = 255  # passable grass
WATER = 1    # impassable water

# Generators register here: name -> fn(spec) -> (grid, map_size, bounds,
# spawns, title). `grid` is grid[y][x] of CLEAR/WATER.
_GENERATORS: dict[str, Callable[[dict], tuple]] = {}


def generator(name: str):
    def _reg(fn):
        _GENERATORS[name] = fn
        return fn
    return _reg


def _blank_arena(w: int, h: int, c: int) -> list[list[int]]:
    """Open arena with a water cordon (the base canvas every generator
    builds on)."""
    if not (16 <= w <= 512 and 16 <= h <= 512):
        raise ValueError(f"arena size out of range: {w}x{h}")
    if not (0 <= c < min(w, h) // 2):
        raise ValueError(f"arena cordon out of range: {c}")
    return [
        [
            WATER
            if (x < c or x >= w - c or y < c or y >= h - c)
            else CLEAR
            for x in range(w)
        ]
        for y in range(h)
    ]


def _stamp_rect(grid: list[list[int]], rect: dict, tile: int) -> None:
    """Paint a rectangle of `tile` onto `grid`. `rect` is {x, y, w, h}
    in playable-area coords (the cordon is honoured)."""
    h = len(grid); w = len(grid[0])
    rx = int(rect["x"]); ry = int(rect["y"])
    rw = int(rect.get("w", 1)); rh = int(rect.get("h", 1))
    for y in range(max(0, ry), min(h, ry + rh)):
        for x in range(max(0, rx), min(w, rx + rw)):
            grid[y][x] = tile


def _default_spawns(w: int, h: int, c: int) -> list[tuple[int, int]]:
    return [
        (c + 4, c + 4), (w - c - 5, c + 4),
        (c + 4, h - c - 5), (w - c - 5, h - c - 5),
    ]


@generator("arena")
def _arena(spec: dict):
    """Open rectangle with a water cordon + optional interior obstacles.

    Params:
      width, height (16..512), cordon (default 2).
      obstacles: list of {x, y, w, h} rectangles painted as WATER.
        Each rectangle is in absolute map coords; the cordon is
        the impassable frame, obstacles paint additional water
        INSIDE the playable area (forces flanking, cover, etc).
      spawns: optional list of [x, y] overriding the 4 corner defaults.
      title: optional human-readable map title.
    """
    w = int(spec.get("width", 128))
    h = int(spec.get("height", 40))
    c = int(spec.get("cordon", spec.get("bounds", 2)))
    grid = _blank_arena(w, h, c)
    for rect in (spec.get("obstacles") or []):
        _stamp_rect(grid, rect, WATER)
    spawns = [tuple(s) for s in spec["spawns"]] if spec.get("spawns") \
        else _default_spawns(w, h, c)
    bounds = (c, c, w - 2 * c, h - 2 * c)
    return grid, (w, h), bounds, spawns, spec.get("title", f"Arena {w}x{h}")


@generator("bridges-arena")
def _bridges_arena(spec: dict):
    """Open arena split by a horizontal or vertical water channel with
    N bridge gaps — the canonical 'chokepoint defense' map.

    Params:
      width, height, cordon as in `arena`.
      axis: 'horizontal' (default) splits along y; 'vertical' splits
        along x.
      channel_y / channel_x: the cell index of the water row/column
        (defaults to the map midpoint).
      channel_width: thickness of the water band (default 2 cells).
      bridges: list of {pos, width} declaring the cell position of
        each crossing (along the axis perpendicular to the channel)
        and how many cells wide the gap is. Default: 3 evenly-spaced
        2-cell bridges.
      obstacles: optional extra WATER rectangles, as in `arena`.
      spawns / title: as in `arena`.
    """
    w = int(spec.get("width", 128))
    h = int(spec.get("height", 40))
    c = int(spec.get("cordon", 2))
    grid = _blank_arena(w, h, c)
    axis = spec.get("axis", "horizontal")
    cw = int(spec.get("channel_width", 2))
    bridges = spec.get("bridges")
    if axis == "horizontal":
        cy = int(spec.get("channel_y", h // 2))
        if bridges is None:
            # 3 evenly-spaced 2-cell bridges
            third = (w - 2 * c) // 4
            bridges = [
                {"pos": c + third, "width": 2},
                {"pos": c + 2 * third, "width": 2},
                {"pos": c + 3 * third, "width": 2},
            ]
        for dy in range(cy, min(h, cy + cw)):
            for x in range(c, w - c):
                grid[dy][x] = WATER
        for br in bridges:
            bx = int(br["pos"]); bw = int(br.get("width", 2))
            for dy in range(cy, min(h, cy + cw)):
                for x in range(max(c, bx), min(w - c, bx + bw)):
                    grid[dy][x] = CLEAR
    elif axis == "vertical":
        cx = int(spec.get("channel_x", w // 2))
        if bridges is None:
            third = (h - 2 * c) // 4
            bridges = [
                {"pos": c + third, "width": 2},
                {"pos": c + 2 * third, "width": 2},
                {"pos": c + 3 * third, "width": 2},
            ]
        for dx in range(cx, min(w, cx + cw)):
            for y in range(c, h - c):
                grid[y][dx] = WATER
        for br in bridges:
            by = int(br["pos"]); bh = int(br.get("width", 2))
            for dx in range(cx, min(w, cx + cw)):
                for y in range(max(c, by), min(h - c, by + bh)):
                    grid[y][dx] = CLEAR
    else:
        raise ValueError(f"bridges-arena axis must be horizontal/vertical, got {axis!r}")
    for rect in (spec.get("obstacles") or []):
        _stamp_rect(grid, rect, WATER)
    spawns = [tuple(s) for s in spec["spawns"]] if spec.get("spawns") \
        else _default_spawns(w, h, c)
    bounds = (c, c, w - 2 * c, h - 2 * c)
    return grid, (w, h), bounds, spawns, spec.get("title", f"Bridges {w}x{h}")


@generator("chokepoint-arena")
def _chokepoint_arena(spec: dict):
    """Two open lobes linked by a single narrow corridor — the canonical
    'force the attacker through one cell' map. Defender bias: small army
    holding the pinch beats a larger army that can't deploy width.

    Params:
      width, height, cordon as in `arena`.
      pinch_x: x of the corridor wall (default w//2).
      pinch_width: thickness of the wall on each side of the corridor
        (default 6 cells of WATER above + below the corridor).
      corridor_width: open height of the corridor gap (default 4).
      corridor_y: y of the corridor centre (default h//2).
      obstacles / spawns / title: as in `arena`.
    """
    w = int(spec.get("width", 96))
    h = int(spec.get("height", 40))
    c = int(spec.get("cordon", 2))
    grid = _blank_arena(w, h, c)
    px = int(spec.get("pinch_x", w // 2))
    pw = int(spec.get("pinch_width", 6))      # wall thickness vertical
    cw = int(spec.get("corridor_width", 4))   # corridor height
    cy = int(spec.get("corridor_y", h // 2))
    # Wall above corridor
    for y in range(c, cy - cw // 2):
        for x in range(max(c, px - pw // 2), min(w - c, px + pw // 2 + 1)):
            grid[y][x] = WATER
    # Wall below corridor
    for y in range(cy + (cw + 1) // 2, h - c):
        for x in range(max(c, px - pw // 2), min(w - c, px + pw // 2 + 1)):
            grid[y][x] = WATER
    for rect in (spec.get("obstacles") or []):
        _stamp_rect(grid, rect, WATER)
    spawns = [tuple(s) for s in spec["spawns"]] if spec.get("spawns") \
        else [(c + 4, h // 2), (w - c - 5, h // 2)]
    bounds = (c, c, w - 2 * c, h - 2 * c)
    return grid, (w, h), bounds, spawns, spec.get("title", f"Chokepoint {w}x{h}")


@generator("naval-arena")
def _naval_arena(spec: dict):
    """Naval-arena: an open rectangle (like `arena`) with a documented
    convention that the SCENARIO YAML supplies a `water_rect:` block to
    declare the playable water band. The generator itself does NOT
    emit map.bin water tiles for the band — the C# tile encoding for
    "playable water that ships can traverse" is not yet plumbed in the
    Rust engine, so the engine reads naval terrain from the scenario
    YAML overlay (`oramap::MapDef::water_cells`).

    Params: width, height, cordon, title. Identical shape to `arena`
    so the generator returns the same grid; the only difference is a
    distinct content-addressed id so callers can tell naval scenarios
    apart in the generated-maps cache.
    """
    return _arena(spec)


def _map_bin(grid) -> bytes:
    h = len(grid)
    w = len(grid[0])
    buf = bytearray()
    buf.append(2)                                  # format version
    buf.extend(struct.pack("<H", w))
    buf.extend(struct.pack("<H", h))
    tiles_off = 17
    buf.extend(struct.pack("<I", tiles_off))
    buf.extend(struct.pack("<I", 0))               # no height layer
    buf.extend(struct.pack("<I", tiles_off + 3 * w * h))
    for x in range(w):                             # COLUMN-MAJOR
        for y in range(h):
            buf.extend(struct.pack("<H", grid[y][x]))
            buf.append(0)
    buf.extend(b"\x00\x00" * (w * h))              # resources: none
    return bytes(buf)


def _map_yaml(map_size, bounds, spawns, title) -> str:
    w, h = map_size
    bx, by, bw, bh = bounds
    spawn_actors = "".join(
        f"\tActor{i}: mpspawn\n\t\tOwner: Neutral\n\t\tLocation: {x},{y}\n"
        for i, (x, y) in enumerate(spawns)
    )
    return (
        "MapFormat: 12\nRequiresMod: ra\n"
        f"Title: {title}\nAuthor: openra-bench\nTileset: TEMPERAT\n"
        f"MapSize: {w},{h}\nBounds: {bx},{by},{bw},{bh}\n"
        "Visibility: Lobby\nCategories: Conquest\nPlayers:\n"
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


def _map_png(grid) -> bytes:
    from PIL import Image

    h = len(grid)
    w = len(grid[0])
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (30, 60, 120) if grid[y][x] == WATER else (80, 160, 60)
    b = io.BytesIO()
    img.save(b, format="PNG")
    return b.getvalue()


def _maps_dir() -> Path:
    """First existing maps dir the loader scans (write generated maps
    there so resolve_map_path finds them by logical id)."""
    from .scenarios.loader import _MAP_DIRS

    for d in _MAP_DIRS:
        if d.is_dir():
            return d
    d = _MAP_DIRS[0]
    d.mkdir(parents=True, exist_ok=True)
    return d


def spec_id(spec: dict) -> str:
    """Stable logical id for a generator spec. An explicit `name`
    (slug) gives a readable id for playback/docs; otherwise the id is
    content-addressed so identical specs share one file."""
    name = spec.get("name")
    if name:
        s = str(name)
        if not s.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"map name must be a slug, got {s!r}")
        return s
    gen = str(spec.get("generator", "arena"))
    key = repr(sorted((k, v) for k, v in spec.items()))
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"gen-{gen}-{digest}"


def materialize(spec: dict) -> str:
    """Build (or reuse) the .oramap for `spec`; return its logical id.

    Idempotent: same spec ⇒ same id ⇒ same bytes; skips rewrite if the
    file already exists with identical content.
    """
    gen = str(spec.get("generator", "arena"))
    if gen not in _GENERATORS:
        raise ValueError(
            f"unknown map generator {gen!r}; known: {sorted(_GENERATORS)}"
        )
    grid, map_size, bounds, spawns, title = _GENERATORS[gen](spec)
    mid = spec_id(spec)
    out = _maps_dir() / f"{mid}.oramap"
    blob = io.BytesIO()
    # Deterministic zip: pin every entry's date_time to a fixed epoch so
    # successive `materialize()` calls with identical content produce
    # byte-identical archives. Without this, `zipfile.writestr` stamps
    # each entry with the current wall-clock time, the idempotent
    # `read_bytes() == data` check always fails, and every test run
    # rewrites every .oramap. Under `pytest -n auto` multiple workers
    # then race-write the same file mid-read on other workers, surfacing
    # as transient parse failures in tests that compile many packs
    # (e.g. test_lh_recovery_after_mid_game_loss, test_manual_review_flow).
    _ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, payload in (
            ("map.yaml", _map_yaml(map_size, bounds, spawns, title).encode()),
            ("map.bin", _map_bin(grid)),
            ("map.png", _map_png(grid)),
        ):
            info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, payload)
    data = blob.getvalue()
    # Idempotent: skip rewrite when on-disk bytes already match. With
    # the deterministic zip above this branch is hit on every call after
    # the first, eliminating the write race.
    if out.exists():
        try:
            if out.read_bytes() == data:
                return mid
        except OSError:
            # A concurrent writer truncated the file mid-read; fall
            # through to the atomic-replace write below.
            pass
    # Atomic write: stage to a unique temp file then `os.replace` into
    # place. `os.replace` is atomic on POSIX/Windows, so a concurrent
    # reader either sees the previous complete archive or the new one,
    # never a half-written file.
    tmp = out.with_suffix(
        out.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        tmp.write_bytes(data)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return mid


def resolve_base_map(base_map: Any) -> str:
    """Pass through a string id; materialize a generator-spec dict to
    its logical id. Used by ScenarioPack.compile."""
    if isinstance(base_map, dict):
        return materialize(base_map)
    return str(base_map)
