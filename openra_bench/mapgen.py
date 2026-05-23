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
import struct
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


@generator("arena")
def _arena(spec: dict):
    """Open rectangle with a water cordon (generalizes rush-hour-arena
    and scout-arena). Params: width, height, cordon."""
    w = int(spec.get("width", 128))
    h = int(spec.get("height", 40))
    c = int(spec.get("cordon", spec.get("bounds", 2)))
    if not (16 <= w <= 512 and 16 <= h <= 512):
        raise ValueError(f"arena size out of range: {w}x{h}")
    if not (0 <= c < min(w, h) // 2):
        raise ValueError(f"arena cordon out of range: {c}")
    grid = [
        [
            WATER
            if (x < c or x >= w - c or y < c or y >= h - c)
            else CLEAR
            for x in range(w)
        ]
        for y in range(h)
    ]
    bounds = (c, c, w - 2 * c, h - 2 * c)
    # Four playable-corner spawn references.
    spawns = [
        (c + 4, c + 4), (w - c - 5, c + 4),
        (c + 4, h - c - 5), (w - c - 5, h - c - 5),
    ]
    return grid, (w, h), bounds, spawns, spec.get("title", f"Arena {w}x{h}")


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
    with zipfile.ZipFile(blob, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("map.yaml", _map_yaml(map_size, bounds, spawns, title))
        zf.writestr("map.bin", _map_bin(grid))
        zf.writestr("map.png", _map_png(grid))
    data = blob.getvalue()
    if not (out.exists() and out.read_bytes() == data):
        out.write_bytes(data)
    return mid


def resolve_base_map(base_map: Any) -> str:
    """Pass through a string id; materialize a generator-spec dict to
    its logical id. Used by ScenarioPack.compile."""
    if isinstance(base_map, dict):
        return materialize(base_map)
    return str(base_map)
