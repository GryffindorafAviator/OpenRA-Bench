"""Pack discovery + loading + map-support gating.

The Rust env currently loads only two hardcoded map geometries
(`rush-hour`, `scout-maginot` — see OpenRA-Rust env.rs). Contributors
may still author meaningful scenarios *today* by varying actors, spawns,
and win conditions on a supported geometry. A pack that names an
unsupported `base_map` still loads and validates, but its compiled
levels carry `map_supported=False` so the runner can skip/flag them
rather than crash. Generic `.oramap` loading lands in Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .schema import LevelName, ScenarioPack

PACKS_DIR = Path(__file__).parent / "packs"

# Dirs scanned for `<base_map>.oramap` terrain files. The Rust engine
# parses real .oramap terrain (map.bin) when handed an absolute path, so
# any map present here is a usable custom map — not a 2-entry allowlist.
_MAP_DIRS = [
    Path.home() / "Projects/OpenRA-RL-Training/scenarios/maps",
    Path.home() / "Projects/openra-rl/maps",
]


def resolve_map_path(base_map: str) -> Path | None:
    """Resolve a logical `base_map` id (with or without `.oramap`) to an
    absolute terrain-file path, or None if no such map exists."""
    name = base_map if base_map.endswith(".oramap") else f"{base_map}.oramap"
    # Allow `base_map` to be an explicit absolute/relative path too.
    direct = Path(base_map)
    if direct.suffix == ".oramap" and direct.is_file():
        return direct.resolve()
    for d in _MAP_DIRS:
        p = d / name
        if p.is_file():
            return p.resolve()
    return None


def load_pack(path: str | Path) -> ScenarioPack:
    """Parse and validate a single pack YAML."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return ScenarioPack(**data)
    except Exception as e:  # noqa: BLE001 — re-raise with file context
        raise ValueError(f"invalid scenario pack {path}: {e}") from e


def discover_packs(directory: str | Path | None = None) -> list[ScenarioPack]:
    """Load every *.yaml pack in `directory` (default: bundled packs/).

    Templates (filenames starting with '_' or 'TEMPLATE') are skipped.
    """
    directory = Path(directory) if directory else PACKS_DIR
    packs: list[ScenarioPack] = []
    for p in sorted(directory.glob("*.yaml")):
        if p.name.startswith(("_", "TEMPLATE")):
            continue
        packs.append(load_pack(p))
    return packs


def is_map_supported(base_map: str) -> bool:
    return resolve_map_path(base_map) is not None


def compile_level(pack: ScenarioPack, level: LevelName):
    """Compile one level, wiring in the map-support flag."""
    return pack.compile(level, map_supported=is_map_supported(pack.base_map))
