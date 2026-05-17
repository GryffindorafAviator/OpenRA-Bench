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

# Logical base_map id -> the Rust scenario alias the env actually loads.
# Extend this as Phase 3 adds real map loading.
SUPPORTED_MAPS: dict[str, str] = {
    "rush-hour-arena": "scenarios/discovery/rush-hour.yaml",
    "scout-maginot": "scenarios/strategy/scout-maginot.yaml",
}

PACKS_DIR = Path(__file__).parent / "packs"


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
    return base_map in SUPPORTED_MAPS


def rust_scenario_alias(base_map: str) -> str:
    """The path/alias to hand the Rust env for this logical map."""
    return SUPPORTED_MAPS[base_map]


def compile_level(pack: ScenarioPack, level: LevelName):
    """Compile one level, wiring in the map-support flag."""
    return pack.compile(level, map_supported=is_map_supported(pack.base_map))
