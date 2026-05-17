"""Contributor-facing scenario layer.

A *scenario pack* is one YAML file describing one decision problem at
three difficulty levels (`easy` / `medium` / `hard`). Each level compiles
to an OpenRA-RL-Training `ScenarioDefinition` plus a declarative
`WinCondition` — so lab mates contribute scenarios with custom bot win
conditions and (schema-wise) custom maps without writing Python.

Public API:
    load_pack(path)               -> ScenarioPack
    discover_packs(dir)           -> list[ScenarioPack]
    pack.compile(level)           -> CompiledLevel  (engine def + win cond)

See CONTRIBUTING.md and packs/TEMPLATE.yaml.
"""

from .loader import discover_packs, load_pack
from .schema import CompiledLevel, Level, ScenarioMeta, ScenarioPack
from .win_conditions import WinCondition, WinContext, evaluate

__all__ = [
    "load_pack",
    "discover_packs",
    "ScenarioPack",
    "ScenarioMeta",
    "Level",
    "CompiledLevel",
    "WinCondition",
    "WinContext",
    "evaluate",
]
