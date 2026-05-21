"""Scenario-pack schema: one YAML -> three difficulty levels.

A pack composes (not forks) the OpenRA-RL-Training `ScenarioDefinition`.
`base` holds the shared engine fields; each level supplies a deep-merge
`overrides` patch plus its own `win_condition` / `fail_condition`. This
keeps a three-level scenario in a single readable file and guarantees
every level stays a valid engine scenario.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import openra_rl_training.scenario as _orts
from openra_rl_training.scenario import ScenarioDefinition

# S0: the Rust engine seeds ore patches around a `mine` actor
# (world.rs: "Seed ore patches around mine actors"), but Training's
# VALID_ACTOR_TYPES (from openra_env.game_data) omits the ore-source
# map props since they aren't units/buildings. Extend the in-place set
# so economy scenarios can place ore. Engine-supported only.
_orts.VALID_ACTOR_TYPES |= {"mine", "gmine"}
from pydantic import BaseModel, Field, field_validator

from .win_conditions import WinCondition

LevelName = Literal["easy", "medium", "hard"]
# "adversarial" = head-to-head reasoning vs a reactive opponent (the
# axis an RTS engine uniquely owns); ranked by a difficulty ladder + Elo.
Capability = Literal["perception", "reasoning", "action", "adversarial"]


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursive dict merge; lists/scalars in `patch` replace wholesale.

    Replacing (not concatenating) lists is deliberate: a level that
    customises `actors` states the full actor list, so diffs stay
    auditable in review.
    """
    out = copy.deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ScenarioMeta(BaseModel):
    """Why this scenario exists — required so the library stays meaningful."""

    id: str = Field(..., description="Unique slug, e.g. partial-info-rescue")
    title: str
    capability: Capability = Field(
        ..., description="Primary P/R/A chain link this scenario stresses"
    )
    real_world_meaning: str = Field(
        ..., min_length=20, description="The real decision this abstracts"
    )
    robotics_analogue: str = Field(
        ..., min_length=10, description="Concrete robotics/agentic parallel"
    )
    author: str = "unknown"
    # Real-world / benchmark anchors — every pack must name at least one
    # external referent (a named benchmark or a real-world capability)
    # so the scenario carries transfer signal, not just RTS novelty.
    # Multi-anchor packs are common (e.g. tool-fidelity → BFCL V4 +
    # τ²-bench + IFBench), so this is a list. The list-non-empty rule
    # is suite-enforced by tests/test_benchmark_anchor_required.py.
    benchmark_anchor: list[str] = Field(
        default_factory=list,
        description="Named benchmarks and/or real-world capabilities this "
        "scenario transfers to (e.g. ['MicroRTS rush-defense', "
        "'SC2LE defend-the-cheese tempo', 'incident-response: defend "
        "production infra under live attack']).",
    )
    # Hygiene (audit): "quarantine" keeps the file on disk and runnable
    # by explicit --packs, but excludes it from the default eval set so
    # it doesn't dilute scores / the leaderboard. Used for the
    # redundant cat-* over-generation and the harvest packs blocked on
    # the engine S0/S1 ore-income prerequisite.
    status: Literal["active", "quarantine"] = "active"
    quarantine_reason: str = ""

    @field_validator("id")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not v.replace("-", "").isalnum() or v != v.lower():
            raise ValueError(f"id must be lowercase kebab-case slug, got {v!r}")
        return v


class Level(BaseModel):
    description: str = Field(..., min_length=10)
    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Deep-merge patch onto pack.base"
    )
    win_condition: WinCondition
    fail_condition: WinCondition | None = None
    # How region objectives are disclosed in the briefing. `exact`
    # gives (x,y) coordinates; `relative` gives only the authored
    # compass label ("the north-east corner") so the model must
    # ground the target on the minimap itself (spatial reasoning).
    objective_coords: Literal["exact", "relative"] = "exact"
    max_turns: int = Field(default=40, ge=1, le=400)
    starting_cash: int | None = Field(
        default=None,
        ge=0,
        description="Per-level economy budget (overrides pack default; "
        "engine default 5000 when unset everywhere).",
    )
    # Procedural-compliance family: any agent command whose tool name is
    # in this list increments signals.tool_violations (bench-side track,
    # see eval_core). Use as a fail clause via the
    # `tool_violations_gte` predicate — usually `tool_violations_gte: 1`
    # for a strict zero-tolerance allowlist. Empty list = no constraint.
    forbidden_tools: list[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    """A named runnable configuration of one pack: pins a difficulty
    `level` and the observation `fog_mode`. The same setup at
    `fog_mode: structured` (text 'Unexplored regions') vs `vision`
    (PNG minimap) becomes two distinct cells, so text-vs-vision is a
    first-class comparison the YAML declares (not just a CLI flag)."""

    name: str = Field(..., description="cell suffix, e.g. easy-structured")
    level: LevelName
    fog_mode: Literal["vision", "structured"] = "vision"
    # Optional override of the level's objective_coords for this cell.
    objective_coords: Literal["exact", "relative"] | None = None

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"config name must be a slug, got {v!r}")
        return v


class CompiledLevel(BaseModel):
    """A single runnable level: validated engine scenario + conditions."""

    model_config = {"arbitrary_types_allowed": True}

    pack_id: str
    level: LevelName
    scenario: ScenarioDefinition
    win_condition: WinCondition
    fail_condition: WinCondition | None
    max_turns: int
    meta: ScenarioMeta
    starting_cash: int | None = None
    map_supported: bool = Field(
        ..., description="False => Rust lacks this map (Phase 3 gate)"
    )
    # Observation channel + the cell label. config_name is None for
    # legacy level cells (pack:level); set for declared configs
    # (pack:config_name).
    fog_mode: str = "vision"
    config_name: str | None = None
    objective_coords: Literal["exact", "relative"] = "exact"
    forbidden_tools: list[str] = Field(default_factory=list)
    # Wave-9 mid-episode scripted events (spawn reinforcements,
    # destroy actors, shorten the deadline). Parsed straight-through
    # to the Rust scenario YAML by `_scenario_to_tmp_yaml` — the
    # engine's `oramap::parse_scenario_yaml` handles the schema.
    # `ScenarioDefinition` (training) doesn't know about this field
    # so it's preserved on the CompiledLevel instead of the inner
    # scenario, and re-attached at YAML-write time.
    scheduled_events: list[dict[str, Any]] = Field(default_factory=list)


class ScenarioPack(BaseModel):
    """The contributor-authored unit. One file = one decision problem."""

    meta: ScenarioMeta
    base_map: str | dict[str, Any] = Field(
        default="rush-hour-arena",
        description="Logical map id, OR a generator spec "
        "{generator: arena, width:.., height:.., cordon:..} that is "
        "materialized to a real .oramap at compile time (see mapgen).",
    )
    base: dict[str, Any] = Field(
        ..., description="Shared ScenarioDefinition fields (actors, factions, tools…)"
    )
    starting_cash: int | None = Field(
        default=None,
        ge=0,
        description="Pack-wide economy budget; a level may override it.",
    )
    levels: dict[LevelName, Level]
    # Optional named configurations. When present, the eval runs ONE
    # cell per config (pack:config_name) instead of the 3 raw levels —
    # lets a pack declare e.g. easy-structured / easy-vision / medium.
    configs: list[ScenarioConfig] | None = None

    @field_validator("levels")
    @classmethod
    def _all_three(cls, v: dict) -> dict:
        missing = {"easy", "medium", "hard"} - set(v)
        if missing:
            raise ValueError(f"pack must define all levels; missing {sorted(missing)}")
        return v

    def compile(self, level: LevelName, *, map_supported: bool = True) -> CompiledLevel:
        lvl = self.levels[level]
        merged = deep_merge(self.base, lvl.overrides)
        merged.setdefault("name", f"{self.meta.title} [{level}]")
        merged.setdefault("description", lvl.description)
        merged.setdefault("base_map", self.base_map)
        # A generator-spec base_map (pack-level or via overrides) is
        # materialized to a real .oramap id before validation, so the
        # rest of the pipeline sees an ordinary map id.
        from ..mapgen import resolve_base_map

        merged["base_map"] = resolve_base_map(merged["base_map"])
        # A declared scripted opponent (enemy.bot/bot_type) must name a
        # known behaviour — fail fast at load, not mid-eval.
        from ..botgen import validate_enemy_bot

        validate_enemy_bot(merged.get("enemy"))
        # Validate against the real engine model so a broken level fails
        # at load time, not mid-eval.
        scenario = ScenarioDefinition(**merged)
        # Wave-9: lift the merged `scheduled_events:` (if any) so
        # `_scenario_to_tmp_yaml` can reattach it to the engine YAML.
        # ScenarioDefinition ignores the field (extra='ignore') so
        # without this step the events would be silently dropped.
        sched_events = list(merged.get("scheduled_events") or [])
        return CompiledLevel(
            pack_id=self.meta.id,
            level=level,
            scenario=scenario,
            win_condition=lvl.win_condition,
            fail_condition=lvl.fail_condition,
            max_turns=lvl.max_turns,
            meta=self.meta,
            starting_cash=lvl.starting_cash
            if lvl.starting_cash is not None
            else self.starting_cash,
            map_supported=map_supported,
            objective_coords=lvl.objective_coords,
            forbidden_tools=list(lvl.forbidden_tools or []),
            scheduled_events=sched_events,
        )

    def config_names(self) -> list[str]:
        return [c.name for c in (self.configs or [])]

    def compile_config(
        self, name: str, *, map_supported: bool = True
    ) -> CompiledLevel:
        """Compile a declared config: its `level` + `fog_mode`, with
        the cell label = the config name (pack:config_name)."""
        cfg = next(
            (c for c in (self.configs or []) if c.name == name), None
        )
        if cfg is None:
            raise KeyError(f"no config {name!r} in pack {self.meta.id}")
        cl = self.compile(cfg.level, map_supported=map_supported)
        cl.fog_mode = cfg.fog_mode
        cl.config_name = cfg.name
        if cfg.objective_coords is not None:
            cl.objective_coords = cfg.objective_coords
        return cl
