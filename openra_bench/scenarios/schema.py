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
    max_turns: int = Field(default=40, ge=1, le=400)
    starting_cash: int | None = Field(
        default=None,
        ge=0,
        description="Per-level economy budget (overrides pack default; "
        "engine default 5000 when unset everywhere).",
    )


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


class ScenarioPack(BaseModel):
    """The contributor-authored unit. One file = one decision problem."""

    meta: ScenarioMeta
    base_map: str = Field(
        default="rush-hour-arena",
        description="Logical map id; loader maps to a Rust-supported map",
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
        # Validate against the real engine model so a broken level fails
        # at load time, not mid-eval.
        scenario = ScenarioDefinition(**merged)
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
        )
