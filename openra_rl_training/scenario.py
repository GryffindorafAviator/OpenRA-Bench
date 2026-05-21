"""Pydantic models for scenario and curriculum YAML definitions.

Scenarios define custom starting conditions for RL training episodes:
units, positions, stances, factions, and termination conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from openra_env.game_data import RA_BUILDINGS, RA_UNITS
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# All valid actor types that can be placed on maps
VALID_ACTOR_TYPES = set(RA_UNITS.keys()) | set(RA_BUILDINGS.keys())

# Unit stances matching OpenRA's UnitStance enum
STANCE_HOLD_FIRE = 0
STANCE_RETURN_FIRE = 1
STANCE_DEFEND = 2
STANCE_ATTACK_ANYTHING = 3

STANCE_NAMES = {
    STANCE_HOLD_FIRE: "HoldFire",
    STANCE_RETURN_FIRE: "ReturnFire",
    STANCE_DEFEND: "Defend",
    STANCE_ATTACK_ANYTHING: "AttackAnything",
}


# ── Randomization models ─────────────────────────────────────────────────────


class TypeFilter(BaseModel):
    """Filter-based type randomization: pick a random unit matching criteria."""

    category: str = Field(..., description="Unit category: infantry, vehicle, aircraft, ship")
    side: str = Field(default="both", description="Faction filter: allied, soviet, both")
    max_cost: Optional[int] = Field(default=None, description="Maximum unit cost")
    min_cost: Optional[int] = Field(default=None, description="Minimum unit cost")
    armor: Optional[str] = Field(default=None, description="Armor type: none, light, heavy")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        v = v.lower()
        valid = {"infantry", "vehicle", "aircraft", "ship"}
        if v not in valid:
            raise ValueError(f"category must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("side")
    @classmethod
    def validate_side(cls, v: str) -> str:
        v = v.lower()
        valid = {"allied", "soviet", "both"}
        if v not in valid:
            raise ValueError(f"side must be one of {sorted(valid)}, got '{v}'")
        return v

    @field_validator("armor")
    @classmethod
    def validate_armor(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.lower()
            valid = {"none", "light", "heavy"}
            if v not in valid:
                raise ValueError(f"armor must be one of {sorted(valid)}, got '{v}'")
        return v

    def matching_types(self) -> list[str]:
        """Return all RA_UNITS keys matching this filter."""
        results = []
        for utype, data in RA_UNITS.items():
            if data.get("category") != self.category:
                continue
            unit_side = data.get("side", "both")
            if self.side != "both" and unit_side not in (self.side, "both"):
                continue
            cost = data.get("cost", 0)
            if self.max_cost is not None and cost > self.max_cost:
                continue
            if self.min_cost is not None and cost < self.min_cost:
                continue
            if self.armor is not None and data.get("armor") != self.armor:
                continue
            results.append(utype)
        return sorted(results)


class PositionOffset(BaseModel):
    """Offset-based position randomization: random within ±offset of base."""

    base: tuple[int, int] = Field(..., description="Base position [x, y]")
    offset: int = Field(..., description="Max offset in cells (applies to both x and y)")

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError(f"offset must be 1-50, got {v}")
        return v


class HealthRange(BaseModel):
    """Range-based health randomization."""

    min: int = Field(default=1, description="Minimum health percentage")
    max: int = Field(default=100, description="Maximum health percentage")

    @model_validator(mode="after")
    def validate_range(self) -> "HealthRange":
        if self.min < 1 or self.max > 100:
            raise ValueError(f"Health range must be 1-100, got {self.min}-{self.max}")
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class ActorRandomization(BaseModel):
    """Per-field randomization options for an actor placement."""

    type: Optional[Union[list[str], TypeFilter]] = Field(
        default=None, description="Type alternatives: list of names or category filter"
    )
    position: Optional[Union[list[tuple[int, int]], PositionOffset]] = Field(
        default=None, description="Position alternatives: preset list or offset from base"
    )
    stance: Optional[list[int]] = Field(default=None, description="Stance alternatives (0-3)")
    health: Optional[HealthRange] = Field(default=None, description="Health range {min, max}")
    facing: Optional[list[int]] = Field(default=None, description="Facing alternatives (0-1023)")

    @field_validator("type")
    @classmethod
    def validate_type_alternatives(cls, v: Optional[Union[list[str], TypeFilter]]):
        if isinstance(v, list):
            if not v:
                raise ValueError("type list must not be empty")
            for t in v:
                if t.lower() not in VALID_ACTOR_TYPES:
                    raise ValueError(f"Unknown actor type in randomize.type: '{t}'")
        return v

    @field_validator("position")
    @classmethod
    def validate_position_alternatives(
        cls, v: Optional[Union[list[tuple[int, int]], PositionOffset]]
    ):
        if isinstance(v, list) and not v:
            raise ValueError("position list must not be empty")
        return v

    @field_validator("stance")
    @classmethod
    def validate_stance_alternatives(cls, v: Optional[list[int]]):
        if v is not None:
            if not v:
                raise ValueError("stance list must not be empty")
            for s in v:
                if s < 0 or s > 3:
                    raise ValueError(f"Stance must be 0-3, got {s}")
        return v

    @field_validator("facing")
    @classmethod
    def validate_facing_alternatives(cls, v: Optional[list[int]]):
        if v is not None:
            if not v:
                raise ValueError("facing list must not be empty")
            for f in v:
                if f < 0 or f > 1023:
                    raise ValueError(f"Facing must be 0-1023, got {f}")
        return v


# ── Core scenario models ─────────────────────────────────────────────────────


class ActorPlacement(BaseModel):
    """A unit or building to spawn at game start."""

    type: str = Field(..., description="Actor type (e.g., '2tnk', 'e1', 'fact')")
    owner: Literal["agent", "enemy", "neutral"] = Field(
        default="agent", description="Which player owns this actor"
    )
    position: tuple[int, int] = Field(..., description="Cell coordinates [x, y]")
    stance: int = Field(
        default=STANCE_ATTACK_ANYTHING,
        description="0=HoldFire, 1=ReturnFire, 2=Defend, 3=AttackAnything",
    )
    health: int = Field(default=100, description="HP percentage 1-100")
    facing: int = Field(default=-1, description="-1=auto, 0-1023 WAngle")
    count: int = Field(default=1, description="Spawn N copies with auto-offset positions")
    spawn_point: Optional[int] = Field(
        default=None,
        description="Spawn point group (0-N). If set, only included when this spawn point is selected. "
                    "None = always included (enemies, neutral).",
    )
    randomize: Optional[ActorRandomization] = Field(
        default=None,
        description="Per-field randomization options (resolved before map generation)",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower()
        if v not in VALID_ACTOR_TYPES:
            raise ValueError(
                f"Unknown actor type '{v}'. "
                f"Valid units: {sorted(RA_UNITS.keys())[:10]}... "
                f"Valid buildings: {sorted(RA_BUILDINGS.keys())[:10]}..."
            )
        return v

    @field_validator("stance")
    @classmethod
    def validate_stance(cls, v: int) -> int:
        if v < 0 or v > 3:
            raise ValueError(f"Stance must be 0-3, got {v}")
        return v

    @field_validator("health")
    @classmethod
    def validate_health(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"Health must be 1-100, got {v}")
        return v

    @field_validator("facing")
    @classmethod
    def validate_facing(cls, v: int) -> int:
        if v != -1 and (v < 0 or v > 1023):
            raise ValueError(f"Facing must be -1 (auto) or 0-1023, got {v}")
        return v

    @field_validator("count")
    @classmethod
    def validate_count(cls, v: int) -> int:
        if v < 1 or v > 50:
            raise ValueError(f"Count must be 1-50, got {v}")
        return v

    @property
    def is_building(self) -> bool:
        return self.type in RA_BUILDINGS


class PlayerSetup(BaseModel):
    """Configuration for the agent player."""

    faction: Literal["allies", "soviet", "random"] = Field(
        default="random", description="Player faction"
    )
    cash: int = Field(default=0, description="Starting cash override")

    @field_validator("cash")
    @classmethod
    def validate_cash(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Cash must be non-negative, got {v}")
        return v


class EnemySetup(PlayerSetup):
    """Configuration for the enemy player."""

    bot_type: str = Field(
        default="", description="AI bot type (empty = no AI, stance-only behavior)"
    )


class TerminationConfig(BaseModel):
    """When to end a scenario episode."""

    max_ticks: int = Field(default=5000, description="Tick limit (0 = unlimited)")
    max_time: Optional[float] = Field(
        default=None,
        description="Time limit in seconds (overrides max_ticks). 25 ticks = 1 second.",
    )
    agent_units_killed: bool = Field(
        default=True, description="End as 'lose' when all agent units destroyed"
    )
    enemy_units_killed: bool = Field(
        default=True, description="End as 'win' when all enemy units/buildings destroyed"
    )

    @field_validator("max_ticks")
    @classmethod
    def validate_max_ticks(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_ticks must be non-negative, got {v}")
        return v

    @model_validator(mode="after")
    def resolve_max_time(self) -> "TerminationConfig":
        """Convert max_time (seconds) to max_ticks if specified."""
        if self.max_time is not None:
            self.max_ticks = int(self.max_time * 25)
        return self


class ScenarioDefinition(BaseModel):
    """Complete scenario definition loaded from YAML."""

    name: str = Field(..., description="Scenario display name")
    description: str = Field(default="", description="Human-readable description")
    base_map: str = Field(default="singles.oramap", description="Base map filename for terrain")
    agent: PlayerSetup = Field(default_factory=PlayerSetup)
    enemy: EnemySetup = Field(default_factory=EnemySetup)
    actors: list[ActorPlacement] = Field(..., description="Units/buildings to spawn")
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    reward: dict[str, float] = Field(default_factory=dict, description="Override reward weights")
    reward_calibration: dict[str, float] = Field(
        default_factory=dict,
        description="Manual overrides for reward calibration constants (auto-computed if empty)",
    )
    tools: list[str] = Field(default_factory=list, description="Allowed tool names (empty = all)")
    interrupts: dict[str, bool] = Field(
        default_factory=dict,
        description="Override interrupt signals: signal_name → enabled/disabled. All enabled by default.",
    )
    planning: bool = Field(default=False, description="Enable pre-game planning phase")
    difficulty: int = Field(default=1, description="Difficulty level for ordering")
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")

    @field_validator("tools")
    @classmethod
    def strip_internal_tools(cls, v: list[str]) -> list[str]:
        """Remove internal-only tools that the LLM should never call directly."""
        _INTERNAL_TOOLS = {"get_game_state", "surrender"}
        return [t for t in v if t not in _INTERNAL_TOOLS]

    @field_validator("actors")
    @classmethod
    def validate_actors_not_empty(cls, v: list[ActorPlacement]) -> list[ActorPlacement]:
        if not v:
            raise ValueError("Scenario must have at least one actor")
        return v

    @model_validator(mode="after")
    def validate_has_agent_actor(self) -> "ScenarioDefinition":
        agent_actors = [a for a in self.actors if a.owner == "agent"]
        if not agent_actors:
            raise ValueError("Scenario must have at least one agent-owned actor")
        return self

    @property
    def agent_actors(self) -> list[ActorPlacement]:
        return [a for a in self.actors if a.owner == "agent"]

    @property
    def enemy_actors(self) -> list[ActorPlacement]:
        return [a for a in self.actors if a.owner == "enemy"]

    @property
    def neutral_actors(self) -> list[ActorPlacement]:
        return [a for a in self.actors if a.owner == "neutral"]


def load_scenario(path: str | Path) -> ScenarioDefinition:
    """Load a scenario definition from a YAML file.

    Args:
        path: Path to the scenario YAML file.

    Returns:
        Parsed and validated ScenarioDefinition.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty scenario file: {path}")

    logger.info("Loading scenario '%s' from %s", data.get("name", "?"), path)
    return ScenarioDefinition.model_validate(data)


def load_scenario_from_string(yaml_string: str) -> ScenarioDefinition:
    """Load a scenario definition from a YAML string.

    Args:
        yaml_string: YAML content.

    Returns:
        Parsed and validated ScenarioDefinition.
    """
    data = yaml.safe_load(yaml_string)
    if data is None:
        raise ValueError("Empty scenario YAML")
    return ScenarioDefinition.model_validate(data)
