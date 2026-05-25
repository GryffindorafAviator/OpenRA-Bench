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
# `tanya` is the Allied hero infantry (added to the engine on the
# wip-tanya branch). The training-side VALID_ACTOR_TYPES is sourced
# from the historical openra_env.game_data table that pre-dates her;
# the engine accepts her actor entry already, we just need the bench
# validator to recognise the type.
_orts.VALID_ACTOR_TYPES |= {"tanya"}
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


# The perception ablation grid: observation channel × fog of war.
#
# Three channels:
#   structured — text briefing + a text 'Unexplored regions' block;
#                NO image. The text-only condition.
#   vision     — text briefing + PNG minimap. The multimodal "both
#                available" condition — note the text briefing already
#                enumerates units/enemies, so the image is a SUPPLEMENT.
#   image      — image-PRIMARY: the text briefing is redacted of every
#                coordinate (and the enemy line dropped); the PNG, with
#                legible per-unit labels, is the ONLY source of spatial
#                state. The clean "can the model read a minimap" probe.
#
# Fog axis: bare name ⇒ fog ON (canonical scoring modality); the
# `-clear` variant reveals the whole map (engine `reveal_map: true` ⇒
# every enemy observed, `explored_percent` 100). A `-clear` cell is a
# perfect-information CONTROL for measuring the perception cost — the
# no-cheat bar applies only to the fogged cells.
PERCEPTION_MODES = (
    "structured", "structured-clear",
    "vision", "vision-clear",
    "image", "image-clear",
)
FogMode = Literal[
    "structured", "structured-clear",
    "vision", "vision-clear",
    "image", "image-clear",
]


class ScenarioConfig(BaseModel):
    """A named runnable configuration of one pack: pins a difficulty
    `level` and the observation `fog_mode`. `fog_mode` spans the 2×2
    perception grid — channel (`vision` PNG minimap vs `structured`
    text 'Unexplored regions') × fog (on, or `-clear` ⇒ no fog) — so
    text-vs-vision AND fogged-vs-clear are first-class comparisons the
    YAML declares, not just CLI flags."""

    name: str = Field(..., description="cell suffix, e.g. easy-structured")
    level: LevelName
    fog_mode: FogMode = "vision"
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
    # Observation modality (channel × fog — see PERCEPTION_MODES) + the
    # cell label. config_name is None for legacy level cells
    # (pack:level); set for declared configs / sweep cells.
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
    # Resource-wave `ore_patches:` — list of `{x, y, amount, radius}`
    # dicts the engine materialises into disks of harvestable ore at
    # world-build time. ScenarioDefinition (training) doesn't know
    # about this field so it's preserved on the CompiledLevel and
    # re-attached at YAML-write time, mirroring `scheduled_events`.
    #
    # Engine clamp footgun (`openra-sim/src/resource.rs`,
    # `seed_ore_patch`): per-cell density is
    # `clamp(ceil(amount / passable_cells), 1, 12)`. The cap of 12
    # ore/cell means a small patch with a big `amount` is wasted —
    # `amount: 2000, radius: 2` (~13 cells) yields only ~156 total ore
    # = ~3900 cash ceiling. To get a useful econ patch, choose
    # `amount` so that `amount >= passable_cells * 12` (i.e. fill the
    # disc to the cap), then GROW the disc via `radius` rather than
    # the per-cell `amount`. Recommended ratio: `amount =~ 12 * pi *
    # radius^2 * 1.2` (the 1.2 covers cells on the disc that turn out
    # to be impassable). At 50 cr/ore the cash ceiling of a patch is
    # `12 * passable_cells * 50 = 600 * passable_cells`. Triaged in
    # ENGINE_FOLLOWUPS_TRIAGE.md finding #2.
    ore_patches: list[dict[str, Any]] = Field(default_factory=list)
    # Naval-MVP overlay: explicit `water_cells:` (list of `[x, y]`) and
    # `water_rect:` (a single `[x, y, w, h]`) blocks declare WATER
    # cells on top of an otherwise-grass map. The engine treats each
    # such cell as ground-impassable and ship-passable. Same lift
    # pattern as `scheduled_events` / `ore_patches` —
    # `ScenarioDefinition` doesn't know about these fields so they
    # ride on the CompiledLevel and are re-attached by
    # `_scenario_to_tmp_yaml`.
    water_cells: list[list[int]] = Field(default_factory=list)
    water_rect: list[int] | None = None
    # Pack-wide no-fog flag, lifted from `ScenarioPack.reveal_map` at
    # compile time. The property below OR's this with the
    # `fog_mode`-derived value so both paths produce the same engine
    # effect (`reveal_map: true` in `_scenario_to_tmp_yaml`).
    pack_reveal_map: bool = False
    # Per-scenario production-tick multiplier (default None ⇒ engine
    # default 1.0 ⇒ unchanged behaviour). Lifted from the pack's
    # `base.build_speed_multiplier` at compile time; the
    # `_scenario_to_tmp_yaml` emitter passes it through to the engine
    # YAML as a top-level key. The single contributor of this field is
    # `adversarial-1v1-macro` (4.0× ⇒ snappier 1v1 episodes); every
    # other pack stays on 1.0 by leaving this None.
    build_speed_multiplier: float | None = None

    @property
    def reveal_map(self) -> bool:
        """No-fog cell? True if either (a) the pack declares
        top-level `reveal_map: true` (e.g. close-range duel packs
        where scouting is not the advertised capability), or (b) the
        fog_mode is a `-clear` perception-ablation cell.
        `_scenario_to_tmp_yaml` emits `reveal_map: true` to the engine
        when this returns true."""
        return self.pack_reveal_map or self.fog_mode.endswith("-clear")

    @property
    def obs_channel(self) -> str:
        """The observation channel — `structured` (text only), `image`
        (image-primary: text redacted, PNG is the sole spatial source),
        or `vision` (text + PNG) — independent of the fog axis."""
        if self.fog_mode.startswith("structured"):
            return "structured"
        if self.fog_mode.startswith("image"):
            return "image"
        return "vision"


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
    # Pack-wide top-level engine extras (RE-APPLIED — was reverted by an
    # agent and the regression cost a full debug cycle). These live next
    # to `base:` so a contributor can declare one set of `ore_patches:` /
    # `water_cells:` / `scheduled_events:` once for the whole pack
    # instead of restating them in every level's `overrides:`. A level
    # may still override by restating the field inside `overrides:`
    # (compile() falls back to the pack-level value only when the merged
    # level value is empty). Without this declaration pydantic silently
    # DROPS the top-level key, so `compile()` sees an empty list and the
    # engine seeds zero ore / zero water cells / fires no events for
    # easy + medium tiers (hard tier worked only because it re-declared
    # the field inside its `overrides:` block).
    ore_patches: list[dict[str, Any]] = Field(default_factory=list)
    water_cells: list[list[int]] = Field(default_factory=list)
    water_rect: list[int] | None = None
    scheduled_events: list[dict[str, Any]] = Field(default_factory=list)
    # Pack-wide no-fog flag. When true the engine reveals the whole
    # map to the agent regardless of `fog_mode` — used by packs whose
    # advertised capability is NOT scouting (e.g. close-range duels)
    # so a contributor doesn't have to declare a separate `-clear`
    # config. Engine wiring: `oramap.rs::ScenarioDef.reveal_map`,
    # plumbed by `eval_core.py::_scenario_to_tmp_yaml`. The
    # CompiledLevel `.reveal_map` property OR's this with the
    # `fog_mode`-derived value so both paths produce the same effect.
    reveal_map: bool = False
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
        # Lift pack-wide → level-override; fall back to pack-level when
        # a level didn't restate the field. (Was reverted by an agent;
        # re-applied with the same pattern that was originally tested.)
        sched_events = list(
            merged.get("scheduled_events") or self.scheduled_events or []
        )
        ore_patches = list(merged.get("ore_patches") or self.ore_patches or [])
        water_cells = [
            list(c) for c in (merged.get("water_cells") or self.water_cells or [])
        ]
        water_rect = merged.get("water_rect")
        if water_rect is None:
            water_rect = self.water_rect
        # Lift `build_speed_multiplier` (default None ⇒ engine 1.0).
        # The merged `base` may carry it directly, or the
        # ScenarioDefinition validator will surface it via the
        # `scenario` object. Read both for robustness.
        build_speed_multiplier = (
            merged.get("build_speed_multiplier")
            if "build_speed_multiplier" in merged
            else getattr(scenario, "build_speed_multiplier", None)
        )
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
            ore_patches=ore_patches,
            water_cells=water_cells,
            water_rect=list(water_rect) if water_rect is not None else None,
            pack_reveal_map=self.reveal_map,
            build_speed_multiplier=build_speed_multiplier,
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
