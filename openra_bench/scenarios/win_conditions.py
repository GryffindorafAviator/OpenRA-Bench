"""Declarative, composable win conditions.

Contributors express the "custom bot win con" entirely in YAML — no
Python. A condition is a tree of composites (`all_of` / `any_of` / `not`)
over leaf predicates evaluated against a `WinContext` (the per-turn
adapter signals plus the rendered state). The same tree also expresses
*failure* conditions, so a scenario can be lost as well as won.

Leaf predicates (key: value):
    explored_pct_gte: float          map % revealed >= value
    enemies_discovered_gte: int      distinct enemy units seen >= value
    buildings_discovered_gte: int    distinct enemy buildings seen >= value
    units_killed_gte: int            agent kill count >= value
    units_lost_lte: int              agent units lost <= value (constraint)
    within_ticks: int                current game tick <= value (deadline)
    after_ticks: int                 current game tick >= value
    reach_region: {x,y,radius}       any agent unit within radius of (x,y)
    all_units_in_region: {x,y,radius}  every agent unit within radius

Adding a predicate = one entry in `_PREDICATES`. Keep them pure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, model_validator


@dataclass
class WinContext:
    """Everything a predicate may read. Pure data, no engine handles."""

    signals: Any  # rust_adapter.EpisodeSignals
    render_state: dict  # rust_adapter.RustObsAdapter.render_state()


def _agent_units(ctx: WinContext) -> list[dict]:
    return ctx.render_state.get("units_summary", []) or []


def _in_radius(units: list[dict], x: int, y: int, r: float) -> int:
    return sum(1 for u in units if (u["cell_x"] - x) ** 2 + (u["cell_y"] - y) ** 2 <= r * r)


# Each predicate: (ctx, value) -> bool. Pure and side-effect free.
_PREDICATES: dict[str, Callable[[WinContext, Any], bool]] = {
    "explored_pct_gte": lambda c, v: c.signals.explored_percent >= float(v),
    "enemies_discovered_gte": lambda c, v: len(c.signals.enemies_seen_ids) >= int(v),
    "buildings_discovered_gte": lambda c, v: len(c.signals.enemy_buildings_seen_ids)
    >= int(v),
    "units_killed_gte": lambda c, v: c.signals.units_killed >= int(v),
    "units_lost_lte": lambda c, v: c.signals.units_lost <= int(v),
    "within_ticks": lambda c, v: c.signals.game_tick <= int(v),
    "after_ticks": lambda c, v: c.signals.game_tick >= int(v),
    "reach_region": lambda c, v: _in_radius(
        _agent_units(c), int(v["x"]), int(v["y"]), float(v.get("radius", 3))
    )
    >= 1,
    "all_units_in_region": lambda c, v: len(_agent_units(c)) > 0
    and _in_radius(_agent_units(c), int(v["x"]), int(v["y"]), float(v.get("radius", 3)))
    == len(_agent_units(c)),
    # S9 economy / production constraints (require the engine economy
    # subsystem; 0/empty on movement-only scenarios).
    "own_units_gte": lambda c, v: len(_agent_units(c)) >= int(v),
    "cash_gte": lambda c, v: c.signals.cash >= int(v),
    "resources_gte": lambda c, v: c.signals.resources >= int(v),
    # Final economy value = spendable cash + stored (uncashed) resources.
    # The scoring target for the harvest/silo economy scenarios.
    "economy_value_gte": lambda c, v: (c.signals.cash + c.signals.resources)
    >= int(v),
    "power_surplus_gte": lambda c, v: (
        c.signals.power_provided - c.signals.power_drained
    )
    >= int(v),
    "has_building": lambda c, v: str(v).lower() in c.signals.own_building_types,
    "buildings_owned_gte": lambda c, v: len(c.signals.own_building_types) >= int(v),
    # Total agent buildings (counts duplicates, unlike buildings_owned_gte
    # which counts distinct types). For "build exactly/at least N".
    "building_total_gte": lambda c, v: len(c.signals.own_buildings) >= int(v),
    # >= n agent buildings of a given type. {type: powr, n: 2}
    "building_count_gte": lambda c, v: sum(
        1 for (t, _, _) in c.signals.own_buildings if t == str(v["type"]).lower()
    )
    >= int(v.get("n", 1)),
    # >= count agent buildings (optionally typed) within radius of (x,y):
    # "defenses to the east", "found a base near the ridge".
    "building_in_region": lambda c, v: sum(
        1
        for (t, bx, by) in c.signals.own_buildings
        if (not v.get("type") or t == str(v["type"]).lower())
        and (bx - int(v["x"])) ** 2 + (by - int(v["y"])) ** 2
        <= float(v.get("radius", 5)) ** 2
    )
    >= int(v.get("count", 1)),
}

LEAF_KEYS = frozenset(_PREDICATES)
COMPOSITE_KEYS = frozenset({"all_of", "any_of", "not"})


class WinCondition(BaseModel):
    """One node: exactly one composite OR one-or-more leaf predicates.

    Leaf form (implicit AND over keys):
        {explored_pct_gte: 60, within_ticks: 6000}
    Composite form:
        {any_of: [{...}, {...}]}   {not: {...}}
    """

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _check_keys(self) -> "WinCondition":
        keys = set(self.__pydantic_extra__ or {})
        if not keys:
            raise ValueError("win_condition node is empty")
        unknown = keys - LEAF_KEYS - COMPOSITE_KEYS
        if unknown:
            raise ValueError(
                f"unknown win-condition keys {sorted(unknown)}; "
                f"valid leaves={sorted(LEAF_KEYS)} composites={sorted(COMPOSITE_KEYS)}"
            )
        if keys & COMPOSITE_KEYS and keys & LEAF_KEYS:
            raise ValueError("cannot mix composite and leaf keys in one node")
        if len(keys & COMPOSITE_KEYS) > 1:
            raise ValueError("at most one composite key per node")
        return self

    def evaluate(self, ctx: WinContext) -> bool:
        node = dict(self.__pydantic_extra__ or {})
        if "all_of" in node:
            return all(WinCondition(**c).evaluate(ctx) for c in node["all_of"])
        if "any_of" in node:
            return any(WinCondition(**c).evaluate(ctx) for c in node["any_of"])
        if "not" in node:
            return not WinCondition(**node["not"]).evaluate(ctx)
        return all(_PREDICATES[k](ctx, v) for k, v in node.items())


def evaluate(cond: WinCondition | dict | None, ctx: WinContext) -> bool:
    if cond is None:
        return False
    if isinstance(cond, dict):
        cond = WinCondition(**cond)
    return cond.evaluate(ctx)
