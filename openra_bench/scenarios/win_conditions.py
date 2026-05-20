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


def _waypoint_sequence(c: "WinContext", v: dict) -> bool:
    """Ordered-visit latch: the agent must reach the waypoints in
    `points` IN ORDER. Progress is monotonic and persists across turns
    on the per-episode `signals.seq_progress` (keyed by `id`), so a
    unit may leave a waypoint after tagging it. The next waypoint only
    advances when ≥`n` agent units are within its radius — reaching a
    later point early does NOT skip ahead, and skipping a waypoint
    means the sequence can never complete. Satisfied once every point
    has been reached in order.

    value: {id: str, points: [{x,y[,radius][,label]}...],
            radius: float = 6, n: int = 1}
    """
    pts = v.get("points") or []
    if not pts:
        return False
    store = getattr(c.signals, "seq_progress", None)
    if store is None or not isinstance(store, dict):
        store = {}
        try:
            c.signals.seq_progress = store  # type: ignore[attr-defined]
        except Exception:  # frozen/stub signals in unit tests
            pass
    key = str(v.get("id", id(v)))
    idx = int(store.get(key, 0))
    units = _agent_units(c)
    r_def = float(v.get("radius", 6))
    need = int(v.get("n", 1))
    # Advance through every consecutive waypoint currently satisfied
    # (waypoints are spaced > 2r apart, so normally ≤1 per turn).
    while idx < len(pts):
        p = pts[idx]
        if _in_radius(
            units, int(p["x"]), int(p["y"]),
            float(p.get("radius", r_def)),
        ) >= need:
            idx += 1
        else:
            break
    store[key] = idx
    return idx >= len(pts)


# Each predicate: (ctx, value) -> bool. Pure and side-effect free.
_PREDICATES: dict[str, Callable[[WinContext, Any], bool]] = {
    "explored_pct_gte": lambda c, v: c.signals.explored_percent >= float(v),
    "enemies_discovered_gte": lambda c, v: len(c.signals.enemies_seen_ids) >= int(v),
    "buildings_discovered_gte": lambda c, v: len(c.signals.enemy_buildings_seen_ids)
    >= int(v),
    "units_killed_gte": lambda c, v: c.signals.units_killed >= int(v),
    # Eliminate the enemy's key economic structures (the training
    # strategy design: destroy fact+proc, NOT brute-force the whole
    # strong enemy). Total count, or all of a named type set.
    "enemy_buildings_destroyed_gte": lambda c, v: getattr(
        c.signals, "enemy_buildings_destroyed", 0
    )
    >= int(v),
    "enemy_key_buildings_destroyed": lambda c, v: all(
        getattr(c.signals, "enemy_buildings_destroyed_types", {}).get(
            str(t).lower(), 0
        )
        >= 1
        for t in (v["types"] if isinstance(v, dict) else v)
    ),
    # Region-scoped variant: every named key type must have been
    # destroyed WITHIN radius of (x,y). Lets a scenario require razing
    # fact+proc at TWO separate bases (one per squad) — the type-only
    # count above is satisfied by levelling a single base, so it cannot
    # enforce genuine simultaneous two-group control. {x,y,radius,types}
    "enemy_key_buildings_destroyed_in_region": lambda c, v: all(
        any(
            str(bt).lower() == str(t).lower()
            and (bx - int(v["x"])) ** 2 + (by - int(v["y"])) ** 2
            <= float(v.get("radius", 8)) ** 2
            for (bt, bx, by) in getattr(
                c.signals, "enemy_buildings_destroyed_records", []
            )
        )
        for t in v["types"]
    ),
    "units_lost_lte": lambda c, v: c.signals.units_lost <= int(v),
    "within_ticks": lambda c, v: c.signals.game_tick <= int(v),
    "after_ticks": lambda c, v: c.signals.game_tick >= int(v),
    "reach_region": lambda c, v: _in_radius(
        _agent_units(c), int(v["x"]), int(v["y"]), float(v.get("radius", 3))
    )
    >= 1,
    # ≥ n agent units within radius of (x,y). Lets a scenario require a
    # real force SPLIT ("≥2 units in EACH of 3 regions"), which
    # reach_region (≥1) cannot express.
    "units_in_region_gte": lambda c, v: _in_radius(
        _agent_units(c), int(v["x"]), int(v["y"]), float(v.get("radius", 3))
    )
    >= int(v.get("n", 1)),
    # Type-filtered region count: ≥ n agent units of a given type within
    # radius of (x,y). Lets a scenario enforce SQUAD IDENTITY at a
    # waypoint ("Squad A — 3 jeeps — at P1; Squad B — 3 tanks — at
    # P2") so a single-squad tour cannot satisfy two type-distinct
    # clauses in series. Pair with `then:` for ordered squad handoff.
    # {type, x, y, radius, n}
    "units_of_type_in_region_gte": lambda c, v: sum(
        1 for u in _agent_units(c)
        if str(u.get("type", "")).lower() == str(v["type"]).lower()
        and (u["cell_x"] - int(v["x"])) ** 2 + (u["cell_y"] - int(v["y"])) ** 2
        <= float(v.get("radius", 3)) ** 2
    ) >= int(v.get("n", 1)),
    # Stateful ordered-route latch (see _waypoint_sequence). Lets a
    # scenario require visiting W1→W2→…→Wk IN ORDER (skip/idle ⇒ never
    # satisfied), which stateless region predicates cannot express.
    "waypoint_sequence": lambda c, v: _waypoint_sequence(c, v),
    "all_units_in_region": lambda c, v: len(_agent_units(c)) > 0
    and _in_radius(_agent_units(c), int(v["x"]), int(v["y"]), float(v.get("radius", 3)))
    == len(_agent_units(c)),
    # S9 economy / production constraints (require the engine economy
    # subsystem; 0/empty on movement-only scenarios).
    "own_units_gte": lambda c, v: len(_agent_units(c)) >= int(v),
    # Strict-spec production: agent units of a given type. `_eq` is the
    # no-overproduction teeth (exactly N, not ≥N); `_gte` is handy as a
    # *fail* predicate ("built too many → violation").
    "unit_type_count_eq": lambda c, v: sum(
        1 for u in _agent_units(c)
        if str(u.get("type", "")).lower() == str(v["type"]).lower()
    ) == int(v["n"]),
    "unit_type_count_gte": lambda c, v: sum(
        1 for u in _agent_units(c)
        if str(u.get("type", "")).lower() == str(v["type"]).lower()
    ) >= int(v["n"]),
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
    # Procedural-compliance / strict-toolban family: triggers when the
    # agent has invoked tools on the level's `forbidden_tools` list at
    # least N times this episode. Typical use is in a fail clause as
    # `tool_violations_gte: 1` (zero-tolerance), but >1 also lets a pack
    # tolerate up to N slips before failing. The counter is maintained
    # by eval_core.run_level — see ScenarioPack/Level.forbidden_tools.
    "tool_violations_gte": lambda c, v: getattr(
        c.signals, "tool_violations", 0
    )
    >= int(v),
}

LEAF_KEYS = frozenset(_PREDICATES)
COMPOSITE_KEYS = frozenset({"all_of", "any_of", "not", "then"})


def _then(ctx: WinContext, v: Any) -> bool:
    """Happened-before composite: each clause must become true IN ORDER
    over the course of the episode. Stateful: the latch persists on
    ``signals.then_progress[id]`` so an early clause that stopped being
    true still counts (analogous to ``waypoint_sequence``).

    YAML form::

        then:
          id: scout-then-counter            # stable key (per-episode)
          clauses:
            - {buildings_discovered_gte: 1}    # FIRST
            - {unit_type_count_gte: {type: e3, n: 4}}  # THEN

    Satisfied once every clause has been observed-true AT LEAST ONCE,
    in order. Reaching a later clause without the earlier one being
    latched does NOT advance — this is what makes "counter chosen
    AFTER scout" enforceable (vs the stateless ``all_of`` which is
    satisfied by any state where every clause happens to be true now).
    """
    if not isinstance(v, dict):
        raise ValueError("then: expects a dict with id + clauses")
    clauses = v.get("clauses") or []
    if not clauses:
        return False
    store = getattr(ctx.signals, "then_progress", None)
    if store is None or not isinstance(store, dict):
        store = {}
        try:
            ctx.signals.then_progress = store  # type: ignore[attr-defined]
        except Exception:  # frozen/stub signals in unit tests
            pass
    key = str(v.get("id", id(v)))
    idx = int(store.get(key, 0))
    # Advance through every consecutive clause currently satisfied
    # (typically ≤1 per evaluation, like waypoint_sequence).
    while idx < len(clauses):
        if WinCondition(**clauses[idx]).evaluate(ctx):
            idx += 1
        else:
            break
    store[key] = idx
    return idx >= len(clauses)


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
        if "then" in node:
            return _then(ctx, node["then"])
        return all(_PREDICATES[k](ctx, v) for k, v in node.items())


def evaluate(cond: WinCondition | dict | None, ctx: WinContext) -> bool:
    if cond is None:
        return False
    if isinstance(cond, dict):
        cond = WinCondition(**cond)
    return cond.evaluate(ctx)
