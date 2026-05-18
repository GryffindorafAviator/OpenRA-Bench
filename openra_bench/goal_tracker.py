"""Per-turn goal tracking for playback.

Two complementary views of "how is the agent doing toward the
objective", recorded every turn so a replay shows the trajectory of
intent, not just the final verdict:

* ``leaf_progress`` — walks the scenario's declarative win-condition
  tree and, for every leaf predicate, reports current vs. target and a
  0..1 ratio. Scenario-specific, directly tied to *this* objective.
* ``reward_vector`` — a fixed, normalized, monotone-cumulative vector
  (economy / military / territory / scouting / objective) that is
  scenario-agnostic and therefore comparable across runs and on the
  leaderboard. Mirrors the training rollout's per-turn reward_vector.

``turn_goal`` bundles both side by side plus a scalar
``objective_progress`` and the boolean ``won``.
"""

from __future__ import annotations

from typing import Any

from .scenarios.win_conditions import LEAF_KEYS, WinContext, evaluate

# Current-value extractors for the numeric leaf predicates. Anything not
# listed is treated as a pure boolean gate (ratio = 1.0 iff satisfied).
_CURRENT: dict[str, Any] = {
    "explored_pct_gte": lambda c: c.signals.explored_percent,
    "enemies_discovered_gte": lambda c: len(c.signals.enemies_seen_ids),
    "buildings_discovered_gte": lambda c: len(c.signals.enemy_buildings_seen_ids),
    "units_killed_gte": lambda c: c.signals.units_killed,
    "enemy_buildings_destroyed_gte": lambda c: getattr(
        c.signals, "enemy_buildings_destroyed", 0
    ),
    "units_lost_lte": lambda c: c.signals.units_lost,
    "within_ticks": lambda c: c.signals.game_tick,
    "after_ticks": lambda c: c.signals.game_tick,
    "own_units_gte": lambda c: len(c.render_state.get("units_summary", []) or []),
    "cash_gte": lambda c: c.signals.cash,
    "resources_gte": lambda c: c.signals.resources,
    "economy_value_gte": lambda c: c.signals.cash + c.signals.resources,
    "power_surplus_gte": lambda c: c.signals.power_provided
    - c.signals.power_drained,
    "buildings_owned_gte": lambda c: len(c.signals.own_building_types),
    "building_total_gte": lambda c: len(c.signals.own_buildings),
}
# `_lte` predicates are satisfied while *below* the bound (constraints).
_LTE = frozenset({"units_lost_lte", "within_ticks"})


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def _leaves(node: Any) -> list[dict]:
    """Flatten a win-condition tree to its leaf predicate dicts."""
    if node is None:
        return []
    if not isinstance(node, dict):
        node = dict(getattr(node, "__pydantic_extra__", {}) or {})
    out: list[dict] = []
    for k, v in node.items():
        if k in ("all_of", "any_of"):
            for child in v:
                out.extend(_leaves(child))
        elif k == "not":
            out.extend(_leaves(v))
        elif k in LEAF_KEYS:
            out.append({k: v})
    return out


def leaf_progress(win_condition: Any, ctx: WinContext) -> list[dict]:
    rows: list[dict] = []
    for leaf in _leaves(win_condition):
        (name, target), = leaf.items()
        satisfied = bool(evaluate({name: target}, ctx))
        cur_fn = _CURRENT.get(name)
        if cur_fn is None or not isinstance(target, (int, float)):
            rows.append({
                "name": name, "target": target, "current": None,
                "ratio": 1.0 if satisfied else 0.0, "satisfied": satisfied,
            })
            continue
        cur = cur_fn(ctx)
        tgt = float(target)
        if name in _LTE:
            ratio = 1.0 if satisfied else _clamp(tgt / cur) if cur else 1.0
        else:
            ratio = 1.0 if satisfied else (_clamp(cur / tgt) if tgt else 0.0)
        rows.append({
            "name": name, "target": target, "current": cur,
            "ratio": round(ratio, 4), "satisfied": satisfied,
        })
    return rows


def reward_vector(signals: Any) -> dict[str, float]:
    """Normalized, scenario-agnostic cumulative progress vector.

    All dimensions are 0..1 and (because the underlying signals are
    monotone over an episode) non-decreasing turn over turn — a true
    cumulative tracker, comparable across scenarios.
    """
    cash = getattr(signals, "cash", 0) + getattr(signals, "resources", 0)
    kills = getattr(signals, "units_killed", 0)
    seen = len(getattr(signals, "enemies_seen_ids", []) or []) + len(
        getattr(signals, "enemy_buildings_seen_ids", []) or []
    )
    return {
        "economy": round(_clamp(cash / 10000.0), 4),
        "military": round(_clamp(kills / 10.0), 4),
        "territory": round(_clamp(
            getattr(signals, "explored_percent", 0.0) / 100.0), 4),
        "scouting": round(_clamp(seen / 10.0), 4),
        "objective": 0.0,  # filled by turn_goal once the win cond is known
    }


def turn_goal(win_condition: Any, ctx: WinContext) -> dict:
    leaves = leaf_progress(win_condition, ctx)
    won = bool(evaluate(win_condition, ctx))
    obj = 1.0 if won else (
        round(sum(r["ratio"] for r in leaves) / len(leaves), 4)
        if leaves else 0.0
    )
    rv = reward_vector(ctx.signals)
    rv["objective"] = obj
    return {
        "leaves": leaves,
        "reward_vector": rv,
        "objective_progress": obj,
        "won": won,
    }
