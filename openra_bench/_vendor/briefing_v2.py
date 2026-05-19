"""Briefing formatter v2: one unit per line, prose movement, no arrow.

Drop-in replacement for openra_env.agent.format_state_briefing — same input
state dict, different output layout. Designed to be conservative w.r.t.
information content (same fields, same numbers) and aggressive w.r.t.
readability and natural-language alignment.

Differences vs v1:
  - Per-unit per-line (always), instead of `Nx<type>[id@(x,y),id@(x,y)]`.
  - Movement rendered as ", moving to (X,Y)" instead of "→(X,Y)".
  - Type column space-padded to 5 chars so IDs/positions align across rows.
  - Idle list preserved at end.
  - Buildings, enemies, production lines unchanged from v1 to limit the
    perturbation to the units block (the noisiest part).
"""
from __future__ import annotations

from collections import defaultdict


_TYPE_PAD = 5  # widest common type code is 4 chars (medi, 2tnk, ...) + 1 space


def format_state_briefing_v2(state: dict) -> str:
    if not isinstance(state, dict) or "tick" not in state:
        return ""

    eco = state.get("economy", {})
    tick = state["tick"]
    cash = eco.get("cash", 0)
    ore = eco.get("ore", 0)
    funds = cash + ore

    parts = [
        f"--- TURN BRIEFING (tick {tick}, ~{tick // 25}s game time) ---",
        f"Funds: ${funds} (cash=${cash} + ore=${ore}) | "
        f"Power: {state.get('power_balance', 0):+d} | "
        f"Harvesters: {eco.get('harvester_count', 0)} | "
        f"Explored: {state.get('explored_percent', 0)}%",
    ]

    minimap = state.get("minimap", "")
    if minimap:
        parts.append(minimap)

    buildings = state.get("buildings_summary", [])
    if buildings:
        base_x = sum(b["cell_x"] for b in buildings) // len(buildings)
        base_y = sum(b["cell_y"] for b in buildings) // len(buildings)
        parts.append(f"Base center: ({base_x},{base_y})")

    units = state.get("units_summary", [])
    if units:
        idle_ids = []
        # Sort by (type, id) so the per-line list groups by type visually.
        sorted_units = sorted(units, key=lambda u: (u.get("type", ""), u.get("id", 0)))
        unit_lines = []
        for u in sorted_units:
            uid = u.get("id", "?")
            utype = (u.get("type", "?") + " " * _TYPE_PAD)[:_TYPE_PAD]
            x, y = u.get("cell_x", "?"), u.get("cell_y", "?")
            line = f"  {uid} {utype} @({x},{y})"
            if u.get("target_x") is not None:
                line += f", moving to ({u['target_x']},{u['target_y']})"
            elif not u.get("idle"):
                act = u.get("activity", "")
                if act and act not in ("Idle", "Unknown", "Wait"):
                    line += f", {act.lower()}"
            unit_lines.append(line)
            if u.get("idle") and u.get("can_attack"):
                idle_ids.append(uid)
        parts.append(f"Units ({len(units)}):")
        parts.extend(unit_lines)
        if idle_ids:
            parts.append(f"Idle: {', '.join(str(i) for i in idle_ids)}")
    else:
        parts.append(f"Units: {state.get('own_units', '?')}")

    # Buildings, enemies, production: identical to v1 to constrain the diff.
    _BLDG_CATEGORY = {
        "tent": "infantry", "barr": "infantry", "weap": "vehicle",
        "hpad": "aircraft", "afld": "aircraft", "syrd": "ship", "spen": "ship",
        "gun": "defense", "ftur": "defense", "tsla": "defense",
        "sam": "defense", "agun": "defense", "pbox": "defense", "hbox": "defense",
    }
    if buildings:
        bldg_parts = []
        for b in buildings:
            cat = _BLDG_CATEGORY.get(b["type"], "")
            cat_str = f"[{cat}]" if cat else ""
            bldg_parts.append(
                f"{b['type']}({b['id']})@({b['cell_x']},{b['cell_y']}){cat_str}"
            )
        parts.append(f"Buildings: {' '.join(bldg_parts)}")
    else:
        parts.append(
            f"Buildings: {state.get('own_buildings', '?')} "
            f"({', '.join(state.get('building_types', []))})"
        )

    enemies = state.get("enemy_summary", [])
    enemy_bldgs = state.get("enemy_buildings_summary", [])
    if enemies or enemy_bldgs:
        enemy_parts = []
        if enemies:
            eby_type = defaultdict(list)
            for e in enemies:
                eby_type[e["type"]].append(e)
            for etype, es in eby_type.items():
                entries = ",".join(f"{e['id']}@({e['cell_x']},{e['cell_y']})" for e in es)
                enemy_parts.append(f"{len(es)}x{etype}[{entries}]")
        if enemy_bldgs:
            ebby_type = defaultdict(list)
            for b in enemy_bldgs:
                ebby_type[b["type"]].append(b)
            for btype, bs in ebby_type.items():
                entries = ",".join(f"{b['id']}@({b['cell_x']},{b['cell_y']})" for b in bs)
                enemy_parts.append(f"{len(bs)}x{btype}[{entries}]")
        all_enemy_pos = (
            [(e["cell_x"], e["cell_y"]) for e in enemies]
            + [(b["cell_x"], b["cell_y"]) for b in enemy_bldgs]
        )
        avg_x = sum(p[0] for p in all_enemy_pos) // len(all_enemy_pos)
        avg_y = sum(p[1] for p in all_enemy_pos) // len(all_enemy_pos)
        parts.append(f"Enemies: {' '.join(enemy_parts)} center ({avg_x},{avg_y})")
    else:
        n_enemy = state.get("visible_enemy_units", 0)
        parts.append(
            f"Enemies: {'none visible' if n_enemy == 0 else f'{n_enemy} visible'}"
        )

    prod = state.get("production_items", [])
    if prod:
        active = [p for p in prod if "@100%" not in p]
        ready = [p.split("@")[0] for p in prod if "@100%" in p]
        parts_prod = []
        if active:
            parts_prod.append(", ".join(active))
        if ready:
            parts_prod.append(f"READY TO PLACE: {', '.join(ready)}")
        parts.append(f"Production: {' | '.join(parts_prod)}")
    else:
        parts.append("Production: IDLE")

    available = state.get("available_production", [])
    if available:
        parts.append(f"Can build: {', '.join(available)}")

    return "\n".join(parts)
