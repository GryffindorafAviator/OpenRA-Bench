"""Deterministic, scenario-scoped game knowledge for the agent.

Frontier models have patchy Red Alert knowledge; a model losing
because it doesn't know `proc` is the refinery is a *knowledge*
confound, not a reasoning signal. So we inject — identically for every
model, no tool, no turn cost — exactly the knowledge needed for the
scenario at hand:

* a glossary of only the actor codes actually present,
* the fixed game model (stances / coordinates / fog),
* a plain-language translation of THIS scenario's machine win/fail
  condition (the model is judged on reasoning toward a *known* goal,
  not on divining the success criterion).

Pure functions, fully unit-testable.
"""

from __future__ import annotations

from typing import Any

# Short, accurate glossary. Keep terse — this is reference, not prose.
ACTOR_GLOSSARY: dict[str, str] = {
    # infantry
    "e1": "rifle infantry (cheap, anti-infantry)",
    "e2": "grenadier infantry",
    "e3": "rocket soldier (anti-vehicle/anti-air)",
    "e6": "engineer (captures/repairs buildings; unarmed)",
    "dog": "attack dog (fast, anti-infantry only)",
    "medi": "field medic (heals nearby infantry; unarmed)",
    "spy": "spy (infiltrates enemy buildings)",
    "thf": "thief (steals enemy credits)",
    # vehicles
    "jeep": "ranger/jeep (fast, light, ideal scout)",
    "1tnk": "light tank (medium armour, anti-vehicle)",
    "2tnk": "heavy/medium tank (main battle tank)",
    "3tnk": "heavy tank (Soviet, strong armour)",
    "4tnk": "mammoth tank (very heavy, dual cannon+missile)",
    "apc": "armoured personnel carrier (transports infantry)",
    "arty": "artillery (long-range, fragile)",
    "harv": "ore harvester (gathers ore for credits; unarmed)",
    "mcv": "mobile construction vehicle (deploys into a fact)",
    "lst": "landing craft / transport",
    # buildings
    "fact": "construction yard (builds structures; LOSS-CRITICAL base)",
    "powr": "power plant (supplies power to structures)",
    "apwr": "advanced power plant",
    "proc": "ore refinery (turns ore into credits; economy core)",
    "barr": "Soviet barracks (trains infantry)",
    "tent": "Allied barracks (trains infantry)",
    "weap": "war factory (builds vehicles)",
    "fix": "service depot (repairs vehicles)",
    "dome": "radar dome (reveals map / tech prerequisite)",
    "silo": "ore silo (stores surplus ore)",
    "gun": "gun turret (anti-vehicle base defence)",
    "pbox": "pillbox (anti-infantry base defence)",
    "tsla": "Tesla coil (powerful anti-everything defence)",
    "sam": "SAM site (anti-air defence)",
}

GAME_MODEL = (
    "GAME MODEL:\n"
    "- Map cells are (x,y); x grows east, y grows south. Tools take "
    "integer cell coords.\n"
    "- Fog of war: you only see near your own units; 'explored' stays "
    "dim once revealed but enemies there may have moved/changed. Scout "
    "to gain information.\n"
    "- Stances: 0=HoldFire (never fire), 1=ReturnFire (only if "
    "attacked), 2=Defend (engage nearby threats, hold position), "
    "3=AttackAnything (auto-engage on sight).\n"
    "- move_units auto-fires opportunistically en route; attack_move "
    "advances while engaging; attack_unit focus-fires one target; "
    "stop cancels current orders; observe passes the turn.\n"
)

TECH_NOTE = (
    "TECH TREE: production needs prerequisites — a barracks "
    "(tent=Allied / barr=Soviet) before infantry; a war factory (weap) "
    "before vehicles; power (powr) sufficient for what's built; "
    "advanced defences (tsla) need power and usually a tech building. "
    "Building unnecessary structures wastes limited credits.\n"
)


def actor_codes(scenario: Any) -> set[str]:
    """Lowercase actor-type codes present in a compiled scenario."""
    out: set[str] = set()
    for a in getattr(scenario, "actors", None) or []:
        t = getattr(a, "type", None) if not isinstance(a, dict) else a.get("type")
        if t:
            out.add(str(t).lower())
    return out


def _condition_codes(node: Any) -> set[str]:
    """Actor codes named inside a win/fail predicate tree (production
    targets like e3/tsla that are NOT pre-placed actors but the model
    is asked to build/destroy — they must still be glossary-explained)."""
    out: set[str] = set()
    if node is None:
        return out
    if not isinstance(node, dict):
        node = dict(getattr(node, "__pydantic_extra__", {}) or {})
    for k, v in node.items():
        if k in ("all_of", "any_of"):
            for c in v:
                out |= _condition_codes(c)
        elif k == "not":
            out |= _condition_codes(v)
        elif isinstance(v, dict):
            if v.get("type"):
                out.add(str(v["type"]).lower())
            for t in v.get("types", []) or []:
                out.add(str(t).lower())
        elif k == "has_building" and isinstance(v, str):
            out.add(v.lower())
    return out


def scenario_primer(compiled: Any) -> str:
    """The knowledge block for THIS scenario: glossary of present
    codes + the fixed game model (+ tech note only if the scenario
    actually allows production)."""
    sc = compiled.scenario
    codes = set(actor_codes(sc))
    codes |= _condition_codes(getattr(compiled, "win_condition", None))
    codes |= _condition_codes(getattr(compiled, "fail_condition", None))
    codes = sorted(codes)
    lines = ["GAME KNOWLEDGE (Command & Conquer: Red Alert)"]
    if codes:
        lines.append("Units/buildings in this scenario:")
        for c in codes:
            lines.append(f"  {c} = {ACTOR_GLOSSARY.get(c, 'unknown actor')}")
    lines.append("")
    lines.append(GAME_MODEL)
    tools = set(getattr(sc, "tools", None) or [])
    if {"build", "place_building"} & tools:
        lines.append(TECH_NOTE)
    return "\n".join(lines).strip()


# ── win/fail predicate → plain language ────────────────────────────────────

def _region(x: Any) -> str:
    if isinstance(x, dict):
        return f"({x.get('x')},{x.get('y')}) r={x.get('radius', 3)}"
    return str(x)


# key -> lazy formatter (only the matched one runs, so a scalar v never
# hits a .get meant for a dict-valued predicate).
_PHRASES: dict[str, Any] = {
    "within_ticks": lambda v: f"before game tick {v}",
    "after_ticks": lambda v: f"not before game tick {v}",
    "units_killed_gte": lambda v: f"destroy ≥{v} enemy units",
    "units_lost_lte": lambda v: f"lose ≤{v} of your own units",
    "explored_pct_gte": lambda v: f"reveal ≥{v}% of the map",
    "enemies_discovered_gte": lambda v: f"spot ≥{v} enemy units",
    "buildings_discovered_gte": lambda v: f"spot ≥{v} enemy buildings",
    "reach_region": lambda v: f"get a unit into region {_region(v)}",
    "all_units_in_region": lambda v: f"get EVERY unit into region {_region(v)}",
    "own_units_gte": lambda v: f"keep ≥{v} units alive",
    "cash_gte": lambda v: f"hold ≥{v} credits",
    "resources_gte": lambda v: f"hold ≥{v} stored ore",
    "economy_value_gte": lambda v: f"reach economy value ≥{v} (cash+ore)",
    "power_surplus_gte": lambda v: f"keep power surplus ≥{v}",
    "has_building": lambda v: f"own a '{v}'",
    "buildings_owned_gte": lambda v: f"own ≥{v} distinct building types",
    "building_total_gte": lambda v: f"own ≥{v} buildings total",
    "building_count_gte": lambda v: f"own ≥{(v or {}).get('n', 1)} "
    f"'{(v or {}).get('type')}' building(s)",
    "building_in_region": lambda v: f"have {(v or {}).get('count', 1)} "
    f"building(s) near ({(v or {}).get('x')},{(v or {}).get('y')})",
    "unit_type_count_eq": lambda v: f"have EXACTLY {(v or {}).get('n')} "
    f"'{(v or {}).get('type')}' (no more, no fewer)",
    "unit_type_count_gte": lambda v: f"have ≥{(v or {}).get('n')} "
    f"'{(v or {}).get('type')}'",
    "enemy_buildings_destroyed_gte": lambda v: f"destroy ≥{v} enemy buildings",
    "enemy_key_buildings_destroyed": lambda v: "destroy the enemy "
    + "+".join(v.get("types", []) if isinstance(v, dict) else v),
}


def _leaf_phrase(key: str, v: Any) -> str:
    fn = _PHRASES.get(key)
    return fn(v) if fn else f"{key}={v}"


def _describe(node: Any, join: str = " AND ") -> str:
    if node is None:
        return ""
    if not isinstance(node, dict):
        node = dict(getattr(node, "__pydantic_extra__", {}) or {})
    if "all_of" in node:
        return join.join(_describe(c) for c in node["all_of"])
    if "any_of" in node:
        return "(" + " OR ".join(_describe(c) for c in node["any_of"]) + ")"
    if "not" in node:
        return "NOT (" + _describe(node["not"]) + ")"
    return join.join(_leaf_phrase(k, v) for k, v in node.items())


def objective_brief(description: str, win_condition: Any,
                     fail_condition: Any, max_turns: int) -> str:
    """Plain-language objective the model sees every turn: the scenario
    prose PLUS the exact machine win/fail criteria (so success is a
    known target, not a guess)."""
    parts = []
    if description:
        parts.append(description.strip())
    win = _describe(win_condition)
    parts.append(f"WIN WHEN: {win}." if win else "WIN: (none defined)")
    fail = _describe(fail_condition, join=" AND ")
    if fail:
        parts.append(f"YOU LOSE IF: {fail}.")
    parts.append(
        f"You have at most {max_turns} decision turns; acting "
        "decisively and early matters."
    )
    return "\n".join(parts)
