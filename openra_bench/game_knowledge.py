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
    "tanya": "Tanya (Allied hero commando; pistol vs infantry, "
    "C4 vs buildings)",
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
    # naval
    "dd": "destroyer (Allied naval; ranged shore bombardment)",
    # aircraft
    "heli": "Hind attack helicopter (flies over terrain/buildings; "
    "anti-ground)",
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
    "mine": "ore mine (resource node harvesters gather from)",
    "gun": "gun turret (anti-vehicle base defence)",
    "pbox": "pillbox (anti-infantry base defence)",
    "brik": "concrete wall (inert obstacle; no weapon — channels but does not kill)",
    "tsla": "Tesla coil (powerful anti-everything defence)",
    "sam": "SAM site (anti-air defence)",
    "agun": "AA gun (Allied anti-air ground turret)",
    "mslo": "missile silo (Soviet superweapon; launches a single "
    "high-damage nuke at a target cell after a charge cycle)",
    "atek": "advanced tech centre (gates top-tier units / superweapons)",
    "hpad": "helipad (builds and rearms helicopters)",
    "afld": "airfield (builds and rearms fixed-wing aircraft; "
    "Soviet equivalent of hpad's role for planes)",
    "syrd": "Soviet shipyard (builds naval units)",
    "spen": "Allied sub pen (builds naval units)",
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

_REGION_KEYS = (
    "reach_region",
    "units_in_region_gte",
    "units_of_type_in_region_gte",
    "all_units_in_region",
)


def _region(x: Any, coords: str = "exact") -> str:
    if isinstance(x, dict):
        if coords == "relative":
            # Disclose only the authored compass label — the model
            # must localize the target on the minimap itself. Fall
            # back to coords if a region lacks a label (authoring bug
            # made visible rather than silently leaking numbers).
            lbl = x.get("label")
            if lbl:
                return str(lbl)
        return f"region ({x.get('x')},{x.get('y')}) r={x.get('radius', 3)}"
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
    # Region phrases are handled by _REGION_PHRASES (coords-aware);
    # these fallbacks keep the exact default for any direct use.
    "reach_region": lambda v: f"get a unit into {_region(v)}",
    "units_in_region_gte": lambda v: (
        f"get ≥{(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"units into {_region(v)}"
    ),
    "units_of_type_in_region_gte": lambda v: (
        f"get ≥{(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"'{(v if isinstance(v, dict) else {}).get('type')}' "
        f"units into {_region(v)}"
    ),
    "all_units_in_region": lambda v: f"get EVERY unit into {_region(v)}",
    "own_units_gte": lambda v: f"keep ≥{v} units alive",
    "cash_gte": lambda v: f"hold ≥{v} credits",
    "resources_gte": lambda v: f"hold ≥{v} stored ore",
    "economy_value_gte": lambda v: f"reach economy value ≥{v} (cash+ore)",
    "power_surplus_gte": lambda v: f"keep power surplus ≥{v}",
    "power_provided_gte": lambda v: f"keep gross power provided ≥{v}",
    "has_building": lambda v: f"own a '{v}'",
    "buildings_owned_gte": lambda v: f"own ≥{v} distinct building types",
    "building_total_gte": lambda v: f"own ≥{v} buildings total",
    "building_count_gte": lambda v: f"own ≥{(v or {}).get('n', 1)} "
    f"'{(v or {}).get('type')}' building(s)",
    "building_hp_pct_gte": lambda v: (
        f"keep your '{(v or {}).get('type')}' building at "
        f"≥{int(float((v or {}).get('pct', 1.0)) * 100)}% HP"
    ),
    "building_in_region": lambda v: f"have {(v or {}).get('count', 1)} "
    f"building(s) near ({(v or {}).get('x')},{(v or {}).get('y')})",
    "unit_type_count_eq": lambda v: f"have EXACTLY {(v or {}).get('n')} "
    f"'{(v or {}).get('type')}' (no more, no fewer)",
    "unit_type_count_gte": lambda v: f"have ≥{(v or {}).get('n')} "
    f"'{(v or {}).get('type')}'",
    "enemy_buildings_destroyed_gte": lambda v: f"destroy ≥{v} enemy buildings",
    "enemy_key_buildings_destroyed": lambda v: "destroy the enemy "
    + "+".join(v.get("types", []) if isinstance(v, dict) else v),
    "enemy_key_buildings_destroyed_in_region": lambda v: (
        "destroy the enemy "
        + "+".join((v or {}).get("types", []))
        + f" at the base near ({(v or {}).get('x')},{(v or {}).get('y')})"
    ),
    "tool_violations_gte": lambda v: (
        "you used a forbidden tool (instant fail)"
        if int(v) <= 1
        else f"you used a forbidden tool ≥{v} times"
    ),
}


# Coords-aware region phrasing (honours objective_coords).
_REGION_PHRASES: dict[str, Any] = {
    "reach_region": lambda v, c: f"get a unit into {_region(v, c)}",
    "units_in_region_gte": lambda v, c: (
        f"get ≥{(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"units into {_region(v, c)}"
    ),
    "units_of_type_in_region_gte": lambda v, c: (
        f"get ≥{(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"'{(v if isinstance(v, dict) else {}).get('type')}' "
        f"units into {_region(v, c)}"
    ),
    "all_units_in_region": lambda v, c: f"get EVERY unit into {_region(v, c)}",
    "waypoint_sequence": lambda v, c: (
        "reach these waypoints IN ORDER (no skipping, no idling): "
        + " → then → ".join(
            _region(p, c)
            for p in ((v.get("points") or []) if isinstance(v, dict) else [])
        )
    ),
}


def _leaf_phrase(key: str, v: Any, coords: str = "exact") -> str:
    if key in _REGION_PHRASES:
        return _REGION_PHRASES[key](v, coords)
    fn = _PHRASES.get(key)
    return fn(v) if fn else f"{key}={v}"


def _describe(node: Any, join: str = " AND ", coords: str = "exact") -> str:
    if node is None:
        return ""
    if not isinstance(node, dict):
        node = dict(getattr(node, "__pydantic_extra__", {}) or {})
    if "all_of" in node:
        return join.join(_describe(c, coords=coords) for c in node["all_of"])
    if "any_of" in node:
        return "(" + " OR ".join(
            _describe(c, coords=coords) for c in node["any_of"]
        ) + ")"
    if "not" in node:
        inner = node["not"]
        inner_d = inner if isinstance(inner, dict) else dict(
            getattr(inner, "__pydantic_extra__", {}) or {}
        )
        # Common fail form {not: {own_units_gte: N}} reads far clearer
        # as a plain loss statement than a double negative.
        if set(inner_d) == {"own_units_gte"}:
            n = inner_d["own_units_gte"]
            return (
                "your whole force is destroyed" if int(n) <= 1
                else f"fewer than {n} of your units remain"
            )
        return "NOT (" + _describe(inner, coords=coords) + ")"
    if "then" in node:
        # The happened-before composite. Read each clause as a stage
        # in an enforced ordered chain.
        v = node["then"] or {}
        clauses = v.get("clauses") or []
        if not clauses:
            return "(empty then:)"
        return (
            "in this exact order: "
            + " → THEN → ".join(
                _describe(c, coords=coords) for c in clauses
            )
        )
    return join.join(_leaf_phrase(k, v, coords) for k, v in node.items())


def objective_brief(description: str, win_condition: Any,
                     fail_condition: Any, max_turns: int,
                     objective_coords: str = "exact") -> str:
    """Plain-language objective the model sees every turn: the scenario
    prose PLUS the exact machine win/fail criteria (so success is a
    known target, not a guess)."""
    parts = []
    if description:
        parts.append(description.strip())
    win = _describe(win_condition, coords=objective_coords)
    parts.append(f"WIN WHEN: {win}." if win else "WIN: (none defined)")
    fail = _describe(fail_condition, join=" AND ", coords=objective_coords)
    if fail:
        parts.append(f"YOU LOSE IF: {fail}.")
    parts.append(
        f"You have at most {max_turns} decision turns; acting "
        "decisively and early matters."
    )
    return "\n".join(parts)


# ── Chinese (ZH) objective generation ─────────────────────────────────────
# Generates Chinese WIN/LOSE conditions directly from the structured
# predicates, avoiding the broken find-and-replace approach.

_DIRECTION_ZH = {
    "NORTH": "北方", "SOUTH": "南方", "EAST": "东方", "WEST": "西方",
    "NORTH-EAST": "东北方", "NORTH-WEST": "西北方",
    "SOUTH-EAST": "东南方", "SOUTH-WEST": "西南方",
    "NE": "东北", "NW": "西北", "SE": "东南", "SW": "西南",
    "north": "北方", "south": "南方", "east": "东方", "west": "西方",
    "north-east": "东北方", "north-west": "西北方",
    "south-east": "东南方", "south-west": "西南方",
    "northern": "北部", "southern": "南部", "eastern": "东部", "western": "西部",
    "far": "最远", "mid": "中部", "centre": "中央", "center": "中央",
    "corner": "角落",
}

_ACTOR_ZH = {
    "e1": "步枪兵", "e2": "掷弹兵", "e3": "火箭兵",
    "e6": "工程师", "dog": "军犬", "medi": "医疗兵",
    "spy": "间谍", "thf": "小偷", "tanya": "谭雅",
    "jeep": "吉普车", "1tnk": "轻型坦克", "2tnk": "中型坦克",
    "3tnk": "重型坦克", "4tnk": "猛犸坦克",
    "apc": "装甲运兵车", "arty": "火炮", "harv": "采矿车",
    "mcv": "基地车", "lst": "登陆艇",
    "dd": "驱逐舰", "heli": "雌鹿武装直升机",
    "fact": "建造厂", "powr": "发电厂", "apwr": "高级发电厂",
    "proc": "矿石精炼厂", "barr": "苏军兵营", "tent": "盟军兵营",
    "weap": "战车工厂", "fix": "维修站", "dome": "雷达",
    "silo": "矿石仓库", "mine": "矿场",
    "gun": "防御炮塔", "pbox": "碉堡", "tsla": "磁暴线圈",
    "sam": "防空导弹", "mslo": "核弹发射井",
}


def _actor_zh(code: str) -> str:
    return _ACTOR_ZH.get(code, code)


def _region_zh(x: Any, coords: str = "exact") -> str:
    if isinstance(x, dict):
        if coords == "relative":
            lbl = x.get("label")
            if lbl:
                return _translate_label_zh(str(lbl))
        return f"区域 ({x.get('x')},{x.get('y')}) 半径={x.get('radius', 3)}"
    return str(x)


def _translate_label_zh(lbl: str) -> str:
    """Translate a compass-direction label to Chinese."""
    import re
    result = lbl
    for en, zh in sorted(_DIRECTION_ZH.items(), key=lambda kv: -len(kv[0])):
        result = re.sub(re.escape(en), zh, result, flags=re.IGNORECASE)
    return result


_PHRASES_ZH: dict[str, Any] = {
    "within_ticks": lambda v: f"在游戏第 {v} 刻之前完成",
    "after_ticks": lambda v: f"不早于游戏第 {v} 刻",
    "units_killed_gte": lambda v: f"消灭 ≥{v} 个敌方单位",
    "units_lost_lte": lambda v: f"己方损失不超过 {v} 个单位",
    "explored_pct_gte": lambda v: f"探索至少 {v}% 的地图",
    "enemies_discovered_gte": lambda v: f"发现至少 {v} 个敌方单位",
    "buildings_discovered_gte": lambda v: f"发现至少 {v} 个敌方建筑",
    "reach_region": lambda v: f"派一个单位到达{_region_zh(v)}",
    "units_in_region_gte": lambda v: (
        f"将至少 {(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"个单位移动到{_region_zh(v)}"
    ),
    "units_of_type_in_region_gte": lambda v: (
        f"将至少 {(v if isinstance(v, dict) else {}).get('n', 1)} 个"
        f"「{_actor_zh((v if isinstance(v, dict) else {}).get('type', '?'))}」"
        f"移动到{_region_zh(v)}"
    ),
    "all_units_in_region": lambda v: f"将所有单位移动到{_region_zh(v)}",
    "own_units_gte": lambda v: f"保持至少 {v} 个单位存活",
    "cash_gte": lambda v: f"持有至少 {v} 资金",
    "resources_gte": lambda v: f"持有至少 {v} 矿石储备",
    "economy_value_gte": lambda v: f"经济总值达到 ≥{v}（资金+矿石）",
    "power_surplus_gte": lambda v: f"电力盈余保持 ≥{v}",
    "power_provided_gte": lambda v: f"总发电量保持 ≥{v}",
    "has_building": lambda v: f"拥有一座「{_actor_zh(v)}」",
    "buildings_owned_gte": lambda v: f"拥有至少 {v} 种不同建筑",
    "building_total_gte": lambda v: f"拥有至少 {v} 座建筑",
    "building_count_gte": lambda v: (
        f"拥有至少 {(v or {}).get('n', 1)} 座"
        f"「{_actor_zh((v or {}).get('type', '?'))}」"
    ),
    "building_hp_pct_gte": lambda v: (
        f"将「{_actor_zh((v or {}).get('type', '?'))}」"
        f"修复到 ≥{int(float((v or {}).get('pct', 1.0)) * 100)}% 血量"
    ),
    "building_in_region": lambda v: (
        f"在 ({(v or {}).get('x')},{(v or {}).get('y')}) 附近"
        f"放置 {(v or {}).get('count', 1)} 座建筑"
    ),
    "unit_type_count_eq": lambda v: (
        f"恰好拥有 {(v or {}).get('n')} 个"
        f"「{_actor_zh((v or {}).get('type', '?'))}」（不多不少）"
    ),
    "unit_type_count_gte": lambda v: (
        f"拥有至少 {(v or {}).get('n')} 个"
        f"「{_actor_zh((v or {}).get('type', '?'))}」"
    ),
    "enemy_buildings_destroyed_gte": lambda v: f"摧毁至少 {v} 座敌方建筑",
    "enemy_key_buildings_destroyed": lambda v: (
        "摧毁敌方的"
        + "+".join(
            _actor_zh(t)
            for t in (v.get("types", []) if isinstance(v, dict) else v)
        )
    ),
    "enemy_key_buildings_destroyed_in_region": lambda v: (
        "摧毁"
        f" ({(v or {}).get('x')},{(v or {}).get('y')}) 附近的敌方"
        + "+".join(_actor_zh(t) for t in (v or {}).get("types", []))
    ),
    "tool_violations_gte": lambda v: (
        "使用了禁止的工具（立即失败）"
        if int(v) <= 1
        else f"使用禁止的工具 ≥{v} 次"
    ),
}

_REGION_PHRASES_ZH: dict[str, Any] = {
    "reach_region": lambda v, c: f"派一个单位到达{_region_zh(v, c)}",
    "units_in_region_gte": lambda v, c: (
        f"将至少 {(v if isinstance(v, dict) else {}).get('n', 1)} "
        f"个单位移动到{_region_zh(v, c)}"
    ),
    "units_of_type_in_region_gte": lambda v, c: (
        f"将至少 {(v if isinstance(v, dict) else {}).get('n', 1)} 个"
        f"「{_actor_zh((v if isinstance(v, dict) else {}).get('type', '?'))}」"
        f"移动到{_region_zh(v, c)}"
    ),
    "all_units_in_region": lambda v, c: f"将所有单位移动到{_region_zh(v, c)}",
    "waypoint_sequence": lambda v, c: (
        "按顺序依次到达以下路径点（不可跳过，不可停滞）："
        + " → 然后 → ".join(
            _region_zh(p, c)
            for p in ((v.get("points") or []) if isinstance(v, dict) else [])
        )
    ),
}


def _leaf_phrase_zh(key: str, v: Any, coords: str = "exact") -> str:
    if key in _REGION_PHRASES_ZH:
        return _REGION_PHRASES_ZH[key](v, coords)
    fn = _PHRASES_ZH.get(key)
    return fn(v) if fn else f"{key}={v}"


def _describe_zh(node: Any, join: str = "，并且", coords: str = "exact") -> str:
    if node is None:
        return ""
    if not isinstance(node, dict):
        node = dict(getattr(node, "__pydantic_extra__", {}) or {})
    if "all_of" in node:
        return join.join(_describe_zh(c, coords=coords) for c in node["all_of"])
    if "any_of" in node:
        return "（" + " 或 ".join(
            _describe_zh(c, coords=coords) for c in node["any_of"]
        ) + "）"
    if "not" in node:
        inner = node["not"]
        inner_d = inner if isinstance(inner, dict) else dict(
            getattr(inner, "__pydantic_extra__", {}) or {}
        )
        if set(inner_d) == {"own_units_gte"}:
            n = inner_d["own_units_gte"]
            return "全军覆没" if int(n) <= 1 else f"存活单位不足 {n} 个"
        return "未能（" + _describe_zh(inner, coords=coords) + "）"
    if "then" in node:
        v = node["then"] or {}
        clauses = v.get("clauses") or []
        if not clauses:
            return "（空的顺序条件）"
        return (
            "按以下顺序完成："
            + " → 然后 → ".join(
                _describe_zh(c, coords=coords) for c in clauses
            )
        )
    return join.join(_leaf_phrase_zh(k, v, coords) for k, v in node.items())


def objective_brief_zh(
    description_zh: str,
    win_condition: Any,
    fail_condition: Any,
    max_turns: int,
    objective_coords: str = "exact",
) -> str:
    """Chinese objective text, structured identically to objective_brief."""
    parts = []
    if description_zh:
        parts.append(description_zh.strip())
    win = _describe_zh(win_condition, coords=objective_coords)
    parts.append(f"胜利条件：{win}。" if win else "胜利条件：（未定义）")
    fail = _describe_zh(fail_condition, join="，并且", coords=objective_coords)
    if fail:
        parts.append(f"失败条件：{fail}。")
    parts.append(f"你最多有 {max_turns} 个决策回合，果断且尽早行动至关重要。")
    return "\n".join(parts)
