"""Prompt/briefing/minimap = the training v2 format, by construction.

Loads the byte-vendored training artifacts (`_vendor/system_v2.txt`,
`briefing_v2.py`, `minimap_v2.py`) and adapts the bench `render_state`
into the exact `state` dict they expect, so a bench transcript is
indistinguishable in format from a training rollout.

The system prompt also carries the FULL RA codex + tech tree (built at
import time from the vendored OpenRA YAML rules / weapons), so every
LLM call sees the same uniform encyclopaedia — equivalent to a human
reading the RA manual before play. Previously only the actors mentioned
in a given pack were surfaced, which left a 1v1 macro pack with a
`syrd` blind to what ships it could produce. (`unit_codex(codes)` is
retained as a legacy filtered fallback.)
"""

from __future__ import annotations

import base64
import importlib.util as _iu
import logging
import re as _codex_re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VENDOR = Path(__file__).parent / "_vendor"


def _load(mod: str):
    spec = _iu.spec_from_file_location(mod, _VENDOR / f"{mod}.py")
    m = _iu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_BRIEF = _load("briefing_v2")          # format_state_briefing_v2(state)
_MM = _load("minimap_v2")              # render(obs, terrain, w, h, bounds, hist)
_SYSTEM_TMPL = (_VENDOR / "system_v2.txt").read_text(encoding="utf-8")


# Curated RA codex: cost / hp / range(cells) / dps / sight(cells) /
# speed. Values from the Red Alert ruleset the engine ships. Kept
# terse — one line per code, matching system_v2's "rng=Xc / DPS / sight
# Nc" vocabulary.
_CODEX: dict[str, str] = {
    "e1":   "Rifle infantry — cheap basic foot soldier, good vs infantry. $100 hp50 rng3c dps6 sight4c foot",
    "e2":   "Grenadier — lobs grenades, strong vs groups/buildings. $150 hp50 rng4c dps9 sight4c foot",
    "e3":   "Rocket soldier — anti-tank/anti-air infantry, weak vs infantry. $300 hp45 rng4c dps12 sight5c foot",
    "e6":   "Engineer — captures or repairs buildings; unarmed, fragile. $400 hp25 unarmed sight4c foot",
    "dog":  "Attack dog — very fast, kills infantry instantly, useless vs vehicles. $200 hp12 rng1c dps10 sight5c foot",
    "medi": "Medic — heals nearby friendly infantry; unarmed. $200 hp80 unarmed sight4c foot",
    "spy":  "Spy — infiltrates enemy buildings (intel/sabotage); unarmed, disguised. $500 hp25 unarmed sight5c foot",
    "thf":  "Thief — steals credits from enemy refineries; unarmed. $500 hp25 unarmed sight4c foot",
    "jeep": "Ranger jeep — fast lightly-armed scout, best for vision. $500 hp150 rng4c dps8 sight7c wheeled",
    "1tnk": "Light tank — cheap, fast, modest armour/gun; anti-vehicle. $700 hp300 rng4c dps14 sight6c tracked",
    "2tnk": "Medium tank — the main battle tank, balanced armour/firepower. $850 hp400 rng4.75c dps22 sight6c tracked",
    "3tnk": "Heavy tank (Soviet) — heavily armoured, hits hard, slow. $1150 hp600 rng5c dps30 sight6c tracked",
    "4tnk": "Mammoth tank — biggest tank, dual cannon+missile (also anti-air), very slow. $2000 hp900 rng5c dps40+ sight6c tracked",
    "arty": "Artillery — long-range siege gun, devastating but fragile and slow. $850 hp75 rng7c dps45 sight5c wheeled",
    "apc":  "Armoured personnel carrier — carries 5 infantry, light gun, tough. $850 hp200 rng4c dps8 sight6c tracked",
    "harv": "Ore harvester — gathers ore for credits; unarmed, heavily armoured. $1100 hp600 unarmed sight4c tracked",
    "mcv":  "Mobile construction vehicle — deploys into a construction yard. $2000 hp600 unarmed sight5c tracked",
    "fact": "Construction yard — builds all structures; LOSS-CRITICAL base building.",
    "powr": "Power plant — supplies power; structures fail without enough power.",
    "apwr": "Advanced power plant — supplies more power than a basic plant.",
    "proc": "Ore refinery — converts harvested ore into credits; the economy core.",
    "barr": "Soviet barracks — trains infantry; prerequisite for foot units.",
    "tent": "Allied barracks — trains infantry; prerequisite for foot units.",
    "weap": "War factory — builds vehicles; prerequisite for tanks/armour.",
    "dome": "Radar dome — reveals the map / tech prerequisite for advanced units.",
    "silo": "Ore silo — stores surplus ore so income isn't capped.",
    "gun":  "Gun turret — fixed anti-armour base defence. rng6c",
    "pbox": "Pillbox — fixed anti-infantry base defence. rng5c",
    "tsla": "Tesla coil — powerful fixed defence vs everything; needs power. rng7c",
    "sam":  "SAM site — fixed anti-air defence (cannot hit ground).",
}


def unit_codex(codes) -> str:
    """Legacy scenario-scoped codex — kept for back-compat (callers
    that still want a filtered view). The system prompt now defaults
    to FULL_CODEX_TEXT + TECH_TREE_TEXT (every model sees the same
    encyclopaedia)."""
    rows = [c for c in sorted(set(codes)) if c in _CODEX]
    if not rows:
        return ""
    body = "\n".join(f"  {c:<5} {_CODEX[c]}" for c in rows)
    return (
        "UNIT CODEX (this scenario). Each line: <code> $cost hp<HP> "
        "rng<attack range, cells> dps<damage/sec> sight<vision, cells> "
        "<movement: foot/wheeled/tracked> (role). Unarmed units list "
        "their function instead of rng/dps; buildings list their "
        "purpose.\n" + body
    )


# ─── Full RA codex built from vendor YAML ─────────────────────────────
# Parses OpenRA-Rust/openra-data/src/embedded/rules/*.yaml and
# weapons/*.yaml at import time. The result is two strings —
# FULL_CODEX_TEXT (per-actor stats: cost / hp / range / DPS / sight /
# movement / role) and TECH_TREE_TEXT (faction + prerequisites,
# grouped by production source) — appended to the system prompt by
# default so every LLM call sees the same uniform reference. Pure
# function: read-only access to the vendored YAML files; if the
# vendor checkout isn't reachable the builders return "" and the
# system prompt falls back to the curated `_CODEX` legacy view.


def _vendor_rules_dir() -> Path | None:
    """Locate the in-repo vendored OpenRA rules directory.
    Reuses the same path the engine compiles into the wheel."""
    repo = Path(__file__).resolve().parents[2]  # …/OpenRA-Bench → parent
    cand = repo / "OpenRA-Rust/openra-data/src/embedded"
    if (cand / "rules").is_dir() and (cand / "weapons").is_dir():
        return cand
    return None


def _parse_ra_yaml(text: str) -> dict:
    """Tab-indented OpenRA YAML. A key with both inline value AND
    children (e.g. `Warhead@1Dam: SpreadDamage` + sub-keys) stores
    the value as `__type__` inside the child dict."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        i = 0
        while i < len(raw) and raw[i] == "\t":
            i += 1
        lines.append((i, raw[i:].rstrip()))
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    n = len(lines)
    for idx, (indent, line) in enumerate(lines):
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        has_children = idx + 1 < n and lines[idx + 1][0] > indent
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if has_children:
                child: dict[str, Any] = {}
                if val:
                    child["__type__"] = val
                if key in parent and isinstance(parent[key], dict):
                    child = {**child, **parent[key]}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = val
        else:
            parent[line.strip()] = ""
    return root


def _deep_merge(into: dict, src: dict) -> dict:
    out = dict(into)
    for k, v in src.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_inherits(actors: dict, name: str, seen: set | None = None) -> dict:
    seen = set(seen or ())
    if name in seen or name not in actors:
        return {}
    seen.add(name)
    body = actors[name]
    merged: dict = {}
    for k in list(body.keys()):
        if k == "Inherits" or k.startswith("Inherits@") or k.startswith("Inherits#"):
            parent_name = body[k]
            if isinstance(parent_name, str):
                merged = _deep_merge(
                    merged, _resolve_inherits(actors, parent_name, seen)
                )
    for k, v in body.items():
        if k == "Inherits" or k.startswith("Inherits@") or k.startswith("Inherits#"):
            continue
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _resolve_weapon(weapons: dict, name: str, seen: set | None = None) -> dict:
    seen = set(seen or ())
    if name in seen or name not in weapons:
        return {}
    seen.add(name)
    body = weapons[name]
    merged: dict = {}
    for k in list(body.keys()):
        if k == "Inherits" or k.startswith("Inherits"):
            parent_name = body[k]
            if isinstance(parent_name, str):
                merged = _deep_merge(
                    merged, _resolve_weapon(weapons, parent_name, seen)
                )
    for k, v in body.items():
        if k == "Inherits" or k.startswith("Inherits"):
            continue
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _parse_range_cells(rng: object) -> float:
    if not rng:
        return 0.0
    m = _codex_re.match(r"(\d+)c(\d+)?", str(rng))
    if m:
        return int(m.group(1)) + int(m.group(2) or 0) / 1024.0
    try:
        return float(rng)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def _weapon_damage(weapon: dict) -> int:
    total = 0
    for k, v in weapon.items():
        if k.startswith("Warhead") and isinstance(v, dict):
            t = str(v.get("__type__", "")).strip()
            if t.startswith("SpreadDamage") or "Damage" in v:
                try:
                    total += int(v.get("Damage", 0))
                except Exception:
                    pass
    return total


def _weapon_summary(weapons: dict, wname: str) -> dict | None:
    w = _resolve_weapon(weapons, wname)
    if not w:
        return None
    dmg = _weapon_damage(w)
    try:
        burst = int(w.get("Burst", 1))
    except Exception:
        burst = 1
    try:
        reload = int(w.get("ReloadDelay", 60))
    except Exception:
        reload = 60
    rng = _parse_range_cells(w.get("Range", "0"))
    # Engine HP / damage scale: vendor / 100. 25 ticks per game second.
    dps = (dmg * burst * 25.0 / reload) / 100.0 if reload > 0 else 0.0
    return {
        "name": wname,
        "dmg": dmg,
        "burst": burst,
        "reload": reload,
        "range_cells": rng,
        "dps": round(dps, 1),
    }


_LOCO_MAP = {
    "foot": "foot", "wheeled": "wheeled", "tracked": "tracked",
    "heavy": "tracked", "amphibious": "amphibious", "naval": "naval",
    "ship": "naval", "submarine": "naval", "boat": "naval",
}
_SRC_MOVE = {
    "aircraft.yaml": "aircraft", "ships.yaml": "naval",
    "vehicles.yaml": "tracked", "infantry.yaml": "foot",
    "structures.yaml": "building", "civilian.yaml": "foot",
}

# Engine aliases (gamerules.rs registers `tanya` as E7).
_ALIASES = {"tanya": "E7"}


def _actor_stats(actors: dict, weapons: dict, code_upper: str) -> dict:
    body = _resolve_inherits(actors, code_upper)
    src = actors.get(code_upper, {}).get("__source__", "")
    out: dict[str, Any] = {"code": code_upper.lower(), "_src": src}
    valued = body.get("Valued", {})
    if isinstance(valued, dict) and "Cost" in valued:
        try:
            out["cost"] = int(valued["Cost"])
        except Exception:
            pass
    health = body.get("Health", {})
    if isinstance(health, dict) and "HP" in health:
        try:
            out["hp"] = int(int(health["HP"]) / 100)
        except Exception:
            pass
    buildable = body.get("Buildable", {})
    if isinstance(buildable, dict) and buildable:
        out["prereqs"] = str(buildable.get("Prerequisites", "")).strip()
        out["queue"] = str(buildable.get("Queue", "")).strip()
        out["buildable"] = True
    else:
        out["buildable"] = False
        out["prereqs"] = ""
        out["queue"] = ""
    rev = body.get("RevealsShroud", {})
    if isinstance(rev, dict) and "Range" in rev:
        out["sight_cells"] = _parse_range_cells(rev["Range"])
    arms: list[dict] = []
    seen_weapons: set[str] = set()
    for k, v in body.items():
        if k == "Armament" or k.startswith("Armament@") or k.startswith("Armament#"):
            if isinstance(v, dict) and "Weapon" in v:
                wname = v["Weapon"]
                if "GARRISON" in k.upper() or str(v.get("Name", "")).lower() == "garrisoned":
                    continue
                if wname in seen_weapons:
                    continue
                seen_weapons.add(wname)
                ws = _weapon_summary(weapons, wname)
                if ws:
                    arms.append(ws)
    out["armaments"] = arms
    if "Aircraft" in body or "Plane" in body or "Helicopter" in body or src == "aircraft.yaml":
        out["movement"] = "aircraft"
    else:
        mob = body.get("Mobile", {})
        if isinstance(mob, dict) and mob:
            loco = str(mob.get("Locomotor", "")).lower().strip()
            out["movement"] = _LOCO_MAP.get(loco, _SRC_MOVE.get(src) or loco or "ground")
        elif src == "ships.yaml":
            out["movement"] = "naval"
        elif src == "structures.yaml" or "Building" in body:
            out["movement"] = "building"
        else:
            out["movement"] = "?"
    out["is_building"] = out["movement"] == "building"
    power = body.get("Power", {})
    if isinstance(power, dict) and "Amount" in power:
        try:
            out["power"] = int(power["Amount"])
        except Exception:
            pass
    return out


# Role glossary — short, accurate. Falls back to a generic role when
# the actor isn't in openra_bench.game_knowledge.ACTOR_GLOSSARY.
_EXTRA_ROLES = {
    "v2rl": "V2 rocket launcher (Soviet long-range artillery)",
    "mnly": "minelayer (drops mines)",
    "mgg": "mobile gap generator (creates shroud)",
    "mrj": "mobile radar jammer",
    "ttnk": "tesla tank (Soviet mobile shocker)",
    "ftrk": "flak truck (Soviet anti-air vehicle)",
    "dtrk": "demolition truck (suicide explosive)",
    "ctnk": "chrono tank (Allied teleporter)",
    "qtnk": "MAD tank (Soviet shockwave)",
    "stnk": "phase transport (Allied stealth APC)",
    "hind": "Hind gunship (Soviet attack helicopter)",
    "yak": "Yak fighter (Soviet ground-attack plane)",
    "mig": "MiG attack plane (Soviet)",
    "tran": "Chinook transport helicopter",
    "badr": "Badger bomber (paradrop carrier)",
    "u2": "U-2 spy plane (recon)",
    "mh60": "Black Hawk helicopter",
    "ca": "cruiser (Allied long-range naval artillery)",
    "ss": "submarine (Soviet stealth naval, anti-ship)",
    "msub": "missile submarine (Soviet long-range)",
    "pt": "PT boat (light naval, anti-sub)",
    "lst": "landing craft (transport across water)",
    "afld": "Soviet airfield (builds planes)",
    "spen": "Allied sub pen (builds submarines/PT)",
    "syrd": "Soviet shipyard (builds DD/cruiser/PT/lst)",
    "iron": "iron curtain (Soviet superweapon: temporary invulnerability)",
    "stek": "Soviet tech centre (gates top-tier units)",
    "pdox": "chronosphere (Allied superweapon: teleport unit)",
    "mslo": "missile silo (Soviet nuke superweapon)",
    "kenn": "kennel (prerequisite for attack dogs)",
    "gap": "gap generator (creates shroud over enemy)",
    "hbox": "camo pillbox (hidden anti-infantry)",
    "ftur": "flame turret (Soviet anti-infantry, short range)",
    "agun": "AA gun (Allied anti-air ground turret)",
    "barb": "barbed wire (inert obstacle)",
    "sbag": "sandbag wall (cheap obstacle)",
    "cycl": "chain-link fence (obstacle)",
    "fenc": "wood fence (obstacle)",
    "wood": "wood (decorative wall)",
    "e7": "Tanya (Allied hero; Colt45 vs infantry, C4 vs buildings)",
}


def _role_for(code: str) -> str:
    # Late import to avoid circulars.
    try:
        from .game_knowledge import ACTOR_GLOSSARY
    except Exception:
        ACTOR_GLOSSARY = {}  # type: ignore[assignment]
    c = code.lower()
    return ACTOR_GLOSSARY.get(c) or _EXTRA_ROLES.get(c, "")


_INFANTRY = ["E1", "E2", "E3", "E6", "DOG", "MEDI", "SPY", "THF", "E7"]
_VEHICLES = ["1TNK", "2TNK", "3TNK", "4TNK", "JEEP", "APC", "ARTY", "HARV",
             "MCV", "V2RL", "MNLY", "MGG", "MRJ", "TTNK", "FTRK", "DTRK",
             "CTNK"]
_AIRCRAFT = ["HELI", "HIND", "YAK", "MIG", "TRAN", "BADR", "U2", "MH60"]
_SHIPS = ["DD", "CA", "SS", "MSUB", "PT", "LST"]
_STRUCTURES = ["FACT", "POWR", "APWR", "PROC", "BARR", "TENT", "WEAP",
               "HPAD", "AFLD", "SPEN", "SYRD", "FIX", "DOME", "SILO",
               "ATEK", "STEK", "IRON", "MSLO", "PDOX", "KENN", "GAP"]
_DEFENSES = ["PBOX", "HBOX", "GUN", "FTUR", "TSLA", "SAM", "AGUN"]
_WALLS = ["BRIK", "SBAG", "CYCL", "FENC", "BARB"]

# Tanya/e7 alias display name
_DISPLAY = {"e7": "tanya"}  # surface tanya since packs use it


def _faction_from_prereqs(prereqs: str) -> str:
    p = prereqs.lower()
    if "structures.allies" in p or "vehicles.allies" in p or "infantry.allies" in p:
        return "Allied"
    if "structures.soviet" in p or "vehicles.soviet" in p or "infantry.soviet" in p:
        return "Soviet"
    # ~barracks (alias for either tent or barr) ⇒ both
    if "~barracks" in p or "techlevel.infonly" in p and "~barracks" in p:
        return "Allied+Soviet"
    parts = [s.strip() for s in p.split(",")]
    for token, fac in (
        ("~tent", "Allied"), ("tent", "Allied"),
        ("~barr", "Soviet"), ("barr", "Soviet"),
        ("~hpad", "Allied"), ("hpad", "Allied"),
        ("~afld", "Soviet"), ("afld", "Soviet"),
        ("~syrd", "Allied"), ("syrd", "Allied"),
        ("~spen", "Soviet"), ("spen", "Soviet"),
        ("kenn", "Soviet"), ("atek", "Allied"),
        ("stek", "Soviet"),
    ):
        if token in parts:
            return fac
    if "~barracks" in p:
        return "Allied+Soviet"
    return "Allied+Soviet"


def _clean_prereqs(prereqs: str) -> str:
    parts = [p.strip() for p in prereqs.split(",")]
    keep: list[str] = []
    for p in parts:
        if not p:
            continue
        if "techlevel" in p or "disabled" in p:
            continue
        if p.startswith("~!") or p.startswith("!"):
            continue
        # strip leading ~ for display (it's an "or-equivalent" marker)
        keep.append(p.lstrip("~"))
    return ", ".join(keep) or "(none from construction yard alone)"


def _format_row(s: dict) -> str:
    code = s["code"]
    disp = _DISPLAY.get(code, code)
    role = _role_for(code)
    parts: list[str] = []
    if s.get("cost") is not None:
        parts.append(f"${s['cost']}")
    if s.get("hp") is not None:
        parts.append(f"hp{s['hp']}")
    arms = s.get("armaments", [])
    if arms:
        a = arms[0]
        parts.append(f"rng{a['range_cells']:.1f}c")
        parts.append(f"dps{a['dps']:.1f}")
        if len(arms) > 1:
            b = arms[1]
            parts.append(
                f"+{b['name']}(rng{b['range_cells']:.1f}c/dps{b['dps']:.1f})"
            )
    elif not s.get("is_building"):
        parts.append("unarmed")
    if s.get("sight_cells") is not None:
        parts.append(f"sight{s['sight_cells']:.1f}c")
    if s.get("movement") and not s.get("is_building"):
        parts.append(s["movement"])
    if s.get("power") is not None and s.get("is_building"):
        pw = s["power"]
        parts.append(f"power{pw:+d}")
    stats = " ".join(parts)
    if role:
        return f"  {disp:<6} {role} — {stats}"
    return f"  {disp:<6} — {stats}"


def _build_full_codex() -> tuple[str, str]:
    """Returns (FULL_CODEX_TEXT, TECH_TREE_TEXT). Empty strings if
    the vendor checkout isn't reachable."""
    base = _vendor_rules_dir()
    if base is None:
        return "", ""
    try:
        actors: dict[str, dict] = {}
        for f in sorted((base / "rules").glob("*.yaml")):
            text = f.read_text(encoding="utf-8")
            for k, v in _parse_ra_yaml(text).items():
                if isinstance(v, dict) and k not in actors:
                    v["__source__"] = f.name
                    actors[k] = v
        weapons: dict[str, dict] = {}
        for f in sorted((base / "weapons").glob("*.yaml")):
            text = f.read_text(encoding="utf-8")
            for k, v in _parse_ra_yaml(text).items():
                if isinstance(v, dict) and k not in weapons:
                    weapons[k] = v
    except Exception as e:
        logger.debug("Vendor codex build failed: %s", e)
        return "", ""

    sections: list[tuple[str, list[str]]] = [
        ("Infantry", _INFANTRY),
        ("Vehicles", _VEHICLES),
        ("Aircraft", _AIRCRAFT),
        ("Ships", _SHIPS),
        ("Structures", _STRUCTURES),
        ("Defenses", _DEFENSES),
        ("Walls", _WALLS),
    ]
    out: list[str] = [
        "UNIT CODEX (full RA reference; identical for every model).",
        "Each row: <code> <role> — $cost hp<HP> rng<attack range, cells> "
        "dps<damage/sec> sight<vision, cells> <movement> [power].",
        "Engine HP / damage scale = vendor / 100; DPS = damage·burst·25 / "
        "reload·100 (the engine ticks at 25/sec).",
    ]
    for title, codes in sections:
        out.append("")
        out.append(f"{title}:")
        for code in codes:
            yc = _ALIASES.get(code.lower(), code)
            s = _actor_stats(actors, weapons, yc)
            # The PBOX in the engine has its M60mg attached at runtime
            # (gamerules.rs::from_ruleset) since the vendor YAML carries
            # an AttackGarrisoned armament only. Inject the same M60mg
            # numbers here so the codex matches engine behaviour.
            if code.lower() in ("pbox", "hbox") and not s.get("armaments"):
                ws = _weapon_summary(weapons, "M60mg")
                if ws:
                    s["armaments"] = [ws]
                    s["sight_cells"] = s.get("sight_cells", 6.0)
            if s.get("cost") is None and s.get("hp") is None:
                continue
            out.append(_format_row(s))

    codex_text = "\n".join(out)

    # Tech tree
    tt: list[str] = [
        "TECH TREE (faction availability + prerequisites).",
        "`~X` in vendor YAML = 'any of the X-providing structures'; "
        "factions are inferred from prereq tags. fact (construction "
        "yard) is the seed building; place_building does NOT require "
        "engine-side adjacency in this bench.",
    ]
    tt_by_section = [
        ("Infantry (built at tent/barr)", _INFANTRY),
        ("Vehicles (built at weap)", _VEHICLES),
        ("Aircraft (built at hpad / afld)", _AIRCRAFT),
        ("Ships (built at syrd / spen)", _SHIPS),
        ("Buildings (built at fact)", _STRUCTURES),
        ("Defenses (built at fact)", _DEFENSES),
        ("Walls (built at fact)", _WALLS),
    ]
    for title, codes in tt_by_section:
        tt.append("")
        tt.append(f"{title}:")
        for code in codes:
            yc = _ALIASES.get(code.lower(), code)
            s = _actor_stats(actors, weapons, yc)
            if not s.get("buildable"):
                continue
            disp = _DISPLAY.get(code.lower(), code.lower())
            faction = _faction_from_prereqs(s.get("prereqs", ""))
            prereqs = _clean_prereqs(s.get("prereqs", ""))
            tt.append(f"  {disp:<6} {faction:<14} prereqs: {prereqs}")
    tech_text = "\n".join(tt)
    return codex_text, tech_text


FULL_CODEX_TEXT, TECH_TREE_TEXT = _build_full_codex()


def _default_reference() -> str:
    """The canonical RA reference appended to every system prompt.
    Falls back to the curated legacy `_CODEX` view when the vendor
    checkout isn't reachable (e.g. wheel-only installs)."""
    if FULL_CODEX_TEXT and TECH_TREE_TEXT:
        return FULL_CODEX_TEXT + "\n\n" + TECH_TREE_TEXT
    # Legacy fallback: the original 30-entry curated codex.
    body = "\n".join(f"  {c:<5} {desc}" for c, desc in _CODEX.items())
    return (
        "UNIT CODEX (legacy fallback; vendor YAML unavailable).\n" + body
    )


def system_prompt(objective_text: str, codex_text: str = "") -> str:
    """Vendored system_v2.txt with {objective} filled, plus the RA
    encyclopaedia (full codex + tech tree) appended.

    If a non-empty `codex_text` is passed, it OVERRIDES the default —
    that's the legacy "scenario-scoped" path (`unit_codex(codes)`).
    Otherwise the full vendor-derived codex + tech tree is used so
    every model sees the same uniform reference.
    """
    s = _SYSTEM_TMPL.replace("{objective}", objective_text or "(none)")
    ref = codex_text if codex_text else _default_reference()
    if ref:
        s = s.rstrip() + "\n\n" + ref
    return s


def _enemy_split(render_state: dict):
    units, bldgs = [], []
    for e in render_state.get("enemy_summary", []) or []:
        rec = {
            "type": e.get("type") or "?",
            "id": e.get("id"),
            "cell_x": int(e.get("cell_x", 0)),
            "cell_y": int(e.get("cell_y", 0)),
        }
        (bldgs if e.get("is_building") else units).append(rec)
    for b in render_state.get("enemy_buildings_summary", []) or []:
        rec = {
            "type": b.get("kind") or b.get("type") or "?",
            "id": b.get("id"),
            "cell_x": int(b.get("cell_x", 0)),
            "cell_y": int(b.get("cell_y", 0)),
        }
        if rec not in bldgs:
            bldgs.append(rec)
    return units, bldgs


def state_from_render(render_state: dict) -> dict:
    """bench render_state → the `state` dict briefing_v2 expects."""
    own_b = render_state.get("own_buildings", []) or []
    enemies, enemy_b = _enemy_split(render_state)
    return {
        "tick": int(render_state.get("game_tick", 0) or 0),
        "economy": {
            "cash": int(render_state.get("cash", 0) or 0),
            "ore": int(render_state.get("resources", 0) or 0),
            "harvester_count": int(render_state.get("harvesters", 0) or 0),
        },
        "power_balance": int(render_state.get("power_provided", 0) or 0)
        - int(render_state.get("power_drained", 0) or 0),
        "explored_percent": round(
            float(render_state.get("explored_percent", 0.0) or 0.0), 1
        ),
        # Deliberately empty: the spatial channel is the PNG minimap
        # (sent as an image). Training strips the ASCII grid from
        # briefings (agent_rollout._strip_ascii_minimap) — it's
        # redundant, token-heavy, and a coordinate-counting crutch.
        "minimap": "",
        # Keep the REAL engine actor id (own_buildings now surfaces it,
        # mirroring units_summary) so repair / sell / power_down /
        # set_primary orders resolve. Fall back to the list index only
        # for legacy obs that predate the id field.
        "buildings_summary": [
            {
                "type": b.get("type"),
                "id": b.get("id") if b.get("id") not in (None, "") else i,
                "cell_x": b.get("cell_x"),
                "cell_y": b.get("cell_y"),
                "hp": b.get("hp"),
                "is_primary": b.get("is_primary"),
            }
            for i, b in enumerate(own_b)
        ],
        "units_summary": render_state.get("units_summary", []) or [],
        "enemy_summary": enemies,
        "enemy_buildings_summary": enemy_b,
        "production_items": list(render_state.get("production", []) or []),
        "available_production": list(
            render_state.get("available_production", []) or []
        ),
    }


import re as _re


def briefing(render_state: dict) -> str:
    text = _BRIEF.format_state_briefing_v2(state_from_render(render_state))
    # Vendored briefing_v2's empty-own-buildings fallback emits a
    # confusing "Buildings: ? ()" for unit-only scenarios (the agent
    # has no base). Sanitize that one degenerate line in our wrapper
    # (the vendored module stays byte-identical for the drift test).
    text = _re.sub(
        r"(?m)^Buildings: (?:\?|0|none)? ?\(\)\s*$",
        "Buildings: none (you command mobile units only; enemy "
        "buildings appear under Enemies once scouted)",
        text,
    )
    return text


def structured_fog(render_state: dict) -> str:
    """Text 'Unexplored regions' block — the structured-fog channel
    that substitutes for the PNG minimap (text-vs-vision A/B)."""
    from .structured_fog import format_structured_fog

    obs = render_state.get("_raw") or {}
    bounds = tuple(render_state.get("bounds", (0, 0, 64, 64)))
    return format_structured_fog(obs, bounds)


# ── Image-primary channel ────────────────────────────────────────────
# The `image` perception modality: the text briefing is redacted of
# every coordinate, so the PNG minimap (with per-unit labels) is the
# ONLY source of spatial state. Units carry a short legible handle
# (`tank-1`, `enemy-2`) shown identically in the briefing roster, on
# the minimap, and as the id the model passes to its tools.
_FRIENDLY_TYPE = {
    "1tnk": "tank", "2tnk": "tank", "3tnk": "tank", "4tnk": "tank",
    "jeep": "jeep", "apc": "apc", "arty": "arty", "harv": "harvester",
    "mcv": "mcv", "e1": "rifle", "e2": "grenadier", "e3": "rocket",
    "e6": "engineer",
}


def _friendly_word(actor_type: object) -> str:
    """A short legible word for an actor type — `1tnk` → `tank`. Falls
    back to the raw type (so buildings read `proc-1`, `pbox-1`)."""
    t = str(actor_type or "").lower()
    return _FRIENDLY_TYPE.get(t, t or "unit")


def perception_labels(
    render_state: dict, prior: dict[str, str] | None = None
) -> dict[str, str]:
    """Per-actor handle map for the image-primary channel: engine id
    (str) → a short legible label. Own actors get a type-word handle
    (`tank-1`, `jeep-2`, `proc-1`); enemy actors get `enemy-N`.

    `prior` (the previous turn's map) is carried forward so a label
    stays pinned to its actor for the actor's whole lifetime — an
    enemy revealed mid-episode keeps the same handle on every later
    turn instead of renumbering when a lower-id foe is later scouted.
    New actors are assigned in engine-id order, continuing the count."""
    labels: dict[str, str] = dict(prior or {})
    # Highest index already used per prefix — so new handles continue
    # the sequence rather than colliding with carried-over ones.
    used: dict[str, int] = {}
    for lab in labels.values():
        pre, _, num = lab.rpartition("-")
        if pre and num.isdigit():
            used[pre] = max(used.get(pre, 0), int(num))

    def _assign(items, prefix_fn):
        for a in sorted(items, key=lambda a: int(a.get("id", 0) or 0)):
            aid = a.get("id")
            if aid is None or str(aid) in labels:
                continue
            pre = prefix_fn(a)
            used[pre] = used.get(pre, 0) + 1
            labels[str(aid)] = f"{pre}-{used[pre]}"

    _assign(
        render_state.get("units_summary", []) or [],
        lambda a: _friendly_word(a.get("type")),
    )
    _assign(
        render_state.get("own_buildings", []) or [],
        lambda a: _friendly_word(a.get("type")),
    )
    enemies = list(render_state.get("enemy_summary", []) or [])
    enemies += list(render_state.get("enemy_buildings_summary", []) or [])
    _assign(enemies, lambda a: "enemy")
    return labels


def briefing_image_primary(render_state: dict, labels: dict[str, str]) -> str:
    """Image-primary briefing: every coordinate redacted. The text
    keeps the non-spatial scaffolding (funds, power, the unit roster as
    label+type handles, production) so the model knows WHAT it has —
    the minimap image is the only place it learns WHERE anything is."""
    st = state_from_render(render_state)
    econ = st["economy"]
    funds = econ["cash"] + econ["ore"]
    out = [
        f"--- TURN BRIEFING (tick {st['tick']}) ---",
        f"Funds: ${funds} (cash=${econ['cash']} + ore=${econ['ore']}) | "
        f"Power: {st['power_balance']:+d} | "
        f"Harvesters: {econ['harvester_count']} | "
        f"Explored: {st['explored_percent']}%",
    ]
    units = sorted(
        st["units_summary"], key=lambda u: int(u.get("id", 0) or 0)
    )
    out.append(f"Your units ({len(units)}) — find each on the minimap:")
    idle = []
    for u in units:
        lab = labels.get(str(u.get("id")), str(u.get("id")))
        out.append(f"  {lab}  ({u.get('type') or '?'})")
        if str(u.get("activity", "")).lower() in ("", "idle"):
            idle.append(lab)
    if idle:
        out.append(f"Idle: {', '.join(idle)}")
    bsum = st["buildings_summary"]
    if bsum:
        out.append(f"Your buildings ({len(bsum)}) — find each on the minimap:")
        for b in bsum:
            lab = labels.get(str(b.get("id")), str(b.get("id")))
            out.append(f"  {lab}  ({b.get('type') or '?'})")
    else:
        out.append("Buildings: none (you command mobile units only).")
    out.append(
        "Enemies: NOT listed here — read the minimap. Enemy markers are "
        "labelled enemy-1, enemy-2, … (only those in your units' sight; "
        "scout the fog to reveal more)."
    )
    prod = st["production_items"]
    out.append(f"Production: {', '.join(prod) if prod else 'IDLE'}")
    out.append(
        "ALL unit and enemy POSITIONS are on the minimap image only — "
        "each marker is labelled with the id shown above; pass that id "
        "to your tools."
    )
    return "\n".join(out)


def minimap_b64(
    render_state: dict, terrain_png: bytes | None,
    explored_history: set | None,
    constant_colors: bool = False,
) -> str | None:
    """Vendored training bitmap minimap (terrain + 3-tier fog + grid +
    axis labels + legend). `constant_colors` ⇒ one colour for all own
    units and one for all enemies (easy/medium); per-type palette
    otherwise (hard). None ⇒ graceful text-only."""
    obs = render_state.get("_raw") or {}
    if not obs or not terrain_png:
        return None
    if constant_colors:
        # Empty type maps → the renderer falls back to a single
        # per-group (own / enemy) style instead of the per-type palette.
        own_types: dict = {}
        enemy_types: dict = {}
    else:
        own_types = {
            str(u["id"]): str(u.get("type") or "?")
            for u in render_state.get("units_summary", []) or []
            if u.get("id") is not None
        }
        enemy_types = {
            str(e["id"]): str(e.get("type") or "?")
            for e in render_state.get("enemy_summary", []) or []
            if e.get("id") is not None
        }
    # Surface the model's OWN buildings as a SEPARATE layer (mirroring
    # `enemy_buildings_summary`) so the vendor renderer can paint them
    # with the building-shape style (filled square + outline) instead of
    # the unit-shape style. Pre-fix history: own buildings were merged
    # INTO `unit_positions`, making the model's base look like a cluster
    # of tiny unit dots — readable only by the rare model that can
    # distinguish small colour variations. Now buildings get the dedicated
    # `filled_square` marker the human-Play tab already uses.
    obs = dict(obs)
    own_buildings_summary = []
    for b in (render_state.get("own_buildings") or []):
        if not isinstance(b, dict):
            continue
        bid = b.get("id")
        bx = b.get("cell_x")
        by = b.get("cell_y")
        btype = b.get("type")
        if bid is None or bx is None or by is None:
            continue
        own_buildings_summary.append({
            "id": str(bid),
            "cell_x": int(bx),
            "cell_y": int(by),
            "type": str(btype) if btype else "?",
        })
    obs["own_buildings_summary"] = own_buildings_summary
    # Carry resource_cells through (the rust_adapter populates it; if
    # the caller already wrote it into obs we keep that).
    if "resource_cells" not in obs and render_state.get("resource_cells"):
        obs["resource_cells"] = render_state["resource_cells"]
    try:
        png = _MM.render(
            obs=obs,
            terrain_png_bytes=terrain_png,
            map_width=int(render_state.get("map_width", 64) or 64),
            map_height=int(render_state.get("map_height", 64) or 64),
            bounds=tuple(render_state.get("bounds", (0, 0, 64, 64))),
            explored_history=(
                explored_history if explored_history is not None else set()
            ),
            own_unit_types=own_types,
            enemy_unit_types=enemy_types,
        )
        if isinstance(png, (bytes, bytearray)) and png:
            return base64.b64encode(png).decode("ascii")
    except Exception as e:  # noqa: BLE001 — vision optional
        logger.debug("minimap_v2 render failed: %s", e)
    return None
