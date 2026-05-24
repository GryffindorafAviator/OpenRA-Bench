"""Prompt/briefing/minimap = the training v2 format, by construction.

Loads the byte-vendored training artifacts (`_vendor/system_v2.txt`,
`briefing_v2.py`, `minimap_v2.py`) and adapts the bench `render_state`
into the exact `state` dict they expect, so a bench transcript is
indistinguishable in format from a training rollout. Adds one bench
extra the user asked for: a scenario-scoped unit codex (accurate RA
ruleset numbers) appended to the system prompt — coherent with the
combat-model section that references rng/DPS/sight.
"""

from __future__ import annotations

import base64
import importlib.util as _iu
import logging
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
_SYSTEM_TMPL = (_VENDOR / "system_v2.txt").read_text()


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


def system_prompt(objective_text: str, codex_text: str = "") -> str:
    """Vendored system_v2.txt with {objective} filled; optional codex
    appended (the only bench-specific addition)."""
    s = _SYSTEM_TMPL.replace("{objective}", objective_text or "(none)")
    if codex_text:
        s = s.rstrip() + "\n\n" + codex_text
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
