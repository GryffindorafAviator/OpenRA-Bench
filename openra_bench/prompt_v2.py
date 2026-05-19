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
    "e1":   "$100  hp50   rng3c  dps6   sight4c  foot",
    "e2":   "$160  hp50   rng4c  dps9   sight4c  foot (grenadier)",
    "e3":   "$300  hp45   rng4c  dps12  sight5c  foot (anti-armour)",
    "e6":   "$500  hp25   unarmed (engineer: capture/repair)  sight4c",
    "dog":  "$200  hp12   rng1c  dps10  sight5c  fast",
    "medi": "$800  hp80   unarmed (heals infantry)  sight4c",
    "spy":  "$500  hp25   unarmed (infiltrate)  sight5c",
    "thf":  "$500  hp25   unarmed (steal credits)  sight4c",
    "jeep": "$600  hp150  rng4c  dps8   sight7c  fast wheeled (scout)",
    "1tnk": "$700  hp300  rng4c  dps14  sight6c  tracked",
    "2tnk": "$850  hp400  rng4.75c dps22 sight6c  tracked (main)",
    "3tnk": "$950  hp450  rng5c  dps30  sight6c  tracked (heavy)",
    "4tnk": "$1500 hp600  rng5c  dps40+ sight6c  tracked (mammoth, +AA)",
    "arty": "$600  hp75   rng7c  dps45  sight5c  slow (siege, fragile)",
    "apc":  "$800  hp200  rng4c  dps8   sight6c  carries 5 infantry",
    "harv": "$1100 hp600  unarmed (gathers ore)  sight4c",
    "mcv":  "$2500 hp600  unarmed (deploys into a fact)  sight5c",
    "fact": "construction yard — builds structures; LOSS-CRITICAL",
    "powr": "power plant (supplies power)",
    "apwr": "advanced power plant",
    "proc": "ore refinery (ore→credits; economy core)",
    "barr": "Soviet barracks (trains infantry)",
    "tent": "Allied barracks (trains infantry)",
    "weap": "war factory (builds vehicles)",
    "dome": "radar dome (tech / map reveal)",
    "silo": "ore silo (stores ore)",
    "gun":  "gun turret — anti-armour defence  rng6c",
    "pbox": "pillbox — anti-infantry defence  rng5c",
    "tsla": "Tesla coil — heavy defence  rng7c (needs power)",
    "sam":  "SAM site — anti-air defence",
}


def unit_codex(codes) -> str:
    rows = [c for c in sorted(set(codes)) if c in _CODEX]
    if not rows:
        return ""
    body = "\n".join(f"  {c:<5} {_CODEX[c]}" for c in rows)
    return (
        "UNIT CODEX (this scenario — cost/hp/range(c)/dps/sight(c)):\n"
        + body
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
        "buildings_summary": [
            {"type": t, "id": i, "cell_x": x, "cell_y": y}
            for i, (t, x, y) in enumerate(
                [(b["type"], b["cell_x"], b["cell_y"]) for b in own_b]
            )
        ],
        "units_summary": render_state.get("units_summary", []) or [],
        "enemy_summary": enemies,
        "enemy_buildings_summary": enemy_b,
        "production_items": list(render_state.get("production", []) or []),
        "available_production": list(
            render_state.get("available_production", []) or []
        ),
    }


def briefing(render_state: dict) -> str:
    return _BRIEF.format_state_briefing_v2(state_from_render(render_state))


def minimap_b64(
    render_state: dict, terrain_png: bytes | None,
    explored_history: set | None,
) -> str | None:
    """Vendored training bitmap minimap (terrain + 3-tier fog + grid +
    axis labels). None ⇒ graceful text-only."""
    obs = render_state.get("_raw") or {}
    if not obs or not terrain_png:
        return None
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
