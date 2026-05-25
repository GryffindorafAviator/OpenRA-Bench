"""Provider-agnostic model agent.

Turns a `RustObsAdapter.render_state()` into a Training-compatible text
briefing (+ optional minimap image), calls a `ChatProvider`, and parses
tool calls back into `openra_train.Command` objects. Exposes an
`agent_fn` matching `eval_core`'s `(render_state, Command) -> [Command]`
contract.

Tool contract mirrors OpenRA-RL-Training so models trained there behave
consistently: `move_units(unit_ids, target_x, target_y)`,
`attack_unit(unit_ids, target_id)`, `observe()`. The scenario's `tools`
list filters which are offered.
"""

from __future__ import annotations

import logging
from typing import Any

from .providers import ChatProvider, ProviderConfig, make_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are commanding units in Command & Conquer: Red Alert.\n"
    "Each turn you receive a BRIEFING (and, when available, a MINIMAP image: "
    "bright=visible, dim=explored, black=unknown fog).\n"
    "Units are listed as `<id> <type> @(x,y)` (with `-> (tx,ty)` if moving).\n"
    "Pass numeric unit IDs to tools, e.g. unit_ids=[1004,1005].\n"
    "Every turn MUST include at least one tool call. Think briefly, then act."
)

_TOOL_SCHEMAS: dict[str, dict] = {
    "move_units": {
        "type": "function",
        "function": {
            "name": "move_units",
            "description": "Move the given units to a map cell. Units auto-fire "
            "opportunistically en route. Use to position/scout/retreat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_x": {"type": "integer"},
                    "target_y": {"type": "integer"},
                },
                "required": ["unit_ids", "target_x", "target_y"],
            },
        },
    },
    "attack_unit": {
        "type": "function",
        "function": {
            "name": "attack_unit",
            "description": "Order the given units to pathfind to and focus-fire "
            "a specific enemy actor id until it dies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "guard": {
        "type": "function",
        "function": {
            "name": "guard",
            "description": "Order the given units to guard (follow and stay "
            "near) a friendly actor id, repositioning as it moves.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "observe": {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Take no action; advance the game and re-observe.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "surrender": {
        "type": "function",
        "function": {
            "name": "surrender",
            "description": "Concede the match. Use only when the position "
            "is unrecoverable; ends the scenario as a loss.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    "enter_transport": {
        "type": "function",
        "function": {
            "name": "enter_transport",
            "description": "Order passenger units (infantry) to walk to "
            "and board a transport actor id (e.g. an APC).",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "unload": {
        "type": "function",
        "function": {
            "name": "unload",
            "description": "Order transport(s) (by id, in unit_ids) to "
            "eject all carried passengers next to it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["unit_ids"],
            },
        },
    },
    "capture_actor": {
        "type": "function",
        "function": {
            "name": "capture_actor",
            "description": "Order engineer(s) (actor_type e6) to walk "
            "to an enemy BUILDING and capture it — on arrival the "
            "building's owner transfers to your player and the "
            "engineer is consumed. Non-engineer units are rejected; "
            "friendly / non-building targets are ignored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "c4_detonate": {
        "type": "function",
        "function": {
            "name": "c4_detonate",
            "description": "Order Tanya (actor_type tanya) to walk to "
            "an enemy BUILDING, plant C4, and instantly destroy it. "
            "Tanya survives the detonation. Non-tanya subjects are "
            "rejected; friendly / non-building targets are ignored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "infiltrate": {
        "type": "function",
        "function": {
            "name": "infiltrate",
            "description": "Order a spy (actor_type spy) or thief "
            "(actor_type thf) to walk into an enemy BUILDING. On "
            "arrival the infiltrator is consumed and one of two "
            "effects fires depending on the infiltrator's type: a "
            "spy reveals every structure owned by the target's "
            "owner (one-shot scan, survives fog); a thief drains a "
            "chunk of the target owner's cash to your player (only "
            "when the target is a proc or silo). Friendly / "
            "non-building targets are ignored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_id": {"type": "integer"},
                },
                "required": ["unit_ids", "target_id"],
            },
        },
    },
    "set_stance": {
        "type": "function",
        "function": {
            "name": "set_stance",
            "description": "Set engagement stance for units: 0=HoldFire, "
            "1=ReturnFire, 2=Defend, 3=AttackAnything.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "stance": {"type": "integer", "minimum": 0, "maximum": 3},
                },
                "required": ["unit_ids", "stance"],
            },
        },
    },
    "set_primary": {
        "type": "function",
        "function": {
            "name": "set_primary",
            "description": "Designate a production building (by id, in "
            "unit_ids) as the PRIMARY producer for its type; newly "
            "produced units of that category spawn from / rally there.",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["unit_ids"],
            },
        },
    },
    "patrol": {
        "type": "function",
        "function": {
            "name": "patrol",
            "description": "Patrol order (accepted; currently a no-op, "
            "matching the reference engine).",
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["unit_ids"],
            },
        },
    },
    "fire_superweapon": {
        "type": "function",
        "function": {
            "name": "fire_superweapon",
            "description": (
                "Fire one of the three superweapons (kind = 'mslo' "
                "nuke / 'iron' iron curtain / 'pdox' chronosphere). "
                "The agent must own a launcher building of the matching "
                "kind AND the weapon must be fully charged; otherwise "
                "the order is silently dropped. Nuke needs target_x / "
                "target_y (the impact cell). Iron curtain needs "
                "target_id (a friendly actor to make invulnerable for "
                "~750 ticks). Chronosphere needs both target_x / "
                "target_y (destination cell) AND target_id (the "
                "friendly actor to teleport)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["mslo", "iron", "pdox"],
                    },
                    "target_x": {"type": "integer"},
                    "target_y": {"type": "integer"},
                    "target_id": {"type": "integer"},
                },
                "required": ["kind"],
            },
        },
    },
}


def _units_xy(name: str, desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}},
                    "target_x": {"type": "integer"},
                    "target_y": {"type": "integer"},
                },
                "required": ["unit_ids", "target_x", "target_y"],
            },
        },
    }


def _units_only(name: str, desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "unit_ids": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["unit_ids"],
            },
        },
    }


def _item_only(name: str, desc: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    }


_TOOL_SCHEMAS.update(
    {
        "attack_move": _units_xy(
            "attack_move", "Move toward a cell, engaging hostiles encountered."
        ),
        "harvest": _units_xy(
            "harvest", "Send harvesters to collect ore at a resource cell."
        ),
        "set_rally_point": _units_xy(
            "set_rally_point", "Set a production building's unit rally cell."
        ),
        "stop": _units_only("stop", "Cancel the units' current orders (go idle)."),
        "deploy": _units_only("deploy", "Transform an MCV into a construction yard."),
        "sell": _units_only("sell", "Sell a building for a partial refund."),
        "repair": _units_only("repair", "Toggle repair on a damaged building."),
        "power_down": _units_only("power_down", "Toggle a building's power."),
        "build": _item_only(
            "build", "Queue production of a unit/building by type (e.g. 'e1')."
        ),
        "cancel_production": _item_only(
            "cancel_production", "Cancel the last queued item of this type (refund)."
        ),
        "place_building": {
            "type": "function",
            "function": {
                "name": "place_building",
                "description": "Place a completed building at a cell.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "target_x": {"type": "integer"},
                        "target_y": {"type": "integer"},
                    },
                    "required": ["item", "target_x", "target_y"],
                },
            },
        },
    }
)
# Aliases tolerated from models trained on slightly different names.
_TOOL_ALIASES = {"attack_target": "attack_unit", "stop_units": "stop"}


# Scenario-agnostic safe default: the core movement/combat verbs every
# scenario needs. A scenario that does not declare `tools:` gets this
# set (NOT all 17 — economy/structure/concede verbs are noise on a
# perception or combat scenario). A scenario opts into more via its
# `tools:` allowlist; `"*"`/`"all"` exposes everything.
DEFAULT_CORE_TOOLS = (
    "move_units",
    "attack_unit",
    "attack_move",
    "stop",
    "observe",
)


def _tool_schemas(allowed: list[str] | None) -> list[dict]:
    """Resolve the tool set offered to the model:

    * unset / empty       → DEFAULT_CORE_TOOLS
    * ["*"] or ["all"]    → every implemented tool
    * explicit list       → exactly those (intersected with known tools;
                            unknown names are ignored, not errors)

    `observe` (the safe no-op) is always included so the agent can
    always emit a valid turn even under the tightest allowlist.
    """
    if not allowed:
        names: list[str] = list(DEFAULT_CORE_TOOLS)
    elif any(a in ("*", "all") for a in allowed):
        names = list(_TOOL_SCHEMAS)
    else:
        names = list(allowed)
    out = [_TOOL_SCHEMAS[n] for n in names if n in _TOOL_SCHEMAS]
    if "observe" not in {t["function"]["name"] for t in out}:
        out.append(_TOOL_SCHEMAS["observe"])  # always allow a no-op
    return out


def build_briefing(render_state: dict, objective: str = "") -> str:
    """Training-style text state. Self-contained (no engine handles)."""
    lines: list[str] = []
    if objective:
        lines.append(f"OBJECTIVE: {objective}")
    lines.append(
        f"tick={render_state.get('game_tick', 0)} "
        f"explored={render_state.get('explored_percent', 0.0):.1f}%"
    )
    own = render_state.get("units_summary", []) or []
    lines.append(f"\nYOUR UNITS ({len(own)}):")
    for u in own:
        act = u.get("activity")
        suffix = f", {act}" if act and act != "idle" else ""
        lines.append(
            f"  {u['id']} {u.get('type') or 'unit'} @({u['cell_x']},{u['cell_y']}){suffix}"
        )
    enemy = render_state.get("enemy_summary", []) or []
    if enemy:
        lines.append(f"\nVISIBLE ENEMIES ({len(enemy)}):")
        for e in enemy:
            kind = "building" if e.get("is_building") else (e.get("type") or "unit")
            lines.append(f"  {e['id']} {kind} @({e['cell_x']},{e['cell_y']})")
    else:
        lines.append("\nVISIBLE ENEMIES: none (scout the fog)")
    # Base / economy state (present on economy/building scenarios).
    if "cash" in render_state:
        net = render_state.get("power_provided", 0) - render_state.get(
            "power_drained", 0
        )
        lines.append(
            f"\nBASE: cash={render_state.get('cash', 0)} power_net={net}"
        )
        obs_b = render_state.get("own_buildings", []) or []
        if obs_b:
            lines.append(f"BUILDINGS ({len(obs_b)}):")
            for b in obs_b:
                lines.append(
                    f"  {b.get('type','?')} @({b['cell_x']},{b['cell_y']})"
                )
        prod = render_state.get("production", []) or []
        if prod:
            lines.append(f"PRODUCING: {', '.join(prod)}")
    return "\n".join(lines)


def _render_minimap_b64(
    render_state: dict, terrain_png: bytes | None = None
) -> str | None:
    """Best-effort minimap PNG. With `terrain_png` uses the training
    renderer (real terrain + an embedded legend the model can read);
    else the bench fallback. None ⇒ graceful text-only."""
    try:
        from .minimap import render_b64

        return render_b64(render_state, terrain_png)
    except Exception as e:  # noqa: BLE001 — vision is optional
        logger.debug("minimap render skipped: %s", e)
        return None


def _to_commands(
    tool_calls: list[dict], Command: Any, label_to_id: dict | None = None
) -> list:
    # In the image-primary channel the model references units by the
    # legible handle shown on the minimap (`tank-1`); map it back to the
    # engine actor id. Numeric ids (every other channel) pass straight
    # through — the lookup simply misses.
    label_to_id = label_to_id or {}

    def _rid(x: Any) -> str:
        return label_to_id.get(str(x), str(x))

    cmds = []
    for call in tool_calls:
        name = _TOOL_ALIASES.get(call.get("name", ""), call.get("name", ""))
        args = call.get("arguments") or {}
        try:
            if name == "move_units":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.move_units(ids, int(args["target_x"]), int(args["target_y"]))
                )
            elif name == "attack_unit":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(Command.attack_unit(ids, _rid(args["target_id"])))
            elif name == "guard":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(Command.guard(ids, _rid(args["target_id"])))
            elif name == "enter_transport":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.enter_transport(ids, _rid(args["target_id"]))
                )
            elif name == "capture_actor":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.capture_actor(ids, _rid(args["target_id"]))
                )
            elif name == "c4_detonate":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.c4_detonate(ids, _rid(args["target_id"]))
                )
            elif name == "infiltrate":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.infiltrate(ids, _rid(args["target_id"]))
                )
            elif name == "observe":
                cmds.append(Command.observe())
            elif name == "surrender":
                cmds.append(Command.surrender())
            elif name == "set_stance":
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(Command.set_stance(ids, int(args["stance"])))
            elif name == "patrol":
                cmds.append(Command.patrol([_rid(i) for i in args["unit_ids"]]))
            elif name in ("attack_move", "harvest", "set_rally_point"):
                ids = [_rid(i) for i in args["unit_ids"]]
                fn = getattr(Command, name)
                cmds.append(fn(ids, int(args["target_x"]), int(args["target_y"])))
            elif name in (
                "stop",
                "deploy",
                "sell",
                "repair",
                "power_down",
                "set_primary",
                "unload",
            ):
                ids = [_rid(i) for i in args["unit_ids"]]
                cmds.append(getattr(Command, name)(ids))
            elif name in ("build", "cancel_production"):
                cmds.append(getattr(Command, name)(str(args["item"])))
            elif name == "place_building":
                cmds.append(
                    Command.place_building(
                        str(args["item"]), int(args["target_x"]), int(args["target_y"])
                    )
                )
            elif name == "fire_superweapon":
                kind = str(args["kind"])
                tx = args.get("target_x")
                ty = args.get("target_y")
                cell = (
                    (int(tx), int(ty))
                    if tx is not None and ty is not None
                    else None
                )
                tid = args.get("target_id")
                tid_str = _rid(tid) if tid is not None else None
                cmds.append(
                    Command.fire_superweapon(kind, cell, tid_str)
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("dropping malformed tool call %s: %s", call, e)
    return cmds


def _image_primary_tools(tools: list[dict]) -> list[dict]:
    """Re-type unit/target handles as strings for the image-primary
    channel: the model references actors by the legible label drawn on
    the minimap (`tank-1`, `enemy-2`), not numeric engine ids. The
    `_to_commands` `label_to_id` map turns them back into engine ids."""
    import copy

    out = copy.deepcopy(tools)
    for t in out:
        props = (
            t.get("function", {}).get("parameters", {}).get("properties", {})
        )
        ui = props.get("unit_ids")
        if isinstance(ui, dict) and ui.get("type") == "array":
            ui["items"] = {"type": "string"}
            ui["description"] = (
                'unit handles EXACTLY as labelled on the minimap, '
                'e.g. ["tank-1","jeep-2"]'
            )
        tid = props.get("target_id")
        if isinstance(tid, dict):
            tid["type"] = "string"
            tid["description"] = (
                'the target actor\'s handle as labelled on the minimap, '
                'e.g. "enemy-1"'
            )
    return out


class ModelAgent:
    """One instance per episode (keeps bounded chat history).

    Usage:
        agent = ModelAgent(cfg, allowed_tools=compiled.scenario.tools,
                            objective=compiled.scenario.description)
        result = run_level(compiled, agent.agent_fn, seed=...)
    """

    def __init__(
        self,
        cfg: ProviderConfig,
        allowed_tools: list[str] | None = None,
        objective: str = "",
        provider: ChatProvider | None = None,
        system_extra: str = "",
        base_map: str = "",
        unit_codex: str = "",
        level: str = "",
        fog_mode: str = "",
        agent_faction: str = "",
        enemy_faction: str = "",
    ):
        self.cfg = cfg
        self.objective = objective
        self._agent_faction = (agent_faction or "").lower() or None
        self._enemy_faction = (enemy_faction or "").lower() or None
        self.tools = _tool_schemas(allowed_tools)
        self.provider = provider or make_provider(cfg)
        self._level = level
        # Scenario config wins over the model-side cfg default.
        self._fog_mode = fog_mode or getattr(cfg, "fog_mode", "vision")
        # Image-primary channel: the text briefing carries no positions —
        # the labelled minimap is the sole spatial source, and the model
        # references units by those labels. Re-type the tool handles to
        # strings; `_labels` / `_label_to_id` are rebuilt each turn.
        self._image_primary = self._fog_mode.startswith("image")
        if self._image_primary:
            self.tools = _image_primary_tools(self.tools)
        self._labels: dict[str, str] = {}
        self._label_to_id: dict[str, str] = {}
        # Real terrain (map.png from the .oramap) for the vendored
        # training bitmap minimap; persistent fog history across turns.
        self._terrain: bytes | None = None
        self._explored_history: set = set()
        if base_map:
            try:
                from .minimap import terrain_png_for

                self._terrain = terrain_png_for(base_map)
            except Exception:  # noqa: BLE001
                self._terrain = None
        # System prompt = vendored training system_v2 (objective lives
        # HERE, not per-turn) + the scenario unit codex. Falls back to
        # the legacy prompt only if the vendored template is missing.
        try:
            from .prompt_v2 import system_prompt as _sysp

            sys_content = _sysp(self.objective, unit_codex)
        except Exception:  # noqa: BLE001
            sys_content = SYSTEM_PROMPT + (
                f"\n\n{system_extra}" if system_extra else ""
            )
        if self._image_primary:
            sys_content += (
                "\n\nPERCEPTION MODE — IMAGE-PRIMARY. The text briefing "
                "lists WHAT units exist but never where anything is. "
                "Every position — your units AND the enemy — is shown "
                "ONLY on the minimap image. Each marker is tagged with a "
                "legible label (tank-1, jeep-2, enemy-1). Read the image "
                "to locate units and threats; pass those exact labels as "
                "the ids in your tool calls (e.g. unit_ids=[\"tank-1\"])."
            )
        self.history: list[dict] = [{"role": "system", "content": sys_content}]
        self.stats = {"turns": 0, "tool_calls": 0, "empty_replies": 0}
        # Audit-format capture (FullPlayback). When `audit_capture` is
        # True the agent stores the per-turn briefing, the provider's
        # literal request/response, and exposes the system prompt so
        # the audit JSONL line for the turn carries everything.
        # Default off — zero overhead for normal runs.
        self.audit_capture: bool = False
        self.last_briefing: str = ""
        self.last_request: dict | None = None
        self.last_response: dict | None = None
        self.system_prompt: str = sys_content
        # Controller contract (openra_bench/controller.py): a ModelAgent
        # IS a Controller — it exposes `name`, `reset`, `act` so the
        # eval loop, the 1v1 harness, and the human-labeling harness can
        # all drive it interchangeably with any other policy backend.
        self.name = getattr(cfg, "model", None) or "model"

    def _image_primary_message(self, render_state: dict) -> dict:
        """Image-primary turn message: a position-redacted text briefing
        plus a labelled minimap — the minimap is the ONLY place the
        model learns where its units and the enemy are."""
        from .prompt_v2 import briefing_image_primary, perception_labels

        # Carry last turn's map forward so a label stays pinned to its
        # actor for the whole episode (stable handles across turns).
        self._labels = perception_labels(render_state, self._labels)
        self._label_to_id = {v: k for k, v in self._labels.items()}
        text = briefing_image_primary(render_state, self._labels)
        faction = self._faction_line()
        if faction:
            text = f"{faction}\n{text}"
        b64 = None
        try:
            import base64
            import io

            from .minimap import render_tactical_minimap

            # Keep the PNG ≤ ~1560px wide so the vision API does not
            # downscale it (which would shrink the unit labels below
            # legibility); the 6px base cell × scale sets the width.
            rows = [
                r for r in (render_state.get("minimap") or "").split("\n")
                if r
            ]
            w = max((len(r) for r in rows), default=64)
            scale = max(2, min(6, 1560 // max(1, w * 6)))
            img = render_tactical_minimap(
                render_state, scale=scale, unit_labels=self._labels,
            )
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, "PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001 — degrade to text-only on render fail
            b64 = None
        if b64:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        return {"role": "user", "content": text}

    def _faction_line(self) -> str:
        """One-line faction reminder for the per-turn briefing. Tells
        the model which subset of the full RA codex IT can produce —
        the codex itself stays uniform across packs."""
        if not self._agent_faction:
            return ""
        ag = self._agent_faction
        en = self._enemy_faction or "(unknown)"
        return f"FACTION: You are {ag}. Your opponent is {en}."

    def _user_message(self, render_state: dict) -> dict:
        # Image-primary channel builds its own (position-redacted)
        # briefing + labelled minimap — dispatch before the text path.
        if self._image_primary:
            return self._image_primary_message(render_state)
        # Briefing = vendored training briefing_v2 (one unit/line,
        # "moving to (x,y)", Idle list). Objective is in the system
        # prompt now, so it's NOT repeated here.
        try:
            from .prompt_v2 import briefing as _v2_brief

            text = _v2_brief(render_state)
        except Exception:  # noqa: BLE001 — never break a turn
            text = build_briefing(render_state, self.objective)
        faction = self._faction_line()
        if faction:
            text = f"{faction}\n{text}"
        # Structured channel: NO image — append the text "Unexplored
        # regions" block instead (text-vs-vision A/B). Covers both
        # `structured` (fogged) and `structured-clear` (no fog — under
        # reveal_map the block reports zero unexplored regions).
        if self._fog_mode.startswith("structured"):
            try:
                from .prompt_v2 import structured_fog as _v2_fog

                text = f"{text}\n\n{_v2_fog(render_state)}"
            except Exception:  # noqa: BLE001
                pass
            return {"role": "user", "content": text}
        if self.cfg.vision:
            # Per-type colours on hard; constant own/enemy on
            # easy/medium; overridable via cfg.minimap_color_mode.
            cm = getattr(self.cfg, "minimap_color_mode", "auto")
            constant = cm == "constant" or (
                cm == "auto" and self._level in ("easy", "medium")
            )
            b64 = None
            try:
                from .prompt_v2 import minimap_b64 as _v2_mm

                b64 = _v2_mm(
                    render_state, self._terrain, self._explored_history,
                    constant_colors=constant,
                )
            except Exception:  # noqa: BLE001
                b64 = None
            if b64 is None:
                b64 = _render_minimap_b64(render_state, self._terrain)
            if b64:
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
        return {"role": "user", "content": text}

    @staticmethod
    def _window(history: list[dict], max_turns: int) -> list[dict]:
        """Wire-history sliding window: keep all leading system
        messages + the last `max_turns` user-led groups. Slicing on a
        user boundary keeps every assistant↔tool pairing intact (only
        whole older groups are dropped, so no dangling tool replies).
        `self.history` itself is untouched — playback keeps the full
        transcript; only what's POSTED is bounded."""
        if max_turns <= 0:
            return history
        lead = 0
        while lead < len(history) and history[lead].get("role") == "system":
            lead += 1
        user_idx = [
            i for i in range(lead, len(history))
            if history[i].get("role") == "user"
        ]
        if len(user_idx) <= max_turns:
            return history
        cut = user_idx[-max_turns]
        return history[:lead] + history[cut:]

    @staticmethod
    def _strip_old_images(history: list[dict]) -> None:
        """Keep only the latest image to bound ViT token cost (mirrors
        Training's _strip_historical_images)."""
        seen = False
        for msg in reversed(history):
            c = msg.get("content")
            if isinstance(c, list):
                if not seen:
                    seen = True
                    continue
                msg["content"] = " ".join(
                    p.get("text", "") for p in c if p.get("type") == "text"
                )

    def agent_fn(self, render_state: dict, Command: Any) -> list:
        self.stats["turns"] += 1
        user_msg = self._user_message(render_state)
        if self.audit_capture:
            # Plain-text briefing capture (image-primary turns carry a
            # list `content`; the text part is the briefing). FullPlayback
            # writes this as the human-readable `briefing` field so the
            # audit JSONL records exactly what the model read.
            c = user_msg.get("content")
            if isinstance(c, str):
                self.last_briefing = c
            elif isinstance(c, list):
                self.last_briefing = "\n".join(
                    p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            # Enable provider-side audit hook for this turn (drain on
            # return). Lazily install the list — providers without the
            # `request_log` attr (e.g. Bedrock stub) silently skip.
            if hasattr(self.provider, "request_log"):
                self.provider.request_log = []
        self.history.append(user_msg)
        self._strip_old_images(self.history)
        wire = self._window(
            self.history, getattr(self.cfg, "max_history_turns", 16)
        )
        reply = self.provider.complete(wire, self.tools)
        if self.audit_capture and hasattr(self.provider, "request_log"):
            log = self.provider.request_log or []
            # One model call per turn (no internal retries here — those
            # are surfaced as a single call with the eventual response);
            # take the LAST entry to be safe.
            if log:
                self.last_request = log[-1].get("request")
                self.last_response = log[-1].get("response")
            else:
                self.last_request = None
                self.last_response = None
            # Drain so the next turn starts clean.
            self.provider.request_log = []
        self.history.append(
            {
                "role": "assistant",
                "content": reply.text or "",
                # Playback-only: the wire layer (providers._wire_messages)
                # strips this before posting, so it never goes back to
                # the model but is preserved in messages.json.
                "reasoning": reply.reasoning or "",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for i, c in enumerate(reply.tool_calls)
                ],
            }
        )
        cmds = _to_commands(reply.tool_calls, Command, self._label_to_id)
        self.stats["tool_calls"] += len(cmds)
        if not cmds:
            self.stats["empty_replies"] += 1
            cmds = [Command.observe()]
        # Satisfy the OpenAI contract: every tool_call needs a tool result.
        for i in range(len(reply.tool_calls)):
            self.history.append(
                {"role": "tool", "tool_call_id": f"c{i}", "content": "ok"}
            )
        return cmds

    # ── Controller contract ──────────────────────────────────────────
    def act(self, observation: dict, Command: Any) -> list:
        """Controller contract — alias of `agent_fn`. Lets a ModelAgent
        be passed straight to `run_level` / the 1v1 harness in place of
        a bare `agent_fn` callable."""
        return self.agent_fn(observation, Command)

    def reset(self, ctx: Any = None) -> None:
        """Controller contract per-episode hook. A ModelAgent is
        constructed once per episode — its bounded chat history starts
        fresh in `__init__` — so reset is a no-op; it exists so the
        agent structurally satisfies the Controller protocol."""
