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


def _render_minimap_b64(render_state: dict) -> str | None:
    """Best-effort minimap PNG. Returns None (text-only fallback) when
    terrain isn't resolvable — vision degrades gracefully in Phase 0."""
    try:
        from openra_rl_training.training.minimap_renderer import render_minimap

        return render_minimap(
            terrain_png=render_state.get("terrain_png"),  # None in Phase 0 -> None
            map_width=render_state.get("map_width", 64),
            map_height=render_state.get("map_height", 64),
            bounds_x=render_state.get("bounds_x", 0),
            bounds_y=render_state.get("bounds_y", 0),
            own_units=render_state.get("units_summary", []) or [],
            enemy_units=render_state.get("enemy_summary", []) or [],
            ascii_minimap=render_state.get("minimap", ""),
        )
    except Exception as e:  # noqa: BLE001 — vision is optional
        logger.debug("minimap render skipped: %s", e)
        return None


def _to_commands(tool_calls: list[dict], Command: Any) -> list:
    cmds = []
    for call in tool_calls:
        name = _TOOL_ALIASES.get(call.get("name", ""), call.get("name", ""))
        args = call.get("arguments") or {}
        try:
            if name == "move_units":
                ids = [str(i) for i in args["unit_ids"]]
                cmds.append(
                    Command.move_units(ids, int(args["target_x"]), int(args["target_y"]))
                )
            elif name == "attack_unit":
                ids = [str(i) for i in args["unit_ids"]]
                cmds.append(Command.attack_unit(ids, str(args["target_id"])))
            elif name == "observe":
                cmds.append(Command.observe())
            elif name == "surrender":
                cmds.append(Command.surrender())
            elif name == "set_stance":
                ids = [str(i) for i in args["unit_ids"]]
                cmds.append(Command.set_stance(ids, int(args["stance"])))
            elif name == "patrol":
                cmds.append(Command.patrol([str(i) for i in args["unit_ids"]]))
            elif name in ("attack_move", "harvest", "set_rally_point"):
                ids = [str(i) for i in args["unit_ids"]]
                fn = getattr(Command, name)
                cmds.append(fn(ids, int(args["target_x"]), int(args["target_y"])))
            elif name in ("stop", "deploy", "sell", "repair", "power_down"):
                ids = [str(i) for i in args["unit_ids"]]
                cmds.append(getattr(Command, name)(ids))
            elif name in ("build", "cancel_production"):
                cmds.append(getattr(Command, name)(str(args["item"])))
            elif name == "place_building":
                cmds.append(
                    Command.place_building(
                        str(args["item"]), int(args["target_x"]), int(args["target_y"])
                    )
                )
        except (KeyError, TypeError, ValueError) as e:
            logger.debug("dropping malformed tool call %s: %s", call, e)
    return cmds


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
    ):
        self.cfg = cfg
        self.objective = objective
        self.tools = _tool_schemas(allowed_tools)
        self.provider = provider or make_provider(cfg)
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.stats = {"turns": 0, "tool_calls": 0, "empty_replies": 0}

    def _user_message(self, render_state: dict) -> dict:
        text = build_briefing(render_state, self.objective)
        if self.cfg.vision:
            b64 = _render_minimap_b64(render_state)
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
        self.history.append(self._user_message(render_state))
        self._strip_old_images(self.history)
        reply = self.provider.complete(self.history, self.tools)
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
        cmds = _to_commands(reply.tool_calls, Command)
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
