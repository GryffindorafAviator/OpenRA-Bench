"""Phase 2 — the human-labeling machine.

Lets a human play the exact scenarios the LLM plays, for a direct
human-vs-LLM comparison on one leaderboard. Design constraints
(user-confirmed):

* the human gets the **same observation the LLM gets** (the rendered
  minimap + the text briefing) — the UI just displays what
  `render_state` already carries;
* the cadence is **turn-based, matching the LLM** — one `act()` call is
  one human decision turn;
* the human interacts by **clicking the minimap**, not typing world
  coordinates — the UI runs `minimap_click_to_cell` to recover the
  world cell, then assembles `HumanAction`s.

Everything here is headless-testable; a browser/terminal UI is a thin
shell over these pieces:

1. `minimap_click_to_cell` / `cell_to_minimap_rect` — the pure
   pixel⇄cell transforms (renderer-agnostic: parameterised by the
   rendered image size).
2. `own_units_at_cell` / `enemy_at_cell` — click-to-selection and
   click-to-target resolution against `render_state`.
3. `HumanAction` + `human_actions_to_commands` — a human's per-turn
   point-and-click gestures, translated into engine `Command`s by
   delegating to the **same** `agent._to_commands` the LLM path uses,
   so human and model emit byte-identical commands.
4. `HumanController` — a Controller (Phase 1 contract) whose `act()`
   pulls the turn's `HumanAction`s from an injected input source and
   records a playback-compatible transcript. The episode then runs
   through `run_level` identically to a model run, so the trace, the
   playback, and the leaderboard entry are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .controller import BaseController, EpisodeContext

# An input source yields the human's gestures for one decision turn.
# Either a callable `(render_state) -> [HumanAction]` (the real UI: a
# blocking browser/terminal round-trip) or a pre-scripted list of turns
# (tests, or replay of a recorded session).
TurnActions = "list[HumanAction]"
InputSource = "Callable[[dict], list[HumanAction]] | Sequence[list[HumanAction]]"


# ── Pixel ⇄ cell transforms ─────────────────────────────────────────


def minimap_click_to_cell(
    px: float,
    py: float,
    img_w: int,
    img_h: int,
    map_cols: int,
    map_rows: int,
) -> tuple[int, int]:
    """Map a click at pixel ``(px, py)`` on a rendered minimap of size
    ``img_w × img_h`` back to a world cell on a ``map_cols × map_rows``
    grid.

    Renderer-agnostic: pass the *actual* rendered image dimensions, so
    this works for either minimap renderer (`minimap.render_png_b64`
    at CELL=6, or the vendored `prompt_v2.minimap_b64`). The result is
    clamped to the playable grid — an out-of-bounds click resolves to
    the nearest edge cell rather than raising."""
    if img_w <= 0 or img_h <= 0 or map_cols <= 0 or map_rows <= 0:
        raise ValueError("image and map dimensions must be positive")
    cx = int(px * map_cols / img_w)
    cy = int(py * map_rows / img_h)
    cx = min(map_cols - 1, max(0, cx))
    cy = min(map_rows - 1, max(0, cy))
    return cx, cy


def cell_to_minimap_rect(
    cx: int,
    cy: int,
    img_w: int,
    img_h: int,
    map_cols: int,
    map_rows: int,
) -> tuple[int, int, int, int]:
    """Inverse of `minimap_click_to_cell`: the pixel rectangle
    ``(left, top, width, height)`` a world cell occupies on the
    rendered minimap. The UI uses it to draw selection highlights."""
    if img_w <= 0 or img_h <= 0 or map_cols <= 0 or map_rows <= 0:
        raise ValueError("image and map dimensions must be positive")
    left = int(cx * img_w / map_cols)
    top = int(cy * img_h / map_rows)
    right = int((cx + 1) * img_w / map_cols)
    bottom = int((cy + 1) * img_h / map_rows)
    return left, top, max(1, right - left), max(1, bottom - top)


# ── Click → selection / target resolution ───────────────────────────


def _cells_near(cx: int, cy: int, radius: int):
    for yy in range(cy - radius, cy + radius + 1):
        for xx in range(cx - radius, cx + radius + 1):
            yield xx, yy


def own_units_at_cell(
    render_state: dict, cx: int, cy: int, radius: int = 1
) -> list[str]:
    """Ids of the agent's own units within `radius` cells of (cx, cy) —
    click-to-select. Buildings are excluded (a click on a building is a
    building selection; see `own_buildings_at_cell`)."""
    near = set(_cells_near(cx, cy, radius))
    out: list[str] = []
    for u in render_state.get("units_summary", []) or []:
        if not isinstance(u, dict):
            continue
        if (int(u.get("cell_x", -99)), int(u.get("cell_y", -99))) in near:
            uid = u.get("id")
            if uid is not None:
                out.append(str(uid))
    return out


def own_buildings_at_cell(
    render_state: dict, cx: int, cy: int, radius: int = 1
) -> list[str]:
    """Ids of the agent's own buildings within `radius` cells of
    (cx, cy) — click-to-select for repair / sell / power_down."""
    near = set(_cells_near(cx, cy, radius))
    out: list[str] = []
    for b in render_state.get("own_buildings", []) or []:
        if not isinstance(b, dict):
            continue
        if (int(b.get("cell_x", -99)), int(b.get("cell_y", -99))) in near:
            bid = b.get("id")
            if bid is not None:
                out.append(str(bid))
    return out


def enemy_at_cell(
    render_state: dict, cx: int, cy: int, radius: int = 1
) -> str | None:
    """Id of the nearest visible enemy actor within `radius` cells of
    (cx, cy), or None — click-to-target for `attack`. Picks the closest
    so a click between two enemies is unambiguous."""
    best: tuple[int, str] | None = None
    for e in render_state.get("enemy_summary", []) or []:
        if not isinstance(e, dict):
            continue
        ex, ey = int(e.get("cell_x", -99)), int(e.get("cell_y", -99))
        d2 = (ex - cx) ** 2 + (ey - cy) ** 2
        if d2 <= radius * radius * 2:
            eid = e.get("id")
            if eid is not None and (best is None or d2 < best[0]):
                best = (d2, str(eid))
    return best[1] if best else None


# ── Human action → engine Command ───────────────────────────────────

# The gesture vocabulary. Each gesture maps onto exactly one LLM tool
# name so the human path reuses `agent._to_commands` verbatim. The
# `move` gesture is the only rename — the engine tool is `move_units`.
_TARGETED_MOVE = {
    "move": "move_units",
    "attack_move": "attack_move",
    "harvest": "harvest",
    "set_rally_point": "set_rally_point",
}
_UNIT_ONLY = {
    "stop", "deploy", "sell", "repair", "power_down",
    "set_primary", "unload", "patrol",
}


@dataclass
class HumanAction:
    """One point-and-click gesture from the human, already resolved to
    world-cell coordinates (the UI ran `minimap_click_to_cell` first).

    `mode` is the gesture; the other fields are filled per gesture:

    * ``move`` / ``attack_move`` / ``harvest`` / ``set_rally_point`` —
      `units` + `target` cell.
    * ``attack`` — `units` + either `target_id` (an enemy actor) or a
      `target` cell (falls back to `attack_move`).
    * ``guard`` — `units` + `target_id` (ally to escort).
    * ``stop`` / ``deploy`` / ``sell`` / ``repair`` / ``power_down`` /
      ``set_primary`` / ``unload`` / ``patrol`` — `units` only.
    * ``set_stance`` — `units` + `stance` (0–3).
    * ``build`` / ``cancel_production`` — `unit_type`.
    * ``place_building`` — `unit_type` + `target` cell.
    * ``observe`` / ``surrender`` — no payload (pass-turn / concede).
    """

    mode: str
    units: list[str] = field(default_factory=list)
    target: tuple[int, int] | None = None
    target_id: str | None = None
    unit_type: str = ""
    stance: int = 0
    note: str = ""  # optional free-text rationale (playback only)
    raw: dict = field(default_factory=dict)  # original click payload

    def to_tool_call(self) -> dict | None:
        """Normalise into the ``{name, arguments}`` tool-call dict that
        `agent._to_commands` consumes. Returns None for a gesture that
        cannot form a valid command (dropped silently, like a malformed
        LLM tool call)."""
        m = self.mode
        if m == "observe":
            return {"name": "observe", "arguments": {}}
        if m == "surrender":
            return {"name": "surrender", "arguments": {}}
        if m == "attack":
            if self.target_id is not None:
                return {
                    "name": "attack_unit",
                    "arguments": {
                        "unit_ids": list(self.units),
                        "target_id": str(self.target_id),
                    },
                }
            # No enemy under the click → close on the cell instead.
            if self.target is not None and self.units:
                return {
                    "name": "attack_move",
                    "arguments": {
                        "unit_ids": list(self.units),
                        "target_x": int(self.target[0]),
                        "target_y": int(self.target[1]),
                    },
                }
            return None
        if m == "guard":
            if not self.units or self.target_id is None:
                return None
            return {
                "name": "guard",
                "arguments": {
                    "unit_ids": list(self.units),
                    "target_id": str(self.target_id),
                },
            }
        if m in _TARGETED_MOVE:
            if not self.units or self.target is None:
                return None
            return {
                "name": _TARGETED_MOVE[m],
                "arguments": {
                    "unit_ids": list(self.units),
                    "target_x": int(self.target[0]),
                    "target_y": int(self.target[1]),
                },
            }
        if m == "set_stance":
            if not self.units:
                return None
            return {
                "name": "set_stance",
                "arguments": {
                    "unit_ids": list(self.units),
                    "stance": int(self.stance),
                },
            }
        if m in _UNIT_ONLY:
            if not self.units:
                return None
            return {"name": m, "arguments": {"unit_ids": list(self.units)}}
        if m in ("build", "cancel_production"):
            if not self.unit_type:
                return None
            return {"name": m, "arguments": {"item": str(self.unit_type)}}
        if m == "place_building":
            if not self.unit_type or self.target is None:
                return None
            return {
                "name": "place_building",
                "arguments": {
                    "item": str(self.unit_type),
                    "target_x": int(self.target[0]),
                    "target_y": int(self.target[1]),
                },
            }
        return None  # unknown gesture — dropped

    def describe(self) -> str:
        """One-line human-readable summary for the playback transcript."""
        if self.mode in ("observe", "surrender"):
            return self.mode
        bits = [self.mode]
        if self.units:
            bits.append(f"units={','.join(self.units)}")
        if self.target_id is not None:
            bits.append(f"target={self.target_id}")
        if self.target is not None:
            bits.append(f"@{self.target[0]},{self.target[1]}")
        if self.unit_type:
            bits.append(f"item={self.unit_type}")
        if self.mode == "set_stance":
            bits.append(f"stance={self.stance}")
        return " ".join(bits)


def human_actions_to_commands(
    actions: Sequence[HumanAction], Command: Any
) -> list:
    """Translate a turn's `HumanAction`s into engine `Command`s by
    delegating to `agent._to_commands` — the exact translator the LLM
    path uses — so a human and a model emit byte-identical commands."""
    from .agent import _to_commands

    calls = [tc for a in actions if (tc := a.to_tool_call()) is not None]
    return _to_commands(calls, Command)


# ── Input sources ───────────────────────────────────────────────────


class ScriptedInputSource:
    """A deterministic, pre-recorded input source — one entry per turn.

    Drives `HumanController` headlessly in tests, and replays a saved
    human session. When the script is exhausted every further turn
    yields a single `observe` (pass-turn)."""

    def __init__(self, turns: Sequence[Sequence[HumanAction]]):
        self._turns = [list(t) for t in turns]
        self._i = 0

    def __call__(self, render_state: dict) -> list[HumanAction]:
        if self._i >= len(self._turns):
            return [HumanAction(mode="observe")]
        turn = self._turns[self._i]
        self._i += 1
        return list(turn)


# ── The Controller ──────────────────────────────────────────────────


class HumanController(BaseController):
    """A Controller (Phase 1 contract) driven by a human.

    `act()` pulls the turn's `HumanAction`s from `input_source` and
    translates them into engine `Command`s. It also records a
    playback-compatible transcript into `self.history` (the same
    `{role, content, tool_calls}` shape `ModelAgent` writes), so a human
    run renders in the existing battle viewer beside model runs and the
    leaderboard entry is directly comparable.

    `input_source` is either a callable ``(render_state) -> [HumanAction]``
    — the real UI's blocking browser/terminal round-trip — or a
    sequence of per-turn action lists (a `ScriptedInputSource`, a plain
    list, or a saved replay)."""

    def __init__(
        self,
        input_source: "InputSource",
        name: str = "human",
    ):
        super().__init__(name=name)
        if callable(input_source):
            self._source = input_source
        else:
            self._source = ScriptedInputSource(input_source)
        self.stats = {"turns": 0, "tool_calls": 0, "empty_replies": 0}
        self._ctx: EpisodeContext | None = None

    def reset(self, ctx: EpisodeContext) -> None:
        """Per-episode hook: stamp context and seed the transcript with
        a system message naming the scenario and objective."""
        self._ctx = ctx
        self.history = [
            {
                "role": "system",
                "content": (
                    f"Human-labeling session — scenario "
                    f"{ctx.pack_id}:{ctx.level} (seed {ctx.seed}). "
                    f"Objective: {ctx.objective or '(see briefing)'}"
                ),
            }
        ]
        self.stats = {"turns": 0, "tool_calls": 0, "empty_replies": 0}

    @staticmethod
    def _briefing(render_state: dict) -> str:
        """The SAME text briefing the LLM is shown (vendored
        `prompt_v2.briefing`, with a defensive fallback)."""
        try:
            from .prompt_v2 import briefing as _v2_brief

            return _v2_brief(render_state)
        except Exception:  # noqa: BLE001 — never break a turn
            return (
                f"tick={render_state.get('game_tick', 0)} "
                f"explored={render_state.get('explored_percent', 0.0):.1f}%"
            )

    def act(self, observation: dict, Command: Any) -> list:
        self.stats["turns"] += 1
        # The human sees exactly the LLM's observation.
        self.history.append(
            {"role": "user", "content": self._briefing(observation)}
        )
        actions = list(self._source(observation) or [])
        calls = [
            tc for a in actions if (tc := a.to_tool_call()) is not None
        ]
        cmds = human_actions_to_commands(actions, Command)
        self.stats["tool_calls"] += len(cmds)
        if not cmds:
            self.stats["empty_replies"] += 1
            cmds = [Command.observe()]
        # Playback-compatible assistant turn: a human-readable summary
        # plus the structured tool_calls, mirroring ModelAgent.agent_fn.
        notes = "; ".join(a.note for a in actions if a.note)
        self.history.append(
            {
                "role": "assistant",
                "content": "; ".join(a.describe() for a in actions)
                or "observe",
                "reasoning": notes,
                "tool_calls": [
                    {
                        "id": f"h{i}",
                        "type": "function",
                        "function": {
                            "name": c["name"],
                            "arguments": c["arguments"],
                        },
                    }
                    for i, c in enumerate(calls)
                ],
            }
        )
        for i in range(len(calls)):
            self.history.append(
                {"role": "tool", "tool_call_id": f"h{i}", "content": "ok"}
            )
        return cmds


# ── Session harness ─────────────────────────────────────────────────


def run_human_session(
    pack_id: str,
    level: str = "easy",
    seed: int = 1,
    input_source: "InputSource" = (),
    playback: Any = None,
    name: str = "human",
):
    """Run a full human-labeling session and return the scored
    `EpisodeResult`.

    Compiles the named pack/level and drives a `HumanController` through
    `run_level` — the IDENTICAL scoring path a model run takes — so the
    result, the per-turn playback, and the leaderboard entry are
    directly comparable to an LLM's run on the same scenario.

    `input_source` is a callable ``(render_state) -> [HumanAction]`` (the
    real UI's blocking browser/terminal round-trip) or a pre-recorded
    sequence of per-turn `HumanAction` lists (a replay)."""
    from .eval_core import run_level
    from .scenarios import load_pack
    from .scenarios.loader import PACKS_DIR, compile_level

    compiled = compile_level(
        load_pack(PACKS_DIR / f"{pack_id}.yaml"), level
    )
    controller = HumanController(input_source, name=name)
    return run_level(compiled, controller, seed=seed, playback=playback)
