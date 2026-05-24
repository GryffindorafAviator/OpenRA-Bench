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


# ── Scenario hints for human play ───────────────────────────────────

_REGION_PREDICATES = {
    "reach_region",
    "units_in_region_gte",
    "units_of_type_in_region_gte",
    "all_units_in_region",
    "building_in_region",
    "enemy_key_buildings_destroyed_in_region",
}


def _actor_field(actor: Any, name: str, default: Any = None) -> Any:
    if isinstance(actor, dict):
        return actor.get(name, default)
    return getattr(actor, name, default)


def _scenario_harvest_points(compiled: Any) -> list[dict]:
    """Authored neutral ore-source markers for human minimap overlays.

    Rust turns `mine` actors into harvestable ore patches, but the normal
    observation does not surface those neutral props. Mirror the engine's
    spawn_point choice, then filter through the current fog before display
    so Play mode does not reveal unexplored ore.
    """
    actors = list(getattr(compiled.scenario, "actors", []) or [])
    mine_actors = [
        actor for actor in actors
        if str(
            _actor_field(actor, "type")
            or _actor_field(actor, "actor_type")
            or ""
        ).lower() in {"mine", "gmine"}
    ]
    out: list[dict] = []
    for actor in mine_actors:
        pos = _actor_field(actor, "position")
        if not pos or len(pos) < 2:
            continue
        out.append({
            "cell_x": int(pos[0]),
            "cell_y": int(pos[1]),
            "type": str(
                _actor_field(actor, "type")
                or _actor_field(actor, "actor_type")
                or "mine"
            ),
            "spawn_point": _actor_field(actor, "spawn_point"),
        })
    return out


def _active_spawn_point(compiled: Any, render_state: dict) -> Any:
    """Infer the engine-selected spawn_point from observed own actors."""
    observed = set()
    for item in (
        list(render_state.get("units_summary") or [])
        + list(render_state.get("own_buildings") or [])
    ):
        if not isinstance(item, dict):
            continue
        try:
            observed.add((int(item.get("cell_x")), int(item.get("cell_y"))))
        except (TypeError, ValueError):
            continue

    counts: dict[Any, int] = {}
    for actor in getattr(compiled.scenario, "actors", []) or []:
        if str(_actor_field(actor, "owner") or "").lower() != "agent":
            continue
        sp = _actor_field(actor, "spawn_point")
        if sp is None:
            continue
        pos = _actor_field(actor, "position")
        if pos and len(pos) >= 2 and (int(pos[0]), int(pos[1])) in observed:
            counts[sp] = counts.get(sp, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _visible_harvest_points(
    points: list[dict], render_state: dict, active_spawn_point: Any = None
) -> list[dict]:
    rows = [r for r in (render_state.get("minimap") or "").split("\n") if r]
    if not rows:
        return []
    visible: list[dict] = []
    for point in points:
        sp = point.get("spawn_point")
        if active_spawn_point is not None and sp != active_spawn_point:
            continue
        try:
            cx = int(point["cell_x"])
            cy = int(point["cell_y"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            0 <= cy < len(rows)
            and 0 <= cx < len(rows[cy])
            and rows[cy][cx] != "#"
        ):
            visible.append(point)
    return visible


def _condition_node(cond: Any) -> dict:
    if cond is None:
        return {}
    if isinstance(cond, dict):
        return cond
    return dict(getattr(cond, "__pydantic_extra__", None) or {})


def _objective_regions_from_condition(cond: Any) -> list[dict]:
    """Extract authored objective regions from a win-condition tree."""
    out: list[dict] = []

    def walk(node: Any) -> None:
        data = _condition_node(node)
        if not data:
            return
        for child in data.get("all_of") or []:
            walk(child)
        for child in data.get("any_of") or []:
            walk(child)
        if "not" in data:
            walk(data["not"])
        then = data.get("then")
        if isinstance(then, dict):
            for child in then.get("clauses") or []:
                walk(child)
        seq = data.get("waypoint_sequence")
        if isinstance(seq, dict):
            default_r = seq.get("radius", 6)
            for i, point in enumerate(seq.get("points") or [], start=1):
                if not isinstance(point, dict):
                    continue
                if "x" in point and "y" in point:
                    out.append({
                        "x": int(point["x"]),
                        "y": int(point["y"]),
                        "radius": float(point.get("radius", default_r)),
                        "label": str(point.get("label") or f"W{i}"),
                    })
        for key in _REGION_PREDICATES:
            v = data.get(key)
            if not isinstance(v, dict) or "x" not in v or "y" not in v:
                continue
            label = v.get("label")
            if not label:
                if key == "units_in_region_gte":
                    label = f">={int(v.get('n', 1))} units"
                elif key == "units_of_type_in_region_gte":
                    label = f">={int(v.get('n', 1))} {v.get('type')}"
                elif key == "building_in_region":
                    label = str(v.get("type") or "building")
                else:
                    label = key.replace("_", " ")
            out.append({
                "x": int(v["x"]),
                "y": int(v["y"]),
                "radius": float(v.get("radius", 3)),
                "label": str(label),
            })

    walk(cond)
    seen = set()
    unique = []
    for r in out:
        key = (r["x"], r["y"], r["radius"], r["label"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique


def _initial_type_by_id(scenario: Any, render_state: dict) -> dict[str, str]:
    """Infer own unit types from authored initial placements.

    Rust currently omits own-unit actor types in some observations, but
    it preserves deterministic actor ids and initial cells. This maps the
    initial visible units back to the scenario placements so the human UI
    can distinguish 1tnk/2tnk/etc. after the units move.
    """
    expected: dict[tuple[int, int], list[str]] = {}
    for actor in getattr(scenario, "actors", []) or []:
        if str(getattr(actor, "owner", "")).lower() != "agent":
            continue
        pos = getattr(actor, "position", None)
        if not pos or len(pos) < 2:
            continue
        cell = (int(pos[0]), int(pos[1]))
        atype = str(getattr(actor, "type", "") or "").lower()
        count = int(getattr(actor, "count", 1) or 1)
        expected.setdefault(cell, []).extend([atype] * count)

    observed: dict[tuple[int, int], list[str]] = {}
    for unit in render_state.get("units_summary") or []:
        if not isinstance(unit, dict) or unit.get("id") is None:
            continue
        cell = (int(unit.get("cell_x", 0)), int(unit.get("cell_y", 0)))
        observed.setdefault(cell, []).append(str(unit["id"]))

    out: dict[str, str] = {}
    for cell, types in expected.items():
        ids = sorted(observed.get(cell, []))
        for uid, atype in zip(ids, types):
            if atype:
                out[uid] = atype
    return out


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


# ── Interactive (GUI-driven) session ────────────────────────────────


class InteractiveSession:
    """A turn-steppable scenario session for GUI-driven human play.

    `run_level` / `run_human_session` run the whole episode in one
    synchronous call — fine when the policy is a function, impossible
    when the policy is a human at a browser. `InteractiveSession`
    inverts the loop: the UI holds control and calls `submit_turn()`
    once per decision turn, exactly mirroring `run_level`'s per-turn
    body (command translation, forbidden-tool accounting, win/fail
    evaluation), so a human's run is scored by the identical rules as
    an LLM's.

    Lifecycle:

        s = InteractiveSession.from_pack("defense-rush-survive", "easy")
        s.render_state()                       # what the human sees
        s.submit_turn([HumanAction(...), ...]) # advance one turn
        ...                                    # repeat until s.done
        s.close()
    """

    def __init__(
        self,
        compiled: Any,
        seed: int = 1,
        record: bool = True,
        playback_root: Any = None,
        player: str = "Human",
    ):
        from openra_rl_training.training.rust_env_pool import RustEnvPool

        from .eval_core import _scenario_to_tmp_yaml
        from .rust_adapter import RustObsAdapter

        if not compiled.map_supported:
            raise RuntimeError(
                f"{compiled.pack_id}: base map not Rust-loadable"
            )
        self.compiled = compiled
        self.seed = seed
        self._tmp_path = _scenario_to_tmp_yaml(compiled)
        self._pool = RustEnvPool(size=1, scenario_path=self._tmp_path)
        self._env = self._pool.acquire()
        self._adapter = RustObsAdapter()
        self._adapter.observe(self._env.reset(seed=seed))
        self._adapter.type_by_id.update(
            _initial_type_by_id(compiled.scenario, self._adapter.render_state())
        )
        self._objective_regions = (
            _objective_regions_from_condition(compiled.win_condition)
            if compiled.objective_coords == "exact" else []
        )
        self._harvest_points = _scenario_harvest_points(compiled)
        self._forbidden = {
            str(t).lower() for t in (compiled.forbidden_tools or [])
        }
        self.turn = 0
        self.outcome = "draw"
        self.done = False
        self._closed = False
        self.save_path: str | None = None
        # Standard-playback recording: a human run produces the IDENTICAL
        # artifact a model run does — a Playback dir (turns.jsonl,
        # per-turn minimap PNGs, messages.json, manifest.json) under the
        # same PLAYBACK_ROOT — so the Battle Viewer and leaderboard treat
        # human and LLM runs apples-to-apples (no separate human format).
        self._player = player
        self._playback = None
        self._finalized = False
        self._history: list[dict] = []
        self._issued = 0
        self._pb_terrain = None
        self._pb_explored: set = set()
        if record:
            self._setup_playback(playback_root)

    # A live engine session is a resource handle, not a value. Gradio's
    # `gr.State` deep-copies whatever it holds — and the RustEnvPool
    # carries an unpicklable `_thread.lock`, so a naive deepcopy raises
    # and the Play tab's Start handler silently fails. Copying a session
    # is meaningless anyway; return the same instance.
    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    @classmethod
    def from_pack(
        cls,
        pack_id: str,
        level: str = "easy",
        seed: int = 1,
        record: bool = True,
        playback_root: Any = None,
        player: str = "Human",
        fog_mode: str = "vision",
    ) -> "InteractiveSession":
        """Compile a pack by id and open a session on it. `fog_mode`
        selects the observation modality — `vision-clear` (or any
        `-clear` mode) reveals the whole map (engine `reveal_map`), the
        no-fog condition of the human-study fog-penalty pair."""
        from .scenarios import load_pack
        from .scenarios.loader import PACKS_DIR, compile_level

        compiled = compile_level(
            load_pack(PACKS_DIR / f"{pack_id}.yaml"), level
        )
        compiled.fog_mode = fog_mode
        return cls(
            compiled, seed=seed, record=record,
            playback_root=playback_root, player=player,
        )

    @property
    def Command(self) -> Any:
        """The engine Command factory."""
        return self._env.Command

    @property
    def max_turns(self) -> int:
        return self.compiled.max_turns

    @property
    def objective(self) -> str:
        """The scenario objective brief — the SAME text an LLM agent is
        given as its goal. What the human must accomplish to WIN."""
        return (self.compiled.scenario.description or "").strip()

    def render_state(self) -> dict:
        """The current observation — the SAME render_state an LLM agent
        is shown for this scenario."""
        rs = self._adapter.render_state()
        if self._objective_regions:
            rs["objective_regions"] = list(self._objective_regions)
        if self._harvest_points:
            visible_points = _visible_harvest_points(
                self._harvest_points, rs, _active_spawn_point(self.compiled, rs)
            )
            if visible_points:
                rs["harvest_points"] = visible_points
        return rs

    def status(self) -> dict:
        """Turn / outcome / done summary for the UI."""
        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "outcome": self.outcome,
            "done": self.done,
            "tick": self._adapter.signals.game_tick,
            "save_path": self.save_path,
        }

    # ── Standard-playback recording ──────────────────────────────────
    def _setup_playback(self, playback_root: Any) -> None:
        """Open a Playback dir — the same one model runs use."""
        try:
            import os
            from datetime import datetime, timezone
            from pathlib import Path

            from .playback import Playback

            root = Path(
                playback_root
                or os.environ.get(
                    "OPENRA_BENCH_PLAYBACK_ROOT",
                    Path(__file__).resolve().parents[1] / "playback",
                )
            )
            run_id = "human-" + datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            safe = "".join(
                c if (c.isalnum() or c in "-_") else "-"
                for c in self._player
            )
            pb = Playback(
                root / f"{run_id}__{safe}",
                cell=f"{self.compiled.pack_id}:{self.compiled.level}",
                seed=self.seed,
            )
            pb.run_id, pb.model = run_id, self._player
            self._playback = pb
            self._history = [{
                "role": "system",
                "content": (
                    f"Human-labeling session — {self.compiled.pack_id}:"
                    f"{self.compiled.level} seed {self.seed}. Objective: "
                    f"{self.objective or '(see briefing)'}"
                ),
            }]
        except Exception:  # noqa: BLE001 — recording never breaks play
            self._playback = None

    def _pb_frame(self, rs: dict) -> "str | None":
        """Per-turn minimap PNG (b64) via the SAME renderer the Play tab
        shows, so saved human frames match the clicked UI frame."""
        try:
            import base64
            import io

            from .minimap import render_tactical_minimap

            img = render_tactical_minimap(
                rs, scale=5, grid=True, legend=True,
            )
            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001
            pass

        try:
            from .minimap import terrain_png_for

            if self._pb_terrain is None:
                self._pb_terrain = terrain_png_for(
                    self.compiled.scenario.base_map
                )
            try:
                from .prompt_v2 import minimap_b64 as _v2_mm

                return _v2_mm(
                    rs, self._pb_terrain, self._pb_explored,
                    constant_colors=self.compiled.level in ("easy", "medium"),
                )
            except Exception:  # noqa: BLE001
                pass
            from .minimap import render_b64

            return render_b64(rs, self._pb_terrain)
        except Exception:  # noqa: BLE001
            return None

    def _record_turn_playback(self, actions, calls, cmds, ctx) -> None:
        """Append one turn to the Playback + the message transcript."""
        if self._playback is None:
            return
        from .goal_tracker import turn_goal

        rs = self.render_state()
        self._issued += len(cmds)
        self._playback.record_turn(
            self.turn, rs, cmds, self._adapter.signals,
            self._pb_frame(rs),
            goal=turn_goal(self.compiled.win_condition, ctx),
        )
        # Transcript turn — same {role, content, tool_calls} shape a
        # ModelAgent writes, so messages.json is uniform across human
        # and LLM runs.
        self._history.append(
            {"role": "user", "content": HumanController._briefing(rs)}
        )
        self._history.append({
            "role": "assistant",
            "content": "; ".join(a.describe() for a in actions) or "observe",
            "reasoning": "; ".join(a.note for a in actions if a.note),
            "tool_calls": [
                {
                    "id": f"h{i}", "type": "function",
                    "function": {
                        "name": c["name"], "arguments": c["arguments"],
                    },
                }
                for i, c in enumerate(calls)
            ],
        })
        for i in range(len(calls)):
            self._history.append(
                {"role": "tool", "tool_call_id": f"h{i}", "content": "ok"}
            )

    def _finalize_playback(self, ctx) -> None:
        """Write messages.json + manifest.json — the same manifest shape
        `run_level` writes for a model run, tagged as a human run."""
        if self._playback is None or self._finalized:
            return
        self._finalized = True
        from .goal_tracker import turn_goal

        final_goal = turn_goal(self.compiled.win_condition, ctx)
        sig = self._adapter.signals
        try:
            self._playback.write_messages(self._history)
            self._playback.finalize({
                "scenario": f"{self.compiled.pack_id}:{self.compiled.level}",
                "pack_id": self.compiled.pack_id,
                "level": self.compiled.level,
                "capability": self.compiled.meta.capability,
                "run_id": self._playback.run_id,
                "model": self._playback.model,
                "agent_type": "Human",
                "seed": self.seed,
                "outcome": self.outcome,
                "turns": self.turn,
                "max_turns": self.max_turns,
                "actions_issued": self._issued,
                "actions_warned": 0,
                "agent_stats": {"turns": self.turn},
                "objective_progress": final_goal.get(
                    "objective_progress", 0.0
                ),
                "reward_vector": final_goal.get("reward_vector", {}),
                "signals": {
                    "economy_value": sig.cash + sig.resources,
                    "explored_percent": round(sig.explored_percent, 2),
                    "units_killed": sig.units_killed,
                    "units_lost": sig.units_lost,
                },
            })
            self.save_path = str(self._playback.dir)
        except Exception:  # noqa: BLE001
            pass

    def submit_turn(self, actions: "list[HumanAction]") -> dict:
        """Advance one decision turn with the human's gestures. Mirrors
        `run_level`'s per-turn body. Returns the updated `status()`."""
        if self.done:
            return self.status()
        from .scenarios.win_conditions import WinContext, evaluate

        Command = self._env.Command
        calls = [
            tc for a in (actions or []) if (tc := a.to_tool_call())
            is not None
        ]
        cmds = human_actions_to_commands(actions or [], Command)
        # Forbidden-tool accounting — identical rule to run_level.
        for tc in calls:
            tn = str(tc.get("name", "")).lower()
            self._adapter.signals.tools_called[tn] = (
                self._adapter.signals.tools_called.get(tn, 0) + 1
            )
            if tn in self._forbidden:
                self._adapter.signals.tool_violations += 1
        conceded = any(a.mode == "surrender" for a in (actions or []))
        if not cmds:
            cmds = [Command.observe()]

        obs, _r, engine_done, _info = self._env.step(cmds)
        self._adapter.observe(obs, done=engine_done)
        self.turn += 1

        ctx = WinContext(
            signals=self._adapter.signals,
            render_state=self._adapter.render_state(),
        )
        if evaluate(self.compiled.win_condition, ctx):
            self.outcome = "win"
        elif evaluate(self.compiled.fail_condition, ctx):
            self.outcome = "loss"
        if conceded:
            self.outcome = "loss"
        if (
            self.outcome != "draw"
            or engine_done
            or self.turn >= self.max_turns
        ):
            self.done = True
        # Standard-playback recording — log this turn, finalize on done.
        self._record_turn_playback(actions or [], calls, cmds, ctx)
        if self.done:
            self._finalize_playback(ctx)
        return self.status()

    def close(self) -> None:
        """Release the engine env. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # An abandoned run (closed before game-over) is still finalized
        # so its Playback dir carries a manifest and stays viewable.
        if (
            self._playback is not None
            and not self._finalized
            and self.turn > 0
        ):
            try:
                from .scenarios.win_conditions import WinContext

                self._finalize_playback(
                    WinContext(
                        signals=self._adapter.signals,
                        render_state=self._adapter.render_state(),
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        elif self._playback is not None and self.turn == 0:
            # Never-played recorded session — drop the empty dir rather
            # than littering the playback root.
            import shutil

            try:
                self._playback._turns_fh.close()
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(self._playback.dir, ignore_errors=True)
        try:
            self._pool.release(self._env)
            self._pool.shutdown()
        finally:
            from pathlib import Path

            Path(self._tmp_path).unlink(missing_ok=True)


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
