"""Rust env -> Training-component schema adapter.

The Rust env (`openra_train.OpenRAEnv`) emits a lean observation:

    keys = unit_positions, unit_hp, enemy_positions, enemy_hp,
           enemy_buildings_summary, explored_cells, explored_percent,
           game_tick, units_killed
    step() -> (obs, reward=0.0 (hardcoded), done: bool,
               info={game_tick, warnings})

`minimap_renderer.render_minimap()` and the prompt builders in
OpenRA-RL-Training expect a different shape (`units_summary`,
`enemy_summary`, an ASCII `minimap`, `terrain_png`). And because the
Rust env hardcodes reward to 0.0, all scoring/diagnostic signals must be
derived here from observation deltas.

This module is the single translation point. It is intentionally pure
(no model / network / file I/O beyond optional terrain load) so it can
be unit-tested against captured Rust observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _cells(obj: Any) -> list[tuple[int, int]]:
    """Normalize explored_cells / position lists to [(x, y), ...]."""
    out: list[tuple[int, int]] = []
    if not obj:
        return out
    for c in obj:
        if isinstance(c, dict):
            out.append((int(c.get("cell_x", 0)), int(c.get("cell_y", 0))))
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append((int(c[0]), int(c[1])))
    return out


# ── Shroud 3-state: engine-truth + Chebyshev fallback ────────────────
# The Rust engine maintains the proper RA 3-state shroud per player
# (`world.rs::shroud`: 0=unexplored, 1=fogged, 2=visible). Two PyO3
# surfaces:
#
#   * `explored_cells` (cumulative union of fogged+visible) — always
#     present.
#   * `visible_cells`  (per-tick visible-only, from
#     `typed_shroud[agent].visible`) — present on the engine HEAD with
#     the "expose typed-shroud visible_cells via PyO3" PR; absent on
#     older HEADs.
#
# `_compute_visible_cells` prefers the engine `visible_cells` when
# present (exact truth, no approximation) and falls back to the
# Chebyshev approximation below otherwise. The fallback unions cells
# within each currently-alive agent unit/building's sight radius (the
# same shape the engine itself uses for shroud reveal), seeded by the
# per-turn explored-delta as a safety net for actor types missing
# from `_SIGHT_BY_TYPE`. Sight ranges are the RA vendor defaults
# (cross-checked via `env.unit_codex`). An actor type not listed
# falls back to `_DEFAULT_SIGHT`; this only affects edge cases for
# the fallback path.
_SIGHT_BY_TYPE: dict[str, int] = {
    # Infantry (sight 4)
    "e1": 4, "e2": 4, "e3": 4, "e4": 4, "e6": 4, "e7": 4,
    "medi": 4, "mech": 4, "spy": 4, "thf": 4, "dog": 4,
    "engineer": 4, "tanya": 6,
    # Vehicles
    "harv": 4, "mcv": 4,
    "1tnk": 5, "2tnk": 6, "3tnk": 6, "4tnk": 5,
    "jeep": 7, "apc": 6, "arty": 5, "v2rl": 5, "ftrk": 5,
    "mnly": 4, "mgg": 5,
    # Aircraft (longer sight)
    "heli": 8, "hind": 8, "yak": 8, "mig": 8, "tran": 7, "u2": 10,
    # Ships
    "dd": 7, "ca": 8, "ss": 7, "msub": 7, "pt": 7, "lst": 5,
    # Buildings
    "fact": 5, "proc": 5, "powr": 4, "apwr": 4,
    "barr": 5, "tent": 5, "weap": 5, "afld": 5, "hpad": 5,
    "spen": 5, "syrd": 5, "silo": 4, "dome": 10,
    # Defenses (the turret IS the sight)
    "pbox": 5, "hbox": 5, "gun": 7, "ftur": 5,
    "sam": 8, "agun": 7, "tsla": 7, "atek": 6, "stek": 6,
}
_DEFAULT_SIGHT = 5


def _within_radius(cx: int, cy: int, r: int, w: int, h: int) -> list[tuple[int, int]]:
    """Cells within Chebyshev radius `r` of (cx, cy), clipped to the
    [0, w) × [0, h) grid. Matches the shape the engine uses to reveal
    shroud (square footprint, matching `world.rs` reveal logic)."""
    out: list[tuple[int, int]] = []
    if r <= 0:
        return [(cx, cy)] if 0 <= cx < w and 0 <= cy < h else []
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    for y in range(y0, y1):
        for x in range(x0, x1):
            out.append((x, y))
    return out


def _units_to_render_list(
    positions: dict[str, Any],
    hp: dict[str, Any] | None,
    type_by_id: dict[str, str] | None = None,
) -> list[dict]:
    """unit_positions {id: {cell_x, cell_y, ...}} -> [{cell_x, cell_y, type, id, hp}]."""
    hp = hp or {}
    type_by_id = type_by_id or {}
    out: list[dict] = []
    _NONCOMBAT = {"harv", "mcv", "medi", "e6", "spy", "thf"}
    for uid, p in (positions or {}).items():
        tgt = None
        if isinstance(p, dict):
            cx, cy = int(p.get("cell_x", 0)), int(p.get("cell_y", 0))
            activity = p.get("activity")
            t = p.get("target")
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                tgt = (int(t[0]), int(t[1]))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            cx, cy, activity = int(p[0]), int(p[1]), None
        else:
            continue
        utype = type_by_id.get(str(uid))
        if not utype and isinstance(p, dict):
            utype = p.get("actor_type")  # engine now emits own-unit type
        utype = utype or "?"
        is_idle = tgt is None and (activity in (None, "", "idle", "Idle"))
        entry = {
            "id": str(uid),
            "cell_x": cx,
            "cell_y": cy,
            "type": utype,
            "hp": float(hp.get(uid, hp.get(str(uid), 1.0)) or 0.0),
            "activity": activity,
            "idle": is_idle,
            # Unknown type ⇒ assume combat-capable (don't hide it from
            # the Idle scan); known non-combat types excluded.
            "can_attack": (str(utype).lower() not in _NONCOMBAT)
            if utype else True,
        }
        if tgt is not None:
            entry["target_x"], entry["target_y"] = tgt
        out.append(entry)
    return out


def _resource_cells_from_spatial(
    spatial: Any, spatial_shape: Any, explored: set[tuple[int, int]] | None = None
) -> list[dict[str, int]]:
    try:
        h, w, c = (int(v) for v in spatial_shape)
    except (TypeError, ValueError):
        return []
    if c <= 5 or h <= 0 or w <= 0 or not spatial:
        return []
    out: list[dict[str, int]] = []
    for y in range(h):
        row = y * w * c
        for x in range(w):
            if explored is not None and (x, y) not in explored:
                continue
            try:
                if float(spatial[row + x * c + 5]) > 0:
                    out.append({"cell_x": x, "cell_y": y})
            except (IndexError, TypeError, ValueError):
                return out
    return out


@dataclass
class EpisodeSignals:
    """Cumulative + per-step signals derived from Rust obs deltas.

    Drives both `reward_funcs` inputs and the P/R/A diagnostic rubrics
    (task #2). Rust gives no reward/result, so every signal lives here.
    """

    units_killed: int = 0
    units_killed_delta: int = 0
    units_lost: int = 0
    explored_percent: float = 0.0
    explored_delta: float = 0.0
    enemies_seen_ids: set[str] = field(default_factory=set)
    enemy_buildings_seen_ids: set[str] = field(default_factory=set)
    # Enemy buildings confirmed destroyed: a building seen earlier that
    # is now absent while we still have vision of its cell (so it's
    # killed, not fogged). Total + per-type — the faithful signal for
    # "eliminate the enemy's key economic structures" objectives.
    enemy_buildings_destroyed: int = 0
    enemy_buildings_destroyed_types: dict = field(default_factory=dict)
    # Per-destroyed-building records (type, last-seen cell_x, cell_y) so a
    # win can require key buildings be destroyed AT a specific region —
    # i.e. raze fact+proc at TWO separate bases (one per squad), which
    # the type-only count cannot express.
    enemy_buildings_destroyed_records: list = field(default_factory=list)
    new_enemies_this_step: int = 0
    new_buildings_this_step: int = 0
    game_tick: int = 0
    done: bool = False
    # S9 economy/production (0/empty until the engine grounds them).
    cash: int = 0
    resources: int = 0  # S1 stored (harvested, not-yet-cashed)
    resource_capacity: int = 0  # S1 storage cap (refineries/silos)
    power_provided: int = 0
    power_drained: int = 0
    harvesters: int = 0
    own_building_types: set[str] = field(default_factory=set)
    # Current agent buildings as (type, cell_x, cell_y) — positions needed
    # for placement/region win-conditions (build defenses in a direction,
    # found a new base near a region).
    own_buildings: list[tuple[str, int, int]] = field(default_factory=list)
    production_items: list[str] = field(default_factory=list)
    # Parallel to `production_items`: per-entry production detail surfaced
    # by the engine (`item` / `progress` 0-1 / `done` bool). The bench
    # historically threw `progress`/`done` away and exposed only the
    # item-name list, which made the manual-play UI confuse "queued
    # in-progress" with "completed and waiting to place" — Build then
    # Place silently no-op'd because the engine rejects `place_building`
    # on a not-yet-`done` queue entry. Surface the full record so the
    # UI / agents can distinguish "Building 30%" from "Ready (click
    # Place)". Engine raw entry shape (see env.rs::production_summary):
    #     {"item": "proc", "progress": 0.0–1.0, "done": false|true}
    production_detail: list[dict] = field(default_factory=list)
    # Per-episode scratch latch for stateful win predicates (e.g.
    # waypoint_sequence's ordered-visit progress, keyed by sequence id).
    # Reset for free: EpisodeSignals is reconstructed each episode.
    seq_progress: dict = field(default_factory=dict)
    # Per-episode latch for the `then:[A,B]` happened-before composite
    # (clauses-satisfied-so-far index, keyed by the `then.id`). Lets a
    # scenario require "scout → THEN commit counter" instead of
    # ``all_of`` which is satisfied by any state where both happen to
    # be true. See win_conditions._then.
    then_progress: dict = field(default_factory=dict)
    # Per-episode tool-use accounting for the strict-toolban / procedural-
    # compliance family. tools_called counts each tool name the agent
    # invoked this episode; tool_violations counts how many of those calls
    # were on the scenario's forbidden_tools list. The `tool_violations_gte`
    # predicate reads from here (typically as a fail clause). Tracking is
    # bench-side (see eval_core.run_level), so scripted policies are
    # graded by the same rule as live models.
    tools_called: dict[str, int] = field(default_factory=dict)
    tool_violations: int = 0
    # Outcome is synthesized (Rust has no result field): a scenario is
    # "won" when all enemy buildings have been discovered AND/OR all
    # enemy units neutralized — refined per-scenario in Phase 2 rubrics.
    outcome: float = 0.0

    def as_reward_kwargs(self) -> dict[str, Any]:
        """Shape expected by OpenRA-RL-Training reward_funcs (game signals)."""
        return {
            "units_killed": self.units_killed,
            "units_lost": self.units_lost,
            "explored_percent": self.explored_percent,
            "enemies_discovered": len(self.enemies_seen_ids),
            "buildings_discovered": len(self.enemy_buildings_seen_ids),
            "outcome": self.outcome,
            "game_tick": self.game_tick,
            "done": self.done,
            "cash": self.cash,
            "resources": self.resources,
            "economy_value": self.cash + self.resources,
            "harvesters": self.harvesters,
            "buildings_owned": len(self.own_building_types),
        }


class RustObsAdapter:
    """Stateful per-episode adapter. One instance per episode.

    Usage:
        ad = RustObsAdapter(scenario_def)
        ad.observe(reset_obs)
        ...loop: ad.observe(step_obs, done=done)
        render_state = ad.render_state()        # for minimap_renderer
        sig = ad.signals                        # for scoring / diagnostics
    """

    def __init__(self, scenario: Any = None, type_by_id: dict[str, str] | None = None):
        self.scenario = scenario
        self.type_by_id = type_by_id or {}
        self.signals = EpisodeSignals()
        self._explored: set[tuple[int, int]] = set()
        # Current-turn shroud-2 (visible) approximation. Recomputed every
        # `observe()` call from the agent's live-actor footprints — see
        # _SIGHT_BY_TYPE and `_compute_visible_cells`. NOT carried across
        # turns: by definition a cell is "visible" only while at least
        # one of your actors can see it RIGHT NOW.
        self._visible: set[tuple[int, int]] = set()
        # Previous-turn explored snapshot — gives an exact lower bound on
        # the visible set (cells newly added to explored THIS turn MUST
        # be currently visible, since the engine only adds to explored
        # via a live reveal). Merged into `_visible` as a safety net for
        # actor types missing from `_SIGHT_BY_TYPE`.
        self._prev_explored: set[tuple[int, int]] = set()
        self._prev_own_ids: set[str] = set()
        self._raw: dict[str, Any] = {}
        self._first_own_count: int | None = None
        # id -> (type, (cell_x, cell_y)) last time the building was seen,
        # for destruction detection (absent + cell explored ⇒ killed).
        self._seen_buildings: dict[str, tuple[str, tuple[int, int]]] = {}
        self._destroyed_bldg_ids: set[str] = set()

    # -- ingestion --------------------------------------------------------
    def observe(self, obs: dict[str, Any], done: bool = False) -> None:
        self._raw = obs or {}
        s = self.signals

        own = self._raw.get("unit_positions", {}) or {}
        own_ids = {str(k) for k in own}
        if self._first_own_count is None:
            self._first_own_count = len(own_ids)
        # Lost = units that disappeared from our roster.
        s.units_lost = max(0, (self._first_own_count or 0) - len(own_ids))
        self._prev_own_ids = own_ids

        prev_kills = s.units_killed
        s.units_killed = int(self._raw.get("units_killed", s.units_killed) or 0)
        s.units_killed_delta = max(0, s.units_killed - prev_kills)

        prev_expl = s.explored_percent
        s.explored_percent = float(self._raw.get("explored_percent", prev_expl) or 0.0)
        s.explored_delta = max(0.0, s.explored_percent - prev_expl)
        # Snapshot previous explored set BEFORE updating, so the "newly
        # revealed this turn" delta can seed the visible approximation.
        self._prev_explored = set(self._explored)
        self._explored.update(_cells(self._raw.get("explored_cells")))
        # Recompute the 3-state visible set from this turn's actor list.
        self._visible = self._compute_visible_cells()

        before_e = len(s.enemies_seen_ids)
        for e in self._raw.get("enemy_positions", []) or []:
            if isinstance(e, dict) and e.get("id") is not None:
                s.enemies_seen_ids.add(str(e["id"]))
        s.new_enemies_this_step = len(s.enemies_seen_ids) - before_e

        before_b = len(s.enemy_buildings_seen_ids)
        visible_b: set[str] = set()
        for b in self._raw.get("enemy_buildings_summary", []) or []:
            if isinstance(b, dict) and b.get("id") is not None:
                bid = str(b["id"])
                s.enemy_buildings_seen_ids.add(bid)
                visible_b.add(bid)
                self._seen_buildings[bid] = (
                    str(b.get("type", "")).lower(),
                    (int(b.get("cell_x", 0)), int(b.get("cell_y", 0))),
                )
        s.new_buildings_this_step = len(s.enemy_buildings_seen_ids) - before_b
        # Destruction: a previously-seen enemy building now absent while
        # an agent unit is right on top of its last cell ⇒ it was
        # killed (not merely fogged after a retreat). Proximity to a
        # *current* unit is the reliable "we have vision here" test —
        # `explored_cells` is cumulative and can't distinguish the two.
        _VIS = 6  # cells; ~unit sight radius
        agent_cells = [
            (int(p.get("cell_x", 0)), int(p.get("cell_y", 0)))
            for p in (own.values() if isinstance(own, dict) else [])
            if isinstance(p, dict)
        ]
        for bid, (btype, (bx, by)) in self._seen_buildings.items():
            if bid in visible_b or bid in self._destroyed_bldg_ids:
                continue
            if any(
                max(abs(ux - bx), abs(uy - by)) <= _VIS
                for ux, uy in agent_cells
            ):
                self._destroyed_bldg_ids.add(bid)
                s.enemy_buildings_destroyed_types[btype] = (
                    s.enemy_buildings_destroyed_types.get(btype, 0) + 1
                )
                s.enemy_buildings_destroyed_records.append(
                    (btype, int(bx), int(by))
                )
        s.enemy_buildings_destroyed = len(self._destroyed_bldg_ids)

        econ = self._raw.get("economy") or {}
        if isinstance(econ, dict):
            s.cash = int(econ.get("cash", s.cash) or 0)
            s.resources = int(econ.get("resources", 0) or 0)
            s.resource_capacity = int(econ.get("resource_capacity", 0) or 0)
            s.power_provided = int(econ.get("power_provided", 0) or 0)
            s.power_drained = int(econ.get("power_drained", 0) or 0)
            s.harvesters = int(econ.get("harvesters", 0) or 0)
        obls: list[tuple[str, int, int]] = []
        for b in self._raw.get("own_buildings", []) or []:
            if isinstance(b, dict) and b.get("type"):
                t = str(b["type"]).lower()
                s.own_building_types.add(t)
                obls.append((t, int(b.get("cell_x", 0)), int(b.get("cell_y", 0))))
        s.own_buildings = obls
        s.production_items = [
            str(p.get("item", "")).lower()
            for p in (self._raw.get("production", []) or [])
            if isinstance(p, dict)
        ]
        # Carry the full per-entry record (item / progress / done) so
        # downstream consumers can tell completed-and-waiting-to-place
        # apart from still-building. See `production_detail` doc above.
        s.production_detail = [
            {
                "item": str(p.get("item", "")).lower(),
                "progress": float(p.get("progress", 0.0) or 0.0),
                "done": bool(p.get("done", False)),
            }
            for p in (self._raw.get("production", []) or [])
            if isinstance(p, dict)
        ]

        s.game_tick = int(self._raw.get("game_tick", s.game_tick) or 0)
        s.done = bool(done)

    # -- render schema ----------------------------------------------------
    def grid_dims(self, margin: int = 4) -> tuple[int, int]:
        """True map (width, height) from the engine's map_info when
        available (S9), else bound from observed extents (legacy
        fallback for envs that don't emit map_info)."""
        mi = self._raw.get("map_info") or {}
        if isinstance(mi, dict) and int(mi.get("width", 0)) > 0 and int(
            mi.get("height", 0)
        ) > 0:
            return int(mi["width"]), int(mi["height"])
        xs, ys = [0], [0]
        for src in (self._explored, _cells(self._raw.get("explored_cells"))):
            for x, y in src:
                xs.append(x)
                ys.append(y)
        for coll in (
            self._raw.get("unit_positions", {}) or {},
            self._raw.get("enemy_positions", []) or [],
            self._raw.get("enemy_buildings_summary", []) or [],
        ):
            items = coll.values() if isinstance(coll, dict) else coll
            for p in items:
                if isinstance(p, dict):
                    xs.append(int(p.get("cell_x", 0)))
                    ys.append(int(p.get("cell_y", 0)))
        return max(xs) + margin, max(ys) + margin

    def _compute_visible_cells(self) -> set[tuple[int, int]]:
        """Per-tick visible (shroud-state-2) cell set.

        Two paths, in priority order:

        1. ENGINE TRUTH — if the PyO3 obs carries `visible_cells`
           (engine PR "expose typed-shroud visible_cells via PyO3"),
           use it verbatim. This is the agent player's
           `typed_shroud.visible` mask, scanned over the playable
           rectangle by the engine — exact, no Chebyshev fudge.
        2. CHEBYSHEV FALLBACK — older engine HEADs only ship
           `explored_cells` (the cumulative fogged+visible union).
           Reconstruct visible as the union of cells within each
           live agent actor's sight radius (unit OR building),
           clipped to the playable rectangle, with cells newly
           added to `explored` THIS turn merged in as a safety net
           for actor types missing from `_SIGHT_BY_TYPE`.

        Either way the result is intersected with `_explored` so
        `visible ⊆ explored` is a hard post-condition (clip any
        out-of-bounds artefacts).
        """
        # Engine-truth path: trust `visible_cells` if the engine
        # surfaced it. An EMPTY list is still authoritative (e.g.
        # the no-agent-actors case), so the presence-of-key check
        # is on `"visible_cells" in self._raw`, not on truthiness.
        if "visible_cells" in self._raw:
            engine_vis = set(_cells(self._raw.get("visible_cells")))
            return engine_vis & self._explored

        w, h = self.grid_dims()
        visible: set[tuple[int, int]] = set()
        # Agent units.
        for uid, p in (self._raw.get("unit_positions", {}) or {}).items():
            if not isinstance(p, dict):
                continue
            cx, cy = int(p.get("cell_x", 0)), int(p.get("cell_y", 0))
            t = str(
                p.get("actor_type")
                or self.type_by_id.get(str(uid), "")
                or ""
            ).lower()
            r = _SIGHT_BY_TYPE.get(t, _DEFAULT_SIGHT)
            visible.update(_within_radius(cx, cy, r, w, h))
        # Agent buildings (buildings reveal shroud the same way).
        for b in self._raw.get("own_buildings", []) or []:
            if not isinstance(b, dict):
                continue
            cx, cy = int(b.get("cell_x", 0)), int(b.get("cell_y", 0))
            t = str(b.get("type", "")).lower()
            r = _SIGHT_BY_TYPE.get(t, _DEFAULT_SIGHT)
            visible.update(_within_radius(cx, cy, r, w, h))
        # Safety net: anything newly explored THIS turn is visible by
        # construction (engine only reveals via a live actor's sight).
        new_explored = set(_cells(self._raw.get("explored_cells"))) - self._prev_explored
        visible.update(new_explored)
        # `visible` must always be a subset of `explored` — drop any
        # out-of-bounds artefacts that survived clipping.
        return visible & self._explored

    def ascii_minimap(self) -> str:
        """Synthesize the ASCII grid the renderer parses.

        3-state encoding (RA shroud):
          '#' = unexplored (full shroud, never seen)
          '.' = fogged    (explored once, NOT currently visible)
          '+' = visible   (currently within an agent actor's sight)

        Faithful to `minimap_renderer._parse_ascii_minimap` for the
        2-state callers (anything != '#' counts as explored). The
        bench's own renderer (`minimap.py`) reads '+' / '.' separately
        to dim the fogged region so the human can SEE that they've
        lost vision of an area — without this the map stayed bright
        behind retreating units (the bug this docstring blocks).
        """
        w, h = self.grid_dims()
        explored = set(self._explored) | set(_cells(self._raw.get("explored_cells")))
        visible = self._visible
        rows = []
        for y in range(h):
            row_chars = []
            for x in range(w):
                if (x, y) in visible:
                    row_chars.append("+")
                elif (x, y) in explored:
                    row_chars.append(".")
                else:
                    row_chars.append("#")
            rows.append("".join(row_chars))
        return "\n".join(rows)

    def render_state(self) -> dict[str, Any]:
        """State dict shaped for minimap_renderer.render_minimap()/prompts."""
        w, h = self.grid_dims()
        own = _units_to_render_list(
            self._raw.get("unit_positions", {}),
            self._raw.get("unit_hp"),
            self.type_by_id,
        )
        enemy = _units_to_render_list(
            {
                str(e.get("id", i)): e
                for i, e in enumerate(self._raw.get("enemy_positions", []) or [])
            },
            self._raw.get("enemy_hp"),
        )
        enemy += [
            {
                "id": str(b.get("id", f"bldg{i}")),
                "cell_x": int(b.get("cell_x", 0)),
                "cell_y": int(b.get("cell_y", 0)),
                "type": b.get("kind") or b.get("type"),
                "hp": float(b.get("hp_pct", 1.0) or 0.0),
                "is_building": True,
            }
            for i, b in enumerate(self._raw.get("enemy_buildings_summary", []) or [])
        ]
        return {
            "units_summary": own,
            "enemy_summary": enemy,
            "minimap": self.ascii_minimap(),
            "map_width": w,
            "map_height": h,
            "bounds_x": 0,
            "bounds_y": 0,
            "game_tick": self.signals.game_tick,
            "explored_percent": self.signals.explored_percent,
            # Cumulative units killed by THIS side, sourced from the engine
            # `kills_per_player` counter (see openra-sim `credit_kill` /
            # `update_kill_counter`). Surfaced into `render_state` so the
            # 1v1 harness's military-progress tie-break (`one_v_one.py::
            # _kills`) sees the real engine value rather than defaulting
            # to 0. Without this, every 1v1 match where both bases
            # survive the deadline collapses to the next tie-break layer
            # (buildings → economy), masking real combat performance.
            "units_killed": self.signals.units_killed,
            # Economy/base state so agents can plan construction.
            "cash": self.signals.cash,
            "resources": self.signals.resources,
            "resource_capacity": self.signals.resource_capacity,
            "economy_value": self.signals.cash + self.signals.resources,
            "power_provided": self.signals.power_provided,
            "power_drained": self.signals.power_drained,
            # Own buildings carry the REAL engine actor id (and hp_pct /
            # is_primary) so an agent can target a building for repair /
            # sell / power_down / set_primary. Mirrors how units_summary
            # keeps the engine unit id — without it `prompt_v2` would
            # fall back to a list-index id the engine's resolver rejects.
            "own_buildings": [
                {
                    "id": str(b.get("id", "")),
                    "type": str(b.get("type", "")).lower(),
                    "cell_x": int(b.get("cell_x", 0)),
                    "cell_y": int(b.get("cell_y", 0)),
                    "hp": float(b.get("hp_pct", 1.0) or 0.0),
                    "is_primary": bool(b.get("is_primary", False)),
                }
                for b in (self._raw.get("own_buildings", []) or [])
                if isinstance(b, dict) and b.get("type")
            ],
            "production": list(self.signals.production_items),
            # Detailed per-entry production status (item / progress 0-1 /
            # done) — the UI uses `done` to distinguish "ready to place"
            # from "still building" so a Place click on an unfinished
            # structure surfaces a clear hint instead of silently
            # no-op'ing (engine rejects the order under the hood).
            "production_detail": [dict(p) for p in self.signals.production_detail],
            # S9 spatial tensor passthrough (flat row-major [y][x][c] +
            # (h,w,c) shape) so multimodal/spatial agents and transfer
            # studies can do grid/occupancy reasoning. Empty when the
            # engine doesn't emit it.
            "spatial": self._raw.get("spatial", []) or [],
            "spatial_shape": tuple(
                self._raw.get("spatial_shape", (0, 0, 0)) or (0, 0, 0)
            ),
            "resource_cells": _resource_cells_from_spatial(
                self._raw.get("spatial", []) or [],
                self._raw.get("spatial_shape", (0, 0, 0)) or (0, 0, 0),
                self._explored,
            ),
            # 3-state shroud (RA semantics):
            #   `explored_cells`  — cumulative union of fogged + visible
            #                       (kept for back-compat with legacy
            #                       consumers that don't care about the
            #                       fogged/visible split).
            #   `visible_cells`   — currently within an agent actor's
            #                       sight (this turn only).
            #   `fogged_cells`    — explored MINUS visible (revealed once
            #                       but no live vision — the dim tint).
            # Lists of [x, y] pairs for JSON-friendliness.
            "explored_cells": sorted([x, y] for (x, y) in self._explored),
            "visible_cells": sorted([x, y] for (x, y) in self._visible),
            "fogged_cells": sorted(
                [x, y] for (x, y) in (self._explored - self._visible)
            ),
            # Raw obs + playable bounds so the vendored training
            # minimap_v2.render (consumes unit_positions/enemy_positions/
            # explored_cells directly) and briefing_v2 can be used
            # verbatim — identical-by-construction with training.
            "_raw": self._raw,
            "bounds": (0, 0, w, h),
            "enemy_buildings_summary": list(
                self._raw.get("enemy_buildings_summary", []) or []
            ),
            "harvesters": self.signals.harvesters,
        }
