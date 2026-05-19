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
        self._explored.update(_cells(self._raw.get("explored_cells")))

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

    def ascii_minimap(self) -> str:
        """Synthesize the ASCII grid the renderer parses for the explored
        mask: '#' = unexplored, '.' = explored. Faithful to
        minimap_renderer._parse_ascii_minimap (anything != '#' = explored).
        """
        w, h = self.grid_dims()
        explored = set(self._explored) | set(_cells(self._raw.get("explored_cells")))
        rows = []
        for y in range(h):
            rows.append("".join("." if (x, y) in explored else "#" for x in range(w)))
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
            # Economy/base state so agents can plan construction.
            "cash": self.signals.cash,
            "resources": self.signals.resources,
            "resource_capacity": self.signals.resource_capacity,
            "economy_value": self.signals.cash + self.signals.resources,
            "power_provided": self.signals.power_provided,
            "power_drained": self.signals.power_drained,
            "own_buildings": [
                {"type": t, "cell_x": x, "cell_y": y}
                for (t, x, y) in self.signals.own_buildings
            ],
            "production": list(self.signals.production_items),
            # S9 spatial tensor passthrough (flat row-major [y][x][c] +
            # (h,w,c) shape) so multimodal/spatial agents and transfer
            # studies can do grid/occupancy reasoning. Empty when the
            # engine doesn't emit it.
            "spatial": self._raw.get("spatial", []) or [],
            "spatial_shape": tuple(
                self._raw.get("spatial_shape", (0, 0, 0)) or (0, 0, 0)
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
