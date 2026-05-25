"""Episode spine: Rust env + adapter + pluggable agent.

This is the Bench-side replacement for Training's `play_episodes_async`
(which is hardwired to the C# gRPC server). It reuses Training *components*
via the adapter; provider-agnostic agents plug in here (Phase 0 follow-up:
openra_bench/agent.py with vLLM/OpenRouter/Bedrock).

An `agent_fn` has signature:
    agent_fn(render_state: dict, Command) -> list[Command]
where `Command` is `openra_train.Command` (move_units/attack_unit/observe).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from openra_rl_training.training.rust_env_pool import RustEnvPool

from .controller import (
    Controller,
    EpisodeContext,
    as_controller,
    introspection_source,
)
from .rust_adapter import EpisodeSignals, RustObsAdapter
from .scenarios.schema import CompiledLevel
from .scenarios.win_conditions import WinContext, evaluate

# A policy is either a bare `agent_fn(render_state, Command) -> [Command]`
# callable (the legacy shape, still accepted everywhere) or a Controller.
AgentFn = Callable[[dict, Any], list]
Policy = "AgentFn | Controller"


def _scenario_to_tmp_yaml(compiled: CompiledLevel) -> str:
    """Serialize a compiled level's ScenarioDefinition to a temp YAML the
    Rust env can load (it reads actors from the given scenario path; the
    map geometry is the Rust-supported base map)."""
    data = compiled.scenario.model_dump(mode="json", exclude_none=True)
    # The Rust loader resolves base_map relative to the scenario file's
    # dir; this temp file lives in /tmp, so a relative ref would silently
    # fall back to rush-hour terrain. Pin it to the resolved absolute
    # .oramap so the *declared* map's real terrain loads.
    from .scenarios.loader import resolve_map_path

    _mp = resolve_map_path(str(data.get("base_map", "")))
    if _mp is not None:
        data["base_map"] = str(_mp)
    # Training's ScenarioDefinition has no economy field; inject the
    # pack's designed `starting_cash` constraint as a top-level key the
    # Rust scenario parser reads (default 5000 when unset).
    if compiled.starting_cash is not None:
        data["starting_cash"] = compiled.starting_cash
    # Per-player cash plumbing footgun: `PlayerSetup.cash` defaults to
    # `int = 0` (not Optional), so an unset `agent: {faction: ...}` in
    # the pack serializes as `{faction: ..., cash: 0}`. With the
    # engine's per-player-cash fix landed, that 0 silently OVERRIDES
    # the top-level `starting_cash:` — production stalls because the
    # agent has no money to consume. Defensive bench-side fix: if
    # `agent.cash` / `enemy.cash` is 0 AND `starting_cash` > 0, STRIP
    # the per-player cash field so the engine falls back to the
    # top-level. (A pack that genuinely wants cash=0 should also set
    # `starting_cash: 0`, in which case there's nothing to fall back
    # to and the behavior is unchanged.)
    _top_cash = int(data.get("starting_cash") or 0)
    for _side in ("agent", "enemy"):
        _block = data.get(_side)
        if isinstance(_block, dict) and _block.get("cash", None) == 0 and _top_cash > 0:
            _block.pop("cash", None)
            data[_side] = _block
    # The Rust engine defaults spawn_mcvs:true → it auto-seeds MCVs at
    # the map's built-in spawn points (e.g. (124,36)), which reveal fog
    # and pollute unit counts for scenarios that never asked for them.
    # Curated packs declare their own actors, so default it OFF; a pack
    # may still opt in via base.spawn_mcvs / a scenario-declared `mcv`.
    data.setdefault("spawn_mcvs", bool(data.get("spawn_mcvs", False)))
    # Wave-9 `scheduled_events:` — preserved on CompiledLevel because
    # the training ScenarioDefinition silently drops the field. Re-emit
    # here so the Rust engine's scenario parser sees it.
    sched = getattr(compiled, "scheduled_events", None) or []
    if sched:
        data["scheduled_events"] = sched
    # Resource-wave `ore_patches:` — same lifting pattern as
    # `scheduled_events`. Each entry is `{x, y, amount, radius}` and
    # the engine (`oramap.rs::read_ore_patches`) materialises it into
    # a disk of ore cells on the terrain at world-build time.
    ore = getattr(compiled, "ore_patches", None) or []
    if ore:
        data["ore_patches"] = ore
    # No-fog perception cells (fog_mode ends with "-clear") flip the
    # engine's `reveal_map` flag: the agent observes the whole map with
    # no fog of war — the clear half of the perception ablation grid.
    if getattr(compiled, "reveal_map", False):
        data["reveal_map"] = True
    # Naval-MVP overlay: forward declared `water_cells:` and
    # `water_rect:` so the Rust engine marks the corresponding
    # terrain cells as ship-passable / ground-impassable. Without
    # this lift the engine's terrain stays all-grass and ships have
    # nowhere to move.
    wc = getattr(compiled, "water_cells", None) or []
    if wc:
        data["water_cells"] = [list(c) for c in wc]
    wr = getattr(compiled, "water_rect", None)
    if wr:
        data["water_rect"] = list(wr)
    fd = tempfile.NamedTemporaryFile(
        "w", suffix=f"_{compiled.pack_id}_{compiled.level}.yaml", delete=False
    )
    yaml.safe_dump(data, fd, sort_keys=False)
    fd.close()
    return fd.name


@dataclass
class EpisodeResult:
    scenario: str
    seed: int
    turns: int
    signals: EpisodeSignals
    outcome: str = "draw"  # "win" | "loss" | "draw"
    actions_issued: int = 0
    actions_warned: int = 0  # commands the engine rejected/warned on
    trace: list[dict] = field(default_factory=list)
    # Final goal-tracker snapshot (always computed, playback or not).
    # `objective_blocking_ratio` is the worst-leaf ratio — i.e. the
    # bottleneck constraint, since an `all_of` win needs every leaf
    # satisfied (see goal_tracker for why min beats mean). `leaves_final`
    # carries the per-predicate `{name, target, current, ratio,
    # satisfied}` snapshot from the final turn — the SOURCE OF TRUTH
    # for "how close to the win" reporting. `objective_progress` is
    # kept as a deprecated alias of `objective_blocking_ratio` for
    # one release so older readers don't crash.
    objective_progress: float = 0.0  # deprecated alias of objective_blocking_ratio
    objective_blocking_ratio: float = 0.0
    leaves_final: list[dict] = field(default_factory=list)
    reward_vector: dict = field(default_factory=dict)


_CMD_NAME_RE = __import__("re").compile(r"Command::([A-Z][A-Za-z0-9]*)")


def _cmd_tool_name(cmd: Any) -> str | None:
    """Decode the snake_case tool name from a Command repr.

    The Rust pyo3 Command enum stringifies as ``Command::VariantName { … }``
    or ``Command::VariantName``. We extract the variant and convert to
    snake_case to match the bench's tool-name vocabulary (and the
    `tools:` / `forbidden_tools:` allowlist keys in YAML). Returns
    ``None`` for anything that doesn't match — defensive; never raises.
    """
    m = _CMD_NAME_RE.search(repr(cmd))
    if not m:
        return None
    variant = m.group(1)
    # CamelCase → snake_case (MoveUnits → move_units, AttackUnit → attack_unit)
    out: list[str] = []
    for i, ch in enumerate(variant):
        if i and ch.isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def scripted_explore_agent(render_state: dict, Command: Any) -> list:
    """Baseline reference agent: walk every unit toward the nearest
    unexplored frontier cell. Exercises the move path; a useful
    lower-bound control for the perception/exploration scenarios.
    """
    grid = render_state["minimap"].splitlines()
    h = len(grid)
    w = len(grid[0]) if grid else 0
    frontier = [
        (x, y)
        for y in range(h)
        for x in range(min(w, len(grid[y])))
        if grid[y][x] == "#"
    ]
    units = render_state.get("units_summary", [])
    if not units or not frontier:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        tx, ty = min(frontier, key=lambda c: (c[0] - ux) ** 2 + (c[1] - uy) ** 2)
        cmds.append(Command.move_units([str(u["id"])], target_x=tx, target_y=ty))
    return cmds


def run_episode(
    scenario_path: str,
    agent_fn: "AgentFn | Controller" = scripted_explore_agent,
    max_turns: int = 40,
    seed: int = 0,
    pool: RustEnvPool | None = None,
) -> EpisodeResult:
    """Run a scenario for a fixed number of turns. `agent_fn` may be a
    bare `agent_fn(render_state, Command) -> [Command]` callable or any
    `Controller`; it is coerced through `as_controller()`."""
    owns_pool = pool is None
    if pool is None:
        pool = RustEnvPool(size=1, scenario_path=scenario_path)
    env = pool.acquire()
    try:
        adapter = RustObsAdapter()
        obs = env.reset(seed=seed)
        adapter.observe(obs)
        controller = as_controller(agent_fn)
        controller.reset(
            EpisodeContext(seed=seed, max_turns=max_turns)
        )
        trace: list[dict] = []
        turns = 0
        issued = warned = 0
        for turns in range(1, max_turns + 1):
            rs = adapter.render_state()
            cmds = controller.act(rs, env.Command) or [env.Command.observe()]
            obs, _reward, done, info = env.step(cmds)
            adapter.observe(obs, done=done)
            issued += len(cmds)
            warned += len(info.get("warnings", []) if isinstance(info, dict) else [])
            trace.append(
                {
                    "turn": turns,
                    "tick": adapter.signals.game_tick,
                    "explored": round(adapter.signals.explored_percent, 2),
                    "kills": adapter.signals.units_killed,
                    "enemies_seen": len(adapter.signals.enemies_seen_ids),
                    "n_cmds": len(cmds),
                }
            )
            if done:
                break
        return EpisodeResult(
            scenario=scenario_path,
            seed=seed,
            turns=turns,
            signals=adapter.signals,
            actions_issued=issued,
            actions_warned=warned,
            trace=trace,
        )
    finally:
        pool.release(env)
        if owns_pool:
            pool.shutdown()


def run_level(
    compiled: CompiledLevel,
    agent_fn: "AgentFn | Controller" = scripted_explore_agent,
    seed: int = 0,
    playback=None,
    full_playback=None,
) -> EpisodeResult:
    """Run one scenario-pack level, scoring against its declarative
    win/fail conditions (checked every turn). Outcome maps to the
    `reward_outcome` convention: win=1.0, draw=0.0 (= loss), loss=0.0.
    A draw is a non-win — the agent failed to satisfy the win predicate
    by the deadline — so it scores as a loss. The `outcome` STRING is
    still emitted as "draw" so audit tooling can distinguish deadline
    timeouts from active fail-condition triggers (the former usually
    signals a scenario defect: missing/inert fail_condition or an
    auto-`done` race).

    `agent_fn` may be a bare `agent_fn(render_state, Command) ->
    [Command]` callable, a `ModelAgent` bound method, or any
    `Controller`; it is coerced through `as_controller()`.
    """
    if not compiled.map_supported:
        raise RuntimeError(
            f"{compiled.pack_id}: base map not Rust-loadable yet (Phase 3). "
            f"Validate-only; cannot execute."
        )
    tmp_path = _scenario_to_tmp_yaml(compiled)
    pool = RustEnvPool(size=1, scenario_path=tmp_path)
    env = pool.acquire()
    try:
        adapter = RustObsAdapter()
        adapter.observe(env.reset(seed=seed))
        # Coerce the policy through the unified Controller contract:
        # a bare agent_fn, a ModelAgent bound method, or a Controller
        # all resolve to a Controller the loop drives identically.
        controller = as_controller(agent_fn)
        controller.reset(
            EpisodeContext(
                pack_id=compiled.pack_id,
                level=compiled.level,
                seed=seed,
                objective=compiled.scenario.description or "",
                max_turns=compiled.max_turns,
            )
        )
        trace: list[dict] = []
        outcome = "draw"
        turns = 0
        issued = warned = 0
        conceded = False
        # Persistent fog history so the SAVED minimap == the image the
        # model actually saw (same vendored _minimap_v2, accumulating).
        _pb_explored: set = set()
        _pb_terrain = None
        # Audit-capture wiring: when a FullPlayback is attached, surface
        # the underlying ModelAgent (if any) and flip on `audit_capture`
        # so per-turn briefing / wire request+response are stashed for
        # the audit JSONL.
        _audit_agent = (
            introspection_source(controller) if full_playback is not None else None
        )
        if _audit_agent is not None and hasattr(_audit_agent, "audit_capture"):
            _audit_agent.audit_capture = True
        # Interrupt-driven mode (step 4): if the scenario enabled any
        # interrupt signals, advance with step_until_event so the agent
        # is re-prompted (debriefed) the moment an event fires
        # (enemy spotted, unit lost, production complete, …) instead of
        # only on fixed tick boundaries. Falls back to fixed step()
        # when no signals are enabled or the env lacks the API.
        _KNOWN_SIGNALS = {
            "enemy_unit_spotted", "enemy_building_spotted", "engage_start",
            "own_unit_destroyed", "production_complete",
        }
        enabled_sig = sorted(
            s for s, on in (compiled.scenario.interrupts or {}).items()
            if on and s in _KNOWN_SIGNALS
        )
        raw_env = getattr(env, "_env", None)
        interrupt_mode = bool(enabled_sig) and raw_env is not None and hasattr(
            raw_env, "step_until_event"
        )
        # Strict-toolban / procedural-compliance accounting: any cmd whose
        # tool name is in compiled.forbidden_tools increments
        # signals.tool_violations (read by the tool_violations_gte
        # predicate). Tracked here so scripted and live-model policies
        # are graded by the exact same rule.
        forbidden = {str(t).lower() for t in (compiled.forbidden_tools or [])}
        for turns in range(1, compiled.max_turns + 1):
            rs = adapter.render_state()
            cmds = controller.act(rs, env.Command) or [env.Command.observe()]
            for _cmd in cmds:
                _tn = _cmd_tool_name(_cmd)
                if _tn:
                    adapter.signals.tools_called[_tn] = (
                        adapter.signals.tools_called.get(_tn, 0) + 1
                    )
                    if _tn in forbidden:
                        adapter.signals.tool_violations += 1
            if not conceded:
                conceded = any("Surrender" in repr(c) for c in cmds)
            interrupt = None
            if interrupt_mode:
                obs, _r, done, info, was_int, reason, _tk = (
                    raw_env.step_until_event(cmds, None, 5, enabled_sig)
                )
                if was_int:
                    interrupt = reason
            else:
                obs, _r, done, info = env.step(cmds)
            adapter.observe(obs, done=done)
            issued += len(cmds)
            warned += len(info.get("warnings", []) if isinstance(info, dict) else [])
            ctx = WinContext(signals=adapter.signals, render_state=adapter.render_state())
            if evaluate(compiled.win_condition, ctx):
                outcome = "win"
            elif evaluate(compiled.fail_condition, ctx):
                outcome = "loss"
            if playback is not None:
                _png = None
                try:
                    from .minimap import terrain_png_for

                    if _pb_terrain is None:
                        _pb_terrain = terrain_png_for(
                            compiled.scenario.base_map
                        )
                    # Same vendored _minimap_v2 the model is sent, with
                    # accumulating fog → saved image == model's image.
                    from .prompt_v2 import minimap_b64 as _v2_mm

                    _png = _v2_mm(
                        rs, _pb_terrain, _pb_explored,
                        constant_colors=compiled.level in ("easy", "medium"),
                    )
                    if _png is None:
                        from .agent import _render_minimap_b64

                        _png = _render_minimap_b64(rs, _pb_terrain)
                except Exception:  # noqa: BLE001 — playback never breaks a run
                    pass
                from .goal_tracker import turn_goal

                playback.record_turn(
                    turns, rs, cmds, adapter.signals, _png,
                    interrupt=interrupt,
                    goal=turn_goal(compiled.win_condition, ctx),
                )
            if full_playback is not None:
                # Mirror the same PNG (when the legacy playback rendered
                # one). Otherwise render on-demand for the audit format.
                _fp_png = locals().get("_png") if playback is not None else None
                if _fp_png is None:
                    try:
                        from .minimap import terrain_png_for
                        if _pb_terrain is None:
                            _pb_terrain = terrain_png_for(
                                compiled.scenario.base_map
                            )
                        from .prompt_v2 import minimap_b64 as _v2_mm
                        _fp_png = _v2_mm(
                            rs, _pb_terrain, _pb_explored,
                            constant_colors=compiled.level in ("easy", "medium"),
                        )
                        if _fp_png is None:
                            from .agent import _render_minimap_b64
                            _fp_png = _render_minimap_b64(rs, _pb_terrain)
                    except Exception:  # noqa: BLE001 — audit never breaks a run
                        _fp_png = None
                try:
                    full_playback.record_turn(
                        turn=turns,
                        tick=adapter.signals.game_tick,
                        obs=rs,
                        briefing=getattr(_audit_agent, "last_briefing", "")
                        if _audit_agent is not None
                        else "",
                        system_prompt=getattr(_audit_agent, "system_prompt", "")
                        if _audit_agent is not None
                        else "",
                        model_request=getattr(_audit_agent, "last_request", None)
                        if _audit_agent is not None
                        else None,
                        model_response=getattr(_audit_agent, "last_response", None)
                        if _audit_agent is not None
                        else None,
                        commands_issued=cmds,
                        engine_warnings=(
                            info.get("warnings", [])
                            if isinstance(info, dict)
                            else []
                        ),
                        signals=adapter.signals,
                        minimap_png_b64=_fp_png,
                        done=bool(done),
                        interrupt=interrupt,
                    )
                except Exception:  # noqa: BLE001
                    pass
            trace.append(
                {
                    "turn": turns,
                    "tick": adapter.signals.game_tick,
                    "explored": round(adapter.signals.explored_percent, 2),
                    "kills": adapter.signals.units_killed,
                    "enemies_seen": len(adapter.signals.enemies_seen_ids),
                    "interrupt": interrupt,
                }
            )
            if outcome != "draw" or done:
                break
        if conceded:
            outcome = "loss"  # the agent chose to concede
        adapter.signals.outcome = {"win": 1.0, "draw": 0.0, "loss": 0.0}[outcome]
        from .goal_tracker import turn_goal

        final_rs = adapter.render_state()
        final_goal = turn_goal(
            compiled.win_condition,
            WinContext(signals=adapter.signals, render_state=final_rs),
        )
        # Terminal frame: the RESOLVED post-action board the moment the
        # episode ends (win/loss). The per-turn record uses the
        # pre-step state, so without this the viewer never shows the
        # winning/losing position. No model action on this frame.
        if playback is not None:
            _fpng = None
            try:
                from .minimap import terrain_png_for

                if _pb_terrain is None:
                    _pb_terrain = terrain_png_for(compiled.scenario.base_map)
                from .prompt_v2 import minimap_b64 as _v2_mm

                _fpng = _v2_mm(
                    final_rs, _pb_terrain, _pb_explored,
                    constant_colors=compiled.level in ("easy", "medium"),
                )
                if _fpng is None:
                    from .agent import _render_minimap_b64

                    _fpng = _render_minimap_b64(final_rs, _pb_terrain)
            except Exception:  # noqa: BLE001 — playback never breaks a run
                pass
            playback.record_turn(
                turns + 1, final_rs,
                [f"(episode end: {('loss' if conceded else outcome)})"],
                adapter.signals, _fpng, interrupt=None, goal=final_goal,
            )
        # `final_goal` always exposes the new `objective_blocking_ratio`
        # key (and the deprecated `objective_progress` alias). Read the
        # new key first; fall back for paranoia in case a hand-built
        # dict is ever injected here.
        _blk = final_goal.get(
            "objective_blocking_ratio", final_goal.get("objective_progress", 0.0)
        )
        result = EpisodeResult(
            scenario=f"{compiled.pack_id}:{compiled.level}",
            seed=seed,
            turns=turns,
            signals=adapter.signals,
            outcome=outcome,
            actions_issued=issued,
            actions_warned=warned,
            trace=trace,
            objective_progress=_blk,
            objective_blocking_ratio=_blk,
            leaves_final=list(final_goal.get("leaves") or []),
            reward_vector=final_goal["reward_vector"],
        )
        if playback is not None:
            # Dump the full model⇄env transcript when the agent is a
            # ModelAgent — the Controller layer surfaces the underlying
            # instance (bound-method __self__ or the Controller itself).
            agent_obj = introspection_source(controller)
            hist = getattr(agent_obj, "history", None)
            if isinstance(hist, list):
                playback.write_messages(hist)
            playback.finalize(
                {
                    "scenario": result.scenario,
                    "pack_id": compiled.pack_id,
                    "level": compiled.level,
                    "capability": compiled.meta.capability,
                    "run_id": getattr(playback, "run_id", None),
                    "model": getattr(playback, "model", None),
                    "seed": seed,
                    "outcome": outcome,
                    "turns": turns,
                    "max_turns": compiled.max_turns,
                    "actions_issued": issued,
                    "actions_warned": warned,
                    "agent_stats": getattr(agent_obj, "stats", None),
                    "objective_progress": result.objective_progress,
                    "objective_blocking_ratio": result.objective_blocking_ratio,
                    "leaves_final": result.leaves_final,
                    "reward_vector": result.reward_vector,
                    "signals": {
                        "economy_value": adapter.signals.cash
                        + adapter.signals.resources,
                        "explored_percent": round(
                            adapter.signals.explored_percent, 2
                        ),
                        "units_killed": adapter.signals.units_killed,
                        "units_lost": adapter.signals.units_lost,
                    },
                }
            )
        if full_playback is not None:
            try:
                full_playback.finalize(
                    outcome=outcome,
                    final_obs=final_rs,
                    manifest_extra={
                        "scenario": result.scenario,
                        "pack_id": compiled.pack_id,
                        "level": compiled.level,
                        "capability": compiled.meta.capability,
                        "seed": seed,
                        "outcome": outcome,
                        "turns": turns,
                        "max_turns": compiled.max_turns,
                        "actions_issued": issued,
                        "actions_warned": warned,
                        "agent_stats": getattr(
                            introspection_source(controller), "stats", None
                        ),
                        "objective_progress": result.objective_progress,
                        "objective_blocking_ratio": result.objective_blocking_ratio,
                        "leaves_final": result.leaves_final,
                        "reward_vector": result.reward_vector,
                    },
                )
            except Exception:  # noqa: BLE001 — never break a run on I/O
                pass
        return result
    finally:
        # Abort the audit recorder if the loop crashed before finalize —
        # leaves a `.partial` on disk for forensics; the resume scanner
        # correctly sees no `.jsonl` and retries the cell.
        try:
            if full_playback is not None and Path(
                full_playback.jsonl_path
            ).exists() is False:
                full_playback.abort()
        except Exception:  # noqa: BLE001
            pass
        pool.release(env)
        pool.shutdown()
        Path(tmp_path).unlink(missing_ok=True)
