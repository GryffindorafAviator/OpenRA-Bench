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

from .rust_adapter import EpisodeSignals, RustObsAdapter
from .scenarios.schema import CompiledLevel
from .scenarios.win_conditions import WinContext, evaluate

AgentFn = Callable[[dict, Any], list]


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
    agent_fn: AgentFn = scripted_explore_agent,
    max_turns: int = 40,
    seed: int = 0,
    pool: RustEnvPool | None = None,
) -> EpisodeResult:
    owns_pool = pool is None
    if pool is None:
        pool = RustEnvPool(size=1, scenario_path=scenario_path)
    env = pool.acquire()
    try:
        adapter = RustObsAdapter()
        obs = env.reset(seed=seed)
        adapter.observe(obs)
        trace: list[dict] = []
        turns = 0
        issued = warned = 0
        for turns in range(1, max_turns + 1):
            rs = adapter.render_state()
            cmds = agent_fn(rs, env.Command) or [env.Command.observe()]
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
    agent_fn: AgentFn = scripted_explore_agent,
    seed: int = 0,
    playback=None,
) -> EpisodeResult:
    """Run one scenario-pack level, scoring against its declarative
    win/fail conditions (checked every turn). Outcome maps to the
    `reward_outcome` convention: win=1.0, draw=0.5, loss=0.0.
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
        trace: list[dict] = []
        outcome = "draw"
        turns = 0
        issued = warned = 0
        conceded = False
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
        for turns in range(1, compiled.max_turns + 1):
            rs = adapter.render_state()
            cmds = agent_fn(rs, env.Command) or [env.Command.observe()]
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
                    from .agent import _render_minimap_b64

                    _png = _render_minimap_b64(rs)
                except Exception:  # noqa: BLE001 — playback never breaks a run
                    pass
                from .goal_tracker import turn_goal

                playback.record_turn(
                    turns, rs, cmds, adapter.signals, _png,
                    interrupt=interrupt,
                    goal=turn_goal(compiled.win_condition, ctx),
                )
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
        adapter.signals.outcome = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
        result = EpisodeResult(
            scenario=f"{compiled.pack_id}:{compiled.level}",
            seed=seed,
            turns=turns,
            signals=adapter.signals,
            outcome=outcome,
            actions_issued=issued,
            actions_warned=warned,
            trace=trace,
        )
        if playback is not None:
            # Dump the full model⇄env transcript when the agent is a
            # ModelAgent (bound-method closure exposes the instance).
            agent_obj = getattr(agent_fn, "__self__", None)
            hist = getattr(agent_obj, "history", None)
            if isinstance(hist, list):
                playback.write_messages(hist)
            playback.finalize(
                {
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
                    "agent_stats": getattr(agent_obj, "stats", None),
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
        return result
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp_path).unlink(missing_ok=True)
