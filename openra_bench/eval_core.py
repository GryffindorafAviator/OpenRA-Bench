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
        for turns in range(1, compiled.max_turns + 1):
            rs = adapter.render_state()
            cmds = agent_fn(rs, env.Command) or [env.Command.observe()]
            obs, _r, done, info = env.step(cmds)
            adapter.observe(obs, done=done)
            issued += len(cmds)
            warned += len(info.get("warnings", []) if isinstance(info, dict) else [])
            ctx = WinContext(signals=adapter.signals, render_state=adapter.render_state())
            if evaluate(compiled.win_condition, ctx):
                outcome = "win"
            elif evaluate(compiled.fail_condition, ctx):
                outcome = "loss"
            trace.append(
                {
                    "turn": turns,
                    "tick": adapter.signals.game_tick,
                    "explored": round(adapter.signals.explored_percent, 2),
                    "kills": adapter.signals.units_killed,
                    "enemies_seen": len(adapter.signals.enemies_seen_ids),
                }
            )
            if outcome != "draw" or done:
                break
        adapter.signals.outcome = {"win": 1.0, "draw": 0.5, "loss": 0.0}[outcome]
        return EpisodeResult(
            scenario=f"{compiled.pack_id}:{compiled.level}",
            seed=seed,
            turns=turns,
            signals=adapter.signals,
            outcome=outcome,
            actions_issued=issued,
            actions_warned=warned,
            trace=trace,
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp_path).unlink(missing_ok=True)
