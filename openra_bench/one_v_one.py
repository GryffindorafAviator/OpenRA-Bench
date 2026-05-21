"""Phase 3 — the 1v1 full-macro adversarial harness.

Drives TWO Controllers (Phase 1 contract) — any mix of LLM / human /
scripted — over ONE shared episode. Each side is fed its own
fog-of-war observation and issues commands into the same engine frame
via the `step_1v1` two-player command channel (engine change shipped
alongside this module). Full macro game: economy, production, tech and
combat all in play; the episode ends when one base falls or the turn
cap is hit.

`step_1v1` builds each side's orders independently (scoped to that
player's unit ownership) and applies them into the SAME first frame, so
neither side moves "first". It returns each player's observation from
its own shroud — so an LLM driving the enemy sees exactly the fogged
view its opponent sees, not a god's-eye board.

Usage:

    from openra_bench.one_v_one import run_1v1
    result = run_1v1(scenario_path, agent_controller, enemy_controller)
    print(result.winner, result.turns)

The two controllers are interchangeable: pass two `ModelAgent`s for
model-vs-model, a `ModelAgent` + a scripted Controller for
agent-vs-bot, or a `HumanController` on either side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .controller import EpisodeContext, as_controller
from .rust_adapter import RustObsAdapter


@dataclass
class OneVOneResult:
    """Outcome of a 1v1 match.

    `winner` is from the agent side's frame of reference: "agent",
    "enemy", or "draw". `reason` records why the match ended."""

    winner: str  # "agent" | "enemy" | "draw"
    reason: str
    turns: int
    ticks: int
    agent_name: str
    enemy_name: str
    agent_trace: list[dict] = field(default_factory=list)
    enemy_trace: list[dict] = field(default_factory=list)


def _alive(render_state: dict) -> bool:
    """A side is still in the game if it has any unit or any building."""
    units = render_state.get("units_summary") or []
    buildings = render_state.get("own_buildings") or []
    return bool(units) or bool(buildings)


def _economy_value(render_state: dict) -> int:
    """Cash + stored resources — the deadline tie-break metric."""
    return int(render_state.get("cash", 0) or 0) + int(
        render_state.get("resources", 0) or 0
    )


def run_1v1(
    scenario_path: str,
    agent_controller: Any,
    enemy_controller: Any,
    seed: int = 0,
    max_turns: int = 200,
) -> OneVOneResult:
    """Run one full-macro 1v1 match and return the result.

    `agent_controller` / `enemy_controller` are each a Controller, a
    `ModelAgent`, or a bare `agent_fn` callable — coerced through
    `as_controller()`. The scenario should leave the enemy side
    externally controlled (no `enemy.bot_type`); if it declares an
    engine bot, that bot co-drives the enemy actors alongside this
    harness's enemy controller.
    """
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    agent = as_controller(agent_controller)
    enemy = as_controller(enemy_controller)

    pool = RustEnvPool(size=1, scenario_path=scenario_path)
    env = pool.acquire()
    try:
        agent.reset(
            EpisodeContext(seed=seed, side="agent", max_turns=max_turns)
        )
        enemy.reset(
            EpisodeContext(seed=seed, side="enemy", max_turns=max_turns)
        )

        agent_ad = RustObsAdapter()
        enemy_ad = RustObsAdapter()
        Command = env.Command

        # Seed each side's first fog-of-war observation WITHOUT stepping:
        # reset() gives the agent's; enemy_observation() gives the
        # enemy's at the same tick-0 state. (A two-player idle bootstrap
        # step would waste a whole decision turn — fatal on a combat map
        # where forces start in contact.) NOTE: the pool rebuilds the
        # underlying `_env` on reset(), so the raw env must be fetched
        # AFTER reset() — fetching it earlier captures a stale env.
        agent_ad.observe(env.reset(seed=seed))
        raw = getattr(env, "_env", None)
        if raw is None or not hasattr(raw, "step_1v1"):
            raise RuntimeError(
                "engine wheel lacks step_1v1 — rebuild the wheel "
                "(maturin develop --release) to run 1v1 matches"
            )
        enemy_ad.observe(raw.enemy_observation())

        agent_trace: list[dict] = []
        enemy_trace: list[dict] = []
        turns = 0
        done = False
        winner = "draw"
        reason = "turn cap reached"

        for turns in range(1, max_turns + 1):
            a_rs = agent_ad.render_state()
            e_rs = enemy_ad.render_state()

            a_cmds = agent.act(a_rs, Command) or [Command.observe()]
            e_cmds = enemy.act(e_rs, Command) or [Command.observe()]
            a_obs, e_obs, done, _info = raw.step_1v1(a_cmds, e_cmds)
            agent_ad.observe(a_obs, done=done)
            enemy_ad.observe(e_obs, done=done)

            agent_trace.append(
                {
                    "turn": turns,
                    "tick": agent_ad.signals.game_tick,
                    "n_cmds": len(a_cmds),
                }
            )
            enemy_trace.append(
                {
                    "turn": turns,
                    "tick": enemy_ad.signals.game_tick,
                    "n_cmds": len(e_cmds),
                }
            )
            if done:
                break

        # Decide the winner from the final boards.
        a_rs = agent_ad.render_state()
        e_rs = enemy_ad.render_state()
        agent_alive = _alive(a_rs)
        enemy_alive = _alive(e_rs)
        if agent_alive and not enemy_alive:
            winner, reason = "agent", "enemy base eliminated"
        elif enemy_alive and not agent_alive:
            winner, reason = "enemy", "agent base eliminated"
        elif not agent_alive and not enemy_alive:
            winner, reason = "draw", "mutual elimination"
        else:
            # Both standing — deadline / turn cap. Tie-break on economy.
            av, ev = _economy_value(a_rs), _economy_value(e_rs)
            if av > ev:
                winner, reason = "agent", "deadline — agent ahead on economy"
            elif ev > av:
                winner, reason = "enemy", "deadline — enemy ahead on economy"
            else:
                winner, reason = "draw", "deadline — even"

        return OneVOneResult(
            winner=winner,
            reason=reason,
            turns=turns,
            ticks=agent_ad.signals.game_tick,
            agent_name=getattr(agent, "name", "agent"),
            enemy_name=getattr(enemy, "name", "enemy"),
            agent_trace=agent_trace,
            enemy_trace=enemy_trace,
        )
    finally:
        pool.release(env)
        pool.shutdown()
