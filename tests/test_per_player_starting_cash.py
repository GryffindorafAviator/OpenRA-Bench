"""Per-player starting-cash plumbing: end-to-end via `openra_train`.

Pins the engine fix that the bench's `agent: {cash: N}` /
`enemy: {cash: M}` scenario-YAML knobs are now honoured per-player at
world build time. Before the fix the engine plumbed a single
`starting_cash: int` into ALL player actors, so an
`agent: {cash: 0}` + `enemy: {cash: 1500}` scenario gave both sides
the SAME cash (0), which broke scenarios like
`spec-thief-steal-cash` (the thief had nothing to steal).

Two sub-tests:
  * `test_agent_and_enemy_cash_independent` — both per-player
    overrides are present; the agent starts at 500 and the enemy at
    1500.
  * `test_back_compat_no_per_player_cash` — when the scenario omits
    `agent: {cash:}` and `enemy: {cash:}`, both slots inherit the
    top-level `starting_cash:` (the pre-fix behaviour, which the
    fix preserves as the default).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


def _scenario(
    actors: list[dict],
    *,
    starting_cash: int = 0,
    agent_cash: int | None = None,
    enemy_cash: int | None = None,
) -> dict:
    agent = {"faction": "allies"}
    if agent_cash is not None:
        agent["cash"] = agent_cash
    enemy = {"faction": "soviet"}
    if enemy_cash is not None:
        enemy["cash"] = enemy_cash
    return {
        "name": "per-player-cash-test",
        "description": "per-player starting-cash plumbing",
        "base_map": "rush-hour-arena",
        "starting_cash": starting_cash,
        "spawn_mcvs": False,
        "agent": agent,
        "enemy": enemy,
        "tools": ["observe"],
        "planning": True,
        "termination": {"max_ticks": 600},
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix="_per_player_cash.yaml", delete=False)
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def test_agent_and_enemy_cash_independent():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    # Minimal scene: one agent rifleman + one enemy rifleman so the
    # world builds cleanly. The cash assertions don't depend on
    # combat — we just snapshot tick 0.
    actors = [
        {"type": "e1", "owner": "agent", "position": [20, 20]},
        {"type": "e1", "owner": "enemy", "position": [60, 60]},
    ]
    path = _scenario_path(
        _scenario(actors, starting_cash=0, agent_cash=500, enemy_cash=1500)
    )
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        obs = env.reset(seed=1)
        ad.observe(obs)

        # The agent's own cash is surfaced as the top-level `cash`
        # field by RustObsAdapter.render_state() (mirrors the
        # `cash` key in the raw obs).
        agent_cash = int(ad.render_state().get("cash", -1))
        assert agent_cash == 500, (
            f"agent slot must honour `agent: {{cash: 500}}` "
            f"(got {agent_cash})"
        )

        # Cross-check enemy cash via the engine's `player_cash`
        # accessor exposed on the env handle. Available since
        # the per-player-cash fix shipped.
        # RustEnvHandle wraps the native `openra_train.OpenRAEnv`;
        # the per-player accessors live on the inner object.
        inner = env._env
        enemy_pid = inner.enemy_player_id
        enemy_cash = inner.player_cash(enemy_pid)
        assert enemy_cash == 1500, (
            f"enemy slot must honour `enemy: {{cash: 1500}}` "
            f"(got {enemy_cash})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_back_compat_no_per_player_cash():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    actors = [
        {"type": "e1", "owner": "agent", "position": [20, 20]},
        {"type": "e1", "owner": "enemy", "position": [60, 60]},
    ]
    # Neither agent.cash nor enemy.cash set ⇒ both slots inherit
    # the top-level 750 (back-compat with every existing pack).
    path = _scenario_path(_scenario(actors, starting_cash=750))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        agent_cash = int(ad.render_state().get("cash", -1))
        assert agent_cash == 750, (
            f"agent slot must inherit lobby `starting_cash: 750` "
            f"when `agent.cash` is omitted (got {agent_cash})"
        )
        # RustEnvHandle wraps the native `openra_train.OpenRAEnv`;
        # the per-player accessors live on the inner object.
        inner = env._env
        enemy_pid = inner.enemy_player_id
        enemy_cash = inner.player_cash(enemy_pid)
        assert enemy_cash == 750, (
            f"enemy slot must inherit lobby `starting_cash: 750` "
            f"when `enemy.cash` is omitted (got {enemy_cash})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
