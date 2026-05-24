"""Engine `termination.{agent,enemy}_units_killed` flags — end-to-end
across the Python boundary.

Pins the engine fix that scenario-YAML termination flags now gate the
auto-`done` paths. Before the fix the flags were silently dropped by
`oramap::parse_scenario_yaml` (no consumer), so a sacrifice / decoy
pack (e.g. `combat-suicide-charge-mission`) that documented
`termination.agent_units_killed: false` still saw the engine end the
run the moment the agent's last unit died — the within_ticks fail
predicate never got a chance to fire, collapsing every wipe to DRAW.

Two sub-tests:
  * `test_default_agent_wipe_ends_run` — back-compat path: with no
    `termination:` block (or with the flag explicitly `true`), an
    agent-side wipe ends the run.
  * `test_agent_units_killed_false_keeps_run_alive_past_wipe` — the
    opt-out: with `termination.agent_units_killed: false`, an
    agent-side wipe does NOT end the run. The episode advances until
    the deadline (or until a fail/win predicate fires bench-side).
  * Mirror pair for `enemy_units_killed`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


def _scenario(
    actors: list[dict],
    *,
    max_ticks: int = 40000,
    agent_units_killed: bool | None = None,
    enemy_units_killed: bool | None = None,
) -> dict:
    termination: dict = {"max_ticks": max_ticks}
    if agent_units_killed is not None:
        termination["agent_units_killed"] = agent_units_killed
    if enemy_units_killed is not None:
        termination["enemy_units_killed"] = enemy_units_killed
    return {
        "name": "termination-flag-smoke",
        "description": "termination flag end-to-end",
        "base_map": "rush-hour-arena",
        "starting_cash": 0,
        "spawn_mcvs": False,
        "agent": {"faction": "allies"},
        "enemy": {"faction": "soviet"},
        "tools": ["observe"],
        "planning": True,
        "termination": termination,
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix="_termination_flags.yaml", delete=False)
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _run_with_observe(env, max_steps: int) -> tuple[int, bool]:
    """Step `env` with a list of one Observe command until done or
    `max_steps` exhausted. Returns (steps_taken, done)."""
    import openra_train

    cmd = openra_train.Command.observe()
    for i in range(max_steps):
        _obs, _r, done, _info = env.step([cmd])
        if done:
            return i, True
    return max_steps, False


# A fragile e1 next to a 4tnk wipes within a few combat ticks. Used
# to exercise the agent-wipe auto-`done` gate.
_AGENT_WIPE_ACTORS = [
    {"type": "e1", "owner": "agent", "position": [20, 20]},
    {"type": "4tnk", "owner": "enemy", "position": [21, 20]},
]


# A 4tnk vs. a passive (stance:0) e1 — the enemy is wiped quickly.
# Used to exercise the enemy-wipe auto-`done` gate.
_ENEMY_WIPE_ACTORS = [
    {"type": "4tnk", "owner": "agent", "position": [20, 20]},
    {"type": "e1", "owner": "enemy", "position": [21, 20], "stance": 0},
]


def test_default_agent_wipe_ends_run():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    # No explicit flag ⇒ engine default true ⇒ agent wipe ends run.
    path = _scenario_path(_scenario(_AGENT_WIPE_ACTORS, max_ticks=40000))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        env.reset(seed=7)
        steps, done = _run_with_observe(env, max_steps=50)
        assert done, (
            f"default agent_units_killed:true must auto-`done` on agent wipe "
            f"(ran {steps} steps without terminating)"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_agent_units_killed_false_keeps_run_alive_past_wipe():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    # Opt-out: agent wipe must NOT end the run.
    path = _scenario_path(
        _scenario(_AGENT_WIPE_ACTORS, max_ticks=40000, agent_units_killed=False)
    )
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        env.reset(seed=7)
        # 300 step.()s × 30 ticks/step = 9000 ticks, well past the
        # ~10 combat ticks needed to wipe the e1 vs. a 4tnk. The run
        # MUST still be alive (default max_ticks=10000 — bumped via
        # the scenario's `termination.max_ticks: 40000`, but the
        # engine's separate cap is still `with_max_ticks` not set
        # here; that's fine because we explicitly check < that cap).
        # Run ~50 steps × ~90 ticks/step ≈ 4500 ticks. The engine's
        # default hard cap is 10000 (DEFAULT_MAX_TICKS), so we stay
        # comfortably below it — anything terminating before then is
        # the auto-`done` path we're suppressing. The agent e1 dies
        # to the adjacent 4tnk within a handful of combat ticks; if
        # the agent-wipe gate were still active, `done` would fire
        # by step 5-ish.
        steps, done = _run_with_observe(env, max_steps=50)
        assert not done, (
            f"agent_units_killed:false MUST keep the run alive past agent wipe "
            f"(terminated at step {steps})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_default_enemy_wipe_ends_run():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    path = _scenario_path(_scenario(_ENEMY_WIPE_ACTORS, max_ticks=40000))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        env.reset(seed=7)
        steps, done = _run_with_observe(env, max_steps=50)
        assert done, (
            f"default enemy_units_killed:true must auto-`done` on enemy wipe "
            f"(ran {steps} steps without terminating)"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_enemy_units_killed_false_keeps_run_alive_past_wipe():
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    path = _scenario_path(
        _scenario(_ENEMY_WIPE_ACTORS, max_ticks=40000, enemy_units_killed=False)
    )
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        env.reset(seed=7)
        # ~50 steps stays comfortably under the engine's default
        # 10000-tick cap; the stance:0 e1 is dead within ~5 combat
        # ticks vs. the 4tnk, so anything > 10 steps proves the
        # enemy-wipe auto-done is suppressed.
        steps, done = _run_with_observe(env, max_steps=50)
        assert not done, (
            f"enemy_units_killed:false MUST keep the run alive past enemy wipe "
            f"(terminated at step {steps})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
