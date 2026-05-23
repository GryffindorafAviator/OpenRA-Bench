"""Spy / thief Infiltrate: engine order end-to-end via `openra_train`.

Two sub-tests mirror the Rust integration tests:

  * `test_spy_reveals_enemy_buildings_via_infiltrate` — a `spy`
    pre-placed adjacent to an enemy `proc` walks in; after the order
    the `enemy_buildings_summary` reports the enemy's structures even
    when their cells are out of the agent's natural sight.
  * `test_thief_steals_cash_via_infiltrate` — a `thf` pre-placed
    adjacent to an enemy `silo` walks in; the agent's cash rises by
    the stolen amount.

Both sub-tests `pytest.skip` gracefully if the installed `openra_train`
wheel pre-dates the `Command.infiltrate` static constructor — the
worktree's engine change ships ahead of the wheel rebuild during a
parallel-agent batch.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


def _scenario(actors: list[dict], starting_cash: int = 0, enemy_cash: int = 0) -> dict:
    return {
        "name": "infiltrate-test",
        "description": "spy / thief Infiltrate order",
        "base_map": "rush-hour-arena",
        "starting_cash": starting_cash,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": starting_cash},
        "enemy": {"faction": "soviet", "cash": enemy_cash},
        "tools": ["observe", "infiltrate"],
        "planning": True,
        "termination": {"max_ticks": 3000},
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile("w", suffix="_infiltrate.yaml", delete=False)
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def _have_infiltrate() -> bool:
    try:
        import openra_train  # noqa: F401
    except Exception:
        return False
    return hasattr(openra_train.Command, "infiltrate")


def test_spy_reveals_enemy_buildings_via_infiltrate():
    pytest.importorskip("openra_train")
    if not _have_infiltrate():
        pytest.skip(
            "installed openra_train wheel predates Command.infiltrate; "
            "rebuild with `maturin develop --release` to enable"
        )
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    # Spy adjacent to one enemy proc; a second enemy structure placed
    # far away (out of the spy's natural sight) verifies the one-shot
    # scan reveals ALL of the target-owner's buildings, not just the
    # one the spy entered.
    actors = [
        {"type": "spy", "owner": "agent", "position": [20, 20]},
        {"type": "proc", "owner": "enemy", "position": [21, 20]},
        {"type": "powr", "owner": "enemy", "position": [110, 90]},
    ]
    path = _scenario_path(_scenario(actors))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        own = ad.render_state().get("units_summary", []) or []
        spy_id = next(
            (u["id"] for u in own if str(u.get("type", "")).lower() == "spy"), None
        )
        assert spy_id is not None, "spy not placed by scenario"

        # Pick the enemy proc id from the initial observation (it is
        # adjacent so the spy already sees it).
        rs = ad.render_state()
        enemy = rs.get("enemy_buildings_summary", []) or []
        proc_id = next(
            (b["id"] for b in enemy if str(b.get("type", "")).lower() == "proc"), None
        )
        assert proc_id is not None, "enemy proc must be visible to spy at start"

        obs, _r, done, _i = env.step(
            [env.Command.infiltrate([str(spy_id)], str(proc_id))]
        )
        ad.observe(obs, done=done)
        # Drive a handful of turns so the consume + reveal happens.
        for _ in range(3):
            obs, _r, done, _i = env.step([env.Command.observe()])
            ad.observe(obs, done=done)
            if done:
                break

        # Spy must be gone (consumed).
        own_after = ad.render_state().get("units_summary", []) or []
        assert not any(
            str(u.get("type", "")).lower() == "spy" for u in own_after
        ), f"spy must have been consumed ({own_after})"

        # The far-away `powr` (out of natural sight) must now appear in
        # `enemy_buildings_summary` thanks to the spy reveal scan.
        rs_after = ad.render_state()
        enemy_after = rs_after.get("enemy_buildings_summary", []) or []
        types = {str(b.get("type", "")).lower() for b in enemy_after}
        assert "powr" in types, (
            f"distant enemy `powr` should be revealed by spy infiltration "
            f"(enemy_buildings_summary={enemy_after})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_thief_steals_cash_via_infiltrate():
    pytest.importorskip("openra_train")
    if not _have_infiltrate():
        pytest.skip(
            "installed openra_train wheel predates Command.infiltrate; "
            "rebuild with `maturin develop --release` to enable"
        )
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    actors = [
        {"type": "thf", "owner": "agent", "position": [20, 20]},
        {"type": "silo", "owner": "enemy", "position": [21, 20]},
    ]
    path = _scenario_path(_scenario(actors, starting_cash=0, enemy_cash=2000))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        own = ad.render_state().get("units_summary", []) or []
        thf_id = next(
            (u["id"] for u in own if str(u.get("type", "")).lower() == "thf"), None
        )
        assert thf_id is not None, "thief not placed by scenario"

        enemy = ad.render_state().get("enemy_buildings_summary", []) or []
        silo_id = next(
            (b["id"] for b in enemy if str(b.get("type", "")).lower() == "silo"), None
        )
        assert silo_id is not None, "enemy silo must be visible to thief at start"

        cash_before = int(ad.render_state().get("economy", {}).get("cash", 0))

        obs, _r, done, _i = env.step(
            [env.Command.infiltrate([str(thf_id)], str(silo_id))]
        )
        ad.observe(obs, done=done)
        for _ in range(3):
            obs, _r, done, _i = env.step([env.Command.observe()])
            ad.observe(obs, done=done)
            if done:
                break

        own_after = ad.render_state().get("units_summary", []) or []
        assert not any(
            str(u.get("type", "")).lower() == "thf" for u in own_after
        ), f"thief must have been consumed ({own_after})"

        cash_after = int(ad.render_state().get("economy", {}).get("cash", 0))
        assert cash_after > cash_before, (
            f"thief infiltration must increase agent cash "
            f"(before={cash_before}, after={cash_after})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
