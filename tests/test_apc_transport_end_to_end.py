"""End-to-end guardrail: APC transport (Command.enter_transport →
APC drives → Command.unload) with infantry alive on the far side.

Pins the full Python-side loop:
  1. An e1 issues `enter_transport(apc)` and walks adjacent → boards
     (passenger is removed from the world and stashed in transport
     cargo).
  2. The APC moves ~30 cells east via `move_units`.
  3. `unload(apc)` ejects the passenger onto an adjacent passable
     cell; the infantry is alive and observable in the visible-units
     list at the new (east) location.

The Rust side has direct unit tests for each leg; this is the
bench-side mirror that proves Python `Command.enter_transport` /
`Command.move_units` / `Command.unload` actually drive the loop
through the env boundary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

# Absolute path to the bundled rush-hour terrain. See test_aa_fires_on_
# aircraft.py for the rationale — older tests relied on the engine's
# HOME-dir fallback to OpenRA-RL-Training, which does not exist on CI.
_BUNDLED_MAP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "maps"
    / "rush-hour-arena.oramap"
)


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_apc_transport.yaml", delete=False
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def test_apc_transport_loads_drives_and_unloads_infantry_alive():
    pytest.importorskip("openra_train")
    pytest.importorskip("openra_rl_training")
    from openra_train import Command
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    # APC at (20, 20), one passenger riflemen one cell south, and a
    # far enemy actor so the world has a valid opponent footprint.
    actors = [
        {"type": "apc", "owner": "agent", "position": [20, 20]},
        {"type": "e1", "owner": "agent", "position": [21, 20]},
        {"type": "e1", "owner": "enemy", "position": [90, 90]},
    ]
    scenario = {
        "name": "apc-transport-test",
        "description": "engine guardrail: APC enter_transport/move/unload",
        "base_map": str(_BUNDLED_MAP),
        "starting_cash": 0,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": 0},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "move_units", "enter_transport", "unload"],
        "planning": True,
        "termination": {"max_ticks": 20000},
        "actors": actors,
    }
    path = _scenario_path(scenario)

    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        rs0 = ad.render_state()
        own0 = rs0.get("units_summary", []) or []
        apc_id = None
        e1_id = None
        for u in own0:
            t = str(u.get("type", "")).lower()
            if t == "apc" and apc_id is None:
                apc_id = str(u["id"])
            elif t == "e1" and e1_id is None:
                e1_id = str(u["id"])
        assert apc_id is not None and e1_id is not None, (
            f"need both APC and e1 in initial units_summary; got {own0}"
        )

        # 1. enter_transport: e1 boards the APC.
        ad.observe(env.step([Command.enter_transport([e1_id], apc_id)])[0])
        boarded = False
        for _ in range(80):
            obs, _r, done, _i = env.step([Command.observe()])
            ad.observe(obs)
            cur = ad.render_state().get("units_summary", []) or []
            # The passenger is removed from the world (stashed as
            # cargo) when it boards. Detect by the e1 disappearing.
            still_there = any(str(u["id"]) == e1_id for u in cur)
            if not still_there:
                boarded = True
                break
            if done:
                break
        assert boarded, (
            f"e1 ({e1_id}) must board the APC ({apc_id}) within the step "
            f"budget; current units={ad.render_state().get('units_summary')}"
        )

        # 2. APC drives ~30 cells east via move_units.
        dest_x, dest_y = 55, 20
        ad.observe(env.step([Command.move_units([apc_id], dest_x, dest_y)])[0])
        arrived = False
        for _ in range(200):
            obs, _r, done, _i = env.step([Command.observe()])
            ad.observe(obs)
            cur = ad.render_state().get("units_summary", []) or []
            apc = next((u for u in cur if str(u["id"]) == apc_id), None)
            if apc is None:
                break
            if abs(int(apc["cell_x"]) - dest_x) <= 2 and abs(int(apc["cell_y"]) - dest_y) <= 2:
                arrived = True
                break
            if done:
                break
        assert arrived, (
            f"APC must move within 2 cells of ({dest_x},{dest_y}) within the "
            f"step budget; final units={ad.render_state().get('units_summary')}"
        )

        # 3. unload: passenger ejected onto a passable cell adjacent
        #    to the APC; visible in units_summary again.
        ad.observe(env.step([Command.unload([apc_id])])[0])
        unloaded = False
        for _ in range(20):
            obs, _r, done, _i = env.step([Command.observe()])
            ad.observe(obs)
            cur = ad.render_state().get("units_summary", []) or []
            e1s = [u for u in cur if str(u.get("type", "")).lower() == "e1"]
            for u in e1s:
                # Must be near the APC's destination, not near the
                # initial (20,20) position — proves the APC actually
                # carried the unit east.
                cx, cy = int(u["cell_x"]), int(u["cell_y"])
                if abs(cx - dest_x) <= 3 and abs(cy - dest_y) <= 3:
                    unloaded = True
                    break
            if unloaded or done:
                break
        assert unloaded, (
            f"unload must eject the e1 near the APC's new location "
            f"({dest_x},{dest_y}); final units={ad.render_state().get('units_summary')}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
