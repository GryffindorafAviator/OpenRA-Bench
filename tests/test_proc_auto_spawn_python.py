"""End-to-end guardrail: a 2nd `proc` auto-spawns its `harv` adjacent
to the NEW proc (not piled on top of the lowest-id proc), and the
fresh harv picks the path-shortest refinery as its delivery target.

Historical footgun (closed by the matching engine commit):
  * `order_place_building` called `spawn_unit("harv", owner)`, which
    routed through `find_spawn_location` — that helper sorts
    production-building candidates by `(!is_primary, id)`, so the
    auto-harv always materialised next to the LOWEST-id proc, never
    the new one. A 2nd refinery placed far from the 1st gained no
    throughput from its own auto-harv.
  * `find_refinery` returned the first (lowest-id) `proc`, so every
    harv deposited at the closest by ID, not by path-distance. Adding
    a refinery near a contested patch never paid off.

Mirror of `OpenRA-Rust/openra-sim/tests/test_proc_auto_spawn_at_new_proc.rs`,
exercised here via the Python `OpenRAEnv` boundary so the bench-side
adapter is pinned too.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


def _scenario(actors, *, agent_cash: int = 5000) -> dict:
    return {
        "name": "proc-auto-spawn-test",
        "description": "engine guardrail: 2nd proc auto-harv lands near the NEW proc",
        "base_map": "rush-hour-arena",
        "starting_cash": agent_cash,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": agent_cash},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "build", "place_building"],
        "planning": True,
        "termination": {"max_ticks": 12000},
        "actors": actors,
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_proc_spawn.yaml", delete=False
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def test_second_proc_autospawns_harv_at_the_new_proc_via_python_env():
    pytest.importorskip("openra_train")
    pytest.importorskip("openra_rl_training")
    from openra_train import Command
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    # Pre-place a small base on the WEST: fact + powr + 1st proc.
    # Enemy gets a single far rifleman so the world has a valid
    # opponent footprint (no draw-on-elim).
    actors = [
        {"type": "fact", "owner": "agent", "position": [10, 10]},
        {"type": "powr", "owner": "agent", "position": [14, 10]},
        {"type": "powr", "owner": "agent", "position": [16, 10]},
        {"type": "proc", "owner": "agent", "position": [10, 14]},
        {"type": "e1", "owner": "enemy", "position": [90, 90]},
    ]
    path = _scenario_path(_scenario(actors, agent_cash=8000))

    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))

        # Pre-place harv id snapshot (so we can identify the new one
        # by id-order later).
        own_units0 = ad.render_state().get("units_summary", []) or []
        pre_harv_ids = {
            int(u["id"])
            for u in own_units0
            if str(u.get("type", "")).lower() == "harv"
        }

        # Queue the 2nd proc. cost ≈ 1400; one Build call enqueues
        # one item, then we step until done.
        ad.observe(env.step([Command.build("proc")])[0])

        # Step until the proc completes (production tick uses
        # ~3 frames per process_frame; loop until the raw observation
        # surfaces the proc item with `done=True`, or budget
        # exhausted). render_state's `production` is collapsed to
        # item strings; we reach for the raw obs via the env handle.
        done_proc = False
        for _ in range(120):
            obs, _r, _d, _i = env.step([Command.observe()])
            ad.observe(obs)
            raw_prod = obs.get("production", []) or []
            if any(
                isinstance(p, dict)
                and str(p.get("item", "")).lower() == "proc"
                and bool(p.get("done", False))
                for p in raw_prod
            ):
                done_proc = True
                break
        assert done_proc, (
            "2nd proc must be completed in the production queue before "
            "place_building can fire"
        )

        # Place the 2nd proc FAR EAST.
        east_x, east_y = 70, 14
        ad.observe(env.step([Command.place_building("proc", east_x, east_y)])[0])
        # Step one more to fire the SpawnUnit frame-end task.
        ad.observe(env.step([Command.observe()])[0])

        rs = ad.render_state()
        own_units = rs.get("units_summary", []) or []
        harvs = [
            u
            for u in own_units
            if str(u.get("type", "")).lower() == "harv"
        ]
        new_harvs = [u for u in harvs if int(u["id"]) not in pre_harv_ids]
        own_b = rs.get("own_buildings", []) or []
        procs = [b for b in own_b if str(b.get("type", "")).lower() == "proc"]
        assert len(procs) >= 2, (
            f"expected ≥2 procs after place_building; got own_buildings={own_b}"
        )
        assert len(new_harvs) >= 1, (
            "placing a 2nd proc must auto-spawn a 2nd harv "
            f"(pre={pre_harv_ids}, "
            f"post_harvs={[(u['id'], u['cell_x'], u['cell_y']) for u in harvs]}, "
            f"procs={procs})"
        )
        new_harv = new_harvs[0]
        hx, hy = int(new_harv["cell_x"]), int(new_harv["cell_y"])

        # Chebyshev distance to the NEW (east) vs OLD (west) proc.
        cheb_east = max(abs(hx - east_x), abs(hy - east_y))
        cheb_west = max(abs(hx - 10), abs(hy - 14))
        assert cheb_east <= 3, (
            f"new harv must spawn within 3 cells of the NEW (east) proc; "
            f"harv at ({hx},{hy}), east proc at ({east_x},{east_y}), "
            f"Chebyshev distance={cheb_east}"
        )
        assert cheb_east < cheb_west, (
            f"new harv must be CLOSER to the new (east) proc than to the "
            f"old (west) proc; east={cheb_east} west={cheb_west}"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
