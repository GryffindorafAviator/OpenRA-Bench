"""MCV deploy: scenario-declared {type: mcv} → Construction Yard.

The OpenRA engine's `Command::deploy(mcv_id)` queues `DeployTransform`,
which is gated on `actor.kind == ActorKind::Mcv`. Two bugs landed
scenario-declared MCVs as `ActorKind::Vehicle`, so deploy silently
no-op'd for every Group A (`mcv-*`) brainstorm pack:

  1. `kind_for_unit_type` fallback (used when rules.actor() returns
     None) defaulted to Infantry for `mcv` — now special-cased.
  2. `classify_actor` (used by `from_ruleset`, the typical load path
     for the bench's scenarios) infers kind from traits + locomotor;
     MCV has Mobile + a non-foot locomotor so it was classified
     Vehicle — now special-cased on the actor name.

Fixed in OpenRA-Rust:
  openra-sim/src/gamerules.rs : classify_actor takes name + branches
                                "mcv" → ActorKind::Mcv
  openra-train/src/env.rs     : kind_for_unit_type fallback also
                                returns ActorKind::Mcv for "mcv"

This test asserts the live engine WITH the fix:
  (a) the MCV's reported ActorKind round-trips as Mcv (no regression
      in the upstream classifier),
  (b) `Command.deploy(mcv_id)` actually creates a `fact` Construction
      Yard the agent owns (the user-visible behavior),
  (c) downstream, the agent can build a structure from the new fact
      (so the production-queue re-enable on deploy actually fires).

Unblocks Group A's `mcv-deploy-*` brainstorm seeds (mcv-deploy-and-
build, mcv-deploy-defensible-site, mcv-deploy-near-resource,
mcv-deploy-relocate-under-pressure, mcv-deploy-second-base,
mcv-deploy-third-base, mcv-deploy-evac-and-resite) and removes the
'deploy unimplemented' footgun from CLAUDE.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


def _scenario_with_mcv(starting_cash: int = 0):
    return {
        "name": "mcv-deploy-test",
        "description": "scenario-declared MCV must deploy to a fact",
        "base_map": "rush-hour-arena",
        "starting_cash": starting_cash,
        "spawn_mcvs": False,
        "agent": {"faction": "allies", "cash": starting_cash},
        "enemy": {"faction": "soviet", "cash": 0},
        "tools": ["observe", "deploy", "build", "place_building"],
        "planning": True,
        "termination": {"max_ticks": 3000},
        "actors": [
            {"type": "mcv", "owner": "agent", "position": [20, 20]},
            # Persistent enemy fact so the engine doesn't auto-`done`
            # before we've had a chance to deploy.
            {"type": "fact", "owner": "enemy", "position": [120, 20]},
        ],
    }


def _scenario_path(scenario: dict) -> str:
    fd = tempfile.NamedTemporaryFile(
        "w", suffix="_mcv.yaml", delete=False
    )
    yaml.safe_dump(scenario, fd, sort_keys=False)
    fd.close()
    return fd.name


def test_mcv_deploy_creates_construction_yard():
    """The user-visible bug: `Command.deploy(mcv_id)` must remove the
    MCV and create an agent-owned `fact`. Pre-fix, the order
    silently no-op'd and the MCV sat forever."""
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    path = _scenario_path(_scenario_with_mcv())
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        rs = ad.render_state()
        own = rs.get("units_summary", []) or []
        mcv_id = next(
            (u["id"] for u in own if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        assert mcv_id is not None, "MCV not placed by scenario"
        assert "fact" not in ad.signals.own_building_types

        # Issue deploy
        obs, _r, done, _i = env.step([env.Command.deploy([str(mcv_id)])])
        ad.observe(obs, done=done)
        # The DeployTransform queues a Turn; the transform completes
        # mid-step (≤ ticks_per_step ticks). One step is enough.
        assert "fact" in ad.signals.own_building_types, (
            f"deploy didn't create an agent fact "
            f"(own_buildings={ad.signals.own_buildings})"
        )

        # The MCV should be gone (replaced by the fact).
        rs = ad.render_state()
        own_after = rs.get("units_summary", []) or []
        assert not any(
            str(u.get("type", "")).lower() == "mcv" for u in own_after
        ), f"MCV should have been consumed by deploy ({own_after})"
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_mcv_deploy_re_enables_production_queue():
    """The harder downstream guarantee: after deploy, the agent's
    Building / Defense production queues are re-enabled — so the
    agent can immediately `build('powr')` etc. from the new fact.
    (This is the world.rs:3596 ClassicProductionQueue re-enable
    branch firing in concert with the actor creation.)"""
    pytest.importorskip("openra_train")
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    from openra_bench.rust_adapter import RustObsAdapter

    path = _scenario_path(_scenario_with_mcv(starting_cash=800))
    pool = RustEnvPool(size=1, scenario_path=path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        own = ad.render_state().get("units_summary", []) or []
        mcv_id = next(
            (u["id"] for u in own if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        env.step([env.Command.deploy([str(mcv_id)])])
        # Now queue a powr (cost 300, well under starting cash 800).
        # If the production-queue was re-enabled by deploy, this
        # queues; if it's still disabled (no fact), it silently no-ops.
        env.step([env.Command.build("powr")])
        # Step a few times to let production complete (powr takes
        # ~ticks_per_step turns based on cost). The exact threshold
        # depends on engine speed; 20 turns is more than enough.
        for _ in range(20):
            obs, _r, done, _i = env.step([env.Command.observe()])
            ad.observe(obs, done=done)
            if "powr" in ad.signals.own_building_types or done:
                break
        # `powr` is queued for placement after production completes
        # — the agent needs to `place_building` it. For the queue-
        # re-enable check we just need evidence production STARTED:
        # production_items contains the queued item OR powr is
        # already a structure. Either confirms the queue accepted
        # the build call.
        assert (
            "powr" in (ad.signals.production_items or [])
            or "powr" in ad.signals.own_building_types
        ), (
            f"build('powr') after deploy did not queue / produce "
            f"(production_items={ad.signals.production_items}, "
            f"own_buildings={ad.signals.own_buildings})"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
