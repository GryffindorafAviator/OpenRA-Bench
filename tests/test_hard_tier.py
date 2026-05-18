"""Hard-tier curation invariant.

The `hard` level of a spatial pack must be *qualitatively* harder, not
number-inflated: it must define ≥2 distinct agent `spawn_point` groups
so `Env` round-robins the start position by seed (a single memorised
opening can't generalise — this is what the held-out-seed /
generalization-gap metric is there to reward). `UPGRADED` grows as
packs are curated, so the suite stays green per commit while the
contract is enforced on everything already done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"

# Packs whose `hard` tier has been curated to the spawn-variation
# contract. Append as each is done (see the applicability matrix —
# economy-*/building-and-planning/strict-production-bom are N/A).
UPGRADED = [
    "adversarial-duel",
    "adversarial-skirmish",
    "adversarial-siege",
]


def _agent_spawn_points(pack_id: str, level: str) -> set:
    c = compile_level(load_pack(PACKS / f"{pack_id}.yaml"), level)
    return {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }


@pytest.mark.parametrize("pid", UPGRADED)
def test_hard_has_multiple_seed_driven_spawn_points(pid):
    sp = _agent_spawn_points(pid, "hard")
    assert len(sp) >= 2, (
        f"{pid}:hard must define ≥2 agent spawn_point groups for "
        f"seed-driven start variation; got {sorted(sp)}"
    )


@pytest.mark.parametrize("pid", UPGRADED)
def test_curated_hard_still_compiles_and_runs(pid):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACKS / f"{pid}.yaml"), "hard")
    assert c.map_supported
    # Different seeds must actually place the agent differently (the
    # whole point of multiple spawn_point groups).
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_bench.eval_core import _scenario_to_tmp_yaml, RustEnvPool

    starts = set()
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"{pid}:hard seeds produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


def test_fail_condition_present_on_curated_hard():
    # Curated hard tiers must be able to emit a loss (no loss==draw
    # degeneracy on the packs we've touched).
    for pid in UPGRADED:
        c = compile_level(load_pack(PACKS / f"{pid}.yaml"), "hard")
        assert c.fail_condition is not None, f"{pid}:hard needs a fail_condition"
