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
    # adversarial-skirmish/-siege consolidated into adversarial-duel
    # (quarantined) — see SCENARIO_QUALITY.md de-dup.
    "artofwar-decoy-sacrifice",
    "artofwar-indirect-approach",
    "artofwar-lure-the-tiger",
    "artofwar-sequenced-citadel",
    "action-sequenced-execution",
    "coordination-staggered-window",
    "harass-response-preserve",
    "strict-sequence",
    "perception-count-the-threat",
    "perception-frontier-reading",
    "perception-target-vs-fog",
    "reasoning-frontier-commit",
    "rush-hour",
    "custom-map-no-enemy",
    "tempo-double-window",
    "economy-harvest-timebox",
    "strategy-trilemma",
    "defense-rush-survive",
    "navigation-confined-hard-only",
    # Rebuilt post-S0/S1 harvest income (Task #14); hard tier defines
    # ≥2 symmetric spawn_point groups around the near patch so the
    # DEEP throughput is equal on each spawn (no opening can be
    # memorised across seeds).
    "economy-harvest-investment",
    "mid-concede-vs-hold",
]

# Consciously NOT spawn-varied, with the reason (keeps the curation
# exhaustive — every active pack is classified, see the coverage test).
NOT_APPLICABLE = {
    "economy-investment": "non-spatial: capital allocation, start pos irrelevant",
    "economy-time-box": "non-spatial: budget-under-clock",
    "economy-force-buildup": "non-spatial: production economy",
    "building-and-planning": "non-spatial: build-order/tech, fixed base",
    "strict-production-bom": "non-spatial: exact bill-of-materials spec",
    "tech-production-planning": "non-spatial: build-order dependency "
    "planning (precedence + power + budget); start position irrelevant",
    "longhorizon-opening-to-assault": "fixed pre-seeded base; the test "
    "is the scout→tech→army→strike phase chain within one budget, not "
    "start-position generalization (single base by design)",
    "reasoning-risk-route": "rigor 5/5 from one tuned safe seam — varying "
    "the start would break the single-solution tuning / seed parity",
    "strategy-dilemma": "win redesigned to destroy fact+proc (faithful "
    "to training); spawn deferred — route-choice puzzle is the decision",
    "strategy-gauntlet": "win redesigned to fact+proc; single defended "
    "corridor — spawn variation would not add a distinct decision",
    "strategy-twobody": "win redesigned to fact+proc; two "
    "simultaneously-controlled groups IS the task — spawn-alternatives "
    "would break intent",
    "action-multiunit-coordination": "hard is held byte-identical to "
    "medium's setup by design; the SOLE controlled variable vs medium "
    "is objective_coords:relative (spatial grounding from the minimap "
    "instead of handed coordinates). Adding seed-driven spawn variation "
    "would introduce a second uncontrolled variable and break the clean "
    "medium→hard attribution.",
}

# No-adversary maps: spawn variation applies but a force-loss
# fail_condition is impossible (nothing can destroy the force).
_NO_ENEMY = {"strict-sequence", "custom-map-no-enemy"}


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
    # degeneracy) — except no-adversary maps where force-loss is
    # impossible by construction (documented in _NO_ENEMY).
    for pid in UPGRADED:
        c = compile_level(load_pack(PACKS / f"{pid}.yaml"), "hard")
        if pid in _NO_ENEMY:
            continue
        assert c.fail_condition is not None, f"{pid}:hard needs a fail_condition"


def test_every_active_pack_is_classified():
    """Curation is exhaustive: every active pack is either spawn-varied
    (UPGRADED) or consciously NOT_APPLICABLE with a stated reason — no
    pack silently skipped."""
    import glob
    import os

    classified = set(UPGRADED) | set(NOT_APPLICABLE)
    missing = []
    for f in glob.glob(str(PACKS / "*.yaml")):
        b = os.path.basename(f)
        if b.startswith(("_", "TEMPLATE")):
            continue
        m = load_pack(f).meta
        if m.status != "active":
            continue
        if m.id not in classified:
            missing.append(m.id)
    assert not missing, f"unclassified active packs (curate or mark N/A): {missing}"
