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
    # First pack to USE the Wave-2 MCV deploy fix: hard tier flips the
    # MCV's start corner (NW vs SW) per seed, so the deploy site and
    # the patrol's relative defensibility differ across seeds.
    "mcv-deploy-and-build",
    # B1 — live ore-patch defense vs Wave 2 raider bot. Hard tier
    # round-robins agent base between a north-flank (y=14..22) and
    # south-flank (y=18..26) staging, both covering the same patch
    # column against three raider vectors (north/central/south).
    "mid-economy-under-fire",
    # Group A seed: hard tier stages the spare MCV NE-leaning vs
    # SE-leaning by seed; the win predicate accepts a deploy at
    # either candidate region (NE or SE) so the model must pick the
    # NEAREST one — cross-map diagonal blows the tick budget.
    "mcv-deploy-second-base",
    # Group A seed: hard tier seed-varies the FRESH MCV's start
    # column (40,20 vs 60,20) with the matching safe shoulders, so a
    # single memorised relocation cell cannot generalise across seeds.
    "mcv-deploy-relocate-under-pressure",
    # Group A econ-first siting seed (Wave 2 MCV deploy fix + post-S0/S1
    # harvest income); hard tier defines 2 symmetric spawn_point groups
    # (north/south base) so each seed gets its own near-patch geometry.
    "mcv-deploy-near-resource",
    # Wave-2 MCV deploy + tri-region operational footprint: 3 MCVs
    # must each deploy at a DISTINCT target region (NE/SE/S-CENTER);
    # hard tier rotates the WHOLE MCV cluster between WEST and EAST
    # starts so the per-MCV nearest-region assignment flips per seed.
    "mcv-deploy-third-base",
    # Wave-2 MCV deploy + threat-axis site choice: hard tier round-
    # robins the MCV start latitude (NW vs SW) per seed; agent must
    # IDENTIFY the safe corner from each start, not memorise a path.
    "mcv-deploy-defensible-site",
    # B2 reasoning pack (Wave-2 then-composite). Hard defines two
    # agent spawn_point groups (NORTH base / SOUTH base) round-
    # robined by seed — the "near" enemy composition flips so a
    # memorised pre-pick from one spawn cannot generalise.
    "mid-tech-switch-on-scout",
    # Wave-4 Group B AGGRO triple expansion (SC2 greedy 3-base macro /
    # startup blitzscale anchor). Three MCVs at a NW staging zone must
    # commit to three DISTINCT eastern target regions; the AGGRO play
    # is to deploy ALL THREE immediately and skip defence. Hard tier
    # rotates the MCV cluster between NW and SW spawns so the
    # per-MCV nearest-region assignment flips per seed.
    "expansion-aggro-3-base-greedy",
    # Wave-4 perception pack: scout reveals the eastern fact AND
    # returns alive to the start region. Hard defines two agent
    # spawn_point groups (NORTH (10,8) / SOUTH (10,32)) round-
    # robined by seed; the win clause is `any_of` over the two
    # matching corner-return regions so a scout must return to
    # its OWN spawn corner.
    "scout-and-survive",
    # Wave-4 AGGRO node of the harvester-harass triple. Hard tier
    # rotates the raider strike force between two west-edge
    # corridors (north y=10..14 vs south y=26..30) so the
    # engagement geometry against the symmetric enemy garrison
    # (two 3tnk defenders + four harvs around proc 80,20) varies
    # per seed.
    "combat-harass-aggro-commit",
    # Group F econ reasoning (Wave-4): Weber multi-source / SC2
    # mineral-patch allocation. Hard defines two agent spawn_point
    # groups (NORTH base y=14 / SOUTH base y=28) round-robined by
    # seed; the four neutral mines stay fixed but the NEAREST patch
    # flips per seed ((16,14) for NORTH, (16,28) for SOUTH), so a
    # memorised "always send to (16,14)" cannot generalise.
    "econ-multi-patch-allocation",
    # Group F opening greenfield seed (Wave-4): cold-start from a
    # single MCV with no buildings / no harvester / no income. Hard
    # tier defines 2 agent spawn_point groups (NORTH (20,14) / SOUTH
    # (20,26)) each with its OWN local ore patch ((26,14)/(26,26))
    # so the deploy site + near-patch geometry flips per seed and a
    # memorised "(20,20) deploy" opening cannot generalise.
    "econ-startup-from-scratch",
    # Wave-4 BALANCED tech-triple pack. Hard defines two agent
    # spawn_point groups (NORTH base / SOUTH base) so the harv-to-
    # proc geometry differs per seed; a memorised opening cannot
    # generalise.
    "tech-balanced-econ-then-tech",
    # Wave-4 BALANCED node of the expansion triple (SC2 standard
    # 2-base macro / sustainable-growth anchor). NW starter base +
    # spare MCV at centre; balanced play is deploy + pbox at EACH
    # base + 3 defenders. Hard rotates the spare MCV's starting
    # column between NW-leaning (drives to NE candidate) and SW-
    # leaning (drives to SE candidate) so the per-seed nearest-
    # target assignment flips.
    "expansion-balanced-2-base-defended",
    # Group F replan-after-loss seed (Wave-4): PlanBench replanning
    # / SC2 replace-killed-workers idiom. The agent starts with 3
    # harvs + indivisible reserve; a raider 4tnk kills 1 harv early
    # (medium) or 2 staggered on hard; agent must build('harv') and
    # re-issue `harvest`. Hard defines 2 agent spawn_point groups
    # (NORTH base y=14 / SOUTH base y=26) with a SHARED central patch
    # geometry, so the replanning task is symmetric per spawn but a
    # memorised base opening cannot generalise.
    "econ-replace-dead-harvester",
    # Wave-4 salvaged-from-worktree packs: agents finished authoring
    # and validating but ran out of usage budget mid-push; rescued
    # YAML+tests from their worktree branches and committed from the
    # main session.
    "lh-opening-to-tech-to-army",
    "lh-tech-pivot-attack",
    "mfb-tech-base-vs-economy-base",
    "combat-harass-balanced-hit-and-run",
    "tech-aggro-all-in",
    # Wave-4 Group D combat-micro pack: SC2 focus-fire / MicroRTS
    # target prioritization / military strike-package doctrine. Hard
    # tier rotates the strike force between two west-edge corridors
    # (NORTH y=10..13 / SOUTH y=26..29) round-robined by seed; the
    # centre enemy squad (2× e3 + 4× e1) is symmetric across y=20 so
    # either spawn faces an equivalent focus-fire decision.
    "combat-focus-fire-priority",
    # Wave-4 Group B secure-expand pack: defend base #1 (under a
    # 2-grenadier `rusher` raid) WHILE deploying a fresh MCV at the
    # eastern target region. Hard tier round-robins the fresh MCV's
    # start latitude (NORTH (30,15) vs SOUTH (30,45)) per seed and
    # the win clause accepts EITHER candidate target region (NE
    # (130,15) or SE (130,45)), so the nearest-corner assignment
    # flips per seed (an off-axis diagonal busts the tick budget
    # and brushes the wrong-corner patrol).
    "mfb-base-1-defend-base-2-build",
    # Wave-4 TURTLE node of the tech triple (SC2 turtle macro /
    # military fortify-before-research doctrine anchor). Hard defines
    # two agent spawn_point groups (NORTH base / SOUTH base) so the
    # base latitude varies by seed; the two hunt bands (north + south)
    # always place and hit whichever latitude the agent occupies, so
    # a memorised pbox-lane placement cannot generalise across seeds.
    "tech-turtle-defensive-tech",
    # Wave-4 Group D reasoning pack — SC2 bait-micro / military feint-
    # and-flank doctrine. Hard tier defines two agent spawn_point groups
    # (NORTH staging y=10 / SOUTH staging y=30) round-robined by seed;
    # 6 enemy guards are split into NW + SW half-clusters around the
    # objective `fact` so the working bait-direction + counter-attack-
    # target pair flips per seed and a single memorised bait line
    # cannot generalise across seeds.
    "combat-bait-counter-attack",
    # Wave-4 Group D combat-micro kite pack (SC2 kiting / cavalry
    # skirmish anchor). Hard tier defines two agent spawn_point groups
    # (NORTH corridor y=10 vs SOUTH corridor y=30) round-robined by
    # seed; the heavy tank is centred mid-latitude so the kite vector
    # flips per seed and no memorised "retreat west on y=20" opening
    # generalises.
    "combat-kite-jeep-vs-tank",
    # Wave-4 Group B TURTLE node of the expansion triple (SC2 fortress
    # macro / 1-base mass-defence; military fortress doctrine; risk-
    # averse single-market deep-investment anchor). Hard tier defines
    # two agent spawn_point groups (NORTH y=12 / SOUTH y=28) round-
    # robined by seed; symmetric hunt bands arrive at BOTH latitudes
    # regardless (enemy actors don't honour spawn_point — CLAUDE.md),
    # so a memorised pbox layout from one spawn does not generalise.
    "expansion-turtle-1-base-fortified",
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
    # PR #8 from yiyu-tian: 4 packs that tighten one decision axis per
    # tier rather than per-spawn variation. UPGRADED would force a
    # second variable per tier and break the clean attribution model.
    "perception-count-the-threat-small-k": "hard tightens K (2→3→4) "
    "and the positional non-obviousness of each squad — adding spawn "
    "variation would dilute the count-and-infer signal. Companion to "
    "perception-count-the-threat (UPGRADED above), which IS spawn-varied.",
    "coordination-ordered-rendezvous": "hard tightens the waypoint "
    "count (2→3→4) and the overall deadline; spawn variation would "
    "compete with the order-enforcement axis for attribution.",
    "tempo-strike-window": "hard tightens the lull length, follow-up "
    "window, and attrition cap (cat-c11 canonical tempo design). "
    "Spawn variation would dilute the tempo-discipline signal.",
    "risk-blockade-bypass": "hard adds a tempo gate (after_ticks) and "
    "shrinks the attrition cap; spawn variation would muddy the "
    "corridor-vs-detour route-choice signal.",
    "proc-ordered-action-strict": "non-spatial ordered build task; "
    "spawn variation would compete with the order-enforcement signal",
    "proc-no-attack-passive-only": "spawn variation would compete with "
    "the ROE-compliance signal — the controlled variable across tiers "
    "is the sentry density / loss cap (the ROE compliance pressure), "
    "and adding a seed-varied agent start would dilute that "
    "attribution. Seed-randomised sentry anchors give within-hard "
    "variation without varying the agent's start.",
    "proc-only-build-no-combat": "non-spatial build task; spawn "
    "variation would compete with the role-discipline signal",
    "strict-toolban-fidelity-under-pressure": "spawn variation would "
    "compete with the tool-discipline signal — the binding measurement "
    "is whether the agent calls a forbidden tool under pressure, and a "
    "seed-varied agent start would introduce a second uncontrolled "
    "variable (route geometry) that competes with that signal for "
    "attribution. Hard tightens the patrol intensity (a second arc) "
    "and the attrition cap (units_lost_lte 1 → 0) instead.",
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
