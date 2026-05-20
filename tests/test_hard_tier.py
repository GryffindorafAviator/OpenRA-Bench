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

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
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
    # Wave-7 econ reasoning: split-routing under heterogeneous
    # round-trip cost (OR vehicle-routing / SC2 worker-distribution /
    # M/M/c). Hard defines two agent spawn_point groups (NORTH base
    # y=14 / SOUTH base y=28) round-robined by seed; four neutral
    # mines stay fixed but the SPAWN-MATCHED (A,B) pair flips per
    # seed (NORTH → (16,14)+(80,14); SOUTH → (16,28)+(80,28)), so a
    # memorised single-pair split cannot generalise.
    "econ-harvester-pathing-optimization",
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
    # Wave-7 Group B reasoning pack — greedy 3-base macro against a
    # deadline (SC2 3-base macro / MicroRTS expansion / industrial
    # site expansion anchor). Hard tier defines two agent spawn_point
    # groups (NORTH base layout y≈20 / SOUTH base layout y≈50)
    # round-robined by seed; the win clause accepts EITHER candidate
    # far-east region ((90,20) or (90,50)) so the agent must place
    # the 3rd proc in line with their actual base latitude. A
    # memorised "place at (90,20)" generalises to NORTH but mis-places
    # on SOUTH.
    "mfb-third-base-against-clock",
    # Wave-7 Group B reasoning pack — concurrent two-base ramp (SC2
    # simultaneous-base macro / multi-plant manufacturing /
    # distributed-systems geo-redundancy anchor). Hard tier defines
    # two agent spawn_point groups (NORTH base latitude y=12 / SOUTH
    # base latitude y=28) round-robined by seed; both candidate
    # target regions move with the bases (NORTH (15,12)+(85,12) vs
    # SOUTH (15,28)+(85,28)), so a memorised "always (15,20) /
    # (85,20)" opening misses both regions on either seed group. A
    # single HoldFire rifleman per spawn group acts as a seed-driven
    # start MARKER so units_summary reveals which spawn fired (no
    # role in the win predicate, which is buildings-only).
    "mfb-two-base-simultaneous",
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
    # Wave-5 Group N coordination pack — diversionary / split-attack
    # assault (SC2 multi-prong + Sun Tzu loud feint anchor). Hard tier
    # defines two agent spawn_point groups (NORTH y≈10..14 / SOUTH
    # y≈26..30) round-robined by seed; the REAL `fact` at (100,10)
    # and the DECOY `powr` at (100,30) place every seed (enemy
    # actors don't honour spawn_point — CLAUDE.md), so the SAME
    # diversionary discipline is tested from either spawn while the
    # bait/strike vector flips per seed.
    "coord-diversionary-attack",
    "def-in-depth",
    # Wave-8 REASONING pack — concurrent defence + construction under
    # active attack (SRE simultaneous incident-response + capacity-add /
    # military business continuity / ops triage during incident anchor).
    # Hard tier defines two agent spawn_point groups (NORTH base y=14 /
    # SOUTH base y=26) round-robined by seed; heavy rush bands always
    # place at BOTH candidate latitudes (enemy actors don't honour
    # spawn_point — CLAUDE.md), so the on-latitude band converges on
    # the active fact and a memorised "build at (10,20)" plan misses
    # the active cluster region. Both streams (pbox build + tank
    # dispatch) must run concurrently from t=0 at the spawn-matched
    # latitude.
    "def-while-building",
    # Wave-7 reasoning pack — distributed defence across three
    # concurrent attack lanes (graph min-cut / military multi-front /
    # distributed-systems load-balancing anchor). Hard tier defines
    # two agent spawn_point groups (NORTH base y=14 / SOUTH base y=26)
    # round-robined by seed; the base latitude (and thus the matching
    # three-zone defensive layout) flips per seed, so a single
    # memorised "send to (25,8)/(25,20)/(25,32)" plan wins one seed
    # and loses the other — the 2/2/2 doctrine must be matched to
    # the actual base latitude.
    "def-multi-direction",
    "def-surprise-flank-react",
    # Wave-7 REASONING pack — intel-driven adaptive defense
    # (PlanBench replanning / military reactive defense / intel-driven
    # facility hardening anchor). Sibling/inverse of
    # def-position-expected-direction (which DECLARES the threat axis
    # in the brief): this pack HIDES the axis, so the agent must
    # SCOUT to find the enemy's forward outpost before committing
    # pillboxes on that lane. Hard tier defines two agent
    # spawn_point groups (NORTH base y=16 / SOUTH base y=24) round-
    # robined by seed; both north (y=4) and south (y=36) enemy
    # outposts always place (enemy actors don't honour spawn_point —
    # CLAUDE.md), so each seed faces the SAME scout-then-fortify
    # discipline from a flipped base latitude. A memorised "always
    # scout/defend NORTH" opening WINS the NORTH-spawn seeds by
    # coincidence but LOSES the SOUTH-spawn seeds (clause 1 of the
    # then-chain never latches AND the south rush razes the
    # SOUTH-spawn fact), so no fixed-axis opening generalises.
    "def-position-revealed-direction",
    # Wave-5 Group F cash-reserve / treasury management pack (SC2 cash
    # overflow / financial runway / operational reserve anchor). Hard
    # tier defines two agent spawn_point groups (NORTH base y=14 /
    # SOUTH base y=28) round-robined by seed; the two near patches
    # sit on the spawn-matched row so throughput is symmetric per
    # spawn, but a memorised opening cannot generalise across seeds.
    "econ-cash-reserve-management",
    # Wave-6 IFBench / BFCL V4 negative-instruction seed — conditional
    # SOP / selective-action discipline anchor. The brief carries an
    # explicit carve-out ("move the JEEPS to (90,20); do NOT move the
    # TANKS"); win checks BOTH halves via type-filtered region
    # (`units_of_type_in_region_gte`). Hard tier defines 2 agent
    # spawn_point groups (NW jeep staging y≈8 / SW jeep staging y≈32)
    # round-robined by seed; the TANKS are duplicated at (8..10, 20)
    # across BOTH groups so the must-not-move region is identical and
    # well-defined across seeds. The spawn variation prevents memorised
    # opening generalisation without diluting the selective-action
    # signal.
    "proc-instruction-following-edge-case",
    # Wave-5 Group F reasoning seed — common-pool contested mining
    # (SC2 contested expansion / Hardin tragedy of commons / TAM
    # contention). Hard defines two agent spawn_point groups (NORTH
    # base y=14 / SOUTH base y=26) round-robined by seed with a
    # shared harv-lane + defender ring at y=18..22 — base orientation
    # flips per seed (rear-guard 3tnk at y=14 vs y=26 marks the
    # spawn group). The enemy raider strike force + dense contested
    # flavor (3 rival harvs + proc) ALWAYS place (enemy actors
    # don't honour spawn_point — CLAUDE.md), so each seed faces the
    # SAME committed-balance test from a different base orientation
    # and a memorised "always defend at y=N" opening cannot
    # generalise.
    "econ-contention-with-enemy",
    # Wave-6 perception pack — military intelligence reconnaissance
    # (recon-and-extract) / SC2 scout-and-return for info / drone
    # surveillance with return-to-base / intel ops: gather then withdraw.
    # Sibling of scout-and-survive (which permits ANY return of a single
    # jeep); this pack requires the WHOLE 2-jeep team to extract INTACT
    # (`units_lost_lte:0` + `units_in_region_gte n:2`) AFTER revealing
    # the specific far enemy fact. Hard defines two agent spawn_point
    # groups (NORTH (10,8) / SOUTH (10,32)) round-robined by seed; the
    # win clause is `any_of` over the two matching corner-return
    # regions, and a second scattered defender group at the inner
    # bands (y=12 / y=28) punishes the near-edge shortcut so a
    # memorised single-detour-band opening cannot generalise.
    "scout-and-report",
    # Wave-6 salvage: agent timeout pre-commit
    "combat-retreat-after-engagement",
    # Wave-8 BCP-evacuation: doomed base, EVAC mobile force east to
    # a pre-designated safe zone before attrition busts the survival
    # cap. Hard tier defines 2 spawn_point groups that flip the
    # safe-zone latitude (NORTH (90,12) vs SOUTH (90,28)) per seed;
    # the base + heavy assault stay fixed (enemy actors don't honour
    # spawn_point — CLAUDE.md), so the per-seed read is "which
    # corridor did MY tanks start in" → "EVAC to the matching zone".
    "def-evacuation",
    # Wave-6 proc-checklist-no-deviation: strict ordered visit-
    # checklist (IFBench step-order / PlanBench strict ordering /
    # aviation pre-flight / SOP no-skip-no-reorder anchor). Hard
    # tier defines 2 spawn_point groups (FAR-WEST x=6 vs NEAR-WEST
    # x=20, both at mid-y so the canonical east-then-turn pathfinder
    # cannot accidentally clip the northern (y=4) or southern (y=36)
    # waypoints en route). _NO_ENEMY pack — only fail is the timeout.
    "proc-checklist-no-deviation",
    # Wave-6 Group I procedural-compliance seed — τ²-bench distractor
    # handling / IFBench irrelevant-tool ignoring / BFCL V4 relevance
    # (refuse to call a tool that isn't needed) / operator discipline
    # (don't use the kitchen-sink palette). FULL tool palette
    # (move/attack/build/place_building/observe) is exposed but
    # forbidden_tools is NOT used — the test is whether the model
    # REASONS about which tools matter, not whether it can follow an
    # explicit allowlist (those are the other procedural packs). Hard
    # defines two agent spawn_point groups (NORTH y=8 / SOUTH y=32)
    # round-robined by seed; both staging latitudes face symmetric
    # flank patrols (central + matching flank band), so a memorised
    # "ignore build, drive y=20 east" opening from one spawn does not
    # generalise — every seed picks the correct spawn-matched lane.
    "proc-tool-use-with-distractor",
    # Wave-6 perception pack — CICERO/Diplomacy deep recon, ERQA off-
    # axis perception, intelligence-ops "search unconventional
    # locations", SC2 hidden-base detection. The map presents an
    # obvious eastern decoy presence (3× e1 pickets at x=92..98); the
    # REAL enemy `fact` is in an off-axis corner (far NW on easy/
    # medium, NW OR SW on hard). Hard tier defines two agent
    # spawn_point groups (NORTH-leaning (15,16) / SOUTH-leaning
    # (15,24)) round-robined by seed; both hidden-base candidates
    # (NW + SW) place every seed (enemies don't honour spawn_point
    # — CLAUDE.md), so the spawn rotation flips which off-axis
    # corner is the SHORTER reach but EITHER hidden-base discovery
    # satisfies the win.
    "scout-discover-hidden-base",
    # Wave-6 robustness reasoning pack — PlanBench replanning / goal-
    # conditional adaptation / SC2 reactive macro pivot / military
    # objective change during ops anchor. The win is `any_of` over
    # TWO destruction paths (fact A at (60,20) heavily defended, fact
    # B at (100,30) lightly defended); commit-A-pure busts the
    # attrition cap, commit-B-pure-on-medium+ runs through A's e3
    # sight envelope and dies, only the recognise-and-pivot policy
    # converges inside cap + clock. Hard defines two agent
    # spawn_point groups (NORTH staging y=12..16 / SOUTH staging
    # y=24..28) round-robined by seed; the two enemy bases always
    # place (enemies don't honour spawn_point — CLAUDE.md) plus a
    # central intercept picket (75,25) bites the pure-B beeline
    # so a memorised opening cannot generalise across seeds.
    "rob-objective-shift-with-or-clause",
    # Wave-6 Group I procedural-compliance pack — defensive ROE drill
    # (IFBench negative-instruction / military ROE / BFCL V4 allowlist /
    # security-guard react-only doctrine). The agent's base (`fact`) is
    # probed by `patrol` raiders; defenders start on HoldFire and must
    # be lifted via `set_stance` so the engine's stance-driven auto-fire
    # racks up the ≥2 (easy/medium) or ≥3 (hard) kills. attack_unit /
    # attack_move are FORBIDDEN. Hard defines TWO agent spawn_point
    # groups (NORTH y=17..19 / SOUTH y=21..23) round-robined by seed;
    # both staging slots cover the same fact column so the defensive
    # footprint is symmetric per spawn but a memorised "defenders at
    # (14,20)" cluster cannot generalise across seeds.
    "proc-only-defend-no-attack",
    # Wave-6 perception coverage pack (SC2 map-control / vision objectives
    # / ERQA coverage / military area dominance / drone-coverage anchor).
    # Win is explored_pct_gte X% within a tick budget with units_lost_lte:0.
    # Hard tier defines two agent spawn_point groups (SW column y=20..36
    # vs NW column y=4..20) of 5 jeeps each, round-robined by seed; the
    # five frontier corners (4 distant corners + a centre diagonal) are
    # symmetric to both spawn columns so the per-jeep nearest-quadrant
    # assignment flips per seed and a memorised "which jeep to which
    # corner" plan cannot generalise.
    "scout-map-reveal-percent-target",
    # Wave-6 compound-incident triage pack (defence + economy + tech
    # under simultaneous pressure). Hard tier defines two agent
    # spawn_point groups (NORTH y=14 / SOUTH y=26) round-robined by
    # seed; the centre-line raider tanks and centre-line hunt squad
    # face the seed-chosen latitude with equal pressure.
    "rob-multiple-simultaneous-pressures",
    # Wave-6 multi-distractor tool-relevance discipline pack (τ²-bench
    # multi-distractor / BFCL V4 cluttered-API anchor). Hard tier
    # defines two agent spawn_point groups (NORTH lane y=4..6 / SOUTH
    # lane y=34..36) round-robined by seed; both lanes route around
    # the same off-path north garrison (y=8..9) and central sentry
    # stack (y=18..22), so a memorised "drive east on y=20" opening
    # cannot generalise across the lane variants.
    "proc-tool-use-multi-distractor",
    # Wave-6 Group E perception pack — parallel multi-region scout
    # (Watch-And-Help concurrent multi-task / SMAC distributed
    # exploration / SC2 multi-scout coverage / military distributed
    # reconnaissance doctrine anchor). 3 jeeps must split across K
    # foggy regions; sending all jeeps to one region or touring all
    # regions with a single jeep loses on the K-discovery bar or the
    # clock. Hard defines two agent spawn_point groups (NW y=6..12 /
    # SW y=28..34) round-robined by seed; the K=4 region layout is
    # symmetric N/S so the sweep corridor (NW→NE on the north edge
    # vs SW→SE on the south edge) flips per seed and a memorised
    # single-corner reading order cannot generalise across spawns.
    "scout-multiple-fog-areas",
    # Wave-6 ROBUSTNESS / adversarial robustness seed — surprise 2nd-
    # wave handling (military follow-on wave doctrine / SC2 second-
    # attack timing / ops incident-after-incident triage anchor). Hard
    # tier defines two agent spawn_point groups (NORTH base y=12 /
    # SOUTH base y=28) round-robined by seed; symmetric Wave-2 squads
    # at BOTH fog corners (NE y=8 / SE y=32) always place (enemy
    # actors don't honour spawn_point — CLAUDE.md), so which corner
    # the surprise wave arrives FROM flips per seed and a memorised
    # "expect Wave 2 from NE" disposition cannot generalise.
    "rob-unexpected-enemy-spawn",
    # Wave-6 ROBUSTNESS / reasoning seed — PlanBench replanning under
    # exogenous loss / SC2 rebuild-after-trade / military force-regen /
    # ScienceWorld error recovery anchor. Agent commands 4 heavy tanks
    # (3tnk) + a production base (fact + powr + weap + fix); a pre-
    # placed enemy 4tnk strike force at the lane mouth lands its
    # opening salvo on tick 0 (stance:3) and kills 1+ agent tanks;
    # the agent must commission replacements from the war factory with
    # the reserve cash AND continue the eastward assault to clear the
    # e1 garrison. Hard tier defines two agent spawn_point groups
    # (NORTH-flank scout (16,14) vs SOUTH-flank scout (16,28)) round-
    # robined by seed; the central combat force + base buildings are
    # SHARED across both groups (per CLAUDE.md, base buildings are
    # duplicated under both spawn_points because ANY agent actor with
    # a spawn_point causes agent actors WITHOUT a spawn_point to be
    # filtered out), so the strike geometry is symmetric but a
    # memorised opening cannot generalise.
    "rob-unit-loss-recovery",
    # Wave-6 Group N coordination pack — military pincer movement /
    # SC2 two-prong attack / envelopment from two directions. Two
    # 3-tank squads start at OPPOSING west-edge latitudes (north y=8 /
    # south y=32) and must converge on a central enemy cluster
    # SIMULTANEOUSLY: the win predicate requires ≥4 tanks in the
    # central region (single squad has only 3 — structurally
    # impossible) AND ≤2 tanks lost (sequenced A-then-B-late shreds A
    # before B arrives → cap busted). Hard defines two agent
    # spawn_point groups round-robined by seed with DIFFERENT cell
    # coordinates (group 0: (5,8)/(5,32); group 1: (3,5)/(3,35)) so
    # a memorised single-cell opening cannot generalise; the cluster
    # is symmetric across y=20 so both spawns face the same pincer
    # decision.
    "combat-pincer-coordination",
    # Wave-6 ACTION cell — escort fragile VIP (harv) through cumulative
    # arty walls; hard tier defines 2 agent spawn_point groups (NORTH y=14
    # / SOUTH y=26) round-robined by seed.
    "combat-protect-vip-escort",
    # Wave-6 Group D combat-micro pack: SC2 flank micro / military
    # flank-maneuver doctrine / force-multiplier-through-angle-of-
    # attack anchor. Hard tier defines two agent spawn_point groups
    # (NORTH staging y=14..17 / SOUTH staging y=23..26) round-robined
    # by seed; the two-column e3 wall at x=60..62 is symmetric across
    # y=20 so either spawn faces the same flank-vs-frontal decision
    # from a flipped bearing, and no memorised opening generalises.
    "combat-flanking-attack",
    # Wave-7 REASONING seed: military divide-and-conquer / defeat-in-
    # detail / SMAC squad-isolation / CICERO splitting anchor. The
    # agent commands 4× 2tnk vs TWO enemy clusters (N at y=15 / S at
    # y=25, each 3× e3 + 1× 1tnk, stance:3). Engaging at the y=20
    # midpoint puts the lead tank in range of BOTH clusters; the
    # winning play is to flank well off-axis, isolate one cluster,
    # destroy it, then pivot to the other. Hard tier defines two
    # agent spawn_point groups (NORTH staging y=10..13 / SOUTH
    # staging y=27..30) round-robined by seed; the two clusters are
    # symmetric across y=20 so each spawn faces the same divide-and-
    # conquer decision but the first-flank-target flips per seed
    # (N spawn engages Cluster A first; S spawn engages Cluster B
    # first) — a memorised "always flank Cluster A first" opening
    # cannot generalise across seeds.
    "combat-divide-and-conquer",
    # Wave-7 combat-formation pack: military tank-wedge doctrine /
    # SC2 formation micro / combined-arms anchor. The agent commands
    # 5× 2tnk and must arrange them in a WEDGE (apex + 2 flankers
    # per side spread across y=18..22) before contacting an eastern
    # cluster (4-5× e3 + 1-2× 1tnk at x=84..86). A COLUMN (single-
    # file east on y=20) concentrates incoming Dragon fire on the
    # lead tank and bleeds the survival bar (own_units_gte:4 fails
    # when 2+ tanks lost); the WEDGE spreads return fire across the
    # formation and clears the cluster intact. Hard defines two agent
    # spawn_point groups (NORTH staging y=12..16 / SOUTH staging
    # y=24..28) round-robined by seed; the central cluster is
    # symmetric across y=20 so either spawn faces an equivalent
    # column-vs-wedge decision and no memorised opening generalises.
    "combat-formation-tank-wedge",
    # Wave-6 perception pack — early-warning intrusion detection
    # paired with targeted intercept (SC2 early-warn scout /
    # NORAD early-warning / IDS / military reconnaissance-in-force
    # anchor). Hard tier defines two agent spawn_point groups
    # (NORTH base y=14 / SOUTH base y=26) round-robined by seed;
    # two hunt bands (N y=10 + S y=30) always place (enemy actors
    # don't honour spawn_point — CLAUDE.md), so the per-seed base
    # latitude varies but the scout-then-intercept doctrine
    # generalises. A memorised "send scout to (40,10) + tanks to
    # (45,10)" opening cannot generalise across seeds.
    "scout-detect-incoming-army",
    # Wave-7 ACTION econ-defense pack — convoy / supply-line protection
    # (SC2 harass defense / military convoy protection / supply-line
    # doctrine anchor). A single harv commutes proc↔mine on a long
    # exposed route; raider 2tnks specifically target the harv.
    # exposed route; a `raider` 1tnk specifically targets the harv.
    # Defenders at base never engage (raider intercepts harv beyond
    # base sight); intended play is to move escorts east to intercept
    # on the route. Hard tier defines two agent spawn_point groups
    # (NORTH route y=14 / SOUTH route y=26) round-robined by seed;
    # symmetric north + south raider waves always place (enemy actors
    # don't honour spawn_point — CLAUDE.md), so each spawn defends
    # its OWN supply lane and a memorised opening cannot generalise.
    "econ-protect-harvester-route",
    # Wave-7 Group D reasoning pack — rock-paper-scissors hard-counter
    # selection (SC2 hard-counter doctrine / military RPS counter /
    # capability-based defense procurement anchor). Cash $2550 funds
    # EITHER 3× 2tnk (the right counter to pure-infantry enemy) OR
    # 8× e3 (wrong counter — anti-tank rockets vs soft targets) OR
    # 25× e1 (1:1 attrition match). Hard tier defines two agent
    # spawn_point groups (NORTH base y=12 / SOUTH base y=28) round-
    # robined by seed; the centre infantry cluster always places at
    # (70,20) (enemy actors don't honour spawn_point — CLAUDE.md),
    # so the composition decision is the same per seed but the lane
    # the agent commits to flips per seed and a memorised opening
    # cannot generalise.
    "combat-vehicle-vs-infantry-counter",
    # Wave-7 REASONING temporal-sequencing pack — SC2 timing-push
    # window / PlanBench temporally-extended goal / cyber attack
    # timing-window anchor. The `then:` happened-before composite
    # enforces a SURVIVAL gate (own_units_gte:4 at T1) latching
    # BEFORE the STRIKE gate (units_killed_gte:K within T2), so
    # premature engagement and stalling both lose. Hard tier defines
    # two agent spawn_point groups (NORTH staging y=12 / SOUTH
    # staging y=28) round-robined by seed; the central enemy turtle
    # cluster + tsla place every seed (enemy actors don't honour
    # spawn_point — CLAUDE.md) and is symmetric across y=20, so
    # both staging latitudes face the same survive-then-strike
    # decision from a flipped approach axis.
    "tp-survive-and-strike-at-window",
    # Wave-7 REASONING pack: concentrated-defense topology — build a
    # TIGHT CLUSTER of pillboxes around the high-value building (the
    # agent fact). Hard tier defines 2 agent spawn_point groups
    # (NORTH fact at y=14 / SOUTH fact at y=26) round-robined by seed;
    # the cluster centre flips with the fact, so a memorised "cluster
    # at (10,20)" plan cannot generalise. Enemies don't honour
    # spawn_point (CLAUDE.md), so the rush band is staged at BOTH
    # candidate latitudes — only the on-latitude band converges on
    # the active fact, but it is heavy enough to overwhelm any
    # defence that isn't a CLUSTER around the correct fact.
    "build-defensive-tower-cluster",
    # Wave-7 REASONING asymmetric-underdog pack — SC2 asymmetric /
    # guerrilla tactics / asymmetric warfare anchor. 2× 2tnk vs 4× e1
    # + 1× 3tnk; head-on loses, off-axis flank-pick (outside the
    # 3tnk's aggro envelope) wins. Hard defines two agent spawn_point
    # groups (NORTH staging y=12 / SOUTH staging y=28) round-robined
    # by seed; the enemy garrison (e1 wall + 3tnk on the east face)
    # ALWAYS places (enemy actors don't honour spawn_point —
    # CLAUDE.md), so either staging latitude faces the same asymmetric
    # flank-vs-charge decision from a flipped approach axis and a
    # memorised opening cannot generalise.
    "adv-asymmetric-weaker-must-win",
    # Wave-7 REASONING / RPS hard-counter pack (INVERSE of combat-
    # vehicle-vs-infantry-counter) — SC2 hard-counter / anti-armor
    # procurement / military RPS anchor. Starting cash ($1800) funds
    # exactly ONE composition vs a pre-placed band of HEAVY tanks
    # (3tnk on easy/medium, 4tnk Mammoths on hard); the agent must
    # build e3 (rocket soldiers, anti-vehicle Dragon launcher) — not
    # 1tnk (light tanks lose attrition to heavy armour, budget buys
    # only ~2) and not e1 (no anti-armour weapon, kill bar fails).
    # Hard tier defines two agent spawn_point groups (NORTH base
    # y=12 / SOUTH base y=28) round-robined by seed; the heavy band
    # is centred mid-latitude (y=20) so both spawns face symmetric
    # pursuit geometry (enemy actors don't honour spawn_point —
    # CLAUDE.md) and a memorised "build e3 at y=20" opening cannot
    # generalise across seeds.
    "combat-rocket-soldier-anti-vehicle",
    # Wave-7 perimeter/firewall reasoning pack — ERQA spatial commit /
    # MicroRTS defense placement / military perimeter (firewall rule
    # placement) anchor. Sibling/inverse of def-tower-line-vs-cluster:
    # that pack enforces CLUSTER at a single bottleneck cell (graph
    # min-cut doctrine); this pack enforces a LINE across the corridor
    # (one pbox per row spanning y=18..22 at x=60, radius 0.5 so only
    # the exact rung cell counts). Hard tier defines two agent
    # spawn_point groups (NORTH base y=12 / SOUTH base y=28) round-
    # robined by seed; the rusher band is centred at y=20 and ALWAYS
    # places (enemy actors don't honour spawn_point — CLAUDE.md), so
    # the corridor LINE is identical across seeds but the agent's base
    # bearing flips per seed and a memorised relative-to-base placement
    # cannot generalise.
    "build-defensive-tower-line",
    # Wave-7 Group I REASONING — opening-phase build-order / power-grid
    # bring-up sequencing (PlanBench task-ordering / SOP compliance /
    # electrical-grid bring-up anchor). Hard tier defines two agent
    # spawn_point groups (NORTH y=12 / SOUTH y=28) round-robined by
    # seed; the pre-placed `fact` (and therefore the build radius and
    # the placement coords for powr/proc) flips per seed, so a
    # memorised "(20,20) opening" cannot generalise. An inert HoldFire
    # `e1` per group surfaces the variation via units_summary (the
    # pack would otherwise be building-only); no `move_units`/
    # `attack_unit` tool is exposed so the e1 is functionally inert
    # and does not interact with the SOP test.
    "build-power-online-first",
    # Wave-7 REASONING pack — cost-optimal build-order (powr → proc →
    # weap) under a tight deadline (PlanBench cost-optimal / BOM-
    # manufacturing critical-path anchor). Hard tier defines two agent
    # spawn_point groups (NORTH base y=14 / SOUTH base y=26) round-
    # robined by seed; ore patches are duplicated at both latitudes so
    # harv income is symmetric per spawn. A memorised "place powr at
    # (14,22)" opening cannot generalise — placement must be computed
    # relative to the actual fact each seed.
    "build-sequence-tech-fastest",
    # Wave-7 REASONING econ pack — SC2 silo management / working-capital
    # allocation / FIFO perishable inventory cap anchor. Cash starts NEAR
    # the proc-only storage cap (1800 vs 2000); income accrues; the
    # any_of clause is silo OR ≥K kills OR ≥2 pbox — three legitimate
    # spend-or-build paths, none of which is "hoard". Hard tier defines
    # two agent spawn_point groups (NORTH base y=14 / SOUTH base y=26)
    # round-robined by seed; ore patches are duplicated at both
    # latitudes so per-spawn throughput is symmetric (neutral / enemy
    # actors don't honour spawn_point — CLAUDE.md). The garrison sits
    # at y=19..21 so attack distance from either spawn is comparable,
    # and a memorised "place silo at (14,18)" opening cannot generalise
    # across seeds.
    "econ-silo-vs-spend",
    # Wave-7 Group D combat-micro / formation pack: military tank-wedge
    # doctrine / SC2 formation micro / combined-arms anchor. Hard tier
    # defines two agent spawn_point groups (NORTH staging y=10..14 /
    # SOUTH staging y=26..30) round-robined by seed; the bracketing
    # corridor (e3 brackets at y=16 and y=24 plus two on-axis 1tnk
    # blockers) is symmetric across y=20 so either spawn faces an
    # equivalent column-vs-wedge decision from a flipped bearing — a
    # memorised "form wedge from y=14" opening cannot generalise.
    "combat-formation-tank-wedge",
    # Wave-7 Group D combat-micro pack: SC2 mirror micro / Lanchester
    # square law / concentration-of-force doctrine. ASYMMETRIC geometry
    # (3 agent tanks bunched at (30,*) vs 3 enemy tanks spread across
    # 3 latitudes at (50,15)/(51,20)/(50,25)) so focus-fire on the
    # closest enemy preserves the force while spread-fire (each tank
    # picks its own nearest enemy) collapses to Lanchester linear and
    # loses 2 of 3 tanks (the load-bearing discrimination on medium).
    # Hard tier defines two agent spawn_point groups (NORTH y=11..13 /
    # SOUTH y=27..29) round-robined by seed; the enemy line always
    # places (enemy actors don't honour spawn_point — CLAUDE.md) so
    # both spawns face the same focus-fire decision from a flipped
    # bearing. Hard discriminates via kill-speed (tight within_ticks
    # 1200) + spawn-variation generalisation.
    "combat-tank-vs-tank-engagement",
    # Wave-7 REASONING pack — CAPEX-vs-OPEX capital allocation under
    # rush pressure (PlanBench resource-allocation / financial
    # allocation / SC2 production: pump units now vs lay second
    # factory anchor). Starting cash ($3400) funds EITHER 4× 2tnk
    # (immediate force, OPEX) OR 1× weap second factory (long-run
    # capacity, CAPEX). Rusher band is incoming at t=0 ⇒ buy-now
    # WINS, build-weap-first LOSES. Hard tier defines two agent
    # spawn_point groups (NORTH base y=14 / SOUTH base y=26) round-
    # robined by seed; rusher bands always place at BOTH latitudes
    # (enemy actors don't honour spawn_point — CLAUDE.md) but the
    # rusher bot targets the agent centroid, so the active threat
    # axis flips per seed. Inert HoldFire e1 per group surfaces the
    # spawn variation via units_summary (the buy-vs-build base is
    # otherwise building-only).
    "econ-buy-vs-build-decision",
    # Wave-7 ACTION pack — cut-off encirclement (Cannae doctrine /
    # SC2 encirclement / "the hammer needs an anvil"). Strike force
    # must position ONE tank in the eastern cut-off region BEFORE
    # engaging the centre cluster with the rest, otherwise either
    # the column gets bled by e3 Dragon fire (brute-east) or the
    # cluster is wiped with no tank east (brute-centre). Hard tier
    # defines two agent spawn_point groups (NORTH staging y=12..15 /
    # SOUTH staging y=25..28) round-robined by seed; the centre
    # cluster is symmetric across y=20 so both spawns face the same
    # encirclement geometry from a flipped flank latitude, and a
    # memorised "always flank via the north" opening cannot
    # generalise across seeds.
    "combat-prevent-retreat",
    # a single central raider at y=20 always places (enemy actors
    # don't honour spawn_point — CLAUDE.md) and beelines on whichever
    # harv exists, so each spawn defends its OWN supply lane on its
    # OWN y-band and a memorised opening cannot generalise.
    "econ-protect-harvester-route",
    # Wave-8 ACTION pack — timed-stance-flip ambush trigger discipline
    # (military ROE ambush doctrine / SC2 stance micro / USMC FM 7-8
    # linear-ambush anchor). 4× 2tnk pre-staged at a choke on stance:0
    # (HoldFire); the agent MUST `set_stance` to AttackAnything AT THE
    # RIGHT TIME (when the inbound hunt squad enters the kill zone) to
    # wipe the bunched column with one concentrated cannon salvo.
    # Hard defines TWO agent spawn_point groups (NORTH choke y=18..20
    # vs SOUTH choke y=20..22) round-robined by seed; both close the
    # same fact column (x=10) so the doctrine generalises but a
    # memorised "fire when enemy reaches (40,20)" cell cannot.
    "def-stance-mgmt-hold-then-attack",
    # Wave-8 REASONING pack — bypass a prepared frontal line via a far
    # off-axis fog lane and strike the undefended HQ from behind (SC2
    # hidden assault / military surprise attack / fog warfare anchor).
    # 4× 2tnk vs a west-facing pbox+e3 wall at x=50, y=15..25; the
    # objective fact is undefended at (100,20). Hard defines TWO agent
    # spawn_point groups (NORTH staging y=14..17 / SOUTH staging
    # y=23..26) round-robined by seed; the line is symmetric across
    # y=20 so each spawn faces the same fog-flank decision but the
    # bypass latitude flips per seed (NORTH spawn → fog via y=2; SOUTH
    # spawn → fog via y=38) — a memorised "always fog-flank via the
    # north" opening cannot generalise across seeds.
    "combat-attack-from-behind-fog",
    # Wave-8 REASONING pack — capacity planning / distributed-systems
    # back-pressure / FIFO inventory. 3-harv high-throughput economy
    # saturates the proc cap (2000) fast; the agent MUST build silos
    # (+3000 cap each) to absorb the overflow or the EV bar is
    # unreachable. Hard defines TWO agent spawn_point groups (NORTH
    # base y=14 vs SOUTH base y=26) round-robined by seed; each spawn
    # has its own near-patch geometry so a memorised silo placement
    # cell cannot generalise.
    "econ-overflow-to-silos",
    # Wave-8 PERCEPTION pack — long-range reconnaissance / chassis
    # selection by speed (ERQA recon / military light-cavalry-over-
    # foot-patrol / SC2 scout-unit selection anchor). A fast jeep and a
    # slow e1 are BOTH pre-placed in the agent base; only the jeep is
    # fast enough to reach the far frontier inside the clock. Hard
    # defines TWO agent spawn_point groups (NORTH base y=8 / SOUTH
    # base y=32) with spawn-matched frontier targets (NE (120, 5) /
    # SE (120, 35)) round-robined by seed; the win clause is `any_of`
    # over the two frontier regions so the jeep must drive to the
    # frontier matching its OWN base latitude (an off-axis cross-
    # diagonal commit blows the budget).
    "scout-far-frontier",
    # Wave-8 PERCEPTION pack — exact-count force sizing (POMDP exact-
    # count / ScienceWorld census / SC2 scout-count anchor). 2 jeep
    # scouts must count the K hidden defenders at the far-east enemy
    # fact, then queue EXACTLY K medium tanks (2tnk) — under-build
    # loses the attrition trade, over-build (queue 7) misses the
    # tight clock. Hard tier defines two agent spawn_point groups
    # (NORTH base y=14 / SOUTH base y=26) round-robined by seed; the
    # K=5 defender cloud + enemy fact always place at mid-y (enemies
    # don't honour spawn_point — CLAUDE.md), so the scout-then-build
    # discipline generalises across spawns but a memorised "scout at
    # y=20" opening cannot.
    "scout-count-defenders",
    # Wave-8 REASONING pack — PlanBench replanning under exogenous loss /
    # disaster recovery / SC2 rebuild-after-trade anchor. The agent
    # inherits a complete production base (fact + proc + powr + weap +
    # harv); a pre-placed low-HP `powr` is destroyed in the opening
    # turn by an adjacent stance:3 `4tnk` strike (1× on easy, 2× on
    # medium/hard); the agent must `build('powr')` + `place_building`
    # adjacent to the surviving `fact` with the indivisible reserve.
    # The win clause uses `then:[not building_count_gte powr, then
    # building_count_gte powr]` so destruction-then-rebuild is enforced
    # by the happened-before latch (CLAUDE.md: `has_building` cannot be
    # used here because its accumulating set never toggles back to
    # false after destruction). Hard tier defines two agent spawn_point
    # groups (NORTH base y=14..22 / SOUTH base y=22..30) round-robined
    # by seed; the full base (fact + proc + powr + weap + harv +
    # defender 3tnk) is duplicated across both groups per CLAUDE.md
    # spawn_point filter rules, and the 2× 4tnk strikers are duplicated
    # at BOTH latitudes (enemy actors don't honour spawn_point) to keep
    # per-spawn strike geometry symmetric — a memorised "(10,18) rebuild
    # cell" cannot generalise across seeds.
    "build-engineer-rebuild-after-loss",
    # Wave-8 REASONING pack — cost-effective recon under an
    # intelligence budget (SC2 Reaper / worker-scout economy / OR
    # cost-effective recon anchor). Cash $900 funds EITHER 1× jeep
    # ($600, wheeled, sight 7c — INTENDED), 1× 1tnk ($700, tracked,
    # sight 6c — over-spend, slower), or 9× e1 ($100 ea, foot,
    # sight 4c — under-spend, too slow). Medium / hard tighten the
    # clock to within_ticks 1200 so only the jeep can field and
    # traverse 91 cells in time. Hard tier defines two agent
    # spawn_point groups (NORTH base y=14 / SOUTH base y=26) round-
    # robined by seed; the frontier marker at (110, 20) is
    # equidistant so the cost-vs-speed decision generalises across
    # spawns while a memorised "always start at (10, 20)" opening
    # does not. Inert HoldFire `e1` per spawn group surfaces the
    # spawn variation via units_summary (the scout base is otherwise
    # building-only at reset).
    "scout-jeep-vs-infantry-cost-effective",
    # Wave-8 ACTION speedrun pack — quickest-path planning to a far
    # objective (human speedrun / quickest-path planning / SC2 worker
    # rush anchor). 3× 2tnk pre-staged at the far west (x=6) must drive
    # directly east and raze the enemy `fact` at the far-east objective
    # region (115,20) inside a TIGHT clock; sub-optimal pathing
    # (south-detour wander) busts the deadline. Hard defines TWO agent
    # spawn_point groups (NORTH staging y≈10 vs SOUTH staging y≈30)
    # round-robined by seed; the objective fact at (115,20) places
    # every seed (enemy actors don't honour spawn_point — CLAUDE.md)
    # and Chebyshev distance from either spawn is identical (109), so
    # the tight clock is consistent across seeds but the bearing-to-
    # objective flips (NORTH spawn → south-east; SOUTH spawn → north-
    # east) and a memorised "drive east on y=20" plan mis-targets
    # from both spawns.
    "tp-rush-objective-very-fast",
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
    # Wave-6 Group J robustness seed. Hard's controlled variable is
    # the strike SEVERITY (one more 4tnk + tent added to the rubble
    # pile). The enemy-bot footgun (CLAUDE.md: enemy actors don't
    # honour spawn_point) makes a clean per-spawn enemy partition
    # geometrically infeasible — foreign-latitude `hunt` 4tnks drift
    # to the active base and over-pressure the defender ring beyond
    # the tuned proc-but-not-fact kill window, converting clean LOSS
    # discrimination into noisy across-seed instability. Spawn
    # variation would compete with the recovery-discipline signal
    # for attribution.
    "rob-cash-depletion-recovery": "hard's controlled variable is "
    "strike severity (proc + tent destroyed vs medium's proc only). "
    "Enemy `hunt` actors don't honour spawn_point (CLAUDE.md "
    "oramap.rs footgun) so foreign-latitude 4tnks drift to the "
    "active spawn and over-pressure the tuned defender ring, "
    "destroying the fact and converting the recovery test into a "
    "fact-defence test. Spawn variation would compete with the "
    "recovery-discipline signal for attribution.",
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
