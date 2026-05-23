# Scenario Uniqueness Audit (Phase 2)

Read-only audit of `openra_bench/scenarios/packs/*.yaml`. Scope: 212 YAML
files (excluding `TEMPLATE.yaml`); 23 in-flight `econ-*` packs were
deliberately skipped (other agents own them), leaving **189 packs probed**.

The probe was a stall policy — `Command.observe()` only — run on the
hard tier, seed=1, of every pack, through the same code path
(`openra_bench.eval_core.run_level`) the bench evaluator uses. Each
result is the engine-evaluated outcome (`win` / `loss` / `draw`),
final game tick, and the static profile read straight from the YAML.

Headline numbers
- 181 / 189 packs (95.8%) — stall LOSSES (healthy: no-cheat bar holds).
- 3 packs — stall WINS (the no-cheat bar is broken).
- 3 packs — stall DRAWS (no real LOSS reachable from a stall play).
- 2 packs — Rust engine panic at `reset(seed=1)` (both
  `adversarial-siege` / `adversarial-skirmish` — already
  `status: quarantine` in-pack so they are NOT in the default eval set,
  but they remain on disk and panic when explicitly invoked).

Quarantined packs are still YAML-discoverable and were probed. They are
flagged `[Q]` in the lists below.

---

## Section 1 — Defect packs

A "defect" here means stall WINS (the lazy bar fell) or stall DRAWS
(no real reachable LOSS — the timeout collapses to draw degeneracy
because the fail predicate never trips). Both are explicit violations
of the "no defect, no cheat" bar in `CLAUDE.md`.

### 1A. Stall WINS (3 — bar fell)

| Pack | Outcome | Turn | Tick | Diagnosis |
|---|---|---|---|---|
| `mid-economy-under-fire` | **WIN** | 11 | 993 | Stall achieves the win predicate without any agent action. Hard win-clause is `economy_value_gte:4000 AND harv≥2 AND units_lost_lte:2`. With the current engine, the 3 starter harvesters auto-harvest (no `harvest` command needed — they begin in `harvest` mode), the perimeter 1tnks auto-fire and kill the lone raider 1tnk, and EV trips 4000 by tick ~993 with zero losses. The pack header says "Stall (only observe)… harvs never harvest → EV stays at 0 → LOSS." That is no longer true under the current engine — harvs in `harvest` mode work without re-issued orders. **Fix**: either remove the harvesters from the starting placement (require the model to issue `Command.harvest`) or replace the `1tnk` raider with a stronger raider wave that out-attritions an idle defense ring. |
| `combat-naval-shore-strike` | **WIN** | 3 | 198 | Stall wins inside 3 turns. The hard tier places two destroyers in the water channel on `stance:2` (Defend), and the entire shore garrison on `stance:0` (HoldFire). The destroyers auto-fire on every garrison unit in range and clear all 5 enemies in <200 ticks; the agent never sends a command. **Fix**: set the destroyers to `stance:0` so the agent must issue an `attack_unit` order; or place the shore garrison just outside the destroyer's auto-target range so the agent must manually fire. The whole capability (target an across-shore target with a ranged unit) collapses because the engine fires automatically. |
| `def-with-ambush` | (intended) **WIN** | 24 | 1233 | **Not a defect — exempt by design**. CLAUDE.md `§Triage coverage`: "1 pack (`def-with-ambush`) is exempt by design (positional-discipline scenario where do-nothing IS the intended policy)." The capability under test is *hold the ambush position*; the four `stance:2` flanker tanks auto-engage the rusher band when it enters weapon range. Listed here only for completeness — no action required. |

### 1B. Stall DRAWS (3 — no LOSS reachable)

A draw means the win predicate was never met AND the fail predicate
never tripped, so the level resolves to `draw` (outcome score 0.5).
This is a defect because it makes the lazy play indistinguishable
from a partial / borderline play, and inflates the lazy score by
treating it as half a win.

| Pack | Outcome | Turn | Tick | Diagnosis |
|---|---|---|---|---|
| `economy-time-box` [Q] | **DRAW** | 80 | 7203 | Already quarantined in-file (reason: "redundant with economy-force-buildup"). The hard tier has **no `fail_condition` at all** (`fail_condition: None`), so a 12000-tick budget that never produces 6 units / 5 buildings simply times out as draw. This is one of the canonical CLAUDE.md defect classes ("no `fail_condition`, or only triggers on full force-wipe; a stall / preserve / partial outcome silently draws"). Since the pack is quarantined the lazy-bar drift is harmless to the default eval; if the pack is ever un-quarantined, add `fail_condition: {after_ticks: 12001}` plus `{not: own_units_gte: 1}`. |
| `spec-spy-infiltrate` | **DRAW** | 3 | 273 | Hard tier `fail_condition: {after_ticks: 4001}` — i.e. the only failure mode is the deadline. But the agent's only combat-relevant units are 2 spies (`spy`) and the scenario also places one defender `e1`. With stall, the spies are passive while the engine auto-`done`s at turn 3 (likely because the defender `e1` falls to auto-fire or because all agent combat units are destroyed). The outcome lands as DRAW because the spy-only force is wiped before the within_ticks ever bites. **Fix**: add `{not: own_units_gte: 1}` (or `unit_type_count_gte:{spy, 1}`) to the `fail_condition.any_of` so a spy-wipe is a LOSS, not a draw. Same fix likely needed for `easy` / `medium` (uninspected; consider auditing). |
| `def-bridge-chokepoint` | (mixed: usually **LOSS** at tick 1623, occasionally **DRAW** in one fresh process) | 18 | 1623 | The first probe (subprocess #1) reported DRAW; three subsequent re-runs from a fresh process consistently reported LOSS (17 losses, fail-clause `not has_building:fact` trips when the procs/fact fall under the rifle assault). This is borderline — the pack is *probably* healthy, but the one-off DRAW observation suggests a possible determinism-edge case (race between the engine auto-`done` and the predicate evaluator). Recommend a 4-seed re-probe (1–4) over a clean Python process to confirm. If the LOSS is stable, no fix required; if the DRAW resurfaces, add `{not: own_units_gte: 1}` to the fail clause. |

### 1C. Engine panics at reset (2)

| Pack | Status | Error |
|---|---|---|
| `adversarial-siege` | `quarantine` (consolidated into `adversarial-duel`) | `pathfinder.rs:175 index out of bounds: len 5120, index 5382` at `env.reset(seed=1)`. The hard tier places actors outside the playable bounds (CLAUDE.md footgun #6) for the chosen spawn. Pack is excluded from default eval; no eval-time impact, but flag the file is still selectable via explicit `--packs`. |
| `adversarial-skirmish` | `quarantine` (consolidated into `adversarial-duel`) | Same panic at `env.reset`. Same situation as above. |

Recommend either deleting these two YAMLs (the consolidation is the
documented future state) or moving them to `packs/_archived/` so
`discover_packs` skips them (it already skips `_*` and `TEMPLATE`).

---

## Section 2 — Duplicate / near-duplicate clusters

Methodology: each pack was fingerprinted by
`(capability, tools, actor_types_sorted, win_predicate_keys_sorted)`.
Five strict (= all four fields equal) clusters surfaced; a second pass
relaxed `tools` and `actor_types` to surface near-duplicates with the
same capability + same win-key set, which surfaced a further ~10
clusters. Each cluster is judged on the per-pack `real_world_meaning`
and the `level_description` of the hard tier — if these tell the same
story, the cluster is genuinely redundant; if not, the predicate-key
match is coincidental and the packs probe distinct skills.

### 2A. Strict duplicates (action required)

1. **`adversarial-siege` + `adversarial-skirmish`** — both already
   quarantined as "consolidated into `adversarial-duel`". **Keep:**
   `adversarial-duel`. **Action:** physical removal (delete or
   `_archive`) — they are the only two packs that crash the engine at
   load, so removing them eliminates the crash class entirely.

2. **`artofwar-decoy-sacrifice` + `artofwar-lure-the-tiger`** — both
   `capability: reasoning`, both use bot `guard`, identical hard-tier
   actor list `[2tnk,e1,e3,fact,jeep]`, identical win-keys
   `{units_in_region_gte, units_lost_lte, within_ticks}`, both
   described as "main-force reaches an objective by diverting a
   leashed defender." **Keep:** `artofwar-lure-the-tiger` (the more
   doctrinally complete framing — leash mechanic; "the strong
   defender" is the load-bearing pull). **Merge / delete:**
   `artofwar-decoy-sacrifice` — its "spend a decoy unit" angle is
   covered by the `units_lost_lte` clause in the lure pack, which
   already permits a small attrition budget the model can spend on
   the bait sub-force.

3. **`scout-cycle-keep-info-fresh` + `scout-track-enemy-movement`** —
   both `capability: perception`, identical hard-tier actor list, both
   use the `then`-chained `units_killed_gte` win predicate, both
   probe "scout multiple times because the world changes between
   observations." Both use the `scheduled_events:` hook (cycle =
   reinforcement waves; track = enemy march legs). **These are
   genuinely distinct skills** — cycle is "re-observe a stale region
   for new arrivals", track is "follow a moving target across the
   map." **Keep both**, but verify the briefing of each clearly
   distinguishes the cause of staleness so the model is not asked to
   solve cycle-by-track-strategy or vice-versa. (No action required if
   the briefings are unambiguous, which they appear to be on
   inspection.)

4. **`build-defensive-skirt-corners` + `build-defensive-tower-line`**
   (strict 4-field match) **AND** the wider quartet
   `build-defensive-tower-cluster` + `def-in-depth-vs-single`
   (same capability + win-key set, near-identical actor list, all use
   `rusher` bot). These four packs all probe "where do you place a
   finite pillbox budget to survive an incoming rush?" with topology
   variants: skirt (one in each map-relative corner of the building),
   cluster (tight wrap), line (across the choke), depth (two thinner
   bands). **Each topology is genuinely distinct in the optimal
   answer**, so all four packs probe a real choice axis. **Keep all
   four** but consider promoting them into a single named "Defense
   Topology Suite" in the eval-cell catalog so the four cells score
   together (one "topology-IQ" metric) rather than as four unrelated
   reasoning packs. (No file action required; this is a catalog /
   metric grouping recommendation.)

### 2B. Near-duplicates by win-predicate keys (informational — same
predicate idiom is used across distinct scenarios; verify each is
load-bearing)

These groups all share an identical `(capability, win_predicate_keys)`
fingerprint. Each group below is **not necessarily a duplicate**, but
flags that all members rely on the same predicate idiom and the
scenarios should diverge on actor composition / pressure / bot to
remain discriminative.

- `(reasoning, then+within_ticks)` — N=10:
  `build-power-online-first`, `build-sequence-tech-cheapest`,
  `build-sequence-tech-fastest`, `build-sequence-tech-most-resilient`
  (not in cluster — was filtered earlier), `lh-econ-army-victory`,
  `lh-opening-to-defense-to-counter`, `lh-opening-to-tech-to-army`,
  `lh-progression-stage-locked`, `rob-objective-change-midway`,
  `tech-balanced-econ-then-tech`,
  `lh-build-army-coordinate-multifront-attack`. These are all
  ordered-multi-phase scenarios. Distinct intents (build-order vs
  long-horizon vs objective-shift); the `then` predicate is the
  common engine machinery, not a duplication signal.

- `(reasoning, building_count_gte+building_in_region+within_ticks)` —
  N=7: `build-sell-and-rebuild-elsewhere`, `def-in-depth`,
  `def-tower-line-vs-cluster`, `expansion-aggro-3-base-greedy`,
  `mcv-deploy-second-base`, `mcv-deploy-third-base`,
  `mfb-mirror-base-east-west`. All probe "build / re-build N
  buildings in a target region." Genuinely distinct intents
  (expansion-greed, mirror-base symmetry, MCV-redeploy under
  pressure); keep all.

- `(reasoning, building_count_gte+units_killed_gte+within_ticks)` —
  N=6: `adv-rps-counter-pick`, `build-tech-skip-decision`,
  `def-counter-battery`, `def-walls-vs-towers`, `def-while-building`,
  `lh-tech-rush-vs-army-rush`. All "kill K enemies and have N
  buildings standing." Reads as honest differentiated tests.

- `(action, own_units_gte+units_killed_gte+within_ticks)` — N=5:
  `combat-flanking-attack`, `combat-harass-aggro-commit`,
  `combat-kite-and-pull`, `combat-kite-jeep-vs-tank`,
  `combat-tanya-vs-rush`. Five combat micro packs — kite, flank,
  harass. Each probes a different micro idiom; keep all but verify
  that the win bar (`units_killed_gte`) is tuned per pack so the
  intended micro is load-bearing, not just "auto-fire wins."
  Two are explicitly kite (`combat-kite-and-pull`,
  `combat-kite-jeep-vs-tank`) — **possible duplicate**: kite-and-pull
  uses generic 2tnk/3tnk vs jeep-vs-tank's unit-class asymmetry.
  Keep both only if `combat-kite-jeep-vs-tank` actually requires the
  jeep speed advantage that `combat-kite-and-pull` doesn't (read the
  briefings; they look distinct).

- `(action, reach_region+units_lost_lte+within_ticks)` — N=4:
  `proc-no-attack-passive-only`, `proc-tool-use-multi-distractor`,
  `proc-tool-use-with-distractor`,
  `strict-toolban-fidelity-under-pressure`. All
  procedural-compliance packs — "reach the goal without using a
  forbidden tool / without attacking." The `proc-tool-use-*` pair is
  a possible duplicate: "with-distractor" (one distractor tool) vs
  "multi-distractor" (multiple). Both are the same skill at different
  scales — recommend collapsing into a single pack with three tiers
  (no distractor / one / many) rather than two top-level packs.

- `(perception, buildings_discovered_gte+units_lost_lte+within_ticks)`
  — N=4: `perception-target-vs-fog`, `scout-detect-enemy-tech`,
  `scout-discover-hidden-base`, `scout-multiple-fog-areas`. All
  "discover K enemy buildings under fog." Each probes a distinct
  discrimination (target-vs-fog = ignore decoys; detect-enemy-tech =
  read the tech-tree; hidden-base = single hidden compound;
  multiple-fog-areas = K disjoint regions). Keep all.

### 2C. Overlapping prefixes worth a catalog review (not duplicates,
but the area is dense)

- `combat-*` (25 packs), `def-*` (18), `scout-*` (14), `build-*` (14),
  `lh-*` (13), `proc-*` (10), `mfb-*` (8), `rob-*` (8), `tp-*` (7),
  `coord-*` (7), `mcv-*` (6), `spec-*` (5), `economy-*` (5),
  `tech-*` (4), `strategy-*` (4), `perception-*` (4), `artofwar-*` (4),
  `tempo-*` (2), `maint-*` (2), `mid-*` (2), `expansion-*` (3),
  `coordination-*` (2), `harass-*` (1), `risk-*` (1), `power-*` (1),
  `navigation-*` (1), `defense-*` (1), `longhorizon-*` (1),
  `custom-*` (1), `building-*` (1), `rush-*` (1), `reasoning-*` (2),
  `action-*` (2), `adv-*` (2), `adversarial-*` (3), `strict-*` (3).
  The combat / def / scout / build axes carry ~70 packs combined; a
  follow-up audit explicitly checking pairwise overlap inside each
  prefix (briefing-level read) would surface ~5–10 more candidate
  collapses.

---

## Section 3 — Capability coverage matrix

Cross-referenced against the **phase × decision-type** matrix sketched
in `PAPER_PLAN.md §12.2` and `CLAUDE.md`. Each cell lists the packs
that primarily measure it (a pack may legitimately serve more than one
cell — only the dominant assignment is listed; cross-cutting packs are
flagged).

### Opening

| Decision | Packs | Density |
|---|---|---|
| MCV deploy site (where to plant) | `mcv-deploy-and-build`, `mcv-deploy-defensible-site`, `mcv-deploy-near-resource`, `mcv-deploy-relocate-under-pressure`, `mcv-deploy-second-base`, `mcv-deploy-third-base` | **6 packs** (dense; good) |
| Build-order commit | `build-sequence-tech-cheapest`, `build-sequence-tech-fastest`, `build-sequence-tech-most-resilient`, `build-tech-skip-decision`, `tech-balanced-econ-then-tech`, `tech-aggro-all-in`, `tech-turtle-defensive-tech`, `tech-production-planning`, `build-power-online-first`, `power-budget-online`, `building-and-planning` | **11 packs** (dense; good) |
| Defense-direction commit (which side to anticipate) | `def-position-expected-direction`, `def-position-revealed-direction`, `def-multi-direction`, `def-surprise-flank-react`, `def-pre-position-mobile-reserve` | **5 packs** (good) |
| Rush-defense | `defense-rush-survive`, `rush-hour`, `build-defensive-tower-cluster`, `build-defensive-tower-line`, `build-defensive-skirt-corners`, `def-in-depth`, `def-in-depth-vs-single`, `def-walls-vs-towers`, `def-tower-line-vs-cluster`, `def-while-building` | **10 packs** (dense) |

### Early-mid

| Decision | Packs | Density |
|---|---|---|
| Harass / harass-preserve | `harass-response-preserve`, `combat-harass-aggro-commit`, `combat-harass-balanced-hit-and-run`, `combat-skirmish-then-disengage`, `combat-bait-counter-attack`, `combat-kite-and-pull`, `combat-kite-jeep-vs-tank`, `combat-retreat-after-engagement`, `combat-suicide-charge-mission` | **9 packs** (dense) |
| Exact-count perception | `perception-count-the-threat`, `perception-count-the-threat-small-k`, `scout-count-defenders`, `scout-detect-incoming-army` | **4 packs** (adequate) |
| Scout-direction commit | `scout-detect-base-direction`, `scout-far-frontier`, `scout-frontier-reading` (=`perception-frontier-reading`), `scout-multiple-fog-areas`, `scout-and-report`, `scout-and-survive`, `scout-discover-hidden-base`, `scout-detect-enemy-tech`, `scout-map-reveal-percent-target`, `reasoning-frontier-commit` | **10 packs** (dense; possible over-coverage) |

### Mid

| Decision | Packs | Density |
|---|---|---|
| Live-economy defense | `mid-economy-under-fire` (**DEFECT — stall wins**), `econ-harvester-defense-raid` [skipped — econ-*], `econ-protect-harvester-route` [skipped] | **~3 packs, 1 defective** — the only non-econ exemplar is broken. **Gap when the in-flight econ-* land — verify they cover.** |
| Tech-switch on scout | `mid-tech-switch-on-scout`, `lh-scout-react-counter`, `adv-rps-counter-pick` | **3 packs** (adequate) |
| Second front | `mfb-base-1-defend-base-2-build`, `mfb-supply-line-link-between-bases`, `mfb-mirror-base-east-west`, `mfb-two-base-simultaneous`, `mfb-third-base-against-clock`, `expansion-balanced-2-base-defended`, `expansion-aggro-3-base-greedy`, `coord-diversionary-attack` | **8 packs** (dense) |
| Replan after loss | `rob-unit-loss-recovery`, `rob-partial-base-loss-continue`, `lh-recovery-after-mid-game-loss`, `build-engineer-rebuild-after-loss`, `def-retreat-and-rebuild`, `build-sell-and-rebuild-elsewhere` | **6 packs** (good) |

### Mid-late

| Decision | Packs | Density |
|---|---|---|
| Concede vs hold | `mid-concede-vs-hold` | **1 pack** (THIN — single exemplar) |
| Isolate vs split | `combat-divide-and-conquer`, `combat-pincer-coordination`, `combat-prevent-retreat`, `combat-formation-tank-wedge` | **4 packs** (adequate) |
| Tempo double-window | `tempo-double-window`, `tempo-strike-window`, `coordination-staggered-window`, `tp-survive-and-strike-at-window` | **4 packs** (adequate) |
| Decoy / lure / feint | `artofwar-decoy-sacrifice`, `artofwar-lure-the-tiger`, `artofwar-indirect-approach`, `artofwar-sequenced-citadel`, `combat-bait-counter-attack`, `coord-diversionary-attack`, `def-with-ambush` | **7 packs** (good) |
| Multi-front | `mfb-rotating-production-pressure`, `mfb-redundant-tech-buildings`, `mfb-tech-base-vs-economy-base`, `rob-multiple-simultaneous-pressures` | **4 packs** (adequate) |

### Late

| Decision | Packs | Density |
|---|---|---|
| Sustained multi-front | `lh-build-army-coordinate-multifront-attack`, `mfb-rotating-production-pressure` | **2 packs** (thin) |
| Base-trade race | None directly | **0 packs (GAP)** |
| Counter-strategy read | `adv-rps-counter-pick`, `combat-attack-from-behind-fog`, `lh-tech-rush-vs-army-rush` | **3 packs** (adequate) |
| Superweapon timing | `spec-nuke-strike`, `spec-tanya-c4-strike`, `spec-engineer-capture`, `spec-spy-infiltrate` (**DEFECT — stall draws**), `spec-thief-steal-cash` | **5 packs, 1 defective** (good once spec-spy is fixed) |
| Credit-only final phase | `lh-credit-only-final-phase` | **1 pack** (thin) |

### Cross-cutting

| Capability | Packs | Density |
|---|---|---|
| Procedural compliance under pressure | `proc-checklist-no-deviation`, `proc-conditional-branch-action`, `proc-instruction-following-edge-case`, `proc-no-attack-passive-only`, `proc-only-build-no-combat`, `proc-only-defend-no-attack`, `proc-ordered-action-strict`, `proc-strict-toolban-fidelity`, `proc-tool-use-multi-distractor`, `proc-tool-use-with-distractor`, `strict-production-bom`, `strict-sequence`, `strict-toolban-fidelity-under-pressure`, `tp-pressure-procedural` | **14 packs** (very dense) |
| Long-horizon multi-phase | `lh-*` (13 packs), `longhorizon-opening-to-assault`, `lh-100-turn-marathon-survival` | **14 packs** (dense) |
| Coordination across squads | `coord-converge-on-target`, `coord-cover-and-move`, `coord-diversionary-attack`, `coord-mutual-support`, `coord-relay-attack`, `coord-relay-vision-chain`, `coord-squad-handoff`, `coordination-ordered-rendezvous`, `coordination-staggered-window`, `action-multiunit-coordination`, `action-sequenced-execution`, `combat-pincer-coordination`, `combat-heli-flank` | **13 packs** (dense) |
| Adversarial 1v1 (full macro) | `adversarial-duel`. The full 1v1 battleground lives in `one_v_one.py` (`openra_bench/`), not as a `meta.capability: adversarial` pack. | **1 pack** (the catalog says this is by design — full macro is the live ladder) |

---

## Section 4 — Capability gaps (uncovered or thin cells)

Cells worth new packs, in priority order:

1. **Base-trade race (mid-late / late)** — *no* current pack tests
   "your base falls while you race the enemy's; whoever finishes first
   wins." Sketch: hard tier places both players' bases halfway to
   killable, no defenders left; the agent has one army worth attacking
   with, the enemy is doing the same to the agent's base; win by
   destroying the enemy's `fact` before they destroy yours. Win
   `building_count_gte:{type:fact,n:0,owner:enemy}` paired with
   `has_building:fact` (own) and an aggressive `within_ticks`. Pack id:
   `combat-base-trade-race`.

2. **Concede-vs-hold (mid-late)** — only `mid-concede-vs-hold`
   currently. Sketch a second pack: `mid-concede-vs-rebuild` where a
   forward outpost is lost-cost (committing reinforcements throws
   good money after bad), and the right call is to abandon it and
   rebuild on a held line. Win condition: keep the rear-line `fact`
   alive AND have ≥4 `tent`/`weap` on the rear line at deadline;
   throwing reinforcements at the forward outpost fails the resource
   budget. Pack id: `mid-concede-forward-rebuild-rear`.

3. **Live economy defense (mid)** — the only non-econ pack
   (`mid-economy-under-fire`) is currently a stall-WIN defect. After
   fixing it, add a sibling pack `mid-economy-rebuild-harvester-line`
   where the agent must re-issue `harvest` commands to a freshly
   produced harvester after a raider kills the starter ones (puts
   the load-bearing capability on the `harvest` order, not on the
   passive auto-harvest).

4. **Long-horizon credit-only final phase** — `lh-credit-only-final-phase`
   is the only exemplar. Sketch sibling: `lh-credit-only-bait-window` —
   the agent must spend the last credits on a single decisive strike
   window rather than on attrition. Different decision (one shot vs
   accumulation).

5. **Sustained multi-front (late)** — thin (2 packs). Sketch:
   `mfb-three-front-rotation` — three simultaneously-attacked bases
   with the agent's army too small to defend all three at once; the
   decision is which two to hold and which one to let fall while the
   army cycles. Cross-pack with `coord-relay-attack`.

6. **MCV deploy under timer + crossfire** — current MCV packs cover
   site selection but not "deploy NOW or the MCV is destroyed by an
   incoming wave that hits in 4 turns." Sketch:
   `mcv-deploy-emergency-relocation` — the starting MCV stands in the
   path of an incoming squad; the agent must deploy + re-build OR
   move + re-deploy further west before the squad arrives.

7. **Information freshness across modalities** — `scout-cycle-keep-info-fresh`
   and `scout-track-enemy-movement` cover this for the structured
   channel, but the perception ablation grid (channel × fog) doesn't
   currently have an information-freshness pack that is explicitly
   easier to solve with the labelled image (the `image` channel is
   advantaged when the model can spot newly-spawned units
   inter-tick). Sketch: `perception-freshness-image-advantage`.

8. **Single-pack `meta.capability: adversarial`** — only
   `adversarial-duel` carries this tag. PAPER_PLAN.md §12.2 notes the
   imbalance. The full 1v1 lives in `one_v_one.py` (correct), but a
   second adversarial-tagged pack that tests "reactive opponent
   selects a counter from a small menu mid-game" (different from RPS
   pre-game commit) would put real teeth on the tag.

9. **Engineer / Tanya / spy under fire** (specialist packs) — only one
   pack each (`spec-engineer-capture`, `spec-tanya-c4-strike`,
   `spec-spy-infiltrate`, `spec-thief-steal-cash`, `combat-tanya-vs-rush`).
   Each specialist has one happy-path scenario. Sketch a "stealth +
   target priority" pack per specialist that requires the model to
   pick *which* of N enemy assets to hit with the one-shot specialist.

10. **Naval / amphibious** — `combat-naval-shore-strike` is the only
    naval pack (and it's currently a stall-WIN defect). Once fixed,
    add `combat-naval-amphibious-landing` (a transport ship lands
    infantry on a contested shore) and `combat-naval-anti-air-vs-bomber`
    pending the air-unit engine work. These are documented as
    out-of-scope in `PAPER_PLAN.md §11.1` for the air variant but the
    naval ones are now feasible thanks to `water_rect:`.

---

## Section 5 — Map mis-bindings (rush-hour-arena used where intent
calls for a custom map)

Of 189 probed packs, **183** use `rush-hour-arena`; **3** use a custom
map (`navigation-confined-hard-only`, `custom-map-no-enemy`,
`strategy-dilemma`/`-gauntlet`/`-twobody`); **1** uses a generator
(`combat-naval-shore-strike`); **2** are quarantined-and-crashing.

Packs whose stated intent calls for a non-rush-hour geometry:

| Pack | Stated intent | Current map | Recommendation |
|---|---|---|---|
| `combat-heli-flank` | Helicopter assault from a flank; the brief explicitly references aircraft flight over terrain. | rush-hour-arena (no terrain features to flank around) | Author / use a custom map with a forested ridge that channels ground units one way and aircraft another. (Engine air-unit support is out-of-scope per PAPER_PLAN §11.1 — pack is currently a ground-only proxy.) |
| `def-bridge-chokepoint` | "A water band cuts the map east-to-west with bridges." | rush-hour-arena + `water_rect` overlay | The `water_rect` overlay correctly synthesizes the bridge geometry on top of rush-hour, but the chokepoint-narrowing aspect ("attackers must pass through 3 narrow openings") works against an open-arena map. Consider a custom `bridges-arena` map authored once — the visual / minimap reading test is more honest when the bridges are real terrain rather than a YAML overlay on an unrelated map. |
| `combat-naval-shore-strike` | Naval ship in a water channel. | uses `naval-arena` generator (correct) — but the generator spec lives in the pack file rather than a named `.oramap`. | When a second naval pack is authored (per gap #10), promote `naval-arena` to a real `data/maps/naval-arena.oramap` so the bench has a canonical naval geometry. |
| `def-with-ambush` | "Concealed flanking defenders catch the band in an L-ambush down a lane toward the construction yard." Doctrinal answer requires real linear terrain. | rush-hour-arena (open) | The pack synthesizes the lane purely with actor placement (a single `e1` fixing defender at x=15, flankers at x=40). Visually the geometry is invisible on the minimap — a custom `corridor-arena` map with a real lane would make the spatial decision legible from the image. (Capability is sound; this is a perception-channel polish issue.) |
| `mfb-supply-line-link-between-bases` | "Supply line corridor between two bases." | rush-hour-arena | Similar polish issue — a custom multi-base map with two separated valleys would make the supply-line decision visible on the minimap rather than implicit in the actor coordinates. |
| `combat-hold-chokepoint` | "Defend a narrow chokepoint corridor." | rush-hour-arena (no narrow corridor) | Same; either author a `chokepoint-arena` map or accept that the chokepoint is implied by enemy spawn geometry alone. |
| `mcv-deploy-second-base`, `mcv-deploy-third-base` | "Plant a second / third base at a defensible site." | rush-hour-arena (uniform; no defensible sites) | The map is symmetric and uniform — every cell is roughly as defensible as every other. The capability becomes "pick a cell that satisfies the `building_in_region` predicate," not "pick a *defensible* cell." Add a custom map with terrain features (cliff edges, narrow passes) so the defensibility geometry is real. |
| `expansion-aggro-3-base-greedy`, `expansion-balanced-2-base-defended`, `expansion-turtle-1-base-fortified` | Expansion decisions over distinct base sites. | rush-hour-arena (only ~3 sensible base sites given map size) | Authoring a wider arena with 5+ distinguishable expansion sites would put real teeth on the trilemma. |

**Summary**: roughly 8–10 packs would benefit from a custom map, but
all are currently *functional* (the no-cheat bar holds). The map
mis-binding is a perception-channel polish issue, not a correctness
issue. **Priority**: author 2 custom maps that unlock several packs
each:
1. `bridges-arena` — would replace `water_rect` overlay on
   `def-bridge-chokepoint` and could host `combat-naval-amphibious-landing`.
2. `chokepoint-arena` — narrow corridor for `combat-hold-chokepoint`,
   `def-with-ambush`, `mfb-supply-line-link-between-bases`.

---

## Section 6 — Recommended actions (prioritized)

### P0 — Hard defects (gates the no-cheat headline number)

1. **Fix `mid-economy-under-fire`** — currently the only stall-WIN
   defect among non-exempt packs. Either:
   - (a) Remove the 3 starter harvesters; require the model to issue
     `Command.harvest(...)` on a fresh harvester (the load-bearing
     capability), OR
   - (b) Upgrade the raider to a force that out-attritions an idle
     defense ring (e.g. 3× `1tnk` + 2× `e3` over 60 turns), so a
     stall play loses harvesters and the `harv,2` clause fails.

2. **Fix `combat-naval-shore-strike`** — set the destroyer `stance` to
   `0` (HoldFire) so the agent must issue an explicit `attack_unit`
   order. Verify the intended capability (manual cross-shore attack)
   still wins; stall now loses by deadline.

3. **Fix `spec-spy-infiltrate`** — add `{not: own_units_gte: 1}` (or
   `{not: unit_type_count_gte: {type: spy, n: 1}}`) to the
   `fail_condition.any_of` so a spy-wipe is a real LOSS, not a
   DRAW. Audit the easy / medium tiers for the same defect.

4. **Delete or `_archive/` `adversarial-siege` and `adversarial-skirmish`** —
   both already quarantined; deletion removes the only two
   load-time-crashing packs.

5. **Re-probe `def-bridge-chokepoint` over seeds 1–4 in clean processes**
   — confirm the one-off DRAW observation was a transient (subsequent
   re-runs gave LOSS consistently); if the DRAW recurs, add
   `{not: own_units_gte: 1}` to its `fail_condition.any_of`.

### P1 — Coverage gaps (new packs to author)

In priority order from §4: `combat-base-trade-race`,
`mid-concede-forward-rebuild-rear`, `mid-economy-rebuild-harvester-line`,
`mfb-three-front-rotation`, `mcv-deploy-emergency-relocation`,
`lh-credit-only-bait-window`, `perception-freshness-image-advantage`,
`combat-naval-amphibious-landing`, and per-specialist target-priority
packs (`spec-engineer-priority-targets`, `spec-tanya-priority-targets`).

### P2 — Duplicate consolidation

1. Merge / delete `artofwar-decoy-sacrifice` into
   `artofwar-lure-the-tiger`.
2. Merge `proc-tool-use-with-distractor` + `proc-tool-use-multi-distractor`
   into a single pack with two tiers.
3. Re-read `combat-kite-and-pull` vs `combat-kite-jeep-vs-tank` and
   either merge or sharpen the asymmetry-of-units distinction in the
   second pack's briefing.

### P3 — Map / catalog polish

1. Author `data/maps/bridges-arena.oramap` and re-bind
   `def-bridge-chokepoint` + (future) `combat-naval-amphibious-landing`.
2. Author `data/maps/chokepoint-arena.oramap` and re-bind
   `def-with-ambush`, `combat-hold-chokepoint`,
   `mfb-supply-line-link-between-bases`.
3. Promote the four "defensive topology" packs
   (`build-defensive-tower-cluster`, `build-defensive-tower-line`,
   `build-defensive-skirt-corners`, `def-in-depth-vs-single`) into a
   single grouped eval-cell suite so they score one "topology-IQ"
   metric.

---

## Appendix — Probe methodology

Each pack was probed by:

```python
import openra_train  # builds the engine; required for env.reset
from openra_bench.scenarios.loader import compile_level, load_pack
from openra_bench.eval_core import run_level

pack = load_pack(pack_yaml_path)
c = compile_level(pack, "hard")
ep = run_level(c, lambda rs, Command: [Command.observe()], seed=1)
# ep.outcome ∈ {"win", "loss", "draw"}; ep.signals.game_tick = final tick.
```

The driver (`/tmp/stall_probe_driver.py`) invoked one subprocess per
pack so a Rust engine panic in any one pack did not abort the run.
189 packs probed in 48 seconds wall-time (Apple M-series).

Hard tier seed=1 only. A full audit would extend to seeds 1–4 (the
documented "hard seed" range in CLAUDE.md) and to easy/medium tiers,
but seed-1 hard is the highest-pressure cell per pack — defects
visible on any tier are visible here.

Static profile was extracted from the compiled `CompiledLevel`
(authoritative — the engine sees the merged scenario, not the raw
YAML).

Raw probe results JSON: `/tmp/stall_probe_results.json` (regeneratable
by re-running `/tmp/stall_probe_driver.py`).
