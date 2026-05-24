# Tier-Progression Audit — Hard-is-Hard-but-Solvable Mathematical Check

Sweep of every active pack × tier (210 packs × 3 = 630 tier-cells)
for monotonicity defects (hard easier than medium on any axis) and
solvability gaps (intended capability physically unable to win).

**Working method (binding):** parse every pack YAML, extract for each
tier the deadline (`within_ticks`/`after_ticks`), reachable max tick
(`93 + 90·(max_turns−1)`), agent / enemy force value (per
`engine_unit_audit.csv` vendor-runtime costs), kill quota, survival
floor + strictness, max-allowed losses, cash goal, scheduled-event
count, win/fail predicate tree size, hard spawn-group count, build-tool
presence. Score each pack on monotonic escalation (survival, kill,
enemy-value, deadline) and tag defects.

**Source of truth:** `audits/tier_progression_audit.csv` (one row per
pack, 71 columns); engine unit costs from `audits/engine_unit_audit.csv`
(`cost_engine` = vendor-runtime, the canonical column).

## 1. Headline counts

- **Total packs audited:** 210
- **OK (no flagged defect):** 184 (88%)
- **Flagged:** 26

Monotonicity score (kill_quota / max_losses / enemy_value /
effective_deadline; +1 per strictly-harder transition, −1 per inversion):

- `weak <50%         `: 119
- `flat              `: 34
- `negative          `: 33
- `partial >=50%     `: 23
- `perfect           `: 1

Most packs land in `weak` (small ABSOLUTE numeric monotonic improvement)
or `flat` — this is NOT a defect signal by itself. Many packs escalate
difficulty by changing AXIS rather than knob (e.g., a bigger map ⇒
looser deadline, but the capability test gets harder). Only the F-INVERT
class — where a HIGHER tier is STRICTLY EASIER on a survival/kill
axis — represents a structural defect.

## 2. Defect histogram

| Class | Count | Meaning |
|---|---|---|
| F-INVERT             | 18 | hard / medium ALLOWS MORE LOSSES (or LOWER kill quota) than the next-lower tier — survival_strictness or kill_quota regression |
| F-INVERT-ENEMY      | 6 | hard enemy_value drops >15% vs medium (smaller enemy on higher tier) |
| F-FLAT              | 0 | every quantitative axis identical across e/m/h (no escalation at all) |
| F-TRIVIAL-HARD      | 1 | medium ≡ hard on every measured axis AND on schedule/spawn/predicate count |
| F-NO-SEED-ROTATION  | 4 | hard tier has <2 spawn groups AND no scheduled_events — likely fails test_hard_tier or trivially memorizable |
| F-UNSOL-TICK / -CASH / -FORCE | 0 / 0 / 0 | with the build-capability + special-verb exclusions, no tier is statically unsolvable |
| F-NEAR-IMPOSSIBLE   | 0 | no tier with <5% solvability margin under the heuristic |

**Note on F-UNSOL = 0**: with `has_build` and `has_special_verb`
exclusions (build-axis packs can grow their force; thief/spy packs
can drain enemy cash without harvesting), no tier is statically
unsolvable under the heuristic. Earlier passes flagged
`spec-thief-steal-cash` / `strategy-trilemma` / `economy-force-buildup`
/ `economy-investment` / `economy-time-box` / `longhorizon-opening-to-assault`
but each has build or special-verb tools and is intentionally solvable.

## 3. F-INVERT (survival / kill-quota regression) — the F9 inversion class

23 packs flagged. The F9 archetype
(combat-flanking-attack pre-fix) was hard allowing more PROPORTIONAL
losses than medium — this is the structural defect we hunt.

### 3a. STRUCTURAL inversions (survival_strictness drops or max_losses rises with same headcount)

These are the highest-priority fixes — the agent is allowed to be
PROPORTIONALLY sloppier at a higher tier.

| pack | break | recommendation |
|---|---|---|
| `action-sequenced-execution` | effective_deadline: easy→medium LOOSER (e=2400, m=3000); effective_deadline: medium→hard LOOSER (m=3000, h=5000); surviv | tighten hard survival floor |
| `combat-formation-tank-wedge` | max_losses: easy→medium LOOSER (e=1, m=2); survival_strictness e=0.80 > m=0.60 | tighten hard survival floor |
| `combat-naval-shore-strike` | max_losses: easy→medium LOOSER (e=0, m=1); survival_strictness e=1.00 > m=0.50 | tighten hard survival floor |
| `def-multi-direction` | survival_strictness m=0.43 > h=0.33 | tighten hard survival floor |
| `def-position-expected-direction` | max_losses: medium→hard LOOSER (m=4, h=5); survival_strictness m=0.43 > h=0.38 | tighten hard survival floor |
| `defense-rush-survive` | max_losses: medium→hard LOOSER (m=0, h=3); survival_strictness m=1.00 > h=0.75 | tighten hard survival floor |
| `expansion-aggro-3-base-greedy` | max_losses: easy→medium LOOSER (e=1, m=2); survival_strictness e=0.50 > m=0.33 | tighten hard survival floor |
| `expansion-turtle-1-base-fortified` | max_losses: easy→medium LOOSER (e=9, m=18); max_losses: medium→hard LOOSER (m=18, h=21); survival_strictness e=0.25 > m= | tighten hard survival floor |
| `harass-response-preserve` | survival_strictness m=0.83 > h=0.71 | tighten hard survival floor |
| `lh-100-turn-marathon-survival` | max_losses: medium→hard LOOSER (m=7, h=11); effective_deadline: easy→medium LOOSER (e=7200, m=8100); effective_deadline: | tighten hard survival floor |
| `rob-unexpected-enemy-spawn` | max_losses: medium→hard LOOSER (m=5, h=7); survival_strictness m=0.29 > h=0.22 | tighten hard survival floor |
| `spec-thief-steal-cash` | max_losses: medium→hard LOOSER (m=0, h=1); effective_deadline: easy→medium LOOSER (e=1350, m=1800); effective_deadline:  | tighten hard survival floor |

### 3b. REVIEW-NEEDED (deadline LOOSENS or kill_quota DROPS on higher tier — often intentional axis shift)

These are flagged but may be intentional — e.g., `combat-retreat-after-engagement`
lowers the kill bar on hard because the capability is RETREAT, not
kills; `adv-rps-counter-pick` lowers kill_quota on hard because the
test is COUNTER-PICKING the right unit, not raw kill volume.
MANUAL REVIEW required to decide:

| pack | break | likely call |
|---|---|---|
| `adv-rps-counter-pick` | kill_quota: medium→hard LOOSER (m=8, h=5); enemy_value: medium→hard LOOSER (m=3200, h=2600); kill_quota m=8 > h=5 | VERIFY — capability shift may justify lower bar |
| `combat-hold-chokepoint` | kill_quota: medium→hard LOOSER (m=11, h=9); kill_quota m=11 > h=9 | VERIFY — capability shift may justify lower bar |
| `combat-retreat-after-engagement` | kill_quota: medium→hard LOOSER (m=3, h=2); enemy_value: medium→hard LOOSER (m=4050, h=3750); kill_quota m=3 > h=2 | VERIFY — capability shift may justify lower bar |
| `combat-tank-vs-tank-engagement` | enemy_value: medium→hard LOOSER (m=5400, h=4550) | VERIFY |
| `def-in-depth-vs-single` | kill_quota: medium→hard LOOSER (m=7, h=6); enemy_value: medium→hard LOOSER (m=3400, h=3200); kill_quota m=7 > h=6 | VERIFY — capability shift may justify lower bar |
| `def-tower-line-vs-cluster` | effective_deadline: medium→hard LOOSER (m=5400, h=6300); kill_quota: medium→hard LOOSER (m=10, h=8); enemy_value: medium | VERIFY — capability shift may justify lower bar |
| `def-while-building` | kill_quota: medium→hard LOOSER (m=7, h=5); enemy_value: medium→hard LOOSER (m=4900, h=4800); kill_quota m=7 > h=5 | VERIFY — capability shift may justify lower bar |
| `econ-protect-harvester-route` | effective_deadline: easy→medium LOOSER (e=4500, m=5400); enemy_value: medium→hard LOOSER (m=4850, h=2850) | INTENTIONAL — bigger map ⇒ more travel ticks |
| `perception-target-vs-fog` | effective_deadline: easy→medium LOOSER (e=2000, m=2300); effective_deadline: medium→hard LOOSER (m=2300, h=2700); enemy_ | INTENTIONAL — bigger map ⇒ more travel ticks |
| `risk-blockade-bypass` | enemy_value: medium→hard LOOSER (m=9200, h=4500) | VERIFY |
| `tp-survive-n-turns` | effective_deadline: easy→medium LOOSER (e=3600, m=4500); effective_deadline: medium→hard LOOSER (m=4500, h=5400); enemy_ | INTENTIONAL — bigger map ⇒ more travel ticks |

## 4. F-INVERT-ENEMY (enemy force drops >15% on higher tier)

6 packs. Often paired with a CAPABILITY swap —
the hard tier may swap fewer-but-stronger enemies for the medium
composition, or test an entirely new axis. Worth a manual sanity check.

| pack | easy enemy $ | medium enemy $ | hard enemy $ |
|---|---|---|---|
| `adv-rps-counter-pick` | 2800 | 3200 | 2600 |
| `combat-tank-vs-tank-engagement` | 4550 | 5400 | 4550 |
| `econ-protect-harvester-route` | 2850 | 4850 | 2850 |
| `perception-target-vs-fog` | 2300 | 3500 | 2700 |
| `risk-blockade-bypass` | 9200 | 9200 | 4500 |
| `tp-survive-n-turns` | 3800 | 3800 | 3200 |

## 5. F-NO-SEED-ROTATION (hard has only 1 spawn group and no scheduled_events)

Per CLAUDE.md, `test_hard_tier` requires hard to produce ≥2 distinct
seed-driven spawns. These packs deterministically reproduce the same
layout across all seeds, defeating the anti-memorisation purpose
of the hard tier:

| pack | hard spawn_groups | hard sched_events |
|---|---|---|
| `combat-naval-shore-strike` | 0 | 0 |
| `def-bridge-chokepoint` | 0 | 0 |
| `proc-only-build-no-combat` | 0 | 0 |
| `tempo-strike-window` | 0 | 0 |

## 6. F-TRIVIAL-HARD

Strict triviality: hard ≡ medium on EVERY measured axis (kill quota,
survival, deadline, enemy value, agent count, enemy count, spawn
groups, scheduled_events count, win-predicate count). After filtering
out packs that escalate via different metrics, only **1** pack
triggers:

| pack | note |
|---|---|
| `proc-only-build-no-combat` | medium ≡ hard on every measured axis (also F-NO-SEED-ROTATION) |

## 7. Top 10 worst by defect-class load

| # | pack | defects | breaks |
|---|---|---|---|
| 1 | `adv-rps-counter-pick` | F-INVERT|F-INVERT-ENEMY | kill_quota: medium→hard LOOSER (m=8, h=5); enemy_value: medium→hard LOOSER (m=3200, h=2600); kill_qu |
| 2 | `combat-naval-shore-strike` | F-INVERT|F-NO-SEED-ROTATION | max_losses: easy→medium LOOSER (e=0, m=1); survival_strictness e=1.00 > m=0.50 |
| 3 | `proc-only-build-no-combat` | F-NO-SEED-ROTATION|F-TRIVIAL-HARD |  |
| 4 | `action-sequenced-execution` | F-INVERT | effective_deadline: easy→medium LOOSER (e=2400, m=3000); effective_deadline: medium→hard LOOSER (m=3 |
| 5 | `combat-formation-tank-wedge` | F-INVERT | max_losses: easy→medium LOOSER (e=1, m=2); survival_strictness e=0.80 > m=0.60 |
| 6 | `combat-hold-chokepoint` | F-INVERT | kill_quota: medium→hard LOOSER (m=11, h=9); kill_quota m=11 > h=9 |
| 7 | `combat-retreat-after-engagement` | F-INVERT | kill_quota: medium→hard LOOSER (m=3, h=2); enemy_value: medium→hard LOOSER (m=4050, h=3750); kill_qu |
| 8 | `combat-tank-vs-tank-engagement` | F-INVERT-ENEMY | enemy_value: medium→hard LOOSER (m=5400, h=4550) |
| 9 | `def-bridge-chokepoint` | F-NO-SEED-ROTATION | effective_deadline: easy→medium LOOSER (e=2700, m=3150); effective_deadline: medium→hard LOOSER (m=3 |
| 10 | `def-in-depth-vs-single` | F-INVERT | kill_quota: medium→hard LOOSER (m=7, h=6); enemy_value: medium→hard LOOSER (m=3400, h=3200); kill_qu |

## 8. Family-level summary

| Family | OK / total | OK% | F-INVERT | F-INVERT-ENEMY | F-NO-SEED |
|---|---|---|---|---|---|
| F10_special               | 5/6 | 83% | 1 | 0 | 0 |
| F11_full_game             | 1/1 | 100% | 0 | 0 | 0 |
| F1_combat_micro           | 22/29 | 76% | 6 | 1 | 1 |
| F2_economy                | 27/28 | 96% | 0 | 1 | 0 |
| F3_defense                | 15/22 | 68% | 6 | 0 | 1 |
| F4_perception             | 18/19 | 95% | 0 | 1 | 0 |
| F5_longhorizon            | 13/14 | 93% | 1 | 0 | 0 |
| F6_build_tech             | 17/17 | 100% | 0 | 0 | 0 |
| F7_procedure              | 21/23 | 91% | 1 | 0 | 1 |
| F8_multifront             | 23/23 | 100% | 0 | 0 | 0 |
| F9_tempo_strategy         | 22/28 | 79% | 3 | 3 | 1 |

**Cleanest:** F6 build_tech and F8 multifront (100% OK) — both
families authored after the F9 inversion was triaged, so the survival
axis is tier-monotonic by construction.

**Most defects:** F3 defense (68%) and F9 tempo_strategy (75%) — many
defense packs use kill-quota or survival-floor regressions on hard
that may or may not be intentional capability shifts.

## 9. Recommendations (prioritised backlog)

### Priority 1 — structural inversions (survival/strictness)

12 packs. Fix by tightening the hard tier's
`own_units_gte` floor so it represents the SAME OR LOWER allowed-loss
FRACTION than medium. Recipe (per CLAUDE.md F9 fix):
`hard_floor = ceil(hard_agent_count * medium_floor / medium_agent_count)`.
Worked example: combat-flanking-attack (already fixed) used
medium 3 of 4 = 75% ⇒ hard 6 of 8 = 75%.

### Priority 2 — review kill-quota / enemy-value drops (24+6 packs)

Spot-check each: is the lower kill bar / smaller enemy on hard
INTENTIONAL (capability shift to micro / counter-pick / retreat)?
If so, document the reason in the pack header. If not, escalate the
bar.

### Priority 3 — seed-rotation (4 packs)

Add a 2nd spawn group OR scheduled_events anti-memorisation hook to
`combat-naval-shore-strike`, `def-bridge-chokepoint`,
`proc-only-build-no-combat`, `tempo-strike-window`. If the pack's
capability genuinely doesn't admit spawn rotation, declare the
`NOT_APPLICABLE` exception in `tests/test_hard_tier.py`.

### Priority 4 — Triviality (1 pack)

`proc-only-build-no-combat` — escalate medium→hard on cash, build
target, or time budget. Currently identical on every measured axis.

## 10. Estimated fix scope

- **Structural fixes:** 12 packs (raise hard survival floor / reduce hard max_losses).
- **Intentional-or-bug review:** 17 packs (kill-quota / enemy-value drops).
- **Seed-rotation fixes:** 4 packs (2nd spawn_point or scheduled_events).
- **Triviality fixes:** 1 pack (proc-only-build-no-combat escalation).
- **Total flagged:** 26 of 210 (12%).

## 11. Caveats / limitations

- This is a STATIC YAML scan. Dynamic difficulty (mid-episode
  reinforcements via `scheduled_events`, scout-cycle re-observation
  freshness, intermediate `then:` chains, perception/fog escalation
  via the hard-tier `objective_coords: relative`) is not captured
  in the metric set.
- The kill-quota TICK heuristic is conservative (~80 ticks/kill +
  200-tick travel); real engagements vary widely with map size and
  weapon range, so F-UNSOL-TICK is gated by a 50% margin to avoid
  false positives.
- The CASH heuristic assumes ~30 cash/tick per harvester and is
  disabled for packs with `build` / `deploy` / `place_building` /
  `infiltrate` / `fire_superweapon` tools (the agent can spawn new
  harvs or steal/drain cash mid-episode).
- `survival_strictness > 1` indicates the `own_units_gte` is a BUILD
  TARGET (not a preserve constraint); these are excluded from the
  inversion check via the `has_build` flag.
- `spawn_point` actors are counted as ONE seed-group's worth (the
  largest), not the union — matching engine per-owner filter semantics.
