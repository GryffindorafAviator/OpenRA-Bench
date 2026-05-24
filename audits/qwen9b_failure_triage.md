# Qwen 9B Failure Triage — easy/medium-win-but-hard-loss

## Scope

Family 1, 29 packs × {easy, medium, hard×4 seeds} = 174 target cells
(journal has 177 lines after retries). Source:
`data/runs/family1-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl`.

Filter applied: pack wins easy OR medium AND has at least one hard
seed NOT winning. **12 packs match.**

(The strict filter — easy/medium-WIN AND >=3 of 4 hard seeds NOT win
— matches a tighter 6-pack subset: `action-sequenced-execution`,
`combat-divide-and-conquer`, `combat-skirmish-then-disengage`,
`combat-tank-vs-tank-engagement`, `combat-vehicle-vs-infantry-counter`,
`harass-response-preserve`. The CSV is broadened to the 12-pack set
so partial hard-tier failures are still captured for the next-wave
fix list.)

## Bucket histogram

| Bucket          | Count |
|-----------------|-------|
| OUR-CONFIG      | 2     |
| MIXED           | 3     |
| MODEL-STRATEGY  | 7     |
| Inconclusive    | 0     |

Sub-bucket breakdown (primary subbucket):

| Subbucket                                            | Count |
|------------------------------------------------------|-------|
| A.1 idle/passivity                                   | 2 |
| A.2 wrong tactic (chase, brute, dispersal, no-doctrine) | 6 |
| A.4 ran short on tight clock                         | 1 (with B.1) |
| B.1 tick budget too tight / inert                    | 1 (with A.2 in mixed) |
| B.4 fail-clause mistriggered on briefing-doctrine    | 2 (one mixed) |

## Top CONFIG defects to fix (ranked by hard-seed loss count)

### 1. combat-vehicle-vs-infantry-counter — `not own_units_gte:1` punishes the scout (4 hard losses / 4 seeds)

- Every hard seed LOSES at turn 4-5 (tick 273-363 of 5400 budget).
- The briefing prescribes "scout with the jeep to verify they carry no
  anti-tank threats". Model follows brief. The scout jeep ends turn 4
  at HP 6.7% inside the 12-strong fogged centre cluster, dies turn 5
  → fail clause `not own_units_gte:1` fires the instant the scout dies
  (the predicate counts mobile units only — the agent's 5 base
  buildings + $1951 + in-flight Build orders don't save it).
- Easy survives because there are only 8 enemies and the scout can
  retreat; hard has 12 enemies under fog and the scouted area is a
  death zone.
- **Fix:** replace the `not own_units_gte:1` clause with `not
  has_building:fact` (or drop it — `after_ticks: 5401` and
  `not has_building:fact` already cover stall and base-wipe). Today's
  clause turns the pack's own scripted doctrine into a tripwire.

### 2. action-multiunit-coordination — hard within_ticks too tight after adding fog interrupts (2 hard losses)

- Hard seed 3 satisfied ALL 3 region predicates at turn 32 / tick 2853
  with 0 units lost, but `within_ticks:2800` burst by 53 ticks → near-
  miss LOSS. The pack adds `objective_coords:relative` + fogged
  interrupts on hard vs medium, yet keeps the same 2800-tick budget.
- **Fix:** raise hard `within_ticks` / `max_turns` by ~15-20% (e.g.
  3300 ticks / ~42 turns) to absorb the per-turn scout-cost the
  fogged-interrupt mode adds.

### 3. action-sequenced-execution — interrupt-mode max_turns under-reaches the deadline (4 hard draws)

- All 4 hard seeds DRAW (CLAUDE.md rule #2 violation: non-completion
  must be a real LOSS, never a DRAW).
- Pack uses `interrupts.enemy_building_spotted: true`. Under interrupt
  mode the engine advances 1-`max_ticks` ticks per turn (variable, not
  the standard 30). With `max_turns=70` the agent hits turn 70 at tick
  ~5400-5500, well below the `after_ticks: 6001` fail threshold.
- hard:1 ended at tick 5388 with obj_progress 0.5; hard:2 at tick 5508
  with one of two waypoint sequences satisfied (B sat=True, A
  cur=None). None of the 4 seeds triggered the `after_ticks` fail —
  pure timeout-DRAW degeneracy.
- This was labelled MODEL-STRATEGY in the CSV (model's multi-route
  coordination failure is the immediate cause), but the underlying
  pack defect is real: the within_ticks/after_ticks deadline never
  bites within max_turns, so even a perfect failure mode reports as
  DRAW. **Both fixes apply** — strengthen the model's coordination
  AND tighten the deadline (e.g. lower within_ticks to ~5000 OR raise
  max_turns to ~100).

### Borderline CONFIG (the "MIXED" entries)

- **combat-tank-vs-tank-engagement** (3 hard losses): all losses end
  3-63 ticks past the 1200-tick deadline with 2/3 kills. `max_turns:15`
  is the tightest budget in any candidate pack; consider bumping to
  18-20 turns to absorb pacing variance, though seed 1 WIN proves
  capability is reachable.
- **combat-formation-tank-wedge** seed 4 (1 hard loss): killed 0/4 in
  region with 4 units alive — `within_ticks:4500` burst by 33 ticks.
  Pure close-miss deadline.
- **harass-response-preserve** (4 hard losses): `units_lost_lte:0`
  hard cap is a knife-edge with 3 simultaneous probes vs 5 stance:0
  HoldFire defenders. Loosening to `units_lost_lte:1` preserves the
  bar (all cheat plays still lose) while giving competent models
  recovery room.

## Top MODEL-strategy patterns

### 1. Passivity / freeze under pressure (A.1)
- `combat-protect-vip-escort` hard:3: 9 Observes / 20 turns (45%
  passivity); model observed through the final 4 turns while the
  harv escort died → `unit_type_count_gte:harv,1` fail.
- `combat-target-priority-highvalue` hard:1: 10 Observes / 18 turns
  (56% passivity); 1 kill of 15 required.
- `combat-skirmish-then-disengage` hard:2 (62.5% observe — 5 of 8
  turns).
- `combat-divide-and-conquer` hard:4 (45.8% observe).
- Pattern: when novel/contested, model defaults to observing.

### 2. Brute-rush instead of doctrine (A.2)
- `combat-divide-and-conquer`: brief warns "middle corridor's choke
  gets cooked"; model walks frontally on seeds 1/2/4. Seed 3 (the
  WIN) used 11 AttackUnit instead of the dispersed AttackMove.
- `combat-attack-from-behind-fog` hard:3: model drove all 4 units
  along y=24 (centre lane) — exact opposite of "attack from behind
  via fog flank".
- `combat-skirmish-then-disengage`: brief says "drive east, gun down
  three, then pull back". Model AttackMoves east and never disengages.
- `harass-response-preserve`: brief says "do NOT chase east" — model
  chases anyway on every loss.

### 3. Tight-clock turn-management waste (A.4)
- `combat-tank-vs-tank-engagement` hard:2/3/4: all three losses end
  3-63 ticks past 1200 with 2/3 kills. Model issued 10-11 MoveUnits
  per episode and only 2 AttackUnits — repositioned instead of
  focus-firing. Seed 1 won with 4 AttackUnit, 3 kills at tick 1038.

### 4. Multi-route coordination collapse (A.2)
- `action-sequenced-execution` hard:1/2: 142-153 MoveUnit commands
  spread across two parallel waypoint sequences A and B. Model
  satisfies one but not both.

### 5. Vague AttackMove sweep instead of intercept (A.2)
- `combat-prevent-retreat` hard:2: 17 AttackMove commands, only 1
  kill, 2/4 units lost. Brief says "block the retreat axis" — model
  doesn't hold a chokepoint, AttackMoves into the open.

## Notes

- All candidate seed traces examined used the `vision` perception
  modality (default). No `-clear` (no-fog) cells inspected.
- 17 of 29 family-1 packs do NOT match the candidate filter — most
  win or lose uniformly across tiers.
- `rush-hour` returned only 2 of 4 hard seeds (seeds 1 and 3 missing
  from the journal); both seeds present were wins, so it is not a
  candidate. Worth a re-run for completeness.
