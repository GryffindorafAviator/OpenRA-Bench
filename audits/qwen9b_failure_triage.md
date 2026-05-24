# Qwen 9B Failure Triage — easy/medium-win-but-hard-loss

## Scope

Family 1, 29 packs × {easy, medium, hard×4 seeds} = 174 target cells
(journal has 177 lines after retries). Source:
`data/runs/family1-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl`.

Filter applied: pack wins easy OR medium AND has >=3 of 4 hard seeds
NOT winning (loss or draw). **6 packs match.**

## Bucket histogram

| Bucket          | Count |
|-----------------|-------|
| MODEL strategy  | 4     |
| CONFIG defect   | 2     |
| Inconclusive    | 0     |

Sub-bucket breakdown:

| Subbucket                                         | Count |
|---------------------------------------------------|-------|
| A.1 idle/passivity                                | 1 (with A.2) |
| A.2 wrong tactic (chase, brute, dispersal)        | 4 |
| A.4 ran short on tight clock                      | 1 (with A.2) |
| B.1 tick budget inert under interrupt-mode turns  | 1 |
| B.4 fail-clause mistriggered on briefing-doctrine | 1 |

## Top CONFIG defects to fix (ranked by hard-seed loss count)

### 1. combat-vehicle-vs-infantry-counter — `not own_units_gte:1` punishes the scout

- **Hard-tier non-wins: 4/4 LOSS**, every loss at turn 4-5 (tick 273-363
  of 5400 budget — agent never even reaches turn 6).
- Brief says "scout with the jeep to verify they carry no anti-tank
  threats". Model follows brief. Jeep ends turn 4 at HP 6.7% in the
  12-strong fogged cluster, dies turn 5. The fail clause `not
  own_units_gte:1` fires the instant the scout dies — `own_units_gte`
  counts mobile units only, so the agent's 5 base buildings + $1951 +
  in-flight Build orders don't save it.
- Easy survives (8 enemies → scout can retreat). Hard's 12 enemies + fog
  make scout survival a gamble the brief mandates the model take.
- **Fix:** drop `not own_units_gte:1` from the fail clause (or replace
  with `not has_building:fact`). The pack's other fail clauses
  (`after_ticks: 5401`, `not has_building:fact`) already cover stall and
  base-wipe; the `own_units` clause exists to prevent suicide-rush but
  in practice traps the scenario's own scripted doctrine.

### 2. action-sequenced-execution — interrupt-mode max_turns under-reaches the deadline

- **Hard-tier non-wins: 4/4 DRAW** (CLAUDE.md rule #2 violation:
  non-completion must be a real LOSS, never a DRAW).
- Pack uses `interrupts.enemy_building_spotted: true`. Under interrupt
  mode the engine advances 1-`max_ticks` ticks per turn (variable),
  not the standard 30. With `max_turns=70` the agent hits turn 70 at
  tick ~5400-5500, well below the `after_ticks: 6001` fail threshold.
- hard:1 ended at tick 5388 with obj_progress 0.5 (neither waypoint
  sequence satisfied); hard:2 at tick 5508 with one of two waypoint
  sequences satisfied (B sat=True, A cur=None). None of the 4 seeds
  triggered the `after_ticks` fail.
- **Fix:** raise `max_turns` to ~100 OR drop `within_ticks` to 5000 and
  `after_ticks` to 5001 so the deadline actually fires within the
  available turn budget. Converts today's degenerate DRAW into the real
  LOSS the no-cheat bar demands.

### 3. harass-response-preserve — `units_lost_lte:0` is a knife-edge (CONFIG-adjacent)

- **Hard-tier non-wins: 4/4 LOSS** at turn 4 / tick 273 — instantly,
  with `units_lost=2`.
- Hard tier escalates from cap 2 (easy) -> cap 1 (medium) -> cap 0
  (hard). 3 simultaneous probes vs 5 stance:0 HoldFire defenders with
  no `set_stance` tool exposed leaves no error budget for a single
  micro slip. Model's failure mode (chase east) is documented as
  intended-to-lose; the pack is technically sound, but bumping
  `units_lost_lte` from 0->1 on hard preserves the bar (every cheat
  policy still loses with 1 casualty allowed) while giving competent
  models recovery room.
- Triage classifies the trace evidence as MODEL.A.2 (model is
  chasing east on every seed), but the pack design itself is brittle.
- **Soft recommendation:** consider loosening hard cap to 1 if all
  wrong-doctrine plays still lose there.

## Top MODEL-strategy patterns

### 1. Passivity / freeze under pressure (A.1)
- combat-skirmish-then-disengage hard:1 (28.6% observe) and hard:2
  (62.5% observe — 5 of 8 turns were observe-only).
- combat-divide-and-conquer hard:4 (45.8% observe — 11 of 24 turns).
- Pattern: when the engagement is novel or contested, the model
  defaults to observing.

### 2. Brute-rush instead of doctrine (A.2)
- combat-divide-and-conquer: brief warns "middle corridor's choke
  gets cooked"; model walks frontally on seeds 1/2/4 anyway.
- combat-skirmish-then-disengage: brief says "drive east, gun down
  three, then pull back into the recovery disc". Model AttackMoves
  east and never disengages.
- harass-response-preserve medium/hard: brief says "do NOT chase east"
  in bold pack design; model chases east anyway.

### 3. Tight-clock turn-management waste (A.4)
- combat-tank-vs-tank-engagement hard:2/3/4: all three losses end
  3-63 ticks past the 1200-tick deadline with 2/3 kills. Model issued
  10-11 MoveUnits and only 2 AttackUnits per episode — repositioned
  instead of focus-firing. Seed 1 won with 4 AttackUnit, 3 kills at
  tick 1038.

### 4. Multi-route coordination collapse (A.2)
- action-sequenced-execution hard:1/2: 142-153 MoveUnit commands
  spread across two parallel waypoint sequences A and B. Model often
  satisfies one but not both (hard:2: B satisfied, A not).

### 5. Scout over-commit (A.2)
- combat-vehicle-vs-infantry-counter (also flagged as CONFIG above):
  model drives the scout deep into a 12-strong fogged cluster on
  every hard seed and never retreats. This is partially the brief's
  fault for saying "scout to verify" without a "retreat after first
  contact" qualifier.

## Notes

- All candidate seed traces examined used the `vision` perception
  modality (default). No `-clear` cells inspected.
- 23 of 29 family-1 packs do NOT match the candidate filter — most
  win or lose uniformly across tiers. The 6 packs in scope here are
  exactly the easy->hard cliff packs.
- `rush-hour` returned only 2 of 4 hard seeds (seeds 1 and 3 missing
  from the journal); both seeds present were wins, so it is not a
  candidate. Worth a re-run for completeness but does not affect
  this triage.
