# Qwen 9B failure triage — easy/medium wins, hard losses

Source: `data/runs/family1-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl`
(159 lines / ~150 of 174 cells complete at audit time; remaining cells are
hard seeds still trickling in).
Model: `Qwen/Qwen3.5-9B` (one repeat per cell, vision modality).

## Scope

Candidates: packs where the model won at least one of easy/medium AND
had at least one hard seed in {loss, draw, error}. Of 29 packs in the
journal, **12 packs (41%)** matched the "ladder breaks at hard" pattern.
The remaining 17 packs either won hard cleanly (e.g.
`combat-harass-aggro-commit` 4/4) or lost at every tier (uniformly hard
packs that are outside this triage's scope).

## Bucket histogram

| Bucket            | Count | Packs                                                                                                                                                       |
|-------------------|------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OUR-CONFIG (pure) |     2 | `action-multiunit-coordination`, `combat-vehicle-vs-infantry-counter`                                                                                       |
| MIXED CONFIG+MODEL|     3 | `combat-formation-tank-wedge`, `combat-tank-vs-tank-engagement`, `harass-response-preserve`                                                                 |
| MODEL-STRATEGY    |     7 | `action-sequenced-execution`, `combat-attack-from-behind-fog`, `combat-divide-and-conquer`, `combat-prevent-retreat`, `combat-protect-vip-escort`, `combat-skirmish-then-disengage`, `combat-target-priority-highvalue` |
| INCONCLUSIVE      |     0 |                                                                                                                                                             |

Net: **~2 packs need a config fix outright**, **3 packs would benefit
from a tick-budget bump**, **7 packs are sound** and discriminate real
model weaknesses (passivity, brute-rushing, ignored doctrine).

## Top CONFIG defects (ranked by hard-seed loss count caused)

### 1. `combat-vehicle-vs-infantry-counter` — fail clause punishes the scripted doctrine (B.4, 4 hard losses)

`fail_condition.any_of: [..., {not: {own_units_gte: 1}}]` paired with a
scenario that ships only ONE pre-spawn unit (a scout jeep) and instructs
the model to "scout with the jeep to verify they carry no anti-tank
threats." Every hard seed: the jeep walks toward the centre cluster as
briefed, takes fire, drops to 0 HP, and the game ends in `loss` at
turn ~4-5 with ~$1951 still in the build queue and a full base intact.

- Seed 1: turn 4, tick 363, jeep at HP 0.066 → episode end loss.
- The fact / barracks / war factory / service depot are all standing
  (`has_building:fact` was still SAT in the goal at episode end).

**Fix options:**
- (a) Pre-spawn a single garrison unit (e.g. one `2tnk` or two `e1`)
  alongside the jeep so a scout death doesn't drop `own_units` to 0.
- (b) Replace `not own_units_gte:1` with `not has_building:fact` alone —
  the win clause already gates on `has_building:fact`, and a base-alive
  scenario should not LOSE just because the only scout dies.

### 2. `action-multiunit-coordination` — tick budget not scaled for the perception cost added on hard (B.1, 2 hard losses)

Hard and medium both use `within_ticks: 2800` on the same map and same
unit composition. Hard's only delta is `objective_coords: relative` +
`enemy_building_spotted` interrupts — i.e. the model has to SCOUT for
buildings whose coords are hidden. That scout cost burns 1-2 turns per
landmark but the deadline stays flat.

Decisive trace — **hard seed 3**: at turn 32 / tick 2853, ALL THREE
`units_in_region_gte` clauses are satisfied AND `units_lost_lte:2`
holds (0 lost). The model literally completed the objective, but
`within_ticks:2800` had already burst by 53 ticks (~1 turn). Hard
seed 4 fails the same way.

**Fix:** raise hard `within_ticks` to ~3300 and `max_turns` to 42 (a
~17% headroom that mirrors the fogged-search overhead). Easy/medium
remain unchanged.

### 3. `combat-formation-tank-wedge` (seed 4 only) — close-miss deadline (B.1, 1 hard loss)

Hard seed 4: model reached the (46,20) staging region with 4 units
alive, but burst `within_ticks: 4500` by 33 ticks (cur=4533 at the
end). One additional turn would have closed the kill quota
(units_killed_gte:4 was at 0).

This pack also has a separate MODEL failure (seed 2: 3 units lost to
brute-rush) so the right action is a small CONFIG bump to ~4800 ticks
to absorb seed-to-seed variance, while accepting that the brute-rush
seed is a genuine model failure.

### 4. `combat-tank-vs-tank-engagement` — `max_turns: 15` is the tightest budget in any candidate pack (B.1, contributing to 3 hard losses)

Hard seed 2: killed 2/3 enemies at the max-turns deadline with all 3
own tanks alive — only burst `within_ticks:1200` by 63 ticks. The
model wasted 2 of 16 turns on `Observe`, which under interrupt mode
costs ~160 ticks. With seeds 3 & 4 also losing on attrition, the 15-
turn / 1200-tick budget feels load-bearing in the wrong direction —
seed 1 won so the capability IS reachable, but the margin is below
the model's noise floor.

**Fix:** bump `max_turns: 15 → 20` and `within_ticks: 1200 → 1500`.

### 5. `harass-response-preserve` — `units_lost_lte: 0` is the harshest fail floor in the suite (B.4, 3 hard losses)

The hard tier requires `units_lost_lte: 0` — zero deaths permitted.
Seed 1 trace: model AttackMoved at turn 3, lost 2 infantry, immediate
fail. The briefing says "preserve" so the model IS supposed to avoid
combat, but a knife-edge zero-loss floor means even competent
positioning can lose to RNG.

**Fix:** loosen to `units_lost_lte: 1`. The doctrine is still
"preserve" (one casualty is still a costly mission) and the win-cond
still demands 3 kills + base intact, but the model gets a one-error
buffer.

## Top MODEL-strategy patterns

### 1. Passivity / Observe-stall (A.1) — 2 packs, both with VIP-survival or kill-quota gates

- `combat-protect-vip-escort` seed 3: **9 Observes / 20 turns (45%)**;
  the model freezes from turn 16 onward while the escort harvester
  takes fire and dies at turn 19. This is the textbook "freeze when
  the pressure shows up" failure.
- `combat-target-priority-highvalue` seed 1: **10 Observes / 18 turns
  (56%)**, 1 kill of 15 required. The model never commits to a
  target-priority engagement.

These two packs are SOUND — they are discriminating exactly the
weakness the names advertise.

### 2. Ignored doctrine / brute-rush (A.2) — 6 packs

The most common pattern: the briefing prescribes a flanking, splitting,
kiting, or holding doctrine and the model issues a frontal `AttackMove`
or `MoveUnits` straight through the enemy centre.

- `combat-attack-from-behind-fog` s3: drove all 4 units along y=24 (the
  centre lane the brief explicitly says to AVOID).
- `combat-divide-and-conquer` s1: killed 8/9 but lost 2 of 4 starters
  to frontal pressure when a split would have kept losses to 0-1.
- `combat-skirmish-then-disengage` s1: 8 `AttackMove` commands, lost all
  4 units, 0 kills, never executed the "disengage" half.
- `combat-prevent-retreat` s2: 17 `AttackMove` sweeps for 1 kill — the
  doctrine demands a chokepoint hold, not a sweep.
- `action-sequenced-execution` hard (all 4 seeds draw): the model issued
  153 MoveUnits orders over 70 turns with extensive backtracking (e.g.
  target_x=15,y=15 after a scout) and only completed 1 of the 2 routes
  before the max_turns hit; 37 of 156 actions were warned (24% format
  churn).
- `harass-response-preserve` seeds 1-4: attack-moved into combat
  instead of preserving (this overlaps with the CONFIG concern above).

### 3. Close-miss after correct intent (A.4) — 1 pack

- `combat-tank-vs-tank-engagement` s2: model executed the engagement
  correctly (3 tanks alive, 2 enemies killed) but ran out of turns
  before landing the third kill. Sits on the boundary between MODEL
  pacing and CONFIG headroom — see Top CONFIG #4.

## Summary

- **12 candidate packs** analysed across hard tier (~44 hard episodes).
- **5 packs** need CONFIG attention (2 outright, 3 light tick-budget
  bumps); the other 7 packs correctly discriminate model weakness.
- The biggest single source of "false losses" is
  `combat-vehicle-vs-infantry-counter`'s `not own_units_gte:1` fail
  clause firing on the scripted scout death (4/4 hard seeds lost on
  this same trigger).
- The dominant MODEL pattern is **doctrine-ignored brute-rush** (6 of
  7 MODEL-bucket packs), not raw passivity — Qwen 9B reads the
  briefing well enough at easy but defaults to frontal `AttackMove` /
  bulk `MoveUnits` at hard when the routes are fogged or the
  composition is asymmetric.
- Passivity (`Observe` >40%) is concentrated in the two
  "preservation" packs (`protect-vip-escort`, `target-priority-highvalue`)
  — these packs are SOUND and exposing a real model weakness.

## Methodology notes

- Outcome matrix built from `_journal__Qwen_Qwen3.5-9B.jsonl`.
- For each candidate pack: read `manifest.json` + `turns.jsonl` for the
  first non-winning hard seed (priority loss > draw > error), and
  compared against the winning easy seed trace from the same pack.
- Goal-tree leaves (`current` vs `target` vs `satisfied`) from the
  final turn pinpointed which predicate flipped to fail; this
  distinguished "deadline burst" (within_ticks) from "attrition fail"
  (not own_units_gte:N) from "kill quota not met" (units_killed_gte:N).
- No engine, YAML, or test edits were made — pure trace analysis.
