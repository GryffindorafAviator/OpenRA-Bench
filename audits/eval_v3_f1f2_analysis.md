# F1+F2 v3 eval analysis: Qwen 3.5-9B vs Gemma 4 31B (post-fix)

Data: post-fix v3 journals for Family 1 (Combat / Micro) and Family 2 (Economy), 168 cells per file (28 packs x 6 cells = easy + medium + 4 hard seeds).

**Critical caveat:** The Gemma 31B v3 runs are heavily incomplete due to Together.ai 429 rate-limit errors during the eval. F1 Gemma: 127 / 168 cells are `error` (125 x 429, 2 x 503); F2 Gemma: 144 / 168 cells are `error` (all 429). Qwen 9B has zero infrastructure errors. Every Gemma stat below is reported BOTH raw (errors counted as non-wins) AND non-error (excluding error cells), and the non-error WR is the only honest comparison metric — but only ~24 % of the Gemma matrix actually ran. **The bench needs a Gemma rerun before declarative model-vs-model claims are safe.**

---

## Section 1 — Headline numbers

### Per-model per-family per-tier win-rate

| Model | Family | Tier | n | Wins | Loss | Draw | Error | WR (raw) | WR (ex_err) |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| Qwen 9B  | F1 | easy   | 28  | 14 | 13 | 1 | 0  | 0.500 | 0.500 |
| Qwen 9B  | F1 | medium | 28  | 12 | 15 | 1 | 0  | 0.429 | 0.429 |
| Qwen 9B  | F1 | hard   | 112 | 55 | 56 | 1 | 0  | 0.491 | 0.491 |
| Gemma 31B| F1 | easy   | 28  | 2  | 0  | 0 | 26 | 0.071 | **1.000** (2/2) |
| Gemma 31B| F1 | medium | 28  | 3  | 1  | 1 | 23 | 0.107 | 0.600 (3/5) |
| Gemma 31B| F1 | hard   | 112 | 21 | 13 | 0 | 78 | 0.188 | 0.618 (21/34) |
| Qwen 9B  | F2 | easy   | 28  | 6  | 22 | 0 | 0  | 0.214 | 0.214 |
| Qwen 9B  | F2 | medium | 28  | 7  | 20 | 1 | 0  | 0.250 | 0.250 |
| Qwen 9B  | F2 | hard   | 112 | 16 | 96 | 0 | 0  | 0.143 | 0.143 |
| Gemma 31B| F2 | easy   | 28  | 1  | 0  | 0 | 27 | 0.036 | **1.000** (1/1) |
| Gemma 31B| F2 | medium | 28  | 0  | 0  | 0 | 28 | 0.000 | n/a (0/0) |
| Gemma 31B| F2 | hard   | 112 | 21 | 2  | 0 | 89 | 0.188 | **0.913** (21/23) |

### Summary stat sheet

| Stat | Qwen 9B | Gemma 31B |
|---|--:|--:|
| F1 overall WR (raw) | 0.482 | 0.155 |
| F1 overall WR (ex_err, 9B=168, Gemma=41) | 0.482 | 0.634 |
| F2 overall WR (raw) | 0.173 | 0.131 |
| F2 overall WR (ex_err, 9B=168, Gemma=24) | 0.173 | 0.917 |
| Mean composite (raw) | F1 0.551 / F2 0.363 | F1 0.164 / F2 0.108 |
| Mean composite (ex_err) | F1 0.551 / F2 0.363 | F1 0.673 / F2 0.755 |
| Total draws | 4 | 1 |
| Total errors | 0 | 271 (266 x 429, 3 x 503) |
| Total cells evaluated | 336 | 336 (only 65 actually ran) |

**Headline (with the ex-err caveat):** when both models actually got a request through, Gemma 31B clearly out-performs Qwen 9B on Family 1 (0.634 vs 0.482) and dominates Family 2 (0.917 vs 0.173 on the 23 hard seeds Gemma completed). The model gap is real and consistent with parameter scale. **But Gemma's apparent 0.91 hard-WR on F2 is over ~23 cells, not 112** — provisional only.

---

## Section 2 — Pack-level comparison

### 2.1 Inverted-difficulty: 9B WINS / Gemma LOSES on the same (pack, tier, seed)

Only 3 cells in the entire matrix. Suggests Qwen is rarely strictly better than Gemma on a head-to-head cell.

| pack | tier | seed |
|---|---|---|
| combat-kite-jeep-vs-tank | hard | 3 |
| combat-skirmish-then-disengage | hard | 1 |
| combat-target-priority-highvalue | hard | 2 |

### 2.2 Expected gap: Gemma WINS / 9B LOSES on the same cell

| pack | n cells |
|---|--:|
| combat-suicide-charge-mission | 4 (all hard seeds Gemma completed) |
| economy-time-box | 4 |
| economy-harvest-investment | 4 |
| economy-harvest-timebox | 2 |
| economy-investment | 2 (other 2 hard seeds Gemma also won; Qwen split 2W/2L) |
| econ-tech-vs-expand-decision | 2 |
| combat-tanya-vs-rush | 2 |
| combat-skirmish-then-disengage | 1 |
| combat-tank-vs-tank-engagement | 1 |
| combat-target-priority-highvalue | 1 |
| combat-kite-jeep-vs-tank | 1 (other seed inverted) |
| economy-force-buildup | 1 |

### 2.3 Both-lose every hard seed (suspected scenario defect OR genuinely hard)

A pack listed here means Qwen loses 4/4 hard seeds. The Gemma column shows what Gemma did when it ran (n / wins / losses); cells marked `(err)` are 429s.

| pack | cap | Qwen hard | Gemma hard | failure_triage bucket | Notes |
|---|---|--:|---|---|---|
| action-sequenced-execution | action | 0W/4L | 0W/0L/4err | MODEL-STRATEGY (A.2) | both fail — waypoint discipline |
| combat-bait-counter-attack | reasoning | 0W/4L | 0W/0L/4err | n/a | Gemma never ran the hard seeds |
| combat-formation-tank-wedge | action | 0W/4L | 0W/0L/4err | MIXED (A.2+B.1) | |
| combat-harass-balanced-hit-and-run | reasoning | 0W/4L | 0W/2L/2err | n/a | Gemma also losing here |
| harass-response-preserve | action | 0W/4L | 1W/3L/0err | MIXED (B.4+A.2) | **Gemma also lossy → genuine difficulty** |
| econ-burn-rate-management | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-cash-reserve-management | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-contested-expansion | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-expansion-timing | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-far-patch-vs-near-patch | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-harvester-pathing-optimization | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-mine-and-grow | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-multi-patch-allocation | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-overflow-to-silos | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-protect-harvester-route | action | 0W/4L | 0W/0L/4err | n/a | |
| econ-quantitative-vs-qualitative-spend | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-recover-from-zero-cash | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-startup-from-scratch | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| econ-target-cash-amount-by-deadline | reasoning | 0W/4L | 0W/0L/4err | n/a | |
| economy-force-buildup | reasoning | 0W/4L | 1W/0L/3err | n/a | **Gemma can solve it** → 9B-only difficulty, not defect |
| economy-harvest-investment | reasoning | 0W/4L | 4W/0L/0err | n/a | **Gemma 4/4** → not a defect, real model gap |
| economy-harvest-timebox | reasoning | 0W/4L | 2W/0L/2err | n/a | Gemma solving it |
| economy-time-box | reasoning | 0W/4L | 4W/0L/0err | n/a | **Gemma 4/4** |

The F2 econ swathe is largely 429-shadowed; we **cannot conclude defect** for most F2 hards. economy-force-buildup, economy-harvest-investment, economy-harvest-timebox, economy-time-box are confirmed solvable (Gemma wins them) → 9B-specific incapacity on F2 econ reasoning, not pack defects.

### 2.4 Both-win every cell (suspected too-easy)

Qwen-9B 4-of-4 hard plus easy+medium wins:

| pack | cap |
|---|---|
| combat-heli-flank | action |
| combat-naval-shore-strike | action |
| combat-pincer-coordination | action |
| combat-stance-mgmt-attack | action |

(`combat-kite-and-pull` and `combat-prevent-retreat` are 4/4 hard but lose easy/medium for 9B, so they are NOT in the "all-win" list — they discriminate at the easy tier.)

Gemma has no pack at 6/6 — but its run is sparse. Of packs where Gemma ran easy + at least one hard and won everything it ran: combat-suicide-charge-mission (4 hard wins, 2 errors), combat-tank-vs-tank-engagement (3 hard wins, 3 errors), combat-tanya-vs-rush (3 hard wins, 3 errors), economy-investment (4 hard wins, 2 errors), economy-time-box (4 hard wins, 2 errors), economy-harvest-investment (4 hard wins, 2 errors).

---

## Section 3 — Outcome distribution per model

| Model | Family | win | loss | draw | error |
|---|---|--:|--:|--:|--:|
| Qwen 9B | F1 | 81 | 84 | 3 | 0 |
| Qwen 9B | F2 | 29 | 138 | 1 | 0 |
| Gemma 31B | F1 | 26 | 14 | 1 | 127 |
| Gemma 31B | F2 | 22 | 2 | 0 | 144 |

### Draws (scenario defects per CLAUDE.md rule #2)

| Model | Cell | seed | turns | notes |
|---|---|--:|--:|---|
| Qwen 9B | combat-flanking-attack:easy | 1 | 26 | weakest link: perception |
| Qwen 9B | combat-focus-fire-priority:medium | 1 | 30 | weakest link: perception |
| Qwen 9B | action-multiunit-coordination:hard | 2 | 31 | 2/3 region predicates satisfied at turn 32; `within_ticks` 3300 reached but third region (south-west powr) not reached. Looks like a soft draw at deadline expiry — predicate fires `draw` rather than `loss` because partial progress + units_lost_lte still holds. |
| Qwen 9B | econ-protect-harvester-route:medium | 1 | 36 | weakest link: perception |
| Gemma 31B | combat-flanking-attack:medium | 1 | 14 | weakest link: perception |

Five draws total across 672 cells (5 / 672 = 0.7 %). All five are on perception-weakest-link cells. Triage:

- **combat-flanking-attack:easy s1 + medium s1** — same pack draws for both models. Strong signal this pack has a fail_condition that doesn't bite on a stall (the `qwen9b_draws_triage.csv` rule B.2 family). Worth a closer look.
- **action-multiunit-coordination:hard s2** — 2 of 3 region waypoints reached at deadline, deadline expires, agent retains units → ends draw. Per CLAUDE.md rule #2, the fail_condition should fire LOSS on missed deadline regardless of units retained. This is a defect.
- **combat-focus-fire-priority:medium s1** and **econ-protect-harvester-route:medium s1** — investigate fail_condition coverage.

---

## Section 4 — Pre-fix vs Post-fix (Qwen 9B only)

168 same-cell pairs available for F1, 57 for F2.

### F1 (168 common cells)

| Pre → Post | n |
|---|--:|
| win → win | 48 |
| loss → loss | 46 |
| loss → win | 31 |
| win → loss | 27 |
| draw → loss | 11 |
| draw → win | 2 |
| loss → draw | 2 |
| win → draw | 1 |

- **Overall WR**: pre = 0.452 → post = 0.482 (delta +0.030; essentially flat).
- **27 cells flipped win → loss** after the leak fix: the leak removal cost 9B those wins.
- **31 cells flipped loss → win**: the F1 CONFIG fixes (e.g. vehicle-vs-infantry-counter, action-multiunit-coordination) more than compensated.
- **13 of the 14 previous draws became decisive** (mostly losses) — the engine deadline / fail_condition rework eliminated the draw degeneracy. Net effect: matrix moved from draw-shadowed to decisive outcomes, with a small WR lift.

### F1 packs where leak removal HURT 9B most (win → loss flips)

| pack | win→loss | loss→win |
|---|--:|--:|
| combat-harass-aggro-commit | 3 | 0 |
| combat-formation-tank-wedge | 3 | 0 |
| combat-flanking-attack | 3 | 1 |
| combat-protect-vip-escort | 3 | 0 |
| combat-target-priority-highvalue | 2 | 0 |
| combat-tanya-vs-rush | 2 | 3 |
| combat-divide-and-conquer | 2 | 1 |
| combat-attack-from-behind-fog | 2 | 0 |

These packs had load-bearing strategy hints in their pre-fix briefings (per `briefing_leak_audit.csv` they were rated HEAVY) — the model was riding the hint, not solving the problem. Now it doesn't.

### F1 packs where CONFIG fixes HELPED 9B most (loss → win)

| pack | loss→win | win→loss |
|---|--:|--:|
| combat-kite-and-pull | 5 | 0 |
| combat-vehicle-vs-infantry-counter | 3 | 0 |
| combat-tank-vs-tank-engagement | 3 | 0 |
| combat-tanya-vs-rush | 3 | 2 |
| combat-skirmish-then-disengage | 2 | 0 |
| combat-retreat-after-engagement | 2 | 1 |
| action-multiunit-coordination | 2 | 0 |
| combat-focus-fire-priority | 2 | 0 |

These match the `qwen9b_failure_triage.csv` "OUR-CONFIG" bucket — scenarios where the scenario YAML was broken (within_ticks too tight, sentinel anti-auto-done, etc.), so a fix unlocked a previously losable pack.

### F2 (57 common cells)

| Pre → Post | n |
|---|--:|
| loss → loss | 33 |
| win → win | 8 |
| win → loss | 9 |
| loss → win | 6 |
| draw → draw | 1 |

- Overall WR: pre = 0.298 → post = 0.246 (delta −0.052; modest regression).
- F2 saw NET regression for 9B after the leak fix (-5 pp). Consistent with F2 leaks being more strategic ("Build a refinery first then…") and 9B being unable to derive the strategy unaided.

#### F2 leak-fix-hurt packs (win→loss):

| pack | win→loss |
|---|--:|
| econ-buy-vs-build-decision | 3 |
| econ-cash-reserve-management | 2 |
| econ-deny-enemy-expansion | 1 |
| econ-quantitative-vs-qualitative-spend | 1 |
| econ-contention-with-enemy | 1 |
| econ-mine-and-grow | 1 |

**Net pre→post 9B delta: F1 +3.0 pp; F2 -5.2 pp.** The leak fix worked exactly as designed on F2 (made the test harder, 9B WR dropped) — and on F1 the simultaneous CONFIG repairs added back more than the leak removal cost.

---

## Section 5 — Per-capability win-rate

Capability tags reported in the journals only span `action` and `reasoning` for F1 and F2 (perception / longhorizon / etc. live in F3-F10).

| Model | Family | Capability | n | non-err | wins | WR (ex_err) |
|---|---|---|--:|--:|--:|--:|
| Qwen 9B | F1 | action | 120 | 120 | 70 | **0.583** |
| Qwen 9B | F1 | reasoning | 48 | 48 | 11 | **0.229** |
| Qwen 9B | F2 | action | 12 | 12 | 2 | 0.167 |
| Qwen 9B | F2 | reasoning | 156 | 156 | 27 | 0.173 |
| Gemma 31B | F1 | action | 120 | 29 | 18 | 0.621 |
| Gemma 31B | F1 | reasoning | 48 | 12 | 8 | **0.667** |
| Gemma 31B | F2 | action | 12 | 2 | 1 | 0.500 |
| Gemma 31B | F2 | reasoning | 156 | 22 | 21 | **0.955** |

The capability split is the biggest model gap:

- **Qwen 9B is sharply weaker at `reasoning` than `action`** (F1: 0.583 action vs 0.229 reasoning). 9B can execute mechanical micro that requires an obvious move, but multi-step "decide whether to expand / save / spend" reasoning chains collapse.
- **Gemma 31B does not show this gap**: its reasoning WR (ex_err) matches or exceeds its action WR. The model-size gap manifests primarily as reasoning competence.

---

## Section 6 — Outlier traces

### Outlier 1 — combat-suicide-charge-mission:hard seed 1
**Qwen 9B loss, Gemma 31B win** (4 of 4 hard seeds Gemma completed went the same way).

Pack expects the agent to push 10 vulnerable units at a single `fact` landmark (50,20) within 7200 ticks while not being wiped en route. The intended verb is concentrated `attack_unit` once the fact is in line-of-sight.

**Qwen trace** (`turns.jsonl`, 81 turns, ends `loss`):
```
t1: AttackMove([1001..1006], 25, 20), AttackMove([1007..1010], 25, 20)
...
t77: AttackMove([1005..1010], 50, 20)
t78: AttackMove([1005..1010], 50, 20)
t79: AttackMove([1005..1010], 50, 20)
t80: Observe
end: units_killed=2 units_lost=10 (all units dead, deadline expired, won=False)
```
Qwen issued AttackMove 43 times across 81 turns — the verb's en-route auto-fire wastes shots on intermediate enemies, the squad gets ground down. Never resolves to `AttackUnit` on the `fact`.

**Gemma trace** (18 turns, ends `win`):
```
t1: MoveUnits([1001..1010], 25, 20)
t2: MoveUnits([1001..1010], 45, 20)
t3: MoveUnits([1001..1010], 47, 20)
...
t16: AttackUnit([1004..1010], target_id=1014)   # the fact
t17: AttackUnit([1004..1010], target_id=1014)
end: units_killed=4 units_lost=3 won=True (1014 destroyed at t17, tick 1533)
```
Gemma uses concentrated `MoveUnits` to close (move-auto-fires opportunistically without abandoning the path — engine fix from CLAUDE.md), then snaps to `AttackUnit(target_id=fact)`. 3 losses, 1 fact down, well inside 7200.

**Failure mode:** verb selection. `AttackMove` is the wrong primitive for a known-target charge; `MoveUnits` + `AttackUnit` is the correct two-phase. Qwen never makes the phase transition.

### Outlier 2 — economy-investment:hard seed 1
**Qwen 9B loss, Gemma 31B win.**

Pack: build 22 units inside 2613 ticks, with starting cash and a working proc.

**Qwen** (30 turns, loss): issues `Build("e1")` once per turn for 28 turns. Final state `own_units_gte = 14 / 22` (ratio 0.636). The fail is not a strategy problem — it's a *throughput* problem: the model only queues one Build per decision frame.

**Gemma** (16 turns, win): mixes Build + MoveUnits in parallel each turn:
```
t1: Build("e1"), Build("e1")
t2: MoveUnits([1006], 18, 25), Build("e1"), Build("e1")
t3: MoveUnits([1007, 1008], 18, 20), Build("e1"), Build("e1")
...
```
Reaches 22 units at tick 1353, well under the 2613 deadline.

**Failure mode:** parallel command issuance. Qwen issues sequential single-verb plans; Gemma stacks multiple Build orders + a positioning order in the same turn. Throughput on F2 packs depends entirely on parallel Build issuance per turn.

### Outlier 3 — combat-target-priority-highvalue:hard seed 2
**Qwen 9B win, Gemma 31B loss** (rare inversion).

Pack: 4 units, 1 high-value target plus distractors, prioritize the high-value.

**Qwen** (26 turns, win): AttackMove to engage perimeter (2 calls), then sustained `AttackUnit(target_id=...)` on enumerated enemy ids for 22 turns. Kills 15 enemies. units_lost=1.

**Gemma** (20 turns, loss): MoveUnits 3 times to close distance, then 16 turns of `AttackUnit` — same verb, similar trace shape — but kills only 9 enemies and loses 2 of 4 units before deadline. Looking at the unit_ids passed, Gemma sometimes drops a unit from the AttackUnit call (e.g. last turn `[1002, 1003, 1004]` — unit 1001 missing). Suggests Gemma micro-managed group composition mid-fight and lost an unattended unit. Qwen kept all four units focus-firing.

**Failure mode:** focus-fire discipline. The cell where Qwen wins is one where less re-planning is better. Sample size 3 (the inverted set is only 3 cells), so this is an isolated quirk, not a systematic Qwen edge.

### Outlier 4 — action-multiunit-coordination:hard seed 2 (DRAW for Qwen, scenario defect candidate)
Qwen's only F1 hard draw. 32 turns.

Goal leaves at end:
```
units_in_region_gte (44,4,r6,n=2)  -> satisfied
units_in_region_gte (12,36,r6,n=2) -> NOT satisfied
units_in_region_gte (44,34,r6,n=2) -> satisfied
within_ticks 3300 (current 2643)   -> satisfied (deadline reached)
units_lost_lte 2 (current 0)       -> satisfied
```

The deadline expired with 2 of 3 region-occupancy predicates met and the units_lost_lte side-condition satisfied. The composite returns `objective_progress=0.8` and `won=False` → DRAW. Per CLAUDE.md rule #2 this is a defect: a missed-deadline run that retains units should LOSS, not draw. The `fail_condition` for this pack does not have an `after_ticks not all_regions_met` clause (or the equivalent). Recommend: add an explicit fail_condition `after_ticks: 3300, not all_three_regions_met` so the deadline bites as a loss.

### Outlier 5 — action-sequenced-execution:hard seed 1 (both-lose pack)
Qwen 63 turns of MoveUnits (96 calls), ends loss with `waypoint_sequence A` at 0/4 satisfied, `waypoint_sequence B` at 0/4. Explored 78.8% of map but never landed any unit inside the radius-6 disk of any sequence point. Pre-existing `qwen9b_failure_triage.csv` bucket = MODEL-STRATEGY (A.2): the model picks routes that don't pass through the actual waypoints. Gemma 0/6 ran (all errors). No new info — this is the previously-documented sequenced-execution failure mode that survived the leak fix.

---

## Section 7 — Recommendations

### Packs that need CONFIG / scenario rework

| Priority | Pack | Issue |
|---|---|---|
| HIGH | `action-multiunit-coordination:hard` | Draw at deadline (rule #2 defect) — missing `after_ticks` fail_condition; explicit triage rec exists |
| HIGH | `combat-flanking-attack:easy/medium` | Both models drew here; perception-weakest-link draws indicate fail_condition does not bite a stall |
| MEDIUM | `combat-focus-fire-priority:medium s1` (Qwen draw) | Investigate fail_condition |
| MEDIUM | `econ-protect-harvester-route:medium s1` (Qwen draw) | Investigate fail_condition |
| LOW | `combat-suicide-charge-mission:hard` | Not a defect — solvable by Gemma — but the Qwen failure mode is "uses AttackMove instead of AttackUnit"; consider whether the briefing should still survive a more direct verb-choice test |

### Packs to consider too-easy (Qwen 6/6 wins, very low signal)

- `combat-heli-flank`, `combat-naval-shore-strike`, `combat-pincer-coordination`, `combat-stance-mgmt-attack`

All four are pre-fix CONFIG-fixed packs; if a 9B-class model wins every cell the discriminator value is gone. Tighten difficulty (smaller agent force / larger enemy / shorter window) on hard tier.

### Packs with strong discrimination signal (use these for benchmarking)

By same-cell disagreement count (excluding error cells):

| Pack | q_W_g_L | q_L_g_W | both_W | both_L | Discrimination quality |
|---|--:|--:|--:|--:|---|
| combat-suicide-charge-mission | 0 | 4 | 0 | 0 | clean Gemma-wins separator |
| economy-time-box | 0 | 4 | 0 | 0 | clean |
| economy-harvest-investment | 0 | 4 | 0 | 0 | clean |
| economy-investment | 0 | 2 | 2 | 0 | clean (parallel build throughput probe) |
| econ-tech-vs-expand-decision | 0 | 2 | 0 | 0 | (low n) clean |
| economy-harvest-timebox | 0 | 2 | 0 | 0 | clean |
| combat-tanya-vs-rush | 0 | 2 | 1 | 0 | mixed but Gemma-leaning |

These are the most informative packs — they reliably separate Gemma from 9B by ≥2 hard seeds. The four F2 economy throughput packs (`economy-time-box`, `economy-harvest-investment`, `economy-investment`, `economy-harvest-timebox`) are an especially clean cluster — they probe the parallel-build-orders-per-turn capability that 9B lacks and Gemma has.

### Packs to retire or rework due to no signal

`combat-suicide-charge-mission`, `combat-pincer-coordination`, `combat-naval-shore-strike`, `combat-heli-flank`, `combat-stance-mgmt-attack`: Qwen sweeps 6/6 or Gemma sweeps every cell it ran. These do not discriminate at the 9B↔31B level.

### Operational recommendation

**The Gemma rerun is the single most important next step.** With 271 / 336 cells lost to provider 429s the model-vs-model comparison is on 19 % of the matrix. Add per-call retry/backoff to the Together.ai client OR cut concurrency on Gemma's eval, then rerun F1+F2. Without this, all "Gemma is better at reasoning" claims are provisional.
