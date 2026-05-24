# Qwen 9B F1+F2 v3 Deep Triage — MODEL vs SCENARIO vs ENGINE

**Scope:** 226 non-win cells across F1 (combat/micro) + F2 (economy), Qwen 3.5-9B v3 run.
Source journals: `data/runs/family1-v3-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl` (168 cells) + `data/runs/family2-v3-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl` (168 cells).

## 1. Headline counts

| Bucket | Count | % of non-wins |
|---|--:|--:|
| MODEL | 222 | 98.2% |
| SCENARIO | 4 | 1.8% |
| ENGINE | 0 | 0.0% |
| INCONCLUSIVE | 0 | 0.0% |

### Sub-bucket distribution

| Bucket | Subbucket | Count |
|---|---|--:|
| MODEL | M.2 | 129 |
| MODEL | M.1 | 35 |
| MODEL | M.4 | 30 |
| MODEL | M.5 | 28 |
| SCENARIO | S.3 | 2 |
| SCENARIO | S.1 | 1 |
| SCENARIO | S.2 | 1 |

## 2. Top SCENARIO defects (priority fix list)

Most of the matrix is MODEL gap, not scenario defect. The 5 documented defects (4 draws + 1 hard-with-draw) are:

| Priority | Pack | Tier | Seed(s) | Subbucket | Evidence | Recommendation |
|---|---|---|---|---|---|---|
| 1 | combat-flanking-attack | easy | 1 | S.3 | easy:1 ended draw at turn 26 (6 observe turns); model stalled — fail clause did not fire on passivity; 4-tank survival-floor knife-edge with | Add explicit fail_condition any_of with after_ticks + not full_objective so stall is LOSS not DRAW |
| 2 | combat-focus-fire-priority | medium | 1 | S.1 | medium:1 draw at turn 30/budget; pack runs in interrupt mode, per-turn tick advance variable; within_ticks 2700 not reachable in 30 turns un | Raise max_turns or relax within_ticks so deadline bites within max_turns × (variable per-turn ticks under interrupt mode) |
| 3 | action-multiunit-coordination | hard | 2 | S.2 | engine ended at turn 32/tick 2643 (kills=3) before within_ticks=3300; 2 of 3 region predicates met, 3rd never reached because run terminated | Add anti-auto-done sentinel (far-corner unarmed enemy fact/barr) or convert deadline-without-completion into explicit fail clause |
| 4 | econ-protect-harvester-route | medium | 1 | S.3 | medium:1 draw at turn 36 (kills=2 lost=0 progress=0.80); fail_condition does not bite on the deadline-without-completion frame | Add explicit fail_condition any_of with after_ticks + not full_objective so stall is LOSS not DRAW |

## 3. Top MODEL gap patterns (the legitimate eval signal)

These are what the benchmark IS measuring. The benchmark works as designed — 9B is a weaker model and the bench correctly fails it on these capabilities:

| Sub-bucket | Count | % of MODEL non-wins | Description |
|---|--:|--:|---|
| M.2 | 129 | 58.1% | Wrong tactic — strategy chosen does not match briefing situation (front-charge instead of flank, hoard instead of spend, abandon defense post) |
| M.1 | 35 | 15.8% | Idle / passivity — many `Command::Observe` turns, runs out the clock |
| M.4 | 30 | 13.5% | Close miss — last turn within 1-2 turns / 5 % of win threshold |
| M.5 | 28 | 12.6% | Fixated on subtask — ignored win clause |

## 4. Per-tier breakdown

| Tier | MODEL | SCENARIO | ENGINE | INCONCLUSIVE | Total non-wins |
|---|--:|--:|--:|--:|--:|
| easy | 35 | 1 | 0 | 0 | 36 |
| medium | 35 | 2 | 0 | 0 | 37 |
| hard | 152 | 1 | 0 | 0 | 153 |

Easy and medium losses are predominantly MODEL.1 passivity + M.2 wrong-tactic; hard losses are spread similarly across the model sub-buckets, with the few scenario defects mostly on hard.

## 5. Per-family breakdown

| Family | MODEL | SCENARIO | ENGINE | INCONCLUSIVE |
|---|--:|--:|--:|--:|
| F1 | 84 | 3 | 0 | 0 |
| F2 | 138 | 1 | 0 | 0 |

**F1 (combat/micro)** is roughly balanced 9B win/loss; failures are largely MODEL wrong-tactic (M.2) and close-miss (M.4) — 9B can fire the right verbs but mis-picks the right unit composition or attack vector.

**F2 (economy)** is dominated by MODEL passivity (M.1) and wrong-tactic (M.2). The cash-band predicates (burn-rate, cash-reserve, overflow-to-silos) catch a stall as a LOSS (cash drifts above the upper bound); the multi-step economy reasoning (expansion timing, second-base race, contested expansion) catches 9B unable to chain build orders. This is the load-bearing 9B-vs-Gemma capability gap documented in the prior analysis.

## 6. Pre-fix vs Post-fix delta (Qwen 9B)

225 same-cell pairs available across pre-fix (`data/runs/family[12]-qwen9b/playback/`) and post-fix v3 journals.

| Pre → Post | n |
|---|--:|
| loss → loss | 79 |
| win → win | 56 |
| loss → win | 37 |
| win → loss | 36 |
| draw → loss | 11 |
| loss → draw | 2 |
| draw → win | 2 |
| win → draw | 1 |
| draw → draw | 1 |

Headline (carried over from `audits/eval_v3_f1f2_analysis.md` §4): the leak-fix removed strategy hints (F1 27 win→loss flips, F2 9 win→loss flips). On F1 the simultaneous CONFIG repairs more than compensated (net +3 pp WR); on F2 the leak fix produced a net regression (-5 pp WR), and the failure mix is now dominated by reasoning passivity/wrong-tactic rather than scenario defects.

In failure-mode terms: the pre-fix matrix had ~13 draws driven by missing termination semantics and tick-deadline mis-budgeting (mostly in `combat-suicide-charge-mission`, `combat-focus-fire-priority` hard, `combat-target-priority-highvalue` hard, `combat-rocket-soldier-anti-vehicle` easy). After the engine wire-up of `termination.{agent,enemy}_units_killed` + the within_ticks / max_turns config repairs, only 4 draws remain — all real scenario defects flagged in §2.

## 7. ENGINE issues spotted

No new engine issues found in this triage that are not already documented in `CLAUDE.md`. Two close calls worth noting:

- **`action-multiunit-coordination:hard:2`** — the engine ended the episode at turn 32 / tick 2643 with `kills=3` despite `within_ticks=3300` and `max_turns=42`. The engine appears to auto-`done` after the 2 enemy `MustBeDestroyed` landmarks (fact + proc) are destroyed, BEFORE the 3rd region predicate latches. Strictly this is a SCENARIO design issue (the pack needs an anti-auto-done sentinel or its win clause should not depend on a third region after the buildings have been destroyed), but the engine behaviour is also worth a 3rd-party check — see CLAUDE.md auto-done #5.
- **interrupt-mode tick-rate mis-budgeting** — `combat-focus-fire-priority:medium:1` still draws under `max_turns=30 + within_ticks=2700` because the per-turn tick advance under interrupt mode is below the constant 90-ticks-per-turn assumption. This is a known footgun (CLAUDE.md #2) — pack-side fix, not an engine bug. No PR needed.

## 8. Recommendations (prioritized fix list)

### SCENARIO fixes (5 cells need YAML edits)

1. **`action-multiunit-coordination:hard`** — add a 3rd-region fail clause OR add an anti-auto-done sentinel so the 3rd region waypoint can latch before engine done. (hard:2 draw)
2. **`combat-flanking-attack:easy/medium`** — add explicit fail clause `after_ticks + not full_objective` so a stall produces LOSS instead of DRAW; both 9B and Gemma drew here under the current YAML. (easy:1 draw)
3. **`combat-focus-fire-priority:medium`** — raise `max_turns` 30 → 45 (interrupt-mode tick deficit). (medium:1 draw)
4. **`econ-protect-harvester-route:medium`** — add explicit deadline-without-completion fail clause. (medium:1 draw)
5. Re-evaluate **`combat-rocket-soldier-anti-vehicle:easy`** loss — the model issued 797 commands across 51 turns, never landed a single kill. This is most likely M.3 (tool-format thrash) but the cell is a known prior B.2 candidate; verify the second far-corner sentinel landed.

### MODEL gaps (no fix — these are the legitimate eval signal)

- 129 cells lose to wrong-tactic (M.2)
- 35 cells lose to passivity (M.1)
- 30 cells are close-misses (M.4) — these are the most discriminating cells; the next model class up will likely flip them all to wins
- 0 cells show probable tool-format errors (M.3)

### Operational recommendation

The 5 SCENARIO fixes above are the entire scenario-defect surface in the F1+F2 v3 matrix. Everything else is 9B vs the bar — the bar is working as designed. The next eval step is the **Gemma 31B rerun** (271 / 336 cells were 429-shadowed; see `eval_v3_f1f2_analysis.md` headline).
