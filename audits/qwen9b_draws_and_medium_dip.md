# Qwen 9B Deep-Dive — 9 Unaccounted Draws + Medium-vs-Hard Inversion

## Scope

Two analyses, additive to `audits/qwen9b_failure_triage.csv`:

1. The **9 DRAW cells** not classified by the prior triage (per CLAUDE.md
   no-cheat bar rule #2, every DRAW is a scenario defect).
2. The **medium-vs-hard win-rate inversion** (medium 37.9% < hard 45.7%).

Prior triage covered: `action-sequenced-execution` hard×4 (B.1),
`action-multiunit-coordination` hard×2 (B.1), `combat-target-priority-highvalue`
hard:3 (re-classified below).

Sources:
- Journal `data/runs/family1-qwen9b/playback/_journal__Qwen_Qwen3.5-9B.jsonl`
- Per-cell traces under `data/runs/family1-qwen9b/playback/2026.../`
- Pack YAMLs under `openra_bench/scenarios/packs/`

---

## 1. The 9 draws — per-pack classification

### combat-suicide-charge-mission (5 draws — STRUCTURAL DEFECT)

The 5 suicide-charge draws share one root cause. The pack documents
this defence:

> `termination.agent_units_killed: false` — without this the engine
> auto-`done`s the run the moment the last attacker dies, even if the
> objective fact has already been razed and within_ticks is still
> satisfied — collapsing the win into a draw.

**The flag is INERT.** A grep across the bench (`openra_bench/`) AND the
engine (`OpenRA-Rust/`) finds no consumer of `agent_units_killed` /
`enemy_units_killed`. The flag exists only in 3 YAMLs
(`combat-suicide-charge-mission`, `strategy-dilemma`, `strategy-twobody`,
`rush-hour`); the engine ignores it and `done=True` from the env breaks
the loop at `openra_bench/eval_core.py:453` regardless. Whichever path
the agent takes that wipes the strike package collapses to DRAW.

Per-cell evidence:

| pack:level:seed | turns / max | final tick / budget | killed | lost | last_unit_count |
|---|---|---|---|---|---|
| suicide-charge:easy:1 | 14 / 62 | 1263 / 5400 within_ticks | 0 | 10 | 0 |
| suicide-charge:medium:1 | 9 / 73 | 813 / 6400 within_ticks | 1 | 10 | 0 |
| suicide-charge:hard:2 | 11 / 82 | 993 / 7200 within_ticks | 1 | 10 | 0 |
| suicide-charge:hard:3 | 12 / 82 | 1083 / 7200 within_ticks | 2 | 10 | 0 |
| suicide-charge:hard:4 | 12 / 82 | 1083 / 7200 within_ticks | 0 | 10 | 0 |

(Suicide-charge:hard:1 WON at turn 12 — agent kept 4 units alive long
enough to raze the fact, then auto-done on enemy-elimination fired the
win.)

**Classification:** B.3 — fail_condition does not bite on agent-force
wipe; only fires on `after_ticks: N+1`, which the auto-done race always
beats.

**Fix:** see "Structural defect" section below.

### combat-rocket-soldier-anti-vehicle:easy:1 — sentinel doubles as target (1 draw)

The only enemy `fact` is at (60,20), 20 cells past the engagement column
at x=40. The pack uses it as both the auto-done sentinel AND the
win-anchor reference; there is no separate sentinel. Trace
(`turns.jsonl` turns 27-31) shows the model issued
`AttackUnit { target_id: "1008" }` (the fact) and the agent's starter
units shot it down at turn 31 (tick 2793 of 4500 within_ticks budget).
Engine auto-done'd on enemy-elimination; the
`unit_type_count_gte:{e3, n:6}` clause was never satisfied (model built
0 e3 — it actually queued 2 e3 on turn 27 but they never finished
because the run ended). 4 of 5 win clauses were green; only the e3-build
clause failed. Outcome: DRAW (no fail clause fired before done).

**Classification:** B.2 — engine auto-`done` race because the only
sentinel was destroyed.

**Fix:** add a second far-corner enemy MustBeDestroyed building (e.g.
`barr` or `powr` at (2,2)) so a chance kill of the visible fact doesn't
auto-done; OR move the sentinel `fact` far off the agent's likely
movement axis (e.g. (62,2) instead of (60,20)).

### combat-focus-fire-priority hard:3 + hard:4 (2 draws)

Pack runs in interrupt mode (`interrupts.enemy_unit_spotted: true` +
`own_unit_destroyed: true`). Under interrupt mode the engine advances
1–`max_ticks` ticks per turn (variable; default 5) — see CLAUDE.md
"interrupt mode tick advance". With `max_turns: 30` the final tick lands
at 1473–2358 depending on interrupt frequency; `within_ticks: 2700` /
`after_ticks: 2701` are above the reachable tick.

| seed | final tick | within_ticks budget | killed/14 | lost | shape |
|---|---|---|---|---|---|
| 3 | 2358 | 2700 | 1 | 0 | far from clearing |
| 4 | 2073 | 2700 | 13 | 0 | **one kill short of WIN** |

Seed 4 in particular is a textbook close-miss: the model was executing
the intended capability cleanly (13 of 14 killed, 0 lost — would have
won at 14/15 of those kills) and the deadline simply never bit because
the per-turn tick advance under interrupt mode is too small for the
declared budget.

**Classification:** B.1 — deadline unreachable in interrupt mode at
declared `max_turns`.

**Fix:** raise `max_turns` from 30 → ~45 so the 2700-tick window is
inside the per-turn variable advance.

### combat-target-priority-highvalue hard:3 (1 draw)

Same interrupt-mode pattern as focus-fire hard. `max_turns: 30` hit at
tick 1473; `within_ticks: 2700` / `after_ticks: 2701` unreachable.
Killed 7 of 15 with 1 unit lost (still above the `own_units_gte:3`
floor), so no fail clause bit. The other 3 hard seeds resolve cleanly
(seed 1 LOSS, seed 2 WIN, seed 4 WIN); only seed 3 hits the deadline
mismatch.

This draw was tagged in the prior CSV as "1 hard draw" without a
sub-bucket; the diagnosis here is B.1 (same root cause as the focus-fire
hard draws — they share the `interrupts:` block + `max_turns: 30` +
`within_ticks: 2700` configuration triad).

**Classification:** B.1 — deadline unreachable under interrupt mode.

**Fix:** same as focus-fire — raise `max_turns` ≥ 45.

---

## 2. `combat-suicide-charge-mission` structural defect — root cause + fix

### Root cause

The pack's anti-DRAW defence is a YAML flag that has no consumer:

```yaml
base:
  termination:
    agent_units_killed: false
    enemy_units_killed: false
```

Search results:
- `grep -rn "agent_units_killed" openra_bench/` → 4 hits, all in YAMLs.
- `grep -rn "agent_units_killed" OpenRA-Rust/` → 0 hits.
- `openra_bench/eval_core.py:453` — `if outcome != "draw" or done: break`
  — exits the run loop on `done=True` regardless of any termination flag.

So when the agent's last unit dies the engine returns `done=True`, the
loop breaks, and the only fail clause (`after_ticks: 5401`) hasn't fired
yet because `tick < 5401` (the wipe happens at tick 800-1300). Result:
`outcome = "draw"`.

The pack's premise — "spend the entire force to destroy a high-value
objective" — is a literal-total-sacrifice scenario that REQUIRES the
engine to keep evaluating after the agent has no units. The auto-done
on agent elimination kills the win condition before the strike package
can plausibly raze the fact even on a perfect commit.

Hard:1 happened to WIN because the agent kept 4 units alive past the
fact's destruction by sheer luck of geometry; that's the only seed
where the engine's auto-done fired on enemy-elimination AFTER the win
condition was satisfied.

### Fix options (recommend the engine wire-up since CLAUDE.md says "fix the engine, do not compromise the pack")

1. **Engine wire-up (recommended)**: plumb
   `termination.agent_units_killed: false` /
   `termination.enemy_units_killed: false` through
   `OpenRA-Rust/openra-data/src/oramap.rs::RawScenarioActor` →
   the scenario's termination block → `env.rs` done-eval. Add a Rust
   test pinning that a wiped agent force does NOT set `done=True` when
   the flag is false. Then add a Python pin in
   `tests/test_combat_suicide_charge_mission.py` that a force-wipe
   under the flag preserves the run to within_ticks.
2. **Pack-level workaround (no engine change)**: replace the inert
   flag with an explicit LOSS clause:
   ```yaml
   fail_condition:
     any_of:
       - {after_ticks: 5401}
       - {all_of: [{after_ticks: 600}, {not: {own_units_gte: 1}}]}
   ```
   `after_ticks: 600` gates the wipe-fail clause so it doesn't fire on
   turn 1 (when units may briefly be in-flight); past tick 600 a wiped
   force is a real LOSS, not a DRAW. This converts every draw cell to
   an explicit LOSS — preserving the no-cheat bar at scoring level
   even though the pack's "literal sacrifice" anchor is no longer
   achievable as a WIN.
3. **Pack-level alt (preserve sacrifice anchor)**: keep the wipe LOSS
   clause AND grant the agent a 1-cell sentinel unit (e.g. a single
   `e1` rifleman at (2,2)) that stays out of the engagement. The
   sentinel keeps `own_units_gte:1` and the engine's auto-done off the
   agent side; the strike package can still be sacrificed and the
   evaluator sees the within_ticks frame after the fact falls.

The engine wire-up is the right answer (CLAUDE.md: "fix the engine, do
not compromise the pack"). Failing that, option 3 preserves the
mission's literal-sacrifice anchor without requiring an engine change.

---

## 3. Medium-vs-hard inversion — table + dominant cause

Family-1 packs where 9B medium win-rate is BELOW hard win-rate
(medium has 1 seed; hard has 4):

| pack | easy | medium | hard | classification | evidence |
|---|---|---|---|---|---|
| combat-attack-from-behind-fog | 1/1 win | 0/1 (loss) | 3/4 win | MODEL+SAMPLE | medium:1 loss is identical model failure mode (front-charge instead of fog flank) to hard:3 loss; the OTHER 3 hard seeds happened to win because the model's vague AttackMove still cleared the easier per-seed cluster sizes; medium uses same max_turns=60 / within_ticks=3500 as hard ⇒ no CONFIG asymmetry. |
| combat-flanking-attack | 0/1 loss | 0/1 loss | 3/4 win | MODEL (hard has 8 tanks vs medium's 4) | medium = 4 tanks, hard = 8 tanks (split N/S). `own_units_gte:3` floor is a knife-edge for 4-tank medium (losing 2 fails), comfortable for 8-tank hard (can lose 5 and still win). Hard's headcount escalation gave the survival floor more slack — escalation is non-monotonic on this pack. |
| combat-hold-chokepoint | 0/1 loss | 0/1 loss | 3/4 win | MODEL+SAMPLE | medium:1 ended at turn 65 (vs max 75) — model failed to hold the chokepoint anchor; hard's per-seed enemy compositions happened to give the model 3 wins in <14 turns each. No CONFIG difference (medium/hard share max_turns=75 / within_ticks=5400). |
| combat-protect-vip-escort | 1/1 win | 0/1 loss | 3/4 win | SAMPLE NOISE | All 4 tiers share max_turns=61 / within_ticks=5400. Medium:1 lost at turn 35 (model passivity); 3 of 4 hard seeds won at turns 31-44. Identical pack; single medium seed simply hit the model's freeze pattern. |
| combat-tanya-vs-rush | 0/1 loss | 0/1 loss | 2/4 win | MODEL (model-format failure) | Easy/medium losses both at turn 2 — model failed to issue any productive command (it likely emitted a malformed call). Hard tier has DIFFERENT spawn topology (2-spawn group); hard:1/2 were 1-2 turn wins (1 kill = full clear in hard's lighter rush). The pack structurally lets a 1-2 turn win happen on hard. |
| combat-target-priority-highvalue | 1/1 win | 0/1 loss | 2W/1L/1D in 4 | MODEL+SAMPLE | medium:1 lost at turn 15 (target prioritization failure); hard seeds had different geometry (NORTH/SOUTH spawn split) — same kill bar, different staging. Hard:3 draw is a B.1 deadline defect (see §1). |

### Dominant cause of the medium-vs-hard inversion

**Mostly SAMPLE NOISE + one CONFIG asymmetry** (flanking-attack's
4-vs-8-tank survival-floor relaxation on hard). With medium at 1 seed
per pack and hard at 4 seeds per pack, a single unlucky medium failure
masks behind an averaged hard win-rate; on 5 of the 6 inverted packs
the model failure mode is the SAME as on hard, the medium just hit it
on the one observed seed.

No pack has medium "harder than hard" by configuration in the strict
sense (same `max_turns`, same `within_ticks`, same `after_ticks` across
tiers). `combat-flanking-attack` is the one outlier: hard's
8-tank-with-N/S-split spawn structurally gives the model more headroom
on the `own_units_gte:3` survival floor than medium's 4-tank-single-
spawn does. That's a real but intentional escalation-mismatch (each
tier was designed to add a discrimination axis but they don't quite
keep difficulty monotonic for a passable model).

---

## 4. Add-to-fix-queue list

### Packs needing CONFIG fixes (4 packs, 9 draw cells)

| pack | cells | fix | bucket |
|---|---|---|---|
| combat-suicide-charge-mission | 5 (1 easy + 1 medium + 3 hard) | wire `agent_units_killed: false` engine-side (or pack-level wipe-LOSS clause + 1-cell sentinel) | B.3 — STRUCTURAL |
| combat-rocket-soldier-anti-vehicle | 1 (easy:1) | add a second far-corner enemy MustBeDestroyed building so a chance kill of the visible fact doesn't auto-done | B.2 |
| combat-focus-fire-priority | 2 (hard:3, hard:4) | raise `max_turns` 30 → 45 to fit `within_ticks: 2700` under interrupt mode | B.1 |
| combat-target-priority-highvalue | 1 (hard:3) | raise `max_turns` 30 → 45 (same root cause as focus-fire hard — they share the `interrupts:` + `max_turns: 30` + `within_ticks: 2700` triad) | B.1 |

### Model-only (no pack fix; medium-vs-hard inversion not a CONFIG bug)

- combat-attack-from-behind-fog medium:1 — model front-charges instead
  of fog-flanking (same pattern as hard:3 loss; prior triage A.2).
- combat-flanking-attack medium:1 — medium's 4-tank/single-spawn
  survival-floor is structurally tighter than hard's 8-tank/split-
  spawn; not a defect, an escalation-monotonicity quirk. Optional:
  loosen medium `own_units_gte:3` → `own_units_gte:2`.
- combat-hold-chokepoint medium:1 — model passivity (timed out at
  turn 65/75 with weak chokepoint hold).
- combat-protect-vip-escort medium:1 — model passivity / freeze
  (same pattern as hard:3 in the prior triage).
- combat-tanya-vs-rush easy:1 + medium:1 — turn-2 model-format
  failures; not a pack defect.

---

## Summary

- **Total drawn-pack count needing CONFIG fixes: 4 packs / 9 draw
  cells**:
  - combat-suicide-charge-mission (5 cells, B.3 — structural; the
    documented `agent_units_killed: false` flag is inert)
  - combat-rocket-soldier-anti-vehicle (1 cell, B.2 — single-sentinel
    auto-done race)
  - combat-focus-fire-priority (2 cells, B.1 — interrupt-mode tick
    deficit)
  - combat-target-priority-highvalue (1 cell, B.1 — same as focus-fire)
- **Medium-vs-hard top 3 inverted packs**:
  1. combat-flanking-attack (0/1 medium vs 3/4 hard — CONFIG
     asymmetry: hard's 8-tank/N-S-split gives the survival floor
     headroom medium's 4-tank/single-spawn does not)
  2. combat-attack-from-behind-fog (0/1 medium vs 3/4 hard — MODEL
     pattern + sample noise)
  3. combat-hold-chokepoint (0/1 medium vs 3/4 hard — MODEL pattern +
     sample noise)
- **Dominant cause — Task 1 (draws):** missing/inert termination
  semantics (suicide-charge's flag never wired) + interrupt-mode
  deadline mis-budgeting (focus-fire/target-priority hard share the
  same triad).
- **Dominant cause — Task 2 (medium-vs-hard):** SAMPLE NOISE (single
  medium seed catches the model's failure mode that hard's 4 seeds
  partially average out), with one real structural escalation-quirk
  on combat-flanking-attack.
