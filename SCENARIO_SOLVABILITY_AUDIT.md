# Scenario Solvability Audit (Static)

**Date:** 2026-05-23  
**Branch:** pr13-revised-rebased  
**Repo:** `/Users/xiaochu/Projects/OpenRA/OpenRA-Bench`  
**Method:** Pure YAML reading + arithmetic. No engine runs. See "Method" below.

## TL;DR

- **208** active scenario packs, **186** static-clean (89.4%), **22** flagged.
- **0 packs are theoretically unsolvable.** No win predicate exceeds available targets / budgets, no inert deadlines on active packs, and the only `MISSING_FAIL` is on `economy-time-box.yaml` which is `status: quarantine` (excluded from the active set).
- **4 packs (WARNING)** carry an authoring mistake (actor `position` outside `MapSize`) that the engine silently drops; works by accident in 3/4 because the dropped actor is an enemy MBD building still counted by `is_terminal` (canonical anti-DRAW effect, just OOB instead of at `(120,20)`).
- **18 packs (INFO)** trigger a rule that the design context defuses: the canonical "after_ticks-in-win + far enemy `fact` anti-DRAW marker" survive idiom (17 packs), or the no-enemy navigation idiom protected by `enemy_started_present=false` (1 pack).
- This audit does NOT certify the no-cheat bar (lazy/brute/stall LOSE; intended capability WINS); that requires the live scripted-policy sweep.

## Top — Counts

| Bucket | Count | Notes |
|---|---:|---|
| Active scenario packs analysed | **208** | excludes `_archive/`, `TEMPLATE.yaml`, and 1 quarantined pack(s) |
| Static-clean packs (no defect signal) | **186** | 89.4% — see TAIL caveat |
| Packs with at least one defect / note | **22** | 10.6% |
| Packs with multiple defects / notes | **22** | usually all 3 tiers share the same predicate idiom |
| Quarantined (excluded) | 1 | `economy-time-box.yaml` — `status: quarantine` |
| Parse errors | 0 | should be 0 |

### Severity rollup of "with-defect" packs

| Severity | Pack count | Meaning |
|---|---:|---|
| **HIGH** | **0** | Theoretical-solvability defect — predicate cannot be satisfied / actor invisible to predicate / engine panic. Real bug. |
| **WARNING** | **4** | Authoring mistake — actor `position` outside the map's `MapSize`. Engine silently drops the terrain placement. Side effect: the actor still appears in `world.actors` but never on the map. For an enemy MBD building this happens to function as the anti-DRAW marker idiom; for any other actor it's just a wasted entry. Either way, normalise the coordinates so a future "fix" doesn't silently invert the pack. |
| **INFO** | **18** | Rule technically applies but the design context defuses it. Surfaced so reviewers can confirm intent (canonical anti-DRAW marker pattern, no-enemy navigation idiom). |

### Per-class histogram

| Defect class | Total occurrences | Distinct packs | What it means |
|---|---:|---:|---|
| `AFTER_TICKS_IN_WIN` | 59 | 17 | `after_ticks` in a win clause. Per `CLAUDE.md`, structurally incompatible with `ConquestVictoryConditions` *unless* the only enemy MBD building is a far-corner (e.g. `(120,20)`) anti-DRAW marker that's effectively unkillable in `max_turns`. All 17 packs flagged here use that pattern (no reachable enemy MBD), so the rule's degenerate-DRAW path is closed by design — flagged INFO. |
| `ACTOR_OUT_OF_BOUNDS` | 10 | 4 | Actor `position` is outside the map's `MapSize`. Engine `terrain.set_occupant` silently drops the placement (`if self.contains(x,y)`). Actor stays in `world.actors`; if it's an enemy MBD building, `is_terminal` still counts it ⇒ accidental anti-DRAW behaviour. |
| `REACH_WITHOUT_ENEMY` | 5 | 2 | `reach_region`/`building_in_region` win predicate with no enemy actors in the pack. Per `env.rs::is_terminal`, when `enemy_started_present == false` the engine never auto-terminates on enemy elimination, so the win predicate IS evaluated. Canonical no-enemy navigation idiom — flagged INFO. |

## Capability families with at least one flagged pack

| Capability | Flagged packs | Notes |
|---|---:|---|
| `action` | 3 | Three packs: 1 OOB anti-DRAW marker (`build-rally-point-management`, fact at (140,20) silently drops yet still gates `is_terminal`), 2 INFO survive-idiom packs. |
| `perception` | 6 | 6 packs: 1 with OOB markers (`perception-frontier-reading` — wasted but win still satisfiable via in-bounds markers), 2 no-enemy navigation idioms (`custom-map-no-enemy`, `navigation-confined-hard-only`), 3 survive/discovery idiom packs. |
| `reasoning` | 13 | Concentrated in `tp-*` survive-N, `lh-*` long-horizon, `tempo-*`, `mid-*` — the survive/timing idiom that legitimately uses `after_ticks` in win + a far anti-DRAW marker (canonical pattern, INFO-severity). |

## Method (the static-only no-defect bar this audit checks)

Per `CLAUDE.md` §"The bar (apply to every scenario you touch)" the engine-side defect set is:

1. **Inert deadline** — `within_ticks` / `after_ticks` set above the tick reachable inside `max_turns` (engine ~90 ticks/turn ⇒ `max_tick = 93 + 90·(max_turns − 1)`). Inert-in-win or inert-in-fail collapses an otherwise-LOSS to a DRAW.
2. **Missing or degenerate `fail_condition`** — a stall / preserve / partial outcome silently DRAWS.
3. **`after_ticks` in a WIN clause** — structurally incompatible with `ConquestVictoryConditions`: the engine auto-`done`s the second the last enemy `MustBeDestroyed` building falls, before the `after_ticks` window opens. Mitigation: a far-corner enemy `fact` marker that's unkillable inside `max_turns`. We downgrade severity to `info` when the pack uses that mitigation (no reachable enemy MBD; the only MBD is a >60-cell-away or x≥110 marker).
4. **Actor positions outside playable bounds** — engine `terrain.set_occupant` silently drops the placement (`if self.contains(x,y)`); actor stays in `world.actors` so an OOB enemy MBD building keeps gating `is_terminal`. For `rush-hour-arena` MapSize is 128×40 ⇒ valid x ∈ [0,127], y ∈ [0,39]. Tailored maps declare their own dims.
5. **Hard-tier seed-driven spawn check** — listed-as-`UPGRADED` in `tests/test_hard_tier.py` but no ≥2 distinct seed-driven spawns. **NOT CHECKED** here; that's already enforced by `tests/test_hard_tier.py`.
6. **Engine auto-terminate on enemy-elimination before win/fail evaluates** — for `reach_region`/`building_in_region` predicates without a persistent enemy. Detected; downgraded to `info` for the canonical no-enemy idiom (`env.rs::is_terminal` short-circuits when `enemy_started_present=false`).
7. **Logical defect** — win predicate unsatisfiable: `units_killed_gte: N` vs available enemy targets, `building_count_gte` vs starting cash + harvest income, etc.

Method limitations:
- Static-only. We don't simulate the engine. Heuristics are deliberate (anti-DRAW marker = enemy MBD building >60-cell Manhattan from any agent actor OR x≥110 — the rush-hour idiom).
- Harvest income estimate is crude (≤6000 if `ore_patches:` or `mine` actors AND a proc/seedable proc exists; `mine`/`gmine` actors auto-seed an ore patch per engine `world.rs`).
- We do NOT validate the no-cheat bar (lazy/brute/stall LOSE; intended capability WINS). That requires the scripted-policy live-engine sweep — see `CLAUDE.md` §"How to validate".

## Body — Per-pack Defect Listing

### HIGH severity (theoretical-solvability defects)

_(none)_

### WARNING severity (authoring mistake; works by accident or wastes the actor)

#### `build-rally-point-management.yaml`

- **[warning] ACTOR_OUT_OF_BOUNDS** — level `easy`, line `184`
  - actor[4] type=fact owner=enemy at (140,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `medium`, line `184`
  - actor[4] type=fact owner=enemy at (140,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `hard`, line `184`
  - actor[9] type=fact owner=enemy at (140,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention

#### `econ-contention-with-enemy.yaml`

- **[warning] ACTOR_OUT_OF_BOUNDS** — level `easy`, line `177`
  - actor[15] type=fact owner=enemy at (128,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `medium`, line `177`
  - actor[14] type=fact owner=enemy at (128,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `hard`, line `177`
  - actor[26] type=fact owner=enemy at (128,20) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention

#### `maint-repair-priority-order.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `235`
  - win.win.all_of[3].after_ticks contains after_ticks=900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `294`
  - win.win.all_of[3].after_ticks contains after_ticks=900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `395`
  - win.win.all_of[3].after_ticks contains after_ticks=900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `hard`, line `366`
  - actor[13] type=fact owner=enemy at (120,44) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention

#### `perception-frontier-reading.yaml`

- **[warning] ACTOR_OUT_OF_BOUNDS** — level `medium`, line `140`
  - actor[2] type=e1 owner=enemy at (30,41) is outside map 128x40 (valid x: 0..127, y: 0..39). non-MBD actor placed out-of-bounds is silently dropped from terrain — never discoverable / shootable / pathable. If the win predicate depends on this actor specifically, the level is unsatisfiable; if other in-bounds actors of the same kind exist, this is a wasted authoring mistake but the win can still be met.
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `medium`, line `143`
  - actor[3] type=proc owner=enemy at (100,41) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention
- **[warning] ACTOR_OUT_OF_BOUNDS** — level `hard`, line `197`
  - actor[7] type=proc owner=enemy at (118,40) is outside map 128x40 (valid x: 0..127, y: 0..39). enemy MBD building authored out-of-bounds — engine silently drops the terrain placement but the actor still gates `is_terminal` ⇒ effectively an anti-DRAW marker. Works by accident; flag so the coordinate doesn't get 'fixed' to inside-bounds (which would let the agent kill it and re-enable auto-done).
  - **Fix:** move actor inside map bounds (e.g. fact at x=124 instead of x=128/140); for anti-DRAW markers, (120,20) is the canonical convention

### INFO (rule technically applies but design-correct)

#### `artofwar-sequenced-citadel.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `69`
  - win.win.all_of[1].after_ticks contains after_ticks=1100; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `94`
  - win.win.all_of[1].after_ticks contains after_ticks=1300; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `129`
  - win.win.all_of[1].after_ticks contains after_ticks=1400; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `build-repair-priority-under-fire.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `231`
  - win.win.all_of[2].after_ticks contains after_ticks=1200; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `288`
  - win.win.all_of[2].after_ticks contains after_ticks=1700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `370`
  - win.win.all_of[2].after_ticks contains after_ticks=1700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `custom-map-no-enemy.yaml`

- **[info] REACH_WITHOUT_ENEMY** — level `easy`, line `?`
  - win uses reach_region/building_in_region and pack has NO enemy actors (no scheduled_events spawn). Per env.rs `enemy_started_present=false` ⇒ engine does NOT auto-terminate on enemy elimination, so this is the canonical no-enemy navigation idiom
  - **Fix:** (no-op; verify intent — pack is a navigation/no-enemy idiom)
- **[info] REACH_WITHOUT_ENEMY** — level `medium`, line `?`
  - win uses reach_region/building_in_region and pack has NO enemy actors (no scheduled_events spawn). Per env.rs `enemy_started_present=false` ⇒ engine does NOT auto-terminate on enemy elimination, so this is the canonical no-enemy navigation idiom
  - **Fix:** (no-op; verify intent — pack is a navigation/no-enemy idiom)
- **[info] REACH_WITHOUT_ENEMY** — level `hard`, line `?`
  - win uses reach_region/building_in_region and pack has NO enemy actors (no scheduled_events spawn). Per env.rs `enemy_started_present=false` ⇒ engine does NOT auto-terminate on enemy elimination, so this is the canonical no-enemy navigation idiom
  - **Fix:** (no-op; verify intent — pack is a navigation/no-enemy idiom)

#### `def-reinforce-the-breach.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `211`
  - win.win.all_of[0].after_ticks contains after_ticks=720; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `274`
  - win.win.all_of[0].after_ticks contains after_ticks=720; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `346`
  - win.win.all_of[0].after_ticks contains after_ticks=720; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `econ-harvester-defense-raid.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `298`
  - win.win.all_of[0].then.clauses[0].after_ticks contains after_ticks=1700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `385`
  - win.win.all_of[0].then.clauses[0].after_ticks contains after_ticks=2700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `521`
  - win.win.all_of[0].then.clauses[0].after_ticks contains after_ticks=3500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `lh-100-turn-marathon-survival.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `198`
  - win.win.all_of[3].after_ticks contains after_ticks=7200; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `272`
  - win.win.all_of[3].after_ticks contains after_ticks=8100; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `372`
  - win.win.all_of[3].after_ticks contains after_ticks=9900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `lh-opening-to-defense-to-counter.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `248`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1100; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `308`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1300; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `411`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `lh-recovery-after-mid-game-loss.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `258`
  - win.win.all_of[2].after_ticks contains after_ticks=1600; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `347`
  - win.win.all_of[2].after_ticks contains after_ticks=1600; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `456`
  - win.win.all_of[2].after_ticks contains after_ticks=1600; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `mfb-rotating-production-pressure.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `299`
  - win.win.all_of[3].after_ticks contains after_ticks=4500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `382`
  - win.win.all_of[3].after_ticks contains after_ticks=4500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `539`
  - win.win.all_of[3].after_ticks contains after_ticks=4500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `mid-concede-vs-hold.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `224`
  - win.win.any_of[0].all_of[2].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `224`
  - win.win.any_of[1].all_of[2].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `314`
  - win.win.any_of[0].all_of[2].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `314`
  - win.win.any_of[1].all_of[2].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `487`
  - win.win.any_of[0].all_of[2].after_ticks contains after_ticks=2400; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `487`
  - win.win.any_of[1].all_of[2].after_ticks contains after_ticks=2400; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `navigation-confined-hard-only.yaml`

- **[info] REACH_WITHOUT_ENEMY** — level `easy`, line `?`
  - win uses reach_region/building_in_region and pack has NO enemy actors (no scheduled_events spawn). Per env.rs `enemy_started_present=false` ⇒ engine does NOT auto-terminate on enemy elimination, so this is the canonical no-enemy navigation idiom
  - **Fix:** (no-op; verify intent — pack is a navigation/no-enemy idiom)
- **[info] REACH_WITHOUT_ENEMY** — level `medium`, line `?`
  - win uses reach_region/building_in_region and pack has NO enemy actors (no scheduled_events spawn). Per env.rs `enemy_started_present=false` ⇒ engine does NOT auto-terminate on enemy elimination, so this is the canonical no-enemy navigation idiom
  - **Fix:** (no-op; verify intent — pack is a navigation/no-enemy idiom)

#### `perception-count-the-threat.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `190`
  - win.win.all_of[2].after_ticks contains after_ticks=800; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `245`
  - win.win.all_of[2].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `301`
  - win.win.all_of[2].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=2) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `scout-cycle-keep-info-fresh.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `274`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `350`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `431`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `scout-track-enemy-movement.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `326`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1400; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `328`
  - win.win.all_of[0].then.clauses[3].after_ticks contains after_ticks=2700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `438`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1100; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `440`
  - win.win.all_of[0].then.clauses[3].after_ticks contains after_ticks=2000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `442`
  - win.win.all_of[0].then.clauses[5].after_ticks contains after_ticks=2900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `555`
  - win.win.all_of[0].then.clauses[1].after_ticks contains after_ticks=1100; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `557`
  - win.win.all_of[0].then.clauses[3].after_ticks contains after_ticks=2000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `559`
  - win.win.all_of[0].then.clauses[5].after_ticks contains after_ticks=2900; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `tempo-double-window.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `118`
  - win.win.all_of[1].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `175`
  - win.win.all_of[1].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `236`
  - win.win.all_of[1].after_ticks contains after_ticks=3000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `tempo-strike-window.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `85`
  - win.win.all_of[0].after_ticks contains after_ticks=2000; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. no enemy MBD buildings at all (bot_type=''); rule technically applies but degenerate-DRAW path unreachable — pattern-typical for survive-N packs
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `109`
  - win.win.all_of[0].after_ticks contains after_ticks=1800; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. no enemy MBD buildings at all (bot_type=''); rule technically applies but degenerate-DRAW path unreachable — pattern-typical for survive-N packs
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `141`
  - win.win.all_of[0].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. no enemy MBD buildings at all (bot_type=''); rule technically applies but degenerate-DRAW path unreachable — pattern-typical for survive-N packs
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `tp-survive-and-strike-at-window.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `167`
  - win.win.then.clauses[0].all_of[0].after_ticks contains after_ticks=1200; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `223`
  - win.win.then.clauses[0].all_of[0].after_ticks contains after_ticks=1500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `308`
  - win.win.then.clauses[0].all_of[0].after_ticks contains after_ticks=1800; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

#### `tp-survive-n-turns.yaml`

- **[info] AFTER_TICKS_IN_WIN** — level `easy`, line `246`
  - win.win.all_of[2].after_ticks contains after_ticks=2700; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `medium`, line `350`
  - win.win.all_of[2].after_ticks contains after_ticks=3600; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead
- **[info] AFTER_TICKS_IN_WIN** — level `hard`, line `470`
  - win.win.all_of[2].after_ticks contains after_ticks=4500; per CLAUDE.md structurally incompatible with ConquestVictoryConditions. only far-corner MBD marker(s) (count=1) and no reachable MBD; canonical anti-DRAW marker pattern — after_ticks DOES bite as designed
  - **Fix:** move after_ticks into fail_condition; encode timing via positioning/landmarks instead

## Tail — Static-Clean Pack List

**Caveat:** This list reports packs with no static-defect signal under the rules in §Method. It does NOT certify the no-cheat bar — that requires the live scripted-policy sweep (`stall`, `brute`, `greedy`, intended-capability) against the engine, per `CLAUDE.md`. Packs in this list still need the runtime sweep to prove every level actually rejects the lazy plays as a real timeout LOSS rather than degenerating to DRAW.

### Static-clean packs (186 of 208 active)

#### action (57)

- `action-multiunit-coordination.yaml`
- `action-sequenced-execution.yaml`
- `combat-flanking-attack.yaml`
- `combat-focus-fire-priority.yaml`
- `combat-formation-tank-wedge.yaml`
- `combat-harass-aggro-commit.yaml`
- `combat-heli-flank.yaml`
- `combat-hold-chokepoint.yaml`
- `combat-kite-and-pull.yaml`
- `combat-kite-jeep-vs-tank.yaml`
- `combat-naval-shore-strike.yaml`
- `combat-pincer-coordination.yaml`
- `combat-prevent-retreat.yaml`
- `combat-protect-vip-escort.yaml`
- `combat-skirmish-then-disengage.yaml`
- `combat-stance-mgmt-attack.yaml`
- `combat-tank-vs-tank-engagement.yaml`
- `combat-tanya-vs-rush.yaml`
- `combat-target-priority-highvalue.yaml`
- `coord-converge-on-target.yaml`
- `coord-cover-and-move.yaml`
- `coord-mutual-support.yaml`
- `coord-relay-attack.yaml`
- `coord-relay-vision-chain.yaml`
- `coord-squad-handoff.yaml`
- `coordination-ordered-rendezvous.yaml`
- `coordination-staggered-window.yaml`
- `def-bridge-chokepoint.yaml`
- `def-engineer-repair-under-fire.yaml`
- `def-stance-mgmt-hold-then-attack.yaml`
- `econ-protect-harvester-route.yaml`
- `harass-response-preserve.yaml`
- `lh-multi-checkpoint-5-plus.yaml`
- `mfb-supply-line-link-between-bases.yaml`
- `mid-economy-under-fire.yaml`
- `proc-checklist-no-deviation.yaml`
- `proc-instruction-following-edge-case.yaml`
- `proc-no-attack-passive-only.yaml`
- `proc-only-build-no-combat.yaml`
- `proc-only-defend-no-attack.yaml`
- `proc-ordered-action-strict.yaml`
- `proc-strict-toolban-fidelity.yaml`
- `proc-tool-use-multi-distractor.yaml`
- `proc-tool-use-with-distractor.yaml`
- `rush-hour.yaml`
- `spec-engineer-capture.yaml`
- `spec-nuke-strike.yaml`
- `spec-spy-infiltrate.yaml`
- `spec-tanya-c4-strike.yaml`
- `spec-thief-steal-cash.yaml`
- `strategy-twobody.yaml`
- `strict-production-bom.yaml`
- `strict-sequence.yaml`
- `strict-toolban-fidelity-under-pressure.yaml`
- `tp-pressure-procedural.yaml`
- `tp-rush-multi-objective.yaml`
- `tp-rush-objective-very-fast.yaml`

#### adversarial (1)

- `adversarial-duel.yaml`

#### perception (12)

- `perception-count-the-threat-small-k.yaml`
- `perception-target-vs-fog.yaml`
- `scout-and-report.yaml`
- `scout-and-survive.yaml`
- `scout-count-defenders.yaml`
- `scout-detect-base-direction.yaml`
- `scout-detect-enemy-tech.yaml`
- `scout-detect-incoming-army.yaml`
- `scout-discover-hidden-base.yaml`
- `scout-far-frontier.yaml`
- `scout-map-reveal-percent-target.yaml`
- `scout-multiple-fog-areas.yaml`

#### reasoning (116)

- `adv-asymmetric-weaker-must-win.yaml`
- `adv-rps-counter-pick.yaml`
- `artofwar-indirect-approach.yaml`
- `artofwar-lure-the-tiger.yaml`
- `build-defensive-skirt-corners.yaml`
- `build-defensive-tower-cluster.yaml`
- `build-defensive-tower-line.yaml`
- `build-engineer-rebuild-after-loss.yaml`
- `build-power-down-defensive.yaml`
- `build-power-online-first.yaml`
- `build-production-throughput-multibuilding.yaml`
- `build-sell-and-rebuild-elsewhere.yaml`
- `build-sequence-tech-cheapest.yaml`
- `build-sequence-tech-fastest.yaml`
- `build-sequence-tech-most-resilient.yaml`
- `build-tech-skip-decision.yaml`
- `building-and-planning.yaml`
- `combat-attack-from-behind-fog.yaml`
- `combat-bait-counter-attack.yaml`
- `combat-divide-and-conquer.yaml`
- `combat-harass-balanced-hit-and-run.yaml`
- `combat-retreat-after-engagement.yaml`
- `combat-rocket-soldier-anti-vehicle.yaml`
- `combat-suicide-charge-mission.yaml`
- `combat-vehicle-vs-infantry-counter.yaml`
- `coord-diversionary-attack.yaml`
- `def-counter-battery.yaml`
- `def-evacuation.yaml`
- `def-in-depth-vs-single.yaml`
- `def-in-depth.yaml`
- `def-multi-direction.yaml`
- `def-position-expected-direction.yaml`
- `def-position-revealed-direction.yaml`
- `def-pre-position-mobile-reserve.yaml`
- `def-retreat-and-rebuild.yaml`
- `def-surprise-flank-react.yaml`
- `def-tower-line-vs-cluster.yaml`
- `def-walls-vs-towers.yaml`
- `def-while-building.yaml`
- `def-with-ambush.yaml`
- `defense-rush-survive.yaml`
- `econ-burn-rate-management.yaml`
- `econ-buy-vs-build-decision.yaml`
- `econ-cash-reserve-management.yaml`
- `econ-contested-expansion.yaml`
- `econ-deny-enemy-expansion.yaml`
- `econ-expansion-timing.yaml`
- `econ-far-patch-vs-near-patch.yaml`
- `econ-harvester-pathing-optimization.yaml`
- `econ-mine-and-grow.yaml`
- `econ-multi-patch-allocation.yaml`
- `econ-overflow-to-silos.yaml`
- `econ-quantitative-vs-qualitative-spend.yaml`
- `econ-recover-from-zero-cash.yaml`
- `econ-replace-dead-harvester.yaml`
- `econ-resource-trade-with-self.yaml`
- `econ-second-base-race.yaml`
- `econ-silo-vs-spend.yaml`
- `econ-startup-from-scratch.yaml`
- `econ-target-cash-amount-by-deadline.yaml`
- `econ-tech-vs-expand-decision.yaml`
- `economy-force-buildup.yaml`
- `economy-harvest-investment.yaml`
- `economy-harvest-timebox.yaml`
- `economy-investment.yaml`
- `expansion-aggro-3-base-greedy.yaml`
- `expansion-balanced-2-base-defended.yaml`
- `expansion-turtle-1-base-fortified.yaml`
- `lh-build-army-coordinate-multifront-attack.yaml`
- `lh-credit-only-final-phase.yaml`
- `lh-defense-tech-second-base.yaml`
- `lh-econ-army-victory.yaml`
- `lh-opening-to-tech-to-army.yaml`
- `lh-progression-stage-locked.yaml`
- `lh-scout-react-counter.yaml`
- `lh-tech-pivot-attack.yaml`
- `lh-tech-rush-vs-army-rush.yaml`
- `longhorizon-opening-to-assault.yaml`
- `maint-sell-and-recoup-cash.yaml`
- `mcv-deploy-and-build.yaml`
- `mcv-deploy-defensible-site.yaml`
- `mcv-deploy-near-resource.yaml`
- `mcv-deploy-relocate-under-pressure.yaml`
- `mcv-deploy-second-base.yaml`
- `mcv-deploy-third-base.yaml`
- `mfb-base-1-defend-base-2-build.yaml`
- `mfb-mirror-base-east-west.yaml`
- `mfb-redundant-tech-buildings.yaml`
- `mfb-tech-base-vs-economy-base.yaml`
- `mfb-third-base-against-clock.yaml`
- `mfb-two-base-simultaneous.yaml`
- `mid-tech-switch-on-scout.yaml`
- `power-budget-online.yaml`
- `proc-conditional-branch-action.yaml`
- `reasoning-frontier-commit.yaml`
- `reasoning-risk-route.yaml`
- `risk-blockade-bypass.yaml`
- `rob-cash-depletion-recovery.yaml`
- `rob-deadline-shortened-midway.yaml`
- `rob-multiple-simultaneous-pressures.yaml`
- `rob-objective-change-midway.yaml`
- `rob-objective-shift-with-or-clause.yaml`
- `rob-partial-base-loss-continue.yaml`
- `rob-unexpected-enemy-spawn.yaml`
- `rob-unit-loss-recovery.yaml`
- `scout-deny-enemy-vision.yaml`
- `scout-jeep-vs-infantry-cost-effective.yaml`
- `strategy-dilemma.yaml`
- `strategy-gauntlet.yaml`
- `strategy-trilemma.yaml`
- `tech-aggro-all-in.yaml`
- `tech-balanced-econ-then-tech.yaml`
- `tech-production-planning.yaml`
- `tech-turtle-defensive-tech.yaml`
- `tp-decision-under-clock.yaml`
- `tp-survive-and-grow.yaml`

