# OpenRA-Bench — Scenario Design-Quality Audit & Scoring

Analysis only. No scenario files changed. Prepared 2026-05-17.
Scope: the 48 `status: active` packs (every pack without
`status: quarantine`); TEMPLATE and all `cat-c*-02..05` quarantine
packs excluded. Grounded against `SCENARIO_AUDIT.md`,
`win_conditions.py`, `scoring.py`, `goal_tracker.py`,
`eval_core.run_level`, the schema, and `openra_env.game_data`
(verified unit/building costs, prerequisites, tech tree).

---

## 1. Executive Summary

The capability taxonomy and predicate grammar are sound, and the
hand-authored families (perception-*, reasoning-*, action-*,
artofwar-*, strict-*) are mostly genuine, well-grounded decision
problems. But the active set has four structural quality problems that
materially weaken it as a discriminating benchmark:

1. **Loss is impossible in ~70% of active packs.** Only a
   `fail_condition` can produce `loss`; `run_level` leaves the outcome
   at `draw` when the win predicate is unmet (eval_core.py L174, L243).
   Every perception-*, reasoning-*, action-*, economy-*, custom-map,
   strict-sequence, building-and-planning, and most `cat-*` pack has
   **no real fail_condition** (the `cat-*` ones carry
   `units_lost_lte: -1`, which is *never* satisfiable because
   `units_lost = max(0, …) ≥ 0` — a dead no-op). Consequence: a
   do-nothing agent and a near-miss agent both score `draw` (0.5
   outcome); the binary outcome only separates win vs. not-win.
   Partial credit via `objective_progress` rescues *some*
   discrimination, but the headline outcome axis is degenerate on the
   majority of the suite.

2. **The `cat-c*` `-00`/`-01` kept representatives are near-exact
   duplicates, not distinct decisions.** `diff cat-c1-00 cat-c1-01`
   differs only in id/title and ±1 `explored_pct` / ±200 ticks.
   c5↔c6, c7↔c8, c9↔c11, c4↔c3 are the same decision with relabelled
   meta. The two kept variants per category add **zero** distinct
   capability — keep one, cut the second.

3. **`always-movement-wins` degeneracy in the navigation/perception
   `cat-*` and several hand-authored packs.** Where the only win
   clauses are `explored_pct_gte` + `within_ticks` with no fail and a
   generous clock, the *baseline scripted frontier agent*
   (`scripted_explore_agent`) already wins — these do not separate
   strong from weak models, only "moves" from "doesn't move."

4. **A genuine solvability defect in `strict-production-bom` hard**:
   the budget cannot cover the required tech path (see §2). Several
   "spend/build" packs are solvable but only test "spend the obvious
   thing," not an allocation trade-off, because the engine has no
   harvest income (documented S0/S1) so there is never scarcity
   pressure beyond a single fixed `starting_cash`.

**Bottom line:** ~9 packs should be CUT now (8 redundant `cat-*`
duplicates + 1 unsolvable level needs a budget fix or cut), ~16 need a
FIX (almost all: "add a real fail_condition so loss≠draw"), and the
hand-authored perception/reasoning/artofwar/strict-bom core is KEEP.
The single highest-leverage repo-wide fix is adding a real
`fail_condition` (timeout-loss or attrition-loss) to every active pack
so the outcome axis stops collapsing loss into draw.

---

## 2. Solvability spot-checks (verified against game_data)

Verified building/unit costs & prereqs: `powr` 300 (no prereq, +100
pwr), `tent`/`barr` 500 (prereq `powr`, −20 pwr), `proc` 1400 (prereq
`powr`), `pbox` 600 (prereq `tent`), `gun` 800 (prereq `weap`),
`tsla` 1200 (prereq `weap`, −75 pwr), `weap` War Factory (~2000),
`fact` 2000, e1 100 / e3 300 (prereq `barr|tent`).

- **strict-production-bom HARD — UNSOLVABLE on budget.** Spec:
  `tsla` + 3 e1 + 2 e3, `starting_cash: 4200`, agent faction soviet.
  `tsla` requires `weap` (≈2000) which is **not pre-placed and not in
  the actor list**. Minimum spend: barr 500 + weap 2000 + tsla 1200 +
  3·e1 300 + 2·e3 600 = **4600 > 4200**. Also tsla is −75 power with
  only one `powr` (+100) minus barr/weap drain → likely power-starved.
  → **FIX (raise hard `starting_cash` to ~5200 and pre-place or
  budget `weap`) or CUT the hard level.** Easy/medium are solvable
  (barr 500 + 3 e1; barr + 3 e1 + 2 e3 within 2200/3000) and a good
  BFCL-style fidelity test.
- **building-and-planning** easy (build powr+tent, cash 6000),
  medium (2 pbox in region, tent pre-placed, cash 5000: 2·600+powr),
  hard (MCV→deploy fact, 2 buildings near (60,20), cash 4000):
  all solvable; hard is tight but feasible. KEEP.
- **economy-investment / economy-time-box / economy-force-buildup**:
  all levels solvable on the given `starting_cash` (2nd proc 1400 +
  2nd powr 300 ≤ budgets; e1 at 100 makes `own_units_gte` trivially
  affordable). Solvable but **decision-thin** — no harvest income
  means "wide vs deep" is not a real trade (both fit the budget); the
  test reduces to "issue the build commands." FIX (see §4).
- **adversarial-* / artofwar-* / reasoning-risk-route /
  reasoning-frontier-commit / perception-***: policies exist
  (kite/focus-fire; bait then run; take the safe detour; commit to
  the correct fog region). Solvable; these are the strongest packs.
- **cat-c3/c4** (`building_total_gte` + `power_surplus_gte: 0`):
  solvable (powr is +100, free of prereq; spam powr). But
  `power_surplus_gte: 0` + `building_total_gte` is trivially gamed by
  building only power plants — see Rigor notes.
- **rush-hour / strategy-gauntlet / strategy-twobody / strategy-
  dilemma**: 14 scattered enemy infantry vs ~32 agent combat units,
  `units_killed_gte` 3–9; solvable, but with stance-3 auto-fighting
  units the agent barely has to act — see Discrimination.

---

## 3. Scored Table (1–5; 5 = best)

solv = solvability · disc = discrimination · grnd = grounding ·
rigor = non-gameable + principled scaling · ovr = overall

| pack id | capability | solv | disc | grnd | rigor | ovr | verdict | one-line note |
|---|---|---|---|---|---|---|---|---|
| action-multiunit-coordination | action | 5 | 4 | 5 | 4 | **4** | KEEP | Real parallel-control test; add fail_condition so non-win≠draw. |
| action-sequenced-execution | action | 5 | 4 | 5 | 4 | **4** | KEEP | after_ticks gates encode ordering well; add timeout fail_condition. |
| adversarial-duel | adversarial | 4 | 4 | 5 | 4 | **4** | KEEP | Has real fail (own_units_gte:1); reactive enemy = unique RTS value. |
| adversarial-siege | adversarial | 4 | 4 | 5 | 4 | **4** | KEEP | Same model as duel; entrenched defender variant — distinct enough. |
| adversarial-skirmish | adversarial | 4 | 4 | 5 | 4 | **4** | KEEP | Outnumbered/maneuver variant; genuine asymmetric tactic. |
| artofwar-decoy-sacrifice | reasoning | 4 | 5 | 5 | 5 | **5** | KEEP | Loss cap forces "spend bait, save army" — true credit assignment. |
| artofwar-indirect-approach | reasoning | 4 | 5 | 5 | 5 | **5** | KEEP | units_lost_lte:0 + hazard short path = clean long-horizon test. |
| artofwar-lure-the-tiger | reasoning | 3 | 4 | 5 | 4 | **4** | FIX | Two-phase intent good, but win has no clause proving the lure happened — reaching region by any survivable route also wins; add an ordering/after_ticks gate. |
| artofwar-sequenced-citadel | reasoning | 5 | 4 | 5 | 4 | **4** | KEEP | after_ticks+within_ticks band genuinely grades order; solid. |
| building-and-planning | reasoning | 4 | 4 | 5 | 4 | **4** | KEEP | 3 genuinely different decisions (tech dep / placement / relocate); add fail_condition. |
| custom-map-no-enemy | perception | 5 | 2 | 4 | 2 | **2** | FIX | Pure nav, no enemy, no fail → baseline move-agent wins easy/medium; only hard (all_units_in_region, tight clock) discriminates. Keep hard, cut/merge easy+medium or add fail. |
| economy-force-buildup | reasoning | 5 | 2 | 3 | 2 | **2** | FIX | e1@100 makes own_units_gte trivial; no scarcity (no harvest). Tests "click build," not allocation. Make win require a building mix + fail on idle. |
| economy-investment | reasoning | 5 | 3 | 4 | 3 | **3** | FIX | Best of the economy packs but "wide vs deep" isn't a real trade w/o income; both paths fit budget. Tighten cash so paths are mutually exclusive; add fail. |
| economy-time-box | reasoning | 5 | 2 | 3 | 2 | **2** | FIX | Near-duplicate of force-buildup + a building_total bar; same triviality. Merge into economy-investment or harden. |
| perception-frontier-reading | perception | 5 | 3 | 5 | 3 | **3** | FIX | Strong concept; but easy/medium = explored_pct only, no fail → baseline frontier agent already wins. Hard (decoys + units_lost_lte:1) is the real test. Add fail to easy/medium. |
| perception-target-vs-fog | perception | 5 | 4 | 5 | 4 | **4** | KEEP | buildings_discovered (not coverage) defeats the blind-sweep agent; medium/hard force inference. Add timeout fail. |
| reasoning-frontier-commit | reasoning | 5 | 4 | 5 | 4 | **4** | KEEP | Single scout = un-hedgeable commitment; decoy buildings vs units is a clean discriminator. |
| reasoning-risk-route | reasoning | 5 | 5 | 5 | 5 | **5** | KEEP | units_lost_lte:0 + lethal short path + clock = textbook non-gameable risk/route. Exemplary. |
| rush-hour | action | 5 | 2 | 3 | 2 | **2** | FIX | ~32 stance-3 auto-fighting agent units vs 14 scattered enemy; kill 3/6/9 in 40 turns is near-automatic. Low discrimination; not load-bearing. Cut to a real sweep (fewer units, coverage+kill, fail on timeout). |
| strategy-dilemma | reasoning | 4 | 2 | 3 | 2 | **2** | FIX | Win = buildings_discovered_gte:1 only → any scout that bumps the enemy base wins; "risk dilemma" framing not enforced by the predicate. Replace win with reach_region + units_lost_lte. |
| strategy-gauntlet | reasoning | 4 | 2 | 3 | 2 | **2** | FIX | Same defect: buildings_discovered_gte:1 with a huge friendly force; corridor never has to be solved. Re-spec to reach_region + attrition. |
| strategy-twobody | action | 4 | 2 | 3 | 2 | **2** | FIX | Two pre-split friendly blobs already near the enemy base; discover-1-building wins with no real coordination. Re-spec to all_units_in_region (true rendezvous). |
| strict-production-bom | action | 3 | 5 | 5 | 5 | **4** | FIX | Excellent BFCL-style exact-spec test (unit_type_count_eq + overproduction fail). HARD level unsolvable on 4200 cash (needs weap+tsla≈4600) — raise cash or cut hard. |
| strict-sequence | action | 5 | 4 | 5 | 4 | **4** | KEEP | Tool allowlist + after_ticks band genuinely tests API-fidelity ordering; add timeout fail so a stalled agent loses, not draws. |
| cat-c1-00 (Frontier Scouting) | perception | 5 | 3 | 4 | 3 | **3** | KEEP | Representative of C1; explored_pct+clock, hard adds units_lost_lte:0. OK as the single C1 kept level. |
| cat-c1-01 | perception | 5 | 2 | 4 | 2 | **2** | CUT | ±1 explored_pct / ±200 ticks vs cat-c1-00. Pure duplicate. |
| cat-c2-00 (Threat Enumeration) | perception | 5 | 3 | 4 | 3 | **3** | KEEP | enemies_discovered + units_lost_lte is a real perception-under-decoy test; keep one. |
| cat-c2-01 | perception | 5 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c2-00 (only tick deltas). |
| cat-c3-00 (Tech Critical Path) | reasoning | 4 | 2 | 4 | 2 | **2** | FIX | building_total_gte + power_surplus_gte:0 is gamed by spamming powr (each +100, no prereq) — doesn't test the tech *path*. Require has_building tent/proc to force the dependency. |
| cat-c3-01 | reasoning | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c3-00. |
| cat-c4-00 (Power-Budget Online) | reasoning | 4 | 2 | 4 | 2 | **2** | FIX | Same gameability as c3 (powr-spam satisfies both clauses); near-identical to c3. Merge c3+c4 into one fixed pack. |
| cat-c4-01 | reasoning | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c4-00 (and of c3). |
| cat-c5-00 (Budget Allocation) | reasoning | 5 | 2 | 3 | 2 | **2** | FIX | powr n:2 + building_total bar, no income → "build 5 cheap buildings"; not an indivisible-budget trade. Same as c6. |
| cat-c5-01 | reasoning | 5 | 2 | 3 | 2 | **2** | CUT | Duplicate of cat-c5-00. |
| cat-c6-00 (Time-Boxed Deploy) | reasoning | 5 | 2 | 3 | 2 | **2** | FIX | own_units_gte+building_total; e1@100 trivial; duplicate decision of c5/economy-time-box. Merge. |
| cat-c6-01 | reasoning | 5 | 2 | 3 | 2 | **2** | CUT | Duplicate of cat-c6-00. |
| cat-c7-00 (Defensive-Direction) | perception | 4 | 3 | 4 | 3 | **3** | KEEP | building_in_region for pbox forces a directional commit; pbox needs tent (pre-placed) — solvable. Keep one. |
| cat-c7-01 | perception | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c7-00. |
| cat-c8-00 (Base-Placement) | perception | 4 | 3 | 4 | 3 | **3** | FIX | Same family as c7 (place building in inferred region). Distinct enough to keep one but merge c7/c8 conceptually; ensure region is reachable/buildable. |
| cat-c8-01 | perception | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c8-00. |
| cat-c9-00 (Commit-vs-Retreat) | reasoning | 4 | 3 | 4 | 3 | **3** | KEEP | units_killed + units_lost_lte is a real risk call; no real fail (dead −1) — FIX-lite: make fail = units_lost_gte cap. |
| cat-c9-01 | reasoning | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c9-00. |
| cat-c10-00 (Force Coordination) | action | 5 | 3 | 4 | 3 | **3** | KEEP | all_units_in_region + units_lost_lte = genuine rendezvous; better-spec'd than strategy-twobody. Keep one. |
| cat-c10-01 | action | 5 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c10-00. |
| cat-c11-00 (Tempo Window) | reasoning | 4 | 3 | 4 | 3 | **3** | KEEP | after_ticks + units_killed + within_ticks genuinely encodes a window; only tempo coverage — keep one. |
| cat-c11-01 | reasoning | 4 | 2 | 4 | 2 | **2** | CUT | Duplicate of cat-c11-00 (and decision-twin of c9). |
| cat-c12-00 (Error Recovery) | reasoning | 4 | 2 | 3 | 2 | **2** | FIX | "Replan after setback" but nothing is destroyed at start — it's just a timed build with after_ticks:1200. Not a recovery test; either script a mid-episode loss or CUT. |
| cat-c12-01 | reasoning | 4 | 2 | 3 | 2 | **2** | CUT | Duplicate of cat-c12-00. |

Aggregate: KEEP 19 · FIX 18 · CUT 11. (TEMPLATE excluded.)

---

## 4. CUT list (concrete — safe to delete now)

These add no distinct capability and/or are non-discriminating:

1. **`cat-c1-01`** — byte-near duplicate of cat-c1-00 (±1 pct/±200 tk).
2. **`cat-c2-01`** — duplicate of cat-c2-00.
3. **`cat-c3-01`** — duplicate of cat-c3-00.
4. **`cat-c4-01`** — duplicate of cat-c4-00 (and decision-twin of c3).
5. **`cat-c5-01`** — duplicate of cat-c5-00.
6. **`cat-c6-01`** — duplicate of cat-c6-00 (decision-twin of c5).
7. **`cat-c7-01`** — duplicate of cat-c7-00.
8. **`cat-c8-01`** — duplicate of cat-c8-00.
9. **`cat-c9-01`** — duplicate of cat-c9-00.
10. **`cat-c10-01`** — duplicate of cat-c10-00.
11. **`cat-c11-01`** — duplicate of cat-c11-00.
12. **`cat-c12-01`** — duplicate of cat-c12-00.

Net: keep exactly one representative per `cat-c*` category (the `-00`),
deleting all 12 `-01` actives — removes 12 files with zero capability
loss. (Category-merge candidates, can be done after: fold c4→c3, c6→c5,
c8→c7, c11→c9 since each pair is the same predicate decision.)

Additionally **CUT or hard-FIX**: `cat-c12-00` (not actually an
error-recovery test — nothing fails mid-episode; it is a delayed timed
build), and `strict-production-bom` HARD level (unsolvable on 4200
cash; either raise to ~5200 + pre-place `weap`, or delete the hard
level and keep easy/medium).

---

## 5. The repo-wide FIX (do this first, highest leverage)

**Add a real `fail_condition` to every active pack that lacks one.**
Today `run_level` only emits `loss` if a `fail_condition` evaluates
true; otherwise an unmet win stays `draw` (eval_core L174/L243). With
no fail, a do-nothing agent draws and a 95%-there agent draws — the
outcome axis cannot rank effort. The `cat-*` `units_lost_lte: -1`
"fail" is a dead no-op (units_lost ≥ 0 always).

Mechanical, non-gameable fixes using the *existing* vocabulary:

- Timed tasks (perception-*, reasoning-*, action-*, economy-*,
  strict-*, building-and-planning, cat-c1/c3/c4/c5/c6/c7/c8/c12):
  `fail_condition: {not: {within_ticks: <max_ticks>}}` — i.e. running
  out the clock without meeting the win is a *loss*, not a draw.
- Attrition tasks (anything with a `units_lost_lte: N` win clause):
  add `fail_condition: {not: {units_lost_lte: N}}` so exceeding the
  cap loses immediately (also prunes wasted ticks).
- `cat-*`: replace the dead `units_lost_lte: -1` with the timeout-loss
  above (one-line generator change; regenerate the kept `-00` files).

This single change restores a 3-way outcome (win/draw/loss) across the
suite and makes `objective_progress` partial credit meaningful instead
of every non-win collapsing to 0.5.

Secondary FIXes (per pack, already noted in the table):

- **strategy-dilemma / strategy-gauntlet / strategy-twobody**: win is
  `buildings_discovered_gte: 1` — satisfied by *seeing* the pre-placed
  enemy base, which a huge friendly stance-3 force does incidentally.
  Re-spec to `reach_region` (the objective) + `units_lost_lte`
  (enforce the risk/coordination the title claims).
- **cat-c3/c4**: add `has_building: tent` (or `proc`) to the win so
  `building_total_gte` can't be met by powr-spam; this restores the
  tech-*dependency* the pack claims to test.
- **economy-investment / -time-box / -force-buildup**: tighten
  `starting_cash` so the wide and deep paths are mutually exclusive
  (currently both fit), and add the timeout fail. Until engine S0/S1
  (ore income) lands these can only ever be "spend the obvious thing."
- **artofwar-lure-the-tiger**: add an `after_ticks` gate or a
  staging `reach_region` on the lure so the win actually requires the
  two-phase plan, not just any survivable route to the objective.
- **rush-hour**: drop it from the "strategy" claim; reduce the
  friendly force and require coverage (`explored_pct_gte`) + kills +
  timeout-fail so it tests coordinated sweep, not auto-battle.

---

## 6. ADD list (top 10 new scenarios to author)

Gaps the active set still has, with win/fail sketches in the existing
predicate grammar and the external benchmark each parallels.

1. **adversarial-counterstrategy-read** (capability: adversarial).
   Opponent commits one of two observable openings (rush near-lane vs
   tech far-corner); agent must scout then commit the dominant
   counter. Win: `all_of[{buildings_discovered_gte:1},{units_killed_gte:N},{within_ticks:T}]`;
   fail: `{not:{own_units_gte:1}}` or timeout. Parallel:
   StarCraft-II-Arena, game-theoretic LLM evals. (Fills the
   opponent-modeling gap — the unique RTS value prop.)

2. **longhorizon-opening-to-assault** (reasoning). One episode,
   40k+ ticks: scout (explored_pct) → tech (`has_building: proc`) →
   army (`own_units_gte`) → strike (`reach_region` enemy base) — all
   four as `all_of`, terminal-only credit, timeout fail. Parallel:
   Terminal-Bench 2.0 / OSWorld 50-step. (Fills long-horizon credit
   assignment — currently thin.)

3. **strict-toolban-fidelity** (action). Achieve `reach_region` but
   the allowlist excludes `attack*`; win additionally requires
   `units_lost_lte: 0` AND a `not{after_ticks:T0}`-then-`reach`
   ordering. Any disallowed-tool call → fail. Parallel: BFCL V4
   relevance / τ²-bench. (Isolates API fidelity as the objective.)

4. **economy-throughput-real** (reasoning) — *gated on engine S0/S1*.
   Once ore income exists: `economy_value_gte: V within_ticks:T`,
   wide (2 proc) vs deep (1 proc + harvesters) genuinely mutually
   exclusive. Parallel: PlanBench cost-optimal. (Replaces the 5
   non-discriminating economy/cat-c5/c6 packs with one real test.)

5. **perception-count-the-threat** (perception). Win:
   `enemies_discovered_gte: K` where K = exact hidden count and
   over/under-scouting wastes the clock; fail timeout +
   `units_lost_lte`. Parallel: ERQA state-estimation. (Strengthens
   the validated transfer target with a counting variant.)

6. **reasoning-replan-after-loss** (reasoning) — a *true* error-
   recovery pack (what cat-c12 only pretends to be). Pre-place the
   agent base in the path of a scripted stance-3 enemy push that
   destroys 1–2 buildings early; win =
   `all_of[{building_total_gte:N},{has_building:tent},{after_ticks:t_loss},{within_ticks:T}]`.
   Parallel: PlanBench replanning. (Requires interrupt/scripted-loss
   support — see §7.)

7. **adversarial-feint-handling** (adversarial). Opponent shows a
   decoy force on one axis; win keyed to committing against the
   *real* axis (`building_in_region`/`units_killed_gte` at the true
   threat), losing if you commit to the decoy (timeout). Parallel:
   opponent modeling / adversarial robustness.

8. **coordination-staggered-window** (action). Two squads must each
   be in *different* regions within *different* tick bands
   (`all_of` of two `reach_region` + paired `after_ticks`/`within_ticks`),
   forcing true parallel scheduling, not a single column. Parallel:
   Watch-And-Help / multi-robot dispatch. (Upgrades the thin
   strategy-twobody/cat-c10 coordination bucket.)

9. **tempo-double-window** (reasoning). Two separate strike windows
   in one episode (`after_ticks:t1`+`units_killed_gte:a` then a
   forced lull then `after_ticks:t2`+`units_killed_gte:b`), so acting
   continuously fails. Parallel: TextStarCraft II tempo. (Tempo is
   currently a single cat-c11 category.)

10. **navigation-confined-hard-only** (perception). Replace the
    trivial custom-map easy/medium with one hard map-read:
    `all_units_in_region` in a far bounded corner, tight clock,
    `fail_condition` on timeout — a real ARC-/ERQA-style spatial
    commit with no degenerate easy tier.

---

## 7. Predicate-vocabulary additions needed

The current leaf set is adequate for §6 #1,2,5,7,8,9,10. To fully
author the others, add:

- **`tool_called`/`tool_not_called: <name>`** (or a scenario-level
  `forbidden_tools` enforced as an automatic `fail`) — needed for
  #3 strict-toolban-fidelity; today the only tool-fidelity signal is
  the cross-cutting `actions_warned/issued` score, not a win/fail
  predicate. This is the most leaderboard-relevant missing primitive
  (BFCL/τ²-bench relevance detection).
- **`building_destroyed_gte` / `own_building_lost_gte`** — needed for
  #6 (detect the scripted setback) and to give offensive packs a real
  "took the base" predicate instead of proxying with
  `buildings_discovered`/`reach_region`.
- **An ordering composite (e.g. `then: [A, B]` meaning A must have
  held true at some tick strictly before B)** — `after_ticks` only
  approximates sequencing by wall-clock; a true happened-before would
  let artofwar-lure-the-tiger / sequenced-citadel grade *order*
  rather than *timing*, and make #2/#6 non-gameable.
- A scripted-adversary / mid-episode-event hook (engine side, already
  flagged in SCENARIO_AUDIT as Phase-1) is the prerequisite for #1,
  #6, #7 to be more than scripted-stance encounters.

---

## 8. Prioritized action summary

**Cut now (12 files, zero capability loss):** all `cat-c*-01` actives
(c1–c12 `-01`). Optionally also retire `cat-c12-00` (mislabelled) and
the `strict-production-bom` hard level (unsolvable) unless budgeted.

**Fix now (one mechanical pass):** add a real `fail_condition`
(timeout-loss; attrition-loss where a `units_lost_lte` win clause
exists) to all ~36 active packs lacking one, and replace the dead
`units_lost_lte:-1` in the kept `cat-*` files. This is the single
change that most improves benchmark discrimination.

**Fix next (per-pack, ~6 packs):** strategy-dilemma/gauntlet/twobody
(win predicate doesn't enforce the claimed decision), cat-c3/c4
(powr-spam gameable — add `has_building`), economy-* (tighten cash so
allocation is a real trade), artofwar-lure-the-tiger (add ordering
gate), rush-hour (re-spec or stop counting it as strategy),
strict-production-bom hard (raise budget or cut).

**Top 10 to add (ranked):** #1 adversarial-counterstrategy-read,
#2 longhorizon-opening-to-assault, #3 strict-toolban-fidelity,
#6 reasoning-replan-after-loss, #7 adversarial-feint-handling,
#8 coordination-staggered-window, #5 perception-count-the-threat,
#9 tempo-double-window, #4 economy-throughput-real (after S0/S1),
#10 navigation-confined-hard-only.

**Predicate additions (in priority order):** (1) tool-call
fidelity predicate / `forbidden_tools` auto-fail; (2)
`building_destroyed_gte`; (3) a happened-before ordering composite;
(4) scripted mid-episode adversary hook (engine, Phase-1).

---
## De-duplication (overlap analysis, 24→21 active)

Win-predicate Jaccard overstates overlap (small vocab); judged at the
decision level. Consolidated the genuinely-aimless dups (same map +
same win signature + same skill):

- **adversarial-siege, adversarial-skirmish → quarantined**:
  same as adversarial-duel (rush-hour-arena, kill-reactive-enemy-within
  -ticks-with-loss-cap); duel's easy→hard ladder + easy/medium configs
  already cover even→outnumbered→entrenched. adversarial-duel is the
  canonical adversarial scenario.
- **economy-time-box → quarantined**: = economy-force-buildup + a
  "keep a building standing" clause (same budget-under-clock decision).
  Kept economy-force-buildup + economy-investment (the latter adds a
  distinct economy-vs-military allocation choice).

Kept (NOT aimless dup — each serves a purpose):
- strategy-dilemma vs strategy-gauntlet: same objective but **different
  maps** (singles-dilemma vs singles-gauntlet) → different route puzzle.
- artofwar-decoy-sacrifice vs -lure-the-tiger: same map but genuinely
  distinct credit-assignment (sacrifice-and-lose vs displace-and-exploit).

Active set: 21 distinct decisions. Quarantined packs remain on disk,
runnable via explicit --packs, excluded from the default sweep.

## Per-scenario closer look — #1 action-multiunit-coordination

Difficulty axis (one new controlled variable per tier, so a tier
failure attributes to a single capability):

- **easy** — 2 regions, exact `(x,y)`. Tests *parallel dispatch*:
  serialized one-group-at-a-time touring blows the binding tick
  deadline (~30+ turns); only genuine simultaneous control (~17–20
  turns) makes it. Split enforced by `units_in_region_gte n=2` (not
  `reach_region`, which one touring unit satisfies).
- **medium** — 3 *dispersed* regions (NE / SW bottom-left / SE),
  exact `(x,y)`, attrition cap, infantry pickets on the two eastern
  lanes. Adds *multi-vector* dispatch: a single eastward column
  cannot reach the SW region.
- **hard** — byte-identical setup to medium (same regions, pickets,
  deadline, attrition); the ONLY changed variable is
  `objective_coords: relative` — the briefing gives compass labels
  ("the NORTH-EAST corner") instead of coordinates. Tests *spatial
  grounding*: the model must localize each target on the minimap
  (each region holds one enemy building as the on-map landmark).
  Deliberately NOT spawn-varied and NO turrets on the path —
  introducing either would add a second uncontrolled variable and
  break the clean medium→hard attribution (recorded as a reasoned
  exception to the task-#23 spawn-variation contract in
  `tests/test_hard_tier.py::NOT_APPLICABLE`).

`objective_coords` (`exact|relative`, schema `Level`/`ScenarioConfig`,
default `exact`) is a reusable knob: any pack can opt into
coordinate-blind objectives. The win predicate always evaluates
against the real engine coordinates regardless of disclosure.

## Per-scenario closer look — #2 action-sequenced-execution

Original defect: the pack claimed to test ordered, non-stalling route
execution but the win condition referenced only the *final* region +
a tick band — a beeline-to-final that skipped every waypoint and
idled to the gate won (verified: model arrived tick 1353, idled ~900
ticks, won). Not model compensation: the scenario didn't enforce what
it advertised.

Fix — new reusable **stateful** predicate `waypoint_sequence`: latches
ordered region-visit progress (monotonic) on a per-episode
`signals.seq_progress` scratch. Wk+1 only counts after Wk; skip /
out-of-order / idle ⇒ never satisfied. Coords-aware phrase
(`exact|relative`). Difficulty axis (one new controlled variable per
tier):

- **easy** — ONE ordered 3-waypoint route, clear lanes, generous
  budget. Run: WIN, W1→W2→W3 in order, tick 1893 < 2400.
- **medium** — TWO ordered routes that *cross*, BOTH required ⇒ must
  split into two parallel columns (serial overruns); attrition cap.
  Run: LOSS (valid) — model split correctly, finished route S in
  order, lost column discipline on N and timed out (the exact
  re-deliberation failure the pack targets).
- **hard** — TWO 4-waypoint serpentine routes on the larger
  `scout-arena` (176×80, `scripts/build_scout_arena_map.py`), given
  ONLY by relative *search bands* (no coords); fogged markers revealed
  by the `enemy_building_spotted` interrupt; seed-varied 2 spawn
  points. Run: LOSS (valid) — machinery all verified working
  (interrupts fired on the big map); model drove to literal corners
  instead of scanning the bands, ignored the revealed positions,
  attrition-bust. Solvable in principle; not compensated.

All tick budgets aligned to ~90 ticks/turn so timeout / attrition /
wipe are reachable **losses** within `max_turns` (no draw degeneracy).
Footgun recorded: a `base_map` placed at the Level level is silently
ignored — it must go inside `overrides:` (guard test added).

## Win-speed bonus (cross-cutting scoring)

Every episode records `win_tick` / `win_turns` / `win_budget` /
`speed` / `composite_base` (score.json + manifest). On a **win** the
composite gets `+SPEED_BONUS·speed` (`SPEED_BONUS=0.05`,
`speed = 1 − win_tick/budget`, budget = tightest `within_ticks` in the
win tree else `max_ticks`). Bounded so a fast win ranks above a slow
win but never lifts a loss above a win or overrides correctness.
Leaderboard carries `win_speed`/`win_turns` and breaks composite/
win-rate ties by faster wins.
