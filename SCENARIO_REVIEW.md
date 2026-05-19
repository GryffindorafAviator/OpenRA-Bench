# OpenRA-Bench — Scenario Improvement Plan

Review + roadmap. Tasks are independently mergeable, each with a level
of effort (LOE) estimate. Grouped by priority. Prepared 2026-05-19.

Companion docs: `SCENARIO_AUDIT.md` (gap analysis vs frontier benches),
`SCENARIO_QUALITY.md` (per-pack 1–5 scoring), `SCENARIO_BRAINSTORM.md`
(engine-prereq blockers), `SCENARIO_CATALOG.md` (historical design
notes from the now-retired auto-generated corpus).

---

## Current state (2026-05-19)

- 27 YAML files in `openra_bench/scenarios/packs/`: 1 TEMPLATE,
  5 quarantined, **21 active** (the `cat-*` auto-generated corpus was
  retired in `e9d5097`).
- Major audit findings already landed: strategy-* win predicates
  rewritten to `enemy_key_buildings_destroyed`, adversarial-duel got
  interrupts + a hard tier, multi-unit-coordination got real fail
  conditions and a coordinate-blind hard tier, 24→21 active de-dup.
- Predicate vocabulary covers spatial / state-estimation / counting /
  ordering (`waypoint_sequence`) / production-spec / building
  placement. Notable absences: tool-call fidelity, building destruction
  by the agent, happened-before composite.
- `run_level` only emits `loss` when a `fail_condition` evaluates true
  (`eval_core.py`). Unmet wins without a fail = `draw`. This is the
  single highest-leverage discrimination axis in the codebase.

## P0 — already shipped

Landed in [PR #6 — *P1: scenario fail_conditions + new long-horizon pack*](https://github.com/yxc20089/OpenRA-Bench/pull/6). Three changes that needed no new predicates:

1. **`rush-hour` easy/medium hardened.** Both levels now require a
   `units_lost_lte` attrition cap and have `fail_condition` on
   force-destruction OR timeout. Easy raised from kill 3 to kill 5
   (cap 6 lost); medium from kill 6 to kill 7 (cap 4 lost). Hard
   adds a timeout-fail to its existing force-destruction-fail. Removes
   the "32 stance-3 units auto-win" path on easy/medium.
2. **`economy-investment` fail_conditions on all 3 levels.** Each
   level now loses on timeout (and hard additionally loses if it
   exceeds the `units_lost_lte: 0` cap). Win/draw/loss is restored.
3. **New pack `longhorizon-opening-to-assault.yaml`** authored. One
   scout-tech-army-strike chain in a single 24k–40k tick episode with
   terminal-only `all_of` credit and timeout-fail. Fills the
   long-horizon credit-assignment gap named in `SCENARIO_AUDIT.md`.

All three changes use only the existing predicate vocabulary, validate
against the win-condition schema, and are isolated to the affected
files (no engine changes).

---

## P0 — remaining (mechanical, no new predicates)

These cost minutes each and unlock the largest discrimination gain.

| # | Task | LOE | Files | Notes |
|---|---|---|---|---|
| P0-1 | Add `fail_condition: {after_ticks: <max_ticks+1>}` to every level missing one in the 7 active packs that still have zero fail clauses | 20 min | `building-and-planning`, `custom-map-no-enemy`, `economy-force-buildup`, `reasoning-risk-route`, `strict-sequence`, plus quarantined `economy-time-box`, `economy-harvest-*` if un-quarantining | Mirror any `units_lost_lte: N` win clause as a fail (`not: {units_lost_lte: N}`) so attrition loses immediately |
| P0-2 | Add `fail_condition` to easy/medium of the 3 packs that only have it on hard (`perception-frontier-reading`, `perception-target-vs-fog`, `reasoning-frontier-commit`) | 10 min | 3 files | Same template as P0-1; today these levels can only win or draw |
| P0-3 | Fix `strict-production-bom` HARD solvability: pre-place `weap` (war factory) and/or raise `starting_cash` from 4200 to ~5200 | 10 min | 1 file | Today the spec requires `tsla` whose tech path costs ~4600 on soviet — currently unsolvable; this is the BFCL-style fidelity anchor and is worth saving |
| P0-4 | Add ordering gate to `artofwar-lure-the-tiger`: require a staging `reach_region` BEFORE `after_ticks: T0` AND the objective `reach_region` AFTER `T0`. Today the win is just "any survivable route to the objective" — the "two-phase lure" narrative isn't enforced by the predicate | 20 min | 1 file | Uses existing predicates only |
| P0-5 | Decide on the economy-investment "split path" framing: as written, easy/medium reward the wide path and hard the deep path. Document this clearly in the level descriptions so the cross-level read is "alternate allocations under tighter clocks", not "choose wisely" | 10 min, doc-only | 1 file | The audit's "wide-vs-deep both fit" critique is already partially addressed by per-level budget tightening |

**P0 total LOE: ~1h 10m. Highest-leverage hour in the repo.**

---

## P1 — single-pack improvements (no new predicates)

| # | Task | LOE | Files | Notes |
|---|---|---|---|---|
| P1-1 | Quarantine or formally retire `cat-c12` analogue thinking from any remaining docs — `error-recovery` cannot exist without mid-episode scripted-loss support; the `after_ticks` "wait then build" framing of the old `cat-c12` is dishonest. Documentation cleanup | 15 min | `SCENARIO_CATALOG.md` (historical doc) | Catalog doc is already marked RETIRED; just remove or amend the C12 row |
| P1-2 | Promote `rush-hour` MEDIUM thresholds onto a new HARDER level and split current hard into `hard` + `expert` with explicit coverage clauses (`explored_pct_gte`). Trim the agent force in a new tier so the sweep is real | 1 day | `rush-hour.yaml` | The audit's "drop friendly force size, require coverage" rebalance; deferred from this commit because the actor list is ported and large |
| P1-3 | Author `perception-count-the-threat`: win on `enemies_discovered_gte: K` where K is the exact hidden count; over-scouting wastes the clock, under-scouting fails the bar | 4 h | new file | Strengthens the validated ERQA transfer target with a counting variant; no new predicates |
| P1-4 | Author `coordination-staggered-window`: two squads must reach two different regions in two different tick bands (`all_of` over paired `after_ticks`/`within_ticks`), forcing genuine parallel scheduling | 4 h | new file | Upgrades the thin coordination bucket beyond strategy-twobody / multi-unit |
| P1-5 | Author `tempo-double-window`: two separate strike windows in one episode, gated by `after_ticks: t1` + kill bar, forced lull, `after_ticks: t2` + second kill bar. Acting continuously fails | 4 h | new file | Tempo coverage was thin once `cat-c11` was cut |
| P1-6 | Author `navigation-confined-hard-only`: one hard level only, `all_units_in_region` in a far bounded corner with tight clock + timeout fail. Replace `custom-map-no-enemy` easy/medium triviality | 3 h | 1 new + 1 edited file | Removes the auto-win baseline that the audit flagged |
| P1-7 | Land engine-prereq S0 (schema-valid ore source field per scenario) and S1 (silo resource storage so `silo` is not inert). Once landed, un-quarantine `economy-harvest-{timebox,investment}` and rewrite them around `economy_value_gte` so they test throughput, not spend | 2–3 days engine + 1 day scenario | engine + 2 quarantined files | Unblocks a real economy bucket; until then those packs inflate coverage claims |

**P1 total LOE: ~5–7 days. Each subtask independently mergeable.**

---

## P2 — new predicate vocabulary (engine-side)

Unblocks the strongest scenarios in the audit's ADD list. Each new
predicate is a one-line entry in `win_conditions.py` plus a signal in
the rust adapter; the work is in the adapter, not the YAML grammar.

| # | New predicate | What it enables | LOE |
|---|---|---|---|
| P2-1 | `forbidden_tools: [list]` (scenario-level field; auto-fail on disallowed tool call) and/or `tool_called` / `tool_not_called` leaves | Strict-action-API fidelity isolated as the objective — BFCL V4 / τ²-bench / IFBench analogue. The audit's #1 unaddressed structural gap | 1 day engine + 1 day scenario authoring (`strict-toolban-fidelity`) |
| P2-2 | `building_destroyed_gte: int` and `own_building_lost_gte: int` (own buildings lost; needed for scripted-loss recovery) | True error-recovery scenario (`reasoning-replan-after-loss`) where a scripted enemy push destroys 1–2 own buildings early and the win is rebuild-around-it. Also gives offensive packs a real "took the base" predicate beyond `enemy_key_buildings_destroyed` | 1 day engine + 1 day scenario authoring |
| P2-3 | `then: [A, B]` happened-before composite (A held true at some tick strictly before B) | True ordering for `artofwar-{lure,sequenced-citadel}`, longhorizon, and any sequenced-phase test. `after_ticks` only approximates by wall-clock, not by "A occurred first" | 1 day engine |

**P2 total LOE: ~4–6 days.** Order matters: P2-1 first (most leaderboard relevance), then P2-2, then P2-3.

---

## P3 — strategic / cross-cutting (multi-day)

| # | Task | LOE | Notes |
|---|---|---|---|
| P3-1 | Implement Phase-1 second-RL-slot adversary (a live RL-controlled enemy, not scripted stance-3 actors) | engine milestone | Unblocks `adversarial-counterstrategy-read`, `adversarial-feint-handling`, opponent-modeling family. The unique RTS value vs every other benchmark; current `adversarial-duel` is the best we have under the scripted-enemy constraint |
| P3-2 | Author `adversarial-counterstrategy-read`: opponent commits one of two observable openings (rush vs tech); agent scouts and picks the dominant counter | 1 day (post P3-1) | Game-theoretic LLM eval analogue (StarCraft-II-Arena) |
| P3-3 | Author `adversarial-feint-handling`: opponent shows a decoy force on one axis; win keyed to committing against the *real* axis | 1 day (post P3-1) | Adversarial robustness / opponent modeling |
| P3-4 | Pre-register external transfer panel: ERQA (primary correlate, validated by lmgame-Bench) + Blocksworld/PlanBench + BFCL V4 + GSM8K (negative control) + a coding eval (negative control). Report per-axis deltas, not aggregate | 2 days harness + 1 day per benchmark wiring | Without this the "rush-hour finetune → ERQA" headline claim is not defensible. lmgame-Bench protocol |
| P3-5 | Per-pack `weakest_link` diagnostic surfaced in the HTML scenario catalog; today it lives in `scoring.py` but the catalog page only shows win/loss outcome | 4 h frontend | Helps users see WHICH of perception/reasoning/action a model failed on |
| P3-6 | Generalization-gap reporting: held-out seeds per pack, report `(train-seed accuracy − held-out accuracy)` as a first-class metric in the leaderboard | 1–2 days | Procgen/SMACv2/ARC-AGI discipline; already designed for in the seed/randomize machinery, but not surfaced |

**P3 total LOE: ~2–3 weeks. P3-1 gates P3-2/P3-3.**

---

## Suggested execution order

```
This commit (P0-this) ─────────────► commit a
                                       │
P0-1 + P0-2 + P0-3 + P0-4 + P0-5 ────► commit b  (one mechanical pass, ~1h)
                                       │
P1-1 (doc cleanup, free) ─────────────► commit c
                                       │
P2-1 (tool-fidelity predicate) ───────► commit d (engine + new pack)
                                       │
P1-3, P1-4, P1-5 (new authoring, P) ──► commits e–g (parallelizable)
                                       │
P2-2 + P1 replan-after-loss ──────────► commit h
                                       │
P3-1 (RL adversary) + P3-2 + P3-3 ────► milestone "live adversary"
                                       │
P3-4 (transfer panel pre-reg) ────────► milestone "headline claim"
```

The commit b mechanical pass and P2-1 (tool-fidelity primitive) are the
two highest-leverage moves not yet made. Everything else is incremental
authoring against an already-sound predicate grammar.

## Out-of-scope (do not pursue)

- **Re-generating a `cat-*` style auto-corpus.** The audit's
  `SCENARIO_CATALOG.md` retirement note is correct: high level count,
  low *distinct-capability* count, near-duplicate variants, mild
  SMAC-style memorization risk. The curated, hand-authored direction is
  right; do not re-introduce a generator-driven family.
- **Adding "Elo leaderboard" claims before P3-1 lands.** With no live
  adversary, head-to-head Elo is only meaningful for fixed-scenario
  composite scoring; framing it as game-theoretic skill is misleading.
- **Tuning `rush-hour` further than P1-2.** It is currently sound as a
  multi-target sweep test (post-this-commit) and a "ported reference
  scenario". Its load-bearing weight for the strategy claim should be
  carried by `strategy-{dilemma,gauntlet,twobody}` and (eventually) the
  adversarial family.
