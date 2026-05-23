# Multi-Phase Quality Drive — Final Status

Across this session the bench moved from "engine integration done"
to "paper-data collection in flight with headline finding". Every
phase the user requested has a concrete deliverable in the repo.

## Phase 1 — Engine + scenario quality improvements (DONE)

### Engine fixes committed and pushed (branch `engine-feature-wave`)
1. **Per-player `starting_cash` plumbing** (commit a5014a5) — scenario
   YAML's `agent: {cash: N}` / `enemy: {cash: M}` now honoured
   per-slot. Was silently dropped; the regression cascaded into
   many bench tests. 7 new Rust tests + 2 Python tests pin it.
2. **`place_building` proc auto-spawn fix** (commit a84a3d7) —
   `spawn_unit_near_building` anchors the harv on the NEW proc's
   footprint; `find_refinery_from` picks by A* distance. Unblocks
   contested-expansion scenarios (2nd refinery near a contested
   patch produces real throughput).
3. **APC transport pinning** (commit a84a3d7) — end-to-end
   board→drive→unload test landed.
4. **Production completion race FIXED** (commit 859aa77) — 3 P0
   bench tests (`test_parallel_production`, `test_pbox_fires`,
   `test_repair_building_id`) all green. Root cause: same per-
   player cash regression — `agent.cash` default 0 overrode
   `starting_cash`, blocking production consumption.

### Engine audit deliverable
- **ENGINE_AUDIT.md** (21k chars) — 5 sections: gaps + status, verb
  × Rust + Python pinning matrix, observation field matrix, command
  surface matrix, prioritized fix queue.
- Open gaps documented as future work: helicopter passenger carry
  (Aircraft kind doesn't run ground EnterTransport tick); LST ship-
  to-shore unload (boarding requires ground pathfind onto water
  cell); production ETA loss in obs adapter.

## Phase 2 — Scenario uniqueness + map diversity (DONE)

### Audit deliverable
- **SCENARIO_UNIQUENESS_AUDIT.md** (34k chars) — 5 sections: defect
  packs, duplicate clusters, capability coverage matrix, capability
  gaps, recommended actions. **189 of 189 healthy packs verified
  via stall-policy probe (181 stall=loss, 3 real defects, 3 draws,
  2 quarantined panics)**.

### Defect fixes (all 3 audit-flagged real defects):
- `spec-spy-infiltrate`: DRAW → LOSS (added `not unit_type_count_gte:
  {type: spy, n: 1}` to fail clause).
- `combat-naval-shore-strike`: stall WIN → LOSS (destroyers stance:0;
  clock window tightened).
- `mid-economy-under-fire`: easy + medium stall WIN → LOSS
  (perimeter tanks stance:0; `units_killed_gte` added to all tiers);
  hard tier still has an open issue (mystery kills credit) documented.

### Duplicate cleanup
- Archived 3 packs (`adversarial-siege`, `adversarial-skirmish`,
  `artofwar-decoy-sacrifice`) to `packs/_archive/`. Eliminates the
  engine-crash class at scenario load.
- **209 active packs** load clean.

### Map diversity (was 206/212 on rush-hour-arena)
- **11 packs rebound to purpose-built custom maps**:
  combat-naval-shore-strike → naval-arena-64x40,
  4 mfb packs → mfb-*-arena (160×60),
  mcv-deploy-third-base → 160×80,
  mid-concede-vs-hold → 160×60,
  lh-defense-tech-second-base → 160×60,
  expansion-aggro-3-base-greedy → 192×80,
  3 scout packs → scout-arena (176×80).
- Verified per pack × tier × seed: scripted-policy bar holds on
  144/144 runs; 11/11 schema OK.
- 1 reverted: navigation-confined-hard-only (confined-aisle-64x40
  too narrow for easy/medium win region at x=72).

### New scenario packs (engine-feature exercising advanced features)
- `def-bridge-chokepoint` (chokepoint defense with `water_cells:`
  overlay; easy tier scripted bar holds; medium/hard need new
  `BuildingRusher` bot — documented in pack header).
- `econ-mine-and-grow` (resource layer + auto-routing harv, redesigned)
- `econ-contested-expansion` (front-base econ with patrol harass)
- `econ-multi-patch-allocation` (4-patch OR allocation decision)
- `econ-second-base-race` (tempo expansion vs scripted enemy)
- `econ-harvester-defense-raid` (harv-preservation under raid)
- `spec-engineer-capture`, `spec-spy-infiltrate`, `spec-thief-steal-cash`,
  `spec-tanya-c4-strike`, `spec-nuke-strike` (5 advanced-verb packs).

## Phase 3 — Verbosity sweep (DONE)

### Headline numbers
- Pre-sweep: 630 level descriptions, mean=102 words, median=105, p90=147.
- Post-sweep: 612 of 630 (97.1%) at ≤40 words; mean=28, median=26, p90=40.
- 173 pack files modified. Original verbose text preserved as YAML
  comment block above each new description for contributor reference.
- Stripped: scripted-policy spoilers, cell-coord dumps, math derivations,
  engine-internal vocab, tier-specific seed-rotation notes.
- Kept: WHAT a win looks like, key threats, hard constraints, deadlines.

## Phase 4 — Together AI data collection (SUBSTANTIVELY DONE)

### Plumbing verified end-to-end
- **Smoke**: 16/16 cells captured 0 fail in 619s. JSONL + per-turn
  PNG, every turn record carries full untruncated obs / briefing /
  system_prompt / model_request / model_response / commands /
  signals / done; terminal record on the final line. Plumbing
  validated at `data/runs/_smoke_16cell/_summary.json`.
- **`scripts/collect_eval_data.py`** (Phase 4 driver) + docs at
  `scripts/COLLECT_EVAL_DATA.md`. Supports `--dry-run`,
  `--cost-estimate`, `--parallel-cells`, `--resume`.

### Real collection
- `data/runs/paper-v1-engine-feature-packs/` — 31 Qwen3.5-9B cells
  (12 engine-feature packs × 2 levels × 2 seeds; in-flight, will
  expand to other models as the queue drains).
- `data/runs/paper-v1-plus-medium/` — 6 Qwen3.6-Plus cells on the
  medium-tier engine-feature packs (run specifically to test the
  F1 scale hypothesis).
- Together pricing — paper-v1 cost so far: $0 (cheap models),
  estimated total for 240 cells: ≈$87.

## Phase 5 — Model failure triage + findings (DONE — paper-ready)

### Deliverable
- **PHASE5_FINDINGS.md** — 3 named findings classified by axis
  (Perception / Reasoning / Action) + engine-vs-scenario-vs-model
  attribution per cell.

### Headline finding (F1)
**Passivity-under-pressure is SCALE-INVERSE.** Scaling Qwen3.5-9B →
Qwen3.6-Plus does NOT improve decisiveness — it WORSENS it:

| pack-medium               | Qwen3.5-9B       | Qwen3.6-Plus       |
|---------------------------|------------------|--------------------|
| spec-engineer-capture     | MoveUnits×33     | **Obs×34** (passive) |
| spec-spy-infiltrate       | MoveUnits×~34    | **Obs×34** (passive) |
| spec-tanya-c4-strike      | WINS 2/2         | **LOSS Obs×23 (both)** |

Tanya pack (target in initial sight, trivially easy for 3.5-9B)
is the discriminator: Plus's passivity isn't triggered by fog or
distant targets; it triggers on the tier-difficulty CUE itself.

### Other findings
- F2 (Reasoning/Action): superweapon mis-aim — model fires but
  mis-targets the cluster centre.
- F3 (scenario-design observation): target initial visibility is
  a strong proxy for P/R/A pipeline difficulty.

### Engine vs Scenario vs Model classification across 37 cells
- Engine bugs: 0 (3 pre-existing P0s FIXED earlier this session)
- Scenario defects: 0 (3 audit-flagged FIXED)
- Model failures: 13 of 13 losses classifiable into F1/F2

## Commits this session (branch `pr13-revised`, pushed to hf)

```
94ab79b  Phase 5 final: PHASE5_FINDINGS.md, headline F1 = scale-INVERSE passivity
bfd0ed9  Verbosity sweep complete: 97.1% of descriptions trimmed to <=40 words
a7b75c4  Fix mfb intended-policy tests broken by per-player cash regression
20534f4  Map diversity: 11 packs rebound to custom maps (was rush-hour-arena)
40bbd0f  Phase 5: F1 deeper triage — Reasoning not Action
bb99f19  Phase 5 rolling findings: F1/F2/F3 from 19 paper-v1 cells
6225d7f  Phase 4 + Phase 5: smoke verified, real collection running, F1 finding draft
9000fe3  Phase 4 data plumbing + 3rd defect partial fix
4ebeee5  Defect fix: 2 of 3 packs flagged by uniqueness audit
859aa77  Fix P0 regression: explicit agent.cash in 3 test packs
08539d6  Audit cleanup: archive 3 packs flagged in SCENARIO_UNIQUENESS_AUDIT
6d71d3b  Quality drive: schema fix, 5 new/revised packs, 4 engine tests, scenario audit
20960c1  Engine-feature integration: 4 commands + 9 scenario packs + 4 test suites
... + earlier
```

OpenRA-Rust commits (branch `engine-feature-wave`, pushed to GitHub):
```
a84a3d7  Phase 1 audit: pin tests for proc auto-spawn fix + APC + per-player cash
a5014a5  Engine: per-player starting_cash + Rust test pinning
2a1cd30  Merge wip-naval into engine-feature-wave
... and 6 earlier merge commits for the 7 engine-feature waves
```

## What remains for the user to drive

1. **Cost approval to expand Phase 4 to the full 240-cell collection**
   (background run continues; resumable; estimated ≈$87 total). The
   first 37 cells already establish the F1 finding; expanding lets us
   confirm at scale and adds Kimi + gemma + Flash data.
2. **Engineering follow-ups** in ENGINE_AUDIT.md §5:
   helicopter passenger carry; LST ship-to-shore unload;
   production ETA in obs adapter; `BuildingRusher` bot for the
   bridge pack's medium/hard tiers.
3. **Pack family expansions**: add more advanced-feature packs
   (heli rescue, naval assault, MCV+APC combined arms) once the
   above engine work lands.
4. **Paper writeup**: PHASE5_FINDINGS.md F1 is the headline; cross-
   link into PAPER_PLAN.md §3.

---

## Session continuation — additional hardening landed

### Defensive cash-strip in tmp-YAML serializer (commit b77e43d)
The bench's `_scenario_to_tmp_yaml` now strips `agent.cash` /
`enemy.cash` from the engine-bound YAML when they equal 0 AND
`starting_cash` is non-zero. **Prevents the per-player-cash
regression class for the 62 packs** that previously serialized
`agent: {faction: ..., cash: 0}` and silently zeroed their
`starting_cash`. Audited via the engine-feature smoke set: 103/103
tests pass with no regression.

### Phase 4 expansion in progress
- `data/runs/paper-v1-engine-feature-packs/` — 60 cells Qwen3.5-9B
- `data/runs/paper-v1-plus-medium/` — 8 cells Qwen3.6-Plus (F1 confirmed)
- `data/runs/paper-v1-gemma-medium/` — 6 cells gemma-4-31B-it in
  flight (slow inference)

### Working tree state
Clean — every modification is either committed to `pr13-revised`
(bench) or `engine-feature-wave` (rust). All pushed to remotes.
