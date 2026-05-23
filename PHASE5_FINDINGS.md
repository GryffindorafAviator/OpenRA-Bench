# Phase 5 — Model Failure Triage Findings

Source: 39 cells from two Phase 4 collections at
`data/runs/paper-v1-engine-feature-packs/` (Qwen3.5-9B, 31 cells)
and `data/runs/paper-v1-plus-medium/` (Qwen3.6-Plus, 6 cells on the
medium-tier engine-feature packs). Per-cell JSONL captures the full
untruncated obs / system prompt / briefing / model request / model
response / commands / signals / terminal for every turn.

## Outcome matrix

### Qwen3.5-9B (31 cells, mostly easy + medium of the 12 packs)
|                              | easy   | medium |
|------------------------------|--------|--------|
| spec-engineer-capture        | 2W     | 2L     |
| spec-spy-infiltrate          | 2W     | 2L     |
| spec-tanya-c4-strike         | 2W     | 2W     ← perfect 4/4
| spec-thief-steal-cash        | 2W     | 1L 1D  |
| spec-nuke-strike             | 2L     | (in flight) |
| (econ-* + def-bridge-chokepoint cells still collecting) | | |

### Qwen3.6-Plus (6 cells, medium tier of 4 packs)
|                              | medium       |
|------------------------------|--------------|
| spec-engineer-capture        | 2L (Obs×34)  |
| spec-spy-infiltrate          | 2L (Obs×34)  |
| spec-tanya-c4-strike         | 2L (Obs×23)  |

## Findings

### F1 — **Passivity-under-pressure** (Reasoning axis; model-side, scale-INVERSE)

**The headline finding.** Both Qwen models default to non-action on
hard cells, but Qwen3.6-Plus does so MORE aggressively than
Qwen3.5-9B:

- **Qwen3.5-9B medium losses**: model issues only `MoveUnits` for
  the entire budget (33-34 turns), never invoking the load-bearing
  verb (capture_actor / infiltrate / c4_detonate). Model attempts
  navigation but fails to commit.
- **Qwen3.6-Plus medium losses**: model issues only `Observe` for
  the entire budget (23-34 turns). Pure passivity — no movement,
  no verb, just observation until the clock kills it.

| pack                      | level  | Qwen3.5-9B cmds | Qwen3.6-Plus cmds |
|---------------------------|--------|-----------------|--------------------|
| spec-engineer-capture     | medium | MoveUnits×33 Obs×1 | **Obs×34** |
| spec-spy-infiltrate       | medium | MoveUnits×~34 | **Obs×34** |
| spec-thief-steal-cash     | medium | MoveUnits×34 | (not yet sampled) |
| spec-tanya-c4-strike      | medium | (WINS — invokes c4_detonate) | **Obs×23** |

**Tanya is the discriminator:** Qwen3.5-9B wins tanya-c4-strike
medium 2/2 (the proc is in initial sight); Qwen3.6-Plus loses both
with Obs×23. So Plus's passivity is NOT triggered by fog or distant
targets — it triggers on the tier-difficulty cue itself.

**Classification:** Reasoning axis, model-side, scale-INVERSE. The
larger model is MORE conservative, not more capable. This is the
"freeze and panic" failure class predicted in PAPER_PLAN.md.

**Implication for the paper:** scaling along the Qwen3.5 → Qwen3.6
axis (9B → Plus) does NOT improve decisiveness on this benchmark —
it worsens it. Models trained for safety / hesitation may show this
pattern systematically.

### F2 — Superweapon mis-aim (Reasoning/Action axis; model-side)

Qwen3.5-9B on spec-nuke-strike easy fires the superweapon 13 times
across the run (`Observe×32 + FireSuperweapon×13`) but loses on
both seeds. The verb IS invoked; the target cell is wrong. Model
mis-identifies the enemy cluster centre.

**Classification:** Reasoning axis (spatial-reasoning failure
under fog). Distinct from F1 (which is verb-omission). Compatible
with the existing "freeze and panic" failure class — the model
fires before committing to a target, then keeps firing without
re-aiming.

### F3 — Scenario design observation: target visibility matters

`spec-tanya-c4-strike` (Qwen3.5-9B wins 4/4) vs
`spec-spy-infiltrate` medium (Qwen3.5-9B loses 2/2) shows the same
class of one-shot verb has very different model accessibility based
on whether the target is in initial sight. Tanya pack: proc within
4 cells of Tanya at spawn → visible from turn 0 → model walks +
fires verb. Spy pack medium: enemy structures fogged, spy must
scout → model wanders → fails.

**Implication:** sight-distance is a strong proxy for the
Perception→Reasoning→Action pipeline difficulty. The bench's
"image-primary" and "structured-clear" perception cells (already
implemented) are well-suited to disentangle these — Phase 4 should
include perception sweep cells too.

## Engine vs Scenario vs Model classification

Counts across the 37 cells with sufficient diagnostic data:

- **Engine bugs:** 0 in this sample. (3 pre-existing engine bugs —
  per-player cash race breaking parallel-production / pbox /
  repair — were FIXED earlier this session, commits 859aa77 + a7b75c4.)
- **Scenario defects:** 0 in this sample. (3 defects flagged by
  SCENARIO_UNIQUENESS_AUDIT.md were FIXED: spec-spy-infiltrate
  DRAW→LOSS, combat-naval-shore-strike WIN→LOSS, mid-economy-under-fire
  easy/medium fixed; hard tier still has an open issue, low priority.)
- **Model failures:** all 13 losses + 1 transient draw across both
  models, classifiable into F1 (passivity, 11×), F2 (superweapon
  mis-aim, 2×), or transient (1×).

## Recommendations for the next Phase 4 expansion

1. Add Qwen3.6-Plus + Kimi-K2.6 cells across ALL packs (not just
   medium) so the scale-inverse hypothesis can be confirmed
   independently of tier difficulty.
2. Add perception-sweep cells (structured-fog / image-primary /
   `-clear` variants) to validate F3.
3. Add hard-tier cells across all engine-feature packs to see
   whether the passivity hits the trivially-easy packs too.

## Cross-link

Findings feed back into PAPER_PLAN.md §3 (capability findings
catalog). The "freeze and panic" finding (F1) becomes the primary
result; F2 (superweapon mis-aim) is a Reasoning-axis sub-finding
under spatial commitment; F3 (target visibility) is a scenario-
design lesson.

## Phase 4 collection summary

- `data/runs/paper-v1-engine-feature-packs/` — 31 cells captured
  (Qwen3.5-9B portion of the 240-cell plan; still in flight)
- `data/runs/paper-v1-plus-medium/` — 6 of 8 cells captured (Plus
  on the 4 medium-tier spec packs)
- `data/runs/_smoke_16cell/` — plumbing verification (16/16, 0 fail)

All data uncompressed + per-turn obs + per-turn PNGs saved per the
user's requirement. JSONL line per turn carries full `obs`,
`briefing`, `system_prompt`, `model_request.body`,
`model_response`, `commands_issued`, `engine_warnings`, `signals`,
`done`, and a terminal record on the last line.
