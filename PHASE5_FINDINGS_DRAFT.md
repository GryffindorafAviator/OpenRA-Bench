# Phase 5 Model Failure Triage — Findings (draft, rolling)

Source: in-progress `paper-v1-engine-feature-packs` collection at
`data/runs/paper-v1-engine-feature-packs/`. 240-cell plan: 5
Together models × 12 engine-feature packs × 2 levels × 2 seeds ×
`vision` fog. As of this snapshot, 19 cells complete (all
Qwen/Qwen3.5-9B — first model in the queue).

## Outcome matrix (Qwen3.5-9B only, 18 of first 48 cells)

|                              | easy   | medium |
|------------------------------|--------|--------|
| spec-engineer-capture        | 2W     | 2L     |
| spec-spy-infiltrate          | 2W     | 2L     |
| spec-tanya-c4-strike         | 2W     | 2W     ← perfect 4/4
| spec-thief-steal-cash        | 2W     | 1L 1D  |
| spec-nuke-strike             | 2L     | (in flight) |

(Other models — Qwen3.6-Plus, qwen3.6-flash, gemma-4-31B-it,
Kimi-K2.6 — still pending in the queue.)

## F1 — Action axis: **special-ability verb omission** (model-side)

The dominant Qwen3.5-9B failure mode: across **6 of 9 losses**, the
model issues ONLY `MoveUnits` (and at most one `Observe`) for the
entire decision budget, never invoking the load-bearing verb the
pack tests:

| pack                      | level  | seed | n_turns | commands              | verb missing |
|---------------------------|--------|------|---------|-----------------------|--------------|
| spec-engineer-capture     | medium |   1  |   34    | MoveUnits×33 Obs×1    | capture_actor |
| spec-engineer-capture     | medium |   2  |   34    | MoveUnits×33 Obs×1    | capture_actor |
| spec-spy-infiltrate       | medium |   1  |   ~34   | MoveUnits×~34         | infiltrate |
| spec-spy-infiltrate       | medium |   2  |   ~34   | MoveUnits×~34         | infiltrate |
| spec-thief-steal-cash     | medium |   1  |   34    | MoveUnits×34          | infiltrate |
| spec-thief-steal-cash     | medium |   2  |   34    | MoveUnits×33 Obs×1    | infiltrate |

**Diagnosis:** the verb is in the tool list with a clear description
(verified in `agent.py`); the model "knows" to position the unit but
doesn't pivot from `move` → `capture/infiltrate` on arrival.

**Classification:** model capability gap on the Action axis. The
Perception (the model sees the target) and Reasoning (the model
decides to move toward it) phases work; the Action phase (committing
to the verb that wins) fails.

**Discriminating evidence for Phase 5b — does scale fix it?** When
Qwen3.6-Plus / gemma / Kimi cells land, compare. Hypothesis: F1 is
scale-correlated; the bigger model pivots; the 9B doesn't.

## F2 — Action axis: **superweapon mis-aim** (model-side)

`spec-nuke-strike easy` lost on both seeds despite the model
ACTUALLY firing the superweapon 13 times across the run. Output:
`Observe:32 FireSuperweapon:13`. So the verb is invoked but the
target cell is wrong (firing on empty terrain or own units).

**Classification:** Reasoning/Action — model has the verb, knows
WHEN to fire (charge timer met), but mis-targets. Distinct from F1
(verb-omission). Compatible with the "freeze and panic" failure
class: model fires before identifying the cluster's centre.

## F3 — Capability: **Tanya C4 is the easiest** (model-side, surprising)

`spec-tanya-c4-strike` is the ONLY pack Qwen3.5-9B wins on every
seed × every level (4 of 4 in 18-cell sample). C4 is a one-shot,
one-target verb (`c4_detonate(unit, building)`). Spy infiltration
should be similarly one-shot but loses on medium (F1). The
difference: spy needs a long walk through fog; Tanya walks to a
proc that's almost adjacent already.

**Classification:** scenario-design observation — Tanya pack's
geometry (proc within initial sight of Tanya) makes the verb
trivially discoverable. Spy pack's geometry hides the targets
behind fog. The model handles "walk to visible target then fire
verb" better than "scout to find target then fire verb."

## F4 — Transient draw on thief-steal-cash easy (smoke; not reproduced)

The smoke (`/tmp/phase4-multi/.../spec-thief-steal-cash__easy__seed1__vision.jsonl`)
ended with `outcome=draw` after ONE turn (`MoveUnits×1`). Phase 4
collection on the same cell now reports `win` and `win`. The smoke
result may have been a transient (model issued an empty command set
or the engine logged the wrong outcome). Not reproducing — drop
from the findings list pending replay.

## Engine vs scenario vs model in this sample

- **Engine bugs:** 0 in this sample (the smoke + 19 Phase 4 cells
  all produced rc=0 + complete JSONL + per-turn PNGs; no engine
  panics).
- **Scenario defects:** 0 in this sample of the engine-feature
  packs (the wider audit at SCENARIO_UNIQUENESS_AUDIT.md found 3
  defects but those are in non-engine-feature packs — already
  fixed in commits 4ebeee5, 9000fe3, this commit removed 2 of
  them via archive).
- **Model failures (Qwen3.5-9B):** 9 losses + 1 transient draw,
  all classifiable into F1 (verb omission, 6×) and F2 (verb
  mis-aim, 2×) and 1 outlier.

## Next actions

1. Wait for paper-v1 collection completion (background, PID 51416,
   ~150-300 minutes total).
2. Re-run this triage on the 240-cell full data. Specifically test:
   - F1 hypothesis: scale-correlated? compare 9B vs Plus vs Kimi
   - F2 hypothesis: does any model hit the cluster reliably?
   - F3 hypothesis: per-pack difficulty ranking — does the C4
     ranking hold across models?
3. Replay any LOSS or DRAW that doesn't fit F1/F2/F3 to classify.
4. Finalize as `PHASE5_FINDINGS.md` (no -DRAFT).
5. Cross-link into PAPER_PLAN.md §3 (findings catalog).

---

## F1 deeper triage (post-tool-list-confirmation)

The tool list IS correctly surfaced in the model request (verified
via `body.tools` drill-down). So `capture_actor` IS available to
Qwen3.5-9B but isn't being invoked. The richer triage from the
JSONL `briefing` field reveals the actual failure shape:

**spec-engineer-capture medium**, the engineer starts at (10, 25) on
the 'different latitude' from the enemy proc at ~(22, 18). With e6
sight-range 4, the proc is **invisible from spawn** (Chebyshev
distance ~13). The briefing shows `Enemies: none visible` on turn 1.

The model walks east-SOUTH-east: (10,25)→(20,25)→(30,25)→(40,25)→
(50,25)→(60,25)→... The proc is NORTH-east, not SOUTH-east. The
model is exploring the wrong half-plane.

**Reclassification of F1:**
- Action axis: NOT broken (the model would issue `capture_actor`
  if the proc were visible)
- Reasoning axis: ROUTE-PLANNING UNDER PARTIAL INFO is the failure.
  The model has the objective brief ("engineer is on a different
  latitude — must path around") but doesn't translate that into a
  search policy that biases toward the OPPOSITE latitude.
- Perception axis: model correctly observes "Enemies: none
  visible" — perception is fine.

This is the same class as the existing "freeze and panic" finding —
the model defaults to a simple direction (east) instead of
reasoning about the briefing's hint about latitude.

**For the paper:** F1 is best framed as **"Reasoning failure: brief
hint not translated to search policy."** Tests if the model can
extract "different latitude → search the other latitude" from a
natural-language briefing.

## Phase 5 status (rolling)

- 19 of 240 cells complete (Qwen3.5-9B only so far).
- F1, F2, F3 documented above.
- F4 (1-turn DRAW) dropped — not reproducing on Phase 4 collection.
- Next data milestone: when Qwen3.6-Plus cells land, retest F1
  scale hypothesis (does the Plus model translate the brief hint?).
