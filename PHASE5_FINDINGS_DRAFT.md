# Phase 5 Model Failure Triage — Findings (draft)

Source: 16-cell smoke run on `Qwen/Qwen3.5-9B`, 4 engine-feature spec
packs × 2 levels × 2 seeds, `vision` fog. Per-cell JSONL at
`/tmp/phase4-multi/`. Full 240-cell paper-v1 collection running in
background.

## Outcome matrix

|                              | easy   | medium |
|------------------------------|--------|--------|
| spec-engineer-capture        | 2W     | 2L     |
| spec-spy-infiltrate          | 2W     | 2W     |
| spec-tanya-c4-strike         | 1W 1L  | 1W 1L  |
| spec-thief-steal-cash        | 1W 1D  | 2L     |

## F1 (new candidate finding) — Action: special-ability verb omission

Across **6 of 7 losses**, Qwen3.5-9B issues ONLY `MoveUnits` (and a
single `Observe`) for the entire decision budget (16-34 turns), never
invoking the scenario's load-bearing verb:

| pack                      | level  | seed | n_turns | commands               | verb missing |
|---------------------------|--------|------|---------|------------------------|--------------|
| spec-engineer-capture     | medium |   1  |   34    | MoveUnits×33 Observe×1 | capture_actor |
| spec-engineer-capture     | medium |   2  |   34    | MoveUnits×33 Observe×1 | capture_actor |
| spec-tanya-c4-strike      | easy   |   2  |   17    | MoveUnits×16 Observe×1 | c4_detonate |
| spec-tanya-c4-strike      | medium |   1  |   23    | MoveUnits×23           | c4_detonate |
| spec-thief-steal-cash     | medium |   1  |   34    | MoveUnits×34           | infiltrate |
| spec-thief-steal-cash     | medium |   2  |   34    | MoveUnits×33 Observe×1 | infiltrate |
| spec-thief-steal-cash     | easy   |   1  |   1     | MoveUnits×1            | infiltrate (DRAW: only 1 turn) |

**Classification: model-side failure.** The system prompt + briefing
expose the verb (we verified `Command.capture_actor` /
`Command.c4_detonate` / `Command.infiltrate` are in the tool list and
the agent.py descriptions are explicit). The model "knows" how to
position the unit but doesn't pivot from `move` → `capture/C4/steal`
on arrival. This is the **Action** axis of the P/R/A triage in
PAPER_PLAN.md (cf. existing finding "freeze and panic" — model
defaults to safe verbs under uncertainty).

**Hypothesis to test in the 240-cell run:** larger models
(Qwen3.6-Plus, Kimi-K2.6) may pivot to the verb. If they do, this is
a Qwen3.5-9B-specific scale/capability gap. If they don't, it's a
prompt-engineering gap (verb description in the tool list too terse
for the model to commit).

## F2 (transient) — DRAW after 1 turn

`spec-thief-steal-cash easy seed 1` ended with `outcome=draw` after
ONE turn (`MoveUnits×1`). Inspection needed:
- Did the engine auto-`done` after the thief was placed adjacent to
  silo (engine sees agent has no other combat units → eliminated)?
- Did the model issue `move` to the silo cell and the engine treat
  arrival as a no-op?
- Is this a bench evaluator bug (1-turn-DRAW shouldn't be possible
  given the pack's `within_ticks: 2700` + `after_ticks: 2701` fail
  clause)?

**Classification: ambiguous — needs replay.** Will diagnose post-
collection: `python3 scripts/view_playback.py
/tmp/phase4-multi/20260523-060837__Qwen_Qwen3.5-9B/spec-thief-steal-cash__easy__seed1__vision.jsonl`.

## Engine vs scenario vs model: count

- Engine bugs: 0 in this sample (the smoke completed cleanly; all
  cells produced JSONL + PNG; no `rc != 0`)
- Scenario defects: 1 candidate (the 1-turn DRAW under F2 — could be
  engine OR scenario; need replay)
- Model failures: 6 cells under F1

## Next actions

1. After 240-cell paper-v1 collection completes (in progress, PID
   51416), re-run this triage on the full data.
2. Replay the F2 DRAW case via `view_playback.py` to classify.
3. Compare F1's frequency across the 5 models. If only Qwen3.5-9B
   exhibits it, file as model-specific. If all models hit it, file
   as a prompt-engineering issue + iterate the agent.py tool
   descriptions.
4. After triage on the full collection, extend
   PHASE5_FINDINGS_DRAFT.md → PHASE5_FINDINGS.md (final).
