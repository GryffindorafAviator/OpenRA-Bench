# Per-Scenario Closer-Look Checklist

The reproducible methodology applied to #1 (action-multiunit-coordination)
and #2 (action-sequenced-execution). Every remaining active pack gets
the same treatment: **closer look → fix defects → re-verify → run each
difficulty easy→medium→hard on the model → inspect the playback →
commit/push**. Read this fully before touching a pack.

Guiding principle: **the benchmark must validly test its stated
capability; do NOT compensate for model weakness, and do NOT
over-engineer.** A model failing a correctly-designed scenario is a
*good* result (real discrimination). Only fix genuine scenario defects.

---

## A. SOLVENCY — can the intended strategy actually win, in budget?

1. **The win predicate must enforce the advertised capability — and
   only that.** The classic defect: the prose claims X but the win
   condition is satisfiable without doing X.
   - #1: "split the force" but `reach_region` (≥1 unit) let one
     touring unit win → switched to `units_in_region_gte` (≥2 in EACH
     region).
   - #2: "ordered route" but only the *final* region was a predicate →
     a beeline that skipped every waypoint won → added the stateful
     `waypoint_sequence` latch (Wk+1 only counts after Wk; skip /
     out-of-order / idle ⇒ never satisfied).
   - Ask: "what is the laziest play that satisfies this win condition?"
     If it isn't the intended capability, the predicate is wrong.
2. **Is the optimal/intended play winnable within the tick budget?**
   Estimate path length; engine advances **~90 ticks per decision
   turn** (`tick ≈ 93 + 90·(turn-1)`). The intended strategy must
   finish comfortably under `within_ticks`; the *inefficient* strategy
   must NOT (that gap is the discrimination).
3. **Coordinate-blind objectives must be solvable.** If
   `objective_coords: relative`, the model can't cell-count — it needs
   a feedback loop (an `enemy_building_spotted` interrupt revealing the
   marker, a landmark building at each region) and a forgiving enough
   radius. A "search band" beats a bare compass word.
4. **Map fits the actors.** Every actor/region coordinate must be
   inside the map's playable bounds. Actors at scout-arena coords on a
   rush-hour map → engine panic. Confirm `compiled.map_supported` and
   that `base_map` resolves to the *intended* map.

## B. STABILITY — deterministic, no crashes, fail is reachable

5. **Non-win must be a real LOSS, not a draw.** Every level needs a
   `fail_condition`. Idiom: `any_of[ {after_ticks: BUDGET+1},
   {not:{units_lost_lte: N}}, {not:{own_units_gte: 1}} ]`.
6. **Tick/turn alignment (critical, easy to get wrong).** A
   `fail after_ticks: K` only bites if K is reachable within
   `max_turns`: require `K ≤ 93 + 90·(max_turns-1)`. Likewise the
   episode must be able to reach `within_ticks` before `max_turns`
   ends, or a staller draws instead of losing. Re-derive this for
   every level after any budget/turn change.
7. **`base_map` override goes INSIDE `overrides:`** — a Level-level
   `base_map` key is silently ignored (it's not a `Level` field).
8. **Smoke the engine path before a full run**: compile the level,
   `_scenario_to_tmp_yaml`, `RustEnvPool` reset(seed) + a few
   `env.step`; if the pack enables interrupts, also drive
   `raw_env.step_until_event([...],None,5,[sig])` ~30 steps — catches
   panics/oob without burning a model run.
9. **Hard-tier spawn contract**: if the pack is in
   `tests/test_hard_tier.py::UPGRADED`, `hard` must keep ≥2
   `spawn_point` groups (seed-varied start). A deliberate exception is
   allowed only with a stated reason added to `NOT_APPLICABLE`.

## C. CAPABILITY — clean difficulty axis, faithful framing

10. **One new controlled variable per tier** so a tier failure
    attributes to a single capability. easy = the bare skill; medium =
    +1 axis (a third region / parallel split / contest / attrition);
    hard = +1 more (relative coords + scouting, larger map, strict
    budget). Avoid stacking uncontrolled variables.
11. **Keep established idioms** (e.g. the single-final-region +
    `[after_ticks, within_ticks]` band for "execute, don't stall";
    fact+proc key-building destruction for adversarial). Don't invent a
    new mechanism when one exists — but DO add a reusable predicate /
    engine feature when the capability genuinely can't be expressed
    (`units_in_region_gte`, `waypoint_sequence`, `enemy_building_
    spotted` interrupt, scripted `enemy.bot`).
12. **Title/description are plain and self-explanatory**; the
    objective brief the model sees (`game_knowledge.objective_brief`)
    must state the exact machine win/fail in plain language with no
    degenerate lines.

## D. RUN & INSPECT (one difficulty at a time)

13. Run on **`qwen/qwen3.6-flash`** via OpenRouter (key in `./.env`,
    git-ignored): `python3 -m openra_bench.run_eval --packs
    openra_bench/scenarios/packs/<pack>.yaml --levels <lvl> --seeds 1
    --provider openrouter --model qwen/qwen3.6-flash --playback
    playback/run1`. easy, then medium, then hard.
14. **Inspect the playback** (`playback/run1/*/<pack>:<lvl>:public/
    seed1/`): `manifest.json` (outcome/turns), `score.json`
    (composite/weakest_link/speed), `turns.jsonl` (per-turn `units`,
    `enemies`, `signals`, `interrupt`, `goal`), `messages.json` (model
    text is in the **`reasoning`** field, not `content`). Reconstruct:
    did the intended mechanism fire? final positions vs win regions?
    `units_lost`? terminal "episode end" frame present?
15. **Classify the outcome**: scenario defect (→ fix, re-verify, re-run)
    vs legitimate model failure (→ record as valid discrimination, no
    change). Cite evidence from the playback.

## E. RE-VERIFY & SHIP

16. `python3 -m pytest tests/ -q` fully green (add/extend focused
    tests for any new predicate/knob/scenario behaviour).
17. `python3 scripts/gen_scenario_docs.py` (regenerate the HTML
    catalog) when prose/objectives change.
18. Commit per fixed scenario, **no Claude co-author line**, using
    `git -c commit.gpgsign=false commit`. Then
    `git fetch -q origin && git rebase -q origin/main && git push -q
    origin HEAD`. Engine changes (OpenRA-Rust) commit+push separately;
    rebuild the wheel (`maturin develop --release`) if the engine
    changed and re-run the affected scenario.
19. Record a "Per-scenario closer look — #N" note in
    `SCENARIO_QUALITY.md` (the defect found, the fix, the
    easy/medium/hard outcomes + verdict).

## Reference: defect patterns already seen

- Win condition doesn't enforce the stated capability (laziest play
  wins). — #1, #2
- `reach_region`/single-region where a *split* or *ordered* visit is
  intended. — #1, #2
- Missing `fail_condition` ⇒ non-win == draw. — #1, #2
- `after_ticks` fail unreachable within `max_turns` (tick/turn
  misalignment). — #2
- Relative objective too literal ("NE corner" → map extreme), region
  inset & unreachable blind, landmark fogged ⇒ unfair. — #1 hard
- `base_map` at Level level silently ignored. — #2 hard
- Actors placed off the resolved map ⇒ engine panic in the interrupt
  path. — #2 hard
- Bench advertises a tool the engine can't execute (1:1 parity). —
  capture_actor / S8
- Inert deadline (`within_ticks` ≫ optimal) ⇒ no anti-stall teeth. —
  #1 easy, #2 easy (acceptable for *easy* only).
