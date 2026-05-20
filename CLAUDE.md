# Working in OpenRA-Bench (Claude Code / Codex / any coding-agent guide)

OpenRA-Bench is a rigorous LLM-agent RTS benchmark on a Rust engine. If
you are an AI coding agent creating, validating, fixing, or extending a
scenario pack, read this once and treat the linked docs as binding.

## The bar (apply to every scenario you touch)

> **No defect. No cheat.**
> Every lazy / brute / stall / blind / shortest-path / wrong-route /
> spam policy must **LOSE** on every level and every hard seed (1–4).
> The intended capability policy must **WIN**. Non-win must be a real
> reachable timeout **LOSS** — never a draw.

A scenario is defective if any of the following hold:

1. The win predicate is satisfiable by a play that ignores the
   advertised capability (the "laziest play wins" inversion).
2. `within_ticks` or `after_ticks` is set above the tick reachable
   within `max_turns` (engine advances ~90 ticks per decision turn ⇒
   `tick ≤ 93 + 90·(max_turns − 1)`); the deadline never bites ⇒ the
   episode times out as a **DRAW**, not a LOSS.
3. There is no `fail_condition`, or it only triggers on full
   force-wipe; a stall / preserve / partial outcome silently draws.
4. The intended capability is not solvable inside the declared budget
   (a scenario nobody can win is also defective).
5. The engine auto-terminates on enemy-elimination before the win/fail
   is evaluated (mitigation: place an unarmed high-HP enemy `fact`
   marker at the objective).
6. Actors are placed outside the map's playable bounds (engine
   panics).
7. The pack is `UPGRADED` in `tests/test_hard_tier.py` but its hard
   tier does not produce ≥2 distinct seed-driven spawns (or there is
   no documented `NOT_APPLICABLE` reason).

## Authoritative docs (read these, in order)

- **`SCENARIO_REVIEW_CHECKLIST.md`** — the closer-look methodology
  (A solvency / B stability / C capability) you follow step by step
  to create or validate a pack.
- **`SCENARIO_QUALITY.md`** — the whole-suite no-cheat pass summary,
  the recurring defect classes the pass eliminated, and the
  predicate-idiom recipe (which predicate makes which capability
  load-bearing), plus engine footguns to avoid.
- **`openra_bench/scenarios/win_conditions.py`** — the predicate
  grammar. If you add a new predicate, you **must** also add a
  `_PHRASES` / `_REGION_PHRASES` translation in
  `openra_bench/game_knowledge.py` (the suite test
  `test_all_predicate_keys_have_a_translation` enforces this).
- **`openra_bench/botgen.py`** + `openra-sim/src/scripted_bot.rs` —
  the scripted opponents (`hunt | rusher | patrol | turtle | guard`)
  declared per-pack as `enemy: {bot_type: <name>}`. `guard` is the
  leashed defender used by the bait/decoy/lure idioms.
- **The 21 no-cheat-redesign commits on `main`** are worked examples
  of every capability/predicate/bot combination. Browse with
  `git log --oneline --grep "no-cheat redesign"` and read the bodies.

## Engine facts you must internalise

- **Ticks/turn:** ~90. Max tick at `max_turns` ≈ `93 + 90·(max_turns
  − 1)`. Any `within_ticks` / `after_ticks` above this is **inert**
  (won't bite) ⇒ draw degeneracy.
- **Engine auto-done:** the engine sets `done=True` when all enemy
  actors are eliminated, or sometimes when an agent unit reaches an
  enemy-key location. Without a persistent enemy actor a win-by-reach
  scenario can end as DRAW. Put an unarmed high-HP enemy `fact`
  marker at the objective.
- **Own-unit `actor_type`** surfaces in `units_summary`
  (`unit_type_count_eq / _gte` work). Predicates relying on it are
  valid.
- **`power_surplus_gte` is currently inert** (obs reports
  `power_provided / power_drained = 0`). Do **not** rely on it as a
  sole discriminator.
- **`deploy` now works** for scenario-declared MCVs (the historical
  "unimplemented" footgun was a two-bug interaction: `classify_actor`
  in `openra-sim/src/gamerules.rs` returned `Vehicle` for MCV, and
  the env.rs `kind_for_unit_type` fallback defaulted to `Infantry`.
  Both fixed; see `tests/test_mcv_deploy.py`.) Scenario actor
  `{type: mcv}` + `Command.deploy([mcv_id])` removes the MCV,
  creates an agent `fact`, and re-enables Building/Defense
  production queues — so a build-radius scenario can launch from a
  single starter MCV.
- **Scripted bot `guard`:** holds its post (`spawn_cell`), auto-fires
  in range, lunges at the nearest foe within `GUARD_AGGRO ≈ 16`,
  snaps back past `GUARD_LEASH ≈ 18` — the bait-able-defender idiom
  proven in #4 / #6 / #7 / #15 / #18.
- **`spawn_point` filter applies ONLY to AGENT actors** — enemy
  actors with no `spawn_point` ALWAYS place, regardless of the chosen
  group (`openra-data/src/oramap.rs::expand_scenario_actors`). You
  cannot vary enemy count/composition by seed via `spawn_point`;
  vary the agent's spawn instead and design symmetric enemy
  placement. If ANY agent actor declares `spawn_point`, every agent
  actor WITHOUT `spawn_point` is filtered OUT — so duplicate
  base/garrison actors across BOTH spawn groups at identical coords.
- **`silo` is NOT MustBeDestroyed** — using it as an objective
  landmark allows premature engine auto-`done` when the *other*
  MustBeDestroyed buildings fall. Use `barr` / `proc` / `powr` /
  `fact` for landmark anchors. (Wall-as-obstacle role is fine.)
- **`after_ticks` in a WIN clause is structurally incompatible with
  ConquestVictoryConditions** — the engine auto-`done`s the second
  the last enemy `MustBeDestroyed` building falls, before the
  `after_ticks` window opens, collapsing the run to DRAW. `after_ticks`
  belongs in `fail_condition`. Encode timed-arrival semantics via
  distance/landmark positioning instead.
- **`move_units` auto-fires opportunistically en route** regardless
  of agent stance (even `stance:0` HoldFire). For perception packs
  with hidden enemies that must be discovered without combat, set
  the HIDDEN actors to `stance:0` themselves (defender side, not
  scout side).
- **`pbox` costs 600** (not the 400 some old specs assumed);
  defense and infantry are SEPARATE production queues so an
  efficient policy queues `build('pbox')` and `build('e1')` in
  parallel from turn 1.
- **`place_building` does NOT enforce build-adjacency** — orders
  work at arbitrary in-bounds coords. Forward-base / far-region
  building is solvable with a single `build + place_building`.
- **`fact` has cost 0** → not buildable via `StartProduction`
  (engine gates on `cost > 0`). Use `proc` as the "second base seed"
  in expand-arm objectives.
- **`not own_units_gte:1`** mis-fires on turn 1 when the agent
  starts unit-less (documented footgun from `economy-force-buildup`).
  Use `after_ticks` + `not has_building:fact` for the unit-less
  start fail clause instead.
- **Certain mid-map cells silently fail to place enemy clusters**
  (e.g. `(50,20)`, `(60,28)`, `(90,30)` observed by A7); nearby
  cells (`(60,10)`, `(100,30)`, `(50,19)/(50,21)`) work. Likewise
  `e1` at some cells doesn't surface in `enemy_positions` — `e3`
  does. For perception packs, use `e3` for hidden clusters and
  verify cluster cells on a smoke run before authoring against them.

## How to validate (deterministic, no model / no network)

For each level + each hard seed (1–4) run scripted policies via
`openra_bench.eval_core.run_level`:

```python
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.eval_core import run_level

c = compile_level(load_pack(PACKS_DIR / "<pack>.yaml"), "easy")
res = run_level(c, my_policy_fn, seed=1)
print(res.outcome, res.turns, res.signals.units_lost)
```

Cover at minimum: `stall` (only `Command.observe()`), a `brute`
beeline to the objective, a `greedy` / `wrong-path` if applicable,
and the **intended** capability policy. The bar (above) must hold.
**No model / OpenRouter / network runs are needed for validation** —
scripted policies are sufficient and free.

## Working on `main` (PRs vs direct push)

- The default branch is `main`.
- Direct pushes to `main` are reserved for the user's batch
  parallel-agent workflow. If you are a one-off agent invoked
  outside that flow, **branch first** and open a PR; do not push to
  `main` without explicit user authorization.
- Commits must NOT include a Claude / AI co-author line.
- The shared engine wheel is rebuilt via
  `cd OpenRA-Rust && PATH=$HOME/.cargo/bin:/opt/anaconda3/bin:$PATH
  maturin develop --release` — **verify the `Installed openra_train`
  line actually printed** (maturin can exit 0 while cargo failed).

## Don'ts (lessons from the cadence)

- Don't add a predicate to `win_conditions.py` without a
  `_PHRASES` translation in `game_knowledge.py`.
- Don't `git add -A` / `git commit -a` — concurrent agents may have
  uncommitted edits to shared files; stage only your own files.
- Don't compensate for model weakness or over-engineer; only fix
  real scenario defects, and keep the established idioms.
- Don't edit `SCENARIO_QUALITY.md` / `docs/scenarios.html` in a
  per-scenario commit — the main session regenerates them at the end.
- Don't edit `OpenRA-Rust/` (the engine) inside a scenario task —
  flag engine needs in your report instead.
