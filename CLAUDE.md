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
- **`power_surplus_gte` / `power_provided_gte` now work** (historical
  footgun fixed). Pre-placed scenario buildings used to be invisible
  to the player's `PowerManager` trait because only
  `order_place_building` updated it, so the obs reported
  `power_provided = power_drained = 0`. The engine now recomputes the
  totals from the live building actors at snapshot time, honouring
  the `PowerDown` toggle (`World.powered_down`). See
  `OpenRA-Rust/openra-sim/tests/test_power_signals.rs` and
  `tests/test_power_signals_python.py`. The new
  `power_provided_gte` predicate (gross provided, ignores drains) is
  the anti-cheat floor for load-shedding packs — see
  `build-power-down-defensive`.
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
- **`spawn_point` filter is PER OWNER** (Wave-9
  `openra-data/src/oramap.rs::expand_scenario_actors`). Each owner
  (agent / enemy) activates the filter INDEPENDENTLY: if any actor of
  that owner declares `spawn_point`, ONLY that owner's actors whose
  `spawn_point` matches the chosen one are kept; that owner's actors
  WITHOUT `spawn_point` are filtered out. Idioms:
  - Pre-Wave-9 (agent-side axis, every existing pack): declare
    `spawn_point` on agent actors → seed→spawn round-robin varies the
    AGENT corner. Duplicate any persistent base/garrison agent actors
    across BOTH spawn groups at identical coords. Enemy actors don't
    declare `spawn_point` → enemy filter inactive → all enemies
    place every seed (back-compat).
  - Wave-9 (enemy-side axis, e.g. `adv-rps-counter-pick`): declare
    `spawn_point` on enemy actors only → the env's
    `new_with_spawn_point` falls back to
    `distinct_enemy_spawn_points` and round-robins the seed across
    enemy compositions while the agent base stays fixed. Persistent
    per-seed enemy markers (e.g. a far-corner `fact` for engine
    auto-`done` mitigation) MUST be duplicated across every enemy
    spawn group, mirroring the agent-side idiom.
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
- **Stance semantics are now four behaviourally-distinct gates**
  (engine fix, pinned by
  `OpenRA-Rust/openra-sim/tests/test_stance_semantics.rs` +
  `tests/test_stance_semantics_python.py`):
  - `stance:0` HoldFire — never auto-engages, even when attacked.
  - `stance:1` ReturnFire — auto-engages an in-range enemy **only
    after itself taking hostile fire** within a 60-tick window
    (`recently_received_fire` gate). A stance:1 unit next to a
    passive (`stance:0`) enemy holds fire indefinitely — it does
    NOT open fire first. This is the load-bearing distinction the
    `combat-stance-mgmt-attack` / `def-stance-mgmt-hold-then-attack`
    packs exploit.
  - `stance:2` Defend (the default when `stance:` is omitted) —
    auto-fires on the closest in-range enemy but never advances.
  - `stance:3` AttackAnything — auto-fires on in-range enemies AND,
    when none are in weapon range, **advances toward the nearest
    visible enemy** (the "hunt" path: `order_move` toward the
    target, then the next-tick scan promotes the encounter to
    Attack). This is the only stance that opens new engagements by
    moving — a scattered-enemy map can be cleared by a single
    stance:3 hunter chaining one hunt move per kill.
  Explicit agent orders (`attack_unit` / `attack_move`) always
  override stance. A stance flip is a real load-bearing verb:
  `set_stance(units, 3)` converts an idle ReturnFire formation
  into an active hunter.
- **`scheduled_events:` — mid-episode scripted hooks** (Wave-9
  engine feature, pinned by
  `OpenRA-Rust/openra-data/tests/test_scheduled_events.rs` +
  `tests/test_scheduled_events.py`). A scenario may declare a
  top-level (or per-level `overrides:`) `scheduled_events:` list;
  each entry fires once when `world_tick >= tick`. Three event
  kinds:
  - `spawn_actors` — inject new actors mid-episode (reinforcement
    waves). `actors:` is a normal actor list (`count:` expands).
    Spawned actors get fresh ids, so a perception count predicate
    (`enemies_discovered_gte`) treats them as additive. They are
    placed via the same path as initial scenario actors (Mobile/
    Health for units, typed Vehicle/Turret components attached).
  - `destroy_actors` — remove every actor matching `filter:`
    (`owner:` + optional circular `region: {x, y, radius}`).
  - `shorten_deadline` — clamp the episode's `max_ticks` DOWN to
    `new_max_ticks` (never grows it).
  Parsed by `oramap.rs::read_scheduled_events`, fired by
  `env.rs::fire_scheduled_events` after each `process_frame`.
  This is the only way to test information-FRESHNESS perception
  (the scout-cycle idiom) — a hidden enemy placed only at t=0
  cannot exercise "re-observe a stale region". Worked example:
  `scout-cycle-keep-info-fresh`. Footgun: a scenario-declared
  `stance:3` AGENT combat unit auto-hunts the whole map; for a
  perception pack keep the agent's combat arm `stance:0` so a
  stall policy can't win for free by self-delivering the army.
- **`pbox` costs 600** (not the 400 some old specs assumed);
  defense and infantry are SEPARATE production queues so an
  efficient policy queues `build('pbox')` and `build('e1')` in
  parallel from turn 1.
- **Multiple production buildings of the same category produce IN
  PARALLEL** (engine fix, pinned by
  `OpenRA-Rust/openra-sim/tests/test_parallel_production.rs` +
  `tests/test_parallel_production.py`). The production tick advances
  a category's queue once per completed, alive production building
  of that category — two `weap` advance the Vehicle queue twice per
  tick, so two war factories roughly DOUBLE vehicle output (likewise
  `tent`/`barr` → Infantry, `hpad`/`afld` → Aircraft, `spen`/`syrd`
  → Ship). Before the fix the engine modelled production as ONE
  per-player queue per category and a 2nd factory added zero
  throughput. Building / Defense queues (fed by the construction
  yard) keep single-stream semantics. This makes "build a 2nd
  factory to hit a throughput quota" a real load-bearing capability
  — see `build-production-throughput-multibuilding`. NOTE: the 2nd
  factory only helps if there is CASH to feed both queues; a
  cash-starved play (e.g. `econ-buy-vs-build-decision`, where a 2nd
  weap leaves only ~1 tank's worth of residual cash) is still a
  losing CAPEX trap — the fix does not break that pack's bar.
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

## Engine blockers: fix the engine, do not compromise the pack

When authoring a pack you may discover that the intended capability
cannot be expressed in the current engine — e.g. an order is a
no-op, a stance doesn't behave as advertised, or some specific
order/predicate isn't surfaced. (Two historical gaps are now
closed: enemy actors DO honour `spawn_point` per-owner, and
`scheduled_events:` provides the mid-episode scripted-event hook —
see the feature notes above.) **The correct move is to fix the
engine, not to retire the pack, weaken the bar, or substitute a
different mechanic that masks the gap.** Concretely:

1. Reproduce the gap with a focused Rust or Python test.
2. Add the test as a failing test in `OpenRA-Rust/openra-sim/tests/`
   (or `openra-data/tests/`, or `tests/test_<feature>.py` on the
   bench side) and make it pass with the minimum change.
3. Rebuild the wheel:
   `cd OpenRA-Rust && PATH=$HOME/.cargo/bin:/opt/anaconda3/bin:$PATH
   maturin develop --release` (verify the `Installed openra_train`
   line printed — maturin can exit 0 while cargo failed).
4. If the change needs a new predicate or signal, also add the
   `_PHRASES` translation in `openra_bench/game_knowledge.py`
   (the suite test `test_all_predicate_keys_have_a_translation`
   enforces this).
5. Ship the engine change + the pack in **separate commits** on
   the same push so reviewers can see "this engine gap was closed
   to unblock this pack".
6. Update this CLAUDE.md to remove the footgun note (or restate it
   as a now-fixed historical pitfall).

Only retire a pack if the engine fix is genuinely out of scope —
e.g. requires a multi-week refactor or contradicts an explicit
design constraint. In that case open a task with the gap, the
attempted fix, and what would be needed, rather than silently
retiring.

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
