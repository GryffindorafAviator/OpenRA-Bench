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
   within `max_turns`; the deadline never bites ⇒ the episode times
   out as a **DRAW**, not a LOSS. **Reachable max tick = `93 + 90·
   (max_turns − 1)`** for non-interrupt mode; interrupt-mode packs
   advance 1-5 ticks per turn (variable). See the full "Ticks/turn"
   note in "Engine facts you must internalise" below for the
   complete formula + common pitfall.
3. There is no `fail_condition`, or it only triggers on full
   force-wipe; a stall / preserve / partial outcome silently draws.
4. The intended capability is not solvable inside the declared budget
   (a scenario nobody can win is also defective).
5. The engine auto-terminates on enemy-elimination before the win/fail
   is evaluated (mitigation: place an unarmed high-HP enemy `fact`
   marker at the objective, OR declare
   `termination.enemy_units_killed: false` to disable the auto-`done`
   for that side; see the engine-fact note below). The mirror flag
   `termination.agent_units_killed: false` keeps the run alive past an
   agent-side wipe — load-bearing for forlorn-hope / suicide-charge
   packs where the within_ticks fail clause must fire AFTER the strike
   package dies.
6. Actors are placed outside the map's playable bounds (engine
   panics).
7. The pack is `UPGRADED` in `tests/test_hard_tier.py` but its hard
   tier does not produce ≥2 distinct seed-driven spawns (or there is
   no documented `NOT_APPLICABLE` reason).
8. **A pre-placed `harv` adjacent to a `proc` auto-routes to nearby
   ore from t=0** (engine `auto_route_idle_harvesters` in
   `OpenRA-Rust/openra-sim/src/world.rs:2233`). This is intentional
   game behaviour, NOT an engine bug. The consequence for pack
   tuning: any `economy_value_gte: N` bar must sit ABOVE the
   unattended auto-harvest ceiling at the pack's deadline; otherwise
   a stall (observe-only) policy WINS by free baseline income, and
   the no-cheat bar is broken. Measure the ceiling with a temporary
   bar of 100000 (so the episode runs to the deadline rather than
   triggering early), then set the real bar comfortably above it.
   Three packs were silently violating this until 2026-05-24:
   `economy-harvest-timebox` easy, `rob-cash-depletion-recovery`
   easy, both since fixed. See `tools/validate_pack_bar.py` (below)
   for the audit script that surfaces this defect class.

## Authoritative docs (read these, in order)

- **`SCENARIO_REVIEW_CHECKLIST.md`** — the closer-look methodology
  (A solvency / B stability / C capability) you follow step by step
  to create or validate a pack.
- **`SCENARIO_QUALITY.md`** — the whole-suite no-cheat pass summary,
  the recurring defect classes the pass eliminated, and the
  predicate-idiom recipe (which predicate makes which capability
  load-bearing), plus engine footguns to avoid.
- **`audits/EDIT_PRINCIPLES.md`** — Family-1 (combat micro) edit
  conventions (§1-10). BINDING for every pack edit: self-contained
  briefings, three-part officer-style structure, plain English with
  RA terms inline-glossed, `map_fit` classification (`fit` / `wide`
  / `large-trivial`), enemy-posture matches briefing doctrine, and
  the §10 map-resizing rule (every `wide` / `large-trivial` pack
  must be shrunk to a bespoke procedural arena, ≤15 cells empty
  pre-engagement traversal).
- **`audits/EDIT_PRINCIPLES_FAMILY2.md`** — Family-2 (economy)
  additions (§11-17): functional harv→ore→proc→cash chain,
  cash-constrained multi-path, wrong-strategy LOSS trap, refinery-
  placement spatial reasoning, stall-LOSES (no auto-harvest
  preemption), `not: {cash_gte: MAX+1}` upper-bound idiom, build-
  prereq cross-check. INHERITS Family-1 §1-10 verbatim — including
  the §10 map-shrink rule, which 16 F2 packs slipped through on
  `rush-hour-arena`; do not let this happen on any future F2 pack.
- **`audits/PRODUCTION_TECH_AUDIT.md`** — construction / unit
  production / tech-tree correctness audit (cross-family). Every
  pack with a `build` or `deploy` capability must satisfy: prereq
  buildings present, faction match, affordability from
  `starting_cash + projected_income`, build-time fits `max_turns`.
  Cross-reference `audits/production_tech_audit.csv` before
  editing; if the row's `issues` field is non-empty, FIX as part of
  the edit.
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
- **`tools/validate_pack_bar.py`** — sweep-runs `observe()`-only
  (stall) against every active pack × level × seed 1–4 and emits
  `audits/pack_bar_status.csv`. Any cell where stall WINS is a
  no-cheat-bar break (cheat policies must LOSE per the bar). Run
  before merging any pack edit:
  ```
  python3 tools/validate_pack_bar.py
  python3 tools/analyze_pack_bar.py audits/pack_bar_status.csv
  ```
  Catches the auto-harvest-baseline defect class (above) and any
  win predicate that's structurally satisfiable without agent
  action (e.g. `within_ticks:N` with no objective predicate).
  The intended-must-WIN half of the bar is still verified by each
  pack's `tests/test_*.py` — the validator is the missing piece
  for the cheats-must-LOSE half.

## Audit-doc workflow (binding for every pack edit)

Before editing any scenario pack:

1. Identify the pack's family (combat / economy / build / defense /
   etc.) and whether its intended play requires `build` / `deploy`.
2. Read the matching `EDIT_PRINCIPLES*.md` doc.
3. Look up the pack's row in `audits/family<N>_*.csv` if present;
   the `map_fit` field tells you whether a map shrink is needed.
4. If the pack has a `build` or `deploy` axis, look up the row in
   `audits/production_tech_audit.csv`; the `issues` field tells you
   whether prereqs / cash / build-time are broken.
5. Apply all flagged fixes in the same edit. Do NOT ship a pack
   edit that leaves a CSV-flagged issue unresolved.
6. If your edit changes a tech-tree-relevant field (new buildings
   added, faction flipped, starting_cash changed), update the
   CSV row.
7. **If your edit changes a pack's `map_size` / `base_map` /
   `starting_cash` / actor list, re-run the matching
   `audits/family<N>_*_build.py` AND
   `audits/production_tech_audit_build.py` in the same commit and
   verify the regenerated CSV row matches your edit.** (Without this
   step the audit doc points at a world that no longer exists — a
   future agent following step 3 reads a stale `map_fit` /
   `afford_at_start` row and may re-do work or lose trust in the
   binding. P0.1 in the PR #30 review documented exactly this
   failure mode.)

## Engine facts you must internalise

- **Embedded RA YAML is the SINGLE source of unit data.** Originally
  seeded from upstream OpenRA at SHA `0938a27` (bleed branch), the RA
  rules / weapons YAML now lives in-repo at
  `OpenRA-Rust/openra-data/src/embedded/{rules,weapons}/*.yaml` and is
  parsed at runtime via `include_str!`. The runtime vendor-checkout
  dependency was removed by the engine refactor that introduced this
  section — history: an earlier `GameRules::defaults()` stub was
  removed in PR #15 making the vendor checkout strict-required, then
  the vendor-checkout dependency itself was removed when the YAML
  files were embedded into the binary. See
  `OpenRA-Rust/VENDOR_DATA.md` for the snapshot / bump procedure.
  All actor / weapon stats — HP, cost, speed, footprint, weapons,
  prereqs — flow through `GameRules::from_ruleset` exactly as before,
  reached via:
  - `GameRules::from_vendor()` — fresh parse of the embedded data
    (no filesystem). Can't fail unless an explicit `OPENRA_VENDOR_DIR`
    override is broken.
  - `GameRules::vendor_cached()` — `OnceLock`-cached clone, suitable
    for tests that spin up many worlds (parity sweeps, etc.).
  - `openra-train/src/env.rs::load_rules_strict()` — runtime entry
    point for the bench wheel. Reads the embedded data by default;
    honours `OPENRA_VENDOR_DIR` as an opt-in override for power users
    testing against an alternate snapshot.
  - `openra-sim/src/world.rs::build_world(map, ..., None, ...)` — the
    `None` fallback hits `vendor_cached()`, so every test that passes
    `None` for rules transparently inherits the embedded RA truth.
  **Footguns this closes**: the stub used to drift from vendor — `fact`
  footprint was 3×2 but vendor is 3×4; pillbox/tanya weapons were
  hand-coded stubs (`AAStub`, `TanyaPistol`) rather than the real
  vendor armaments (`Nike`, `Colt45`). Every pack edit now sees
  vendor numbers without exception. **Footguns this exposes** (carried
  in this CLAUDE.md): a handful of legacy tests pinned to stub HP /
  weapon names (sync_hash_verify recorded sync hashes against stub
  footprints; SAM Nike missile projectile path isn't fully simulated)
  are marked `#[ignore]` with FIXMEs pointing here. If you author a
  new pack and discover an actor stat that surprises you, read the
  vendor YAML — not the audit CSVs from before PR #15 — for the
  truth.
- **Ticks/turn — the empirical numbers** (confirmed by 3+ audit
  passes and 12+ in-tree test files):
  - **Non-interrupt mode**: **~90 ticks per decision turn**. Reachable
    max tick = **`93 + 90·(max_turns − 1)`** (≈ `90·max_turns`).
    This is the formula the test suite enforces — every pack ships
    a `test_timeout_loss_is_reachable` test using exactly
    `93 + 90 * (max_turns − 1)` (`grep -lE '93 \+ 90 \*' tests/`).
  - **Interrupt mode** (any pack with a non-empty `interrupts:`
    block, dispatching through `step_until_event`): **1-5 ticks per
    decision turn**, breaking on the first interrupt signal. The
    bench's call site (`openra_bench/eval_core.py:348`) passes
    `max_ticks=5` explicitly, so per-turn tick advance is variable
    in `[1, 5]`. Read `info["ticks_advanced"]` rather than computing
    arithmetically. **Reachable max tick ≤ `5·max_turns`** in the
    worst case (no early signal), but the agent typically sees far
    fewer because the interrupt fires.
  - **Common pitfall** (the reason this section is long): the engine
    constant `DEFAULT_TICKS_PER_STEP = 30` in
    `openra-train/src/env.rs:33` is NOT the per-decision-turn value.
    The bench advances frames 3× per `env.step()` (search
    `process_frame` in `env.rs`), giving the ~90 figure above.
    Don't quote the `30` figure — it's an internal constant, not
    the budget every audit needs.
  - **Defect rule**: any `within_ticks` / `after_ticks` above the
    reachable tick is **inert** (won't bite) ⇒ DRAW degeneracy
    (CLAUDE.md "no defect, no cheat" criterion #2).
  - **`termination.max_ticks` is honoured exactly** (historical
    footgun fixed). The engine used to ignore the scenario YAML
    `termination.max_ticks` field and always use the constant
    `DEFAULT_MAX_TICKS = 10000`, effectively capping every scenario
    at `max_turns ≤ 110`. The parser now surfaces the field on
    `MapDef.max_ticks`, and `Env::new_with_spawn_point` applies the
    value EXACTLY — no clamp. Long-horizon packs (F11
    vertical-strike with `max_turns 140-180` ⇒ reachable max tick
    ≈ 16203) declare any budget their capability requires (e.g.
    `termination: {max_ticks: 16500}`). When the YAML omits the
    field the env falls back to `DEFAULT_MAX_TICKS = 10000` for
    back-compat. Pinned by
    `openra-data/tests/test_scenario_termination_parse.rs` (parser
    coverage) and
    `openra-train/tests/test_max_ticks_unbounded.rs` (end-to-end:
    the engine actually runs past tick 10000 under a declared
    higher budget).
- **Engine auto-done:** the engine sets `done=True` when all enemy
  actors are eliminated, or sometimes when an agent unit reaches an
  enemy-key location. Without a persistent enemy actor a win-by-reach
  scenario can end as DRAW. Put an unarmed high-HP enemy `fact`
  marker at the objective, OR set
  `termination.enemy_units_killed: false` (see next note).
- **`termination.{agent,enemy}_units_killed` are now load-bearing
  flags** (engine fix, pinned by
  `OpenRA-Rust/openra-data/tests/test_scenario_termination_parse.rs`,
  `OpenRA-Rust/openra-train/tests/env_termination_flags.rs`, and
  `tests/test_termination_flags_python.py`). Historical footgun: the
  flags were documented in `TerminationConfig` and on several pack
  headers (`combat-suicide-charge-mission`, `strategy-dilemma`,
  `strategy-twobody`, `rush-hour`, `mid-concede-vs-hold`,
  `econ-deny-enemy-expansion`, `coord-diversionary-attack`) but had
  no consumer — `oramap::parse_scenario_yaml` silently dropped the
  `termination:` block, so the engine's `is_terminal` check always
  auto-`done`d on either side's wipe. That collapsed sacrifice /
  decoy / forlorn-hope packs to DRAW the moment the agent's last
  unit died (the `within_ticks` fail clause never had a chance to
  fire). Triaged in `audits/qwen9b_draws_and_medium_dip.md` (the
  5-cell `combat-suicide-charge-mission` DRAW cluster). The parser
  now reads the `termination:` block; the two gating flags surface on
  `MapDef` as `terminate_on_{agent,enemy}_units_killed` (default
  `true` ⇒ legacy behaviour) and `Env::is_terminal` gates the
  corresponding wipe path behind them. A scenario opts out by
  declaring the flag `false`:
    * `termination.agent_units_killed: false` — keep the run alive
      past an agent-side wipe (sacrifice anchor: lose the strike
      package without ending the run, so a within_ticks fail clause
      can score the post-wipe arrival).
    * `termination.enemy_units_killed: false` — keep the run alive
      past an enemy-side wipe (the proper fix instead of the
      far-corner `fact` marker workaround for the auto-done race).
  A surrendered agent is still defeated regardless of the flag
  (`Command::Surrender` is an explicit player action, not the
  units-killed auto-done).
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
- **`move_units` auto-fires opportunistically en route, and a
  moving unit is a normal target** (engine fix, pinned by
  `OpenRA-Rust/openra-sim/tests/test_moving_unit_takes_fire.rs`).
  A unit executing a `Move` activity now shoots in-range hostiles in
  passing WITHOUT abandoning its move, and is itself hittable by
  in-range enemies — there is no "sprint-invincibility" any more (a
  unit can no longer cross a kill zone untouched on a long move
  order). The opportunistic move-fire RESPECTS stance: `stance:0`
  HoldFire never fires while moving; `stance:1` ReturnFire fires
  only after taking recent hostile fire; `stance:2`/`stance:3` fire
  on the nearest in-range enemy. For perception packs with hidden
  enemies that must be discovered without combat, set the HIDDEN
  actors to `stance:0` themselves (defender side, not scout side).
- **`attack_unit` on an out-of-sight target paths normally** (engine
  fix, pinned by
  `OpenRA-Rust/openra-sim/tests/test_attack_unit_no_teleport.rs`).
  An explicit `Attack` order against an enemy beyond weapon range
  now closes distance at the attacker's real `Mobile` speed
  (identical to a plain `move`); the chase used to warp a full cell
  per tick (~24x for infantry), teleporting the unit dozens of
  cells in one decision frame.
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
- **`reveal_map:` — the no-fog perception cell** (pinned by
  `OpenRA-Rust/openra-data/tests/test_reveal_map.rs` +
  `tests/test_perception_ablation.py`). A scenario may declare a
  top-level `reveal_map: true`; the agent player then observes the
  whole map with NO fog of war — every enemy actor is reported
  regardless of shroud and `explored_percent` is 100. Parsed by
  `oramap.rs::parse_scenario_yaml` (mirrors `spawn_mcvs`), applied
  in `env.rs` (`is_visible_to` short-circuits true for the agent;
  `refresh_explored_cells` fills the playable rectangle). This is
  the no-fog half of the **perception ablation grid**. The bench
  `fog_mode` spans 3 observation channels × 2 fog states = 6 cells
  (`openra_bench/scenarios/schema.py::PERCEPTION_MODES`):
  - `structured` — text briefing + a text 'Unexplored regions'
    block; no image.
  - `vision` — text briefing + PNG minimap. NOTE the briefing
    already enumerates units/enemies with coordinates, so the image
    is a redundant SUPPLEMENT — `vision` does NOT isolate
    image-reading.
  - `image` — image-PRIMARY: `prompt_v2.briefing_image_primary`
    redacts every coordinate from the text; the labelled minimap
    (`render_tactical_minimap(..., unit_labels=...)`) is the sole
    spatial source. The model references units by the on-map
    handle (`tank-1`, `enemy-2`); `agent._to_commands` maps the
    handle back to the engine id. The clean "can the model read a
    minimap" probe.
  Each channel has a fogged form and a `-clear` (no-fog) form. A
  `-clear` cell is a perfect-information CONTROL that isolates the
  perception cost — a stall/observe policy WINS a `-clear`
  perception pack (perception removed), so the no-cheat bar
  applies ONLY to the fogged cells, never the `-clear` ones. Run
  the full grid with `run_eval --perception-sweep` (expands every
  `pack:level` into `pack:level:<mode>`); the human Play tab stays
  on the canonical `vision` (fogged) modality.
- **Handoff ablation** (`openra_bench/handoff.py`, `run_eval
  --handoff-sweep`). A `HandoffController` lets a `prefix` controller
  play the first K turns, then the model inherits the live game state
  ("take over from here" — a pure STATE handoff, no transcript). A
  `stall` prefix hands the model a losing position (recovery test); a
  replayed winning trajectory (`TrajectoryController`, sourced from a
  `--handoff-bank` of Playback runs) hands it a winning one
  (capitalize-on-advantage). Sweep cells are
  `pack:level:handoff-{base,bad,good}`. Every result carries a
  `passivity` stat — the fraction of the model's turns spent on
  `observe`/`stop` only — the freeze-and-panic signal. A replayed
  trajectory MUST come from the same `pack:level:seed` (engine actor
  ids are seed-deterministic).
- **`pbox` costs 600** in the bench engine (verified against vendor
  RA YAML loaded by `env.rs::load_rules_with_fallback`; note the
  `gamerules.rs::defaults()` stub says 400 but the runtime value is
  600 — see `audits/engine_unit_audit.csv`); defense and infantry
  are SEPARATE production queues so an efficient policy queues
  `build('pbox')` and `build('e1')` in parallel from turn 1.
- **`pbox` is now an active direct-fire tower** (engine fix,
  pinned by `OpenRA-Rust/openra-sim/tests/test_pbox_fires.rs` +
  `tests/test_pbox_fires.py`). RA's `pbox` is an
  `AttackGarrisoned` defense — in C# its fire comes from infantry
  loaded into its `Cargo`, so the YAML carries NO direct
  `Armament` trait. The engine does not model garrisoning, so a
  BUILT `pbox` used to stand inert (the auto-target loop's
  `weapons.first()` returned `None`). `GameRules::from_ruleset`
  now assigns the canonical RA anti-infantry pillbox weapon
  `M60mg` (Damage 1000 × Burst 5, ReloadDelay 30, Range 4c0,
  anti-infantry `Versus None:150` — a burst one-shots an `e1`) to
  garrison-only ground-turret defenses (`pbox`, `hbox`) when they
  carry no explicit `Armament`. `M60mg` is weaker and shorter-
  ranged than the `gun` turret's `TurretGun` (Damage 6000, Range
  6c512), matching the pbox's role as the cheap anti-infantry
  pillbox. Defense packs can now make the pbox load-bearing via a
  `units_killed_gte` clause (a built pbox is the kill source). The
  `def-walls-vs-towers` idiom — a `scheduled_events: spawn_actors`
  rush injected AFTER the defense has time to build serially, with
  NO pre-placed agent combat screen — is the way to make a
  build-pbox policy genuinely WIN via pbox kills while stall /
  wrong-placement still LOSE.
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
- **`scout` is NOT a tool / `Command` verb** — it is a NOUN used in
  pack briefings and pack names (`scout-and-report`,
  `mid-tech-switch-on-scout`, …) to refer to a scout-role UNIT
  (typically `jeep`) and the SCOUTING capability (move into the
  fog to reveal hidden actors / buildings). The actual verb is
  `move_units` (drive the jeep into the unexplored region); the
  predicate that gates the WIN is `buildings_discovered_gte` /
  `enemies_discovered_gte` (snapshot of the agent's fog-of-war
  scan, populated by the engine's `is_visible_to` + the
  adapter's `enemies_seen_ids`). No pack currently lists `scout`
  in `tools:` (verified by a YAML scan), and there is no
  `Command::Scout` variant in `openra-train/src/env.rs`. The F9
  audit's "scout tool semantics unclear" caveat was a
  false-positive on the NOUN; nothing to change.
- **`fact` has cost 0** → not buildable via `StartProduction`
  (engine gates on `cost > 0`). Use `proc` as the "second base seed"
  in expand-arm objectives.
- **`health:` on a pre-placed actor NOW WORKS** (historical footgun
  fixed). The Rust scenario parser
  (`oramap.rs::RawScenarioActor`/`ScenarioActor`) used to parse only
  `actor_type / owner / position / count / spawn_point / stance` and
  silently dropped a `health:` line, so an actor placed with
  `health: 40` spawned at full HP. The parser now carries
  `health: N` (an HP PERCENTAGE, 1-100) through to the spawned
  actor's `Health` trait (`hp = max_hp * N / 100`, clamped ≥1).
  Pre-placed damaged buildings/units are the basis for the
  repair-triage / disaster-recovery idiom. Pinned by
  `OpenRA-Rust/openra-data/tests/test_actor_health.rs` +
  `tests/test_actor_health_field.py`.
- **Building actor ids ARE surfaced for `repair` / `sell` /
  `power_down` / `set_primary`** (historical footgun fixed).
  `RustObsAdapter.render_state()` used to build `own_buildings` as
  `{type, cell_x, cell_y}` — it dropped the engine actor id, so
  `prompt_v2` assigned `id = list-index` and `env.rs::resolve_owned`
  rejected the bogus id ⇒ no agent could target a building. The
  adapter now includes the REAL engine `id` (plus `hp` / `is_primary`)
  in `own_buildings` / `buildings_summary`, mirroring how
  `units_summary` keeps the real unit id. Pinned by
  `tests/test_repair_building_id.py`.
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
- **`place_building('proc')` now auto-spawns the new harv at the
  NEW proc's footprint and binds it to the closest refinery by
  PATH DISTANCE** (engine fix, pinned by
  `OpenRA-Rust/openra-sim/tests/test_proc_auto_spawn_at_new_proc.rs`
  + `tests/test_proc_auto_spawn_python.py`). Historical footgun:
  the engine routed the auto-harv through `find_spawn_location`,
  which sorts candidates by `(!is_primary, id)` — so a 2nd proc
  placed far from the 1st always materialised its harv at the
  LOWEST-ID proc, and `find_refinery` returned the lowest-id proc
  unconditionally. The combined effect: expansion to a contested
  patch was a no-op (the new harv trekked back to the old
  refinery, and the old harv kept depositing at the old
  refinery). The fix: a new `spawn_unit_near_building(actor,
  unit_type, owner, building_id)` anchors the spawn scan on the
  NEW proc's footprint, and `find_refinery_from(owner, cell)`
  picks the proc with the shortest A* path from `cell` (with
  fallback to Chebyshev-nearest then lowest-id). A 2nd refinery
  placed near a contested patch now produces real throughput.
  **Existing harvesters do NOT re-snap** to the new proc — the
  re-resolve only fires when the stored refinery id is stale
  (proc destroyed / never existed). To reroute live harvesters,
  the agent must `set_primary` on the new proc or sell the old
  one.
- **Thief `Infiltrate` is a no-op against any non-`proc` /
  non-`silo` enemy building** (engine match-arm intent). The thf
  walks to the target, is consumed, and 0 cash is drained. The
  Python tool description (`infiltrate`) already documents this:
  the cash-drain branch is gated on `proc | silo`. Bench
  scenarios that want the thief to load-bear must direct it at a
  refinery or silo specifically.
- **`stance:0` HoldFire defenders never return fire even when
  attacked** — engine-intended (pinned by
  `test_stance_semantics.rs::test_stance_0_holds_fire`). The
  defenders die silently. For a defense scenario where the model
  is expected to flip stance under threat: pre-place defenders at
  `stance:0`, expose `set_stance` in `tools:`, and gate the win
  on combat damage so a stall play (no stance flip) loses by
  having the base destroyed without resistance.
- **Per-player starting cash is now plumbed end-to-end** (engine
  fix, pinned by `OpenRA-Rust/openra-sim/tests/test_per_player_starting_cash.rs`
  + `OpenRA-Rust/openra-data/tests/test_per_player_starting_cash.rs`
  + `tests/test_per_player_starting_cash.py`). A scenario YAML's
  `agent: {cash: N}` / `enemy: {cash: M}` is honoured per slot;
  back-compat path (neither override set) falls back to the
  top-level `starting_cash:`. This is the wiring the thief
  `spec-thief-steal-cash` and asymmetric-econ packs depend on.
- **`Command.fire_superweapon` is the only superweapon verb**
  (no other `Command::*` variant fires nukes / iron curtain /
  chrono). Tool entry: `fire_superweapon{kind, target_x?, target_y?,
  target_id?}`. End-to-end pin:
  `tests/test_superweapons_python.py` (Python) +
  `OpenRA-Rust/openra-sim/tests/test_superweapons.rs` (Rust). The
  engine validates (a) the agent owns a launcher building of the
  matching `kind`, (b) the weapon is fully charged (charge time
  is hard-coded 100 ticks per kind for tests; real-play values
  live in `gamerules.rs`); a failed validation is logged and the
  order is dropped silently. Nuke needs `target_cell`; iron
  curtain needs `target_id` only; chrono needs both
  (`target_cell` = destination, `target_id` = friendly actor to
  teleport).

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

## 1v1 LLM-vs-LLM mode

The bench supports head-to-head LLM-vs-LLM matches via the engine's
`step_1v1` two-player command channel
(`OpenRA-Rust/openra-train/src/env.rs::step_1v1`) and the
`openra_bench.one_v_one.run_1v1` harness. The canonical adversarial
cell is `openra_bench/scenarios/packs/adversarial-1v1-macro.yaml` —
a mirrored full-macro arena across three rungs (60×60 / 80×80 /
96×72 bridges + naval) where each commander starts with an
identical roster (`fact` + `proc` + `harv` + `powr` + `weap` +
`tent` + 2 combat units, $2000), each side sees ONLY its own
fog of war, and the winner is decided from the FINAL boards (base
elimination > economy tie-break).

**Pack convention for adversarial 1v1 cells**:

* `capability: adversarial` so `pairwise.py` / `adversarial.py`
  ladder picks the pack up.
* **NO `reveal_map`**. Both sides explore through fog; asymmetric
  information is the whole point. The engine's `reveal_map`
  flag only applies to the AGENT viewer
  (`env.rs:1938`), so it would break the symmetric contract
  even if enabled — don't add it.
* Mirrored symmetric layout: identical roster on both sides,
  identical `starting_cash`, identical factions, ore patches
  mirrored around the map centre, contested middle patch outside
  starting sight.
* Per-side `agent: {bot_type: ''}` AND `enemy: {bot_type: ''}` so
  no scripted engine bot co-drives either side — the Controller
  is the only hand on the wheel.
* No load-bearing per-side win/fail predicates. The schema
  requires a `win_condition`; declare a symmetric stub
  (`not has_building: fact` for both clauses); `run_1v1` decides
  the winner from final boards. Predicates exist only for schema
  hygiene.
* `max_turns: 120` is the working budget — reachable max tick ≈
  93 + 90·119 = 10803, well clear of any plausible deadline.
* Hard-tier rule: a perfectly symmetric arena cannot use
  `spawn_point` rotation without introducing a per-seed bias
  (one corner gets attacked from the centre vs the rim). Vary
  MAP TOPOLOGY (bridges, naval lanes) on hard instead, and add
  the pack to `tests/test_hard_tier.py::NOT_APPLICABLE` with the
  symmetric-arena rationale.

**CLI invocation**:

```
python3 -m openra_bench.run_eval \
    --mode 1v1 \
    --packs openra_bench/scenarios/packs/adversarial-1v1-macro.yaml \
    --levels easy,medium,hard \
    --seeds 1,2,3 \
    --provider openrouter --model anthropic/claude-3.5-sonnet \
    --opponent openrouter:openai/gpt-4o-mini \
    --side-swap \
    --out 1v1_stats.json
```

`--opponent` accepts `scripted:<kind>` (`stall`, `rusher`) for
deterministic baselines or `<provider>:<model>` for LLM
opponents. `--side-swap` plays each match twice with sides
swapped and reports the aggregate as a draw when one half wins
and the other loses — the symmetric-arena tie-break.

`--provider scripted:stall` (and `scripted:rusher`) is the
analogous escape hatch on the AGENT side, useful for CLI smoke
tests and for the fairness probe.

**Fairness contract**:

* Both sides have fog of war (no `reveal_map`).
* Identical roster, cash, faction, stances at t=0.
* Both sides receive their OWN shroud-scoped observation through
  `enemy_observation()` and the per-player observation channels
  of `step_1v1`.
* No engine-side asymmetry beyond the engine's residual playerRNG
  slot ordering, which `--side-swap` is designed to neutralise.
* `run_1v1` is deterministic given identical inputs (pinned by
  `tests/test_one_v_one_macro.py`).

## Production eval workflow (multi-hour sweeps)

A v1.0 Qwen 9B sweep had 205/653 journal↔disk mismatches — entries
the journal said were `done` but where the on-disk `score.json` was
missing or carried a different `outcome`. The default `--resume` is
trusting (skips any cell whose journal row says it's done) and that
trust does NOT hold for long sweeps. For any production run across
multiple models / modes, use the v11 hardening flags:

* **`--strict-resume`** — on resume, every journaled cell is
  cross-checked against its on-disk `score.json`. Rows where the
  score.json is missing / corrupt / carries a different `outcome`
  are DROPPED from the resume set (the cell re-runs). Recommended
  for any multi-hour sweep. The dropped rows are logged to stderr
  so the operator can see what was re-run and why.

* **`--ignore-run-id`** — a journal carries a `_meta` header with
  the `run_id` of the process that wrote it. Re-opening a journal
  with a DIFFERENT current run_id aborts unless this flag is
  passed. The flag is the explicit "I am merging two sweep runs
  into one journal" knob; without it, parallel sweeps pointed at
  the same journal silently produced a frankenstein run.

* **`--adaptive-concurrency`** — monitors the rolling per-cell
  error rate over a 20-cell window; halves concurrency when error
  rate exceeds 10%, restores when a clean 50-cell stretch (<2%
  errors) is observed. Every change is logged to stderr with the
  trigger. The right knob for provider rate-limit storms — the
  sweep slows down instead of aborting.

The `RunJournal` also dedupes `_key`s in-memory inside the append
lock (raises `DuplicateJournalKey` on a same-process re-append),
so the v1.0 `adversarial-duel:easy`-appearing-twice footgun cannot
happen any more.

### Status command

```
python3 -m openra_bench.run_eval status --out data/runs/v1.1-qwen-9b/
```

Prints the run dir, the journal's `_meta` header (run_id, model,
code SHA), per-outcome counts (win / loss / draw / error), the
journaled mean composite, the last cell completed, and a count of
on-disk `score.json` files for cross-reference. Tolerates partial
/ empty / corrupt journals (Ctrl-C-shaped torn tails are common
during long sweeps).

### Recommended invocation

For a real multi-hour multi-model sweep:

```
python3 -m openra_bench.run_eval \\
  --provider openrouter --model <model> \\
  --playback data/runs/<run-name>/ \\
  --concurrency 8 --adaptive-concurrency \\
  --strict-resume \\
  --max-spend 50.0 \\
  --out data/runs/<run-name>/stats.json
```

A killed process re-launched with the same `--playback` + `--model`
resumes losslessly (strict gate re-verifies every entry); the same
re-launch with a DIFFERENT process gets a `JournalRunIdMismatch`
unless `--ignore-run-id` is passed.

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
