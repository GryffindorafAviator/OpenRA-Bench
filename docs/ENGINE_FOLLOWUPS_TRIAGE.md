# Engine Follow-ups — Triage against `OpenRA-Rust/main` HEAD `8ca8989` (post-merge)

Cross-checks the seven findings in `docs/ENGINE_FOLLOWUPS.md` against
`OpenRA-Rust/main` HEAD `8ca8989` (the engine-feature-wave merge,
2026-05-23). The previous version of this doc (HEAD `feb8981`)
classified five findings as "unreproducible — feature lives only on
`engine-feature-wave`"; that branch is now merged so each finding has
a real status against the current `main`.

## Executive summary

| # | Finding | Post-merge status | Action |
|---|---------|-------------------|--------|
| 1 | Tick-rate doc drift | DOC-ONLY (engine constant is 30/step) | **FIXED** — `OpenRA-Bench/CLAUDE.md` updated |
| 2 | Ore density clamps at 12/cell | BROKEN (clamp still at 12) | DOC-FIXED — `schema.py::ore_patches` docstring covers the recipe |
| 3 | Nuke AoE kills not credited to `kills_per_player` | **FIXED** (engine fix shipped this run) | Engine fix + 2 Rust tests landed |
| 4 | Explicit `harvest(unit_id, x, y)` drifts off target | BROKEN (harv re-binds to nearest patch after first deposit) | Failing `#[ignore]`d test pinned; **TODO engine fix** |
| 5 | ReturnFire stance breaks kite-and-pull | PARTIALLY FIXED (stance:1 fix landed); residual is balance/empirical | TODO empirical re-measure of `combat-kite-and-pull` |
| 6 | Arena generator's default 4-corner mpspawns | DOC-ONLY (auto-MCV varies per seed; engine doesn't offset pre-placed actors) | **FIXED** — `ENGINE_FOLLOWUPS.md` rewording |
| 7 | Pre-placed `proc` overlapping `ore_patches:` | BROKEN (patch is seeded, but proc footprint blocks pathing — silent sterilisation) | Failing `#[ignore]`d test pinned; **TODO validator** |

Wheel rebuilt cleanly: `Installed openra_train-0.1.0` on
`maturin develop --release` against HEAD `8ca8989`. `openra-sim` lib
+ integration tests are green except for known pre-existing
fixture/environment failures (see end of doc).

---

## Finding 1 — Tick-rate discrepancy (doc drift)

**Status on `main`:** DOC-ONLY (engine constant is 30/step, not 90).

`openra-train/src/env.rs:33` declares `pub const
DEFAULT_TICKS_PER_STEP: u32 = 30;` and `Env::step` (env.rs:351-364)
advances exactly that many ticks per `step()`. The bench's
`RustEnvHandle.__init__` does not call `with_ticks_per_step` so every
`env.step` advances 30 ticks. Under `step_until_event` (interrupt
mode), the loop runs UP TO `max_ticks` (default 5 in bench code) and
breaks on the first signal — per-turn advance is variable in
`[1, 5]`.

**Action taken:** `OpenRA-Bench/CLAUDE.md` updated in two spots
(defective-scenario list and engine-facts list). The "~90 ticks/turn"
phrase is gone; replaced with the actual `DEFAULT_TICKS_PER_STEP`
constant + a note to read `info["ticks_advanced"]` for interrupt-mode
runs.

---

## Finding 2 — Ore density clamps at 12 cells

**Status on `main`:** BROKEN, but documented.

Clamp lives at `openra-sim/src/resource.rs:94-95`:
```rust
let per_cell = ((patch.amount.max(1) + passable_cells - 1) / passable_cells)
    .clamp(1, 12) as u8;
```
Ceiling is 12 ore/cell; raising the cap is a real (though small)
balance change. The cheapest mitigation is the docstring (option
(c) in the spec) — already applied to `schema.py`.

**Action taken:** `openra_bench/scenarios/schema.py::MapDef.ore_patches`
docstring now reproduces the clamp formula and the recommended ratio
`amount ≈ 12 · π · radius² · 1.2`, plus the 50 cr/ore conversion. New
econ packs that use `ore_patches:` get the recipe inline.

**Optional follow-up (NOT in this commit):** raise the cap to 24/cell
in `resource.rs:95` and update the `seed_patch_fills_disk` density
assertion. Out of scope for triage; documented for whoever wants to
pick it up.

**Bench-side workarounds in place:** `econ-mine-and-grow.yaml`,
`econ-multi-patch-allocation.yaml`, `econ-second-base-race.yaml`,
`mcv-deploy-second-base.yaml` already inflate `amount` to fill the
cap; the new docstring formalises that recipe so future packs don't
re-discover the footgun.

---

## Finding 3 — Nuke AoE kills not credited to `kills_per_player`

**Status on `main`:** **FIXED** in this triage pass.

Pre-fix `world.rs::detonate_nuke` (lines 2101-2141) removed dead
actors via `self.actors.remove(&id)` without ever incrementing
`kills_per_player`. The `_owner` parameter was even prefix-`_`'d to
silence the unused-variable warning, telegraphing the bug.

**Engine fix shipped:** `world.rs::detonate_nuke` now uses `owner`
and credits `kills_per_player` once per dead actor whose owner is
NOT the firing owner (mirrors the projectile-resolve scoping —
friendly-fire kills do NOT credit, so an agent can't game
`units_killed_gte` by nuking its own units).

**Pinning:** `OpenRA-Rust/openra-sim/tests/test_nuke_credits_kills.rs`
— two tests:
- `nuke_credits_each_enemy_kill_to_firing_owner` — fire mslo at a
  cluster of 5 enemy e1s, assert `kills_for_player(owner)` rises by
  exactly 5.
- `nuke_does_not_credit_friendly_fire_kills` — fire mslo at agent's
  own e1 cluster, assert no credit.

**Existing test still green:** `test_superweapons.rs::mslo_nuke_kills_enemy_cluster`
runs unchanged (the new credit logic is additive).

**Bench-side cleanup queued (TODO, NOT this commit):**
- `OpenRA-Bench/openra_bench/scenarios/packs/spec-nuke-strike.yaml`
  currently uses `enemy_buildings_destroyed_gte` with a `silo`-cluster
  workaround (lines ~62-70 by triage finding #3). With the fix in
  place, the hard-tier predicate can switch to `units_killed_gte: 5`
  against an infantry cluster — restoring the original "nuke clears
  infantry cluster" semantics.

---

## Finding 4 — Explicit `harvest(unit_id, x, y)` orders drift off target

**Status on `main`:** BROKEN.

`world.rs::order_harvest` (lines 1372-1402) is idempotent and stores
`last_harvest_cell: Some(target)` — so the FIRST `FindingOre` cycle
honours the explicit target. But `tick_harvesters` (line 2261) calls
`terrain.find_nearest_resource(center, 15)` with `center =
last_harvest_cell.or(loc)`, and `last_harvest_cell` is overwritten
each successful harvest tick (line 2328) to the actually-harvested
cell. After the first deposit, the harv's "explicit target" has
already drifted to wherever it last mined; subsequent FindingOre
cycles search around that point and pick up whatever ore is closest,
without remembering which patch the original `harvest()` order
selected. With multiple patches inside a 15-cell window, the
explicit allocation evaporates within ~30 ticks.

**Pinning:** `openra-sim/tests/test_harvest_explicit_target.rs::explicit_harvest_target_stays_bound_to_far_patch`
— `#[ignore]`d. Sets up a near patch (10,15) and far patch (40,15),
issues an explicit Harvest order targeting the FAR patch, runs 3000
ticks, asserts every harvested cell stays within Chebyshev distance
≤ 4 of the far patch. Currently the harv would re-bind to whichever
patch is closer to its mining position. Flip to `#[test]` after the
fix lands.

**TODO engine fix:** add `bound_patch: Option<(x, y, radius)>` to
`Activity::Harvest`, set by `order_harvest` from the explicit target
(radius derived from the registered `OrePatchDef` containing the
cell, default 3). FindingOre's resource search restricts to cells
inside the patch disc when `bound_patch` is set. Cleared by a new
Harvest order to a different target or by `stop`. Multi-step change
(touches `Activity::Harvest` shape, `order_harvest`, `tick_harvesters`,
all `#[derive]`'d serialization tests) — not a one-liner; pinned-test
+ TODO is the right form.

**Bench-side workarounds:** `econ-multi-patch-allocation.yaml:65-87,
212-216` pre-stages one harv beside each patch and relies on the
auto-route binding each to the nearest proc — the workaround dodges
the explicit-allocation discipline by hand-placing harvs at the
target. Once the engine fix lands, the pack docstring and pre-staged
positions can be removed; `harvest()` becomes load-bearing again.

---

## Finding 5 — ReturnFire-stance auto-fire breaks kite-and-pull

**Status on `main`:** PARTIALLY FIXED — empirical re-measure needed.

The original finding's option (b) — "Pure ReturnFire that fires only
after taking hits" — landed in commit `9e999e2 fix(engine): clarify
stance — stance:1 true return-fire-only`. Pinned by
`openra-sim/tests/test_stance_semantics.rs::test_stance_1_return_fire_only_against_passive_enemy`.
A stance:1 unit next to a passive (stance:0) enemy now holds fire
indefinitely.

But the **scenario** breaks for a different reason: once a stance:1
raider takes one hit from a 3tnk (hunt bot), it auto-fires forever
(the 60-tick window self-refreshes on each subsequent hit). So the
1v1 kite question becomes a balance question: does a passive
ReturnFire 2tnk out-trade an aggressive 3tnk by auto-fire alone? On
the OLD stance:1 (auto-engage on any in-range enemy) the answer was
yes — the original finding's empirical claim. On the NEW stance:1
the answer depends on exact weapon DPS values which changed across
the wave merges; uncertain without a fresh empirical run.

**TODO empirical re-measure:** run the stand-still policy
(no agent orders) on `combat-kite-and-pull.yaml` against `main` HEAD
`8ca8989`. Suggested test:
`OpenRA-Rust/openra-sim/tests/test_kite_1v1.rs` — 1 2tnk at (10,10)
stance:1 vs 1 3tnk at (40,10) stance:3 on a 60×30 arena, run 500
frames with no orders, assert 2tnk dies. If it passes, finding #5 is
genuinely fixed and the bench's 3-raider focus-fire workaround in
`combat-kite-and-pull.yaml:181-183` can be reverted to a true 1v1.
If it fails, the residual is balance: tune the 2tnk's `90mm` reload
delay or damage in `gamerules.rs`.

**Bench-side workaround:** `combat-kite-and-pull.yaml` uses a
3-raider stack to substitute focus-fire micro for distance-control
(documented inline). Workaround is correct given current uncertainty;
remove only after the empirical re-measure resolves.

---

## Finding 6 — Arena generator's default 4-corner mpspawns rotate per-seed

**Status on `main`:** DOC-ONLY (engine code does not offset
pre-placed actor coords).

Re-reading the engine source: `build_scenario_actor`
(`openra-train/src/env.rs:2338+`) places each scenario actor at its
literal `sa.position` — there is NO offset math anywhere. The
finding's claim "authored `position: [6, 5]` actually appears at
y=29-33 on some seeds" is not substantiated by the code. The
practical effect that DOES exist: `assign_spawn_points`
(`world.rs:4684+`) picks one mpspawn per seed for the auto-spawned
MCV, deterministic on `seed`. So when a pack relies on the
auto-MCV's corner being a specific one (not its position absolute),
that DOES vary per seed.

**Action taken:** `docs/ENGINE_FOLLOWUPS.md` finding #6 reworded to
remove the misleading "engine offsets pre-placed actors" claim.

**Bench-side workaround:** the `spawns: [[6, 5]]` declaration in
`perception-target-vs-fog.yaml:62-68, 157, 200, 252` (and any other
arena-generator pack that hand-places agent buildings AND relies on
a fixed auto-MCV corner) is correct and stays.

**Optional follow-up (NOT in this commit):** change
`openra_bench/mapgen.py::_default_spawns` default to a single centred
mpspawn so position invariants hold without the workaround. Touches
default behaviour for every existing arena-generator pack — out of
scope for a doc-cleanup pass.

---

## Finding 7 — Pre-placed `proc` inside an `ore_patches:` disc silently empties patch

**Status on `main`:** BROKEN (with refined understanding).

`env.rs::build_world_for_episode` lines 958-1029 seed ore patches
FIRST (line 966-974: `seed_ore_patch` for each declared patch), then
inject scenario actors. A `proc` placed inside a patch disc occupies
its footprint cells via `terrain.occupy_footprint`, marking them
ground-impassable WITHOUT clearing the resource layer. Result: ore
IS placed (the original finding's "silently zero ore" wording is
slightly off), but `find_path` cannot reach those cells from the
harvester. The harvester loops "FindingOre → can't path → fail"
indefinitely on the cells under the proc footprint.

**Pinning:** `openra-data/tests/test_proc_overlapping_patch_warns.rs`
— `#[ignore]`d. Loads a scenario with `proc` at (40,10) and
`ore_patches: [{x:40, y:10, amount:5000, radius:3}]`, asserts the
(currently nonexistent) `validate_layout` returns a warning string
containing `proc_overlaps_patch`. Flip to `#[test]` after the
validator ships.

**TODO engine fix:** add `oramap::validate_layout(&MapDef) -> Vec<String>`
that returns one warning per overlapping `(building, patch)` pair.
Surface via `OpenRAEnv.last_warnings` at world-build. Cheap; the
optional second fix (skip ore-seeding on cells already occupied by a
building footprint, or re-order seed-after-actor-injection) is more
invasive — leave for later.

**Bench-side workaround:** trivial (place the proc just outside the
patch radius). Already widely applied; no specific files to clean
up.

---

## Pre-existing test failures (NOT this triage pass)

Documented for clarity so they aren't conflated with finding-related
failures:

- `openra-sim` lib test `gamerules::tests::defaults_have_all_common_units`
  — MCV vs Vehicle kind classification regression, predates this
  pass (logged in `OpenRA-Bench/ENGINE_AUDIT.md`).
- `openra-sim` integration tests `sync_hash_verify` (×2),
  `debug_orders_and_hashes` — sync-hash reference fixtures stale
  after recent merges.
- `openra-sim` integration test `tank_attack_terminates_when_no_armament_loaded`
  + asset-loading data tests (`extract_*`, `find_*`,
  `parse_infantry_yaml`, `e1_rifleman_typed_fields`,
  `render_*`, etc.) — all need the vendored OpenRA mod dir at
  `OpenRA-Rust/vendor/OpenRA/mods/ra/` (not present in this
  workspace; environmental, not code-related).
- `openra-data` `test_per_player_starting_cash`, etc. — earlier these
  failed with `MissingBaseMap` because the `rush-hour-arena.oramap`
  fixture wasn't in `openra-data/tests/fixtures/`. Fixed in this pass
  by copying the map from `OpenRA-Bench/data/maps/`. All four
  per_player_starting_cash + 3 ore_patches data tests are green now.

## Summary of changes shipped this pass

**Engine (OpenRA-Rust):**
- `openra-sim/src/world.rs::detonate_nuke` — credit nuke AoE kills to
  firing owner (mirrors projectile-resolve path; friendly-fire is
  excluded).
- `openra-sim/tests/test_nuke_credits_kills.rs` — new (2 tests).
- `openra-sim/tests/test_harvest_explicit_target.rs` — new
  (`#[ignore]`d, pins finding #4).
- `openra-data/tests/test_proc_overlapping_patch_warns.rs` — new
  (`#[ignore]`d, pins finding #7).
- `openra-data/tests/fixtures/rush-hour-arena.oramap` — copied in
  from `OpenRA-Bench/data/maps/` so existing data tests run cleanly
  in this workspace.

**Bench (OpenRA-Bench):**
- `CLAUDE.md` — finding #1 doc fix in two places.
- `openra_bench/scenarios/schema.py::MapDef.ore_patches` docstring —
  finding #2 doc fix (clamp formula + recommended ratio).
- `docs/ENGINE_FOLLOWUPS_TRIAGE.md` — this file (refreshed against
  post-merge HEAD).
- `docs/ENGINE_FOLLOWUPS.md` — finding #6 reworded to drop the
  misleading "engine offsets pre-placed actors" claim.

**TODOs surfaced:**
- finding #4: implement `bound_patch` on `Activity::Harvest` (see test).
- finding #5: empirical re-measure on `combat-kite-and-pull` to
  verify whether the stance:1 fix is sufficient.
- finding #7: implement `oramap::validate_layout` (see test).
- finding #2: optional cap raise from 12 → 24/cell.
- spec-nuke-strike.yaml: switch hard-tier predicate to
  `units_killed_gte` now that finding #3 is fixed.
