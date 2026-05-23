# ENGINE_AUDIT — Phase 1

End-to-end completeness audit of every engine verb + observation
field, written 2026-05-22 against `OpenRA-Rust@engine-feature-wave`
HEAD `a5014a5` and `OpenRA-Bench@pr13-revised`.

Scope: pin tests for every gap that scripted-policy validation has
caught, audit advanced-feature surface coverage (Rust + Python),
audit observation completeness, and verify the
`Command::*` ↔ Python static ↔ agent.py tool-entry surface is
complete.

---

## 1. Engine gaps — status

| # | Gap | Status | Pinning |
|---|-----|--------|---------|
| 1 | `place_building('proc')` auto-spawned harv at lowest-id proc (not new one); `find_refinery` returned lowest-id proc unconditionally ⇒ 2nd refinery far from 1st added no throughput | **FIXED** in `openra-sim/src/world.rs` this phase. Added `spawn_unit_near_building(unit_type, owner, building_id)` (anchors scan on the NEW building's footprint), `find_refinery_from(owner, from_cell)` (path-shortest with Chebyshev fallback), and rewired `order_place_building` + `harvester_start_delivery` stale-id resolve to use them. Existing harvs do NOT re-snap (only stale-id resolve calls the path-shortest helper); to reroute live harvs the agent must `set_primary` on the new proc or sell the old one. | `OpenRA-Rust/openra-sim/tests/test_proc_auto_spawn_at_new_proc.rs` (1 test) + `OpenRA-Bench/tests/test_proc_auto_spawn_python.py` (1 test) |
| 2 | Thief `Infiltrate` drained 0 cash because `enemy: {cash: N}` was historically ignored | **FIXED upstream** (per-player starting cash plumb landed in commit `a5014a5`). Verified `tests/test_per_player_starting_cash.py` passes against the rebuilt wheel. Engine + Rust + Python tests all green. | `OpenRA-Rust/openra-sim/tests/test_per_player_starting_cash.rs` (3) + `OpenRA-Rust/openra-data/tests/test_per_player_starting_cash.rs` (4) + `tests/test_per_player_starting_cash.py` (2) |
| 2b | Thief `Infiltrate` against any non-`proc`/`silo` building is a no-op (engine match-arm intent) | **INTENT — DOCUMENTED**. Tool description in `agent.py` already states "thief drains a chunk of the target owner's cash to your player (only when the target is a proc or silo)". Added a note to bench `CLAUDE.md` engine-footguns block. | `openra-sim/tests/test_infiltrate.rs::thief_infiltration_steals_enemy_cash` covers the proc-targeted happy path. |
| 3 | Stance:0 (HoldFire) units don't return fire even when attacked ⇒ defenders die silently | **INTENT — already-pinned**; added an explicit defender-perspective note to bench `CLAUDE.md` so pack authors don't author defense scenarios that silently lose to a stall policy. | `openra-sim/tests/test_stance_semantics.rs::test_stance_0_holds_fire` |
| (bonus, found this audit) | `fire_superweapon` had no `agent.py` tool entry — model couldn't issue superweapon orders | **FIXED** — added `_TOOL_SCHEMAS["fire_superweapon"]` with `kind / target_x / target_y / target_id` parameter schema + a `_to_commands` case that maps `(target_x, target_y) → cell tuple` and forwards `target_id` as string. Bumped `tests/test_tools.py::test_wildcard_exposes_everything` from 21 → 25 (covers every Command variant now). | `OpenRA-Bench/tests/test_superweapons_python.py` (4 tests: nuke / iron / chrono / launcher-validation) — none existed pre-audit. |
| (bonus, found this audit) | No Rust unit test exercised the full APC `EnterTransport → Move → Unload` loop end-to-end | **FIXED** — added a single integration test that boards an e1 into an APC, drives ~30 cells east, unloads, and asserts the passenger lands within 4 cells of the destination and is back in the active actor map. | `OpenRA-Rust/openra-sim/tests/test_apc_transport.rs` (1) + `OpenRA-Bench/tests/test_apc_transport_end_to_end.py` (1) |

### Pre-existing failures observed during audit (NOT caused by this phase's changes)

These were already failing on `engine-feature-wave` HEAD when I
checked out the branch. Documented here so they aren't conflated
with this phase's diff:

- `openra-sim` lib test `gamerules::tests::defaults_have_all_common_units` — MCV vs Vehicle kind classification regression.
- `openra-sim` integration tests `sync_hash_verify` + `debug_sync` — sync-hash reference fixtures are stale and need regeneration after the recent engine merges (`a5014a5`, `2a1cd30`, `9f2181b`, `0a13243`, `b828c3b`).
- `OpenRA-Bench/tests/test_parallel_production.py::test_two_war_factories_outproduce_one` — a single war factory produces 0 tanks in the test budget (not 1+). Loop times out; `place_building` reports `PLACE BLOCKED: pbox not completed in queue` repeatedly in the related `test_pbox_fires.py`. The shared symptom suggests a production-queue advance regression in one of the recent merges (`order_place_building.has_completed` evaluates false even after the build timer expired). Out of scope for Phase 1; flagged for Phase 2.
- `OpenRA-Bench/tests/test_pbox_fires.py::test_built_pbox_kills_enemy_e1` — same root cause as parallel_production (pbox never gets placed, so it never fires).

---

## 2. Advanced-feature pinning matrix (verb × Rust × Python)

`Command::*` = the engine verb in `openra-train/src/command.rs`.
A ✓ in "Rust" means there is at least one `cargo test`-runnable
test in `openra-sim/tests/` or `openra-data/tests/` that exercises
the order through `process_frame`. A ✓ in "Python" means there is
a `pytest`-runnable test in `OpenRA-Bench/tests/` that exercises
the order through `Command.<verb>` + the `RustEnvHandle.step`
boundary.

| Verb | Rust pinning test | Python pinning test |
|------|-------------------|---------------------|
| `MoveUnits` | `openra-sim/tests/move_activity_replay.rs` + `parity_move_vs_csharp.rs` | `tests/test_resource_economy.py`, many combat packs |
| `AttackUnit` | `openra-sim/tests/test_attack_unit_no_teleport.rs` + `combat_one_v_one.rs` | many combat tests (`test_combat_*.py`) |
| `AttackMove` | covered via combat scenarios | covered via combat scenarios |
| `Guard` | covered via Move tests (Guard is follow-subset) | `tests/test_combat_protect_vip_escort.py` |
| `SetPrimary` | (no dedicated test — exercised via `primary_buildings` set / `find_spawn_location` sort key) | `tests/test_repair_building_id.py` |
| `EnterTransport` | **NEW: `openra-sim/tests/test_apc_transport.rs`** | **NEW: `tests/test_apc_transport_end_to_end.py`** |
| `Unload` | **NEW: `openra-sim/tests/test_apc_transport.rs`** | **NEW: `tests/test_apc_transport_end_to_end.py`** |
| `Stop` | covered via Move tests | covered |
| `Deploy` | (covered via env-level integration) | `tests/test_mcv_deploy.py`, `test_mcv_deploy_*.py` |
| `Build` | `openra-sim/tests/test_parallel_production.rs` | `tests/test_parallel_production.py` (PRE-EXISTING FAILURE — see §1) |
| `CancelProduction` | (no dedicated test; verb has small surface — refunds last-queued item) | — |
| `PlaceBuilding` | **NEW: `openra-sim/tests/test_proc_auto_spawn_at_new_proc.rs`** + `openra-sim/tests/test_pbox_fires.rs` | **NEW: `tests/test_proc_auto_spawn_python.py`** + `tests/test_build_*.py` packs |
| `Harvest` | `openra-sim/tests/test_resource_layer.rs` | `tests/test_resource_economy.py`, `test_economy_harvest.py` |
| `Sell` | (no dedicated test; exercised by `tests/test_maint_sell_and_recoup_cash.py`) | `tests/test_maint_sell_and_recoup_cash.py`, `test_build_sell_and_rebuild_elsewhere.py` |
| `Repair` | (no dedicated rust test; covered by repair pack tests) | `tests/test_build_repair_priority_under_fire.py`, `test_def_engineer_repair_under_fire.py`, `test_repair_building_id.py` |
| `PowerDown` | `openra-sim/tests/test_power_signals.rs` | `tests/test_power_signals_python.py`, `test_build_power_down_defensive.py` |
| `SetRallyPoint` | (no dedicated test; covered by rally-point pack tests) | `tests/test_build_rally_point_management.py` |
| `SetStance` | `openra-sim/tests/test_stance_semantics.rs` (4 tests) | `tests/test_stance_semantics_python.py` (4 tests) |
| `Patrol` | (no-op verb — accepted, no behaviour) | — |
| `Surrender` | (covered via env-level integration) | `tests/test_surrender.py` |
| `Observe` | covered everywhere (no-op verb) | covered everywhere |
| `C4Detonate` | `openra-sim/tests/test_tanya_c4.rs` (3 tests) | `tests/test_tanya_c4.py` (1) |
| `CaptureActor` | `openra-sim/tests/test_capture.rs` (3) | `tests/test_engineer_capture.py` (1) |
| `Infiltrate` | `openra-sim/tests/test_infiltrate.rs` (2) | `tests/test_infiltrate.py` (2) |
| `FireSuperweapon` | `openra-sim/tests/test_superweapons.rs` (5) | **NEW: `tests/test_superweapons_python.py` (4)** |

### Helicopter / Naval (transport-class verbs)

| Capability | Rust test | Python test | Status |
|------------|-----------|-------------|--------|
| Helicopter pickup / drop (passenger carry) | n/a — engine `transport_capacity()` advertises `tran` (chinook) at 5 but the C# `Cargo` integration for helicopters is NOT wired into the Move activity (helicopters use `Aircraft` kind, not the ground-transport Mobile-board path) | n/a | **GAP — DOCUMENTED**. Helicopters can attack ground targets (covered by `test_aircraft.rs::heli_flies_over_impassable_terrain`, `heli_kills_vehicle_behind_obstacle_wall`) but cannot carry passengers. Bench scenarios must not declare helicopter transport as a load-bearing capability. |
| Naval landing craft (LST) ship-to-shore unload | engine `transport_capacity()` advertises `lst` at 5; no dedicated test pins ship-to-shore unload | n/a | **GAP — DOCUMENTED**. The `EnterTransport` activity tick uses `find_path` (ground), not naval; an infantry trying to board an LST in deep water cannot path there. The LST itself moves on water via `find_path_for_kind(naval=true)`. End-to-end ship-to-shore requires either (a) the LST docking adjacent to a shore cell the infantry can reach by land, or (b) an unload-while-on-water followed by a sink. Not currently exercised by any test. Flagged for Phase 2. |

---

## 3. Observation completeness matrix

`obs key` = the field on the `PyDict` returned by `OpenRAEnv.step`
(see `openra-train/src/observation.rs::to_pydict`). "Present" = the
field exists in every observation; "Tested" = at least one test
asserts on its value.

| Obs key | Present | Tested | Notes |
|---------|---------|--------|-------|
| `unit_positions` (own units `{id → {cell_x, cell_y, actor_type, activity, target?, attacking_target_id?}}`) | ✓ | ✓ | `actor_type` enables `unit_type_count_*` predicates; `attacking_target_id` distinguishes Attack from Move. |
| `unit_hp` (`{id → hp_fraction}`) | ✓ | ✓ | Adapter surfaces as `units_summary[].hp`. |
| `enemy_positions` (visible enemy mobile actors `[{cell_x, cell_y, id, actor_type}]`) | ✓ | ✓ | Fog-filtered through player_0's shroud. |
| `enemy_hp` (`{id → hp_fraction}`) | ✓ | ✓ | |
| `enemy_buildings_summary` (`[{cell_x, cell_y, id, type, hp_pct}]`) | ✓ | ✓ | Adapter merges into `enemy_summary` for the briefing. `hp_pct` is per-building 0..1. |
| `units_killed` (cumulative int) | ✓ | ✓ | Drives `units_killed_gte` predicate. |
| `game_tick` (int) | ✓ | ✓ | |
| `explored_percent` (float 0..100) | ✓ | ✓ | Drives `explored_percent_gte` predicate. |
| `explored_cells` (`[(x,y)]`) | ✓ | ✓ | Sticky per-cell reveal set. |
| `economy.cash` | ✓ | ✓ | Per-player; adapter surfaces as `cash`. |
| `economy.power_provided` / `power_drained` | ✓ | ✓ | `power_provided_gte` + `power_surplus_gte` predicates. |
| `economy.harvesters` (count int) | ✓ | (covered indirectly via `units_summary` actor_type) | Standalone count; no dedicated `harvester_count_*` predicate today. |
| `economy.resources` / `resource_capacity` | ✓ | ✓ | `resources_full_pct` style predicates use these. |
| `own_buildings` (`[{id, type, cell_x, cell_y, hp_pct, is_primary}]`) | ✓ | ✓ | `id` is the REAL engine actor id (footgun closed in prior phase). |
| `production` (`[{item, progress, done}]`) | ✓ | (partial) | Adapter currently collapses to `production_items: [str]`, dropping `progress` / `done`. The `done` flag IS in the raw obs (used by `tests/test_proc_auto_spawn_python.py` directly via `obs["production"]`); the adapter loss is by design (briefing simplicity). **GAP**: the briefing-level production view doesn't surface ETA; the model can see what's queued but not when it lands. |
| `map_info` (`{width, height}`) | ✓ | ✓ | Drives bounds-correct minimap rendering. |
| `spatial` (flat row-major `[y][x][c]` with `c=6`) + `spatial_shape` (h,w,c) | ✓ | ✓ | Channels: `0` passable, `1` fog (1 visible / 0.5 explored / 0 unknown), `2` own-unit density, `3` visible-enemy-unit density, `4` own building, `5` resource present. `SPATIAL_CHANNELS = 6` constant in `observation.rs`. Documented in `observation.rs` doc-comment. |
| `ore_cells` (`[{cell_x, cell_y, amount}]`) | ✓ | ✓ (`test_resource_economy.py`) | Global (NOT fog-gated) per-cell ore inventory. |
| minimap PNG | ✓ (rendered by bench `minimap.py`) | ✓ (`tests/test_minimap.py`, `test_battle_viewer.py`) | Bench-side; not in the raw obs dict but produced by `_render_minimap_b64` in `agent.py`. |
| bounds (playable rectangle) | ✓ (via `map_info`) | ✓ | |
| `enemy_summary` (broader enemy-actor list including units) | ✓ via adapter `render_state()` (it concatenates `enemy_positions` + `enemy_buildings_summary` with `is_building` flag) | ✓ | This is bench-side composition, not a raw-obs key. |

### Observation gaps flagged

1. **`production` ETA**: the briefing-level `production` field loses `progress` / `done` because `RustObsAdapter` collapses to a list of item names. For a "what's coming online next?" planning prompt, the model has to estimate from cash deltas. Recommend either surfacing the per-item ETA in `render_state()` or documenting that the model must use the raw obs.
2. **`spatial` documented but discoverability is low**: `SPATIAL_CHANNELS = 6` lives in the engine doc-comment; the bench doesn't surface a schema describing what channel means what. Add a sentence to `agent.py::build_briefing` or the prompt-v2 system text.
3. **Helicopter cargo & LST ship-to-shore unload**: out of scope for the observation pass but listed under §2 — neither is exercised today.

---

## 4. Command surface matrix (Rust variant × Python static × agent tool entry)

Cross-check of `Command::*` in `openra-train/src/command.rs` against
`PyCommand` staticmethods (same file) against `_TOOL_SCHEMAS` and
`_to_commands` in `openra_bench/agent.py`.

| Rust variant | Python staticmethod | `_TOOL_SCHEMAS` entry | `_to_commands` case | Notes |
|--------------|---------------------|-----------------------|---------------------|-------|
| `MoveUnits` | `move_units` | ✓ `move_units` | ✓ | |
| `AttackUnit` | `attack_unit` | ✓ `attack_unit` (+ alias `attack_target`) | ✓ | |
| `AttackMove` | `attack_move` | ✓ `attack_move` | ✓ (generic case for `attack_move` / `harvest` / `set_rally_point`) | |
| `Guard` | `guard` | ✓ `guard` | ✓ | |
| `SetPrimary` | `set_primary` | ✓ `set_primary` | ✓ (generic `unit_ids` case) | |
| `EnterTransport` | `enter_transport` | ✓ `enter_transport` | ✓ | |
| `Unload` | `unload` | ✓ `unload` | ✓ (generic `unit_ids` case) | |
| `Stop` | `stop` | ✓ `stop` (+ alias `stop_units`) | ✓ | |
| `Deploy` | `deploy` | ✓ `deploy` | ✓ | |
| `Build` | `build` | ✓ `build` | ✓ | |
| `CancelProduction` | `cancel_production` | ✓ `cancel_production` | ✓ | |
| `PlaceBuilding` | `place_building` | ✓ `place_building` | ✓ | |
| `Harvest` | `harvest` | ✓ `harvest` | ✓ (generic case) | |
| `Sell` | `sell` | ✓ `sell` | ✓ | |
| `Repair` | `repair` | ✓ `repair` | ✓ | |
| `PowerDown` | `power_down` | ✓ `power_down` | ✓ | |
| `SetRallyPoint` | `set_rally_point` | ✓ `set_rally_point` | ✓ (generic case) | |
| `SetStance` | `set_stance` | ✓ `set_stance` | ✓ | |
| `Patrol` | `patrol` | ✓ `patrol` | ✓ | No-op verb in engine. |
| `Surrender` | `surrender` | ✓ `surrender` | ✓ | |
| `Observe` | `observe` | ✓ `observe` (always force-included by `_tool_schemas`) | ✓ | |
| `C4Detonate` | `c4_detonate` | ✓ `c4_detonate` | ✓ | |
| `CaptureActor` | `capture_actor` | ✓ `capture_actor` | ✓ | |
| `Infiltrate` | `infiltrate` | ✓ `infiltrate` | ✓ | |
| `FireSuperweapon` | `fire_superweapon` | **✓ `fire_superweapon` (ADDED THIS PHASE)** | **✓ (ADDED THIS PHASE)** | Previously: tool entry was missing — the model could not fire superweapons even on a scenario that exposed `tools: ["*"]`. Fixed. |

**Total: 25 enum variants, 25 Python staticmethods, 25 tool entries, 25 `_to_commands` cases.** Bumped `tests/test_tools.py::test_wildcard_exposes_everything` from 21 → 25 (was already out of sync before this phase).

---

## 5. Prioritized fix queue

In rough priority order (P0 = scenario-blocking, P3 = nice-to-have):

### P0 — scenario-blocking

1. **`place_building` "completion" race regression** — the pre-existing failures in `tests/test_parallel_production.py` and `tests/test_pbox_fires.py` (engine logs `PLACE BLOCKED: <type> not completed in queue` even after the build timer should have expired) point at a regression in `order_place_building`'s `is_done()` check or in the production-queue tick advance. Likely landed in one of the recent merges (`2a1cd30` naval, `9f2181b` air, `0a13243` resource, `b828c3b` superweapon). Affects every build-and-place scenario.
   - **Who-affected**: every `build-*` pack and the parallel-production / pbox guardrails.
   - **Effort**: ~half day to bisect the merge that introduced it + targeted fix.

### P1 — capability gap closing real-world packs

2. **Helicopter passenger carry** — `transport_capacity("tran") == 5` but `EnterTransport` path uses ground pathfinding only; a `tran` actor cannot actually board passengers via the `Mobile` activity tick. Either implement aircraft-load (Aircraft kind needs its own board tick), or drop `tran` from `transport_capacity` to make the no-op explicit.
   - **Who-affected**: any scenario that wants helicopter insert/extract (none today, but the bench has at least three drafted heli scenarios).
   - **Effort**: 1–2 days (aircraft activity surface).

3. **Naval landing craft (LST) ship-to-shore unload** — `transport_capacity("lst") == 5` but the boarding path requires the passenger to reach the LST cell via ground pathfind; on water that's impossible. The C# parity here is "infantry boards at shore + LST docks → infantry rides + LST unloads back on shore". Needs either a shore-adjacency rule for `EnterTransport`, or an explicit `Dock` activity that puts the LST adjacent to a shore cell before boarding.
   - **Who-affected**: naval scenarios (none today; was an aspirational pack).
   - **Effort**: 2 days (touches the EnterTransport tick + naval move).

### P2 — observability

4. **Production ETA surfacing in `render_state()`** — adapter collapses `production` to item-name list; the briefing can't say "tank in 4 turns" without the model reading the raw obs. Surface as `production: [{item, eta_ticks, done}]` in `RustObsAdapter.render_state()`.
   - **Who-affected**: every reasoning pack ("how many turns of cash do I have to spare?").
   - **Effort**: ~1 hour.

5. **Spatial-tensor channel schema in prompt-v2** — `SPATIAL_CHANNELS = 6` is documented in engine code but not in the system prompt the model sees. Add a one-line description to `briefing_image_primary` so an image-channel model knows what each plane means.
   - **Who-affected**: `image-*` perception ablation cells.
   - **Effort**: ~30 minutes.

### P3 — small footguns

6. **`find_refinery_from` fallback when no path exists** — currently falls back to Chebyshev-nearest then lowest-id. If the only proc has a path-blocked footprint (e.g. surrounded by walls), the harv binds anyway and then deadlocks. Could surface a warning in `last_warnings`.
   - **Effort**: ~1 hour.

7. **Existing harvesters do NOT re-snap to the new proc after `place_building('proc')`** — by design (avoids churning a stable supply chain), but documented as a footgun in bench `CLAUDE.md`. If the pack wants per-base supply chains, the model has to `set_primary` on the new proc.
   - **Effort**: 0 — already documented.

8. **`SetPrimary` lacks a dedicated Rust unit test** — exercised indirectly via `find_spawn_location`'s `primary_buildings` sort key but never in isolation.
   - **Effort**: ~1 hour to add.

9. **`CancelProduction` lacks any dedicated test** (Rust or Python). Small verb surface, but a model that frees up cash by cancelling the last queued item should be pinned.
   - **Effort**: ~1 hour to add.

---

## Files touched in Phase 1

### Engine (rebuilt the wheel via `maturin develop --release`; verified `Installed openra_train` printed)

- `openra-sim/src/world.rs` — added `spawn_unit_near_building`, `find_refinery_from`, `spawn_unit_at`; refactored `spawn_unit` to share `spawn_unit_at`; wired `order_place_building` proc-harv auto-spawn to use the new helpers; wired `harvester_start_delivery` stale-id resolve to prefer path-shortest.

### New tests

- `OpenRA-Rust/openra-sim/tests/test_proc_auto_spawn_at_new_proc.rs` — 1 test
- `OpenRA-Rust/openra-sim/tests/test_apc_transport.rs` — 1 test
- `OpenRA-Bench/tests/test_proc_auto_spawn_python.py` — 1 test
- `OpenRA-Bench/tests/test_apc_transport_end_to_end.py` — 1 test
- `OpenRA-Bench/tests/test_superweapons_python.py` — 4 tests

### Bench surface

- `OpenRA-Bench/openra_bench/agent.py` — added `fire_superweapon` tool entry + `_to_commands` case.
- `OpenRA-Bench/tests/test_tools.py` — corrected `test_wildcard_exposes_everything` expectation (21 → 25).
- `OpenRA-Bench/CLAUDE.md` — appended footgun bullets for proc auto-spawn, thief Infiltrate intent, stance:0 defender silent-death intent, per-player cash plumbing, `fire_superweapon` Python surface.
- `OpenRA-Bench/ENGINE_AUDIT.md` — this file.

All changes uncommitted per the Phase 1 constraint.
