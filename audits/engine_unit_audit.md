# Engine Unit Audit — Costs, HP, Prereqs, Features

Single source of truth for every actor in the OpenRA-Bench engine.
Cross-references the engine ground truth against every bench-side claim
(PRODUCTION_TECH_AUDIT.md, EDIT_PRINCIPLES_FAMILY2.md §17,
EDIT_PRINCIPLES_FAMILY11.md §70, `openra_bench/prompt_v2.py::_CODEX`, and
pack briefing prose).

## Methodology

- **Engine truth source**: the bench engine loads the vendored RA mod YAML at runtime via
  `openra-train/src/env.rs::load_rules_with_fallback` (`env.rs:2428-2466`). Vendor YAML files
  live at `OpenRA-Rust/vendor/OpenRA/mods/ra/rules/{structures,vehicles,aircraft,ships,infantry}.yaml`.
  These are the AUTHORITATIVE per-actor numbers used at runtime.
- **`gamerules.rs::defaults()`** (`openra-sim/src/gamerules.rs:272-623`) is a CI/no-vendor fallback
  *only* — its values differ from vendor for many actors (see Discovery #1 below). It is the
  table the recent bench doc fixes silently aligned to.
- **Build time** is `cost × 60 / 100` = `cost × 0.6` ticks per item
  (`openra-sim/src/world.rs::ProductionItem::new` line 78-79, called with modifier `60` at
  `world.rs:901`). At ~90 ticks/decision-turn the per-turn unit-build cost is
  `cost / 150` turns. The bench doc lists "Build (sec)" assuming 30 ticks/sec.

## 1. Total actors audited: 64

Breakdown by kind:

| Kind | Count |
|---|---|
| Building | 27 |
| Vehicle | 13 |
| Aircraft | 4 |
| Ship | 7 |
| Infantry | 13 |

## 2. Mismatch histogram

| Category | Actors flagged |
|---|---|
| Doc cost mismatch (any bench doc disagrees with engine vendor value) | 20 |
| Doc prereq mismatch (hard prereqs differ) | 13 |
| `defaults()` vs vendor divergence (CI fallback drifts from runtime) | 41 |

## 3. Critical discovery — `defaults()` table is a stale CI fallback, NOT runtime truth

The bench's recent cost-mismatch fixes (proc 2000→1400, tent 500→400, heli 1200→2000,
2tnk 850→800, pbox 600→400, hbox 800→600, e7 1200→600) silently aligned the docs to
`gamerules.rs::defaults()` values. But at runtime the engine loads the vendored RA YAML
(`env.rs::load_rules_with_fallback`), and vendor values differ from the defaults for many actors:


| Slug | Vendor (runtime) | `defaults()` (CI only) | What the doc was "fixed to" |
|---|---|---|---|
| `2tnk` | $850 | $800 | $800 |
| `3tnk` | $1150 | $1500 | $950 |
| `4tnk` | $2000 | $1800 | $1500 |
| `agun` | $800 | $600 | $— |
| `apc` | $850 | $800 | $800 |
| `arty` | $850 | $600 | $600 |
| `atek` | $1500 | $2800 | $— |
| `barr` | $500 | $400 | $— |
| `ctnk` | $1350 | $2000 | $— |
| `dome` | $1500 | $2800 | $— |
| `e2` | $150 | $160 | $160 |
| `e4` | $300 | $200 | $— |
| `e6` | $400 | $500 | $500 |
| `e7` | $1800 | $600 | $600 |
| `fact` | $2000 | $0 | $— |
| `gap` | $800 | $500 | $— |
| `gun` | $800 | $600 | $600 |
| `harv` | $1100 | $1400 | $1400 |
| `hbox` | $750 | $600 | $600 |
| `heli` | $2000 | $1200 | $2000 |
| `hind` | $1500 | $1200 | $— |
| `iron` | $2000 | $2800 | $— |
| `jeep` | $500 | $600 | $600 |
| `lst` | $500 | $700 | $— |
| `mcv` | $2000 | $2500 | $2500 |
| `medi` | $200 | $600 | $800 |
| `mnly` | $800 | $500 | $— |
| `mslo` | $2500 | $5000 | $— |
| `msub` | $2000 | $1800 | $— |
| `pbox` | $600 | $400 | $400 |
| `pdox` | $1500 | $2800 | $— |
| `sam` | $700 | $750 | $— |
| `shok` | $350 | $400 | $— |
| `spen` | $800 | $650 | $2000 |
| `stek` | $1500 | $2800 | $— |
| `syrd` | $1000 | $650 | $2000 |
| `tent` | $500 | $400 | $400 |
| `tsla` | $1200 | $1500 | $— |
| `ttnk` | $1350 | $1500 | $— |
| `v2rl` | $900 | $700 | $— |
| `yak` | $1350 | $800 | $— |

**Action**: re-verify whether the recent doc fixes match the *runtime* values (vendor)
or only the *fallback* (`defaults()`). For each row above where `cost_doc` matches the
`defaults()` column but NOT the vendor column, the doc is still wrong at runtime.

## 4. Top mismatches (largest |doc − engine vendor| cost delta)

| Slug | Engine vendor | Doc claim | Δ | Source |
|---|---|---|---|---|
| `e7` | $1800 | $600 | $1200 | audits/PRODUCTION_TECH_AUDIT.md |
| `spen` | $800 | $2000 | $1200 | audits/EDIT_PRINCIPLES_FAMILY11.md §70 |
| `syrd` | $1000 | $2000 | $1000 | audits/EDIT_PRINCIPLES_FAMILY11.md §70 |
| `medi` | $200 | $800 | $600 | openra_bench/prompt_v2.py _CODEX |
| `4tnk` | $2000 | $1500 | $500 | openra_bench/prompt_v2.py _CODEX |
| `afld` | $500 | $1000 | $500 | audits/EDIT_PRINCIPLES_FAMILY11.md §70 |
| `hpad` | $500 | $1000 | $500 | audits/EDIT_PRINCIPLES_FAMILY11.md §70 |
| `mcv` | $2000 | $2500 | $500 | audits/PRODUCTION_TECH_AUDIT.md;openra_bench/prompt_v2.py _CODEX |
| `harv` | $1100 | $1400 | $300 | audits/EDIT_PRINCIPLES_FAMILY2.md §17;audits/PRODUCTION_TECH_AUDIT.md;openra_bench/prompt_v2.py _CODEX |
| `arty` | $850 | $600 | $250 | openra_bench/prompt_v2.py _CODEX |
| `3tnk` | $1150 | $950 | $200 | openra_bench/prompt_v2.py _CODEX |
| `gun` | $800 | $600 | $200 | audits/EDIT_PRINCIPLES_FAMILY2.md §17;audits/PRODUCTION_TECH_AUDIT.md |

## 5. Pack briefing leak risk — packs quoting a specific cost

Packs whose `description:` field mentions a specific `$N` near a slug name where
`N` is plausibly an individual-unit price (within $200) but differs from the engine cost.
Stating the wrong number in the LLM-visible briefing misinforms the agent.

Total leak instances: 35 across 7 slugs.

| Slug | Engine $ | # packs leaking | sample (pack: doc-quoted $) |
|---|---|---|---|
| `2tnk` | $850 | 12 | combat-vehicle-vs-infantry-counter.yaml: \$800 |
| `jeep` | $500 | 11 | scout-jeep-vs-infantry-cost-effective.yaml: \$600 |
| `tent` | $500 | 4 | build-power-online-first.yaml: \$400 |
| `powr` | $300 | 3 | build-engineer-rebuild-after-loss.yaml: \$200 |
| `1tnk` | $700 | 2 | lh-tech-rush-vs-army-rush.yaml: \$800 |
| `e3` | $300 | 2 | lh-tech-rush-vs-army-rush.yaml: \$100 |
| `3tnk` | $1150 | 1 | econ-replace-dead-harvester.yaml: \$1100 |

## 6. Full mismatch list (one row per actor with any doc/engine disagreement)

| Slug | Kind | Engine $ | Doc $ | Engine prereqs | Doc prereqs | Mismatch summary |
|---|---|---|---|---|---|---|
| `2tnk` | Vehicle | $850 | $800 | fix,vehicles.allies | weap,fix | cost: doc=$800 engine=$850 |
| `3tnk` | Vehicle | $1150 | $950 | fix,vehicles.soviet | — | cost: doc=$950 engine=$1150 |
| `4tnk` | Vehicle | $2000 | $1500 | fix,stek,vehicles.soviet | — | cost: doc=$1500 engine=$2000 |
| `afld` | Building | $500 | $1000 | dome | tent | cost: doc=$1000 engine=$500; prereqs: doc=tent engine=dome |
| `apc` | Vehicle | $850 | $800 | vehicles.soviet | — | cost: doc=$800 engine=$850 |
| `arty` | Vehicle | $850 | $600 | dome,vehicles.allies | — | cost: doc=$600 engine=$850 |
| `dd` | Ship | $1000 | $1000 | dome,syrd | syrd | prereqs: doc=syrd engine=dome,syrd |
| `e2` | Infantry | $150 | $160 | barr | — | cost: doc=$160 engine=$150 |
| `e6` | Infantry | $400 | $500 | barracks | tent | cost: doc=$500 engine=$400 |
| `e7` | Infantry | $1800 | $600 | atek,tent | tent,fix | cost: doc=$600 engine=$1800; prereqs: doc=tent,fix engine=atek,tent |
| `gun` | Building | $800 | $600 | tent | tent,fix | cost: doc=$600 engine=$800; prereqs: doc=tent,fix engine=tent |
| `harv` | Vehicle | $1100 | $1400 | proc | weap | cost: doc=$1400 engine=$1100; prereqs: doc=weap engine=proc |
| `hbox` | Building | $750 | $600 | tent | tent,fix | cost: doc=$600 engine=$750; prereqs: doc=tent,fix engine=tent |
| `heli` | Aircraft | $2000 | $2000 | atek,hpad | hpad | prereqs: doc=hpad engine=atek,hpad |
| `hpad` | Building | $500 | $1000 | dome | tent,proc,fact | cost: doc=$1000 engine=$500; prereqs: doc=tent,proc,fact engine=dome |
| `jeep` | Vehicle | $500 | $600 | vehicles.allies | weap | cost: doc=$600 engine=$500 |
| `mcv` | Vehicle | $2000 | $2500 | fix | weap,fix | cost: doc=$2500 engine=$2000; prereqs: doc=weap,fix engine=fix |
| `medi` | Infantry | $200 | $800 | tent | — | cost: doc=$800 engine=$200 |
| `pbox` | Building | $600 | $400 | tent | tent | cost: doc=$400 engine=$600 |
| `powr` | Building | $300 | $300 | — | fact | prereqs: doc=fact engine=∅ |
| `spen` | Building | $800 | $2000 | anypower | proc | cost: doc=$2000 engine=$800; prereqs: doc=proc engine=anypower |
| `syrd` | Building | $1000 | $2000 | anypower | proc | cost: doc=$2000 engine=$1000; prereqs: doc=proc engine=anypower |
| `tent` | Building | $500 | $400 | anypower | fact | cost: doc=$400 engine=$500 |
| `thf` | Infantry | $500 | $500 | barr,dome | tent,fix | prereqs: doc=tent,fix engine=barr,dome |

## 7. Recommended fixes

### 7a. Doc realignment — choose ONE truth (vendor) and align every doc to it

Per Discovery #1, the bench has been silently maintaining two parallel cost tables
(vendor YAML — runtime — vs `gamerules.rs::defaults()` — CI fallback). The
**vendor values are what the LLM-facing simulator actually uses**. Recommend:

1. Update `audits/PRODUCTION_TECH_AUDIT.md` cost column to vendor values across the board.
2. Update `audits/EDIT_PRINCIPLES_FAMILY2.md §17` build prereqs table to vendor values.
3. Update `audits/EDIT_PRINCIPLES_FAMILY11.md §70` (hpad/afld 1000 → 500; syrd 2000 → 1000;
   spen 2000 → 800; dd 1000 ✓; heli 2000 ✓).
4. Update `openra_bench/prompt_v2.py::_CODEX` cost numbers (e.g. `2tnk` $850 ✓, `harv` $1100 ✓,
   `3tnk` change from $950 to $1150, `4tnk` from $1500 to $2000, `e7` add at $1800).
5. Align `gamerules.rs::defaults()` to vendor as a follow-up so CI tests match runtime.

### 7b. Briefing prose pass

Sweep the 56 packs whose briefings hard-code a unit price (`/tmp/unit_cost_leaks.txt`)
and replace specific `$N` per-unit quotes with engine-vendor values OR rephrase as
"roughly $X" to absorb future engine tuning. The leak audit at
`audits/briefing_leak_audit.md` already flagged 140/168 briefings as HEAVY-leak (the
strategy/spend leak is independent of the cost-truth question).

### 7c. Engine — fold defaults() onto vendor as a follow-up

`openra-sim/src/gamerules.rs::defaults()` is reachable only when the vendor dir is
absent (CI without submodules). Bench tests using `GameRules::defaults()` directly
(`world.rs::actor_speed`, sync tests, the `defaults_have_all_common_units` test) will
see different numbers than production runs. Align defaults to vendor in a separate
commit to remove the dual-truth.

## 8. Previously-undocumented engine facts

- **Build time = cost × 0.6 ticks**. `ProductionItem::new(name, cost, 60)` is called
  with `build_duration_modifier=60`. `total_time = cost × 60 / 100`. So a $1400 `proc`
  builds in 840 ticks ≈ 9.3 turns at 90 ticks/turn; a $2000 `weap` in 1200 ticks ≈ 13.3
  turns. Multiple production buildings of the same category produce in PARALLEL
  (CLAUDE.md notes this) so the wall-clock per-unit time halves with two factories.
- **`fact` has cost 2000 in vendor YAML** (vs cost 0 in `defaults()`). The engine special-
  cases cost-0 actors to refuse `StartProduction` (`world.rs:894` `if cost > 0`). With the
  vendor ruleset loaded, `fact` becomes a *buildable* construction yard (the production
  branch fires). With `defaults()` it's `0` and unbuildable. **Pack authors who assume**
  **`fact` cannot be built must verify which ruleset path their test uses.**
- **`pbox` has NO `Armament` in vendor YAML** — the engine attaches `M60mg` at runtime
  for garrison-only defenses without an explicit armament (`gamerules.rs:153-160`). This
  is the CLAUDE.md "pbox is now an active direct-fire tower" fix.
- **`e7` (tanya) vendor cost is $1800**, not $600 (`defaults()`). Real-play Tanya is
  3x more expensive than the defaults table suggests. The engine code aliases `tanya`→`e7`
  via the special path at `gamerules.rs:229-233`.
- **Heavy tanks (`3tnk`, `4tnk`) cost $1150 / $2000** in vendor; `defaults()` says $1500 /
  $1800. Soviet 3tnk is more expensive than allied 2tnk by $300 (not $150).
- **Naval prereqs**: `dd`/`ca` need dome (vendor YAML); `defaults()` mirrors this. `lst`
  has NO `~syrd` prereq in vendor (only `~techlevel.low`) — unlike `defaults()` which gates it.
- **`sub` is NOT in vendor YAML** — the bench's `defaults()` `sub` slug ($950) is a
  duplicate of `ss` and never gets vendor-loaded. Packs that use `type: sub` will fall
  through to the default actor.
- **Weapon binding**: every armed actor's `weapons` field at runtime comes from
  `Armament:` / `Armament@*:` blocks in vendor YAML (e.g. `e3` carries RedEye + Dragon;
  `dd` carries Stinger + DepthCharge + StingerAA — three weapons). The `best_weapon_against`
  selector (`gamerules.rs:663`) picks per target armor class.
- **Sight ranges (vendor)**: jeep `7c0` (7 cells), heli/hind no explicit `RevealsShroud`
  in vendor (engine defaults to 4 cells for units), tsla has `RevealsShroud` tied to
  its attack range. The bench `_CODEX` claims `sight4c` for heli, `sight6c` for 2tnk; the
  vendor doesn't set these explicitly so the engine default of 4 wins for both.

## 9. Engine-data gaps (slugs referenced by packs but not in engine)

Actor types referenced in pack YAMLs but NOT present in engine vendor/defaults:

| Slug | Pack references |
|---|---|

(`e1`/`e3`/etc are in engine — only true gaps are listed above.)

