# OpenRA-Bench — Scenario Brainstorm & Economy Family Spec

Design document. No engine or existing-pack changes. Two runnable
economy packs were drafted and validate OK (see end).

All win conditions below use ONLY leaves/composites that exist in
`win_conditions.py`: `explored_pct_gte, enemies_discovered_gte,
buildings_discovered_gte, units_killed_gte, units_lost_lte,
within_ticks, after_ticks, reach_region, all_units_in_region,
own_units_gte, cash_gte, harvesters_gte, power_surplus_gte,
has_building, buildings_owned_gte, building_total_gte,
building_count_gte{type,n}, building_in_region{type?,x,y,radius,count}`;
composites `all_of / any_of / not`.

---

## Verified ground-truth constraints (this drove the design)

1. **No schema-valid ore source.** The Rust engine seeds an ore patch
   in a ~4-cell radius around any actor of type `mine`
   (`openra-sim/src/world.rs` ~L4014, `map_actor.actor_type == "mine"`
   at ~L3908). **But** the Python `ScenarioDefinition` actor whitelist
   (`openra_rl_training.scenario.VALID_ACTOR_TYPES`) has **no `mine` /
   `gmine`**, so a pack that spawns one fails validation. The 5
   `.oramap` files (`map.yaml` inspected) embed **no resource tiles**
   either — only `mpspawn` actors.
   ⇒ **Harvest income is 0 in any runnable pack today.** `harv` + `proc`
   spawn fine but find no ore. `cash_gte` is only satisfiable from
   `starting_cash`, and `harvesters_gte` only counts the auto-spawned
   harvester (it never earns).
2. **`silo` is schema-valid but inert.** `silo` is in
   `VALID_ACTOR_TYPES`, but `PlayerResources.resource_capacity` is
   hardcoded `0` (`openra-sim/src/traits/mod.rs:76,172`,
   `world.rs:3609`) and the deposit path converts ore straight to cash
   ignoring capacity (`world.rs:1467-1473`). Building silos has **no
   economic effect**.
3. **`starting_cash` works** (per-level / pack-wide, schema confirmed),
   build/train with tech-tree prerequisites works, `proc` auto-spawns
   one `harv` on creation (`world.rs:1154`).
4. Only `rush-hour-arena` is Rust-loadable today (CONTRIBUTING §5);
   the four `singles-*` maps are schema-valid but Phase-3-gated.

**Design consequence:** runnable economy scenarios test the
**spend / allocation** decision over a fixed budget (the part the lead
identified as the real test). Harvest-throughput and silo-storage
variants are fully specced but tagged engine-prereq.

---

# 1. Requested family — Economy-Time-Box  (RUNNABLE TODAY)

**Pack:** `packs/economy-time-box.yaml` — validates **OK**.
**Capability:** reasoning.
**Decision:** within a fixed tick budget, convert a fixed cash budget
into BOTH deployed units AND a standing economy; idle cash and
all-in-on-units both fail.

real_world_meaning — Time-bounded capital deployment: a finite,
non-replenishing budget must become mission capability *and* persistent
infrastructure before a hard deadline.
robotics_analogue — A field robot on a finite energy budget scheduling
expenditure between mission work and standing up persistent
infrastructure before a deadline.

| Level | starting_cash | win_condition (real grammar) | scaling axis |
|---|---|---|---|
| easy | 4000 | `all_of[own_units_gte:4, building_total_gte:4, within_ticks:22000]` | loose clock & budget |
| medium | 2400 | `all_of[own_units_gte:5, has_building:tent, building_total_gte:4, within_ticks:16000]` | adds tech-dep building, tighter |
| hard | 1500 | `all_of[own_units_gte:6, building_total_gte:5, units_lost_lte:0, within_ticks:12000]` | lean budget, short clock, attrition cap |

Difficulty comes from the decision getting harder (less budget slack,
a tech-dependency constraint, an attrition cap + short clock), not
bigger target numbers. Predicates: `own_units_gte`, `building_total_gte`,
`has_building`, `units_lost_lte`, `within_ticks`. Commands: `build`,
`place_building`, `move_units`, `deploy`, `stop`. Map: `rush-hour-arena`.

**Engine status: runnable now.**

### Follow-up (leaderboard scoring, not a predicate today)
A *final-economy-value* score = `cash + Σ asset_value(owned units &
buildings)` at the deadline would reward "spent it well" beyond the
binary `cash_gte`. No such predicate exists; propose it as a scoring /
leaderboard metric computed from the render state at episode end (sum
buildable costs of `own_buildings` + `units_summary`, add `signals.cash`).
Do **not** add it as a win leaf — pass/fail stays in the grammar above.

### Harvest-income variant — **needs S0 + S1** (see §Engine-prereqs)
Same shape but income comes from harvesting and `cash_gte` rises over
time: `all_of[cash_gte:N, own_units_gte:K, harvesters_gte:1,
within_ticks:T]`. Requires S0 (schema-valid ore source so harvesting
actually earns) — and S1 only if silo storage is part of the test.

---

# 2. Requested family — Economy-Investment  (allocation decision)

**Pack:** `packs/economy-investment.yaml` — validates **OK** (runnable
variant).

real_world_meaning — Capital allocation under one indivisible budget:
the same cash buys either a *wide* economy (more depot/throughput
capacity) or a *deep* one (more collectors/force); the budget covers
exactly one coherent path, so splitting builds neither.
robotics_analogue — Fleet capital allocation: a fixed budget buying
either added depot/processing capacity or added collector agents — the
throughput-vs-collection trade, where indecision yields neither.

### (a) Runnable-today variant — RUNNABLE NOW
Income is the fixed `starting_cash`; the allocation is verified by
build predicates (no silo, no harvest dependency).

| Level | starting_cash | win_condition (real grammar) | path tested |
|---|---|---|---|
| easy | 4500 | `all_of[building_count_gte{proc,2}, building_count_gte{powr,2}, within_ticks:22000]` | wide, loose |
| medium | 3200 | `all_of[building_count_gte{proc,2}, building_count_gte{powr,2}, own_units_gte:3, within_ticks:16000]` | wide + utilisation, tighter |
| hard | 2600 | `all_of[own_units_gte:6, building_count_gte{proc,1}, units_lost_lte:0, within_ticks:14000]` | deep (force) under lean budget + attrition |

Difficulty escalates by removing budget slack so the allocation must be
committed (a split budget clears no bar), tightening the clock, and
adding an attrition cap. Predicates: `building_count_gte{type,n}`,
`own_units_gte`, `units_lost_lte`, `within_ticks`. Commands as above.
Map `rush-hour-arena`.

### (b) Full version — **ENGINE-PREREQ (needs S0 + S1)**
The faithful test the lead asked for: enough cash for **two refineries
OR one refinery + extra harvesters**, scored by **harvested cash at the
deadline**, plus **silos that raise the harvest cap**.

- Levels: easy = budget covers either path comfortably (commit to one);
  medium = budget covers exactly one (the bottleneck differs: 2×proc =
  unload throughput, 1×proc+harv = gather rate); hard = lean budget +
  short clock + silo decision (must build `silo` to avoid capping the
  resource pool before a return trip).
- Win grammar (valid leaves, *meaningful only after S0/S1*):
  `all_of[cash_gte:N, harvesters_gte:H, within_ticks:T]` for the
  allocation; the silo sub-goal is observable as
  `building_count_gte{type:silo,n:K}` but only *matters* after S1.
- **Engine status: needs S0 (income) + S1 (storage).** Without them the
  two allocations are economically identical (income 0) so the
  scenario is not discriminating — hence variant (a) ships instead.

---

## Engine-prerequisite specs

### S0 — schema-valid ore source (smaller, unblocks all harvest income)
Today the only ore mechanism is `actor_type == "mine"`, which the
Python schema rejects. Two non-engine-code options the project could
take (documented here, not implemented):
- Add `mine`/`gmine` to `VALID_ACTOR_TYPES` in
  `openra_rl_training/scenario.py` (one list entry; the Rust seed loop
  at `world.rs:4014` already handles it), **or**
- Add a `resource_fields: [{x,y,radius,density}]` field to
  `ScenarioDefinition` plumbed to `terrain.set_resource(...)`
  (`openra-sim/src/terrain.rs:193`) at world init.
Either makes harvest income real and unblocks the §1 harvest variant
and §2(b) allocation scoring (without silos).

### S1 — silo / resource-storage modeling (the part to flag)
Goal: harvested resource enters a **capped pool**; cash is realised
from the pool gradually (or on cap pressure), and `silo` buildings
raise the cap. Precise change (Rust):

1. **`openra-sim/src/traits/mod.rs`** — `TraitState::PlayerResources {
   cash, resources, resource_capacity }` already has the fields; stop
   hardcoding `resource_capacity: 0` at the two constructor sites
   (`mod.rs:172`, `world.rs:3609`). Give players a small base capacity
   (e.g. from `fact`/`proc`).
2. **`openra-sim/src/world.rs` deposit path (~L1467-1473, `Unloading`
   arm of `tick_harvesters`)** — instead of `player.set_cash(current +
   value)`, add to `PlayerResources.resources`, clamped to
   `resource_capacity`; overflow is lost (matches OpenRA). Add a
   per-tick drain that converts `resources → cash` at a fixed rate
   (the realisation of stored ore).
3. **Silo capacity contribution** — when an actor of type `silo`
   (and `proc`, in real OpenRA) becomes owned/active, add its
   `Storage` amount to the owning player's `resource_capacity`;
   subtract on loss. Implement as a recompute over owned buildings in
   the building add/remove path (mirror how `PowerManager`
   provided/drained is recomputed), reading a per-building storage
   constant (silo ≈ 3000, proc ≈ 2000 in RA defaults).
4. **(optional) win/score** — no `resources_gte` predicate exists;
   either add one to `_PREDICATES` in `win_conditions.py` (one entry,
   pure: `c.signals.resources >= int(v)`) and expose `resources` on
   `EpisodeSignals`, or keep pass/fail on `cash_gte` and surface
   stored-value only in the leaderboard score.

With S1, building silos genuinely changes optimal play (don't cap;
don't over-invest in storage you can't fill), making §2(b)
discriminating.

---

## 3. Broader brainstorm — 13 scenario ideas across capabilities

Each: capability · real_world_meaning / robotics_analogue · win
(real grammar) · predicates+commands · map · engine status.
None duplicate the existing 13 packs.

### Perception

**P1 — Resource-Patch Discovery Under Fog**
- perception. RWM: locating *where* to forage before committing
  collectors is the real foraging problem, not the path to a known
  patch. RA: SLAM-time frontier selection for a resource map.
- win: `all_of[explored_pct_gte:55, within_ticks:9000]`
- uses `explored_pct_gte`,`within_ticks`; `move_units`,`observe`,`stop`.
  Map `rush-hour-arena`.
- engine status: runnable now.

**P2 — Count-the-Threats Before Acting**
- perception. RWM: estimating force size from partial sightings before
  engaging is a recon-quality problem distinct from combat. RA:
  multi-target detection/enumeration under occlusion.
- win: `all_of[enemies_discovered_gte:N, units_lost_lte:0,
  within_ticks:T]` (levels raise N, add decoy idle units, tighten T).
- uses `enemies_discovered_gte`,`units_lost_lte`,`within_ticks`;
  `move_units`,`observe`. Map `rush-hour-arena`.
- engine status: runnable now.

**P3 — Identify the Production Building (signature ID)**
- perception. RWM: classifying an installation by observable signature
  before acting on it. RA: semantic recognition of a target structure
  from sensor returns.
- win: `all_of[buildings_discovered_gte:1, within_ticks:T]`; hard adds
  decoy buildings + tighter clock.
- uses `buildings_discovered_gte`,`within_ticks`;
  `move_units`,`observe`. Map `rush-hour-arena`.
- engine status: runnable now.

### Reasoning

**R1 — Tech-Tree Critical Path**
- reasoning. RWM: ordering prerequisite-coupled tasks (B needs power
  needs ...) to hit a deadline is scheduling, not building. RA:
  task-graph scheduling with hard precedence + a makespan deadline.
- win: `all_of[has_building:tent, building_total_gte:4,
  power_surplus_gte:0, within_ticks:T]`; levels shorten T, raise depth.
- uses `has_building`,`building_total_gte`,`power_surplus_gte`,
  `within_ticks`; `build`,`place_building`,`deploy`. `rush-hour-arena`.
- engine status: runnable now.

**R2 — Power-Budget Brownout Avoidance**
- reasoning. RWM: keeping a power budget non-negative while expanding —
  an online resource-constraint problem. RA: a robot keeping net power
  ≥ 0 while bringing subsystems online.
- win: `all_of[power_surplus_gte:0, building_total_gte:N,
  within_ticks:T]`; hard raises N with scarce power slack.
- uses `power_surplus_gte`,`building_total_gte`,`within_ticks`;
  `build`,`place_building`,`power_down`. `rush-hour-arena`.
- engine status: runnable now.

**R3 — Defensive-Direction Commitment**
- reasoning. RWM: committing limited defenses to the *right* approach
  under directional uncertainty. RA: sensor/effector placement toward
  the most likely intrusion bearing.
- win: `all_of[building_in_region{type:pbox,x:40,y:20,radius:9,
  count:2}, within_ticks:T]`; levels move/duplicate the threat axis.
- uses `building_in_region`,`within_ticks`;
  `build`,`place_building`. `rush-hour-arena`.
- engine status: runnable now.

**R4 — Economy-Time-Box** (§1) — reasoning — engine status: runnable now.

**R5 — Economy-Investment (a)** (§2a) — reasoning — runnable now;
full (b) **needs S0 + S1**.

### Action

**A1 — Synchronised Multi-Squad Arrival**
- action. RWM: executing a plan so dispersed agents *converge*
  simultaneously — coordination/execution, not planning. RA:
  multi-robot rendezvous with a timing constraint.
- win: `all_of[all_units_in_region{x:90,y:20,radius:6},
  units_lost_lte:0, within_ticks:T]`; levels split start positions
  wider, tighten T.
- uses `all_units_in_region`,`units_lost_lte`,`within_ticks`;
  `move_units`,`attack_move`,`stop`. `rush-hour-arena`.
- engine status: runnable now.

**A2 — Sequenced Build-then-Relocate**
- action. RWM: executing an ordered build→deploy→reposition sequence
  without dropping a step under a clock. RA: a fixed manipulation
  sequence (assemble → carry → place) executed to deadline.
- win: `all_of[building_in_region{x:60,y:20,radius:10,count:2},
  units_lost_lte:1, within_ticks:T]`.
- uses `building_in_region`,`units_lost_lte`,`within_ticks`;
  `move_units`,`deploy`,`build`,`place_building`. `rush-hour-arena`.
- engine status: runnable now.

**A3 — Harvester Re-tasking Under Interruption**
- action. RWM: keeping a foraging loop alive by promptly re-tasking
  collectors when the route is disrupted. RA: closed-loop recovery of a
  collection cycle after an interruption.
- win: `all_of[harvesters_gte:1, cash_gte:N, within_ticks:T]` — only
  meaningful once harvesting earns.
- uses `harvesters_gte`,`cash_gte`,`within_ticks`;
  `harvest`,`move_units`,`stop`. `rush-hour-arena`.
- engine status: **needs S0** (no schema-valid ore source today).

### Economy / Tech / Multi-objective

**E1 — Economy-Time-Box harvest variant** — see §1 — **needs S0**
(+ S1 if silos are in the test).

**E2 — Economy-Investment full** — see §2(b) — **needs S0 + S1**.

**M1 — Earn-and-Survive (multi-objective)**
- reasoning (multi-objective). RWM: holding an income loop *and* a
  defensive constraint simultaneously — competing objectives under one
  budget. RA: a robot foraging while maintaining a safety constraint.
- win: `all_of[cash_gte:N, units_lost_lte:0, building_total_gte:K,
  within_ticks:T]` (the `cash_gte` arm is real income only post-S0;
  pre-S0 it degenerates to a spend test like §1).
- uses `cash_gte`,`units_lost_lte`,`building_total_gte`,`within_ticks`;
  `harvest`,`build`,`move_units`. `rush-hour-arena`.
- engine status: spend-only form runnable now; income form **needs S0**.

---

## Summary

- **Runnable now:** P1, P2, P3, R1, R2, R3, R4 (economy-time-box),
  R5/§2a (economy-investment runnable variant), A1, A2, and the
  spend-only form of M1 — all expressible in the existing grammar on
  `rush-hour-arena`.
- **Engine-prereq S0** (schema-valid ore source — a one-line
  whitelist add or a `resource_fields` scenario field; the Rust seed
  loop already exists): unblocks any *harvest income* scenario — A3,
  E1, the income form of M1, and the §2(b) allocation scoring.
- **Engine-prereq S1** (silo storage: stop hardcoding
  `resource_capacity`, route deposits through a capped `resources`
  pool in `tick_harvesters`, recompute capacity from owned
  `silo`/`proc` buildings, optional `resources_gte` predicate):
  required only for the *silo* mechanic in §2(b).
- **Final-economy-value scoring** (cash + asset value at deadline) is a
  desirable leaderboard metric, NOT a win predicate — there is no such
  leaf and one should not be assumed.

### Files written
- `openra_bench/scenarios/packs/economy-time-box.yaml` — validator: `OK`
- `openra_bench/scenarios/packs/economy-investment.yaml` — validator: `OK`
- `SCENARIO_BRAINSTORM.md` (this file)
