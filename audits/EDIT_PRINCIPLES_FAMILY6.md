# Family-6 (Build / Tech / Power) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-6.** In particular §9.5
(no-solution-leak) and the §10 map-shrink rule are binding here.
Family-2 §18 (no per-policy income / damage / build-count tables)
applies to every pack in F6 that has a cash budget or harvester
income axis. If anything below conflicts with F1 / F2 §18, the F1 /
F2 §18 rule wins.

The single most important inherited rule is **§7 + §10's map-resizing
clause**: every pack must be classified `fit` / `wide` /
`large-trivial`, and every `wide` / `large-trivial` pack is a backlog
item for the YAML-edit phase. `base_map: rush-hour-arena` on a
build / tech / power pack is **almost always `large-trivial`** —
128×40 is far larger than any single-base build-order decision needs
(the agent never moves out of x≤24 in the intended play), and the
enemy is usually a far-east static `fact` marker placed at (115..125,
20) for the anti-DRAW guard. The build / tech / power test happens
inside a ≤24-cell radius of the agent's `fact`. 16 of the 17 F6 packs
inherit `rush-hour-arena` and only `def-position-expected-direction`-
class custom arenas are `fit`; the rest are `large-trivial` by the
F1 §7 rule of thumb (engagement zone ≤ 30 cells on a 128-cell map ⇒
`large-trivial`).

---

## §37. Build sequence IS the test — validate the FULL chain

Build / tech packs ordering `powr → proc → weap → fix → 2tnk` are
testing the agent's reasoning over a multi-hop prerequisite chain.
The win predicate must validate the **full chain** end-to-end, not
just the end product, or a model that hand-waves the intermediate
steps can latch the chain on partial observation.

Canonical idiom (proven in `build-sequence-tech-fastest`,
`build-sequence-tech-cheapest`, `build-sequence-tech-most-resilient`,
`tech-balanced-econ-then-tech`, `tech-aggro-all-in`):

```yaml
win_condition:
  all_of:
    - then:
        id: <pack-tier>
        clauses:
          - {has_building: powr}    # phase 1: power gate
          - {has_building: proc}    # phase 2: economy gate
          - {has_building: weap}    # phase 3: tech step 1
          - {has_building: dome}    # phase 4: optional tech step 2
    - within_ticks: T
```

Predicate guidance:

- **`then:` happened-before composite is mandatory** for chain
  ordering. A flat `all_of:[has_building:powr, has_building:proc,
  has_building:weap]` accepts ANY order, including one that
  bypasses the engine prereq via a bug. The `then:` chain forbids
  "weap observed first" even if the engine itself accepts it.
- **`has_building` (one-shot ever-seen) is OK in the `then:`
  intermediates** — the predicate need only latch ONCE that powr
  existed. But **the survival floor must use `building_count_gte`
  (live count)** so a destruction (e.g. the scheduled strike in
  `build-sequence-tech-most-resilient`) actually drops the clause
  to false.
- **`then:` plus a `unit_type_count_gte` final clause** is the
  right shape when the chain produces a tank army
  (`build-production-throughput-multibuilding`,
  `build-sequence-tech-most-resilient`).

A `then:` chain that omits an intermediate (e.g. only
`[powr, weap]` skipping `proc`) is a defect — the model can satisfy
the chain by building `weap` while the engine internally satisfies
the `proc` prereq, but the predicate hasn't tested whether the
agent's plan included the `proc` step.

## §38. Parallel production — the multi-factory throughput axis

The engine fix (CLAUDE.md "Multiple production buildings of the same
category produce IN PARALLEL") makes a second `weap` (or `tent`,
`hpad`, `spen`) genuinely double the per-tick output. A pack that
advertises a parallel-production capability MUST set the win bar so
a single-factory baseline misses the quota:

- The quota (`unit_type_count_gte: {2tnk, n: K}`) must be SOLVABLE
  with two factories saturated and UNSOLVABLE with one factory
  saturated, inside `within_ticks`.
- Measure the single-factory ceiling first (e.g. on
  `build-production-throughput-multibuilding`: single weap fields 5
  tanks by tick 2613, fields 6 by tick 3063). Set `within_ticks` IN
  the gap (2613) so K=6 is single-factory LOSS / two-factory WIN.
- The Building queue (fed by `fact`) is NOT parallelised — a 2nd
  `fact` is impossible to build (cost 0; MCV deploy only). Only
  the Infantry / Vehicle / Aircraft / Naval queues parallelise.
- Cash must cover BOTH the second factory AND the quota. A
  cash-starved 2nd-factory play is still a CAPEX trap (see
  `econ-buy-vs-build-decision`); the CSV column `wrong_strategy_loss`
  should call this out where it bites.

## §39. Power discipline — `power_provided_gte` is the anti-cheat floor

The engine fix that pre-placed buildings now contribute to
`PowerManager` totals (CLAUDE.md, see
`tests/test_power_signals_python.py`) opens up two new predicates:

- **`power_surplus_gte: N`** — net (provided − drained). Bites for
  load-shedding (positive surplus required) AND for negative-load
  (overdrawn) packs.
- **`power_provided_gte: N`** — gross (provided only, ignores
  drains). The **anti-cheat floor for load-shedding packs**: a model
  that "fixes" the surplus by powering-down or selling the lone
  `powr` ALSO fails this clause.

Power packs (`build-power-down-defensive`, `build-power-online-first`,
`power-budget-online`, `build-engineer-rebuild-after-loss`,
`build-sequence-tech-most-resilient`) must:

1. **Pair `power_surplus_gte:0` with `power_provided_gte:N`** when
   the test is load-shedding or grid bring-up. Surplus-only is
   gamed by selling the powr (drained → 0 too, surplus = 0,
   "win"). Floor-only is gamed by overbuilding `powr` and ignoring
   drainers (surplus → −∞, "win"). Both together discriminate.
2. **Include `building_count_gte:{drainer, n:1}` for each drainer
   the agent is forbidden to sell.** `power_down` is reversible;
   `sell` is destructive. The drainer-survival floor enforces the
   reversible-only constraint.
3. **NEVER use `has_building:proc`** in a power-discipline pack —
   the destructive `sell` would still satisfy `has_building` (it
   accumulates ever-seen). Use `building_count_gte:{type:proc,n:1}`
   (live count).

`power_down` is a real verb (toggles `World.powered_down`, omits the
building's contribution from BOTH provided and drained totals — so
power-down of a `powr` reduces provided, power-down of a drainer
reduces drained). Surface it in `tools:` for load-shed packs;
surface `sell` alongside as the destructive contrast.

## §40. Sell-and-rebuild uses building actor ids (post engine fix)

The engine fix that surfaces real `id` on every building in
`own_buildings` / `buildings_summary` (CLAUDE.md, see
`tests/test_repair_building_id.py`) makes `repair`, `sell`,
`power_down`, `set_primary` actually targetable by the agent.
Pre-fix, the bench assigned `id = list-index` and `env.rs::
resolve_owned` rejected it. Post-fix, the obs reports the engine
id and the agent can target by id.

Implications for F6 packs:

- **`build-sell-and-rebuild-elsewhere`** depends on the agent
  reading the exposed `proc`'s id from `own_buildings`,
  calling `sell([proc_id])`, and the engine refunding 50% (700cr
  for a 1400cr proc). The pack is unsolvable without the engine fix.
- **`build-repair-priority-under-fire`** depends on the agent
  reading the proc / weap ids from `own_buildings` and calling
  `repair([proc_id])` / `repair([weap_id])` — repairing the WRONG
  id is the documented decoy (the pre-damaged pbox).
- **`build-power-down-defensive`** uses building ids for
  `power_down([drainer_id])` — the agent must read each drainer's
  id from `own_buildings` and shed the right combination. Hard
  tier (NORTH/SOUTH spawn flip) is the real test that ids are read
  dynamically, not memorised by index.

A pack that advertises `repair` / `sell` / `power_down` /
`set_primary` without exposing the matching tool in `base.tools:`
is a defect.

## §41. Build-time budget — ticks-to-build vs `max_turns`

The RA-mod canonical formula is `build_ticks = cost × 60 / 100`
(see `world.rs:71-72,832`, `ProductionItem::new`,
`build_duration_modifier = 60`). At the engine's nominal 30
ticks/second this is `build_seconds = cost × 60 / 100 / 30 =
cost × 0.02`. Non-interrupt-mode packs advance exactly 30 ticks
per `env.step()` (CLAUDE.md), so **1 decision turn ≈ 1 build
second**, and the audit must check that

```
sum(build_seconds of intended chain) ≤ max_turns
```

with comfortable slack. The intended-chain timing measured on the
live engine for the F6 canonical chains:

| Chain | Cost | Build seconds | Build turns |
|---|---|---|---|
| `powr` | 300 | 6s | ~6 turns (~tick 180-273) |
| `proc` | 1400 | 28s | ~28 turns (proc completes by ~tick 1263) |
| `weap` | 2000 | 40s | ~40 turns (weap completes by ~tick 2613) |
| `fix` | 1200 | 24s | ~24 turns |
| `2tnk` | 800 | 16s | each tank ~16 turns from a saturated weap |
| `dome` | 2800 | 56s | ~56 turns |
| `tsla` | 1200 | 24s | ~24 turns |
| `gun` | 600 | 12s | ~12 turns |
| `pbox` | 400 | 8s | ~8 turns |

Critical-path totals for the F6 chains:

- `powr → proc → weap` = 6 + 28 + 40 = **74 seconds ≈ 74 turns**.
- `powr → proc → weap → fix → 2tnk×3` = 6 + 28 + 40 + 24 + 16
  ≈ **114 seconds**. With parallel queues (Building serial,
  Vehicle parallel-saturated) the 3 tanks overlap, so the
  end-to-end is ≈ 114 turns serial, ≈ 100 turns with overlap.
- `powr → proc → weap → tsla` = 6 + 28 + 40 + 24 = **98 seconds**.
- `pbox×8 + gun + weap + dome` (turtle hard) = (8×8) + 12 + 40 +
  56 = **172 seconds** serial. The Defense queue and Building
  queue run independently, so pbox + gun can overlap with weap +
  dome, but the absolute floor is the slower of (defense path)
  vs (tech path).

A pack whose `max_turns × 1 build-second` is below the critical-
path total is a **`build-time-over-budget`** defect (the production-
tech audit flags this). Three F6 packs are currently flagged:

- `tech-turtle-defensive-tech` (98s critical path > 90 turns hard
  ceiling — though parallel Defense/Building queues partially
  rescue it; documented in the production-tech CSV as
  `build-time-over-budget:98s>90t`).
- `tech-aggro-all-in` (82s > 50 turns medium, 37 turns hard).
- `power-budget-online` (98s > 71 turns easy, 67 turns med/hard).
- `build-production-throughput-multibuilding` (48s > 35 turns
  easy, 33 turns med/hard — but the parallel-weap fix DESIGNS
  for this; the override is "two queues saturated = ~24s
  effective", which fits).
- `build-sequence-tech-cheapest` / `-fastest` (66s > 40 / 35
  turn ceiling — flagged but the within_ticks (3200/2800) is
  measured to fit empirically; the flag is conservative against
  the formula).

Cross-check is part of the audit row — use the `production_tech_
audit.csv` `issues` field. A non-empty `issues` is a backlog item
for the YAML-edit phase.

## §42. Tech-gate prereq cross-check (§17 binding)

Inherited from F2 §17. Every buildable advertised by the briefing
or required by the intended-capability play must have the full
prereq chain in `agent.actors` AT THE TIER THE PACK STARTS AT:

| Buildable | Allied Prereq | Soviet Prereq | Cost | Build sec |
|---|---|---|---|---|
| `e1` | `tent` | `barr` | 100 | 5 |
| `e3` | `tent` | `barr` | 300 | 8 |
| `1tnk` | `weap` | `weap` | 700 | 16 |
| `2tnk` | `weap`+`fix` (allies) | n/a | 800 | 18 |
| `3tnk` | n/a (allies) | `weap`+`fix` | 950 | 20 |
| `4tnk` (mammoth) | `weap`+`fix`+`dome` | n/a | 1700 | 30 |
| `harv` | `weap` | `weap` | 1400 | 25 |
| `mcv` | `weap`+`fix` | `weap`+`fix` | 2500 | 40 |
| `proc` | `fact`+`powr` | `fact`+`powr` | 1400 | 28 |
| `powr` | `fact` | `fact` | 300 | 8 |
| `weap` | `proc` | `proc` | 2000 | 40 |
| `fix` | `weap` | `weap` | 1200 | 24 |
| `tent` | `fact` | n/a | 400 | 10 |
| `barr` | n/a | `fact` | 300 | 8 |
| `pbox` | `tent` | n/a | 400 | 8 |
| `gun` | `tent`+`fix` | n/a | 600 | 14 |
| `tsla` | `barr`+`weap` | `barr`+`weap` | 1200 | 24 |
| `dome` | `proc` | `proc` | 2800 | 56 |

Cross-references: `audits/production_tech_audit.csv`. If `issues`
is non-empty for a pack, that's a defect for the YAML-edit phase
(EITHER add the missing building to `agent.actors` OR rewrite the
briefing to not advertise that build OR boost `starting_cash`).

## §43. Faction discipline — Allies vs Soviet roster

Most bench packs are Allies (`agent: {faction: allies}`). The
Soviet-specific F6 packs (`tech-aggro-all-in`,
`tech-production-planning` mid-game tier) use Soviet rosters with
`barr` (not `tent`), `tsla` (Soviet super-defense), `3tnk` (heavy
tank, Soviet medium-tank-equivalent), `4tnk` (mammoth).

A pack whose `agent: {faction: ussr}` (or soviet) advertises
Allied-only units (`2tnk`, `tent`, `pbox`, `gun`) is a defect. The
production-tech audit's `faction` column flags this.

## §44. `fact` is cost 0 — can't be built; MCV deploy only

`fact` (Construction Yard) is cost 0 in the engine, so
`StartProduction` is gated out and the engine silently refuses
`Command.build('fact')` (CLAUDE.md). A pack that advertises "build
a second base / second fact" must EITHER:

- Pre-place a 2nd `fact` (rare; `def-retreat-and-rebuild` does
  this with a forward base and a safe-zone fact at different
  cells), OR
- Grant an `mcv` + expose the `deploy` tool. The `mcv` deploy
  spawns a new `fact` at the MCV's cell (CLAUDE.md
  `tests/test_mcv_deploy.py`).

The `proc` is the "second base seed" idiom in expand-arm
objectives (`econ-second-base-race`, `econ-mine-and-grow`) —
`proc` IS buildable (cost 1400, prereq `fact`+`powr`).

## §45. `deploy` MCV — the only path to a new fact

The engine fix that `deploy` works for scenario-declared MCVs
(CLAUDE.md `tests/test_mcv_deploy.py`) closes the historical
"unimplemented" footgun. Two bugs were both fixed:

- `classify_actor` in `gamerules.rs` used to return `Vehicle`
  for MCV; now returns the deployable class.
- The env.rs `kind_for_unit_type` fallback used to default to
  `Infantry`; now correctly classifies MCV.

A scenario actor `{type: mcv, owner: agent, position: [x,y]}` +
`Command.deploy([mcv_id])` removes the MCV, creates an agent
`fact` at the MCV's cell, and re-enables the Building / Defense
production queues anchored on the new `fact`. The build-radius
follows the new fact. This is the canonical "launch a base from
a single starter MCV" idiom (used implicitly by `spawn_mcvs:
true` packs and explicitly by `def-retreat-and-rebuild`).

## §46. `scheduled_events` for mid-episode strikes (build-engineer recovery)

The Wave-9 `scheduled_events.destroy_actors` hook (CLAUDE.md, see
`tests/test_scheduled_events.py`) is the canonical way to inject
a mid-episode building loss. F6 packs using it:

- **`build-sequence-tech-most-resilient`** — `destroy_actors`
  razes the exposed `powr` at tick 1500, testing N+1 redundancy.
  The redundant powr must be PRE-BUILT (before tick 1500) for the
  win to latch.
- **`build-engineer-rebuild-after-loss`** uses a pre-placed
  adjacent enemy `4tnk` at `stance:3` instead of a scheduled
  event — same effect (the powr dies in the first ~90 ticks)
  but the destruction is in-engine combat, not a script. Either
  is fine; the scheduled-event form is preferred when the strike
  must not be intercept-able by combat (the strike is the test,
  not the defense).

A `then:[not powr, powr]` happened-before latch over the live
`building_count_gte` is the canonical recover-from-loss idiom:
clause A latches on destruction, clause B latches on rebuild.
Stall fails clause A (never destroyed); build-without-noticing-
loss fails clause B (never rebuilt).

## §47. Hard tier spawn-variation contract (CLAUDE.md spawn_point footgun)

Every F6 hard tier must declare ≥2 `spawn_point` groups (typically
NORTH y=12-14 / SOUTH y=26-28) so a memorised opening cannot
generalise. Per CLAUDE.md oramap.rs rules:

- **ANY agent actor with `spawn_point` causes agent actors WITHOUT
  `spawn_point` to be filtered out.** So the FULL agent base
  (fact + tent + powr + proc + ...) must be DUPLICATED across
  both spawn groups at spawn-matched cells.
- **Enemy and neutral actors don't honour `spawn_point`** — they
  always place. Either declare per-spawn enemy compositions (the
  Wave-9 enemy-side rotation idiom) or duplicate the strike at
  both latitudes (the agent-side rotation idiom — what most F6
  hard tiers do).
- **`scheduled_events` filters with `region:` are per-group
  duplicated** — the `destroy_actors` region must be duplicated
  at NORTH and SOUTH (the dormant-latitude region simply removes
  nothing).
- **Inert spawn-witness `e1`** (HoldFire stance + no `move_units`
  / `attack_unit` tool exposed) per spawn group is the hard-tier
  contract — the units_summary spawn-variation test
  (`tests/test_hard_tier.py`) requires a seed-varied agent UNIT in
  `units_summary`. A building-only base produces empty
  units_summary on every seed and breaks the contract.

## §9.5 / §18 — no solution leak (binding from F1 / F2)

Forbidden in F6 briefings:

- **Per-policy outcome tables** ("a single weap fields 5 tanks,
  two weaps field 7"; "the tech play has only powr+proc by the
  deadline"). Cut from the briefing prose; OK in the YAML's
  `# ENGINE NOTE` block for contributors.
- **Build-sequence prescriptions by verb** ("Build powr first,
  then proc, then weap, then 2tnk."). Forbidden. The model must
  derive the chain from the prereq table + the win predicate.
  **The win predicate IS allowed to enumerate the chain** (the
  `then:` clauses) — that's the goal description, not the
  strategy.
- **Cash arithmetic done for the model** ("Cash 3700 covers
  powr+proc+weap with $50 slack"). Forbidden. The model has
  cash + costs in the obs.
- **"Intended play is X" / "The trap is Y" / "Stalling LOSES"**.
  Forbidden. The forces + win predicate + tick budget tell the
  model what to aim at; the rest is up to it.

Allowed:

- The starting state ("Construction Yard, Barracks, Power Plant,
  $1500").
- The win predicate in plain English ("Field six medium tanks
  within about 30 turns").
- Relative-direction landmarks ("an enemy barracks at the
  forward zone (62,20), roughly 38 cells east").
- Constraint reminders that aren't strategy ("no ore, no income"
  is OK because it constrains the search space; "and only the
  cheapest chain wins" is NOT OK).

The F6 packs as-shipped contain a `description: >` field on every
level — these are the briefings the model sees. The lengthy `#
Original (pre-verbosity-sweep) description preserved for
contributors` block in many YAMLs IS allowed (it's a comment, not
shown to the model). Audit the `description: >` field, not the
preserved comment.

## §48. Engine auto-`done` mitigation — persistent enemy fact

Every F6 pack must include a persistent unarmed enemy `fact` far
from the agent (canonical: `(120,20)` or `(115,30)`) — see
CLAUDE.md auto-done footgun. The agent has no offensive tools in
most F6 packs (build-only / repair-only / sell-only palettes), so
the enemy `fact` cannot be destroyed and the episode runs to the
deadline. WITHOUT this marker, the engine auto-`done`s on
all-enemies-eliminated (a stray sentry kill collapses the chain),
collapsing every non-winning play to DRAW (F1 §5 violation).

Hard tiers must duplicate the enemy `fact` across both spawn
groups OR rely on the no-spawn-point rule (enemy actors don't
honour spawn_point — the single marker places every seed). The
single-marker form is simpler and used by every F6 hard tier.

---

## Family-6 audit CSV column contract

`audits/family6_build_tech.csv` extends F1 + F2 columns with
build-specific columns:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget |
starting_cash | tech_chain | build_time_check | leak_flag |
production_audit_issue
```

- `starting_cash` — int, the cash the agent starts with at this
  tier.
- `tech_chain` — one short sentence: the intended buildables this
  tier requires, as a comma-list with prereqs noted ("powr → proc
  → weap, 6× 2tnk").
- `build_time_check` — `ok` if the critical-path build-time
  estimate (§41) fits `max_turns × 1s` with slack, otherwise the
  diff ("66s > 40 turns — flagged in production_tech_audit").
- `leak_flag` — `clean` / `verb-prescription` / `arithmetic` /
  `policy-table` per F1 §9.5 + F2 §18. Set when the `description:
  >` field contains a forbidden phrase.
- `production_audit_issue` — verbatim from
  `audits/production_tech_audit.csv` `issues` column, or empty.

Same `map_fit` discipline as F1: every `wide` / `large-trivial`
row is a backlog item for the YAML-edit phase to shrink to a
bespoke arena. The build/tech/power decision usually only needs
a ≤30-cell radius around the agent's base; the canonical custom
arena is 64×40 (a fit) or 80×40 (a wide).
