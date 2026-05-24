# Family-8 (Multi-front base / MCV / Coordination) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-8.** F2 §18 (no solution
leak) inherits as well, because multi-base / MCV / coordination
briefings are EXACTLY the place where a briefing slides from "describe
the situation" into "name every base, name every verb, name the
intended split" — the temptation to dump the whole plan is highest in
this family. F3 §22 (`scheduled_events` for timed pressure) and F3
§24-26 (build prereqs, map-shrink, two-spawn-point hard tiers) inherit
for the MFB packs that have a defense or build-under-pressure axis.
If anything below conflicts with F1 principles, F1 wins (briefing /
map / win-readout conventions stay identical across families).

The single most important inherited discipline is still **F1 §7 +
§10's map-resizing clause**, but Family-8 introduces a meaningful
exception: a 160x60 / 160x80 / 128x64 arena CAN be `wide-justified`
when the inter-base distance IS the test (deploy a 2nd / 3rd base 100+
cells from base #1; mirror a coordinated build across a 70-cell span;
hold a corridor midpoint 35 cells from each base). The audit must
distinguish:

- **wide-justified** — the large map encodes the multi-front geometry
  itself. Examples: `mfb-base-1-defend-base-2-build` (base #1 at x=15,
  target region at x=130 — the 115-cell MCV drive IS the secure-expand
  budget); `mcv-deploy-third-base` (160x80 — three target regions
  fanned across two latitudes); `mfb-two-base-simultaneous` (two
  yards 70 cells apart — the parallel-queue capacity test).
- **wide** — significant empty traversal, but the decision still bites
  (some coord-* packs on `rush-hour-arena` 128x40 — engagement at
  x=50-60 with 6-10 cells of pre-engagement march).
- **large-trivial** — the multi-front geometry doesn't bite hard, but
  the agent still has to march 30+ cells of empty ground before any
  decision matters. Treat as `wide` only if the multi-front element
  is genuinely load-bearing; otherwise demote.

---

## §48. Multi-front base management

MFB packs (`mfb-*`) require attention-splitting across N≥2 bases.
Every base in a pack must have a **load-bearing role** — a base
whose loss / inactivity doesn't change the win/lose outcome is
decoration, and the pack collapses to a single-base test.

Per-pack contract:

- Name each base's role (TECH / ECON / DEFENSE / EXPANSION /
  STANDBY). If two bases share the same role, justify why
  duplication is the test (e.g. `mfb-redundant-tech-buildings` —
  the test IS standing up the second weap so the first can fall;
  `mfb-rotating-production-pressure` — primary toggle is the test).
- The win predicate must touch ≥2 bases. Patterns:
  - `building_in_region: {x: <west>}` AND `building_in_region:
    {x: <east>}` — both bases must end with the right building
    placed (`mfb-mirror-base-east-west`, `mfb-two-base-simultaneous`,
    `mfb-tech-base-vs-economy-base`, `mfb-third-base-against-clock`).
  - `building_count_gte: {type: fact, n: 2}` + region clause for
    the new base location (`mfb-base-1-defend-base-2-build`).
  - `units_killed_gte` + corridor-occupancy region clause to force
    a unit onto the inter-base corridor (`mfb-supply-line-link-between-bases`).
- A single-base policy must demonstrably LOSE. Specifically:
  - "All-defend / all-tech" (ignore expansion) — the expansion
    clause never fires → deadline LOSS.
  - "All-expand" (ignore defense) — primary base falls →
    `not has_building:fact` LOSS.
  - "Split too thin" — neither side completes inside the clock →
    deadline LOSS.
- Scripted-policy validation: a brute single-base policy MUST hit
  the deadline / asset-loss fail clause. If the pack's
  `fail_condition` lacks a base-loss clause AND lacks a tight
  enough `after_ticks`, the single-base play wins by accident and
  the pack is a stall/single-base degeneracy.

## §49. MCV deploy is a real load-bearing verb

`deploy` is no longer the historical "unimplemented" footgun — the
Wave-2 fix means a scenario-declared `{type: mcv}` actor becomes a
buildable agent `fact` on `Command.deploy([mcv_id])` and re-enables
the Building / Defense queues. Pinned by
`tests/test_mcv_deploy.py` (Python) + the engine MCV-deploy fix
described in CLAUDE.md. MCV packs (`mcv-*` and MFB packs that
include an MCV) must make the deploy decision load-bearing in one
of the following ways:

- **Site choice** — `building_in_region: {type: fact, x, y, radius}`
  on the target region. A deploy in the wrong region satisfies
  `has_building: fact` but not the region clause. Examples:
  `mcv-deploy-near-resource` (deploy adjacent to the ore patch),
  `mcv-deploy-defensible-site` (deploy in the terrain-shielded west
  pocket), `mcv-deploy-second-base` (deploy in the eastern
  expansion disc), `mcv-deploy-third-base` (three required region
  clauses).
- **Timing** — `after_ticks` LOSS gate that fires if the MCV is
  still alive (undeployed) past tick N. Examples:
  `mcv-deploy-and-build` (`after_ticks: 2000 AND not has_building:
  fact` → if you didn't deploy in time, LOSS).
- **Relocation under pressure** — pre-placed MCV is in a death
  position (central lane under a `rusher` band). The deploy MUST
  happen at a different cell. Examples:
  `mcv-deploy-relocate-under-pressure` (MCV at (60,20) lane centre;
  must move to (60,8) or (60,32) shoulder before deploying).

A `mcv-*` pack whose win is satisfied by deploying ANYWHERE on the
map is a defective site-choice test. Cross-check: every `mcv-*`
pack's win must include either a `building_in_region` constraint
OR a `<region-AND-time>` constraint that punishes wrong placement.

The engine still does NOT auto-snap an MCV's deploy site to a
clean cell — a deploy on impassable terrain or inside an obstacle
silently dies. Pre-validate target regions are clear of obstacles
on the procedural arena `obstacles:` list (the `mcv-deploy-and-build`
arena has six obstacle pads pushed to the corners exactly so the
deploy disc stays clean).

## §50. Coordination idioms

Coord packs (`coord-*` / `coordination-*`) test multi-unit
synchronisation. Win predicates must require ALL units (or all
required squads) to arrive, not just one.

Allowed predicate shapes:

- **Joint-arrival threshold** — `units_in_region_gte: {x, y,
  radius, n: K}` where K is the SUM of every dispatchable squad.
  A single-squad sortie delivers `K/squad_count` units → predicate
  unsatisfied. Examples: `coord-converge-on-target` (n=9 across
  three 3-tank squads), `coord-mutual-support` (n=5 with a 6-tank
  squad), `coordination-staggered-window` (n=3 at each of 2-3
  docks simultaneously).
- **Sequenced `then:` clauses** — `then: {clauses: [...]}`
  enforces ORDER. A single squad cannot satisfy two type-filtered
  clauses (e.g. `units_of_type_in_region_gte: jeep` then
  `units_of_type_in_region_gte: 2tnk`). Examples:
  `coord-relay-attack` (kill enemy tanks FIRST, then mop up
  infantry), `coord-squad-handoff` (jeeps to P1, then tanks to P2,
  then jeeps to P3), `coordination-ordered-rendezvous`
  (waypoint_sequence with 2-4 ordered stops).
- **Multi-region simultaneity** — multiple `units_in_region_gte`
  clauses each requiring `n≥3` at separate (x, y) coordinates.
  Single-column tours fail because by the time the column reaches
  region B, region A has emptied. Example:
  `coord-relay-vision-chain` (3-4 relay regions each requiring 1
  jeep — exactly four jeeps means every region needs exactly one
  scout, with no slack).

A coord pack whose win is `units_in_region_gte: n=1` at a single
location is NOT testing coordination — it's a path test.
Multi-unit synchronisation requires either a joint threshold
(n ≥ squad_count) OR multiple simultaneous regions.

Footgun: an `all_of: [units_in_region_gte A, units_in_region_gte
B]` MUST evaluate at the SAME tick (the engine evaluates the win
predicate once per tick — a one-tick window where both are true
suffices). The `then:` clauses latch in order so a region A
visit at t=300 + a region B visit at t=900 satisfies the chain.
For TRUE simultaneity (the staggered-window / supply-line
midpoint idiom), use plain `all_of:` with two `units_in_region_gte`
clauses inside a single `within_ticks` window.

## §51. Per-base economy and the auto-routing harvester defect

Multi-base packs that include economy (`harv` + `proc`) have a
specific failure mode: a single auto-routing harvester defeats
the multi-base test. The engine binds a NEW harvester (auto-spawned
by `place_building('proc')`) to the NEAREST refinery by path
distance, BUT existing harvesters do NOT re-snap (CLAUDE.md
engine note). Implications:

- If only ONE harvester exists across two bases and it's spawned at
  base #1's refinery, building a refinery at base #2 produces zero
  income (no harvester routes there). The "second base" is
  decorative.
- The fix is to either (a) supply each base with its own
  pre-placed `harv` + `proc`, OR (b) seed the auto-routing case
  on top of the multi-base requirement (each base must place its
  OWN `proc`, which auto-spawns its own harvester).
- A pack that pre-places one harv and one proc at base #1 and
  asks the model to "expand to base #2" must EITHER (a) require
  the model to also build a 2nd harv via `weap`, OR (b) accept
  that base #2 is a placement / topology test (not an economy
  test) — and the win predicate must reflect that (no
  `economy_value_gte` clause).
- `mcv-deploy-near-resource` solves this cleanly by pre-placing
  proc+harv at base #1 and using `economy_value_gte` ONLY (the
  new MCV-deployed yard at the ore patch doesn't have to itself
  produce income — the existing harv keeps depositing while the
  new yard just satisfies `building_in_region`). The harvester
  auto-routes back to the original proc; the model's deploy site
  only matters for the topology clause.
- `mfb-mirror-base-east-west` and `mfb-two-base-simultaneous`
  test PLACEMENT of mirror refineries, not income — so the
  auto-routing harv defect doesn't apply (no `economy_value_gte`).
- `mfb-tech-base-vs-economy-base` is the diagnostic case: it
  requires `economy_value_gte: 2500` AND `unit_type_count_gte:
  harv:2`. Without the harv:2 clause a single auto-routing harv
  at the wrong-side refinery could clear the income bar on the
  starter ore patch; the harv:2 clause is the anti-cheat that
  makes the SE-side build genuinely load-bearing.

**Audit flag**: any multi-base pack with `economy_value_gte` but
no `unit_type_count_gte: harv` (or pre-placed-harv-per-base) is
a candidate auto-routing defect — surface it as a CSV note.

## §52. Supply-line / inter-base packs

`mfb-supply-line-link-between-bases` (and any future inter-base
escort pack) tests inter-base resource shuttling / interdiction
along the corridor between two bases. The engine does NOT model
explicit supply orders (no `escort` verb, no protected-route
shape), so the test must be expressed via:

- **Corridor occupancy** — `units_in_region_gte: {x: <midpoint>,
  y: <corridor-latitude>, radius, n: 1}` to force the agent to
  PARK a unit on the corridor midpoint (interdict-anything
  semantics). Coupled with raiders that probe FROM the corridor's
  flanks: `scheduled_events: spawn_actors` (or just static
  `bot_type: rusher` raid bands) to make the corridor
  occupancy bite.
- **Both-bases-alive** — `building_count_gte: {type: fact, n: 2}`
  + `building_count_gte: {type: proc, n: 2}` to require both
  bases survive (proxy for "supply line preserved").
- **Kill quota** — `units_killed_gte: N` so the corridor garrison
  must actively engage raiders (not just sit on the cell).

A "supply line" pack whose win is ONLY both-bases-alive is a
double-defense pack, not a supply-line pack. The corridor
occupancy clause is what makes "the link between the bases" the
test. The engine does not currently support a literal supply-line
shape (escort orders, protect-the-route flag) — the corridor
occupancy idiom above is the closest expressible approximation.

If a pack genuinely needs literal supply-line mechanics (e.g. a
periodic resource truck that has to be escorted between bases)
that is an **engine-gap candidate** — add a `convoy_route`
predicate or a `scheduled_events: spawn_actors` + region clause
on the spawned actor, and flag for engine work.

## §53. MCV deploy disc must be clear of obstacles

A target deploy region for `mcv-*` packs is a `building_in_region`
predicate centred at `(x, y, radius)`. The agent must drive the MCV
INSIDE that disc, then call `deploy`. If the disc overlaps an
arena `obstacles:` rectangle the deploy silently fails (engine
places the new `fact` on an open cell — but if no open cell is
inside the disc, the deploy lands OUTSIDE the disc and the
`building_in_region` clause never satisfies).

Pre-flight check during audit: read the arena `obstacles:` list
and confirm every win-clause `building_in_region` disc has ≥1
clear cell within radius. The `mcv-deploy-and-build` arena has
obstacles at (30,4), (30,32), (50,10), (50,26), (70,15), (70,23)
— pushed to the corners exactly so the deploy footprint stays
clean. Verify before authoring against any new arena.

## §54. Stall and lazy policies must lose

(F1 §10 binding, restated for emphasis with Family-8 specifics.)

Coordination / MFB packs are the family most prone to the stall
degeneracy because the win is often expressed as positional
("be in region X"). If the agent's pre-placed units are already
inside the region at t=0 or are auto-pulled by a hunt bot into
the region, the stall policy WINS. Required guards:

- For `units_in_region_gte` packs, the staging position must be
  ≥10 cells from the target region. Verify in audit.
- For `building_in_region` packs, the target region must require
  an explicit `build` / `deploy` / `place_building` — the agent
  must NOT start with a `fact` (or whichever building type) in
  the region.
- For `then:` sequenced clauses, the FIRST clause must not be
  trivially satisfied (no pre-placed units already inside region 1).
- For `units_killed_gte` clauses, the enemy must be reachable
  ONLY by the agent's deliberate engagement order (no enemies
  that walk into the agent's idle defenders for free, unless a
  parallel building clause prevents the stall from winning that
  way).

The intended policy must dominate every stall / lazy / single-base
/ single-squad / brute-charge policy on every level + every hard
seed (1-4). The F1 no-cheat bar is binding.

---

## Family-8 audit CSV column contract

`audits/family8_multifront.csv` uses the F1 column set:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget
```

`map_fit` values:

- `fit` — engagement and decision are commensurate with the map.
- `wide-justified` — large map (≥120 cells wide or ≥60 cells tall),
  but the multi-front geometry encodes the test (inter-base
  distance, multi-region simultaneity).
- `wide` — large map, decision still bites but with significant
  empty traversal.
- `large-trivial` — most of the map is empty pre-engagement; the
  multi-front aspect is decorative.

Tick budget convention: `tick_budget = max_turns × 90 + 3`
(historical F1 convention; non-interrupt-mode actual is 30
ticks/step, but interrupt-mode packs and the audit-row contract
keep the 90 ticks/turn figure for consistency with prior families).

Same `map_fit` discipline as F1: `wide-justified` rows DO NOT
require shrinking (the geometry IS the test); `wide` and
`large-trivial` rows are backlog items for the YAML-edit phase.

Cross-cutting audit checks (run per pack, surface as a row note
if any fire):

1. **No-leak audit** — does the briefing name the winning verb
   per region ("send jeeps south, tanks north"), the deploy site
   explicitly ("deploy at (60,12)"), or the build order? F2 §18
   forbids these.
2. **Multi-base load-bearing** — does the win predicate touch ≥2
   bases (MFB) or ≥squad-count units (coord)? §48 / §50.
3. **MCV decision load-bearing** — does the win predicate punish
   wrong-site / no-deploy? §49.
4. **Auto-routing harv** — does the pack mix `economy_value_gte`
   with a single pre-placed harv? §51.
5. **Stall sanity** — is the agent's staging ≥10 cells from every
   `units_in_region_gte` target? §54.
6. **Tick budget** — `within_ticks ≤ 93 + 90·(max_turns − 1)`
   (F1 §5). Note as `inert-deadline` if the bar can't fire.
7. **Tech-gate cross-check** — for MFB packs with `build` /
   `place_building`, are the prereq buildings present? F2 §17 +
   F3 §24.
8. **Spawn-point hard tier** — does the hard tier declare ≥2
   spawn_point groups, with persistent agent actors duplicated
   across both groups? F3 §26 / CLAUDE.md spawn-filter footgun.
