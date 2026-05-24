# Family-4 (Scout / Perception / Navigation) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-4, INCLUDING §9.5 (no
solution leak) and §10 (map-shrink for `wide` / `large-trivial`).**

It also INHERITS `audits/EDIT_PRINCIPLES_FAMILY2.md` §18 (no-leak — the
F2 generalisation of the rule). The Family-2 economy-specific rules
(§11-17) do not inherit by default; they are referenced only when a
perception pack carries an economy axis (e.g. `scout-jeep-vs-infantry-cost-effective`'s build-the-right-scout decision, or
`scout-count-defenders`'s scout-then-size-the-attack-force decision —
those packs cross-check `audits/PRODUCTION_TECH_AUDIT.md` for the
prereq chain). If anything below conflicts with F1, the F1 principle
wins.

The single most important inherited rule is **§7 + §10's map-resizing
clause**: every pack must be classified `fit` / `wide` /
`large-trivial`, and every `wide` / `large-trivial` pack must be
shrunk to a bespoke procedural arena before the audit is "done".
`base_map: rush-hour-arena` on a perception pack is almost always
`large-trivial` for a tight-clock cell (the decision under test
collapses to "did the scout reach the fog band yet?") and almost
always `wide` for a coverage / long-range pack where the distance is
the decision. This rule was IMPLICIT in the F2/F3 work; do not let it
slip on F4.

---

## §26. Fog channel discipline — the perception ablation grid

The bench has THREE observation channels × TWO fog states = SIX
perception cells per (pack, level), declared in
`openra_bench/scenarios/schema.py::PERCEPTION_MODES`:

| channel | fogged | `-clear` (no fog) |
|---|---|---|
| `structured` | text briefing + 'Unexplored regions' block, no image | full coordinates revealed in text |
| `vision` | text briefing + PNG minimap (briefing already has coords; image is a SUPPLEMENT, not isolated) | text + image with no fog |
| `image` | image-PRIMARY (coords redacted in text; labelled minimap is sole spatial source) | image-PRIMARY with no fog |

Run the full grid via `run_eval --perception-sweep` (expands every
`pack:level` into `pack:level:<mode>`). The Play tab stays on the
canonical `vision` (fogged) modality.

**The no-cheat bar applies ONLY to the fogged cells**, never to the
`-clear` ones. A `-clear` cell is a perfect-information CONTROL that
isolates the perception cost — a stall / observe-only policy is
EXPECTED to win the `-clear` form of a perception pack, because the
perception requirement has been removed. The bar binds only on
fogged channels (structured / vision / image with fog on).

Audit consequence: a perception pack whose `briefing_RA` (the
canonical-text briefing audited in the CSV) is written assuming
fog is OFF (e.g. "the army is at (38,10), then (70,20), then
(96,30)") leaks the answer for the `image` and `image-clear` cells
both. Briefings must be written assuming the `image`-PRIMARY
fogged form is the binding case — i.e. the canonical text used
for `briefing_image_primary` cannot enumerate the hidden actors'
coordinates. (See §29 for `reveal_map` as a tool-side counterpart.)

## §27. Hidden enemy stance — `stance:0` is mandatory for perception packs

For any perception pack that hides enemies in fog, the HIDDEN actors
MUST be `stance:0` HoldFire. Per CLAUDE.md "stance respect on
move-fire": a `stance:2`/`stance:3` hidden enemy advertises itself
the instant a scout enters its weapon range (or, worse, with
`stance:3`, advances on the agent and self-delivers into the agent's
vision for free — collapsing the perception bar to a stall WIN).

Stance contract per perception sub-idiom:

| Sub-idiom | Hidden enemy stance | Reason |
|---|---|---|
| Discovery (`buildings_discovered_gte`, `enemies_discovered_gte`) | `stance:0` HoldFire | scout glides past, no auto-fire reveals enemy position; discovery is FREE |
| Count-and-commit (`unit_type_count_eq` + `enemies_discovered_gte`) | `stance:0` | scout-then-build channel pure; over-commit/under-commit teeth are the test, not bleed |
| Cycle / Track (`scheduled_events` re-spawn or re-locate) | `stance:0` on every leg's spawn | a stance:3 band auto-hunts the agent; stance:2 fires on the scout — both collapse the perception bar |
| Counter-recon (kill the enemy scout before its report) | `stance:0` on the ENEMY scout (its post is the test, not engagement); AGENT strike force also `stance:0` so a stall can't auto-clear it for free | mutual HoldFire isolates the explicit attack as the load-bearing verb |

The AGENT's combat arm in perception packs is also `stance:0` when
the pack has units that could auto-hunt the whole map otherwise.
`scout-cycle-keep-info-fresh`, `scout-track-enemy-movement`,
`scout-deny-enemy-vision`, and `perception-count-the-threat-small-k`
all explicitly note this in their header comments — a scenario-
declared `stance:3` ground unit auto-hunts on the live engine, so a
do-nothing stall could win.

Stance defects to flag in audit:

- HIDDEN enemy at `stance:2` or `stance:3` ⇒ auto-fire / auto-hunt
  collapses the perception bar.
- AGENT combat unit at `stance:3` (default) in a perception pack ⇒
  stall wins for free by self-delivering the army.
- Counter-recon enemy scout NOT at `stance:0` ⇒ it fires back, the
  agent's first attack opens a brawl, not a clean intercept; the
  test conflates counter-recon with combat.

## §28. `scout-cycle-keep-info-fresh` idiom — `scheduled_events: spawn_actors`

Information FRESHNESS perception (re-observe a stale region) is
only testable via the Wave-9 `scheduled_events` hook. A scout pack
that needs to test "you scouted region X at t=0 but it changed by
t=T" MUST inject the change with `scheduled_events: spawn_actors`
(or `destroy_actors`) — a static-at-t=0 placement cannot test
freshness.

Worked example: `scout-cycle-keep-info-fresh` spawns 3× 2tnk
reinforcements at (78,20) at tick 1500 — a deep-fog cell that the
agent's initial scout sweep passes through EMPTY at t<300. Only a
re-cycled scout sees the reinforcement; a one-shot scout latches
the first detection bar but stalls on the post-reinforce bar.

Related idiom (`scout-track-enemy-movement`): pair
`spawn_actors(new leg)` + `destroy_actors(old leg)` with a 60-tick
overlap so the relocated band has fresh actor ids (a
`destroy_actors` frees ids; a `spawn_actors` that fired AFTER the
free would RECYCLE those freed ids — collapsing
`enemies_discovered_gte`, which counts UNIQUE ids).

The same hook is referenced in F3 §22 for wave injection; in F4 it
is the load-bearing primitive for the information-loop perception
tests.

## §29. `reveal_map` vs `explored_pct_gte` — perception controls vs scouting

Two distinct map-vision facts:

- **`reveal_map: true`** (top-level scenario flag, per CLAUDE.md
  reveal_map note) — every enemy actor is reported regardless of
  shroud; the agent observes the whole map with no fog. This is the
  no-fog half of the perception ablation grid (§26). Packs that
  declare `reveal_map: true` are PERCEPTION CONTROLS — they expose
  the gold-standard observation so the perception cost can be
  measured by ablation. No F4 pack currently declares
  `reveal_map: true` at the pack level; the `-clear` perception
  modes apply the flag at sweep time.
- **`explored_pct_gte: X`** (win predicate) — measures the AGENT's
  cumulative shroud-cleared coverage of the playable area. This is
  a SCOUTING bar (the agent must drive scouts into fog to lift
  shroud). Packs that score on this predicate test path-planning
  for coverage, not target discovery — `scout-map-reveal-percent-target` is the sole F4 cell on this axis. (Versus
  `buildings_discovered_gte` / `enemies_discovered_gte`, which test
  finding specific actors.)

A perception pack should pick ONE bar: coverage (broad sweep) OR
discovery (find specific landmark). Mixing them dilutes the test
and tends to admit "drive east and never look at the map" wins.

## §30. `enemies_discovered_gte` semantics — DISCOVERED, not ALIVE

`enemies_discovered_gte: K` counts UNIQUE enemy actor ids EVER
observed by any agent unit. Three consequences:

1. A killed enemy that was previously discovered KEEPS counting.
   So a `buildings_discovered_gte` / `enemies_discovered_gte` clause
   is the right bar for "did the scout SEE the thing" — NOT
   `units_killed_gte`, which counts kills regardless of discovery
   order.
2. Fresh-id-spawned enemies (Wave-9 `spawn_actors` reinforcements,
   relocation legs) are ADDITIVE to the discovered set. So a cycle
   / track pack can use a growing `enemies_discovered_gte` count
   across each leg's bar (initial swarm = K1, after first
   reinforce = K2 > K1, etc.) — but ONLY if a scout has vision
   when the new ids spawn. This is the load-bearing teeth in
   `scout-cycle-keep-info-fresh` and `scout-track-enemy-movement`.
3. Buildings do NOT count toward `enemies_discovered_gte`; they
   surface in `enemy_buildings_*` instead. Use
   `buildings_discovered_gte` for building discovery. So a building
   distractor (e.g. the `silo` in `perception-count-the-threat`
   medium) does NOT inflate the unit count — testing
   unit-vs-building discrimination is clean.

Audit consequence: a scout pack whose win is `units_killed_gte`
without a corresponding `enemies_discovered_gte` (or the kill bar
without an `_in_region` clause forcing the scout step to precede)
is testing combat, not perception. Flag.

## §31. Distance-vs-detection trade-off — scout chassis vs the closing enemy

Many F4 packs test "go far enough to detect WITHOUT being seen" or
"detect WITHIN the report window before the enemy completes its
intel". The relevant engine speeds (CLAUDE.md + empirical, see
header comments in `scout-jeep-vs-infantry-cost-effective` and
`scout-multiple-fog-areas`):

| Chassis | Build cost / time | Travel speed | Sight |
|---|---|---|---|
| `jeep` (wheeled scout) | $600 / ~540 ticks | ~6 ticks/cell | 7c |
| `1tnk` (light tank, tracked) | $700 / ~630 ticks | ~7-8 ticks/cell | 6c |
| `2tnk` (medium tank, tracked) | $800 / ~540 ticks | ~15 ticks/cell on RH-arena | 6c |
| `e1` (rifle, foot) | $100 / ~90 ticks | ~15 ticks/cell | 4c |
| `e3` (rocket, foot) | $300 / ~270 ticks | ~15 ticks/cell | 4c |

The jeep is the canonical scout chassis: fastest travel, widest
sight, lowest cost. Packs that test chassis-selection
(`scout-far-frontier` — jeep vs e1 pre-placed) or
cost-effective-recon (`scout-jeep-vs-infantry-cost-effective` —
$900 budget, three buildable options) MUST tune the deadline so
the WRONG chassis demonstrably misses the clock and the RIGHT
chassis comfortably makes it. Easy is the bare-skill rehearsal
("send anything east"); medium/hard tighten the clock until the
chassis-pick bites. Cross-check with `audits/PRODUCTION_TECH_AUDIT.md` when the pack has `build` tools.

A counter-recon pack (`scout-deny-enemy-vision`) inverts the
trade-off: the AGENT'S strike force must be fast enough to reach
the enemy SCOUT before the scout's report timer (`within_ticks` =
the report window) closes. Tank speed sets the floor on the
report window — too short and even a perfect intercept misses; too
long and a stall wins.

## §32. Briefings and image-PRIMARY mode — coordinate leak audit

`image`-PRIMARY perception mode (per §26) redacts every coordinate
from the text and gives the model only a labelled minimap. The
canonical briefing audited in `briefing_RA` is the text the model
sees in `structured` and `vision` modes — but every coordinate it
NAMES is also a coordinate the model can read OUT OF THE IMAGE in
the image-PRIMARY form, so a coordinate dump in `briefing_RA` is
not a leak for the `image` mode per se.

What IS a leak in `image`-PRIMARY: enumerating which cells are
DECOYS vs the real target, the post-reinforce direction the army
will march, the exact corner the spawn-rotated target is in for
this seed. Those are answers to the perception question, not just
landmarks.

Specifically forbidden in `briefing_RA` for F4 packs:

- "The target is at (X,Y)" when fog hides whether the target IS
  there. (Allowed in `-clear`. Forbidden in fogged.)
- "The army marches A → B → C, A at tick T1, B at tick T2" — the
  scout pack's whole purpose is reading the relocation; the
  briefing must say "the army is on the march" without listing
  every leg's coordinates and tick. (Compare `scout-track-enemy-movement` description prose to the YAML's
  `scheduled_events` — the prose currently enumerates every leg's
  cell + tick, which is a near-total leak of the tracking step.
  Flag in the audit.)
- "Bunching loses, fan to all four corners" — strategy
  prescription (also forbidden by §9.5).
- "The decoys are at (X1,Y1) and (X2,Y2); the real target is at
  (X3,Y3)" — names the answer of the target-vs-decoy
  discrimination outright.

Allowed:

- "Two unexplored pockets east; the near pocket is decoy, the far
  one holds the target" — direction-relative landmarks, fog-state
  description, but NOT the coords.
- "Three candidate corners; only one holds the real base — scout
  to find which" — names the candidate set as a topology, lets
  the model do the search.

Audit consequence: F4 briefings carrying explicit fog-pocket
coordinates are leak-rich and need a rewrite in the YAML-edit
phase. Many existing F4 packs do this (every relocation leg
cell + tick named in `scout-track-enemy-movement`,
`scout-cycle-keep-info-fresh`; every decoy + real cell named in
`perception-target-vs-fog`; every hidden cluster cell named in
`perception-count-the-threat`). Flag per pack.

## §33. `then:` ordered chains as the perception-cycle predicate

A `then:` composite latches each clause once true and advances IN
ORDER (`win_conditions.py::_then`). Two valid F4 idioms:

- **Cycle** (`scout-cycle-keep-info-fresh`): 3 clauses —
  `enemies_discovered_gte:K_initial`, `after_ticks:T_reinforce`,
  `enemies_discovered_gte:K_total > K_initial`. The third clause
  is reachable ONLY if a scout had vision at T_reinforce — the
  one-shot scout policy latches clauses 1+2 but stalls on 3.
- **Track** (`scout-track-enemy-movement`): 5-7 clauses for 3-4
  legs, each leg adding one detection bar. Same shape: each
  later bar requires another full band's worth of unique ids,
  reachable only if the scout re-acquired the army at the new
  leg.

The `then:` chain MUST satisfy `K_(i+1) > K_i + (one full band's
worth)` so a one-shot scout latched at K_i cannot also latch
K_(i+1) by accident (the initial swarm alone must NOT be big
enough to satisfy the post-reinforce bar). This is the explicit
fix the `medium` tier of `scout-cycle-keep-info-fresh` makes
(swarm=6, post-reinforce bar=8 — swarm alone is short by 2).

Audit consequence: a perception-cycle pack with K_2 ≤ K_1 + (swarm
size) admits the one-shot scout WIN — defect, flag.

## §34. Map-fit shorthand for F4

Specific to F4 (in addition to F1 §7's general rule):

- A pack on `rush-hour-arena` (128×40) where the agent's scouts
  spawn at x≈10 and the target / enemy is at x≈100-120 is
  `large-trivial` for a tight-clock cell ("did the scout reach
  the fog band yet?" not "is the perception read correct?"). The
  intended `wide`-or-tighter form ships the scout at x≈20 and the
  target at x≈45 on a 64×40 arena, preserving the perception step
  but cutting the empty traverse.
- A pack on `rush-hour-arena` where the test IS distance-vs-speed
  (`scout-far-frontier`, `scout-jeep-vs-infantry-cost-effective`)
  is `wide` — the distance is the load-bearing variable and the
  arena width is sized to be the discriminator.
- A pack on `rush-hour-arena` testing coverage
  (`scout-map-reveal-percent-target`) is `fit` for the coverage
  test on hard (70% of 128×40 is genuinely hard) but `wide` for
  easy (30%).
- A pack on a custom arena (e.g. `scout-arena`, `scout-and-survive-arena`, `confined-aisle-64x40`) declared in
  the YAML's `base_map.generator` block is presumed `fit`; verify
  by reading the obstacles + spawn + target geometry.

---

## Family-4 audit CSV column contract

`audits/family4_perception.csv` uses the F1 column set (no new
columns needed; perception-specific signals live in
`enemy_posture` and `posture_issue` per F1 §8):

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget
```

Same `map_fit` discipline as F1/F2/F3: any `wide` / `large-trivial`
row is a backlog item for the YAML-edit phase to shrink to a
bespoke arena.

Tick budget convention follows F1: `tick_budget = max_turns × 90 +
3` (the historical F1 audit convention against 90 ticks/turn; the
YAML's `within_ticks` / `after_ticks` values are read directly
from the predicate and reported in the win/lose readouts).
