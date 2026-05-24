# Family-10 (Special weapons + Misc) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-10.** F2's §18 no-solution-leak
extension is referenced where relevant (it is identical in spirit to F1
§9.5; the F10-specific leak patterns are catalogued in §66 below). If
anything below conflicts with F1 principles, the F1 principle wins (we
keep briefing/map conventions identical across families).

The single most important inherited rule is **§7 + §10's map-resizing
clause**: every pack must be classified `fit` / `wide` / `large-trivial`,
and every `wide` / `large-trivial` pack must be shrunk to a bespoke
procedural arena before the audit is "done". Most F10 packs are already
on tailored arenas (the spec-* packs each ship their own per-tier map)
— the holdout is `rush-hour` (the historical 128x40 reference pack) and
`custom-map-no-enemy` (uses the bespoke `singles-maginot` confined
custom map). Both are deliberately on non-shrunk maps for their own
specific reasons (see §65 and §64).

---

## §60. Superweapon packs — launcher present + charge respected

A superweapon (nuke `mslo`, iron-curtain emitter, chronosphere `pdox`)
fires via `Command::FireSuperweapon { kind, target_cell?, target_id? }`
— the ONLY engine verb that fires nukes / IC / chrono (no other
`Command::*` variant routes there). The engine validates:

- The agent owns a launcher BUILDING of the matching `kind` (e.g. for
  `kind: "mslo"` the agent must own an `mslo` actor).
- The weapon is fully CHARGED. Tests use a hard-coded 100-tick charge
  per kind (`SuperweaponKind::charge_ticks`); real-play values live in
  `gamerules.rs`.
- Per-kind argument shape:
  - nuke (`mslo`) needs `target_cell` (an `(x, y)` cell).
  - iron-curtain needs `target_id` (the friendly actor to bestow).
  - chrono (`pdox`) needs BOTH `target_cell` (destination) and
    `target_id` (friendly actor to teleport).

A failed validation is logged and the order is DROPPED SILENTLY — no
error to the agent, the charge stays consumed-or-not depending on the
gate that failed. The pack is responsible for ensuring the charge window
fits inside `max_turns × 30` (non-interrupt) or `max_turns × max_ticks`
(interrupt).

Superweapon-pack invariants:

- The launcher building must be PRESENT on the agent's actor list (or
  buildable from the pack's starting cash + tech, but the standard
  idiom is pre-placed launcher).
- `within_ticks` ≥ 100 + decision-latency budget. Spec-nuke-strike
  uses 1800, which comfortably covers the ~100-tick charge plus ~22
  decision turns. A `within_ticks` shorter than the charge time is a
  defect (the model cannot fire in time even with optimal play).
- A SPOTTER unit must be in vision range of the target cluster if the
  win predicate counts destruction (the bench's
  `enemy_buildings_destroyed_gte` requires the destruction to surface
  through `enemy_buildings_seen_ids` — a building only counts once
  it has been seen). Place the spotter just OUTSIDE the blast radius
  (R=4 Chebyshev for nuke) so the strike survives the spotter.
- Briefings must NOT spoil the target cell. Acceptable: "a cluster of
  enemy ore silos sits across the map; a friendly jeep already has
  eyes on it." Forbidden: "fire the nuke at (80, 30)" or "the cluster
  is at (140, 12)". The model reads the observation to find the
  cluster centroid; that's the test.
- Engine gap (logged in `spec-nuke-strike.yaml`): `detonate_nuke` does
  NOT credit `kills_per_player`, so `units_killed_gte` is unsafe for
  nuke packs. Use `enemy_buildings_destroyed_gte` (adapter-side from
  disappeared-but-recently-seen building ids).

## §61. Spy / thief infiltrate — consumed-on-use, target restrictions

Both infiltrators fire `Command::Infiltrate { unit_id, target_id }`
and are CONSUMED on contact (the actor is removed; the order is
single-use). Target restrictions differ:

- **`spy` (Allied spy)** — works against ANY enemy building. On
  adjacency triggers a one-shot reveal scan: every building owned by
  the target's owner enters the agent's `enemy_buildings_seen_ids`
  set, surviving fog. The load-bearing win predicate is
  `buildings_discovered_gte: K` where K > (buildings the spy can see
  by walking up to the target alone) — so the scan, not the walk, is
  what fires the win.
- **`thf` (thief)** — cash-steal branch is gated on `proc | silo`
  ONLY (the engine match-arm intent). Against any non-proc/non-silo
  enemy building the thief walks up, is consumed, and 0 cash drains.
  The Python tool description already documents this. The win
  predicate is `cash_gte: M` with `starting_cash: 0` and no other
  income, so the M can ONLY come from the steal.

Infiltrate-pack invariants:

- The infiltrator must spawn at `stance: 0` HoldFire. The engine
  default for an agent unit is stance:3 AttackAnything which makes
  the unit auto-march toward the nearest visible enemy — abandoning
  the agent's intended walk path.
- `tools:` must expose `infiltrate`. The brute / wander / wrong-tool
  policies need their verbs available (`move_units`, `attack_unit`,
  `observe`) so the bench can demonstrate they LOSE — that's the
  no-cheat bar.
- The fail clause should include `not: {unit_type_count_gte: {type:
  spy_or_thf, n: 1}}` so a destroyed-infiltrator run terminates
  immediately as a LOSS instead of running to deadline. (Some F10
  packs use the generic `not own_units_gte:1` which is OK too as
  long as the agent starts with at least one unit.)
- Spy reveal-scan win bar must require K > 1 (more than the walked-
  to building alone) — otherwise a walk-up-to-the-proc play that
  ALSO reveals the proc by LOS satisfies the predicate without the
  scan firing. Spec-spy-infiltrate uses K=4/6/7 across tiers, all
  strictly greater than the LOS-visible count.
- Thief cash bar must be set so each `infiltrate` yields enough but
  no other cash source closes the gap. Spec-thief-steal-cash uses
  cash_gte:400 (easy/medium, 1 thief = ~500 drain) and cash_gte:800
  (hard, 2 thieves required).

## §62. Engineer capture — `e6` + `Command::CaptureActor`

The engineer (`e6`) walks to an enemy building and captures it via
`Command::CaptureActor { unit_id, target_id }`: ownership transfers
to the capturer's player AND the engineer is consumed. The captured
building surfaces in the agent's `own_buildings` and the win predicate
is typically `has_building: <type>` (the captured building now reads
as agent-owned).

Engineer-capture invariants:

- `tools:` must expose `capture_actor`. `move_units` / `attack_unit`
  / `observe` are present so brute and wrong-tool policies can be
  observed to LOSE.
- The engineer carries NO weapon — sending it on `attack_unit` is a
  no-op (the engineer stands still). A briefing must NOT instruct
  the model to "attack the building with the engineer".
- A brute "destroy the building" play is too slow to clear the
  90 000-HP `proc` inside `max_turns × 30` with non-engineer units,
  so the deadline bites: a no-engineer-verb policy LOSES.
- Win predicate: `has_building: proc` (or whatever building type
  the capture targets) is sufficient because `has_building` checks
  the AGENT's building inventory. Once captured, the proc is the
  agent's. Pair with `within_ticks: T`; add `fail_condition:
  {after_ticks: T+1}` for the deadline LOSS.
- Per the no-solution-leak rule (§9.5), briefings should NOT name
  the verb. "Seize the refinery" is allowed; "capture_actor the
  refinery" is not.

## §63. Tanya C4 — instant-destroy

Tanya's C4 fires via `Command::C4Detonate { unit_id, target_id }`:
on adjacency to an enemy building, the building is INSTANTLY
destroyed; Tanya SURVIVES the detonation. This is the destroy-verb
counterpart to the engineer's capture-verb — same walk-up pattern,
different terminal effect.

Tanya-C4-pack invariants:

- `tools:` must expose `c4_detonate`. `move_units` / `attack_unit` /
  `set_stance` are present for the brute/wrong-tool diagnostic.
- Tanya's PISTOL is anti-infantry — it deals ~500 dps and would take
  >100 outer ticks to chew through a `proc` (90 000 HP). A brute
  `attack_unit` policy on a proc times out before the clock under
  any sensible deadline.
- The win predicate is `enemy_buildings_destroyed_gte: 1`. Pair with
  `within_ticks`. The deadline must be tight enough that a brute
  attack-the-building-with-pistol policy LOSES; spec-tanya-c4-strike
  uses 1500/2000/2700 across tiers.
- **Cross-reference Tanya combat from F1**: packs like
  `combat-tanya-vs-rush`, `combat-tanya-pistol-snipe`, and Tanya
  appearances in mixed-force F1 packs test Tanya's ANTI-INFANTRY
  pistol micro. F10's `spec-tanya-c4-strike` is the BUILDING-
  demolition cell — different verb, different test. Do not
  duplicate Tanya combat semantics in F10; F10's Tanya cell is
  load-bearing on the C4 verb specifically.
- Defenders may sit near the proc (e.g. stance:0 baits in medium,
  stance:2 cover in hard). Their role is to make a brute pistol
  trade SLOW, not to gate the C4 walk-up. Tanya's HP comfortably
  absorbs the transit damage.

## §64. `custom-map-no-enemy` — pure navigation, not a capability test

This pack tests map-loading + perception + the `move_units` verb.
There is NO adversary on the map (the pack is whitelisted in
`tests/test_hard_tier.py::_NO_ENEMY`). It is a PURE NAVIGATION cell:

- No combat, no kill predicate, no defender.
- The only failure mode is missing the deadline (`after_ticks` LOSS).
- Hard tier escalates to `objective_coords: relative` + a seed-
  rotated spawn latitude — the same machinery the F1 hard tiers use
  to defeat memorisation.
- This pack does NOT need the full no-cheat bar (stall/brute/etc.
  all collapse to the same "did you reach the zone in time"
  question). It's a SMOKE TEST + DEADLINE PROBE, not a capability
  test in the F1 sense.
- Briefing rule §9.5 still applies (don't dump every coordinate),
  but the goal-region coordinate is intrinsically known and shipping
  it (easy/medium) is fine. The hard tier hides the coordinate
  behind a `label:` field that the bench surfaces as a relative-
  direction description.

## §65. `rush-hour` — historical baseline pack, special handling

`rush-hour` is the ORIGINAL reference pack the training-anchor maps
descended from. It is a multi-corner search-and-destroy sweep on the
128x40 `rush-hour-arena` and is the only pack still on that map by
deliberate choice (every other F10 pack ships a tailored arena).

Special handling rules:

- **DO NOT include `rush-hour` in any production family edit cycle
  unless explicitly invoked.** It is the historical baseline; edits
  to it ripple into the legacy comparability of every prior eval
  bench. Touch it only when the user names it explicitly.
- Map size 128x40 is INTRINSIC to the search-and-destroy capability
  (the 4-corner spawn × 22-target zigzag covers the whole arena).
  Classify as `fit` — the wide map IS the test.
- Briefings are F1-officer-style (already cleaned in a prior pass).
  Do not re-edit unless a specific defect is found.

## §66. F10-specific solution-leak patterns

The general no-solution-leak rule (F1 §9.5, F2 §18) applies; F10
introduces a handful of family-specific leak shapes to watch for:

- **Naming the special verb in prose** — "fire the superweapon",
  "capture_actor the refinery", "infiltrate the silo", "c4_detonate
  the proc". Allowed: "demolish the refinery", "seize the
  refinery", "drain the depot", "strike the cluster". The model
  must derive the right verb from the `tools:` list, the unit type,
  and the situation.
- **Spoiling the target coordinate for a charge-and-fire pack** —
  "fire the nuke at (80, 30)". Allowed: "the cluster sits in the
  south-east corner" — relative direction + cluster description.
- **Naming the wrong-target as a verbatim trap** — "the Power Plant
  gives nothing and burns your thief" is borderline (it telegraphs
  the trap). Acceptable form: "two enemy buildings sit on the
  approach; only one will yield cash" — the model must figure out
  which one. Spec-thief-steal-cash currently uses the explicit
  form for clarity; flag as a minor leak candidate for a future
  edit pass.
- **Stating the brute-policy outcome** — "a brute pistol attack on
  the ore refinery times out before the clock" (spec-tanya-c4-
  strike hard). Borderline; the F1 audit rule is to remove these.
  Acceptable form: "Tanya's pistol cannot demolish the refinery in
  time" — states the constraint without naming the failure path.

## §67. Tick-budget bound for F10 packs

The F1 audit convention is `tick_budget = max_turns × 90 + 3` (the
interrupt-mode empirical step count). Several F10 packs are
non-interrupt-mode (no `interrupts:` block) so the engine advances
exactly 30 ticks/turn — at `max_turns: 22`, the tick at the deadline
is 660, not 1980. Verify per-pack:

- `spec-engineer-capture` — no `interrupts:` → 30 ticks/turn →
  max tick = 750/1050/1500 (easy/medium/hard). Within_ticks =
  2000/3000/4000 are well above the reachable tick → DEFECT?
  Actually no — these packs declare `termination: {max_ticks:
  8000}` and `planning: true` in `base:`, and the bench runs them
  with an effective per-turn advance that varies. Verify by smoke
  run; flag for engine-followup if a stall policy DRAWS instead of
  LOSING.
- `spec-nuke-strike` — declares `interrupts: {enemy_unit_spotted:
  true}` → interrupt mode → ~1–5 ticks/turn (variable). The
  `within_ticks: 1800` against `max_turns: 22` is reachable only
  because the charge gate ENTERS the loop at tick 100, and the
  agent has ~22 decision turns to fire after. Verify the deadline
  bites by running stall.
- `spec-tanya-c4-strike` / `spec-thief-steal-cash` / `spec-spy-
  infiltrate` — no `interrupts:` → 30 ticks/turn → the deadline
  bites only if `within_ticks ≤ 30 × max_turns`. Check each row.

Audit the values per pack and flag mismatches in the
`posture_issue` column.

---

## Family-10 audit CSV column contract

`audits/family10_special.csv` uses the F1 column set:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget
```

Tick-budget convention follows F1 (`max_turns × 90 + 3`), with a note
that the actual non-interrupt advance is 30 ticks/turn — see §67.

Scope (7 packs, ~21 rows):

- `spec-engineer-capture` (3 levels)
- `spec-nuke-strike` (3 levels)
- `spec-spy-infiltrate` (3 levels)
- `spec-tanya-c4-strike` (3 levels)
- `spec-thief-steal-cash` (3 levels)
- `rush-hour` (3 levels — historical baseline, do not edit)
- `custom-map-no-enemy` (3 levels — pure navigation, no enemy)

`TEMPLATE.yaml` is SCAFFOLDING (the `meta.id` is `TEMPLATE` and the
file is a starter template for new pack authors). It is NOT a real
pack; SKIP for audit purposes.
