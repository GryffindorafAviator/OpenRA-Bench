# Family-3 (Defense) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-3.** Family-2 economy rules
(§11-§17) do NOT inherit by default but are referenced when a defense
pack carries an economy axis (e.g. `def-retreat-and-rebuild` rebuilds a
`proc`). What follows are the ADDITIONAL defense-specific rules.
If anything below conflicts with F1 principles, the F1 principle wins
(we keep briefing/map conventions identical across families).

The single most important inherited rule is **§7 + §10's
map-resizing clause**: every pack must be classified `fit` / `wide` /
`large-trivial`, and every `wide` / `large-trivial` pack must be
shrunk to a bespoke procedural arena before the audit is "done".
`base_map: rush-hour-arena` on a defense pack is almost always
`large-trivial` (128×40 is far larger than any single-base, single-wave
defense decision needs) and MUST be rewritten to a custom map sized
to the actual base + wave-spawn + intercept-zone geometry. The 14 of
22 F3 packs still on `rush-hour-arena` are the backlog the YAML-edit
phase has to clear.

---

## §18. Wave timing is the load-bearing dimension

Defense packs test whether the model holds against a TIMED wave.
The wave must arrive at a deterministic tick, with sufficient
pre-wave time for the model to position defenders / build towers /
flip stance / set up triage.

Two valid timing idioms:

- **Pre-placed attackers** — enemy band already on the map at t=0;
  the test is the agent's reaction speed and target prioritisation.
  Use this only when the engagement is meant to bite immediately
  (e.g. `def-evacuation` — the assault is on the doomed base from
  tick 0; `def-counter-battery` — the artillery is already shelling
  the fact).

- **`scheduled_events: spawn_actors`** — the wave is injected at a
  declared tick AFTER the agent has had time to act (see §22).
  Always state the wave tick in the briefing ("at tick 1500", "around
  turn 17", "mid-episode") so the model knows the clock.

A pack whose wave arrives BEFORE the intended build/positioning
completes is defective: the model is asked to react to a threat that
materially can't be met. Inverse: a wave whose intended response
fits in < ½ of `max_turns` makes the deadline inert (stall
collapses to DRAW not LOSS — F1 §5 violation).

## §19. Pre-placed defenders vs build-your-defense

Two valid sub-idioms; each pack must clearly be ONE of them.

- **Pre-placed defenders** — agent starts with `e1` / `e3` / `2tnk` /
  `pbox` / `gun` already in position. Tests positioning + stance +
  focus-fire + reserve commitment under threat. No `build` /
  `place_building` tools needed; cash usually 0.
  Examples: `def-bridge-chokepoint`, `def-counter-battery`,
  `def-evacuation`, `def-stance-mgmt-hold-then-attack`,
  `def-pre-position-mobile-reserve`, `def-reinforce-the-breach`,
  `def-multi-direction`, `def-with-ambush`.

- **Build-your-defense** — agent has `fact` + `tent` (+ `powr`) + cash
  and must construct pillboxes / refinery / depth bands before the
  wave hits. Tests construction-under-pressure and tech-tree
  reasoning. Cross-reference `audits/PRODUCTION_TECH_AUDIT.md` for
  the prereq chain.
  Examples: `def-in-depth`, `def-in-depth-vs-single`,
  `def-position-expected-direction`, `def-position-revealed-direction`,
  `def-tower-line-vs-cluster`, `def-walls-vs-towers`,
  `def-while-building`, `defense-rush-survive`, the three
  `build-defensive-*` packs.

A few packs are hybrid (`def-engineer-repair-under-fire`,
`def-retreat-and-rebuild`, `def-surprise-flank-react`) — the
hybrid form is fine as long as the briefing names BOTH the
pre-placed defenders AND the build/deploy verb required.

## §20. `pbox` is now load-bearing (engine fix)

A built `pbox` fires `M60mg` anti-infantry burst (engine fix, pinned
by `OpenRA-Rust/openra-sim/tests/test_pbox_fires.rs`). Pre-engine-fix,
a built pbox stood inert and a "build pbox" predicate was a topology
proxy that never had to actually fire. NOW any defense pack
advertising "build pillboxes" must have a load-bearing
`units_killed_gte` clause where the pbox is the kill source. A pure
`building_count_gte: pbox` win without a kill bar lets a stall play
that places pboxes off the rush axis pass — the topology clause
satisfies without the active defence ever doing anything.

The canonical idiom for a build-pbox pack is:

```yaml
win_condition:
  all_of:
    - building_count_gte: {type: pbox, n: K}    # quantity floor
    - building_in_region: {type: pbox, x:?, y:?, radius:?, count:?}  # topology
    - units_killed_gte: M                       # pbox MUST fire
    - building_count_gte: {type: fact, n: 1}    # base survival
    - within_ticks: T
```

Every active F3 build-pbox pack in the suite uses this shape — see
`def-tower-line-vs-cluster`, `def-walls-vs-towers`, the
`build-defensive-*` triple, `def-while-building`. The
`def-in-depth` and `def-position-expected-direction` packs do NOT
include a `units_killed_gte` clause yet — that's a backlog
upgrade (the topology proxy is currently strong because the rush
geometry forces the pbox arc onto the rush path, but per the rule
above the kill bar should be added explicitly).

## §21. Stance discipline

Many defense packs flip stances under threat (CLAUDE.md `stance:0`
/ `stance:1` idioms). The win predicate must reward correct stance
management, not just survival.

- `def-stance-mgmt-hold-then-attack` — pre-placed tanks on `stance:0`
  (HoldFire). The model must `set_stance` to lift engagement
  authority; an un-flipped stance means the tanks die silently and
  the fact falls. The win requires BOTH `units_killed_gte` (the
  tanks must have actively engaged) AND `has_building: fact` AND
  `own_units_gte` (the tanks must survive). The stance flip is the
  load-bearing verb.

- `def-bridge-chokepoint` — defenders pre-placed on `stance:0`. Model
  must call `set_stance` AND move defenders to the bridge mouths.

- `def-engineer-repair-under-fire` — defenders pre-placed on
  `stance:0`. Model must commit via `attack_unit` (auto-fires en
  route regardless of HoldFire idle — engine note in CLAUDE.md).
  The `repair` tool is the OTHER load-bearing verb.

`stance:0` defenders without a `set_stance` tool exposed in the
pack's `tools:` list is a defect. Likewise a defense pack whose win
predicate is `units_killed_gte` alone (no survival floor) lets a
`stance:3` self-sacrificing charge satisfy the bar — survival must
be in the predicate.

## §22. `scheduled_events.spawn_actors` for wave injection

The canonical way to deliver a defense wave is the Wave-9
`scheduled_events: spawn_actors` hook (pinned by
`OpenRA-Rust/openra-data/tests/test_scheduled_events.rs`). Static
enemy clusters at t=0 don't test "react to incoming" — they test
scouting / pathing. Use scheduled wave injection when:

- The pack tests "build the defence WHILE the threat is incoming"
  (`def-while-building`, `defense-rush-survive`, the
  `build-defensive-*` triple — wave at tick 1500-2200, AFTER serial
  pbox build completes).
- The pack tests "fortify, THEN react to a developed threat"
  (`def-tower-line-vs-cluster`, `def-walls-vs-towers` — wave at
  tick 1800-2000 after the tower line is up).
- The pack tests "intel says X, actual threat is Y" — late wave
  injection from the surprise axis (`def-surprise-flank-react` —
  wave at tick 800/1400 from the supposedly safe flank).
- The pack tests "react to a breach mid-episode"
  (`def-reinforce-the-breach` — heavier breach wave at tick 450 on
  one of two lanes).

Conversely, pre-placed waves are correct when:

- The base is structurally doomed from t=0 and the test is the
  RETREAT decision, not the response window (`def-evacuation`,
  `def-retreat-and-rebuild`).
- The engagement is positional / target-priority and the wave
  composition is the static probe (`def-counter-battery` —
  artillery is already shelling at t=0).

A pack with a static t=0 wave AND a long pre-engagement march is
testing scouting under defense flavour — flag as a posture mismatch.

## §23. Wave success metric is asymmetric

Defense WIN is "wave fully exhausted + base intact"; defense LOSS
is "base destroyed OR critical asset lost".

- Use `building_count_gte: {type: fact, n: 1}` (PRESENT-TENSE) NOT
  `has_building: fact` (ONE-SHOT EVER-SEEN). The latter latches
  true the moment the fact is observed and never falls back — a
  destroyed fact still satisfies `has_building` and the loss never
  triggers.
- The `fail_condition.not: {building_count_gte: {type: fact, n: 1}}`
  is the asset-loss LOSS path; pair with `after_ticks: T+1` for
  the deadline LOSS path.
- For "tanks-alive defenses" use `own_units_gte: N` carefully — a
  unit-less start mis-fires the `not own_units_gte:1` clause on
  turn 1 (CLAUDE.md footgun, well-documented in
  `economy-force-buildup`). Use `not has_building:fact` (or specific
  asset) for the unit-less-start fail clause instead.
- For "loss cap" defenses (kite the wave but don't bleed too much)
  use `units_lost_lte: N` in the win + `not units_lost_lte: N` in
  the fail.

The `defense-rush-survive` / `def-with-ambush` / `def-surprise-flank-react`
packs combine three or four of these clauses; the
`def-position-expected-direction` pack uses `own_units_gte: 3` plus
the topology clause. All variants are valid as long as the LOSS
path is real-reachable inside `max_turns × 30`.

## §24. Build-prereq cross-check for tower builds

Any defense pack advertising `build('pbox')` / `build('gun')` /
`build('proc')` must have the prereq chain in `agent.actors`
(CLAUDE.md tech-tree and `audits/PRODUCTION_TECH_AUDIT.md`):

| Buildable | Prereqs (Allies) | Cost | Build sec |
|---|---|---|---|
| `pbox` | `tent` (Defense queue) | 600 | 12 |
| `gun` | `tent` + `fix` | 600 | 14 |
| `proc` | `fact` | 1400 | 28 |
| `powr` | `fact` | 300 | 8 |
| `silo` | `proc` | 150 | 5 |

The Defense queue is gated by the INFANTRY building (`tent`), not
by the construction yard alone — a pack saying "build pillboxes"
without `tent` is a tech-gate defect. Cross-reference
`audits/production_tech_audit.csv` for the explicit per-pack table.

`build-defensive-tower-line`'s hard tier needs 8 × `pbox` (4800cr =
6 rungs + 2 rebuilds) and lists `tent + powr` — clean.
`tech-turtle-defensive-tech` (NOT an F3 pack but adjacent) needs
`weap + fix + dome + gun + pbox` in 90 turns and has
`tight-cash:need=5400,have=3000` PLUS `build-time-over-budget:98s>90t`
— flagged as DEFECT in the tech audit.

## §25. Map shrink rule still applies (F1 §10 binding)

Defense decisions usually need a tight engagement zone (8-15 cells
from defender to wave-spawn corridor). A `rush-hour-arena` 128×40
hosting a single-lane defense at x=10..30 is `large-trivial` —
~98 cells of empty east traversal that the wave's pathfinder has to
chew through before the actual defense decision bites. The model is
effectively asked "did the wave arrive yet?" instead of "is your
defense correctly positioned?".

Default rule for the YAML-edit phase: cut the empty pre-engagement
traversal to ≤15 cells. For most F3 packs that lands at a 56×40 or
64×40 custom arena (preserving the agent base + wave-spawn band +
the defensive zone), with positions translated 1:1 from the original
layout.

Packs already on a tailored arena (custom `base_map.generator: arena`
or `chokepoint-arena` with explicit `width`/`height`) are `fit` and
do not need shrinking — these are the F3 packs that have already
been through a Wave-12 / Wave-13 redesign. Packs still on
`rush-hour-arena` 128×40 are the backlog.

## §26. Two-spawn-point hard tier contract still applies

Every F3 pack's `hard` tier must declare ≥2 agent spawn_point
groups (typically NORTH `y≈14` / SOUTH `y≈26`) so the defense
geometry round-robins by seed. A memorised "place pbox at (24,20)"
opening cannot generalise to the y=14 / y=26 latitudes. Patterns:

- AGENT-side rotation (most F3 packs) — all agent actors carry
  spawn_point; enemy actors do not, so the rush always materialises
  regardless of which agent latitude is active. The `rusher` /
  `hunt` bot then commits to whichever agent centroid is live.
- ENEMY-side rotation (`def-tower-line-vs-cluster` hard) — enemy
  actors carry spawn_point (per-owner filter, Wave-9), so the
  forcing geometry (concentrated thrust vs wide-front) varies per
  seed while the agent base stays canonical.

Persistent base / sentinel actors must be duplicated across BOTH
spawn groups at identical coords (CLAUDE.md spawn_point footgun).

---

## Family-3 audit CSV column contract

`audits/family3_defense.csv` uses the F1 column set (no new columns
needed — the F2 economy columns aren't required for defense packs
unless the pack has an economy axis, in which case the `briefing_RA`
text covers it inline):

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget
```

Same `map_fit` discipline as F1: any `wide` / `large-trivial` row is
a backlog item for the YAML-edit phase to shrink to a bespoke arena.

Tick budget convention follows F1: `tick_budget = max_turns × 90 + 3`
(non-interrupt mode is exactly 30 ticks/step, but the historical F1
audit convention measured against 90 ticks/turn for the interrupt-mode
empirical step — keep the F1 number for consistency; the YAML's
`within_ticks` and `after_ticks` values are read directly from the
predicate and reported in the win/lose readouts).
