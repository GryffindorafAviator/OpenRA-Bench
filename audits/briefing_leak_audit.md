# Briefing leak audit — Family-1 (combat) + Family-2 (economy)

By-eye review of every `description:` block in every F1 + F2 pack
(combat-*, action-*, harass-*, econ-*, economy-*) for prose that
LEAKS the intended strategy / play-style instead of just describing
the situation. Per F1 §9.5 + F2 §18 of the principles docs, the
briefing must be a SITUATION REPORT — forces, objective, constraint,
fog state — and the model should DERIVE strategy from it. Any
sentence that names the winning verb, prescribes a unit/build count,
or warns about which non-intended policy loses is a leak.

## 1. Coverage

- 56 F1/F2 packs read by eye (action: 2, combat: 27, harass: 1,
  econ: 21, economy: 5).
- 168 level briefings classified (every pack carries 3 tiers).

## 2. Histogram

| Class   | Count | Share |
|---------|-------|-------|
| CLEAN   |   8   |  4.8% |
| MILD    |   8   |  4.8% |
| MEDIUM  |  12   |  7.1% |
| HEAVY   | 140   | 83.3% |

The bench is dominated by HEAVY-leak briefings. Only 8 of 168
briefings (~5%) are pure situation reports. The CLEAN cohort:

- `combat-bait-counter-attack` (all 3 tiers — already trimmed in a
  recent verbosity sweep; the contributor-only commented-out
  "Original" descriptions still carry the strategy spoilers, but the
  active `description:` field is clean).
- `combat-naval-shore-strike` easy.
- `economy-harvest-timebox` (all 3 tiers — straightforward "reach
  $N within K turns", states only the resources visible to the
  agent and the deadline).
- `econ-silo-vs-spend` hard.

## 3. Top 10 worst offenders (HEAVY)

Forty-five packs are HEAVY across all three tiers; ranking them by
severity within HEAVY would be cosmetic. The ten that are the most
egregious — multiple leak phrases per briefing, complete strategy
scripts, named verbs, and per-policy outcome arithmetic — are:

1. **combat-rocket-soldier-anti-vehicle** (all 3) — literally names
   the unit type AND the count: "Train six rocket soldiers (anti-
   vehicle launchers) from the Infantry Barracks, advance east"
   AND volunteers the negative-policy comparisons ("Rifle infantry
   can't dent armour; ... light tanks lose the attrition trade").
   The win predicate already requires `unit_type_count_gte:e3:6`.

2. **combat-vehicle-vs-infantry-counter** (all 3) — names the
   counter ("Three medium tanks walk through small-arms fire"),
   dumps per-option dollar arithmetic ("rockets waste cost-per-
   effect"), AND tells the model to "Build the three tanks".

3. **combat-formation-tank-wedge** (all 3) — names the formation
   ("Form an inverted-V wedge"), prescribes the cell-level
   placement ("leader on y=20 absorbing the blocker, flankers
   offset to y=18 and y=22"), AND volunteers the negative-policy
   ("Driving as a single column ... puts the lead tank inside
   every Dragon's range at once").

4. **combat-kite-and-pull** + **combat-kite-jeep-vs-tank** (all 6
   briefings between them) — script the kite cycle with the literal
   threshold ("When the heavy closes within seven cells, move your
   tank back along the lane; otherwise shoot from range").

5. **combat-prevent-retreat** (all 3) — explicit cell-level routing
   for the flank ("slip one tank around the flank (north via
   y=5..10 or south via y=30..35, outside rocket range)") + the
   3-vs-1 force-split ("the other three grind the cluster down
   from the west").

6. **combat-harass-balanced-hit-and-run** (all 3) — the entire
   pulse cycle is dictated: "Strike one worker, immediately retreat
   west of x=50 (past its leash), let it snap back, then re-engage."

7. **combat-suicide-charge-mission** (all 3) — explicit doctrine
   statement AND negative-policy ("Keeping the strike package alive
   is NOT possible and NOT required. Commit every unit straight at
   the yard, focus-fire through the defenders, accept the losses").

8. **econ-contested-expansion** (all 3) — three-step instruction
   sequence ("Build a second Refinery ($1400) and place it AT the
   contested patch ... and escort with your two Light Tanks") with
   dollar arithmetic and explicit placement and escort verb.

9. **econ-recover-from-zero-cash** (all 3) — a literal Cash-Threshold
   step-by-step script ("once you have $300, queue a Power Plant
   and place it adjacent to the yard ... when you reach $2000,
   build a War Factory ... then queue a second and third harvester").

10. **economy-investment** (all 3) — names the exact composition
    AND the exact arithmetic ("$1700 cash ... precisely the cost of
    one Ore Refinery ($1400) plus one supporting Power Plant ($300)
    ... Build and place both"; hard: "field 22 rifle infantry (22 ×
    100 = $2200)").

## 4. Recurring leak archetypes (for the fix agent)

Five archetypes account for the bulk of the leaks. They appear
across both families; eliminating them mechanically would clean up
most of the briefings.

### A. The named-verb prescription

`flank`, `kite`, `focus-fire`, `pulse`, `pincer`, `wedge`,
`retreat`, `commit`, `escort`, `intercept`. These verbs appear in
the imperative ("Form a wedge", "Drive the defenders east to
intercept") and ARE the answer.

Affected: `combat-flanking-attack`, `combat-divide-and-conquer`,
`combat-pincer-coordination`, `combat-formation-tank-wedge`,
`combat-prevent-retreat`, `combat-harass-balanced-hit-and-run`,
`combat-kite-and-pull`, `combat-kite-jeep-vs-tank`,
`combat-protect-vip-escort`, `combat-retreat-after-engagement`,
`combat-skirmish-then-disengage`, `combat-suicide-charge-mission`,
`combat-target-priority-highvalue`, `econ-contention-with-enemy`,
`econ-contested-expansion`, `econ-protect-harvester-route`,
`harass-response-preserve`. (17 packs / ~50 briefings.)

### B. Negative-policy comparisons ("the other play loses")

Sentences that explicitly say what the non-intended policy does
and why it fails: "a frontal charge dies in the crossfire",
"stalling drifts cash above the ceiling", "spreading fire trades
the column", "splitting fire across both heavies lets them grind
you down", "chasing east leaves the Refinery undefended", "a buy
at turn six plateaus near $10000 and just misses". These are an
explicit "DON'T do X" — the model only needs the constraint
predicate.

Affected: nearly every HEAVY pack. Particularly egregious:
`econ-burn-rate-management` (per-policy outcome arithmetic),
`econ-expansion-timing` (turn-by-turn break-even arithmetic),
`econ-tech-vs-expand-decision` (per-option dollar plateau),
`combat-attack-from-behind-fog` ("A head-on charge dies in the
crossfire"), `combat-harass-aggro-commit` ("Charging the
harvesters while the heavy is alive lets it pick you off").

### C. Build-count / unit-count / cell-coord prescriptions

The briefing dictates the answer: "train six rocket soldiers",
"build a second power plant ($300 is the cheapest add)", "place
it adjacent to the patch", "field 22 rifle infantry", "two power
plants at $300 each fit cleanly", "concentrate ALL four tanks'
fire on one rocket soldier at a time".

Affected: every economy-* pack, most econ-* packs, the
RPS-counter combat packs. Most of these prescriptions duplicate
information already in the win predicate (`unit_type_count_gte`,
`building_total_gte`, `place_building`+region clauses).

### D. Stance / tool-call naming

"Lift Tanya's hold-fire order and attack", "Change the tanks'
orders to Attack-Anything", "Flip the escort to Defend or
AttackAnything stance", "Move them onto the corridor mouth (about
x=44..46, y=18..21)". These name the exact tool call and target
coordinates — the model is being told to call `set_stance(units,
3)` or `attack_move(units, x, y)`.

Affected: `combat-stance-mgmt-attack`, `combat-tanya-vs-rush`,
`econ-harvester-defense-raid`. (Smaller archetype, ~9 briefings.)

### E. Per-policy outcome arithmetic

"A single harvester plateaus near $4100", "stall yields ~3650",
"three tanks shooting one target will drop it fast", "a buy at
turn 10 already tops out near $7000 and misses". These tell the
model the expected NUMERICAL outcome of each policy — the
clearest version of the F1 §9.5 / F2 §18 forbidden pattern.

Affected: most econ packs with cash bars
(`econ-expansion-timing`, `econ-tech-vs-expand-decision`,
`econ-resource-trade-with-self`, `economy-harvest-investment`,
`econ-burn-rate-management`).

## 5. F3 sightings (not exhaustive)

The task scope was F1+F2 only, but several leak archetypes are
likely shared with F3 (the defense packs `def-*`). The fix agent
should expect a similar distribution there. F3 has its own
in-flight shrink-rewrite pass so a separate audit is the cleanest
path.

## 6. Suggested next steps

The bench's `description:` fields are currently load-bearing for
the model's strategy. The fix would be a sweep that:

1. Strips Archetype A (named-verb prescriptions) — replace each
   imperative with a neutral situation phrase ("an enemy heavy
   tank approaches from the east on a hunt order; the heavy
   out-trades a medium head-on", not "kite the heavy").
2. Strips Archetype B (negative-policy comparisons) — delete
   every "stalling does X / a frontal charge does Y" sentence.
   The win/fail predicates already enforce the outcome.
3. Strips Archetype C (build-count prescriptions) — delete every
   "build N of unit X" sentence; let the win predicate do the
   job. Preserve the dollar costs only where the model truly
   needs them to evaluate options (and even then, prefer the
   `unit_codex` channel — the briefing should describe the
   problem, not solve the budget arithmetic for the model).
4. Strips Archetype D (stance/tool naming) — never name the tool
   call by name; describe the world-state ("your tanks are on
   Return-Fire and the scattered enemies are not engaging") and
   let the model derive the verb.
5. Strips Archetype E (per-policy arithmetic) — delete every
   "plateaus at $X", "tops out near $Y", "by tick Z" outcome
   claim; replace with the resource-state baseline ("your
   harvester yields about $95/turn" is allowed; "one harvester
   plateaus near $4100 by the deadline" is not).

Counting the CSV: 8 CLEAN + 8 MILD briefings already pass the bar.
The remaining 152 (12 MEDIUM + 140 HEAVY) need at least one of the
five archetype strips above. Roughly half the HEAVY briefings need
a near-total rewrite (Archetypes A + C + E in the same paragraph);
the others can be saved with surgical excisions of one or two
sentences.

Where a pack's win predicate alone fully encodes the intended
behaviour (e.g. `units_in_region_gte:{n:4}` for pincer,
`unit_type_count_gte:2tnk:3` for the hard-counter packs,
`building_count_gte:proc:2` AND a region clause for the
contested-expansion pack), the briefing can be safely trimmed to
"forces + objective number + deadline + faction context" with no
loss of solvability — the predicates do the work.
