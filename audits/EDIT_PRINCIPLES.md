# Scenario Audit — Edit Principles

These are the conventions used to produce `audits/family1_combat_micro.csv`
(the per-(pack, level) audit table with rewritten briefings, plain-English
win/lose conditions, and a map-fit flag). Apply the same rules to every
subsequent family audit so the resulting briefings and conditions remain
consistent across the suite.

---

## 1. Each level briefing is SELF-CONTAINED

The model (and the human reader) sees ONE level at a time, with no memory
of the easier tiers. Every briefing must work cold.

- **Never** write "the same X as before", "same as medium", "denser
  cluster, same target". Repeat the mission framing, force composition,
  and objective from scratch in each tier.
- Same applies to `win_condition` / `lose_condition` columns — do not
  write "Same as medium".
- "At the same time" / "at the same moment" referring to within-mission
  coordination is fine — it does not cross-reference another tier.

## 2. Three-part briefing structure: framing → given → objective

Every briefing follows the same shape:

1. **Mission framing** — one short sentence naming the kind of operation
   (parallel strike, hit-and-run harassment, base-defence response, etc.).
2. **What is given** — the player's force, plainly listed, with rough
   positions where they matter ("4 medium tanks on the west edge",
   "2 helicopters and a Construction Yard").
3. **Target / objective** — what the enemy is, what to do, what the
   constraint is. End with the key constraint (loss cap, deadline,
   "do not bunch up", etc.) if there is one.

Example shape:

> Commander, this is a flanking strike against a defended Construction
> Yard. You are given 4 medium tanks on the west edge. An enemy
> anti-tank line — 3 rocket infantry — sits across the middle of the
> map at x=50. Charging it head-on will cost the whole column. Slip
> your tanks around the line through the fog and destroy the
> unguarded Construction Yard sitting behind it at (100,20).

## 3. Plain English; no jargon without inline explanation

The audience includes models and humans without Red Alert context.
Explain RA terms inline the first time they appear in the briefing.

| RA term | Inline gloss |
|---|---|
| `pbox` / pillbox | "pillbox (stationary defensive turret)" |
| stance / hold-fire / Return-Fire | "on hold-fire orders — they will NOT shoot, even if the enemy walks under their guns" / "on Return-Fire (they only shoot back when shot at)" |
| kite / kiting | "fire from range, then move back before the enemy closes" |
| `harv` / harvester | "harvester (unarmed ore-collector truck)" |
| Tanya | "Tanya — an elite commando: fast, tough, with a pistol that one-shots riflemen" |
| Mammoth / `4tnk` | "Mammoth tank (very heavy)" |
| `fact` / `proc` / `powr` / `tent` / `weap` / `fix` | "Construction Yard / Ore Refinery / Power Plant / Infantry Barracks / War Factory / Service Depot" |
| guard bot | "they hold their post but lunge at any nearby enemy" |
| Lanchester's square law | rephrase as plain mechanics ("three tanks shooting one target will drop it fast; every enemy lost cuts their firepower") |

Avoid: `2tnk` / `e1` / `e3` shorthand, "DPS", "centroid", "out-trade",
"bracketing fire corridor", "leash radius", "aggro radius".

## 4. Complete sentences, real-general tone

- Use full sentences. Don't write "Denser block, same rule." Write
  "The enemy block is denser, but the rule is unchanged: …".
- Em-dashes are fine as appositives ("a wounded heavy tank at 35% HP —
  it hits harder than you, so a head-on fight loses"), but the clause
  after the em-dash must still complete the sentence.
- Tone is a briefing officer, not a barking drill sergeant. "Commander,
  this is a …" is the standard opener.
- Avoid "Commander, …, … — overwhelm.". Prefer full verbs:
  "overwhelm them before they break the line."

## 5. Win / lose conditions are mechanical, separate from briefing

- `win_condition` is a plain-English readout of the YAML predicate
  ("Destroy the rear Construction Yard with ≥2 tanks alive, within
  5400 ticks."). Numbers in real units.
- `lose_condition` lists the exact fail clauses, including the tick
  deadline ("Fewer than 2 tanks left, or deadline (5401 ticks).").
- Keep them short and concrete; do not repeat the briefing's narrative.

## 6. Force columns are positional + counted, not RA-jargon-y

`agent_force` / `enemy_force` use natural unit names with counts and
cell coordinates where they matter:

- `4× medium tank @(6,18-21) on Return-Fire stance`
- `1× harvester VIP (unarmed, 70% HP) + 4× medium tank escort`
- `23-cell vertical wall of pillboxes (stationary defensive turrets)
  at x=50 + 3× rifle infantry cluster behind it`

`@(x,y)` for a single spot. `@(x, y1-y2)` for a column. `×N` for count.
`sp0` / `sp1` etc. for seed-spawn-driven variants when present.

## 7. `map_fit` classification

Three values:

- **`fit`** — map proportions match the test. The agent's force and the
  enemy are at distances commensurate with the decision the scenario is
  meant to probe.
- **`wide`** — the map has significant empty traversal but the decision
  still bites. Candidate for tightening but not broken.
- **`large-trivial`** — most of the map is empty traversal; the decision
  under test is dwarfed by a long, decision-free march. The scenario
  becomes search-and-destroy instead of the advertised capability.

Rule of thumb: if the agent's spawn is at x≈6 on a 128×40 map and the
engagement happens at x≈50, that's >40 cells of empty drive before any
decision matters → `large-trivial`. A 40-cell engagement on a 128-cell
map = `wide`. A 20-cell decision on a 64-cell map = `fit`.

## 8. Enemy posture must match the mission premise

Each enemy unit has a `stance` and the enemy slot can declare a
`bot_type`. The posture set on the YAML must match what the briefing
claims the enemy will do. Mismatches are real defects — the model
follows the briefing's doctrine but the enemy never behaves the way
the briefing says.

Stance semantics (per `CLAUDE.md`):

| stance | name | behaviour |
|---|---|---|
| `0` | HoldFire | Never engages, even when attacked. Will silently die. |
| `1` | ReturnFire | Auto-fires on attacker only after taking hostile fire. Never advances. |
| `2` | Defend | Auto-fires on closest in-range enemy. Never advances. (Default if stance omitted.) |
| `3` | AttackAnything | Auto-fires AND advances toward nearest visible enemy. The only stance that opens new engagements by moving. |

Bot types (`enemy: {bot_type: …}` on the pack-level base):

| bot_type | role |
|---|---|
| `hunt` | actively closes on the agent's centroid |
| `rusher` | one-shot rush from spawn toward the agent |
| `patrol` | wanders a patrol path |
| `turtle` | defends own base, no excursions |
| `guard` | leashed defender — holds post, lunges at any enemy within ~16, snaps back past leash ~18 |

### Posture defect patterns to flag

| Briefing says… | Posture must include… | Wrong posture → defect |
|---|---|---|
| "the enemy will flee/run when struck" | st3 + east-of-cluster anchor, or scripted move | st2 → enemy never flees |
| "the enemy actively hunts/closes on you" | st3 OR `bot_type: hunt` (verify bot is unblocked by stance) | st0/1/2 alone → enemy stays put |
| "rush" / "incoming attack" / "wave" | st3 OR `bot_type: rusher`/`hunt` | st0/1/2 → no rush |
| "trade is unwinnable head-on; you must kite" | st2/st3 so enemy fires back | st0 → no incoming fire, doctrine moot |
| "ambush the heavies / counter the armour" | enemy armour st2+ so it fires back | st0 → tanks win, counter is not load-bearing |
| "concentrated rocket fire breaks the line" | line st2+ so it fires while being engaged | st0 → no return fire, mission trivialised |
| "defenders hold a chokepoint; you must storm" | enemy st3 or hunt bot — the attacker | enemy st2 — attacker stays put |
| Stance-management drill (agent has hold-fire) | enemy st0/st1 is INTENTIONAL — isolates the test | n/a |
| Stealth / search-and-destroy under fog | enemy st1 (ReturnFire) is intentional — they don't betray themselves | st3 would auto-reveal |

### When to flag vs change

- **Briefing matches mechanic** → no change needed (most common).
- **Defect** → either rewrite the briefing to match the actual mechanic, or change the stance/bot to match the briefing. Prefer changing the stance when the doctrine is the point of the test (because rewriting the briefing weakens the capability under test).
- **Verify** → run a scripted-policy probe to confirm the bot actually drives the unit. `bot_type: hunt` plus `stance:2` is suspect — `stance:2`'s "never advance" rule may block the bot's `OrderMove`. If it does, the test is silently broken.

## 9. Tick budgets noted, not edited (in audit phase)

`max_turns` and `tick_budget` are captured exactly as the YAML has them.
The audit phase does not change deadlines. If `max_turns` is unset on
a pack (rare, e.g. `rush-hour`), note `'unset'` so the YAML-edit phase
can fix it.

## 9.5. No solution leak — describe the situation, not the answer

The briefing tells the model WHAT IT HAS and WHAT IT MUST ACHIEVE. It
does NOT tell the model HOW to achieve it. The benchmark measures the
model's reasoning; if the briefing hands over the strategy, the model
isn't being measured — the briefing is being measured.

**Forbidden** in the description prose (allowed in top-of-file YAML
`# ENGINE NOTE` comments for contributors):

- **Per-policy outcome tables** — "4 tanks frontal lose 3/4; flank
  loses 1/4". Lookup table for the solution. Cut.
- **Strategy prescriptions by verb** — "Flank the line", "Kite the
  Mammoth", "Focus-fire the rocket infantry first". The model must
  derive doctrine from the forces vs the enemy + the constraint.
- **"Intended play is X" / "Optimal is X" / "The correct strategy
  is X".** Anywhere the briefing names the winning policy by verb.
- **Negative policy comparisons** — "A frontal charge costs all
  units"; "Stall yields only 2 kills"; "Sprinting through the
  middle dies to the line". Telegraphing what NOT to do.
- **Damage / income / kill arithmetic done for the model** — "Your
  3 tanks deal 6000 DPS to the Mammoth's 9000 HP". The model has
  units + observation; it can compute.
- **Exhaustive coordinate dumps** — Two or three load-bearing
  landmarks are fine ("Construction Yard at (44,4) NE", "ore
  patches east"); enumerating every cell is briefing-as-blueprint.

**Allowed**:

- The forces given ("4 medium tanks + 3 light tanks on west edge").
- The objective with a number ("Destroy the rear CY within 5400
  ticks").
- Relative-direction landmarks ("Construction Yard north-east",
  "ore patches east of your base", "rush approaches from south").
- Constraints / threats / stance / fog state ("on hold-fire orders",
  "raider approaches at ~tick 1500", "no losses allowed",
  "coordinates HIDDEN — only direction labels").
- Faction / tech context the model can't otherwise derive.

For Family-2 (economy) packs, the F2 supplement (§18 of
`EDIT_PRINCIPLES_FAMILY2.md`) adds economy-specific forbiddens
(per-harvester income tables, build-count prescriptions).

**Lint heuristic** for reviewers: if you delete the briefing and the
test suite still tells you the right policy, the briefing was
load-bearing for the model only. That's the bar.

## 10. Pack-edit phase (later) follows the same principles

When we move from the audit CSV to actually editing the YAML files, the
briefing/win/lose text inside the YAML uses the same prose conventions
above. Additionally:

- **Map resizing** for `wide` and `large-trivial` packs: shrink the
  arena (custom procedural map) so the agent's spawn and the engagement
  are roughly within one short march of each other. Default rule:
  cut the empty pre-engagement traversal to ≤15 cells.
- **Position rewrites**: keep relative positions of the engagement
  intact, just translate the whole layout to the smaller map.
- **No semantic change** to the capability under test or the win
  predicate — only the map size, actor positions, and prose change.
- **Engine bar still applies** (see `SCENARIO_REVIEW_CHECKLIST.md`):
  every stall/lazy/brute policy must still LOSE; the intended-capability
  policy must still WIN; tick budgets must remain reachable.
- **One pack per commit**, descriptive commit message ("audit: rewrite
  briefing + shrink map to fit for combat-flanking-attack").

## 10. CSV column contract

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | briefing_RA | win_condition |
lose_condition | max_turns | tick_budget
```

One row per (pack, level). All fields are quoted in the emitted CSV
so newlines or commas inside briefings don't break parsing.
