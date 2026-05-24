# Family-9 (Tempo / Time-pressure / Strategy / Adversarial) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-9.** Family-2 §18 (economy
no-leak supplement) and Family-3 §18-§26 (defense conventions) do NOT
inherit by default, but are referenced when a Family-9 pack has an
economy axis (the `tp-survive-and-grow`, `mid-economy-under-fire`,
`expansion-*` packs) or a defense axis (the `tp-survive-n-turns`,
`expansion-turtle-1-base-fortified` packs). If anything below
conflicts with F1, the F1 principle wins.

The single most important inherited rule is **§7 + §10's
map-resizing clause**: every pack must be classified `fit` / `wide` /
`large-trivial`, and every `wide` / `large-trivial` pack should be
shrunk to a bespoke procedural arena before the audit is "done".
Many F9 packs are still on `rush-hour-arena` 128×40 — those are the
backlog.

**Family-9-specific nuance**: adversarial / multi-objective / multi-
base packs can LEGITIMATELY justify a wide canvas because the
test's geometry inherently requires distance (two-body objectives,
opposing-corner garrisons, 3-base expansion). The F9 audit
distinguishes **`wide-justified`** (the wide canvas IS the test —
two-body strikes, multi-corner expansion) from **`wide`** (the wide
canvas is excess relative to the decision under test, and a tighter
arena would isolate the capability cleaner). `large-trivial` is
reserved for packs where the wide canvas is purely decision-free
march that dwarfs the actual capability probe.

---

## §53. Tempo / time-pressure packs (`tp-*`, `tempo-*`)

Tempo packs use TIGHT clocks to force a commit-vs-defer decision.
Three sub-idioms:

- **Quick-decision-under-clock** (`tp-decision-under-clock`,
  `tp-rush-objective-very-fast`, `tp-rush-multi-objective`) — the
  clock is short enough that "wait and see" loses on the timer
  while "commit blind" loses on the wrong target. The capability is
  budget-rationed reconnaissance + decisive commit.
- **Survive-then-strike** (`tp-survive-and-strike-at-window`,
  `tempo-strike-window`, `tempo-double-window`) — the win predicate
  is a `then:` chain: a survival/observe gate latches first
  (`after_ticks: T1` clause), then a strike clause becomes
  consequential. Premature engagement triggers a `premature_action`
  fail; late finish blows the within_ticks deadline.
- **Survive-N-turns / survive-and-grow** (`tp-survive-n-turns`,
  `tp-survive-and-grow`) — pure survival predicate: keep yard +
  units alive past `after_ticks: T` AND inside `within_ticks: T+x`,
  with a `scheduled_events: spawn_actors` wave (or hunt-bot raid
  band) generating the pressure.

**Tick reachability** (CLAUDE.md): non-interrupt mode advances
exactly `DEFAULT_TICKS_PER_STEP = 30` ticks per `env.step()`; max
tick at `max_turns` ≈ `30 · max_turns`. The historical F1/F2/F3
convention measures `93 + 90·(max_turns-1)` (the interrupt-mode
empirical step), which is the looser upper bound. Verify
`within_ticks` ≤ `93 + 90·(max_turns − 1)` AND that the intended
policy's reachable tick fits the window. A `tp-*` pack whose
intended policy CANNOT fit the budget is defective.

**Verify the "wait-and-see" stall LOSES on time**: every `tp-*`
pack must have `after_ticks: within_ticks+1` in its `fail_condition`
(real reachable clock LOSS). If `after_ticks` is missing or above
`30·max_turns`, the stall draws — F1 §5 violation.

## §54. Strategy dilemma / trilemma / two-body packs

Multi-objective packs with FORCING choices (can't have it all).
Each branch must be SOLVABLE individually but the wrong branch must
LOSE, AND a hedged "do a little of every branch" must LOSE.

- `strategy-dilemma` — two routes, one direct and lethal, one
  flanking and survivable, both reach the same yard+refinery.
- `strategy-trilemma` — three CAPEX arms (EXPAND, TECH, ARMY) with
  a cash budget that funds EXACTLY one. Hedging leaves no arm
  complete.
- `strategy-twobody` — two enemy bases on opposite map edges,
  both must be razed; one army can't reach both in time so
  forces must be SPLIT and committed simultaneously.
- `strategy-gauntlet` — one defended corridor; the test is
  attrition management (clear the gate cleanly, preserve the
  follow-up force for the economy).

The win predicate must encode the forcing choice via:
- `any_of` over branch-arms (`strategy-trilemma`),
- `all_of` over per-target clauses (`strategy-twobody`,
  `enemy_key_buildings_destroyed_in_region` per base),
- a tight `units_lost_lte` cap that punishes the un-survivable
  branch (`strategy-dilemma`).

**Two-body / multi-base packs LEGITIMATELY justify a wide map.**
The test IS "your one army cannot reach both bases — split or
lose." Shrinking the canvas trivialises the choice. Classify as
`wide-justified`, not `large-trivial`.

## §55. Adversarial / RPS counter-pick packs

True adversarial play needs the engine `step_1v1` channel (model
vs model). The current bench `adversarial-duel` opponent is a
scripted enemy with `stance:2` / `bot_type:''`; the "adversarial"
framing is partly aspirational. `adv-rps-counter-pick` (rock-paper-
scissors counter-pick under fog) is the pure capability probe —
the enemy archetype rotates per seed (`spawn_point` on the enemy
group, CLAUDE.md ENEMY-side rotation) and the agent must scout +
match the counter.

For RPS / counter-pick packs:
- The agent's build budget must fund EXACTLY one counter (no slack
  for "build a hedged army"; F2 §12 logic).
- The scout verb must be a real load-bearing observation step —
  fog OR `objective_coords: relative` — so a memorised "build X"
  opener loses on the seed where Y is the live archetype.
- Persistent enemy markers (the sentinel `fact`) MUST be
  duplicated across every enemy `spawn_point` group at identical
  coords (CLAUDE.md Wave-9 per-owner filter rule).

**`adv-asymmetric-weaker-must-win`** (an outnumbered/asymmetric
fight) is NOT RPS — it's a maneuver-not-attrition probe. The wins
require flank routing because head-on charges out-trade. Win
predicate uses `units_killed_gte + own_units_gte + has_building:
fact` to gate survival; the loss path is `not own_units_gte:1` for
unit wipe + `after_ticks` for the clock.

## §56. Art-of-war idioms (`artofwar-*`)

Three classical maneuver-warfare idioms. The briefing must describe
the SITUATION (the opportunity, the threat geometry), NOT the
doctrine (the maneuver name itself):

- **Indirect approach** (`artofwar-indirect-approach`) — a frontal
  wall blocks the lane; a flanking route exists but is longer. The
  briefing names the wall and the objective; the briefing does NOT
  say "use the indirect approach" or "flank the wall." The model
  derives the maneuver from forces + threat geometry.
- **Lure the tiger** (`artofwar-lure-the-tiger`) — a `guard`-bot
  defender holds post but lunges at proximity (CLAUDE.md
  `guard_bot` ~aggro 16, leash 18). A decoy unit pulls the guard
  off post; the main force then strikes through the now-undefended
  arc. The briefing names the leashed guard and the leash behaviour
  ("holds its post but lunges at any enemy within ~16 cells, then
  snaps back past ~18") — NOT "lure the tiger" by name.
- **Sequenced citadel** (`artofwar-sequenced-citadel`) — an ordered
  waypoint chain (A → B → C); intermediate stops latch
  prerequisites, the terminal seize is a timed window. The win
  predicate uses `waypoint_sequence` + `after_ticks` for the
  prerequisite hold + `within_ticks` for the citadel deadline.

Forbidden in any artofwar briefing: the doctrine name as a verb
prescription ("flank the wall", "lure the guard", "split your
force at A then push to C"). Allowed: the wall position, the
guard's leash behaviour, the waypoint coordinates as
location-labels for the win predicate.

## §57. Risk packs (`risk-*`, `reasoning-risk-*`, `reasoning-frontier-*`)

High-variance plays where the "safe" path loses on average. The
intended-policy must WIN deterministically — not "majority of
seeds". A non-deterministic capability is NOT a capability the
benchmark can grade. Every F9 risk pack reads as a
COST-COMPARISON problem the model must solve from observation,
not a probabilistic gamble.

Pattern in the current suite:
- `risk-blockade-bypass` — a heavy central garrison out-trades you
  if you push it; a lighter detour costs ~600 ticks but ≤1 loss.
  Win = `reach_region + within_ticks + units_lost_lte`. The cap is
  the teeth — a push loses on the cap, a stall loses on the clock.
- `reasoning-risk-route` — same shape: lethal short lane vs
  longer safe edge; the cap is `units_lost_lte: 0`.
- `reasoning-frontier-commit` — multiple candidate regions, only
  one carries the real survivor; tight clock permits exactly ONE
  decisive trip; decoys lie elsewhere. The capability is "pick the
  winner and commit", not exploration.

A risk pack must have at LEAST one survivable path that the
intended policy commits to. A pack where every path loses on the
cap is broken (no-cheat bar violation).

## §58. Expansion packs (`expansion-*`)

1-base / 2-base / 3-base expansion packs force trade-offs:

- `expansion-turtle-1-base-fortified` — single base, no MCV, must
  fortify; `pbox` × N + `gun` × M; hunt-bot raid wave incoming.
  Cross-references F3 §20 (`pbox` load-bearing) and §24 (build
  prereqs: `tent` + `fix` for `gun`).
- `expansion-balanced-2-base-defended` — NW starter base + spare
  MCV; deploy at distant target region; build a defensive screen
  at BOTH bases. Cross-references F2 (economy axis) and F3
  (defense axis).
- `expansion-aggro-3-base-greedy` — 2/3 MCVs (depending on tier),
  deploy at distinct corner regions, no economy/build phase. Pure
  multi-deploy logistics.

These packs LEGITIMATELY use a wide canvas (160×60, 192×80) —
the multi-base geometry IS the test. Classify as `wide-justified`.

Tech-gate cross-check is binding (F3 §24): a "build pbox" pack
must surface `tent` (and `powr` for power); a "build gun" pack
must surface `tent + fix`. A "deploy MCV" pack must include `mcv`
in the starter set (CLAUDE.md `deploy` is now wired end-to-end).

## §59. Mid-game switch packs (`mid-*`)

Mid-game decision points — concede-vs-hold, tech-switch-on-scout,
economy-under-fire. The model is dropped into a partially-played
state and asked to react. Three patterns:

- **Concede-vs-hold** (`mid-concede-vs-hold`) — two bases under
  simultaneous attack; one is salvageable, one isn't. The win is
  "hold one + concede the other" by consolidating the flex force.
  The briefing must describe the state ("two bases under attack,
  heavy on east, light on west"), NOT the answer ("consolidate
  on the west"). The current briefings LEAK ("consolidate the flex
  squad on the lighter WEST side, let EAST fall") — flagged.
- **Tech-switch-on-scout** (`mid-tech-switch-on-scout`) — full
  base, cash for ONE counter; scout the enemy composition first
  (latches a `then:` chain), then commit. The forcing structure
  is the same as RPS counter-pick (§55).
- **Economy-under-fire** (`mid-economy-under-fire`) — harvesters
  working while a raider tank probes the patch line. The
  capability is "don't chase — the static ring intercepts."
  Cross-references F2 §11-§15.

The decision-input must be OBSERVABLE in the obs (visible enemy
advance, cash counter, time elapsed). A concede-vs-hold pack
whose "heavy push" axis is identical-looking to the "light push"
axis is defective — the model can't pick the salvageable side.

## §9.5. No solution leak (repeated for F9 emphasis)

F1 §9.5 binding. F9-specific forbidden phrases:

- "Take the northern flank." / "Take the safer route." / "Pick the
  detour." — naming the route by direction.
- "Concede the EAST base, consolidate on the WEST." — naming the
  decision by side.
- "Pick ONE arm: EXPAND, TECH, or ARMY." with arm names spelled
  out as scripted-policy verbs.
- "Dispatch both prongs at once." / "Send both squads
  simultaneously." — naming the dispatch sequencing by verb.
- "Pull back to the lull safe-point at (55,36)." — naming the
  waypoint by coordinate + verb.
- "Use the lull to assemble at a launch point." — naming the
  pre-positioning maneuver by verb.

Several current F9 briefings leak this way (flagged in the audit
CSV `posture_issue` / leak column). The audit phase records the
leak; the YAML-edit phase rewrites the briefing per F1 §9.5.

Allowed (per F1 §9.5):
- The forces given ("3 tanks, 2 jeeps at the west; full base online").
- The objective with a number ("Raze both yards within 5000 ticks,
  ≤10 losses").
- Threat geometry ("a heavy garrison at (50,20) out-trades you
  head-on"; "two simultaneous hunt waves").
- Constraints / timing ("strike window opens at tick 3000",
  "scout the corner first to latch the prerequisite").

## §60. Engine plumbing notes (F9-specific)

- **`then:` win clauses** (`tp-survive-and-strike-at-window`,
  `mid-tech-switch-on-scout`) — the latches are ordered;
  Clause 1 must latch FIRST and remain true, Clause 2 evaluates
  after. The fail tree carries a `premature_action` clause
  (`units_killed_gte: 1` + `within_ticks: T1-1`) so a pre-window
  kill is an instant LOSS. Verify the premature clause references
  `within_ticks: T1-1`, not `T1` (off-by-one).
- **`scheduled_events.spawn_actors`** (`tp-survive-n-turns`,
  `expansion-aggro-3-base-greedy` medium/hard) — wave injection at
  declared ticks; the wave actor's `stance` controls hunting. A
  wave injected with `stance:3` (AttackAnything) closes
  aggressively; `stance:2` (Defend) holds at the spawn cell. The
  intended pressure profile must match the briefing.
- **`scheduled_events.shorten_deadline`** — clamps `max_ticks`
  DOWN mid-episode. Not currently used by any F9 pack in scope
  but available for "the clock just got tighter" twists.
- **`forbidden_tools:`** (`tp-pressure-procedural`) — a
  tool-allowlist that triggers an instant FAIL when an excluded
  verb is called. The bench's `tool_violations` counter is the
  BINDING rule; the `tools:` allowlist alone is advisory.
- **`step_1v1` channel** — the only channel for true adversarial
  (model-vs-model) play. The current `adversarial-duel` opponent
  is scripted; flag this in `posture_issue` for the YAML-edit
  phase to consider promoting (or downgrading the capability tag).
- **`spawn_point` per-owner filter** — CLAUDE.md Wave-9; relevant
  to every hard tier in F9. Persistent base buildings must be
  duplicated across every group at identical coords. Hard tiers
  in `tp-decision-under-clock`, `adv-rps-counter-pick`, and
  `mid-tech-switch-on-scout` use ENEMY-side rotation; most others
  use AGENT-side rotation.
- **`reveal_map: true`** — perception ablation control (F9 packs
  don't use it currently, but a tempo pack with `reveal_map: true`
  is the perfect "perception removed" control — the time-pressure
  decision should hold even with full vision).

---

## Family-9 audit CSV column contract

`audits/family9_tempo_strategy.csv` uses the F1 column set plus a
`wrong_strategy_loss` column (carried over from F2 §13 because the
forcing-choice idiom is the family's load-bearing structure):

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget |
wrong_strategy_loss
```

- `map_fit` values: `fit` / `wide` / `wide-justified` /
  `large-trivial`. F9 introduces `wide-justified` for adversarial
  / two-body / multi-base maps where the wide canvas IS the test.
- `wrong_strategy_loss` — one short sentence: which plausible
  wrong-strategy LOSES (stall, commit-blind, hedge, dither,
  over-scout, etc.) and why.

Tick-budget convention follows F1: report `93 + 90·(max_turns-1)`
as the tick ceiling (interrupt-mode empirical step). YAML's
`within_ticks` and `after_ticks` are read directly and reported in
the win/lose readouts.
