# Family-7 (Procedure / Strict / Maintenance / Robustness) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-7.** Family-2 economy rules
(§11-§17) and Family-3 defense rules (§18-§26) do NOT inherit by
default; the F2 §18 (no-solution-leak supplement) IS referenced where
a procedure pack carries an economy axis (the maint-* / rob-cash-* /
rob-multi-pressure packs). What follows are the ADDITIONAL
Family-7-specific rules. If anything below conflicts with F1
principles, the F1 principle wins.

Family-7 is the **procedural-compliance + adaptive-robustness** family.
The benchmark axis these packs probe is NOT tactical micro, defense
geometry, or economy reasoning — it is the model's ability to:

- follow an explicit ORDERED procedure (`waypoint_sequence` / `then`
  chain) under a restricted action API,
- honour a TOOL ALLOWLIST or `forbidden_tools` ban even under
  temptation,
- IGNORE irrelevant distractor tools by reasoning about relevance
  rather than by an explicit ban,
- evaluate a runtime CONDITION and dispatch to the matching IF/ELSE
  branch,
- RE-PLAN when an exogenous event (scheduled `spawn_actors`,
  `shorten_deadline`, `destroy_actors`) changes the operating
  conditions mid-episode,
- repair / sell / divest the right buildings under a maintenance budget.

The packs in scope (22 packs × 3 levels = 66 rows):

- **proc-\*** (10): `proc-checklist-no-deviation`,
  `proc-conditional-branch-action`,
  `proc-instruction-following-edge-case`,
  `proc-no-attack-passive-only`,
  `proc-only-build-no-combat`,
  `proc-only-defend-no-attack`,
  `proc-ordered-action-strict`,
  `proc-strict-toolban-fidelity`,
  `proc-tool-use-multi-distractor`,
  `proc-tool-use-with-distractor`.
- **strict-\*** (3): `strict-production-bom`, `strict-sequence`,
  `strict-toolban-fidelity-under-pressure`.
- **maint-\*** (2): `maint-repair-priority-order`,
  `maint-sell-and-recoup-cash`.
- **rob-\*** (8): `rob-cash-depletion-recovery`,
  `rob-deadline-shortened-midway`,
  `rob-multiple-simultaneous-pressures`,
  `rob-objective-change-midway`,
  `rob-objective-shift-with-or-clause`,
  `rob-partial-base-loss-continue`,
  `rob-unexpected-enemy-spawn`,
  `rob-unit-loss-recovery`.

---

## §42. Toolban packs (proc / strict)

A *toolban* pack constrains the agent's verb surface via TWO
overlapping mechanisms — the `tools:` allowlist on the pack base
(what the agent SEES as available in the prompt) and the
`forbidden_tools:` list on the level (what the BENCH counts as a
violation, incrementing the `tool_violations_gte` predicate). The
binding mechanism is the per-level `forbidden_tools` list — a model
that ignores the `tools:` hint and calls an attack verb anyway is
still graded by `forbidden_tools`. (See `eval_core._cmd_tool_name` +
`tests/test_forbidden_tools.py`.)

**Audit must cross-check that the intended-capability policy does
not need a banned verb.** If a pack's win predicate requires
`units_killed_gte` but the allowlist omits both `attack_unit` and
`attack_move` AND omits `set_stance`, the pack is structurally
unsolvable. The canonical proc-only-defend idiom solves this by
exposing `set_stance` on the allowlist so the defenders auto-fire
once lifted from `stance:0` HoldFire — the kills come from
stance-driven auto-fire, NOT from an attack call.

Footguns to flag:

| Allowlist | Win predicate | Solvable? |
|---|---|---|
| `[move_units, observe, stop]` | reach-region | ✓ |
| `[move_units, observe, stop]` | `units_killed_gte` | ✗ unsolvable — no attack verb, no stance verb (engine auto-fire only triggers if move_units en-route auto-fire connects, which is engine-side opportunism, not a guaranteed kill) |
| `[move_units, set_stance, stop, observe]` | `units_killed_gte:K` + defenders `stance:0` | ✓ (the load-bearing canonical idiom) |
| `[build, place_building, observe, stop]` | `building_total_gte` + `has_building:weap` | ✓ (no combat verb needed) |
| Full 9-tool palette + win is `reach_region` | reach + `units_lost_lte:0` | ✓ (the distractor cell — the relevant tool is `move_units`, the rest are noise) |

Also re-confirm: `move_units` auto-fire en route is ENGINE-side, NOT
a tool call, so it does NOT trip the `tool_violations` counter (per
CLAUDE.md). A move-only policy that has its column auto-defend
opportunistically en route is still procedurally clean.

The jeeps / riflemen carry an EXPLICIT `stance: 0` (HoldFire) in
several proc-* / strict-* packs to suppress the move_units en-route
auto-fire AND prevent the engine-default `stance:3` (AttackAnything)
from self-delivering the column at the far enemy `fact` marker
(which would let a stall WIN for free). Audit must verify the
explicit stance is present anywhere a stall-must-LOSE bar is
declared.

## §43. Mid-episode change (rob-\*) and observability

A *mid-episode-change* pack injects new state at a declared tick via
`scheduled_events:` — `spawn_actors` (a fresh reinforcement wave),
`destroy_actors` (a scripted strike razes some actors), or
`shorten_deadline` (the budget clamps DOWN to a smaller `max_ticks`).
The changed state MUST be discoverable from the observation channel
(`units_summary`, `buildings_summary`, the minimap, or an interrupt
payload), NOT from external hints.

In the F7 scope:

- `maint-repair-priority-order` injects a grenadier strike at
  tick 720 via `spawn_actors`; the model can detect it via
  `units_summary` and `enemy_unit_spotted` interrupt.
- `rob-deadline-shortened-midway` fires `shorten_deadline` at
  tick 1000 clamping `max_ticks` to 2000.
- `rob-partial-base-loss-continue` fires `destroy_actors` at
  tick 1200 razing the SOUTH outpost.
- `rob-cash-depletion-recovery` does NOT use `scheduled_events` —
  the strike is pre-placed `hunt` 4tnks that path to the proc at
  t=0. The "mid-episode" framing comes from the rule, not from a
  scheduled hook.
- `rob-unexpected-enemy-spawn` does NOT use `scheduled_events` —
  Wave 2 is pre-placed at a fog corner from t=0 and only becomes
  visible when it closes in. (The "surprise" is fog-of-war
  occlusion, not a scripted spawn.)
- `rob-multiple-simultaneous-pressures` does NOT use
  `scheduled_events` — all three threat axes (hunt squad, raid
  tanks, tech deadline) are live at t=0.
- `rob-objective-change-midway` does NOT use `scheduled_events` —
  the "objective change" is structural (a `then:` ordered chain
  where phase 2 only counts after phase 1 latches).
- `rob-objective-shift-with-or-clause` does NOT use
  `scheduled_events` — both objectives are present from t=0 and
  the agent picks the feasible one via `any_of`.
- `rob-unit-loss-recovery` does NOT use `scheduled_events` — the
  "loss" is rhetorical: the column starts at 4/5 of the
  establishment quota; there is no actual mid-episode destruction.

**Observability defect to flag:** if a rob-\* pack's mid-episode
change is NOT signalled in the observation channel AND the briefing
does not tell the model the rule (e.g. "watch for a second wave at
the NE corner"), the pack tests memorisation of an external hint,
not adaptive re-planning. The fix is either (a) add an interrupt
(`enemy_unit_spotted` / `own_building_destroyed`) so the change
surfaces as a hard signal, or (b) make the observation channel
sufficient (the new actors land within agent vision).

**Briefing pre-disclosure trade-off:** several rob-\* packs (notably
`rob-deadline-shortened-midway`, `rob-unexpected-enemy-spawn`,
`rob-partial-base-loss-continue`, `rob-unit-loss-recovery`)
pre-disclose the mid-episode rule in the briefing prose ("at tick
1000 the deadline gets cut", "a second wave is hidden at a fog
corner"). This is a controlled leak: the model is told the rule
upfront so the capability under test is the REACTION (commit fast /
keep reserves / press on / build the missing unit) rather than the
DISCOVERY. Audit notes this as a leak when the rule essentially
hands over the strategy ("commit four tanks east at full speed from
turn 1"); audit accepts it when the rule is structural (e.g.
"establishment requires 5 tanks; you have 4") and the action is
still derivable.

## §44. Distractor packs (proc-tool-use-\*)

A *distractor* pack lists a LARGE tool surface but only ONE tool is
load-bearing for the win. There is NO `forbidden_tools` list (the
distractor tools are permitted, just useless). The discrimination is
whether the model REASONS about which tools are RELEVANT and ignores
the rest — every irrelevant tool call either replaces the units'
move order, lures units off-path, or burns a turn that the
within_ticks budget cannot spare.

The two F7 distractor packs are:

- `proc-tool-use-with-distractor` — full palette + ONE off-lane
  passive sentry (easy), a patrol harasser (medium/hard). The win
  is reach + zero losses.
- `proc-tool-use-multi-distractor` — 9 tools + a passive off-path
  garrison + (on hard) a central corridor sentry stack. The win is
  reach + zero losses on a TIGHT clock.

Audit must verify:

- Distractor tools are *plausible* — the agent has a `fact` + cash
  so `build` actually queues something instead of silently no-op-ing
  (the temptation must be operational, not theoretical). A
  build-distractor cell with `starting_cash: 0` and no `fact`
  would be a non-distractor (the model can SEE the tool is dead
  from turn 1).
- Distractor tools are NOT load-bearing — the win predicate does
  not require any of them. If the win predicate is
  `building_total_gte:6 + has_building:weap`, the `build` tool is
  load-bearing, not a distractor.
- The tick budget genuinely BITES on a distractor-spam policy. The
  intended-policy travel time should leave a comfortable margin
  (≥ 2×) over the deadline so a couple of probing `build` calls
  don't accidentally win — but a policy that fixates on the build
  palette every turn must miss the deadline.

**Footgun observed:** `proc-tool-use-multi-distractor` hard tier
claims the central sentry stack "brackets y=18..22 with stance:3
fire arcs" but the YAML actor declarations set the sentry stack to
`stance:0` (HoldFire). The corridor denial is therefore weaker than
the briefing claims — the model can drive straight through y=20
without taking fire. Whether this matters depends on whether the
spawn lanes (y=4..6 NORTH / y=34..36 SOUTH) require crossing the
central band; in this pack they don't, so the discrepancy is
cosmetic but worth flagging.

## §45. Conditional branch (proc-conditional-branch-action)

A *conditional branch* pack hands the agent an explicit IF/ELSE
runbook and gates the WIN via a two-branch `any_of` where exactly
one branch is satisfiable per seed. The agent must (1) scout to
OBSERVE the condition, (2) BRANCH on what it sees, (3) EXECUTE the
matching procedure.

The trigger MUST be in `signals` (a unit / building that the agent
discovers via `buildings_discovered_gte` or `enemies_discovered_gte`
or the live `units_summary` / `enemy_buildings` channel), NOT
narrated in the briefing.

In the F7 scope, only `proc-conditional-branch-action` is a true
conditional-branch pack. The condition (outpost flank) flips per
seed via the Wave-9 ENEMY-side `spawn_point` axis: spawn_point=0
puts the enemy `tent` outpost NORTH; spawn_point=1 puts it SOUTH.
The agent base is FIXED (no agent `spawn_point`). The model
discovers the live flank by scouting a jeep into vision.

Audit must verify:

- The condition flips by seed (the `spawn_point` round-robin in
  the env round-robins seeds 1..4 → groups 1,0,1,0 — seeds 1/3
  one branch, seeds 2/4 the other).
- The non-live branch is STRUCTURALLY DEAD on each seed (no SOUTH
  tent exists when the enemy is NORTH, so branch B clause 2 can
  never latch on a NORTH seed; a fixed "always SOUTH" policy
  cannot accidentally win).
- The condition is REACHABLE by the scout (the easy tier
  pre-positions the scout in vision so the BRANCH skill is tested
  without the scout-move skill; medium/hard require an explicit
  scout move).
- The persistent far-east enemy `fact` is duplicated across BOTH
  enemy spawn groups so it lands every seed (CLAUDE.md per-owner
  `spawn_point` footgun).

## §46. Strict-sequence and ordered packs

A *strict-sequence* pack enforces ORDER via either:

- `waypoint_sequence` (n:1 latching) — Wk+1 only counts AFTER Wk
  was tagged (`proc-checklist-no-deviation`, `strict-sequence`).
- `then:` composite over a list of clauses — phase 2 only opens
  after phase 1 latches (`rob-objective-change-midway`,
  `rob-cash-depletion-recovery`'s `then:` chain).
- `then:` over `building_in_region` clauses + explicit out-of-order
  TRAPS in `fail_condition` (`proc-ordered-action-strict`).
  Note: the `then` operator alone is greedy late-credit (see
  `tests/test_then_composite.py`), so an out-of-order placement
  would be retroactively credited. The fail traps close the hole.

Audit must verify:

- The latching mechanism actually prevents wrong-order WIN. For a
  pure `waypoint_sequence` pack this is automatic; for a `then:`
  over building placements the per-tier `fail_condition`
  out-of-order traps must enumerate every pair (B-before-A,
  C-before-A, C-before-B, ...).
- The tick budget bites within `max_turns`. `tick ≤ 93 + 90·(max_turns-1)`
  is the F1 ceiling convention; every `within_ticks` / `after_ticks`
  must sit below the ceiling so a non-finisher LOSES on the
  deadline rather than draws.
- Geometry doesn't accidentally satisfy the latch via path
  coincidence. The pack-side note for `proc-checklist-no-deviation`
  explicitly states "the geometry deliberately makes a
  beeline-to-final impossible to satisfy the latch (W3 at mid-y,
  W1 north, W2 south — a straight east-bound move from the start
  column does NOT pass within radius 5 of W1 or W2)" — this
  invariant is the load-bearing test of the strict-order skill.

## §47. Repair / sell (maint-\*) — building actor ids

The `maint-*` packs use the `repair` verb (`maint-repair-priority-order`)
or the `sell` verb (`maint-sell-and-recoup-cash`) which require the
engine to surface the building actor id in the obs channel. This is
a post-engine-fix capability (see CLAUDE.md "Building actor ids ARE
surfaced for `repair` / `sell` / `power_down` / `set_primary`"). Audit
must confirm the pack:

- exposes `repair` / `sell` in `tools:`,
- pre-places the target buildings with an explicit `health: N`
  percentage (now honoured post-engine-fix per
  `openra-data/tests/test_actor_health.rs`),
- structures the wrong-priority LOSS clearly: the win predicate
  must reference the HIGH-VALUE buildings (`proc`, `weap`, `fact`
  for maint-repair) and a decoy/no-op repair on the LOW-VALUE
  buildings (`pbox`, `fix`) wastes the maintenance window. A
  predicate of "any building still alive" lets a panic-repair on
  the loudest wear indicator accidentally satisfy.

`maint-repair-priority-order` uses an explicit
`after_ticks: 900` IN the win clause (NOT just in fail) — this is
the CLAUDE.md "`after_ticks` in WIN is structurally incompatible
with ConquestVictoryConditions" footgun, accepted here because
the agent has NO attack tools and no units, so the engine's
conquest auto-`done` cannot fire. Audit notes this as a known
acceptable exception, not a defect.

`maint-sell-and-recoup-cash` requires the agent to recognise the
indivisible reserve cannot cover the critical purchase from
starting cash alone — the sell-refund-then-buy chain is the only
solution. Audit must verify:

- the obsolete asset cluster (pbox / tsla / hbox / dome) sells for
  EXACTLY enough refund to fund the critical purchase
  (cash + total refund ≥ weap + 3×2tnk, with small slack),
- the agent has no income source (`proc` with no reachable ore
  patch) so cash cannot refill,
- the `building_count_gte:{fact}` clause guards a present-tense
  survival check (not the latched `has_building` accumulator).

## §9.5 + §10 + §17 (inherited)

F1 §9.5 no-solution-leak still applies. A briefing line like "the
intended play is to sell the obsolete defences and reinvest into the
war factory" is a leak — it names the winning policy by verb. The
allowed shape is "starting cash $3500, the war factory needs $2000
and three tanks need ~$2400, no income" — the model derives the
recipe.

F1 §10 map-shrink discipline applies to every F7 pack that sits on
`base_map: rush-hour-arena` 128×40. The proc-checklist / strict-
sequence / rob-deadline packs explicitly USE the 128-cell traversal
(visiting waypoints across the full arena, racing to the far-east
fact) — for those packs `map_fit` is `fit`, the 128-cell sweep IS
the test. For the proc-only-build / proc-only-defend /
proc-tool-use-* / strict-toolban-* / maint-* / rob-multi-pressure
packs the decision (build, defend, recall, allocate cash) happens
in a tight base footprint at the WEST edge with the deep east used
only as a sentinel-fact landing strip — these are `wide` (the
decision still bites but most of the arena is empty traversal). A
custom 64×40 or 96×40 arena would tighten the visualisation, but
the procedural-compliance signal is intact on either map. The
audit captures this distinction in `map_fit`.

F1 §17 (and PRODUCTION_TECH_AUDIT.md) tech-tree cross-check applies
to every F7 pack with `build` in `tools:` — confirm prereqs:
`weap` needs `proc`, `2tnk` needs `weap`+`fix`, `pbox` needs `tent`,
`tsla` needs `weap`+`tent`+`dome` (Soviet path) etc.
`strict-production-bom` hard tier requires `tsla` and explicitly
flags the need for extra `powr` to keep `power_surplus_gte:0` — a
tech-tree audit must confirm the prereq chain (proc → weap → tent →
dome → tsla on the Soviet side) is achievable inside the
starting-cash budget.

## §48. Robustness packs — "real LOSS, not DRAW" still applies

F1 §5 (no draw degeneracy) applies in full. A `rob-*` pack that
silently degenerates to DRAW on a stall (because the engine
auto-`done`s on enemy-elim before the deadline bites, or because
the win clause latches early without the mid-episode change firing)
is a defect, not a hard pack. The recurring guard idiom is:

- Persistent unarmed enemy `fact` far east keeps the episode alive
  past any enemy-elim auto-`done` (every F7 pack uses this).
- `after_ticks: T+1` in `fail` is reachable inside `max_turns`
  (verify `T+1 ≤ 93 + 90·(max_turns-1)`).
- Present-tense `building_count_gte` instead of one-shot
  `has_building` for survival clauses (so a destroyed fact
  immediately trips the fail clause).
- For `rob-multi-pressure` the harv count is in BOTH win AND fail
  via `unit_type_count_gte:{harv,1}` / `not ...` — a focus-tech
  policy that abandons the harv ring trips fail.

## §49. F1 CSV column contract (Family-7 extension)

`audits/family7_procedure.csv` extends the F1 CSV with two
procedural columns AND keeps the F3 enemy_posture columns:

```
pack | level | capability | map_name | map_size | map_fit | tools |
forbidden_tools | scheduled_events | agent_force | enemy_force |
enemy_posture | posture_issue | briefing_RA | win_condition |
lose_condition | max_turns | tick_budget
```

- `forbidden_tools` — comma-separated list from the per-level
  `forbidden_tools:` (or `[]` when the pack uses the outcome-graded
  distractor idiom).
- `scheduled_events` — short tag: `spawn@T:type×N`,
  `destroy@T:filter`, `shorten@T:new_max_ticks`, or `none`.

Same `map_fit` discipline as F1: any `wide` / `large-trivial` row is
a backlog item for the YAML-edit phase to shrink to a bespoke arena.
F7 packs disproportionately sit on `rush-hour-arena` 128×40 because
the procedural test doesn't intrinsically need a tight engagement
zone; many of these are `wide` not `large-trivial` (the deep east is
the sentinel-fact landing strip, not a long pre-engagement march).

Tick budget convention follows F1: `tick_budget = max_turns × 90 + 3`
(the empirical interrupt-mode step convention; the YAML's actual
`within_ticks` / `after_ticks` are recorded in the win/lose
readouts). For non-interrupt-mode packs (no `interrupts:` block) the
real per-step advance is 30 ticks but the audit holds the column
constant for cross-family comparability.
