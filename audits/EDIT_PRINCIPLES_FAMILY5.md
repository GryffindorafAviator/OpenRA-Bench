# Family-5 (Long-horizon) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-5.** It also inherits the
Family-2 §18 no-solution-leak rule (which is the F1 §9.5 rule
restated for economy verbs; F5 packs frequently include economy +
build axes so the F2 wording is binding here too).

Family-3 §17-§26 conventions (tech-prereq cross-check; `pbox`
load-bearing; stance discipline; scheduled_events idiom; wave success
asymmetry) apply WHEREVER the F5 pack carries a defense axis. The 14
F5 packs collectively touch every other family's verb-set —
construction (build/place_building), economy (proc/harv), combat
(attack_move/set_stance), perception (scout/jeep in scout band),
disaster recovery (scheduled_events.destroy_actors) and superweapons
in some sibling work — so the F5 audit is a CROSS-FAMILY synthesis.
The rules below are the ADDITIONAL discipline that long-horizon-
specific packs need.

The single most important inherited rule is **§7 + §10's map-resizing
clause**: every pack must be classified `fit` / `wide` / `large-trivial`,
and every `wide` / `large-trivial` pack must be shrunk to a bespoke
procedural arena before the audit is "done". F5 packs are
overwhelmingly on `rush-hour-arena` 128×40 (or larger custom 160×60
arenas for the build-army / credit-only / defense-tech packs); a
long-horizon CAPABILITY does NOT justify a long-horizon MAP — the
test is the sequencing decision, not the march across empty cells.
Default rule (carried forward from F2/F3): cut empty pre-engagement
traversal to ≤15 cells PER PHASE, and cap total arena size to the
travel distance the intended-play actually has to cross.

---

## §32. Phase markers must be testable AND load-bearing

Long-horizon packs (multi-checkpoint, stage-locked progression,
3-phase/4-phase/5-phase macro chains) live or die on whether each
phase predicate is GENUINELY satisfiable only by the phase activity
the briefing names. Use the Wave-2 `then:` happened-before composite
to gate each phase; never use a flat `all_of` of terminal-only
phase markers as the chain spine (the assault army marching east
auto-satisfies "buildings discovered" or "phase reached", collapsing
the chain into one tick).

Bad patterns to flag:

- A 3-phase chain where the terminal clause (`enemy_key_buildings_
  destroyed`) implies all earlier clauses (`buildings_discovered_gte:
  1`, `own_units_gte: 1`) — discovering and owning are both true the
  instant the assault force walks into the target's vision, so the
  chain collapses to one evaluation. Fix: rekey earlier phases on
  REGION presence (`units_of_type_in_region_gte` — see
  `lh-tech-pivot-attack`'s scout-band-keyed jeep clause) or on
  EXPLICIT building-state milestones (`has_building: weap` is not
  satisfied by attack motion).

- A "scout" phase keyed on `buildings_discovered_gte` where the
  natural march path crosses the target's vision band. The discovery
  fires for free when the army walks in.

- A "produce N units" clause where N is already met by pre-placed
  starters. The phase latch fires at t=0 and provides no test —
  remove the pre-placed unit count from the threshold, or raise N
  above starter count.

Good patterns (lifted from the F5 packs that survive scrutiny):

- `lh-progression-stage-locked` STAGE 5 (`units_killed_gte`) is paired
  with a STAGE-defense fail clause (`enemy_buildings_destroyed_gte:1`)
  so a hunt-everything brute that razes a sentinel fact LOSES instead
  of drawing — the kill-stage stays a UNIT-kill, not a base-raze.
- `lh-opening-to-defense-to-counter` PHASE-2 (`units_killed_gte: N,
  after_ticks: T1`) pairs a kill count with a HOLD-PAST-TIME guard
  so a fast win cannot bypass the defence window.
- `lh-tech-pivot-attack` SCOUT clause is keyed on `units_of_type_
  in_region_gte: {type: jeep, x:120, y:18, radius:12, n:1}` — the
  jeep's POSITION, not building-discovery — so the assault army
  cannot satisfy it for free.

## §33. Tick budgets ARE the test

Reachable max tick (non-interrupt mode) ≈ `93 + 90·(max_turns − 1)`.
Long-horizon packs run `max_turns` 60-160; the discipline is twofold.

**A. The deadline must bite.** `within_ticks` and the `fail_condition`
`after_ticks: T+1` must both be ≤ reachable max. F1 §5 violation:
`within_ticks` above reachable tick collapses the run to DRAW. Audit
every F5 pack against `max_turns × 90 + 3` as the ceiling.

**B. The deadline must SLACK ENOUGH for the intended play.** Unlike
F1/F3 where a clock is the anti-stall teeth, an F5 chain's intended
play has 3-6 sequenced phases AND must traverse 100+ cells for the
counter. The clock must fit `(build_time × N) + (march_distance ÷
move_speed)` with realistic budget. Empirical pin: easy/medium F5
packs leave 10-30% headroom (e.g. `lh-opening-to-tech-to-army` easy
max=100 → reachable 9003, within=8999 — slack ~0%, but intended
play finishes at ~8500; medium max=80 → reachable 7203, within=7200
— very tight).

A pack whose intended play finishes at ~95% of `within_ticks` is on
the edge of a perfectly-played LOSS — flag as `tight-clock` in the
audit.

**Interrupt-mode caveat.** Packs that set `interrupts:` (e.g.
`lh-opening-to-defense-to-counter`, `lh-tech-rush-vs-army-rush`,
`lh-recovery-after-mid-game-loss`) advance variable ticks per turn
(default `max_ticks=5`, but turn ends on the first interrupt). The
ENGINE NOTE in those packs reads the actual `info["ticks_advanced"]`
rather than the 90-ticks/turn estimate. The F1 audit-CSV convention
is to still report `tick_budget = max_turns × 90 + 3` (CLAUDE.md
acknowledges the historical convention; the actual fail check uses
the YAML `after_ticks` value verbatim).

## §34. Mid-episode events are the canonical surprise vector

`scheduled_events:` (Wave-9 engine feature, pinned by
`OpenRA-Rust/openra-data/tests/test_scheduled_events.rs`) is the
authoritative mechanism for "react to a mid-episode event". Three
kinds:

- `spawn_actors` — inject reinforcement waves (Family-3 §22 idiom).
- `destroy_actors` — wipe agent assets in a region (the canonical
  disaster-recovery event — `lh-recovery-after-mid-game-loss`
  destroys the proc + a forward tank cluster at tick 1500).
- `shorten_deadline` — clamp `max_ticks` DOWN mid-episode (none of
  the 14 F5 packs use this — candidate idiom for a "the clock just
  moved up" surprise).

A long-horizon pack that wants to test REACT-TO-SURPRISE without
`scheduled_events` is structurally weaker: the agent observes the
threat at t=0 and pre-plans. Recovery / mid-game-disaster packs
MUST use `scheduled_events.destroy_actors`; "wave-during-build"
defense packs MUST use `scheduled_events.spawn_actors`.

Gate the win behind `after_ticks: T_event + 100` (or similar) so a
race-the-clock policy cannot bypass the event window — see
`lh-recovery-after-mid-game-loss` easy: `after_ticks: 1600` ≥ event
tick 1500, so the proc clause is true at tick 1600 only if rebuilt.

## §35. Recovery-after-loss uses `health: N%` (engine wired)

The pre-placed actor `health: N` field (PERCENTAGE 1-100, engine
fix pinned by `openra-data/tests/test_actor_health.rs`) is the
basis for the repair-triage / disaster-recovery idiom: a pack can
spawn buildings at e.g. `health: 40` so the model must `repair`
before the wave hits. `lh-recovery-after-mid-game-loss` uses the
`scheduled_events.destroy_actors` variant instead (the wipe is total,
not partial) — both are valid recovery idioms. Future F5 recovery
packs (a "you start at 35% HP — survive the next wave") should use
the `health:` field; the engine handles it natively.

## §36. Multi-phase win predicates: every clause must be load-bearing

A `then:` chain or `all_of` win composite is only as strong as its
weakest clause. Each clause must be:

1. **Independently testable.** A clause that fires for free
   (pre-placed actors satisfy it, or natural play satisfies it
   without the briefed activity) is redundant. The audit must flag
   clauses where the discriminating policy (stall / brute / wrong-
   path) IS NOT failed by the clause.

2. **Ordered correctly.** `then:` advances strictly clause-by-clause;
   `all_of` does not enforce order but a final terminal clause that
   implies earlier ones (e.g. razing the fact implies discovering
   it) collapses an `all_of` to one tick.

3. **Compatible with the rest of the predicate.** `after_ticks` in a
   WIN clause requires a persistent enemy `fact` marker (see
   CLAUDE.md auto-done footgun + §10 below) so the engine doesn't
   end the episode the instant the last MustBeDestroyed building
   falls.

The 14 F5 packs use `then:` for 9 of them and a flat `all_of` for
the remaining 5 (`lh-100-turn-marathon-survival`,
`lh-tech-rush-vs-army-rush`, `lh-recovery-after-mid-game-loss`,
`longhorizon-opening-to-assault`, partly `lh-opening-to-defense-to-
counter`). The `all_of`-only packs are valid when their clauses are
either:

- Single-event clauses with no cross-implication (the marathon's
  4-clause "yard alive AND ≥3 units alive AND ≥2 pbox built AND
  after_ticks survived" — none imply each other), or
- Backed by a `scheduled_events` mid-episode trigger that enforces
  the ordering EXTERNALLY (recovery uses `after_ticks: 1600 ≥
  disaster tick 1500` so the proc clause cannot pre-satisfy).

A flat `all_of` without either of those guards is `chain-leak` —
flag it.

## §9.5 inherited — no solution leak

The F1 §9.5 / F2 §18 no-solution-leak rule is BINDING. F5-specific
forbidden phrasings (lifted from common leak patterns in the
existing long-horizon descriptions):

- **Per-phase prescriptions.** "First build the refinery, then the
  war factory, then four tanks, then attack." This is the chain
  spelled out for the model. The briefing should describe the
  STATE (forces, enemies, deadline) and the OBJECTIVE (the win
  clauses). The model must derive the ORDER from the `then:`
  semantics revealed in the win condition.

- **"Phases must happen in order" + the order.** "Phases must happen
  in this order: scout, react, counter" leaks the same information.
  The fact that there are phases is a structural feature of the win
  predicate — describing them in the briefing as a recipe is a
  leak.

- **Per-policy outcome enumerations.** "Stalling loses on the clock.
  Skipping the opening loses because the rush razes the base.
  Rebuilding without attacking loses on the kill bar." This is the
  policy-comparison table the F1 §9.5 rule forbids — a lookup table
  for what NOT to do.

- **"Optimal play is X" / "The intended sequence is X".** Anywhere
  the briefing names the winning policy by verb.

ALLOWED in F5 briefings:

- The forces and enemies present (with relative-direction
  landmarks).
- The numbers in the win condition ("kill ≥6 rushers AND yard
  alive AND ≥3 pillboxes within 5400 ticks").
- The fact that the win has multiple clauses (the model sees the
  YAML predicate at evaluation time — the briefing can summarise
  it as "five things must be true at the deadline" without
  spelling out the order).
- Constraints: "tech is already up", "tech starts unpowered", "the
  base will be sabotaged mid-game", "no losses allowed".

**Lint heuristic:** if you delete the briefing and the win predicate
still tells a competent model what to do, the briefing was load-
bearing for the model only. F5 briefings should be situation
reports, not strategy guides. The 14 existing F5 packs are flagged
with `leak` in the audit CSV where they currently violate this.

## §10 inherited — map-shrink rule still applies

The 14 F5 packs largely use `rush-hour-arena` 128×40 or a custom
160×60 generator arena (for `lh-build-army-coordinate-multifront-
attack`, `lh-credit-only-final-phase`, `lh-defense-tech-second-base`).
The shrink discipline:

- **Single-front packs on 128×40 with the action at x<60** are
  `large-trivial` (60+ cells of empty east traversal). Candidates
  to shrink to 80×40 or 96×40.
- **Two-corner packs on 160×60** (build-army-multifront, credit-
  only) are legitimately wide — the 120-cell NE↔SE separation IS
  the test, so `fit` rather than `wide`. Document the separation
  as a load-bearing feature.
- **Multi-phase progression packs** where each phase happens at a
  distinct x-band can be `fit` even on a long map (the march
  distance IS the credit-assignment test). Document the phase-by-
  phase geometry.

The audit row classifies each (pack, level) as `fit`/`wide`/`large-
trivial` per the F1 rule of thumb: if the agent spawn is at x≈10 and
the engagement at x≈115 (typical F5 layout), that's >100 cells of
empty drive → `large-trivial` UNLESS the long march IS the test (e.g.
the credit-only pack's whole point is the 120-cell commit). In F5,
"the long march IS the test" applies to packs whose `capability` is
specifically credit-assignment or distance commitment; everywhere
else (defense-while-build, recovery, tech-rush) the long march is
incidental and `large-trivial` applies.

## §17 inherited — tech-prereq cross-check

F5 packs that include `build`/`deploy` in `tools:` must satisfy the
F3 §24 / `PRODUCTION_TECH_AUDIT.md` tech-prereq chain:

| Buildable | Prereqs (Allies) | Cost | Build sec |
|---|---|---|---|
| `pbox` | `tent` (Defense queue) | 400 (NB. F3 doc says 600; the engine value is the canonical one — check `gamerules.rs`) | 12 |
| `proc` | `fact` (`anypower` for placement) | 1400 | 28 |
| `powr` | `fact` | 300 | 8 |
| `tent` | `fact` | 400 | 10 |
| `weap` | `proc` | 2000 | 40 |
| `fix` | `weap` | 1200 | 24 |
| `2tnk` | `weap` + `fix` | 800 | ~12 |
| `1tnk` | `weap` | 700 | ~10 |
| `e1` | `tent` | 100 | ~3 |
| `e3` | `tent` | 300 | ~5 |
| `jeep` | `weap` | 600 | ~6 |
| `gun` | `tent` + `fix` | 600 | 14 |

Cross-check every F5 pack: if the briefing or win condition names
`2tnk` (medium tank), the pack must give `weap + fix`. If it names
`weap`, the pack must give `proc`. The F5 audit CSV's
`tech_chain_ok` column records this.

Several F5 packs (the chain packs) deliberately pre-place SOME of
the chain (e.g. `lh-opening-to-tech-to-army` pre-places `fix` so the
chain's "tech up" phase is `build('weap')`, not also building the
service depot). This is GOOD design — the test should be the SINGLE
build verb the phase names, not a hidden sub-chain.

---

## Family-5 audit CSV column contract

`audits/family5_longhorizon.csv` uses the F1+F2 columns plus three
long-horizon-specific columns:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget |
phase_chain | chain_idiom | leak_flags
```

- `phase_chain` — short summary of the win clauses ("powr → proc →
  M=3500 → weap → kill 2" / "scout (jeep band) → counter (4×e3) →
  raze fact" / "open (powr+proc) → defend (7 kills, hold T) →
  counter (raze far east)"). Records the chain length and kind for
  cross-reference with `lh-multi-checkpoint-5-plus`.
- `chain_idiom` — one of `then`-strict / `all_of`-terminal /
  `all_of`-scheduled / `hybrid`. Flags packs that use a flat
  `all_of` without scheduled-event ordering (potential chain-leak).
- `leak_flags` — comma-separated list of detected leak patterns in
  the YAML description (`per-phase-prescription`, `outcome-table`,
  `order-spelled-out`, `clean`).

Same `map_fit` discipline as F1/F2/F3: any `wide` / `large-trivial`
row is a backlog item for the YAML-edit phase. F5 packs with
legitimately-load-bearing long marches (credit-only, build-army-
multifront) keep `fit` even on 160×60 — see §10 above.

Tick budget convention follows F1: `tick_budget = max_turns × 90 + 3`
recorded for cross-family consistency; the actual fail cutoff is
the YAML `after_ticks` value, reported in the win/lose readout.
