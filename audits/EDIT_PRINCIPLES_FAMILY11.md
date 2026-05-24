# Family-11 (Full-Game Sequence) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-11.** Family-2 §18 (no
solution leak, no per-policy / income / build-count arithmetic in
the briefing) is binding here. Family-3 §22 (`scheduled_events` for
timed pressure), §24 (tech-prereq cross-check), §26 (two-spawn-point
hard tiers) inherit because every F11 pack carries a build + a
defense axis. Family-5 §32-§36 (multi-phase win predicates,
phase-marker load-bearing, mid-episode events, `then:` discipline)
inherit because F11 IS the longest-horizon family in the suite —
its packs ARE 4-6-phase chains. Family-6 §37-§44 (build-chain
validation, parallel production, tech-gate prereqs, build-time vs
`max_turns`) inherit because every F11 pack requires building
production buildings (`weap`, `hpad`, `syrd`, `fix`). Family-8 §48
(per-base load-bearing role) inherits where an F11 pack spans
multiple bases.

If anything below conflicts with F1 / F2 §18, the F1 / F2 §18 rule
wins. F11's specific contribution is the **full-arc integration
test**: plan → cash → build → produce VERTICAL force → attack — the
single family where every other family's verb-set must be exercised
inside one episode.

The single most important inherited rule remains **§7 + §10's
map-resizing clause**. F11 packs are LARGE by necessity (the army
must traverse from west base to east enemy, combined-arms packs
need land + water + air space) but `wide-justified` is the
honest classification for an F11 pack whose multi-arm geometry
requires that scale — see §76 below. A pack that COULD do the same
test on a 96-cell-wide arena but ships on 160×60 is still `wide`.

---

## §68. Full-sequence chain validation

F11 packs test the COMPLETE chain `plan → cash → build → produce
→ attack`. The win predicate must validate the full chain, not
just the end product. A predicate that only checks "≥4 medium
tanks alive" is satisfied by a model that ignores production and
delivers the starter force across the map; a predicate that only
checks "≥1 enemy fact razed" is satisfied by a starter-force
beeline (the brute / lazy play F1 §10 forbids).

Canonical idiom (lifted from F6 §37 and extended to F11's
combined-arms / multi-arm shape):

```yaml
win_condition:
  all_of:
    - then:
        id: <pack-tier>
        clauses:
          - {has_building: weap}                     # P1: ground production up
          - {has_building: hpad}                     # P2: air production up
          - {unit_type_count_gte: {type: 2tnk, n: 2}}    # P3: ground force
          - {unit_type_count_gte: {type: heli, n: 1}}    # P4: air force
          - {enemy_buildings_destroyed_gte: 2}       # P5: strike home
    - within_ticks: T
    - {building_count_gte: {type: fact, n: 1}}       # base survival floor
```

Predicate guidance:

- **`then:` is mandatory** for the full chain. A flat
  `all_of: [has_building:weap, has_building:hpad,
  unit_type_count_gte:..., enemy_buildings_destroyed_gte:...]`
  accepts ANY order, including "raze first, then claim the chain
  satisfied at the terminal frame" — which a starter-force beeline
  with pre-placed weap/hpad could satisfy for free. The `then:`
  chain forbids "destruction observed before production
  completed".
- **`has_building:weap` and `has_building:hpad` (ever-seen, one-
  shot)** are correct for the production gates — the building
  must EXIST at some point, not necessarily survive to the
  terminal frame. If the test is also resilience-of-the-chain
  (rebuild after attrition), use `building_count_gte` (live
  count) instead.
- **Force-composition clauses use `unit_type_count_gte` per arm**
  separately, so "all tanks no heli" cannot satisfy "≥1 heli".
- **Final strike clause** is `enemy_buildings_destroyed_gte: N` or
  `enemy_key_buildings_destroyed_in_region`. The N must be ≥2 so
  a one-building incidental kill cannot collapse the chain.

A `then:` chain that omits the production phase (e.g. only
`[unit_count, enemy_destroyed]` skipping `has_building:hpad`) is
a defect — the model can satisfy "≥1 heli" with a pre-placed heli
and never have to build, so the production capability isn't tested.

## §69. Offensive enemy graduation

The Family-11 spec calls for OFFENSIVE enemies escalating by tier.
This distinguishes F11 from a build-only pack (F6) where the enemy
is a static `fact` marker. F11 enemies pressure the agent's plan
on a real clock; the model must allocate phase-budget between
defense (survive the pressure) and offense (build the army).

Three canonical enemy-threat patterns. Every F11 pack picks one
and documents it in the `enemy_threat_schedule` audit column.

- **Static-then-scheduled** — light static garrison at t=0 (no
  pressure) + `scheduled_events: spawn_actors` waves at fixed
  ticks (e.g. tick 1500, 3000). Lets the model open with econ /
  tech, then react. Best for "build then strike" packs where the
  EARLY game is build-focused. Used by `f11-vertical-strike-
  ground-air`, `f11-vertical-strike-naval`, `f11-econ-tech-army-
  strike`, `f11-rebuild-after-attrition`.
- **Offensive-throughout** — enemy `bot_type: rusher` or
  `bot_type: hunt` from t=0 + scheduled escalation waves. The
  model must defend AND build simultaneously. Best for
  "defense then counter" idiom. Used by `f11-defense-then-counter`.
- **Reactive (scout-and-pivot)** — enemy openly produces ONE
  arm at t=0 (visible at fixed scout cell) → model picks counter
  → enemy SWITCHES arm mid-episode via `scheduled_events:
  spawn_actors` (a new arm appears). Tests re-perception and
  re-pivot. Used by `f11-pivot-on-scout`.

Tier escalation discipline (F3 §22 / F5 §34 lineage):

| Tier | Wave 1 (early) | Wave 2 (mid) | Wave 3 (late, hard only) |
|---|---|---|---|
| easy | static garrison only OR 2-3 unit nuisance @ ~1500 | none | none |
| medium | 3-4 unit @ ~1500 | 4-5 unit @ ~3000 | none |
| hard | 4-5 unit @ ~1500 | 5-6 unit @ ~3000 | 6-7 unit @ ~4500 |

The waves must FIRE before the deadline (`tick < within_ticks`).
A wave scheduled at tick 3000 against a `within_ticks: 2800`
predicate is dead-code; flag as `inert-wave` in the audit.

## §70. Combined-arms verbs

F11 packs exercise the combined-arms verb set: ground (`weap` →
`1tnk`/`2tnk`/`harv`/`jeep`/`mcv`), air (`hpad` → `heli` /
`afld` → `yak`/`mig`), navy (`syrd` → `dd`/`sub`/`ca` /
`spen` → Soviet equivalents). Engine state for F11 authoring:

- **Aircraft ActorKind** is wired (CLAUDE.md, see
  `combat-heli-flank`). `heli` straight-line move ignores
  ground-impassable cells AND building footprints. Pre-placed
  heli works; BUILT heli via `build('hpad') → build('heli')` is
  the untested path — F11 is the first family to exercise it.
- **Ship locomotor** is wired (CLAUDE.md, see
  `combat-naval-shore-strike`). `dd` destroyer's primary
  armament reaches across the shoreline. Pre-placed `dd` works;
  BUILT `dd` via `build('syrd') → build('dd')` is the untested
  path — F11 (specifically `f11-vertical-strike-naval`) is the
  first to exercise it. **Engine-gap candidate**: if
  `build('syrd')` does not work or `build('dd')` does not work,
  flag immediately.
- **Allied vs Soviet rosters.** F11 packs default Allies:
  ground (`weap` → `2tnk`), air (`hpad` → `heli`), navy
  (`syrd` → `dd`). Soviet equivalents (`afld` → `yak`/`mig`,
  `spen` → `sub`) are valid but the Allied roster is the
  canonical Family-11 substrate (consistent with F6 / F8 / F10).
- **Tech-prereq chain** (extends F6 §42):

  | Buildable | Allied Prereq | Cost | Build sec |
  |---|---|---|---|
  | `hpad` | `tent` (canonical) + `proc` or `fact` (varies) | 1000 | ~20 |
  | `heli` | `hpad` | **2000** | ~24 |
  | `syrd` | `proc` + adjacent water cell | 2000 | ~40 |
  | `dd` | `syrd` | 1000 | ~20 |
  | `afld` | `tent` (Soviet variant) | 1000 | ~20 |
  | `spen` | `proc` + water (Soviet variant) | 2000 | ~40 |

  **Heli cost correction (2026-05-24)**: previous draft said $1200;
  verified against vendored RA YAML during the F11 engine-risk
  verification — actual canonical cost is **$2000**. Per-tier cash
  budgets in §73 must accommodate $2000/heli, not $1200. All F11
  packs requiring built helis: budget at least $2000 per heli unit
  the win predicate demands.

  Costs above are CANONICAL RA-mod values, NOT yet pinned in the
  bench's PRODUCTION_TECH_AUDIT.md (which only enumerates
  ground-arm + econ + power). **Audit-phase action item**:
  cross-reference engine `gamerules.rs` for the exact costs
  before the per-pack YAML authoring wave commits.
- **Map terrain requirements.** Naval packs need a water rect
  (declared via `water_rect: [x, y, w, h]` in `base:`). Air
  packs need no terrain change (`heli` ignores terrain). Combined-
  arms packs need both: an open arena (default) with a water
  band on one edge. See §76.

## §71. Phase budget allocation

Each phase needs a minimum tick budget. F11 packs are the longest
in the suite. Empirical floors (turn counts; 1 turn ≈ 30 ticks
non-interrupt or 1-5 ticks interrupt — F11 packs use non-interrupt
by default unless they expose an `interrupts:` block):

| Phase | Minimum turns | Notes |
|---|---|---|
| Opening econ | 15-25 | proc + harv income ramp; first 1-2 refineries productive |
| Tech (single arm) | 10-15 | weap (40s) OR hpad (20s) — count from fact-build start |
| Tech (combined) | 20-30 | weap + hpad serial; weap + syrd needs proc done first |
| Production | 15-30 | ≥2 ground tanks (16s each, parallel-saturated cuts to 8) + ≥1 air/naval unit |
| Strike | 10-20 | march across 80-120 cells + engagement of 2-3 enemy buildings |

Total per tier (rough envelope):

- **easy** 80-120 turns (loose: pre-placed proc/fact, lean tech)
- **medium** 100-150 turns (full chain, single-arm enemy pressure)
- **hard** 120-180 turns (combined-arms enemy, spawn-point variation)

A pack whose `max_turns` falls below the floor for its phase set
is `tick-budget-too-tight` — flag in the audit. Cross-check
against F6 §41's build-time table (`max_turns × 1s ≥ critical-
path total`).

## §72. Wrong-arm trap (the load-bearing discrimination)

Most F11 packs have a "right arm" and a "wrong arm". The win
predicate's `unit_type_count_gte` per arm and the enemy's
composition together discriminate:

- **Enemy ground only + agent all-air = WIN** (heli kills ground
  without taking fire if no aa)
- **Enemy ground + aa = agent all-air LOSES** (heli shot down)
- **Enemy coastal + agent all-navy = WIN** (dd shells shore)
- **Enemy inland + agent all-navy = LOSS** (dd can't reach
  inland building — no targets in range)
- **Enemy mixed (ground + aa + coastal) = agent must build all
  three** (single-arm policy fails some kill clause)

The pack's `force_composition_required` (CSV column) lists the
minimum vertical force; `wrong_arm_traps` enumerates the 2-3
single-arm policies that LOSE. F11 PACK IS DEFECTIVE if no
single-arm policy loses (the combined-arms test isn't
load-bearing). Audit-check: every F11 pack must have at least 1
documented wrong-arm trap that the intended-capability play
avoids.

Engine state to verify:

- `aa` defenders (Anti-Air) — engine wired? Cross-check the
  Rust gamerules for an `aa` actor that fires on `Aircraft`. If
  not, the "enemy has aa" trap for `f11-vertical-strike-ground-
  air` must be expressed via something else (e.g. enemy heli
  intercept, scheduled enemy-heli spawn, range-1 AA infantry
  variant). **Engine-gap candidate**: surface this before
  authoring `f11-vertical-strike-*` packs.
- Naval shore-strike works (pinned by `combat-naval-shore-
  strike` — `dd` armament reaches inland by ~3 cells from the
  shoreline). The "inland building beyond dd range" trap for
  `f11-vertical-strike-naval` is the building at x=50+ cells
  east of the water rect — well past any naval reach.

## §73. Pre-placed seed buildings

F11 agents start with `fact` + `proc` + 1-2 starter harv (so the
econ chain is already producing on turn 1) + minimal `powr`. The
rest of the tech chain (`tent`, `weap`, `hpad`, `syrd`, `fix`)
must be BUILT by the agent. Rationale:

- The `fact` is required for any subsequent `place_building`
  call (build-radius is anchored on `fact`).
- The `proc` + 1-2 starter harv mean the econ axis is already
  producing on turn 1 — the F11 test is NOT "start econ from
  scratch" (that's F2's `econ-startup-from-scratch`); it IS
  "use ongoing income to fund a tech + force chain".
- A `powr` is required to build `proc` and most production
  buildings — pre-place 1 powr (`tent` needs power too).
- `tent`, `weap`, `hpad`, `syrd`, `fix` ARE the chain the model
  must build. Pre-placing any of them removes that step from the
  test.

Cash budget per tier (the model must afford ≥1 of each required
production building + ≥N units; the chain feasibility check is
F6 §41 critical-path):

- **easy**: starting_cash $5500-6500 + econ income → fund weap
  ($2000) + hpad ($1000) + ≥2 tanks ($1600) + ≥1 heli ($2000) =
  **$6600**. Tight without econ income; deliberate (the econ axis
  must work).
- **medium**: starting_cash $7000-8000 + econ income → fund
  weap + hpad + fix + ≥3 tanks + ≥1 heli = **$9000**. Income
  required.
- **hard**: starting_cash $8000-9500 + econ income → fund weap
  + hpad + syrd + fix + ≥3 tanks + ≥1 heli + ≥1 dd = **$10000+**.
  Income mandatory; cash-starved play LOSES.

Audit-check: every F11 pack's starting_cash + ~30-turn econ income
floor must fit the cheapest winning chain. If not, flag as
`cash-budget-too-tight` (F2 §12 violation).

## §74. Scripted attrition idiom (`f11-rebuild-after-attrition`)

The Wave-9 `scheduled_events.destroy_actors` hook (CLAUDE.md, see
`tests/test_scheduled_events.py`) is the canonical way to wreck
part of the agent base mid-episode. F11's
`f11-rebuild-after-attrition` uses it to test recovery +
persistence: a production building (typically the `weap`) is
destroyed at tick ~800-1000; the agent must rebuild + finish the
force without losing the run.

Required scaffolding (extends F5 §34):

- **`termination.agent_units_killed: false`** — the engine's
  default termination on all-agent-units-eliminated would end
  the run before the rebuild can fire. Disable it. (The
  `enemy_units_killed: false` mirror enables long-running
  enemy-pressure packs but isn't load-bearing here unless the
  agent is expected to kill all enemies which would auto-`done`.)
- **`scheduled_events.destroy_actors` at tick T_attrition**
  (canonical T ≈ 800-1200). The `filter:` clause must match the
  building precisely (`owner: agent` + `region: {x, y, radius:
  3}` centred on the pre-placed building). The radius must be
  tight (`≤4`) so an unrelated agent unit isn't wiped.
- **Rebuild gated by `after_ticks` in the win clause**: the
  `has_building:weap` (or `building_count_gte: weap, n:1`) must
  be evaluated AFTER T_attrition + ~200 ticks (the rebuild
  window). The canonical `then:[not-weap-after-attrition, weap-
  rebuilt]` happened-before latch (F6 §46) enforces this.

Hard tier: TWO destroy events at different ticks (T1=800 wipes
the weap, T2=2000 wipes the proc) so the agent must rebuild TWO
buildings under continuing pressure. Each rebuild clause is a
`then:` checkpoint.

Wrong-strategy LOSS:

- **Pre-attrition-only force** — the model builds the army
  early (≥4 tanks by tick 700) and beelines the enemy; the
  attrition wipes the weap → the force is fixed (no more
  production) but if the kill clause requires ≥N production
  buildings ALIVE at the terminal frame the run LOSES.
- **Ignore rebuild** — the model treats the attrition as
  acceptable loss and pushes with remnants; the kill bar is
  set so the remnant force is insufficient → LOSS.

## §75. Scout-pivot idiom (`f11-pivot-on-scout`)

The pivot idiom tests model RE-PERCEPTION: the enemy's
composition is observable at a fixed time (via scout/jeep at a
known cell, or via fog reveal at a scheduled tick), then the
enemy SWITCHES arm mid-episode forcing the model to re-scout.

Required scaffolding:

- **Hard tier MUST use `scheduled_events.spawn_actors`** to
  inject the SECOND arm at a fixed tick (e.g. tick 2000). The
  first arm is pre-placed and visible from t=0; the model picks
  a counter; then the second arm appears and the original
  counter fails.
- **Win predicate enumerates per-arm counter**: `then:[scout-
  region-hit, unit_type_count_gte:counter1, scout-region-hit-
  again, unit_type_count_gte:counter2, kill-enemy-prod-
  building]`. The two scout clauses force re-perception (the
  model can't just memorise the first arm; the second arm is
  observable only via a fresh scout move).
- **Easy tier**: single arm, no switch. Tests basic RPS choice.
- **Medium tier**: single arm, but the enemy has TWO production
  buildings (one for each arm); the model must observe which
  one is producing (which is `set_primary`) to know what's
  coming. Lighter pivot.
- **Hard tier**: full mid-episode switch via `scheduled_events`.

Wrong-strategy LOSS:

- **Same-arm-as-enemy** — the model produces the SAME arm as
  the enemy (no RPS counter); force composition matches enemy's
  but the kill clause is gated by a tech that the matched arm
  can't satisfy → LOSS.
- **Ignored-scout** — the model produces a default counter
  (e.g. tanks) without ever scouting; the enemy's first arm is
  one the default doesn't counter → LOSS.
- **Locked-in (hard tier)** — the model picks the correct
  counter for arm-1 but doesn't re-scout after the switch;
  arm-2 invalidates the counter → LOSS.

## §76. Map terrain requirements

F11 packs span multiple terrain domains. Each must be declared
in the YAML overrides. Reference patterns:

- **Pure ground** (`f11-econ-tech-army-strike`, `f11-defense-
  then-counter`, `f11-rebuild-after-attrition`): default
  arena, no water needed. 96×40 or 112×40.
- **Ground + air** (`f11-vertical-strike-ground-air`, `f11-
  pivot-on-scout`): default arena. The "air corridor" is
  conceptual — `heli` ignores terrain so any open arena works.
  Document the air-mobility advantage in the briefing (e.g.
  "an open air corridor over the enemy base"). 96-112×40.
- **Ground + naval** (`f11-vertical-strike-naval`): naval-
  arena generator + `water_rect: [x, y, w, h]` on one edge.
  Canonical: `water_rect: [x_water_start, 2, w_water,
  height-4]` (a vertical channel along an edge). The enemy
  must have at least one building INSIDE the dd's range from
  the shoreline (~3 cells inland) AND at least one building
  OUTSIDE the dd's range (deep inland, ≥10 cells past the
  shoreline). 112×40 minimum (96 too tight for both axes).
- **Full combined arms** (`f11-full-combined-arms`): naval-
  arena + water_rect on one side; the rest of the map is
  open. The agent's `fact` must NOT be adjacent to the water
  (the `syrd` must be specifically BUILT next to the water,
  which is itself a placement decision). 128×40 minimum;
  preferred 128×48 to fit three target regions (coastal,
  midfield, deep inland) without crowding.

Water_rect syntax (from CLAUDE.md `oramap.rs:961` and
`combat-naval-shore-strike`):

```yaml
base:
  ...
  water_rect: [15, 2, 2, 36]    # [x, y, w, h] — 2-cell-wide channel
                                # at x=15..16, y=2..37
```

The water cells are SHIP-passable, GROUND-impassable. Building
`syrd` requires an ADJACENT water cell on the cell-x+1 face of
the building footprint (engine convention — verify on a smoke
run before authoring against any new arena, see §53 from F8).

## §77. No-cheat bar specifics

Inherits F1 §10 verbatim. F11-specific stall / lazy / brute /
wrong-arm policies that must LOSE:

- **Stall (only `observe`)** — LOSES on the deadline. No army
  built, no enemy razed, `then:` chain never advances past P1.
  Cash floor — agent's starting_cash burned by no income
  expansion is irrelevant since the win predicate requires
  buildings + units + kills, all of which stall fails.
- **Rush-with-pre-built-force** — LOSES because the F11 packs
  pre-place only `fact + proc + harv(s) + powr` (no army), so
  there's nothing to rush with. A 1-2 unit beeline gets killed
  by enemy defenders OR fails the unit-count clause AND the
  kill clause simultaneously.
- **Over-tech** — model builds weap + hpad + syrd + fix + dome
  + 1 of each unit (rich tech tree, anemic force); the
  deadline bites before the kill bar latches. LOSES on the
  deadline.
- **Under-tech** — model builds only weap + tanks; the hpad /
  syrd / heli / dd clause never satisfies → `then:` stalls
  forever → deadline LOSS.
- **Wrong-arm** (per §72) — the chosen arm cannot kill the
  enemy composition (e.g. all-air vs AA, all-navy vs inland).
  Kill clause never latches → deadline LOSS.
- **Over-econ** (≥3 procs, no army) — relevant for
  `f11-econ-tech-army-strike`. Cash piles up; production never
  starts; deadline LOSS.
- **Under-econ** — relevant for `f11-econ-tech-army-strike`
  and `f11-vertical-strike-*`. Single pre-placed proc + single
  harv is insufficient to fund the combined-arms chain inside
  `within_ticks`; the model must build a SECOND harv (via
  weap) OR a SECOND proc. Without that the chain runs out of
  cash → deadline LOSS.

Validate per F1 §10: scripted-policy probes for each of the 4
wrong-strategy policies must produce LOSS on every seed (1-4) on
every tier. The F11 audit row's `wrong_arm_traps` column
enumerates the specific policies that must lose.

## §78. Engine-gap candidates for the per-pack authoring wave

F11 is the first family to exercise several engine paths that
prior families left untested. The audit-phase action items below
must be resolved BEFORE the per-pack YAML authoring wave commits.
Each is an `engine-gap candidate` that may require an engine fix
(per CLAUDE.md "Fix the engine, don't compromise the pack").

1. **`build('hpad')` + `build('heli')` round-trip.** The
   pre-placed-heli path is wired (`combat-heli-flank`). The
   BUILT-heli path is untested. Smoke-test: pack an agent with
   `fact + tent + proc + cash` and call
   `build('hpad') → place_building → build('heli')`. Verify a
   heli surfaces in `units_summary`. If not, file engine fix.
2. **`build('syrd')` + `build('dd')` round-trip.** Same concern,
   naval side. Pre-placed `dd` works (`combat-naval-shore-
   strike`). Built `dd` needs a `syrd` adjacent to water; the
   build-adjacency rule for `syrd` is the gap.
3. **AA defenders.** If `f11-vertical-strike-ground-air`'s
   wrong-arm trap requires an enemy AA building/unit that
   downs a heli, an `aa` actor must exist and fire on
   Aircraft. Cross-check engine `gamerules.rs`.
4. **`scheduled_events.spawn_actors` for Aircraft / Ship.**
   The Wave-9 spawn-actors test pins ground actors (per
   CLAUDE.md). Verify that a scheduled-event spawning a
   `heli` or `dd` correctly attaches the Aircraft/Ship
   locomotor and the actor lives in `enemies_in_range_*`
   predicates. If not, file engine fix.
5. **`termination.agent_units_killed: false`** behaviour with
   `place_building` after the wipe. The
   `f11-rebuild-after-attrition` pack depends on
   `place_building` working AFTER the agent has lost combat
   units (the buildings remain so build-radius is still valid).
   Verify.

Each engine-gap above is a PRE-FLIGHT item. The audit CSV's
`engine_gap` column (if any) lists the candidate; the per-pack
agent in the next wave must verify or file.

---

## Family-11 audit CSV column contract

`audits/family11_full_game.csv` uses the F1 base columns plus
F11-specific columns:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_threat_schedule |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget |
production_buildings_required | force_composition_required | wrong_arm_traps
```

New columns (F11-specific):

- **`enemy_threat_schedule`** — one short string describing the
  enemy pressure schedule per tier (per §69). E.g.
  `"static garrison + scheduled_events hunt-tank wave @ tick 1500
  (med, hard) + heli wave @ tick 3000 (hard)"`.
- **`production_buildings_required`** — comma-separated list of
  buildings the agent must BUILD (not pre-placed) to satisfy the
  win predicate. E.g. `"weap, hpad, fix"`. Cross-references the
  `then:` chain in `win_condition`.
- **`force_composition_required`** — minimum vertical force per
  the win clause. E.g. `"≥2× 2tnk + ≥1× heli"`. Reads off the
  per-arm `unit_type_count_gte` clauses.
- **`wrong_arm_traps`** — 2-3 documented LOSING policies per
  §72 + §77. E.g. `"all-tank LOSES (enemy has aa heli that
  shreds tanks-only); all-heli LOSES (enemy has aa); navy-only
  LOSES (no coastal target on this map)"`.

Same `map_fit` discipline as F1: `wide-justified` rows DO NOT
require shrinking (the combined-arms geometry IS the test);
`wide` and `large-trivial` rows are backlog items for the YAML-
edit phase. F11 packs are mostly `wide-justified` (the army
march + the enemy depth ARE the test) but `f11-defense-then-
counter` and `f11-rebuild-after-attrition` are `fit` on a
96×40 (the test is local to the agent's base).

Tick budget convention follows F1: `tick_budget = max_turns ×
90 + 3` recorded for cross-family consistency; the actual fail
cutoff is the YAML `after_ticks` value.

Cross-cutting audit checks (run per pack, surface as a row note
if any fire):

1. **No-leak audit** (F1 §9.5 + F2 §18) — does the briefing name
   the winning verb, the build order ("first build hpad, then
   heli"), or the wrong-arm traps explicitly?
2. **Full-chain win predicate** (§68) — does the predicate gate
   on production buildings + force composition + kills, in a
   `then:` chain?
3. **Offensive enemy** (§69) — does the pack actually have an
   enemy that pressures the agent, or just a static `fact`
   marker?
4. **Tech-prereq cross-check** (§70 + F6 §42) — is the agent
   given the minimal prereqs (`fact + proc + powr + harv`) but
   not the production buildings the chain requires?
5. **Phase budget** (§71) — is `max_turns × 30 ≥ critical-path
   tick total` with comfortable slack (≥10% headroom)?
6. **Wrong-arm trap** (§72) — does ≥1 documented single-arm
   policy LOSE on the kill clause?
7. **Engine-gap pre-flight** (§78) — does the pack rely on an
   untested engine path (`build('hpad')`, `build('syrd')`, AA
   defenders, scheduled-air-spawn)?
