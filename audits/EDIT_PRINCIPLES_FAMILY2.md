# Family-2 (Economy) — Edit Principles

**This doc INHERITS every rule in `audits/EDIT_PRINCIPLES.md` (Family-1)
verbatim — §1-10 apply unchanged to Family-2.** What follows are the
ADDITIONAL economy-specific rules. If anything below conflicts with
F1 principles, the F1 principle wins (we keep the briefing/map
conventions identical across families).

The single most important inherited rule is **§7 + §10's
map-resizing clause**: every pack must be classified `fit` / `wide` /
`large-trivial`, and every `wide` / `large-trivial` pack must be
shrunk to a bespoke procedural arena before the audit is "done".
`base_map: rush-hour-arena` on an economy pack is almost always
`large-trivial` (128×40 is far larger than any single
harvest/refinery/expansion decision needs) and MUST be rewritten to a
custom map sized to the actual ore-patch + refinery + spawn geometry.
This rule was IMPLICIT in the prior F2 work and 16 packs slipped
through still on `rush-hour-arena` — do not let this happen again.

---

## §11. Economy chain must be functional end-to-end

Every economy pack must verify the harvester → ore → refinery → cash
chain actually produces cash on a smoke run, not just in theory.

- The agent must own (or be able to build) ≥1 `proc` (Ore Refinery).
- ≥1 `harv` (Harvester) must be present, OR the pack must teach
  building one (give `weap` + budget for `harv` cost $1400).
- ≥1 ore patch must be reachable by the harv within ≤25 cells from
  the refinery (the auto-route picks the closest patch by path
  distance; longer hauls drop income to near zero).
- The bench RNG seed must not place the agent on an island the
  harvester can't reach (verify on a smoke run).

A pack that "tests economy" but where the harv can't reach ore in
the budget is a defective pack, not a hard one.

## §12. Starting cash must constrain multiple winning paths

The cash and force grant must allow at least TWO distinct winning
strategies (so the scenario tests judgment, not the one cookbook
recipe) but neither must be trivially affordable from `starting_cash`
alone.

- Bad: starting_cash 10000 with a $400 win-objective spend ⇒ the
  decision dimension is removed (any random spend wins).
- Bad: starting_cash 100 with a $1500 mandatory spend ⇒ the
  capability cannot fire on turn 1 and the harv-income lag converts
  the cell to a stall-vs-deadline race instead of an economy
  decision.
- Good: starting_cash 1500 with a winning path of either
  `2 × harv ($2800) + 1 × proc ($2000)` OR `1 × proc ($2000) + 1 ×
  pbox ($600) + 4 × e1 ($400)` — the model must pick.

## §13. Wrong-strategy LOSS trap

There must be a plausible-but-wrong play that **LOSES** (not just
"performs sub-optimally"). The classic economy traps:

- **Over-expansion:** building a 2nd `proc` without harvesters
  drains cash, the empty refinery returns zero, and the original
  base runs dry ⇒ LOSS. Anti-pattern is well-documented in
  `econ-buy-vs-build-decision` and `econ-cash-reserve-management`.
- **Hoarding:** stall and never spend ⇒ deadline passes with the
  cash bar unsatisfied (use `not: {cash_gte: MAX+1}` upper bound to
  punish overflow).
- **Capex-only:** spend all cash on `proc`/`weap`/`fact` with no
  units built ⇒ kill-count clause unmet at deadline.

The intended-capability policy must dominate every brute / stall /
hoard / capex-only / wrong-expansion policy on every level + every
hard seed (1-4). The F1 no-cheat bar is binding.

## §14. Spatial reasoning — refinery placement vs ore-patch distance

A refinery's effective income falls steeply with path distance to the
nearest ore patch. The engine's `place_building('proc')` now binds
the new harv (auto-spawned at the new refinery) to the nearest
refinery BY PATH DISTANCE (CLAUDE.md engine note), so the player's
placement choice is load-bearing:

- Pack must surface ≥2 ore patches at distinguishable distances.
- The "obvious" refinery placement and the "good" placement must
  differ in income enough to flip the outcome.
- Pre-existing harvs do NOT re-snap to a new refinery — only the
  auto-spawned new harv binds to it. To test "expand to closer
  patch", the pack must give the agent the path to verify income
  uplift (turns to break-even on the $2000 proc + $1400 harv
  outlay).

## §15. Stall must LOSE (auto-harvest preemption defect)

The most common F2 defect: a stall policy WINS because the
pre-placed harvs auto-harvest enough to satisfy the cash bar without
any decision. Verify by running the stall policy (`Command.observe()`
only) — if it WINS, the pack is broken. Common fixes:

- Set `cash_gte` floor high enough that auto-harvest alone is
  insufficient.
- Set `cash_gte` AND a units-killed / building-built clause so the
  cash must be CONVERTED to action.
- Use `not: {cash_gte: MAX+1}` upper bound so hoarding overflows the
  bank → LOSS.
- Add a `harv_killed` / `proc_destroyed` enemy threat that the
  agent must defend against; stall = harv dies = LOSS.

## §16. Cash-band upper bound is the burn-rate teeth

For "spend at the right rate" tests (burn-rate management), use the
composite `not: {cash_gte: MAX+1}` predicate — the engine supports
`not` over any clause, and there is NO `cash_lte` predicate.
Document the band as `[MIN, MAX]` in the briefing and win readout.

Bursty income (each harv cycle dumps ~600 cr at once) means the band
must be wide enough (~±400 cr) to be hit by even the intended
policy. Validate by smoke-running the intended policy on every seed
and confirming the latch fires.

## §17. Build prerequisites must be present on the agent's side

If the briefing says "build pillboxes" / "build tanks" / "build a
second refinery", the agent's starting set must include the
prerequisite chain:

| Buildable | Prereqs (Allies) | Cost |
|---|---|---|
| `harv` | `weap` | $1400 |
| `2tnk` (medium tank) | `weap` + `fix` | $850 |
| `1tnk` (light tank) | `weap` | $700 |
| `jeep` | `weap` | $600 |
| `e1` (rifle) | `tent` | $100 |
| `e3` (rocket) | `tent` | $300 |
| `pbox` (pillbox) | `tent` (defense queue, gated by infantry building) | $600 |
| `gun` (turret) | `tent` + `fix` | $600 |
| `proc` (refinery) | `fact` | $1400 |
| `powr` (power) | `fact` | $300 |
| `silo` | `proc` | $150 |
| `weap` (war factory) | `proc` | $2000 |
| `fix` (service depot) | `weap` | $1200 |
| `tent` (barracks) | `fact` | $500 |

**Cross-check is part of the audit row** — see
`audits/PRODUCTION_TECH_AUDIT.md` for the explicit per-pack table.

A pack that says "build a tank" but doesn't give `weap` + `fix` is a
defect. So is a pack that says "expand to a 2nd refinery" but
doesn't surface the $2000 + $1400 (proc + auto-harv) in projected
cash.

---

## Family-2 audit CSV column contract

`audits/family2_economy.csv` extends the F1 CSV with two economy
columns:

```
pack | level | capability | map_name | map_size | map_fit | tools |
agent_force | enemy_force | enemy_posture | posture_issue |
briefing_RA | win_condition | lose_condition | max_turns | tick_budget |
starting_cash | economy_setup | wrong_strategy_loss
```

- `starting_cash` — int, the cash the agent starts with.
- `economy_setup` — one short sentence: harvesters/refineries/ore-patch
  layout + income estimate ("2 harvs on 2 near patches @ ~190 cr/turn
  combined").
- `wrong_strategy_loss` — one short sentence: which plausible
  wrong-strategy LOSES, and why.

Same `map_fit` discipline as F1: any `wide` / `large-trivial` row is
a backlog item for the YAML-edit phase to shrink to a bespoke arena.
