# Engine follow-ups discovered during Wave 12+13 per-scenario map redesign

Issues that fall outside the per-scenario-map work but were surfaced by
the agents while doing it. Each item is a discrepancy between what the
bench can EXPRESS (the scenario spec / capability description) and what
the engine actually DOES. Most have working pack-side workarounds; a few
genuinely block a capability from being tested cleanly.

Filed 2026-05-23 from Wave 12 + Wave 13 agent reports.

---

## 1. Tick-rate discrepancy (medium severity)

**Finding** (def-position-revealed-direction agent, Wave 13)

CLAUDE.md says "engine advances ~90 ticks per decision turn".
Empirically measured under interrupt mode: **~64 ticks/turn**, not 90.

**Impact**: per-pack `within_ticks` / `after_ticks` deadlines tuned via
the CLAUDE.md formula `tick ≤ 93 + 90·(max_turns − 1)` may either bite
LATER than intended (silent draw-degeneracy) or NOT AT ALL inside
`max_turns`. The agent had to tighten `within_ticks: 5400 → 3600` to
make the deadline actually bite.

**Action options**
- (a) Re-measure tick-rate under all bench code paths (interrupt vs
  non-interrupt; varying `parallel_cells`) and update CLAUDE.md with
  the empirical number.
- (b) Audit every pack's `within_ticks` against the empirical rate;
  flag packs whose deadline is now unreachable.
- (c) Stabilize the engine's tick-per-turn to a documented constant.

---

## 2. Ore density clamps at 12 cells (medium severity)

**Finding** (econ-mine-and-grow agent, Wave 12)

`ore_patches: {x, y, amount, radius}` density formula is
`clamp(amount / cells, 1, 12)`. The cap of 12 ore/cell means a small
patch with a big `amount` is wasted — the original
`amount: 2000, radius: 2` (~13 cells) yielded only ~156 total ore =
~3900 cash ceiling, insufficient for the medium/hard cash bars.

**Workaround**: bump `amount` to 10000-20000 and use `radius: 4` so the
denser disc actually fills. Documented in CLAUDE.md as a footgun, not
yet a permanent fix.

**Action options**
- (a) Document the clamp prominently in `openra_bench/scenarios/schema.py`
  docstring for `ore_patches:`.
- (b) Raise the cap (e.g. 24/cell) so smaller patches behave intuitively.
- (c) Have `ore_patches:` validation warn when `amount > cells * 12`.

---

## 3. Nuke AoE kills not credited to `kills_per_player` (medium severity)

**Finding** (spec-nuke-strike agent, Wave 12)

`openra-sim/src/world.rs::detonate_nuke` (around line 2133) removes
actors via `self.actors.remove(id)` without bumping
`self.kills_per_player[owner]`. Result: `units_killed` stays 0 for
nuke AoE kills.

**Impact**: any pack that wants `units_killed_gte: N` for a nuke
scenario is unwinnable. The original `spec-nuke-strike` pack used
`units_killed_gte: 5` and was unwinnable on every seed.

**Workaround**: pivoted the pack to destroy BUILDINGS (`silo` clusters)
and use `enemy_buildings_destroyed_gte` instead. Works but limits the
nuke-targeting test to building targets.

**Fix**: mirror the data-driven attack path's credit-on-removal line
(`world.rs:4237`) inside `detonate_nuke`. Small Rust PR.

---

## 4. Explicit `harvest(unit_id, x, y)` orders silently ignored (high severity)

**Finding** (econ-multi-patch-allocation agent, Wave 12)

The bench's `harvest` command issues an order, but the engine's
auto-route system (`World::auto_route_idle_harvesters`) re-overrides
any explicit assignment the moment the harvester's owner has a `proc`.
Result: stall, all-to-FAR, all-to-MID, uniform-split, and intended
2N+1M policies ALL yield the same `ev=14250` on
`econ-multi-patch-allocation medium`.

**Impact**: ANY pack measuring harvester-allocation decision quality
(econ-multi-patch-allocation, econ-far-patch-vs-near-patch,
econ-protect-harvester-route, econ-harvester-pathing-optimization)
cannot discriminate stall from intended; the no-cheat bar can't be
enforced on those families.

**Fix**: respect an explicit `harvest` order — disable
`auto_route_idle_harvesters` for any harvester that has a
user-issued `Harvest` activity within the last N ticks. Small Rust
PR but load-bearing for ~4 packs and the whole "harvester allocation"
capability.

---

## 5. ReturnFire-stance auto-fire breaks kite-and-pull discrimination (high severity)

**Finding** (combat-kite-and-pull agent, Wave 13)

The canonical kiting capability is: ranged unit retreat-fires while a
slower-but-stronger melee unit chases. The intended policy WINS by
keeping range; a stall policy (just stand still on ReturnFire) should
LOSE because the kiter eats hits without dealing enough damage back.

**Empirical reality**: 1 medium tank cannot kill 1 heavy via the kite
cycle within `within_ticks` regardless of HP tuning. **The cell where
the kite WINS is also the cell where stall WINS** — the heavy walks
into the passive ReturnFire kiter, which auto-fires and kills it
without any kite needed.

**Impact**: the canonical 1v1 kite-and-pull test cannot be expressed.
The combat-kite-and-pull pack worked around it by using a 3-raider
composition (focus-fire micro instead of true kite-and-pull). This
LOSES the bench's ability to test kite-and-pull as a distinct
capability.

**Fix options**
- (a) ReturnFire balance: reduce auto-fire DPS so a passive kiter
  actually loses to a heavy in 1v1.
- (b) New stance: "Pure ReturnFire" that fires ONLY after taking hits
  (already partially specified in CLAUDE.md's stance section — verify
  it's actually working as specified).
- (c) Engine `attack_unit_kite` order that explicitly retreats-fires
  with kite cadence (most invasive, but cleanest semantic).

This is the highest-leverage of the 7 findings — it removes a whole
capability from the bench.

---

## 6. Arena generator's default 4-corner mpspawns rotate per-seed (medium severity)

**Finding** (perception-target-vs-fog agent, Wave 13)

`openra_bench/mapgen.py::_arena` emits 4 default corner mpspawns.
The engine picks one mpspawn per seed and **OFFSETS pre-placed actor
coords relative to it** — authored `position: [6, 5]` actually appears
at y=29-33 on some seeds.

**Impact**: per-scenario maps using the `arena` generator with
default spawns get unexpectedly seed-rotated, breaking position
invariants. Several Wave 12 packs may be affected silently.

**Workaround**: declare exactly one mpspawn per generator spec:
`spawns: [[6, 5]]`. Documented inline in the affected pack.

**Action options**
- (a) Default `arena` generator to a SINGLE centred mpspawn so
  position invariants hold by default.
- (b) Add a runtime warning when a generator emits multiple mpspawns
  AND the scenario has `position:` actors that could be offset.
- (c) Add `mapgen.py` docstring warning.

---

## 7. Pre-placing `proc` inside an `ore_patches:` disk silently empties patch (low severity, easy workaround)

**Finding** (mcv-deploy-near-resource agent, Wave 13)

If you pre-place a `proc` actor at a cell that lies INSIDE an
`ore_patches:` disc, the patch silently renders zero ore. The harv
"harvests" at 0 cr/turn.

**Workaround**: place `proc` just OUTSIDE the patch radius. Trivial
once known.

**Fix**: validation-time error or warning when a pre-placed proc/silo
overlaps an `ore_patches:` disc.

---

## Summary

| # | Issue | Severity | Workaround? | Cap effort to fix |
|---|---|---|---|---|
| 1 | Tick-rate ~64 not ~90 | M | Tighten deadlines | Re-measure + doc update |
| 2 | Ore density clamps at 12 | M | Bump amount | Raise cap or doc |
| 3 | Nuke kills not credited | M | Use building-destroy predicate | Small Rust PR |
| 4 | Explicit harvest ignored | **H** | None — kills capability | Small Rust PR |
| 5 | ReturnFire breaks kite | **H** | Use focus-fire comp instead | Engine balance change |
| 6 | Arena default mpspawns rotate | M | Declare single spawn | Default change |
| 7 | Proc-in-patch empties patch | L | Place outside | Validation warning |

**The two HIGH-severity findings (#4 + #5) each remove an entire
capability axis from the bench: harvester-allocation and kite-and-pull
micro.** Worth a focused engine PR pair before the next data-collection
campaign.

The medium findings either have clean workarounds (already applied) or
are doc/spec polish.

---

## Suggested next actions

1. **Top priority**: schedule an engine PR to address #4 (harvester
   allocation) — unblocks an entire econ family.
2. **Second priority**: address #5 (ReturnFire balance) — restores
   kite-and-pull as a real capability.
3. **Bench follow-ups**: re-measure tick rate (#1), add validation
   warnings for #2/#7, update mapgen default (#6) — bundleable into
   one bench-side PR.
4. **Nuke kill credit** (#3) is independent and small; ship whenever.
