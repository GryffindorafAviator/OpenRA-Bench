# Construction, Unit Production & Tech-Tree Audit

A separate audit dimension that crosses all families. Every pack that
declares a build/produce capability — or implicitly assumes the
player can build something during the run — must satisfy the tech
chain. This catches a class of defects that briefing/map audits miss:
the player is told to build a tank but doesn't have a war factory,
or is told to expand to a refinery without the prerequisite
construction yard.

---

## The bar

Every buildable advertised by the briefing or required by the
intended-capability play must:

1. **Have its production building present** on the agent's side at
   start, OR be itself buildable in the same run from prereqs the
   agent has.
2. **Have its tech-gate building present** (e.g. `2tnk` needs both
   `weap` and `fix`; `pbox` needs `tent`).
3. **Be affordable** within the projected cash trajectory
   (`starting_cash` + harv income over the build duration).
4. **Be buildable within the tick budget** (build time + tech-gate
   resolution must fit the reachable tick budget — for non-interrupt
   packs that is `93 + 90·(max_turns − 1)` ticks, i.e. ~90 ticks per
   decision turn; interrupt-mode packs vary per turn — read
   `info["ticks_advanced"]`).
5. **Match the faction** — Allies vs Soviet rosters diverge. Most
   bench packs are Allies; the pack-level `agent: {faction: ...}`
   must match the buildables.

## Allies tech tree (anchor — most bench packs)

```
fact (CY)  → tent (barracks)  → e1 (rifle), e3 (rocket), e4 (flamer*)
           → powr (power)
           → proc (refinery)  → silo, weap (war factory)
                                     → 1tnk, 2tnk, jeep, harv, mcv*
                                     → fix (service depot) → 2tnk*, mtnk
           → tent + fix       → gun (turret), pbox (pillbox**), hbox*

* Allies / Soviet differs. Bench's `2tnk` (Allies medium) reads via
  fix as the "tech tank" gate.
** pbox: requires tent (infantry building); see CLAUDE.md note that
  the engine assigns M60mg to garrison-only defenses lacking
  Armament, so a built pbox actually fires anti-infantry.
```

## Costs and build times (canonical, RA-mod defaults)

| Slug | Name | Cost | Build (sec*) | Prereqs |
|---|---|---|---|---|
| `e1` | Rifle infantry | 100 | 5 | `tent` |
| `e3` | Rocket infantry | 300 | 8 | `tent` |
| `e6` | Engineer | 500 | 10 | `tent` |
| `e7` (tanya) | Commando | 600 | 30 | `tent` + `fix` |
| `thf` | Thief | 500 | 15 | `tent` + `fix` |
| `jeep` | Ranger | 600 | 14 | `weap` |
| `1tnk` | Light tank | 700 | 16 | `weap` |
| `2tnk` | Medium tank | 800 | 18 | `weap` + `fix` |
| `mtnk` | Mammoth* | 1700 | 30 | `weap` + `fix` + tech |
| `harv` | Harvester | 1400 | 25 | `weap` |
| `mcv` | Construction MCV | 2500 | 40 | `weap` + `fix` |
| `fact` | Construction Yard | (from MCV deploy) | — | — |
| `tent` | Barracks (Allies) | 400 | 10 | `fact` |
| `powr` | Power Plant | 300 | 8 | `fact` |
| `proc` | Ore Refinery | 1400 | 28 | `fact` |
| `weap` | War Factory | 2000 | 30 | `proc` |
| `fix` | Service Depot | 1200 | 22 | `weap` |
| `silo` | Ore Silo | 150 | 5 | `proc` |
| `gun` | AA/AT Turret | 600 | 14 | `tent` + `fix` |
| `pbox` | Pillbox | 400 | 12 | `tent` |
| `hbox` | Camo Pillbox* | 600 | 14 | `tent` + `fix` |

\*"Build seconds" is the canonical RA-mod value. The bench engine
advances roughly 90 ticks per decision turn (formula
`93 + 90·(max_turns − 1)` total reachable ticks; see CLAUDE.md), so
at the engine's nominal 30 ticks per second the build-second column
corresponds to roughly `secs / 3` decision turns — e.g. a `weap` (30s)
becomes ~10 turns of agent decisions. Parallel production
(2× weap on the same queue) roughly halves the per-unit build time —
see CLAUDE.md note "Multiple production buildings of the same
category produce IN PARALLEL".

## Audit CSV per pack (one row per pack, not per level)

`audits/production_tech_audit.csv`:

```
pack | family | buildables_required | tech_gates_present | tech_gates_missing |
afford_at_start | afford_by_deadline | build_in_budget | faction | issues
```

- `buildables_required` — comma-list of slugs the agent must build
  OR rely on (e.g. `2tnk, harv, proc, pbox`).
- `tech_gates_present` — list of agent-owned buildings at t=0 (or
  scripted_events spawn) that gate the buildables.
- `tech_gates_missing` — any required prereq NOT in `_present`.
  Non-empty ⇒ DEFECT.
- `afford_at_start` — bool, can the cheapest winning build complete
  from `starting_cash` alone?
- `afford_by_deadline` — bool, can `starting_cash + projected_income
  × max_turns` fund the intended build?
- `build_in_budget` — bool, does the sum of build times (serialized
  on a single production building) fit in `max_turns`?
- `faction` — `allies` / `soviet` / `mixed`.
- `issues` — short freeform notes; populate when any bool above is
  False or when a faction mismatch is detected.

## Common defect patterns

1. **Missing `fix` for `2tnk`-build packs.** The Allied medium tank
   requires both `weap` AND `fix`; many F2 packs grant `weap` only,
   silently blocking the tech-tank build path.
2. **Missing `tent` for `pbox` / `gun` packs.** The defense queue is
   gated by the infantry building (tent), not by the construction
   yard alone.
3. **`weap` without `proc`.** A war factory needs a refinery as
   prereq — even if you don't intend to harvest, the prereq still
   gates.
4. **Faction mismatch.** A Soviet `agent: {faction: ussr}` cannot
   build `2tnk` (medium tank, Allies-only) or `mtnk` (Mammoth). The
   bench default is Allies; only flip if intentional.
5. **Build-time exceeds deadline.** A 30-turn `max_turns` cannot
   serialize `weap` (30s) + `fix` (22s) + `2tnk` (18s) — needs ≥70
   turns or pre-placed prereqs.
6. **`harv` not surfaced as buildable** when the win requires
   replacing a dead harv (the `econ-replace-dead-harvester` idiom
   needs `weap` exposed so the agent can produce a new harv).
7. **`fact`-cost edge.** `fact` is cost 0 in the engine (can't be
   built via `StartProduction`); the only way to acquire a new fact
   is `deploy` an MCV. Packs claiming "build a second base" must
   either pre-place a 2nd `fact` or grant an `mcv` + expose the
   `deploy` tool.

## When to flag vs fix

- **`tech_gates_missing` non-empty** → DEFECT — either add the
  missing building to the agent's start, OR rewrite the briefing
  to not advertise that build.
- **`afford_at_start = False` AND `afford_by_deadline = False`**
  → DEFECT — boost `starting_cash` or shorten the build chain.
- **`build_in_budget = False`** → DEFECT — pre-place prereqs OR
  extend `max_turns`.
- **Faction mismatch** → fix the faction or the unit list to match.

Use `audits/production_tech_audit_build.py` to generate the CSV
programmatically (read each YAML, parse `agent: {actors: [...]}`,
extract building list, cross-reference against buildables advertised
in `briefing:` and required by `win_condition.kills_required` etc.).

---

## Binding for future agents

When ANY agent edits a scenario pack:

1. **Read `audits/EDIT_PRINCIPLES.md`** (family-1 base rules)
2. **Read `audits/EDIT_PRINCIPLES_FAMILY2.md`** if the pack is in
   family-2 (economy) or shares an economy axis
3. **Read this doc** if the pack's intended-capability play
   requires any `build` order or `deploy` order — i.e. the
   construction/production/tech tree is load-bearing
4. **Cross-reference `audits/production_tech_audit.csv`** for the
   pack's row; if `issues` is non-empty, FIX the issues as part of
   the edit
5. **Update the CSV row** if the edit changes any tech-tree-relevant
   field (new buildings added, faction flipped, starting_cash
   changed)
