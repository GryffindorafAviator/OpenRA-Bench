# PR #31 stale coord-bound predicate sweep

PR #31 (the "tick-fix py39-compat" merge) bulk-resized 151 packs from
`rush-hour-arena` to fixed-recipe generated arenas (128x40 / 128x48 /
136x48) without per-pack reanalysis. Our `pr13-revised` branch's
F4-F9 audit wave had ALREADY shrunk a different subset of packs to
bespoke sizes (64x40 / 80x40 / 96x40 / 112x40 / 120x40), so the
three-way merge collided on 39 files (resolved per-pack per
`EDIT_PRINCIPLES.md`) and auto-merged the rest. Two confirmed
defects in the auto-merged path were already found and patched at
merge time:

* `longhorizon-opening-to-assault.yaml` — predicate `reach_region:
  (115,32)` did not match the new enemy `fact@(80,18)/(80,22)`
  cluster (R4+R5 in the merge commit message; took OURS).
* `strategy-trilemma.yaml` — briefing rewrite vs no-leak briefing
  (R1; took OURS).

This sweep checks the remaining auto-merged set for the same class
of bug.

## Scope

Candidates: every YAML in `openra_bench/scenarios/packs/` that PR #31
touched (`git diff <merge-base>..origin/main`), minus the `_archive/`
tree. Of those 153 candidates, **52 packs declared at least one
coord-bound predicate** (`reach_region`, `units_in_region_gte`,
`units_of_type_in_region_gte`, `all_units_in_region`,
`building_in_region`, `enemy_key_buildings_destroyed_in_region`).
The other 101 PR-#31-touched packs use only signal predicates
(`has_building`, `units_killed_gte`, etc.) and are coord-immune by
construction.

## Methodology

Per pack, per tier (easy / medium / hard):

1. Resolve the effective arena bounds (top-level `base_map:` merged
   with any per-tier `overrides.base_map:`).
2. Resolve the effective actor set (top-level `base.actors:` or the
   tier `overrides.actors:` REPLACEMENT if present), expanding
   `count:` to instance positions.
3. Walk `win_condition` and `fail_condition` collecting every
   coord-bound predicate.
4. Per predicate, compute:
   * **D2** — is (x, y) outside `[cordon, width-cordon) x [cordon,
     height-cordon)`? Engine refuses to place actors outside the
     cordoned interior; `_in_radius` happily evaluates the
     predicate but no agent unit can stand there ⇒ unsatisfiable.
   * **D1** — nearest LOGICAL target actor (enemy landmark for
     `reach_region`/`units_in_region_gte`; enemy building for
     `enemy_key_buildings_destroyed_in_region`; agent buildings
     skipped because `building_in_region` is a "self-build in empty
     terrain" check); Chebyshev distance > `radius + 5`.
   * **D3** — briefing prose names different cell coords than the
     predicate AND the predicate is far from any matching actor.
     Builders' `building_in_region` predicates are excluded
     because the build target is intentionally in empty terrain.
   * **D4** — no matching target actor exists anywhere.
5. Cross-reference the briefing description (`(x, y)` and bare
   `x, y` patterns) to suppress false positives where the briefing
   explicitly names the predicate coord (intended waypoint, not
   stale).

## Results

* Packs scanned: **52**
* Predicate rows evaluated: **303**
* Defective predicate rows: **13** across **5 packs**
* Defect histogram (rows, not packs):

| class | count | meaning |
| --- | --- | --- |
| D1 | 2 | distance > radius + 5 from logical target actor |
| D2 | 11 | predicate coord out of arena bounds |
| D3 | 0 | briefing prose names different cells than predicate |
| D4 | 0 | no matching target actor anywhere |

OK rows: 290 / 303.

### Defective packs

| pack | tiers | class | summary |
| --- | --- | --- | --- |
| `proc-instruction-following-edge-case` | easy/medium/hard | D2 | predicate `units_of_type_in_region_gte: (90, 20, r=6)` but arena is **32x48** — x=90 is 58 cells past the right edge. Pack is unwinnable on every tier. |
| `proc-no-attack-passive-only` | easy/medium/hard | D2 | `reach_region: (110, 20, r=6)` but arena is **104x32** — x=110 is 6 cells past the right edge. Pack is unwinnable on every tier. |
| `coordination-staggered-window` | easy/medium/hard | D2 | `units_in_region_gte: (20, 36, r=8)` but arena is 124x40 cordon=4 ⇒ y must be `< 36`. The y=36 enemy `proc` marker is itself at the boundary. Predicate is on the boundary line, may evaluate but is fragile and at minimum mismatches engine cordon contract. |
| `lh-tech-pivot-attack` | medium/hard | D2 | `units_of_type_in_region_gte: (80, 38, r=10)` but arena is 96x40 cordon=2 ⇒ y must be `< 38`. The enemy `fact@(82, 38)` actor is also at the boundary. |
| `proc-conditional-branch-action` | easy/medium | D1 | `enemy_key_buildings_destroyed_in_region: {types:[tent], x:20, y:36, r:12}` with the nearest enemy `tent` at `(20, 4)` (distance 32). **INTENTIONAL DEAD BRANCH** — the easy/medium tiers ship only the NORTH tent; the SOUTH branch exists to keep the win-schema symmetric so "always SOUTH" loses. Not a defect; flagged here for completeness. |

### Top 5 priority fixes (real defects)

1. **`proc-instruction-following-edge-case`** — easy/medium/hard.
   Arena is 32x48 but predicate wants jeeps at (90, 20). Predicate
   coord is 58 cells past `width=32`. NO play can satisfy this on
   any tier. Either resize the arena (≥96 wide) or move the
   destination predicate inside bounds (e.g. `(28, 20)` if the
   intent is a far-east cell).
2. **`proc-no-attack-passive-only`** — easy/medium/hard. Arena is
   104x32 but `reach_region: (110, 20)`. x=110 is 6 past the right
   edge. Move the predicate to `(98, 20, r=6)` or similar, OR
   widen the arena to 120x32. The enemy `gun` cluster on each
   tier sits in [55, 90] x [18, 22] — the intended "creep past the
   gun line to the far east" win bar still works at x≈98.
3. **`coordination-staggered-window`** — easy/medium/hard. Arena
   is 124x40 cordon=4 (so y in [4, 36)). Predicate `(20, 36, r=8)`
   AND the matching enemy `proc@(20, 36)` actor both sit on the
   y=36 boundary. Move both to y=32 (well inside bounds) — radius
   8 already keeps the predicate forgiving.
4. **`lh-tech-pivot-attack`** — medium/hard. Arena 96x40 cordon=2
   (y in [2, 38)). Predicate `(80, 38, r=10)` AND enemy `fact@(82,
   38)` actor both sit on the y=38 boundary. Move both to y=34 or
   raise the height to 44 cordon=2 so y=38 becomes interior.
5. **`proc-conditional-branch-action`** (verification only, NOT a
   fix). Confirm the dead-branch intent is documented in the
   pack-level comment (it is — lines 270-289 explain "always
   SOUTH" loses by design). No fix required.

### Estimated fix scope

* **4 packs** with true OOB defects need real fixes.
* Each fix is small (move 1-3 coords by 4-10 cells OR bump arena
  dimensions by 4-10 cells). No new actor placement, no new
  predicate, no engine change.
* Estimated fix wave: **~4 packs**, ≤30 lines of YAML diff per
  pack, no engine work, no `.oramap` regeneration needed (these
  pack arenas regenerate from their YAML's `base_map:` block at
  load time, except for the explicitly-cached `data/maps/`
  binaries — those will need a rebuild only if the arena
  dimensions change).
* `proc-conditional-branch-action` needs no fix; the D1 flag is a
  documented dead branch.

## Confidence and false-positive notes

* This sweep does NOT validate `then:` ordered-clause latch
  semantics or `scheduled_events:` mid-game spawns — only static
  predicate-vs-actor and predicate-vs-arena geometry.
* Briefing-prose matching uses both `(x, y)` and bare `x, y`
  patterns. Briefings that discuss "row x=60" or "y=12 lane" in
  prose without literal numeric pairs were treated as "no
  briefing coord" — false-negatives for D3 are possible but the
  D2 / D1 / D4 checks (which are the load-bearing ones for "is
  this predicate actually evaluable") are independent of briefing
  content.
* The `mfb-tech-base-vs-economy-base` (predicate (140, 60, r=10))
  initially appeared OOB against the top-level `base_map:` 128x40
  but every tier overrides to 160x80 cordon=4, so the predicate
  IS in bounds (it sits at y=60 which is interior of the y in
  [4, 76) range). Correctly classified OK after the tier-override
  fix in the sweep script.
* The `lh-build-army-coordinate-multifront-attack` /
  `lh-credit-only-final-phase` predicates at (130, 15) /
  (130, 45) likewise look OOB against 128x40 but every tier
  overrides to 160x60 cordon=4, putting both inside bounds.
  Classified OK.

## Outputs

* `audits/pr31_reach_region_sweep.csv` — every predicate row with
  defect classification and recommendation.
* This narrative.
