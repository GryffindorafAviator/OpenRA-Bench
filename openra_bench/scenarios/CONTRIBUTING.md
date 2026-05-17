# Contributing a Scenario Pack

A **pack** is one YAML file in `packs/` describing **one decision problem
at three difficulty levels** (`easy` / `medium` / `hard`). You write only
YAML — no Python.

## Rules

1. **It must mean something.** `meta.real_world_meaning` and
   `meta.robotics_analogue` are required and reviewed. Example: *path
   planning algorithms are solved; deciding **which** unexplored region
   to commit to under partial information is not — that is the search-
   and-rescue problem.*
2. **One capability focus.** `meta.capability` ∈ `perception` |
   `reasoning` | `action` — the link of the Perception→Reasoning→Action
   chain this scenario primarily stresses.
3. **Three real levels.** Difficulty must come from the *decision*
   getting harder (less information, more decoys, tighter deadline,
   stronger defenses) — not just bigger numbers.
4. **Custom win condition, declaratively.** Use the grammar below; the
   "bot" / objective is whatever the win condition says.
5. **Map.** `base_map: rush-hour-arena` works today. Other maps are
   schema-valid but skipped until the Rust generic-map loader (Phase 3).

## File shape

```yaml
meta:
  id: partial-info-rescue          # lowercase-kebab, unique
  title: "Rescue Under Partial Information"
  capability: reasoning
  real_world_meaning: >
    Pathfinding is solved; choosing which unexplored area to search
    first with limited fuel/time is the actual rescue problem.
  robotics_analogue: "UAV search-and-rescue frontier selection"
  author: "your-name"

base_map: rush-hour-arena
base:                              # shared ScenarioDefinition fields
  agent: {faction: allies}
  enemy: {faction: soviet}
  tools: [move_units, attack_unit, stop_units]
  planning: true
  actors:
    - {type: jeep, owner: agent, position: [5, 5], count: 3}
    - {type: e1,   owner: enemy, position: [60, 20], stance: 2}
  termination: {max_ticks: 8000}

levels:
  easy:
    description: "Target in the nearest unexplored quadrant."
    overrides: {}                  # deep-merge patch onto base
    win_condition: {all_of: [{buildings_discovered_gte: 1}, {within_ticks: 6000}]}
    fail_condition: {units_lost_lte: -1}      # optional
    max_turns: 30
  medium:
    description: "Two plausible regions; one is a decoy."
    overrides:
      actors:                       # full list replaces base.actors
        - {type: jeep, owner: agent, position: [5, 5], count: 2}
        - {type: e1,   owner: enemy, position: [90, 30], stance: 2}
    win_condition: {all_of: [{buildings_discovered_gte: 1}, {within_ticks: 5000}]}
    max_turns: 35
  hard:
    description: "Three regions, decoys, tight deadline, attrition."
    overrides: { ... }
    win_condition: { ... }
    max_turns: 40
```

## Win-condition grammar

Composites: `all_of: [..]`, `any_of: [..]`, `not: {..}`. Leaves (a node
with multiple leaves is an implicit AND):

| Leaf | Meaning |
|---|---|
| `explored_pct_gte: <float>` | map % revealed ≥ value |
| `enemies_discovered_gte: <int>` | distinct enemy units seen ≥ value |
| `buildings_discovered_gte: <int>` | distinct enemy buildings seen ≥ value |
| `units_killed_gte: <int>` | agent kills ≥ value |
| `units_lost_lte: <int>` | agent losses ≤ value (constraint) |
| `within_ticks: <int>` | reached by game tick ≤ value (deadline) |
| `after_ticks: <int>` | only after game tick ≥ value |
| `reach_region: {x,y,radius}` | ≥1 agent unit within radius of (x,y) |
| `all_units_in_region: {x,y,radius}` | every agent unit within radius |

`win_condition` is checked every turn; first turn it holds → **win**.
`fail_condition` likewise → **loss**. Neither by `max_turns` → **draw**.

## Validate before opening a PR

```bash
python -m openra_bench.scenarios.validate packs/your-pack.yaml
```
