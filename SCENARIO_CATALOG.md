# OpenRA-Bench Scenario Catalog (research-grounded, 200 levels)

Goal: a benchmark whose scenarios **generalize on reasoning** — we
empirically observed finetuning on the rush-hour scenario also lifted
**ERQA**. The literature explains why and how to make that systematic.

## Why this transfers (literature)

- **Game/spatial RL → planning transfer is real but axis-specific.**
  lmgame-Bench (Hu et al., 2025, arXiv:2505.15146): RL on simplified
  Sokoban lifted Blocksworld-1D 17.9→32.7%, Blocksworld-2D 9.0→29.5%,
  WebShop 7.0→19.1% — but *not* GSM8K/SQL. Transfer targets are
  spatial / planning / embodied, **not** math.
- **RL not SFT.** "Understanding Transferability of LLM Reasoning"
  (arXiv:2507.00432): RL generalizes (minimal representation drift);
  trajectory SFT catastrophically forgets. Use these scenarios with
  verifiable-reward RL, not trajectory SFT.
- **ERQA is reasoning-sensitive.** ERQA (Gemini Robotics, DeepMind,
  arXiv:2503.20020): 400 MCQ over spatial / trajectory / action /
  state-estimation / multi-view / task reasoning; CoT alone moves it
  +4–6.5pts → a reasoning-shaping finetune (rush-hour) plausibly moves
  its spatial/trajectory/task subscores. Our suite is built to drive
  exactly those axes.
- **Anti-memorization is mandatory.** SMAC→SMACv2 (arXiv:2212.07489):
  a timestep/ID open-loop policy beat SMAC → memorization, not
  reasoning. Procgen (arXiv:1912.01588), Crafter (arXiv:2109.06780),
  ARC-AGI (arXiv:1911.01547): procedurally varied, held-out instances;
  report the **generalization gap**. Distractor/decoy decorrelation of
  "obvious cue" vs "correct action" (shortcut-learning, arXiv:2412.05152).

## Capability taxonomy → measurement

Consolidated from the general-reasoning, embodied/spatial, and
games-as-eval surveys. Metrics borrow canonical instruments: **SPL**
(Anderson et al., 2018, arXiv:1807.06757) for "reach/find under fog";
**coverage-under-budget / time-to-find / region-commitment regret**
(Active Neural SLAM, Chaplot et al., ICLR 2020, arXiv:2004.05155;
frontier exploration, Yamauchi 1997); **Goal-Condition Success +
path-length weighting** (ALFRED, arXiv:1912.01734); **predicate
satisfaction** (PlanBench, arXiv:2206.10498; BEHAVIOR-1K,
arXiv:2403.09227); **win/attrition under a difficulty ladder**
(TextStarCraft II, arXiv:2312.11865; SmartPlay, arXiv:2310.01557).

| Cap | Dimension | Primary literature |
|---|---|---|
| PERC | spatial/state-estimation under partial obs | ERQA; SpatialVLM (2401.12168); VSR (TACL'23) |
| FRONT | which-unexplored-region-to-commit | Active Neural SLAM; frontier exploration; exploration-aware EQA (2503.11117) |
| PLAN | sequential planning under constraints | PlanBench; ALFWorld (2010.03768) |
| TECH | precondition / tech-graph ordering | PlanBench (Mystery-Blocksworld); BEHAVIOR-1K BDDL |
| ECON | resource allocation / multi-objective | PlanBench cost-optimal; SmartPlay |
| COORD | multi-unit / multi-agent coordination | Watch-And-Help (2010.09890); SMAC |
| RISK | risk assessment / replanning under partial info | PlanBench replanning; StarCraft II Arena (ICLR'25) |
| TEMPO | timing / tempo windows | TextStarCraft II; SmartPlay |

## The 200-level catalog (12 categories, ~67 three-level packs)

From the engine-grounded end-to-end RA dissection (28 decision points,
phases opening→scouting→economy→tech→army→engagement→defense→assault→
endgame). Every win-condition uses ONLY predicates in
`win_conditions.py`; all on `rush-hour-arena` (only Rust-loadable map
until S10). Difficulty ladder = the *decision* gets harder (less info,
more decoys, tighter clock, committed-not-loose budget, attrition
cap) — never just bigger numbers (Procgen/SMACv2 rule).

| # | Category | Cap | Levels | Win pattern (real grammar) | Status |
|---|---|---|---|---|---|
| C1 | Frontier Scouting | FRONT | 18 | `all_of[explored_pct_gte:P,(units_lost_lte:0),within_ticks:T]` | now |
| C2 | Threat Enumeration | PERC | 18 | `all_of[enemies_discovered_gte:N / buildings_discovered_gte:K, units_lost_lte:0, within_ticks:T]` | now |
| C3 | Tech Critical Path | TECH | 18 | `all_of[has_building:X, building_total_gte:N, power_surplus_gte:0, within_ticks:T]` | now |
| C4 | Power-Budget Online | PLAN | 15 | `all_of[power_surplus_gte:0, building_total_gte:N, within_ticks:T]` | now |
| C5 | Budget Allocation (spend) | ECON | 18 | `all_of[building_count_gte{A,n}, building_count_gte{B,n}, within_ticks:T]` | now |
| C6 | Time-Boxed Capital Deploy | ECON | 18 | `all_of[own_units_gte:K, building_total_gte:M,(units_lost_lte:0),within_ticks:T]` | now |
| C7 | Defensive-Direction Commit | PERC | 18 | `all_of[building_in_region{type:pbox,...,count:k}, within_ticks:T]` | now |
| C8 | Base-Placement & Staging | PERC | 15 | `all_of[building_in_region{type:fact,...},(reach_region),within_ticks:T]` | now |
| C9 | Commit-vs-Retreat | RISK | 18 | `all_of[units_killed_gte:N, units_lost_lte:L, within_ticks:T]` | now |
| C10 | Force Coordination | COORD | 18 | `all_of[all_units_in_region{...}, units_lost_lte:0, within_ticks:T]` | now |
| C11 | Tempo / Timing Window | TEMPO | 15 | `all_of[after_ticks:t0, units_killed_gte:N, within_ticks:T1]` | now |
| C12 | Error Recovery / Replan | RISK | 15 | `all_of[has_building:X, building_total_gte:N, after_ticks:t0, within_ticks:T]` | now |

≈ **201 levels / ~67 packs**, all runnable on the current engine.
Economy categories are scoped as **spend/conservation** decisions
(`starting_cash`), the discriminating choice today; an
**Economy-Harvest** 13th category is **now RUNNABLE** — engine task
**#14** (S0 ore-seeding + idempotent harvest, S1 capped resource pool
from refineries/silos + `economy_value`) is done. Shipped packs:
`economy-harvest-timebox` (earn max economic value in a time budget,
keep harvesters productive) and `economy-harvest-investment` (split a
budget between widening collection and silo storage so burst yield is
not lost — the storage decision). Win = `economy_value_gte`
(=cash+stored resources) / `harvesters_gte` / `has_building:silo`
within `within_ticks`; integ-tested end-to-end (test_economy_harvest).

## Generation & integ-test strategy ("each run + validated")

- Packs are **generated** by `tools/gen_catalog.py` (procedurally
  parameterized per the anti-memorization rule), written flat into
  `openra_bench/scenarios/packs/` with a `cat-` prefix.
- **Validation**: `python -m openra_bench.scenarios.validate` compiles
  all 3 levels of every pack against the engine `ScenarioDefinition`
  + win-grammar (schema gate).
- **Engine run**: the existing parametrized
  `tests/test_robustness.py::test_bundled_pack_runs_on_engine`
  auto-covers every `packs/*.yaml` — each catalog pack's easy level
  must actually execute on the Rust engine (terrain resolves, actors
  load, scores), not merely validate.
- **Catalog coverage test**: `tests/test_catalog.py` asserts all 12
  categories are present, ≈200 levels exist, every pack carries a
  capability tag, and difficulty is monotone (tighter clock / less
  info per level) — the literature's controlled-ladder requirement.
- **Transfer panel (follow-up)**: pre-register ERQA + Blocksworld +
  WebShop external eval to quantify generalization gap (lmgame-Bench
  protocol); report per-axis deltas, not aggregate.

## Cross-cutting rules (enforced by the generator)

1. No enemy-destruction predicate exists → offensive success =
   `units_killed_gte` + positional `reach_region`/`building_in_region`.
2. Difficulty = decision hardness (info↓, decoys↑, clock↓, budget
   committed, attrition cap), never raw scaling.
3. Per-pack `real_world_meaning` + `robotics_analogue` are mandatory
   and tie to the mapped capability dimension + its literature.
