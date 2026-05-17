# OpenRA-Bench → Unified Eval Stack Plan

## Goal

Turn **OpenRA-Bench** into an efficient, customizable harness to evaluate model
performance on **spatial, multi-modal, and complex multi-target multi-step
reasoning / planning**, by reusing the mature eval stack from
**OpenRA-RL-Training** on top of the **Rust** environment (OpenRA-Rust).

Two evaluation modes:

1. **Fixed-scenario eval** — N episodes of one model on a controlled scenario,
   scored with verifiable rubrics + composite score + 95% CIs.
2. **Pairwise adversarial 1v1** — two models in one 1v1 map, win-rate / Elo.

Scenarios must be *controlled* and isolate specific capabilities (e.g. the
observed failure: model cannot connect an unexplored region to a target area to
explore — a vision/perception + spatial-planning deficiency).

## Core framing: the Perception → Reasoning → Action chain

Every scenario evaluates one chain: **read the (visual + symbolic) state →
form a multi-step plan → emit valid commands that execute it**. The eval must
*attribute* failure to a specific link, not just report a score:

- **Perception** — did the model correctly read the minimap/state? (e.g.
  locate the unexplored region and the target). Probe via state-readback /
  forced-choice checks derived from ground-truth obs.
- **Reasoning** — given correct perception, did it form a valid plan? (e.g.
  connect unexplored→target, sequence multi-target order). Probe via plan
  quality vs optimal (path length, target ordering, sub-goal coverage).
- **Action** — did it emit syntactically/semantically valid commands that
  realize the plan? Probe via action-validity rate and plan↔execution drift.

Per-scenario rubrics carry one diagnostic per link so a low score points at the
broken link. This is the primary product differentiator vs a raw win-rate bench.

## Decisions (locked)

- **Repo**: refactor OpenRA-Bench in place; it imports Training's eval stack.
  OpenRA-RL-Training stays source of truth for the engine code.
- **Backend**: Rust only (`openra_train` PyO3). C# (openra-rl) is slow/fragile
  and is dropped from the eval path. Rust must be made faithful to the C#
  reference where scenarios require it.
- **Multi-modal**: reuse Training's `minimap_renderer.render_minimap()` PNG,
  injected as `image_url` in the agent prompt.

## Source components reused from OpenRA-RL-Training

| Component | Path | Role in Bench |
|---|---|---|
| Episode engine | `openra_rl_training/training/agent_rollout.py` (`play_episodes_async`) | Drives the real model loop (currently Bench's agent fn is a no-op) |
| Reward dims | `training/reward_funcs.py` | Per-scenario weighted scoring |
| Rust pool | `training/rust_env_pool.py` | The only backend |
| Minimap | `training/minimap_renderer.py` | Multi-modal observation |
| Scenarios/rubrics | `scenarios/*.yaml`, `curricula/*.yaml`, verifiable metrics | Controlled tasks + pass/fail |
| CI comparison | `scripts/build_eval_comparison.py` | Stat-sound model comparison |

## Rust faithfulness gap (drives sequencing)

- Commands: 3/22 (Move, Attack, Observe). Missing: Build/Train/Harvest/Deploy/
  Sell/Repair/Stance/Transport/Power/RallyPoint/Guard/Patrol…
- Observations: ~30% of C# proto. Missing: economy, production, military stats,
  spatial tensor, kill_events, result/reward fields.
- Scenarios: 2 hand-built (rush-hour, scout-maginot). No generic `.oramap` load.
- Engine: movement (A*), combat, projectiles, fog, static defenses = done.
  Economy, production/tech, transport, multi-armament = not done.

→ Movement/combat/fog scenarios + combat-only 1v1 work **today**. Economy /
production / tech scenarios require Rust engine work first.

## CRITICAL FINDING (verified end-to-end, local)

`play_episodes_async` (agent_rollout.py:4815) is **hardwired to the C# gRPC
server** via `openra_env.mcp_ws_client.OpenRAMCPClient` and is entangled with
TRL (tokenizer, prompt_ids/completion_ids, worker pool, partial cache). It is
**not reusable as-is on Rust**. `rust_env_pool` is used only by the lighter
`rollout.py` path.

Also: `minimap_renderer.render_minimap()` expects `state["minimap"]` (ASCII),
`units_summary`, `enemy_summary` — **none of which the Rust env emits.**

Verified live Rust obs schema (`openra_train` rush-hour, local wheel,
Python 3.12, anaconda):
```
keys = enemy_buildings_summary, enemy_hp, enemy_positions, explored_cells,
       explored_percent, game_tick, unit_hp, unit_positions, units_killed
unit_positions = {actor_id: {cell_x, cell_y[, target, activity, ...]}}
step() -> (obs, reward=0.0 hardcoded, done:bool, info={game_tick, warnings})
```
No `minimap` ASCII, no economy/military/result/reward, no terrain.

**Consequence:** Phase 0 builds a Bench-side episode loop that reuses
*components* (reward_funcs, minimap_renderer, scenario loader, action parser)
behind a **Rust→schema adapter** (`openra_bench/rust_adapter.py`). The adapter
is the crux and overlaps directly with the "make Rust faithful" workstream:

- `unit_positions` → `units_summary` (renderer/prompt schema)
- `enemy_positions` + `enemy_buildings_summary` → `enemy_summary`
- synthesize ASCII `minimap` from `explored_cells` + scenario map dims
- load `terrain_png` from the scenario's base `.oramap` (as Training does)
- derive scoring signals (kills, discovery, exploration, outcome) from obs
  deltas since Rust `reward` is hardcoded 0.0 — feeds reward_funcs + the
  P/R/A diagnostics directly.

## Target Bench layout

```
OpenRA-Bench/
  openra_bench/
    eval_core.py      # thin wrapper over play_episodes_async, Rust backend forced
    agent.py          # REAL model agent (OpenAI-compatible), minimap multimodal
    scenarios/        # controlled eval scenarios (symlink/copy + Bench-authored)
    rubrics.py        # verifiable + composite scoring (reuse Training)
    pairwise.py       # 1v1 adversarial orchestration + Elo
  evaluate.py         # fixed-scenario CLI (rewritten to use eval_core)
  compare.py          # CI comparison front-end (wraps build_eval_comparison)
  app.py              # leaderboard (fed by both modes)
```

## Phases

### Phase 0 — Integration spine (no Rust changes)
- Bench depends on `openra_rl_training` + `openra_train`.
- `eval_core.py`: wrap `play_episodes_async`, force Rust pool.
- `agent.py`: real OpenAI-compatible model agent w/ minimap PNG.
- Rewrite `evaluate.py` → fixed-scenario eval producing `eval_stats.json`.
- `compare.py` → 95% CI tables. Wire results into `app.py` leaderboard.
- Validate on rush-hour + scout-maginot (these *are* the perception tasks).

### Phase 1 — Adversarial 1v1 (combat-only, current mechanics)
- Rust: add a second RL-controlled player slot in a 1v1 map (both sides accept
  Commands; remove scripted-enemy assumption).
- Bench `pairwise.py`: two-model orchestration, win-rate + Elo, leaderboard.

### Phase 2 — Controlled scenario library (current mechanics)
- Author perception/spatial scenarios that isolate the unexplored→target
  connection failure; maze/chokepoint pathfinding; multi-target prioritization.
- Verifiable rubrics per scenario (intelligence_pct, path-optimality, etc.).

### Phase 3 — Rust mechanics expansion (unlocks scenario families)
- 3a Economy: ore/cash/harvester obs + HARVEST cmd + economy reward.
- 3b Production/tech: production queue, BUILD/TRAIN, available_production, power.
- Each sub-phase ships its scenario family + rubrics.

## Test coverage (live engine, no mocks)

`tests/test_rust_integration.py` — 17 tests, ~1.9s, boots real
`openra_train` with rule-based bots (idle / charge / hunter):
- tool correctness: move_units reaches target; idle units hold; attack
  path; reset schema; same-seed determinism (bit-for-bit).
- corner cases: empty command list, invalid unit id (warns, no raise),
  invalid attack target, out-of-bounds move — all safe.
- invariants: explored% and units_killed monotonic non-decreasing;
  discovery set cumulative.
- stack: adapter signal tracking; win-condition predicates + composites
  + unknown-key rejection; **deterministic win/fail plumbing** (trivially
  true win/fail conditions ⇒ exact outcome, not bot-skill dependent);
  all authored packs run end-to-end.

## Sequencing (locked)

`Phase 0 → Phase 2 (+ P/R/A diagnostics) → Phase 1 → Phase 3`

Scenario breadth + per-link diagnostics first: fastest path to exposing real
model strengths/weaknesses on current Rust mechanics. Adversarial 1v1 and the
economy/production engine work follow.

## Model provider abstraction (Phase 0)

`openra_bench/agent.py` exposes a provider-agnostic agent. Adapters:

- **openai-compatible** (default): covers local vLLM (matches Training's rollout
  path) and **OpenRouter** (test target). Same Chat Completions + multimodal
  `image_url` for the minimap PNG. Base URL + key from config/env.
- **bedrock**: separate adapter (AWS SDK / Converse API), added when needed.

Selected via Bench config (`provider`, `base_url`, `model`, `api_key_env`).
Phase 0 validates with OpenRouter.
