# Vendored dependencies

OpenRA-Bench vendors a small, faithful subset of two sibling repos so the
benchmark runs without external source checkouts, and so the evaluation
stack is frozen for reproducibility.

## `openra_rl_training/` — from OpenRA-RL-Training
- `scenario.py` — `ScenarioDefinition`, `VALID_ACTOR_TYPES`
- `training/reward_funcs.py` — `DEFAULT_REWARD_WEIGHTS` (composite scorer)
- `training/rust_env_pool.py` — `RustEnvPool` (wraps the engine)
- `training/minimap_renderer.py` — `render_minimap` (terrain minimap)

## `openra_env/` — from OpenRA-RL
- `game_data.py` — `RA_UNITS` / `RA_BUILDINGS` (consumed by `scenario.py`)

These are **verbatim copies** — do not hand-edit. To update: re-copy from
the source repos and re-run the full suite (`pytest tests/`).

## NOT vendored
- **`openra_train`** — the Rust engine, a compiled extension. Build it
  from the OpenRA-Rust repo: `maturin develop --release`. It cannot be
  vendored as source.
- **`openra_env.client` / `openra_env.models`** — the legacy gRPC client.
  The bench never imports them at runtime (the only reference is example
  code inside a docstring), so they are intentionally left out.
