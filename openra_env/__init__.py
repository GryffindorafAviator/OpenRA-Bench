"""Vendored subset of OpenRA-RL's `openra_env` — see VENDOR.md.

Only `game_data` (RA_UNITS / RA_BUILDINGS, consumed by the vendored
`openra_rl_training.scenario`) is vendored. The gRPC `client` / `models`
are deliberately NOT vendored — the bench never imports them at runtime
(the one reference is example code inside a docstring).
"""
