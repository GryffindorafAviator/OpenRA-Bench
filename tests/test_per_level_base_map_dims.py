"""Regression pin: per-level `overrides.base_map` dims must propagate
into every engine grid (shroud / explored / terrain / pathfinding).

Origin: Qwen 9B v1.0 sweep crashed on
`mfb-third-base-against-clock:hard:seed1` with

    pyo3_runtime.PanicException: index out of bounds: the len is 5120
    but the index is 6422

5120 == 128 * 40 (the easy/medium dims), 6422 == y * width + x for the
south-mirror agent harv at [22, 50] in a 128-wide flat array. The hard
tier overrides `base_map.{width:128, height:64}` while easy/medium use
128 × 40 — `mfb-third-base-against-clock` is the only pack in the
suite that flips per-level dims under a shared `base_map.name:` slug
(see `tests/.../mapgen.py::spec_id`).

The fix landed before the engine HEAD documented in this pin (the
sweep was run before PRs #16/#17/#18 merged into engine `main`); the
panic no longer reproduces. This test locks the bench-side behaviour
in place: every level compiles, the engine accepts the resulting
scenario yaml, `reset()` succeeds, and a handful of `observe()` steps
do NOT panic — across every hard-seed (1-4), since hard is the
only tier that triggers the 128 × 64 dims and seeds 1/2 hit the
south-mirror spawn group (y = 50) that surfaced the original crash.
"""

from __future__ import annotations

import pytest

from openra_bench.eval_core import _scenario_to_tmp_yaml
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level


PACK_ID = "mfb-third-base-against-clock"


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_per_level_base_map_dims_no_panic(level: str) -> None:
    """Compile + reset + step the pack at every level. Pre-fix this
    crashed on `hard` (128 × 64) when a stale shared oramap or grid
    sized at the pack-level (128 × 40) was reused for the hard tier.
    """
    openra_train = pytest.importorskip("openra_train")

    pack = load_pack(PACKS_DIR / f"{PACK_ID}.yaml")
    compiled = compile_level(pack, level)
    yaml_path = _scenario_to_tmp_yaml(compiled)

    # Drive every seed: hard's `spawn_point: 1` (south mirror at y=50)
    # is the spawn group that surfaced the original index-out-of-bounds
    # — exercising all four seeds guarantees both groups are hit even
    # if the round-robin order shifts.
    for seed in (1, 2, 3, 4):
        env = openra_train.OpenRAEnv(str(yaml_path), seed)
        env.reset()
        for _ in range(10):
            result = env.step([openra_train.Command.observe()])
            done = result[2] if len(result) >= 3 else False
            if done:
                break


def test_hard_seed1_specifically_does_not_panic() -> None:
    """Tight pin on the exact sweep cell that crashed
    (`mfb-third-base-against-clock:hard:seed1`). Keep this even though
    `test_per_level_base_map_dims_no_panic` already covers it — a future
    refactor that drops the parametrize must still trip on this name.
    """
    openra_train = pytest.importorskip("openra_train")

    pack = load_pack(PACKS_DIR / f"{PACK_ID}.yaml")
    compiled = compile_level(pack, "hard")
    yaml_path = _scenario_to_tmp_yaml(compiled)
    env = openra_train.OpenRAEnv(str(yaml_path), 1)
    env.reset()
    for _ in range(10):
        result = env.step([openra_train.Command.observe()])
        done = result[2] if len(result) >= 3 else False
        if done:
            break
