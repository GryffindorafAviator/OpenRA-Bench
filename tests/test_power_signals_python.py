"""Python-adapter guardrail for the power-signal engine fix.

This is the bench-side mirror of OpenRA-Rust/openra-sim/tests/
test_power_signals.rs. It loads a tiny scenario with one `powr` and
one `proc` pre-placed for the agent, resets the env, steps a few
turns, and asserts that the Python observation surfaces non-zero
`power_provided` / `power_drained` via the rendered state.

Before the recompute-at-snapshot fix the totals stayed at zero for
any pre-placed scenario building (only `order_place_building` updated
the PowerManager trait), so `power_surplus_gte` / `power_provided_gte`
were inert in scenarios. This test pins the fix in place.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: power-signals-test
      title: 'Power signals smoke'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'sensor surfaces non-zero readings'
    base_map: rush-hour-arena
    starting_cash: 1000
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 1000}
      actors:
        - {type: powr, owner: agent, position: [20, 20]}
        - {type: proc, owner: agent, position: [24, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'tiny pre-placed power pack'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      medium:
        description: 'tiny pre-placed power pack'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
      hard:
        description: 'tiny pre-placed power pack'
        starting_cash: 1000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 100
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="power_signals_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_pre_placed_powr_and_proc_surface_nonzero_power():
    pack_path = _pack_tmp()
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        scen_path = _scenario_to_tmp_yaml(compiled)
    except Exception:
        pack_path.unlink(missing_ok=True)
        raise
    pool = RustEnvPool(size=1, scenario_path=scen_path)
    env = pool.acquire()
    try:
        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        render0 = ad.render_state()
        p0 = int(render0.get("power_provided", 0) or 0)
        d0 = int(render0.get("power_drained", 0) or 0)
        assert p0 > 0, (
            "Pre-placed `powr` must surface `power_provided > 0` in the "
            f"Python obs; got {p0}. (Engine bug: pre-placed buildings "
            "bypass order_place_building.)"
        )
        assert d0 > 0, (
            "Pre-placed `proc` must surface `power_drained > 0` in the "
            f"Python obs; got {d0}."
        )

        # Step a couple of decision turns — totals stay consistent.
        from openra_train import Command
        for _ in range(3):
            obs, _r, _done, _i = env.step([Command.observe()])
            ad.observe(obs)
        render1 = ad.render_state()
        p1 = int(render1.get("power_provided", 0) or 0)
        d1 = int(render1.get("power_drained", 0) or 0)
        assert p1 == p0, f"power_provided drifted: {p0} -> {p1}"
        assert d1 == d0, f"power_drained drifted: {d0} -> {d1}"
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
