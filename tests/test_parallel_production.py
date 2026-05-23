"""Python-adapter guardrail for the parallel-production engine fix.

Bench-side mirror of OpenRA-Rust/openra-sim/tests/
test_parallel_production.rs. Multiple production buildings of the same
category must produce IN PARALLEL (OpenRA parity): two war factories
roughly double vehicle throughput.

Before the fix the engine modelled production as ONE per-player queue
per category — a 2nd war factory added zero throughput. This test
pins the fix: a base with TWO pre-placed `weap` buildings, spam-
building `2tnk` with ample cash, must field meaningfully more tanks in
a fixed turn budget than the same base with ONE `weap`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level


# easy = ONE war factory; medium = TWO war factories. Same map, same
# cash, same build policy — only the factory count differs.
_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: parallel-production-test
      title: 'Parallel production smoke'
      capability: reasoning
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'two assembly lines build twice as fast'
    base_map: rush-hour-arena
    starting_cash: 60000
    base:
      agent: {faction: allies, cash: 60000}
      enemy: {faction: soviet, cash: 0}
      tools: [build, observe]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 20000}
      actors:
        - {type: fact, owner: agent, position: [10, 20]}
        - {type: powr, owner: agent, position: [13, 20]}
        - {type: powr, owner: agent, position: [13, 23]}
        - {type: proc, owner: agent, position: [16, 20]}
        - {type: fix,  owner: agent, position: [16, 23]}
        - {type: weap, owner: agent, position: [19, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      # win/fail are intentionally unreachable so the episode runs the
      # full `max_turns` budget — the test reads the peak tank count
      # from the policy closure, not the episode outcome.
      easy:
        description: 'one war factory'
        starting_cash: 60000
        win_condition: {unit_type_count_gte: {type: 2tnk, n: 9999}}
        fail_condition: {after_ticks: 999999}
        max_turns: 30
      medium:
        description: 'two war factories'
        starting_cash: 60000
        win_condition: {unit_type_count_gte: {type: 2tnk, n: 9999}}
        fail_condition: {after_ticks: 999999}
        max_turns: 30
        overrides:
          actors:
            - {type: fact, owner: agent, position: [10, 20]}
            - {type: powr, owner: agent, position: [13, 20]}
            - {type: powr, owner: agent, position: [13, 23]}
            - {type: proc, owner: agent, position: [16, 20]}
            - {type: fix,  owner: agent, position: [16, 23]}
            - {type: weap, owner: agent, position: [19, 20]}
            - {type: weap, owner: agent, position: [22, 20]}
            - {type: fact, owner: enemy, position: [120, 20]}
      hard:
        description: 'two war factories'
        starting_cash: 60000
        win_condition: {unit_type_count_gte: {type: 2tnk, n: 9999}}
        fail_condition: {after_ticks: 999999}
        max_turns: 30
        overrides:
          actors:
            - {type: fact, owner: agent, position: [10, 20]}
            - {type: powr, owner: agent, position: [13, 20]}
            - {type: powr, owner: agent, position: [13, 23]}
            - {type: proc, owner: agent, position: [16, 20]}
            - {type: fix,  owner: agent, position: [16, 23]}
            - {type: weap, owner: agent, position: [19, 20]}
            - {type: weap, owner: agent, position: [22, 20]}
            - {type: fact, owner: enemy, position: [120, 20]}
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="parallel_prod_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def _spam_tank_policy(peak: dict):
    """Keep the Vehicle queue saturated (always several 2tnk queued so
    build TIME, not idle gaps, is the only constraint) and record the
    peak number of finished 2tnk seen in `units_summary`."""

    def pol(obs, Cmd):
        summary = obs.get("units_summary", []) or []
        tanks = sum(
            1 for u in summary if (u.get("type") or u.get("kind")) == "2tnk"
        )
        peak["tanks"] = max(peak.get("tanks", 0), tanks)
        prod = obs.get("production", []) or []
        in_flight = sum(1 for p in prod if p == "2tnk")
        cmds = []
        # Keep at least 4 tanks queued at all times.
        for _ in range(max(0, 4 - in_flight)):
            cmds.append(Cmd.build("2tnk"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds

    return pol


def test_two_war_factories_outproduce_one():
    pack_path = _pack_tmp()
    try:
        pack = load_pack(pack_path)
        one_weap = compile_level(pack, "easy")
        two_weap = compile_level(pack, "medium")
    except Exception:
        pack_path.unlink(missing_ok=True)
        raise

    peak_one: dict = {}
    peak_two: dict = {}
    try:
        run_level(one_weap, _spam_tank_policy(peak_one), seed=1)
        run_level(two_weap, _spam_tank_policy(peak_two), seed=1)
    finally:
        pack_path.unlink(missing_ok=True)

    tanks_one = peak_one.get("tanks", 0)
    tanks_two = peak_two.get("tanks", 0)
    print(
        f"1 weap fielded {tanks_one} tanks; "
        f"2 weap fielded {tanks_two} tanks (same turn budget)"
    )

    assert tanks_one >= 1, (
        f"sanity: a single war factory should field at least one tank "
        f"(got {tanks_one})"
    )
    assert tanks_two > tanks_one, (
        f"two war factories must out-produce one (got 2-weap={tanks_two} "
        f"vs 1-weap={tanks_one}); parallel production is not working"
    )
    assert tanks_two >= 1.6 * tanks_one, (
        f"two war factories must field >= 1.6x the tanks of one "
        f"(got 2-weap={tanks_two} vs 1-weap={tanks_one})"
    )
