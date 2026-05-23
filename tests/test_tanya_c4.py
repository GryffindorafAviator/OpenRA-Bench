"""End-to-end guardrail: `Command.c4_detonate` lets Tanya walk to an
enemy building and instantly destroy it. Tanya survives the
detonation (MVP: no escape animation — she just stays put unharmed).

Mirror of the Rust engine acceptance suite
(`OpenRA-Rust/openra-sim/tests/test_tanya_c4.rs`) — exercises the
same loop via the Python `OpenRAEnv` boundary so the bench-side
`Command.c4_detonate` shim is pinned.

NOTE: this test depends on the wheel having been rebuilt after the
Tanya-C4 engine commit (the new `Command.c4_detonate` staticmethod
was added in that commit). Until the wheel is rebuilt the test is
SKIPPED, not failed, so a stale environment doesn't mask other test
results.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: tanya-c4-test
      title: 'Tanya C4 detonates enemy proc (engine smoke)'
      capability: action
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'hero-asset demolition: walk-up, plant, exit'
    base_map: rush-hour-arena
    starting_cash: 0
    base:
      agent: {faction: allies}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, c4_detonate]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 6000}
      actors:
        - {type: tanya, owner: agent, position: [16, 20]}
        - {type: proc, owner: enemy, position: [22, 20]}
    levels:
      easy:
        description: 'tanya walks ~6 cells east and C4s the proc'
        starting_cash: 0
        win_condition: {enemy_buildings_destroyed_gte: 1}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
      medium:
        description: 'placeholder'
        starting_cash: 0
        win_condition: {enemy_buildings_destroyed_gte: 1}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
      hard:
        description: 'placeholder'
        starting_cash: 0
        win_condition: {enemy_buildings_destroyed_gte: 1}
        fail_condition: {after_ticks: 9001}
        max_turns: 60
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="tanya_c4_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_tanya_c4_destroys_enemy_proc_via_command_c4_detonate():
    from openra_train import Command

    if not hasattr(Command, "c4_detonate"):
        pytest.skip(
            "Command.c4_detonate not available — rebuild the wheel "
            "(maturin develop --release) after the Tanya-C4 engine commit."
        )

    from openra_bench.rust_adapter import RustObsAdapter

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

        own_units = render0.get("own_units", []) or []
        tanyas = [u for u in own_units if str(u.get("type", "")).lower() == "tanya"]
        assert len(tanyas) == 1, (
            f"expected exactly one tanya in own_units, got {own_units}"
        )
        tanya_id = str(tanyas[0]["id"])

        enemy_buildings = (
            render0.get("enemy_buildings")
            or render0.get("enemies", [])
            or []
        )
        proc_id = None
        for b in enemy_buildings:
            if str(b.get("type", "")).lower() == "proc":
                proc_id = str(b.get("id"))
                break
        if proc_id is None:
            for a in render0.get("actors", []) or []:
                if (
                    str(a.get("type", "")).lower() == "proc"
                    and str(a.get("owner", "")) not in (
                        "agent",
                        str(render0.get("agent_player_id", "")),
                    )
                ):
                    proc_id = str(a.get("id"))
                    break
        assert proc_id is not None, (
            "could not find enemy proc id in initial observation "
            f"(render0 keys={list(render0.keys())})"
        )

        # Issue C4Detonate and step until the proc is gone (or done).
        destroyed = False
        for _ in range(40):
            obs, _r, done, _i = env.step(
                [Command.c4_detonate([tanya_id], proc_id)]
            )
            ad.observe(obs)
            rs = ad.render_state()
            enemy_b = (
                rs.get("enemy_buildings")
                or rs.get("enemies", [])
                or []
            )
            still_proc = any(
                str(b.get("type", "")).lower() == "proc"
                and str(b.get("id")) == proc_id
                for b in enemy_b
            )
            if not still_proc:
                destroyed = True
                break
            if done:
                break

        assert destroyed, (
            "Command.c4_detonate should have destroyed the enemy proc "
            "within the step budget"
        )

        # Tanya survives the detonation.
        own_units_final = ad.render_state().get("own_units", []) or []
        live_tanya = [
            u for u in own_units_final if str(u.get("type", "")).lower() == "tanya"
        ]
        assert live_tanya, (
            "tanya should survive the C4 detonation; she is missing from "
            "own_units after the blast"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
