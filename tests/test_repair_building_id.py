"""End-to-end guardrail: `repair` heals a damaged pre-placed building
when targeted via the REAL engine actor id.

The engine repair logic (`world.rs`) works when given the real
building actor id, but `RustObsAdapter.render_state()` previously
built `own_buildings` as `{type, cell_x, cell_y}` — it dropped the
id. `prompt_v2.py` then assigned `id = list-index`, and
`env.rs::resolve_owned` rejected the bogus id, so no agent could
target a building for `repair` / `sell` / `power_down` /
`set_primary`.

This test:
  1. pre-places a `proc` at `health: 35` (depends on the Gap-1 fix),
  2. confirms the engine surfaces a real `id` in `own_buildings`,
  3. issues `Command.repair([real_id])` and steps the env,
  4. asserts the building's HP fraction climbs.

Mirror: relies on the Gap-1 engine fix (`health:` field) and the
Gap-2 adapter fix (real id in `own_buildings` / `buildings_summary`).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
from openra_bench.prompt_v2 import state_from_render
from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios.loader import compile_level, load_pack


_PACK_YAML = textwrap.dedent(
    """\
    meta:
      id: repair-building-id-test
      title: 'Repair via real building id smoke'
      capability: action
      author: engine-test
      real_world_meaning: 'engine guardrail; not a benchmark scenario'
      robotics_analogue: 'actuator targets a real asset by id, not index'
    base_map: rush-hour-arena
    starting_cash: 5000
    base:
      agent: {faction: allies, cash: 5000}
      enemy: {faction: soviet, cash: 0}
      tools: [observe, repair]
      spawn_mcvs: false
      planning: true
      termination: {max_ticks: 4000}
      actors:
        - {type: proc, owner: agent, position: [20, 20], health: 35}
        - {type: fact, owner: agent, position: [24, 20]}
        - {type: fact, owner: enemy, position: [120, 20]}
    levels:
      easy:
        description: 'pre-placed damaged proc to be repaired by id'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 120
      medium:
        description: 'pre-placed damaged proc to be repaired by id'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 120
      hard:
        description: 'pre-placed damaged proc to be repaired by id'
        starting_cash: 5000
        win_condition: {within_ticks: 9000}
        fail_condition: {after_ticks: 9001}
        max_turns: 120
    """
)


def _pack_tmp() -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="repair_id_pack_")
    Path(path).write_text(_PACK_YAML)
    return Path(path)


def test_repair_via_real_building_id_heals_damaged_proc():
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
        from openra_train import Command

        ad = RustObsAdapter()
        ad.observe(env.reset(seed=1))
        render0 = ad.render_state()

        own = {
            str(b.get("type", "")).lower(): b
            for b in render0.get("own_buildings", []) or []
        }
        assert "proc" in own, f"proc must surface in own_buildings: {own}"
        proc = own["proc"]

        # Gap 2: the REAL engine actor id must be present (not a
        # list-index). It must be a non-empty string the resolver
        # accepts.
        real_id = str(proc.get("id", ""))
        assert real_id != "", (
            "own_buildings must surface the real engine actor id "
            f"(got {proc.get('id')!r}). Without it `repair` cannot "
            "target the building."
        )

        # And prompt_v2's buildings_summary must propagate that same
        # real id (not enumerate()).
        st = state_from_render(render0)
        bs = {
            str(b.get("type", "")).lower(): b
            for b in st.get("buildings_summary", [])
        }
        assert str(bs["proc"]["id"]) == real_id, (
            "prompt_v2.buildings_summary must keep the real engine id; "
            f"got {bs['proc']['id']!r} vs {real_id!r}"
        )

        hp0 = float(proc.get("hp", 1.0) or 0.0)
        assert hp0 < 0.6, (
            f"proc with health:35 should start damaged; got hp={hp0}"
        )

        # Issue repair against the real id and step several turns.
        hp_last = hp0
        for _ in range(20):
            obs, _r, done, _i = env.step([Command.repair([real_id])])
            ad.observe(obs)
            rs = ad.render_state()
            cur = {
                str(b.get("type", "")).lower(): b
                for b in rs.get("own_buildings", []) or []
            }
            if "proc" in cur:
                hp_last = float(cur["proc"].get("hp", hp_last) or hp_last)
            if done or hp_last >= 0.99:
                break

        assert hp_last > hp0 + 0.05, (
            "`Command.repair([real_building_id])` must heal a damaged "
            f"pre-placed building; hp went {hp0:.3f} -> {hp_last:.3f}. "
            "(Bug: building id dropped ⇒ resolve_owned rejects the "
            "list-index id ⇒ repair is a no-op.)"
        )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(scen_path).unlink(missing_ok=True)
        pack_path.unlink(missing_ok=True)
