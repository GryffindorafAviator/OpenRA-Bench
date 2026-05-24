"""Regression: the engine default spawn_mcvs:true auto-seeded MCVs at
the map's spawn points (e.g. (124,36)), revealing a fog blob from
turn 1 and polluting unit counts. The bench must emit spawn_mcvs:false
so only scenario-declared actors exist."""
from __future__ import annotations
from pathlib import Path
import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.eval_core import _scenario_to_tmp_yaml

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def test_emitted_scenario_disables_auto_mcv():
    import yaml
    c = compile_level(
        load_pack(PACKS / "action-multiunit-coordination.yaml"), "easy")
    d = yaml.safe_load(open(_scenario_to_tmp_yaml(c)))
    assert d.get("spawn_mcvs") is False


def test_reset_reveals_only_agent_spawn_no_corner_blob():
    pytest.importorskip("openra_train")
    import openra_train as ot
    c = compile_level(
        load_pack(PACKS / "action-multiunit-coordination.yaml"), "easy")
    o = ot.OpenRAEnv(_scenario_to_tmp_yaml(c), 1).reset()
    ec = {tuple(x) for x in (o.get("explored_cells") or [])}
    # On the 48x40 audit-tight map: no phantom MCV in the bottom-right
    # corner (would be near (44,36) on this map).
    blob = [p for p in ec if p[0] >= 35 and p[1] >= 30]
    assert blob == [], f"phantom reveal in bottom-right: {blob[:8]}"
    # only the agent's own top-left spawn is revealed
    assert max(p[0] for p in ec) < 20 and max(p[1] for p in ec) < 20
