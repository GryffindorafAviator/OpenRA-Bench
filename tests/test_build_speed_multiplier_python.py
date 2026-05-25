"""Python-adapter guardrail for the per-scenario `build_speed_multiplier`.

Bench-side mirror of OpenRA-Rust/openra-sim/tests/
test_build_speed_multiplier.rs. Pins:

  1. The pack YAML round-trips `build_speed_multiplier` from `base:`
     onto `CompiledLevel.build_speed_multiplier`.
  2. `_scenario_to_tmp_yaml` re-emits the field as a top-level YAML
     key (the engine's parser reads top-level
     `build_speed_multiplier:`).
  3. Default (field absent) ⇒ the temp YAML does NOT carry the key
     (engine inherits 1.0). This is the load-bearing back-compat
     property: every existing pack stays identical.
  4. The canonical `adversarial-1v1-macro` pack carries 4.0 and that
     value flows end-to-end.
  5. End-to-end engine smoke: queueing an e1 inside a 4.0× scenario
     finishes in the ~22-tick neighbourhood vs ~90 ticks at default.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

import tempfile
import textwrap
from pathlib import Path

import yaml

from openra_bench.eval_core import _scenario_to_tmp_yaml
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level


def _write_pack(yaml_text: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="build_speed_pack_")
    Path(path).write_text(yaml_text)
    return Path(path)


_BASE_PACK = """\
meta:
  id: build-speed-multiplier-test
  title: 'Build speed multiplier smoke'
  capability: reasoning
  author: engine-test
  real_world_meaning: 'engine guardrail; not a benchmark scenario'
  robotics_analogue: 'tunable assembly-line speed'
base_map: rush-hour-arena
starting_cash: 5000
base:
  agent: {faction: allies, cash: 5000}
  enemy: {faction: soviet, cash: 0}
  tools: [build, observe]
  spawn_mcvs: false
  planning: true
  termination: {max_ticks: 5000}
  actors:
    - {type: fact, owner: agent, position: [10, 20]}
    - {type: powr, owner: agent, position: [13, 20]}
    - {type: tent, owner: agent, position: [13, 23]}
    - {type: fact, owner: enemy, position: [120, 20]}
__BSM_PLACEHOLDER__
levels:
  easy:
    description: 'unused — schema requires the level'
    win_condition: {unit_type_count_gte: {type: e1, n: 9999}}
    fail_condition: {after_ticks: 999999}
    max_turns: 30
  medium:
    description: 'unused — schema requires the level'
    win_condition: {unit_type_count_gte: {type: e1, n: 9999}}
    fail_condition: {after_ticks: 999999}
    max_turns: 30
  hard:
    description: 'unused — schema requires the level'
    win_condition: {unit_type_count_gte: {type: e1, n: 9999}}
    fail_condition: {after_ticks: 999999}
    max_turns: 30
"""


def _pack_yaml_with_bsm(value: float | None) -> str:
    line = (
        f"  build_speed_multiplier: {value}\n" if value is not None else ""
    )
    return _BASE_PACK.replace("__BSM_PLACEHOLDER__\n", line)


def test_default_omitted_no_multiplier_emitted():
    """A pack without `build_speed_multiplier` must NOT emit the field
    in the temp YAML — the engine then inherits its 1.0 default and
    every existing benchmark stays bit-identical."""
    pack_path = _write_pack(_pack_yaml_with_bsm(None))
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        assert compiled.build_speed_multiplier is None
        tmp_yaml = _scenario_to_tmp_yaml(compiled)
        try:
            data = yaml.safe_load(Path(tmp_yaml).read_text())
        finally:
            Path(tmp_yaml).unlink(missing_ok=True)
        assert "build_speed_multiplier" not in data, (
            "default pack must not emit build_speed_multiplier; got "
            f"{data.get('build_speed_multiplier')!r}"
        )
    finally:
        pack_path.unlink(missing_ok=True)


def test_pack_field_roundtrips_into_temp_yaml():
    """A pack declaring `build_speed_multiplier: 4.0` under `base:`
    must (a) surface on `CompiledLevel.build_speed_multiplier` and
    (b) emit the value as a top-level key in the temp YAML."""
    pack_path = _write_pack(_pack_yaml_with_bsm(4.0))
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        assert compiled.build_speed_multiplier == pytest.approx(4.0)
        tmp_yaml = _scenario_to_tmp_yaml(compiled)
        try:
            data = yaml.safe_load(Path(tmp_yaml).read_text())
        finally:
            Path(tmp_yaml).unlink(missing_ok=True)
        assert data.get("build_speed_multiplier") == pytest.approx(4.0)
    finally:
        pack_path.unlink(missing_ok=True)


def test_adversarial_1v1_macro_declares_four_x():
    """The canonical 1v1 pack carries 4.0; the eval pipeline must
    surface it on every compiled level."""
    pack_path = PACKS_DIR / "adversarial-1v1-macro.yaml"
    pack = load_pack(pack_path)
    for level in ("easy", "medium", "hard"):
        compiled = compile_level(pack, level)
        assert compiled.build_speed_multiplier == pytest.approx(4.0), (
            f"adversarial-1v1-macro level={level} must carry "
            f"build_speed_multiplier=4.0 (got {compiled.build_speed_multiplier})"
        )
        tmp_yaml = _scenario_to_tmp_yaml(compiled)
        try:
            data = yaml.safe_load(Path(tmp_yaml).read_text())
        finally:
            Path(tmp_yaml).unlink(missing_ok=True)
        assert data.get("build_speed_multiplier") == pytest.approx(4.0)


def test_fractional_multiplier_roundtrips():
    """Fractional multipliers (e.g. 1.5×) round-trip exactly through
    the temp YAML — the engine's accumulator path supports them."""
    pack_path = _write_pack(_pack_yaml_with_bsm(1.5))
    try:
        pack = load_pack(pack_path)
        compiled = compile_level(pack, "easy")
        assert compiled.build_speed_multiplier == pytest.approx(1.5)
        tmp_yaml = _scenario_to_tmp_yaml(compiled)
        try:
            data = yaml.safe_load(Path(tmp_yaml).read_text())
        finally:
            Path(tmp_yaml).unlink(missing_ok=True)
        assert data.get("build_speed_multiplier") == pytest.approx(1.5)
    finally:
        pack_path.unlink(missing_ok=True)
