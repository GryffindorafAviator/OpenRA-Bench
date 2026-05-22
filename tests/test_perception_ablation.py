"""The perception ablation grid: observation channel × fog of war.

`fog_mode` spans 3 channels × 2 fog states = 6 cells:
  structured / structured-clear  — text only, no image
  vision     / vision-clear      — text briefing + PNG minimap
  image      / image-clear       — image-PRIMARY: text redacted of all
                                   positions; the labelled minimap is
                                   the sole spatial source.
The `-clear` cells disable fog (engine `reveal_map: true`) — they are
perfect-information CONTROLS that isolate the perception cost; the
fogged cells keep the no-cheat bar.

Decomposition the grid enables:
  score(vision-clear) − score(vision)     = partial-observability cost
  score(vision)       − score(image)      = text-scaffolding crutch
  score(vision)       − score(structured) = visual-channel advantage
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import _scenario_to_tmp_yaml, run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.schema import PERCEPTION_MODES, ScenarioConfig

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
_PACK = "perception-count-the-threat.yaml"


def test_perception_modes_are_the_3x2_grid():
    assert set(PERCEPTION_MODES) == {
        "structured", "structured-clear",
        "vision", "vision-clear",
        "image", "image-clear",
    }


@pytest.mark.parametrize("mode", PERCEPTION_MODES)
def test_scenario_config_accepts_every_mode(mode):
    cfg = ScenarioConfig(name="cell", level="easy", fog_mode=mode)
    assert cfg.fog_mode == mode


def test_compiled_level_reveal_map_and_channel_properties():
    """`-clear` ⇒ reveal_map True; channel is independent of fog."""
    c = compile_level(load_pack(PACKS / _PACK), "easy")
    cases = {
        "structured": (False, "structured"),
        "structured-clear": (True, "structured"),
        "vision": (False, "vision"),
        "vision-clear": (True, "vision"),
        "image": (False, "image"),
        "image-clear": (True, "image"),
    }
    for mode, (reveal, channel) in cases.items():
        c.fog_mode = mode
        assert c.reveal_map is reveal, mode
        assert c.obs_channel == channel, mode


def test_tmp_yaml_emits_reveal_map_only_for_clear_modes():
    """The no-fog cells must write `reveal_map: true` into the engine
    scenario YAML; the fogged cells must not mention it at all."""
    c = compile_level(load_pack(PACKS / _PACK), "easy")
    for mode in PERCEPTION_MODES:
        c.fog_mode = mode
        data = yaml.safe_load(Path(_scenario_to_tmp_yaml(c)).read_text())
        if mode.endswith("-clear"):
            assert data.get("reveal_map") is True, mode
        else:
            assert "reveal_map" not in data, mode


def test_no_fog_cell_reveals_what_the_fogged_cell_hides():
    """End-to-end: an identical perception scenario, observe-only.
    Under fog a stall sees nothing; under `reveal_map` every enemy is
    observed and the whole map is explored — the perception control."""
    pack = load_pack(PACKS / _PACK)
    stall = lambda rs, Command: [Command.observe()]  # noqa: E731

    fogged = compile_level(pack, "medium")
    fogged.fog_mode = "vision"
    f = run_level(fogged, stall, seed=1)

    clear = compile_level(pack, "medium")
    clear.fog_mode = "vision-clear"
    c = run_level(clear, stall, seed=1)

    # Fog hides the enemies from an observe-only policy.
    assert len(f.signals.enemies_seen_ids) == 0
    assert f.signals.explored_percent < 50.0
    # No-fog reveals every enemy and the entire map.
    assert len(c.signals.enemies_seen_ids) > 0
    assert c.signals.explored_percent > 99.0


def test_perception_sweep_expands_to_six_cells_per_level():
    from openra_bench.run_eval import evaluate

    out = evaluate(
        [PACKS / _PACK], levels=["easy"], seeds=[1],
        perception_sweep=True, dry_run=True,
    )
    expected = {
        f"perception-count-the-threat:easy:{m}" for m in PERCEPTION_MODES
    }
    assert set(out["cells"]) == expected, out["cells"]


# ── Image-primary channel ────────────────────────────────────────────

def _render_state(mode: str) -> dict:
    """A live render_state for the count-the-threat pack under `mode`."""
    from openra_bench.rust_adapter import RustObsAdapter
    from openra_rl_training.training.rust_env_pool import RustEnvPool

    c = compile_level(load_pack(PACKS / _PACK), "medium")
    c.fog_mode = mode
    pool = RustEnvPool(size=1, scenario_path=_scenario_to_tmp_yaml(c))
    ad = RustObsAdapter()
    ad.observe(pool.acquire().reset(seed=1))
    return ad.render_state()


def test_perception_labels_are_friendly_and_unique():
    from openra_bench.prompt_v2 import perception_labels

    labels = perception_labels(_render_state("image-clear"))
    assert labels, "expected labels for own + enemy actors"
    # Each engine id maps to a distinct legible handle.
    assert len(set(labels.values())) == len(labels)
    assert any(v.startswith("tank-") for v in labels.values())
    assert any(v.startswith("enemy-") for v in labels.values())


def test_image_primary_briefing_has_no_coordinates():
    """The redacted briefing must carry WHAT, never WHERE."""
    import re

    from openra_bench.prompt_v2 import (briefing_image_primary,
                                        perception_labels)

    rs = _render_state("image-clear")
    text = briefing_image_primary(rs, perception_labels(rs))
    # No `@(x,y)` coordinate tokens, and the enemy line is not enumerated.
    assert not re.search(r"@\(\d+,\s*\d+\)", text), text
    assert "8xe3" not in text and "1xbarr" not in text
    assert "minimap" in text.lower()


def test_image_primary_minimap_labels_every_unit():
    from openra_bench.minimap import render_tactical_minimap
    from openra_bench.prompt_v2 import perception_labels

    rs = _render_state("image-clear")
    labels = perception_labels(rs)
    img = render_tactical_minimap(rs, scale=3, unit_labels=labels)
    assert img is not None and img.size[0] > 0


def test_to_commands_translates_labels_to_engine_ids():
    """In the image channel the model passes minimap handles; the agent
    maps them back to engine ids. Numeric ids pass straight through."""
    import openra_train

    from openra_bench.agent import _to_commands

    Command = openra_train.Command
    label_to_id = {"tank-1": "1004", "enemy-1": "1006"}
    cmds = _to_commands(
        [{"name": "attack_unit",
          "arguments": {"unit_ids": ["tank-1"], "target_id": "enemy-1"}}],
        Command, label_to_id,
    )
    assert '"1004"' in repr(cmds[0]) and '"1006"' in repr(cmds[0])
    # Empty map (every other channel) — numeric ids untouched.
    passthru = _to_commands(
        [{"name": "stop", "arguments": {"unit_ids": [1004]}}], Command, {},
    )
    assert '"1004"' in repr(passthru[0])


def test_image_primary_tools_retype_handles_as_strings():
    from openra_bench.agent import _image_primary_tools, _tool_schemas

    base = _tool_schemas(["move_units", "attack_unit"])
    img = _image_primary_tools(base)
    for t in img:
        props = t["function"]["parameters"]["properties"]
        if "unit_ids" in props:
            assert props["unit_ids"]["items"]["type"] == "string"
        if "target_id" in props:
            assert props["target_id"]["type"] == "string"
    # The base schemas are deep-copied, not mutated.
    assert (
        base[0]["function"]["parameters"]["properties"]
        ["unit_ids"]["items"]["type"] == "integer"
    )
