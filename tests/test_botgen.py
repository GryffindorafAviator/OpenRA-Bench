"""YAML-referenceable scripted opponent (botgen) — bench surface.

The behaviour runs engine-side; here we test the bench-facing
catalogue + compile-time validation (unknown name fails fast) and
that a known bot survives compile/serialization.
"""
from __future__ import annotations

import yaml
import pytest

from openra_bench.botgen import BEHAVIORS, read_enemy_bot, validate_enemy_bot
from openra_bench.scenarios.loader import PACKS_DIR
from openra_bench.scenarios.schema import ScenarioPack


def test_behaviour_catalogue():
    assert BEHAVIORS == {
        "hunt", "rusher", "patrol", "turtle", "guard", "raider",
    }


def test_read_enemy_bot_field_and_alias():
    assert read_enemy_bot({"faction": "soviet", "bot_type": "Hunt"}) == "hunt"
    assert read_enemy_bot({"bot": "rusher"}) == "rusher"      # alias
    assert read_enemy_bot({"faction": "soviet"}) is None
    assert read_enemy_bot({"bot_type": ""}) is None           # default
    assert read_enemy_bot(None) is None


def test_validate_rejects_unknown_behaviour():
    assert validate_enemy_bot({"bot_type": "turtle"}) == "turtle"
    with pytest.raises(ValueError):
        validate_enemy_bot({"bot_type": "stampede"})


def _pack_with_bot(bot: str) -> dict:
    base = yaml.safe_load((PACKS_DIR / "adversarial-duel.yaml").read_text())
    base["base"]["enemy"] = {"faction": "soviet", "bot_type": bot}
    for lv in base["levels"].values():
        lv["overrides"] = {}
    return base


def test_compile_accepts_known_and_rejects_unknown_bot():
    good = ScenarioPack(**_pack_with_bot("hunt")).compile("easy")
    # The real contract: bot_type survives serialization into the tmp
    # YAML the Rust parser reads.
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    import os

    tmp = _scenario_to_tmp_yaml(good)
    try:
        assert "bot_type: hunt" in open(tmp).read()
    finally:
        os.unlink(tmp)
    with pytest.raises(ValueError):
        ScenarioPack(**_pack_with_bot("not-a-bot")).compile("easy")
