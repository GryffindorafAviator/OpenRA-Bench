"""Scenario `configs:` — named runnable configurations.

A pack may declare configs (each pins a difficulty level + fog_mode);
the eval then runs ONE cell per config (pack:config_name) instead of
the raw 3 levels. adversarial-duel uses this for the text-vs-vision
study: easy (structured-text fog) / medium (PNG minimap fog).
"""

from __future__ import annotations

from pathlib import Path

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


def test_adversarial_duel_declares_easy_structured_and_medium_vision():
    p = load_pack(PACKS / "adversarial-duel.yaml")
    by = {c.name: c for c in (p.configs or [])}
    assert set(by) == {"easy", "medium"}
    # easy = duel observed via structured TEXT fog (no minimap image)
    assert by["easy"].level == "easy"
    assert by["easy"].fog_mode == "structured"
    # medium = same duel family via the PNG minimap
    assert by["medium"].level == "medium"
    assert by["medium"].fog_mode == "vision"


def test_compile_config_sets_level_fog_and_cell_label():
    p = load_pack(PACKS / "adversarial-duel.yaml")
    cl = p.compile_config("easy", map_supported=True)
    assert cl.level == "easy"
    assert cl.fog_mode == "structured"
    assert cl.config_name == "easy"
    # interrupts declared in base flow to the compiled scenario
    it = getattr(cl.scenario, "interrupts", {}) or {}
    assert it.get("enemy_unit_spotted") and it.get("own_unit_destroyed")


def test_run_eval_emits_one_cell_per_config(monkeypatch):
    from openra_bench import run_eval

    captured: dict = {}

    def fake_compile_config(self, name, *, map_supported=True):
        captured.setdefault("configs", []).append(name)

        class C:
            map_supported = False  # short-circuit into skipped, no engine
        c = C()
        c.meta = self.meta
        return c

    monkeypatch.setattr(
        type(load_pack(PACKS / "adversarial-duel.yaml")),
        "compile_config", fake_compile_config, raising=False,
    )
    out = run_eval.evaluate(
        packs=[PACKS / "adversarial-duel.yaml"],
        levels=["easy", "medium", "hard"], seeds=[1],
    )
    # configs supersede the 3-level enumeration
    assert sorted(captured["configs"]) == ["easy", "medium"]
    sk = " ".join(out["skipped"])
    assert "adversarial-duel:easy" in sk


def test_legacy_pack_without_configs_still_uses_levels():
    p = load_pack(PACKS / "rush-hour.yaml")
    assert p.configs is None
    c = compile_level(p, "easy")
    assert c.config_name is None and c.fog_mode == "vision"


def test_agent_fog_mode_param_overrides_cfg():
    from openra_bench.agent import ModelAgent
    from openra_bench.providers import ProviderConfig

    a = ModelAgent(
        ProviderConfig(vision=True, fog_mode="vision"),
        allowed_tools=["observe"],
        provider=type("P", (), {"complete": lambda *x, **k: None})(),
        fog_mode="structured",
    )
    assert a._fog_mode == "structured"  # scenario config wins
