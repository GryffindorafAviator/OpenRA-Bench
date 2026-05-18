"""Scenario-controlled tool allow/deny with a default core set.

A scenario decides which actions the model may call via its `tools:`
allowlist; if it declares none it gets DEFAULT_CORE_TOOLS (not every
tool). `observe` is always available. Full integ coverage of the
resolution rules + the scenario→agent path.
"""

from __future__ import annotations

from openra_bench.agent import (
    DEFAULT_CORE_TOOLS,
    ModelAgent,
    _TOOL_SCHEMAS,
    _tool_schemas,
)
from openra_bench.providers import ProviderConfig


def _names(allowed):
    return {t["function"]["name"] for t in _tool_schemas(allowed)}


def test_unset_gives_default_core_not_everything():
    n = _names(None)
    assert n == set(DEFAULT_CORE_TOOLS)
    assert n == _names([])  # empty list == unset
    assert "observe" in n and "surrender" not in n and "build" not in n
    assert len(n) < len(_TOOL_SCHEMAS)  # NOT all tools


def test_explicit_allowlist_is_exactly_honored():
    n = _names(["build", "place_building"])
    assert n == {"build", "place_building", "observe"}  # +safe no-op
    assert "move_units" not in n  # not in the allowlist → disallowed


def test_wildcard_exposes_everything():
    assert _names(["*"]) == set(_TOOL_SCHEMAS)
    assert _names(["all"]) == set(_TOOL_SCHEMAS)
    assert len(_names(["*"])) == 18


def test_unknown_tool_names_are_ignored_not_errors():
    # A typo'd / unimplemented tool name must not crash the eval; it is
    # simply dropped, and observe still keeps the turn valid.
    assert _names(["frobnicate", "move_units"]) == {"move_units", "observe"}
    assert _names(["definitely-not-a-tool"]) == {"observe"}


def test_observe_always_present_even_if_excluded():
    assert "observe" in _names(["attack_unit"])
    assert "observe" in _names(["build"])


def test_scenario_tools_flow_into_the_agent():
    # The scenario's `tools:` list is what ModelAgent is constructed
    # with → the model only sees those (+observe).
    agent = ModelAgent(
        ProviderConfig(vision=False),
        allowed_tools=["harvest", "build"],
        provider=type("P", (), {"complete": lambda *a, **k: None})(),
    )
    got = {t["function"]["name"] for t in agent.tools}
    assert got == {"harvest", "build", "observe"}


def test_default_core_is_movement_combat_only():
    # Sanity: the default core is the universal movement/combat verbs,
    # so it's safe on perception/combat scenarios without leaking
    # economy/structure/concede verbs.
    assert set(DEFAULT_CORE_TOOLS) == {
        "move_units", "attack_unit", "attack_move", "stop", "observe"
    }
