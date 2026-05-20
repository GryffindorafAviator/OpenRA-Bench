"""forbidden_tools / tool_violations: bench-side procedural-compliance.

Tests the new schema field + signal + predicate end-to-end. The
real-world anchor is BFCL V4 / τ²-bench / IFBench: the agent has a
strict allowlist and any disallowed tool call must trip a fail clause.

Three things must hold:

1. The cmd-repr → tool-name decoder maps every Command variant to its
   snake_case allowlist key (move_units, attack_unit, …).
2. eval_core.run_level increments signals.tool_violations exactly once
   per disallowed call, regardless of whether the policy is scripted or
   wrapped by ModelAgent.
3. The `tool_violations_gte` predicate reads that counter and fails the
   episode (typical use: `tool_violations_gte: 1` in fail_condition).
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from openra_bench.eval_core import _cmd_tool_name, run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"


# ── 1. cmd-repr decoder ───────────────────────────────────────────────────


def test_cmd_tool_name_extracts_snake_case_from_command_repr():
    """The Rust Command enum stringifies as `Command::VariantName { … }`
    or `Command::VariantName`. _cmd_tool_name must produce the bench's
    snake_case tool key for every variant — covers the cmd type names
    actually observed live on the engine in this session."""
    # MoveUnits with payload
    class _Stub:
        def __init__(self, r: str):
            self.r = r

        def __repr__(self) -> str:
            return self.r

    assert _cmd_tool_name(_Stub('Command::MoveUnits { unit_ids: ["1"], '
                                'target_x: 10, target_y: 10 }')) == "move_units"
    # bare variant
    assert _cmd_tool_name(_Stub("Command::Observe")) == "observe"
    # multi-word variant
    assert _cmd_tool_name(_Stub('Command::AttackUnit { unit_ids: ["1"], '
                                'target_id: "2" }')) == "attack_unit"
    assert _cmd_tool_name(_Stub('Command::AttackMove { … }')) == "attack_move"
    assert _cmd_tool_name(_Stub('Command::Stop { unit_ids: ["1"] }')) == "stop"
    assert _cmd_tool_name(_Stub("Command::Surrender")) == "surrender"
    assert _cmd_tool_name(_Stub('Command::SetStance { stance: 2 }')) == "set_stance"
    assert _cmd_tool_name(_Stub('Command::PlaceBuilding { … }')) == "place_building"
    # Non-Command reprs → None, never raise
    assert _cmd_tool_name(_Stub("something else")) is None
    assert _cmd_tool_name(_Stub("")) is None


# ── 2. predicate reads from signals.tool_violations ───────────────────────


def test_tool_violations_gte_predicate_reads_signals():
    """tool_violations_gte: N is true iff signals.tool_violations >= N."""
    sig = types.SimpleNamespace(tool_violations=0)
    ctx = WinContext(signals=sig, render_state={"units_summary": []})
    assert evaluate({"tool_violations_gte": 1}, ctx) is False
    sig.tool_violations = 1
    assert evaluate({"tool_violations_gte": 1}, ctx) is True
    sig.tool_violations = 5
    assert evaluate({"tool_violations_gte": 3}, ctx) is True
    assert evaluate({"tool_violations_gte": 6}, ctx) is False


def test_tool_violations_default_zero_for_legacy_signals():
    """Signals that predate the field (e.g. raw types.SimpleNamespace
    in unit tests) should default to 0, not raise."""
    sig = types.SimpleNamespace()  # no tool_violations attr
    ctx = WinContext(signals=sig, render_state={"units_summary": []})
    assert evaluate({"tool_violations_gte": 1}, ctx) is False


# ── 3. _PHRASES has a translation (suite invariant elsewhere too) ─────────


def test_tool_violations_gte_has_a_phrase_translation():
    from openra_bench.game_knowledge import _PHRASES

    assert "tool_violations_gte" in _PHRASES
    p = _PHRASES["tool_violations_gte"]
    assert "forbidden tool" in p(1)
    assert "≥3" in p(3)


# ── 4. forbidden_tools roundtrips through schema ──────────────────────────


def test_forbidden_tools_roundtrips_through_compiled_level():
    """A pack that declares forbidden_tools on a Level surfaces it on
    the CompiledLevel. CompiledLevel.forbidden_tools is what
    eval_core.run_level reads."""
    # Use an existing pack and inject forbidden_tools at the Level model
    # level (we don't need to write a new YAML to verify roundtripping).
    pack = load_pack(PACKS / "custom-map-no-enemy.yaml")
    pack.levels["easy"].forbidden_tools = ["attack_unit", "attack_move"]
    c = compile_level(pack, "easy")
    assert c.forbidden_tools == ["attack_unit", "attack_move"]


# ── 5. live engine: forbidden cmds increment counter; allowed don't ───────


def test_forbidden_tools_counted_live_via_run_level():
    """The end-to-end test: a scripted policy that issues an
    attack_unit (forbidden) on every turn must accumulate
    tool_violations > 0 by episode end, and the tool_violations_gte:1
    fail clause must fire ⇒ outcome == 'loss'."""
    pytest.importorskip("openra_train")

    pack = load_pack(PACKS / "custom-map-no-enemy.yaml")
    # Hot-patch the easy level: forbid attack_unit + put it in fail.
    pack.levels["easy"].forbidden_tools = ["attack_unit", "attack_move"]
    from openra_bench.scenarios.win_conditions import WinCondition

    pack.levels["easy"].fail_condition = WinCondition(
        any_of=[
            {"after_ticks": 10_000},
            {"tool_violations_gte": 1},
        ]
    )
    c = compile_level(pack, "easy")
    assert c.forbidden_tools == ["attack_unit", "attack_move"]

    def violator(rs, Command):
        # observe + a single forbidden attack_unit on a non-existent
        # target id (the engine warns, but the bench tracks the tool
        # name BEFORE the engine evaluates it — so the violation
        # counts even when the engine refuses the order).
        return [Command.observe(), Command.attack_unit(["1"], target_id="99999")]

    res = run_level(c, violator, seed=1)
    # bench-side accounting must have flagged ≥1 forbidden call …
    assert res.signals.tool_violations >= 1, res.signals.tools_called
    # … and the fail predicate must have fired.
    assert res.outcome == "loss", res.outcome


def test_allowed_tools_dont_increment_violations():
    """The same harness with NO forbidden_tools and only allowed cmds
    must yield tool_violations == 0 (no false positives)."""
    pytest.importorskip("openra_train")

    pack = load_pack(PACKS / "custom-map-no-enemy.yaml")
    c = compile_level(pack, "easy")
    assert c.forbidden_tools == []

    def stall(rs, Command):
        return [Command.observe()]

    res = run_level(c, stall, seed=1)
    assert res.signals.tool_violations == 0
    # Tools_called should at least have observe (the bench's default
    # fallback and the policy's explicit call).
    assert "observe" in res.signals.tools_called
