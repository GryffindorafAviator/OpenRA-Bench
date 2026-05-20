"""strict-toolban-fidelity-under-pressure — B3 procedural compliance.

The bar: the intended movement-only policy WINS on every level and
every hard seed; stall (only observe) and brute (attack_unit — a
FORBIDDEN tool) LOSE on every level. Non-win is a real reachable
timeout LOSS (the after_ticks 4501 fail clause fits inside
max_turns 55 ⇒ 4501 ≤ 93 + 90·54 = 4953 — verified below).

The pack's procedural-compliance signal is binary: did the policy
invoke a forbidden tool? — so the BINDING discriminator is brute
(attack_unit) tripping `tool_violations_gte: 1` and INSTANT-failing.
A wrong-path policy that walks straight east through the patrol with
`move_units` only is a LEGITIMATE win path (the engine doesn't
reliably kill a fast-moving column with the harasser, and that is
acceptable — the procedural rule is the test, not detour necessity;
this mirrors BFCL V4 where refusing the forbidden tool IS the
success criterion even if other valid tools also reach the goal).
We therefore test (1) intended WINS, (2) stall LOSES (deadline),
(3) brute LOSES (tool_violations) — and document wrong-path's
acceptable-WIN behaviour as a sanity check on the procedural
predicate.

Validation is scripted (no model / network) — these policies are
exhaustive proxies for the procedural failure / success modes and
exercise the predicate teeth directly. See CLAUDE.md "How to
validate" for the harness; tests/test_forbidden_tools.py covers the
underlying primitive's unit-level behaviour.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "strict-toolban-fidelity-under-pressure.yaml"


# ── unit-level predicate / metadata checks ───────────────────────────


def _ctx(units_xy=(), tick=1000, lost=0, violations=0):
    """Synthesize a WinContext for predicate-level checks."""
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        tool_violations=violations,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "strict-toolban-fidelity-under-pressure"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # The four anchors from the B3 spec.
    assert "bfcl" in joined
    assert "τ²-bench" in joined or "tau" in joined or "distractor" in joined
    assert "ifbench" in joined or "instruction-following" in joined
    assert "soc" in joined or "runbook" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None
        # The whole point of the pack: forbidden_tools is set on every
        # level (the bench's strict-toolban primitive).
        assert c.forbidden_tools == ["attack_unit", "attack_move"], lvl
        # Tools allowlist excludes the forbidden tools — defensive
        # double-bind so a model honouring `tools:` ALSO doesn't invoke
        # the forbidden ones.
        tools = set(c.scenario.tools or [])
        assert "attack_unit" not in tools, lvl
        assert "attack_move" not in tools, lvl
        assert {"move_units", "stop", "observe"} <= tools, lvl


def test_predicates_easy_win_and_fail_modes():
    """Easy: intended state satisfies win; stall/brute/attrition each
    trip a distinct fail clause."""
    c = compile_level(load_pack(PACK_PATH), "easy")
    units_at_egress = [(110, 20), (109, 19), (110, 21)]
    units_at_start = [(8, 19), (8, 20), (8, 21)]

    # Intended state: reached, 0 lost, 0 violations, in time → WIN
    assert evaluate(
        c.win_condition, _ctx(units_at_egress, tick=2200, lost=0, violations=0)
    )
    # Not reached → win not satisfied
    assert not evaluate(
        c.win_condition, _ctx(units_at_start, tick=2200, lost=0, violations=0)
    )
    # Past deadline → fail (and deadline is reachable within max_turns)
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=4502, lost=0, violations=0)
    )
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 4501 must be reachable within max_turns — else "
        "timeout draws instead of losing"
    )
    # ONE forbidden tool call → instant fail (the procedural-compliance
    # binding rule)
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=200, lost=0, violations=1)
    )
    # Attrition cap busted (>1 lost on easy) → fail
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=2200, lost=2, violations=0)
    )


def test_predicates_medium_same_caps_as_easy():
    """Medium: same attrition cap as easy (≤1) but a bigger patrol
    (2-stack) as the controlled variable. The predicate teeth are
    identical to easy."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    units_at_egress = [(110, 20), (109, 19), (110, 21)]
    assert evaluate(
        c.win_condition, _ctx(units_at_egress, tick=2200, lost=1, violations=0)
    )
    # ONE forbidden tool call → instant fail
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=200, lost=0, violations=1)
    )
    # Deadline reachable
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_strict_zero_loss():
    """Hard tightens attrition to ZERO (units_lost_lte: 0) and adds a
    SECOND patrol vector. Even one lost agent fails."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    units_at_egress = [(110, 20), (109, 19), (110, 21)]
    # Intended: reached, 0 lost, 0 violations → WIN
    assert evaluate(
        c.win_condition, _ctx(units_at_egress, tick=2200, lost=0, violations=0)
    )
    # ONE loss (cap is 0) → fail
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=2200, lost=1, violations=0)
    )
    # ONE forbidden tool call → instant fail
    assert evaluate(
        c.fail_condition, _ctx(units_at_egress, tick=200, lost=0, violations=1)
    )
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_timeout_loss_reachable_on_every_level():
    """No draw degeneracy: after_ticks 4501 must fit inside max_turns
    on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl


def test_hard_has_two_patrol_vectors():
    """Hard's distinguishing controlled variable vs medium: a SECOND
    patrol vector (the northern arc) — doubled salient distraction.
    Verified at the actor level (enemy non-fact actors form ≥2 clusters
    on distinct y-bands)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    enemy_non_marker = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and str(a.type).lower() != "fact"
    ]
    ys = {a.position[1] for a in enemy_non_marker}
    assert len(ys) >= 2, (
        f"hard must have ≥2 distinct patrol y-anchors; got {sorted(ys)}"
    )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Idle: only observe. Misses the deadline ⇒ LOSS."""
    return [Command.observe()]


def _brute_policy(rs, Command):
    """Brute: invoke the FORBIDDEN attack_unit tool on every turn.
    The bench's tool_violations counter increments BEFORE the engine
    evaluates the order (tests/test_forbidden_tools.py), so even an
    attack on a non-existent target id counts ⇒ INSTANT LOSS by
    tool_violations_gte: 1."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    enemies = [
        e for e in (rs.get("enemy_summary", []) or [])
        if not e.get("is_building")
    ]
    target_id = str(enemies[0]["id"]) if enemies else "99999"
    cmds = []
    for u in units:
        cmds.append(Command.attack_unit([str(u["id"])], target_id))
    return cmds


def _wrong_path_policy(rs, Command):
    """Walk straight east through the patrol with `move_units` only
    (an allowed tool). The procedural-compliance rule is satisfied
    (no forbidden tool calls), and the patrol's chip-fire does not
    reliably kill 3 fast-moving agent infantry, so this WINS — which
    is FINE: the test of this pack is the procedural rule, not detour
    necessity (see module docstring)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=110, target_y=20)
        )
    return cmds


def _intended_detour_policy(rs, Command):
    """Movement-only detour around the patrol via the southern lane
    (y≈30 — safe across all hard seeds; on easy/medium the central
    patrol alone leaves both lanes free). Three waypoints: (40,30),
    (90,30), (110,20). Uses only allowed tools (move_units)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux = u["cell_x"]
        if ux < 40:
            tx, ty = 40, 30
        elif ux < 90:
            tx, ty = 90, 30
        else:
            tx, ty = 110, 20
        cmds.append(
            Command.move_units([str(u["id"])], target_x=tx, target_y=ty)
        )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_movement_only_policy_wins(level):
    """The intended movement-only detour WINS on every level. On hard
    we cover seeds 1..4 (the spec's seed sweep; this pack has no
    spawn_point round-robin — see test_hard_tier::NOT_APPLICABLE —
    but the seed sweep still validates determinism)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _intended_detour_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended movement-only detour should "
            f"WIN, got {res.outcome} after {res.turns} turns "
            f"(lost={res.signals.units_lost}, "
            f"violations={res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # Sanity: the disciplined policy never invokes a forbidden tool.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: intended policy unexpectedly tripped "
            f"{res.signals.tool_violations} tool violations "
            f"(should only call move_units): "
            f"{dict(res.signals.tools_called)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses_on_deadline(level):
    """Stall (only observe) LOSES because the deadline bites
    (after_ticks 4501 is reachable within max_turns)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (deadline), got "
            f"{res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick})"
        )
        # Stall never invokes a forbidden tool — the LOSS must come
        # from the deadline, not from tool_violations.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: stall should not invoke any tools "
            f"beyond observe; got {dict(res.signals.tools_called)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_unit_loses_by_tool_violation(level):
    """Brute (attack_unit, FORBIDDEN) LOSES immediately by
    tool_violations_gte: 1 — the procedural-compliance binding rule.
    The exact INSTANT-fail is the load-bearing signal of this pack."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _brute_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute (attack_unit) must LOSE by "
            f"tool_violations, got {res.outcome} after {res.turns} "
            f"turns (violations={res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # The LOSS must be attributed to forbidden tool calls (the
        # binding rule), not to attrition or the deadline.
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute should accrue ≥1 tool violation "
            f"(attack_unit is forbidden); got "
            f"{res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)}"
        )
        # And the violation should fire on turn 1 (instant fail) —
        # not many turns later — because each brute turn issues
        # several attack_unit calls.
        assert res.turns <= 3, (
            f"{level} seed={s}: brute should INSTANT-fail (≤3 turns); "
            f"got {res.turns} turns"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_wrong_path_straight_east_does_not_trip_forbidden_tools(level):
    """Sanity: wrong-path (walk straight east through the patrol with
    `move_units` only) is a LEGITIMATE play under the procedural rule
    — only allowed tools are invoked, so tool_violations stays 0 even
    when this is the "lazy" policy. Whether it wins or loses on
    attrition/deadline is downstream of the rule; the binding signal
    (no forbidden tool calls) is satisfied. This documents the
    procedural-rule semantics: the test is BINARY (did you invoke a
    forbidden tool?), not "did you pick the optimal route?"."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _wrong_path_policy, seed=s)
        # The procedural rule is satisfied (no forbidden invocation).
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: wrong-path uses move_units only — "
            f"tool_violations must stay 0; got "
            f"{res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)}"
        )
        # And the only tool actually invoked beyond the harness'
        # implicit observe is move_units (not attack_unit / attack_move).
        called = set(res.signals.tools_called.keys())
        assert "attack_unit" not in called, called
        assert "attack_move" not in called, called
