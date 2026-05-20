"""proc-strict-toolban-fidelity — clean BFCL-style allowlist fidelity.

Sibling of strict-toolban-fidelity-under-pressure (B3 *pressure*
variant: a patrol harasser tempts the agent to invoke the forbidden
attack tool). This pack is the QUIET baseline cell — no enemy
combatant, no harassing temptation; the test is purely whether the
agent restricts itself to the declared tool surface under no other
stressor than the briefing.

The bar:
  • intended (move_units the jeeps east — optionally with an
    occasional `build`) WINS on every level and every hard seed.
  • stall (only observe) LOSES on every level by the after_ticks
    deadline (reachable within max_turns on all 3 tiers — verified
    arithmetically below).
  • brute (attack_unit on ANY id, including a non-existent one)
    LOSES INSTANTLY on every level by tool_violations_gte:1 (the
    bench's tool_violations counter increments BEFORE the engine
    evaluates the order — see tests/test_forbidden_tools.py).
  • Each of the five "extra" forbidden tools (set_stance, harvest,
    repair, sell, set_rally_point) ALSO triggers tool_violations_gte:1
    on a single invocation — exercised below.

Validation is scripted (no model / no network) — sufficient and free,
per CLAUDE.md "How to validate".
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
PACK_PATH = PACKS / "proc-strict-toolban-fidelity.yaml"

ALLOWED = {"move_units", "build", "place_building", "observe", "stop"}
FORBIDDEN = [
    "attack_unit", "attack_move", "set_stance",
    "harvest", "repair", "sell", "set_rally_point",
]


# ── unit-level predicate / metadata checks ───────────────────────────


def _ctx(units_xy=(), buildings=(), tick=1000, violations=0):
    """Synthesize a WinContext for predicate-level checks."""
    own_buildings = list(buildings)
    own_building_types = {t for (t, _, _) in own_buildings}
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        tool_violations=violations,
        own_buildings=own_buildings,
        own_building_types=own_building_types,
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
    assert pack.meta.id == "proc-strict-toolban-fidelity"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # The three anchors from the spec.
    assert "bfcl" in joined and "v4" in joined and "relevance" in joined
    assert "ifbench" in joined and "instruction" in joined
    assert "soc" in joined and "runbook" in joined and "strict" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None
        # The whole point of the pack: every forbidden tool is on
        # the level's forbidden_tools list.
        assert set(c.forbidden_tools) == set(FORBIDDEN), lvl
        # Tools allowlist excludes the forbidden tools.
        tools = set(c.scenario.tools or [])
        for f in FORBIDDEN:
            assert f not in tools, (lvl, f)
        assert ALLOWED <= tools, (lvl, tools)


def test_predicates_win_satisfied_by_jeeps_in_region_and_fact_alive():
    """Win clause: ≥3 units in region (90,20,r=6) AND ≥1 agent fact
    AND within the deadline. All three clauses checked."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK_PATH), lvl)
        units_at_egress = [(90, 20), (91, 20), (90, 21)]
        with_fact = [("fact", 10, 20)]
        # Intended state → WIN
        assert evaluate(
            c.win_condition, _ctx(units_at_egress, with_fact, tick=1500)
        ), lvl
        # No fact (e.g. destroyed) → win not satisfied (and fail trips)
        assert not evaluate(
            c.win_condition, _ctx(units_at_egress, [], tick=1500)
        ), lvl
        assert evaluate(
            c.fail_condition, _ctx(units_at_egress, [], tick=1500)
        ), lvl
        # Past the deadline → fail
        deadline = (
            5400 if lvl == "easy" else 3600
        )
        assert evaluate(
            c.fail_condition,
            _ctx(units_at_egress, with_fact, tick=deadline + 1),
        ), lvl
        # ONE forbidden-tool call → instant fail (binding rule)
        assert evaluate(
            c.fail_condition,
            _ctx(units_at_egress, with_fact, tick=200, violations=1),
        ), lvl


def test_timeout_loss_reachable_on_every_level():
    """No draw degeneracy: after_ticks must fit inside max_turns
    on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        deadline = 5401 if lvl == "easy" else 3601
        reachable = 93 + 90 * (c.max_turns - 1)
        assert deadline <= reachable, (
            f"{lvl}: after_ticks {deadline} not reachable within "
            f"max_turns={c.max_turns} (reachable={reachable}) — "
            f"timeout would draw instead of losing"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard's distinguishing controlled variable vs medium: ≥2 agent
    spawn_point groups (NW staging / SW staging), round-robined by
    seed. test_hard_tier.py::UPGRADED contract."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must have ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_hard_fact_present_in_both_spawn_groups():
    """If ANY agent actor declares spawn_point, every agent actor
    WITHOUT spawn_point is filtered OUT (CLAUDE.md). So the agent
    `fact` must appear in BOTH spawn groups so the
    building_count_gte:fact:1 win clause is well-defined for every
    seed."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    fact_sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent" and str(a.type).lower() == "fact"
    }
    assert {0, 1} <= fact_sp, (
        f"agent fact must appear in BOTH spawn_point groups; "
        f"got {sorted(fact_sp)}"
    )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Idle: only observe. Misses the deadline ⇒ LOSS."""
    return [Command.observe()]


def _brute_attack_unit_policy(rs, Command):
    """Brute: invoke the FORBIDDEN attack_unit tool on every turn —
    even on a non-existent target id (tests/test_forbidden_tools.py
    confirms the counter increments BEFORE the engine evaluates the
    order). ⇒ INSTANT LOSS by tool_violations_gte:1."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    enemies = [
        e for e in (rs.get("enemy_summary", []) or [])
        if not e.get("is_building")
    ]
    target_id = str(enemies[0]["id"]) if enemies else "99999"
    return [Command.attack_unit([str(u["id"])], target_id) for u in units]


def _intended_move_policy(rs, Command):
    """Intended: move_units the jeeps east to (90,20). Uses only
    `move_units` (an allowed tool). The corridor is empty of enemy
    combatants so the jeeps cruise through unmolested."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=90, target_y=20)
        for u in units
    ]


# Forbidden-tool brute variants — one per extra forbidden tool. Each
# should INSTANT-fail by tool_violations_gte:1 on turn 1, exercising
# the full forbidden_tools list (not just attack_unit).


def _brute_set_stance_policy(rs, Command):
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [Command.set_stance([str(u["id"])], 2) for u in units]


def _brute_attack_move_policy(rs, Command):
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move([str(u["id"])], target_x=90, target_y=20)
        for u in units
    ]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_movement_only_policy_wins(level):
    """The intended jeep-transit policy WINS on every level and every
    seed (easy/medium use seeds 1..4 for sanity; hard exercises the
    two spawn_point groups round-robined across 1..4)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _intended_move_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended move_units policy should "
            f"WIN, got {res.outcome} after {res.turns} turns "
            f"(violations={res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # The disciplined policy never invokes a forbidden tool.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: intended policy unexpectedly tripped "
            f"{res.signals.tool_violations} tool violations: "
            f"{dict(res.signals.tools_called)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses_on_deadline(level):
    """Stall (only observe) LOSES because the deadline bites
    (after_ticks reachable within max_turns)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (deadline), got "
            f"{res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick})"
        )
        # The LOSS must come from the deadline, not tool_violations.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: stall should not invoke any tools "
            f"beyond observe; got {dict(res.signals.tools_called)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_unit_loses_by_tool_violation(level):
    """Brute (attack_unit, FORBIDDEN) LOSES immediately by
    tool_violations_gte:1 — the binding procedural rule."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _brute_attack_unit_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute (attack_unit) must LOSE by "
            f"tool_violations, got {res.outcome} after {res.turns} "
            f"turns (violations={res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute should accrue ≥1 tool violation; "
            f"got {res.signals.tool_violations}, "
            f"tools_called={dict(res.signals.tools_called)}"
        )
        # Instant-fail: each brute turn issues several attack_unit
        # calls, so the violation lands on turn 1.
        assert res.turns <= 3, (
            f"{level} seed={s}: brute should INSTANT-fail (≤3 turns); "
            f"got {res.turns} turns"
        )


@pytest.mark.parametrize(
    "policy_factory,tool_name",
    [
        (_brute_attack_move_policy, "attack_move"),
        (_brute_set_stance_policy, "set_stance"),
    ],
)
def test_extra_forbidden_tools_also_instant_fail_on_easy(
    policy_factory, tool_name
):
    """The non-attack_unit forbidden tools (set_stance, attack_move)
    ALSO trip tool_violations_gte:1 — verifies the WHOLE
    forbidden_tools list is wired, not just attack_unit."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    res = run_level(c, policy_factory, seed=1)
    assert res.outcome == "loss", (
        f"easy seed=1: brute ({tool_name}) must LOSE by "
        f"tool_violations, got {res.outcome} after {res.turns} turns "
        f"(violations={res.signals.tool_violations}, "
        f"tools_called={dict(res.signals.tools_called)})"
    )
    assert res.signals.tool_violations >= 1, (
        f"easy seed=1: brute ({tool_name}) should accrue ≥1 tool "
        f"violation; got {res.signals.tool_violations}, "
        f"tools_called={dict(res.signals.tools_called)}"
    )
