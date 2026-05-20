"""proc-no-attack-passive-only — ROE-style recon-only compliance.

The bar (per CLAUDE.md): the intended (move-around-the-sentry-line)
policy WINS on every level and every hard seed (1–4); the brute
(attack_unit), stall (only observe), and wrong-path (drive straight
through the y=20 sentry lane) policies LOSE on every level and every
hard seed. Non-win is a real reachable timeout LOSS — never a draw.

Validation is scripted (no model / network). The four policies below
are exhaustive proxies for the failure modes the pack discriminates:
they exercise the predicate teeth (tool_violations_gte, units_lost_lte,
after_ticks, reach_region) directly through eval_core.run_level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "proc-no-attack-passive-only.yaml"


# ── unit-level predicate / schema checks (no engine) ──────────────────


def _ctx(
    units_xy=(),
    tick=1000,
    killed=0,
    lost=0,
    tool_violations=0,
):
    """Synthesize a WinContext for predicate-level checks. Mirrors the
    field set the live signals object exposes during eval_core."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        tool_violations=tool_violations,
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
    assert pack.meta.id == "proc-no-attack-passive-only"
    assert pack.meta.capability == "action"
    # benchmark_anchor must name the BFCL V4 + IFBench + ROE anchors
    # the pack is built against (suite-enforced by
    # test_benchmark_anchor_required, but spot-checked here too).
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "bfcl" in joined
    assert "ifbench" in joined or "negative instruction" in joined
    assert "roe" in joined or "rules-of-engagement" in joined or "rules of engagement" in joined
    assert "reconnaissance" in joined or "recon" in joined
    # meta required-prose fields populated
    assert "rules-of-engagement" in pack.meta.real_world_meaning.lower() \
        or "rules of engagement" in pack.meta.real_world_meaning.lower()
    assert "surveillance" in pack.meta.robotics_analogue.lower() \
        or "robot" in pack.meta.robotics_analogue.lower()
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        # Every level must carry the procedural-compliance allowlist:
        # attack_unit / attack_move are FORBIDDEN; observe / move / stop
        # are allowed (and tools: stays out of the forbidden set).
        assert "attack_unit" in c.forbidden_tools
        assert "attack_move" in c.forbidden_tools


def test_each_level_has_tool_violations_fail_clause():
    """The fail clause MUST carry `tool_violations_gte: 1` so a single
    forbidden call ends the episode as a LOSS — zero-tolerance ROE."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # The fail clause is `any_of[..., {tool_violations_gte:1}, ...]`.
        # Exercise the predicate end-to-end on a synthetic context.
        # Keep ≥1 unit alive so the `not own_units_gte:1` clause
        # doesn't fire as a confound; vary only the tool_violations
        # counter so the test isolates THIS predicate.
        units = [(8, 18), (8, 20), (8, 22), (9, 20)]
        ctx0 = _ctx(units_xy=units, tick=100, tool_violations=0)
        ctx1 = _ctx(units_xy=units, tick=100, tool_violations=1)
        assert not evaluate(c.fail_condition, ctx0), (
            f"{lvl}: clean turn must not fail"
        )
        assert evaluate(c.fail_condition, ctx1), (
            f"{lvl}: a single forbidden call must fail the episode"
        )


def test_predicates_per_level_attrition_cap_and_deadline():
    pack = load_pack(PACK_PATH)
    # The intended winning context: a jeep IS in the recon region,
    # the loss cap is honoured, the deadline isn't past.
    in_region = [(110, 20), (8, 18), (8, 20), (8, 22)]
    expected_caps = {"easy": 2, "medium": 1, "hard": 0}
    for lvl, cap in expected_caps.items():
        c = compile_level(pack, lvl)
        # Intended: zero losses, in region, tick well under deadline.
        assert evaluate(
            c.win_condition,
            _ctx(units_xy=in_region, tick=3000, lost=0, tool_violations=0),
        ), f"{lvl}: a clean recon arrival must satisfy the win"
        # Attrition cap busted → fail
        assert evaluate(
            c.fail_condition,
            _ctx(units_xy=in_region, tick=3000, lost=cap + 1, tool_violations=0),
        ), f"{lvl}: losing {cap + 1} (cap {cap}) must fail"
        # Deadline reachable LOSS, not a draw
        assert evaluate(
            c.fail_condition,
            _ctx(units_xy=in_region, tick=4502, lost=0, tool_violations=0),
        ), f"{lvl}: past after_ticks must fail (reachable inside max_turns)"
        # The fail `after_ticks: 4501` must fit inside the tick budget.
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 must be reachable within max_turns "
            f"(else timeout draws instead of losing); "
            f"max_turns={c.max_turns}"
        )


def test_intended_predicate_wins_only_when_actually_at_recon_point():
    """The win condition MUST require a jeep at the recon point —
    sitting at home base must not satisfy it. The reach_region radius
    (6) is generous enough to admit any reasonable arrival within the
    observation footprint, but not the spawn area."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Sitting at spawn — NOT a win, even with zero losses & in time.
        spawn_only = [(8, 18), (8, 20), (8, 22), (9, 20)]
        assert not evaluate(
            c.win_condition,
            _ctx(units_xy=spawn_only, tick=3000, lost=0),
        ), f"{lvl}: sitting at spawn must NOT win"
        # Adjacent to recon point (within radius 6) — wins.
        near_recon = [(108, 22)]
        assert evaluate(
            c.win_condition,
            _ctx(units_xy=near_recon, tick=3000, lost=0),
        ), f"{lvl}: arrival within recon radius must win"


def test_pack_declares_no_spawn_point_groups_under_not_applicable():
    """This pack is in tests/test_hard_tier.py::NOT_APPLICABLE with the
    documented reason "spawn variation would compete with the
    ROE-compliance signal". So hard must NOT carry seed-driven spawn
    groups (the compete-with-signal claim has to be literally true)."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert groups == {0}, (
        f"hard must use a single agent spawn group (NOT_APPLICABLE rationale); "
        f"got {sorted(groups)}"
    )


def test_pack_is_in_not_applicable_with_roe_reason():
    """Round-trip check that the hard-tier curation matrix records
    this pack as NOT_APPLICABLE with the ROE-compliance reason."""
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "proc-no-attack-passive-only" in NOT_APPLICABLE, (
        "pack must be classified in NOT_APPLICABLE with a stated reason"
    )
    assert "proc-no-attack-passive-only" not in UPGRADED, (
        "pack is NOT spawn-varied; should not appear in UPGRADED"
    )
    reason = NOT_APPLICABLE["proc-no-attack-passive-only"].lower()
    assert "roe" in reason or "compliance" in reason, (
        f"NOT_APPLICABLE rationale should reference the ROE-compliance "
        f"signal; got {reason!r}"
    )


# ── engine-driven scripted policies (4 policies × 4+1+1 seeds) ───────


def _intended_policy(rs, Command):
    """Detour-around-sentries policy. Routes the jeep column via the
    NORTH bypass (y=8): first leg WEST→(25,8), middle leg →(100,8),
    then drop SOUTH to the recon point (110,20). Move-only orders —
    no attack_unit / attack_move is ever issued."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if ux < 25 and uy > 12:
            tx, ty = 25, 8
        elif ux < 100:
            tx, ty = 100, 8
        else:
            tx, ty = 110, 20
        cmds.append(Command.move_units([str(u["id"])], target_x=tx, target_y=ty))
    return cmds


def _stall_policy(rs, Command):
    """Issue only observe(); never move. The recon point is never
    reached → within_ticks deadline fires → after_ticks LOSS."""
    return [Command.observe()]


def _wrong_path_policy(rs, Command):
    """Drive straight east along y=20 — through the sentry kill-zone.
    Move-only (no forbidden tool calls); jeeps are stance:0 so they
    don't auto-return-fire. The sentries (stance:3 e3 bracketing
    y=18..22) bleed the column past the attrition cap on every tier."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=110, target_y=20)
        )
    return cmds


def _brute_policy(rs, Command):
    """Issue a forbidden attack_unit on EVERY turn. The bench tracks
    the command NAME the agent issued, not whether the engine has a
    real target for it, so the violation counts even when the target
    id is a stub (see tests/test_forbidden_tools.py::violator).
    `tool_violations_gte: 1` in the fail clause then ends the episode
    as a LOSS on the very first turn."""
    return [Command.observe(), Command.attack_unit(["1"], target_id="99999")]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_policy_wins(level):
    """Intended detour wins on every level; on hard, across all 4 seeds."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended detour must WIN, "
            f"got {res.outcome} after {res.turns} turns "
            f"(lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick}, "
            f"viol={res.signals.tool_violations})"
        )
        # No spurious tool violations from the intended (move-only) policy.
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: intended policy issued forbidden tool "
            f"{res.signals.tool_violations} time(s); should be 0"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Pure-observe stall misses the recon point → after_ticks LOSS
    (the deadline must be reachable inside max_turns — verified in
    test_predicates_per_level_attrition_cap_and_deadline)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE on the deadline, "
            f"got {res.outcome} (tick={res.signals.game_tick})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_wrong_path_policy_loses(level):
    """Drive straight east on y=20 — jeep column eats sentry fire,
    attrition cap busts. LOSS on every level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    expected_cap = {"easy": 2, "medium": 1, "hard": 0}[level]
    for s in seeds:
        res = run_level(c, _wrong_path_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: wrong-path through sentry lane must LOSE "
            f"on attrition; got {res.outcome} "
            f"(lost={res.signals.units_lost}, cap={expected_cap}, "
            f"tick={res.signals.game_tick})"
        )
        # Attrition is the mechanism (or, on hard with cap=0, any
        # loss). No tool violations should be issued — wrong-path is a
        # move-only failure mode (the discrimination is purely loss-cap,
        # not procedural).
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: wrong-path policy must not violate; "
            f"violations were {res.signals.tool_violations}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_policy_loses_on_tool_violation(level):
    """Brute issues a forbidden attack_unit each turn → tool_violations
    accumulates → fail clause `tool_violations_gte: 1` fires → LOSS.
    Holds on every level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_policy, seed=s)
        assert res.signals.tool_violations >= 1, (
            f"{level} seed={s}: brute must trip ≥1 tool_violation; "
            f"got {res.signals.tool_violations} "
            f"(tools_called={res.signals.tools_called})"
        )
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute must LOSE via "
            f"tool_violations_gte:1; got {res.outcome}"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: `after_ticks: 4501` must fit inside max_turns
    on every level (~90 ticks/turn ⇒ tick ≤ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl
