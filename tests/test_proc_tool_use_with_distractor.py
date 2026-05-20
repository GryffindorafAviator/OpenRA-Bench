"""proc-tool-use-with-distractor — IGNORE the irrelevant build palette.

Wave-6 procedural-compliance seed (τ²-bench distractor handling /
IFBench irrelevant-tool ignoring / BFCL V4 relevance / operator
discipline). The pack's whole point: the tool palette is BIG
(`[move_units, attack_unit, build, place_building, observe]`) but the
win path needs only `move_units`. `forbidden_tools` is INTENTIONALLY
empty — the test is whether the model can REASON about tool relevance,
not whether it can follow an explicit allowlist (those are the other
procedural packs: strict-toolban-fidelity-under-pressure,
proc-only-build-no-combat, proc-no-attack-passive-only).

The bar (per CLAUDE.md): the intended movement-only policy WINS on
every level and every hard seed (1..4); the stall (only observe) and
build-distractor (queue build orders every turn, no movement) policies
LOSE on every level and every hard seed. Non-win is a real reachable
timeout LOSS (after_ticks 4501 fits inside max_turns 55 ⇒ 4501 ≤
93 + 90·54 = 4953 — verified below). No draw degeneracy.

Validation is scripted (no model / network) — the three policies below
are exhaustive proxies for the three failure / success modes the pack
discriminates (intended / stall / build-distractor), per the Wave-6
spec for this cell.
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
PACK_PATH = PACKS / "proc-tool-use-with-distractor.yaml"


# ── unit-level predicate / metadata checks (no engine) ───────────────


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
    assert pack.meta.id == "proc-tool-use-with-distractor"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # The four anchors named in the Wave-6 spec for this cell.
    assert "τ²-bench" in joined or "tau" in joined or "distractor" in joined
    assert "ifbench" in joined or "irrelevant" in joined
    assert "bfcl" in joined
    assert "kitchen-sink" in joined or "kitchen sink" in joined or "operator" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None
        # The full tool palette is exposed (spec). The distractor tools
        # (build / place_building) AND attack_unit are visible.
        tools = set(c.scenario.tools or [])
        assert {"move_units", "attack_unit", "build", "place_building", "observe"} <= tools, (
            f"{lvl}: full tool palette must be exposed; got {sorted(tools)}"
        )
        # `forbidden_tools` MUST be empty on every level — the WHOLE
        # point is to test reasoning-based irrelevance, not allowlist-
        # enforcement. (The other procedural packs are the allowlist-
        # enforcement versions.)
        assert c.forbidden_tools == [], (
            f"{lvl}: forbidden_tools must be empty (the test is "
            f"reasoning about relevance, not allowlist enforcement); "
            f"got {c.forbidden_tools}"
        )


def test_predicates_win_only_when_at_egress_with_zero_losses_in_time():
    """The win predicate: reach_region(110,20,r=6) AND units_lost_lte:0
    AND within_ticks:4500. Verify each clause is load-bearing."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Intended winning context: a jeep at the egress, zero losses,
        # well under the deadline.
        at_egress = [(110, 20), (8, 20), (8, 21)]
        assert evaluate(
            c.win_condition, _ctx(at_egress, tick=2200, lost=0)
        ), f"{lvl}: a clean arrival must win"
        # Not at the egress → win not satisfied.
        at_start = [(8, 19), (8, 20), (8, 21)]
        assert not evaluate(
            c.win_condition, _ctx(at_start, tick=2200, lost=0)
        ), f"{lvl}: sitting at spawn must NOT win"
        # Any loss (cap is 0) → win not satisfied (even if at egress).
        assert not evaluate(
            c.win_condition, _ctx(at_egress, tick=2200, lost=1)
        ), f"{lvl}: zero-loss cap must be load-bearing on the win clause"
        # Past the deadline → win not satisfied (even if at egress + 0 lost).
        assert not evaluate(
            c.win_condition, _ctx(at_egress, tick=4501, lost=0)
        ), f"{lvl}: deadline must be load-bearing on the win clause"


def test_fail_predicates_real_loss_each_clause():
    """Each fail clause is a real reachable LOSS — no draw degeneracy."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        at_egress = [(110, 20), (8, 20), (8, 21)]
        # Past the deadline → fail.
        assert evaluate(
            c.fail_condition, _ctx(at_egress, tick=4502, lost=0)
        ), f"{lvl}: past after_ticks must fail"
        # Any loss (cap is 0) → fail.
        assert evaluate(
            c.fail_condition, _ctx(at_egress, tick=2200, lost=1)
        ), f"{lvl}: any unit lost must fail"
        # Force wipe → fail (no `own_units_gte:1`).
        assert evaluate(
            c.fail_condition, _ctx((), tick=2200, lost=3)
        ), f"{lvl}: force wipe must fail"
        # Clean turn (no fail predicate) — must NOT fail.
        assert not evaluate(
            c.fail_condition, _ctx(at_egress, tick=2200, lost=0)
        ), f"{lvl}: clean turn must not trigger fail"
        # The fail `after_ticks: 4501` must fit inside the tick budget.
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 must be reachable within "
            f"max_turns ({c.max_turns}); else timeout draws instead of losing"
        )


def test_no_forbidden_tools_clause_is_the_whole_point():
    """The pack INTENTIONALLY does not use `forbidden_tools` /
    `tool_violations_gte` — distinguishing it from B3
    (strict-toolban-fidelity-under-pressure), proc-only-build-no-combat,
    and proc-no-attack-passive-only, which all DO use them. This test
    asserts the cell's identity: it is reasoning-based relevance, not
    rule-following compliance."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert not c.forbidden_tools, (
            f"{lvl}: forbidden_tools must stay empty — the test is "
            f"REASONING about which tools are relevant, not whether "
            f"the model can follow an explicit allowlist"
        )
        # And the fail predicate must NOT carry tool_violations_gte
        # (which would only fire when forbidden_tools is non-empty,
        # but assert structurally to make the intent explicit).
        fc_repr = repr(c.fail_condition)
        assert "tool_violations_gte" not in fc_repr, (
            f"{lvl}: fail_condition must not depend on tool_violations "
            f"(the test is reasoning-based, not enforcement-based); "
            f"got {fc_repr}"
        )


def test_hard_has_two_spawn_groups_for_seed_variation():
    """The hard tier registers in test_hard_tier.py::UPGRADED, so it
    must define ≥2 distinct agent spawn_point groups so a memorised
    opening cannot generalise across seeds. This pack uses NORTH (y=8)
    vs SOUTH (y=32) staging."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert sp == {0, 1}, (
        f"hard must define exactly 2 agent spawn_point groups "
        f"(NORTH / SOUTH); got {sorted(sp)}"
    )


def test_pack_is_in_upgraded_with_distractor_rationale():
    """Round-trip check that the hard-tier curation matrix lists this
    pack in UPGRADED (it IS spawn-varied, per the Wave-6 spec for the
    hard tier)."""
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "proc-tool-use-with-distractor" in UPGRADED, (
        "pack must be classified in UPGRADED — hard tier is spawn-varied "
        "(N/S staging round-robin) per the Wave-6 spec"
    )
    assert "proc-tool-use-with-distractor" not in NOT_APPLICABLE, (
        "pack uses spawn variation on hard — it must not appear in "
        "NOT_APPLICABLE"
    )


# ── engine-driven scripted policies (3 policies × seeds 1..4) ────────
#
# Per Wave-6 spec: "test × seeds 1..4 × {stall, build-distractor
# (wastes turns building), intended-direct-move}".


def _intended_direct_move_policy(rs, Command):
    """Movement-only policy: drive every jeep east to the egress
    region. Uses only `move_units` (the one relevant tool); ignores
    the build palette and attack_unit entirely. WINS on every level
    and every hard seed."""
    units = rs.get("units_summary", []) or []
    jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
    if not jeeps:
        return [Command.observe()]
    cmds = []
    for u in jeeps:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=110, target_y=20)
        )
    return cmds


def _stall_policy(rs, Command):
    """Issue only observe(); never move. The deadline (after_ticks
    4501, reachable inside max_turns) fires → LOSS."""
    return [Command.observe()]


def _build_distractor_policy(rs, Command):
    """The exact failure mode the cell is built to catch: the model
    fixates on the visible-but-irrelevant build palette and spends
    every turn queuing structures instead of moving. The intended
    `move_units` calls never go out, the jeeps never advance, and the
    deadline fires → LOSS. (The build orders themselves are NOT
    forbidden — they don't trip any violation counter; the LOSS comes
    from running out the tick clock on a non-progressing policy.)"""
    cmds = [
        Command.build("powr"),
        Command.place_building("powr", 6, 18),
    ]
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_direct_move_wins(level):
    """The movement-only policy WINS on every level and every seed.
    Hard sweeps seeds 1..4 to validate spawn-variation determinism."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _intended_direct_move_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended movement-only policy must WIN, "
            f"got {res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick}, "
            f"lost={res.signals.units_lost}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # The intended policy uses move_units ONLY — no build /
        # place_building / attack_unit calls. (Sanity: the policy
        # itself only invokes move_units.)
        called = set(res.signals.tools_called.keys())
        assert "build" not in called, called
        assert "place_building" not in called, called
        assert "attack_unit" not in called, called
        # And no tool_violations are tracked (forbidden_tools is empty
        # by design — see test_no_forbidden_tools_clause_is_the_whole_point).
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: no forbidden_tools on this pack ⇒ "
            f"tool_violations must stay 0; got {res.signals.tool_violations}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses_on_deadline(level):
    """Stall (only observe) LOSES because the deadline bites
    (after_ticks 4501 is reachable within max_turns 55)."""
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


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_distractor_policy_loses_on_deadline(level):
    """The cell's headline failure mode: a policy that fixates on the
    visible-but-irrelevant build palette (and place_building) and never
    issues a movement order LOSES on the deadline. The build calls are
    NOT forbidden (no tool_violations counter is configured), so the
    LOSS is purely on the deadline — exactly mirroring τ²-bench's
    "irrelevant tool in palette" failure mode and BFCL V4's "calling a
    tool that isn't needed wastes turns" failure mode."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _build_distractor_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: build-distractor must LOSE on the "
            f"deadline, got {res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick}, "
            f"lost={res.signals.units_lost}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # The build calls themselves are NOT forbidden — the LOSS must
        # come from the deadline, not from a tool_violation. (This is
        # the cell's identity: reasoning-based irrelevance, not
        # rule-based enforcement.)
        assert res.signals.tool_violations == 0, (
            f"{level} seed={s}: build is not on forbidden_tools; "
            f"tool_violations must stay 0 even when build is spammed; "
            f"got {res.signals.tool_violations}"
        )
        # And the policy IS observably invoking the distractor (so the
        # test is exercising the right failure mode, not silently
        # passing on an empty-cmds short-circuit).
        called = res.signals.tools_called
        assert called.get("build", 0) >= 1, (
            f"{level} seed={s}: build-distractor policy must actually "
            f"invoke `build` (cell is meaningless otherwise); got {dict(called)}"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: `after_ticks: 4501` must fit inside max_turns
    on every level (~90 ticks/turn ⇒ tick ≤ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl
