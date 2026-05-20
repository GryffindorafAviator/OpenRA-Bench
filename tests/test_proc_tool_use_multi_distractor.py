"""proc-tool-use-multi-distractor — Wave-6 cluttered-API tool relevance.

The bar (per CLAUDE.md): the intended (move-only, ignore-distractors)
policy WINS on every level and every hard seed (1–4); stall (only
observe) and use-multiple-distractors (re-orders units every turn
with non-move tools / wastes turns on build / harvest / set_rally_
point) policies LOSE on every level and every hard seed. Non-win is
a real reachable timeout LOSS — never a draw.

The pack's discriminator is OUTCOME-graded (reach + zero loss + on
time), NOT tool-name graded — there are no `forbidden_tools` and
the bench's `tool_violations` counter stays at 0 for every policy.
The model is given a 9-tool cluttered API (move_units, attack_unit,
attack_move, build, place_building, harvest, set_stance, set_rally_
point, observe) and must self-select the relevant subset
(move_units + observe). Off-path enemies are stance:0 (HoldFire)
so they never pursue a "clean" column — the cost of invoking an
attack tool is the TURNS BURNED ordering it, not a forced kill-
zone. Same for build/harvest/etc.: they don't fail engine-side,
they just waste the model's turn budget while the within_ticks
clock keeps ticking.

Validation is scripted (no model / network) — these four policies
are exhaustive proxies for the cluttered-API failure / success
modes and exercise the predicate teeth (reach_region, units_lost_
lte, within_ticks/after_ticks) directly through eval_core.run_level.
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
PACK_PATH = PACKS / "proc-tool-use-multi-distractor.yaml"


# ── unit-level predicate / schema checks (no engine) ──────────────────


def _ctx(units_xy=(), tick=1000, lost=0, violations=0):
    """Synthesize a WinContext for predicate-level checks. Mirrors the
    field set the live signals object exposes during eval_core."""
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
    assert pack.meta.id == "proc-tool-use-multi-distractor"
    assert pack.meta.capability == "action"
    # benchmark_anchor must name the τ²-bench + BFCL V4 + IFBench +
    # junior-operator anchors from the Wave-6 spec.
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors
    joined = " ".join(anchors).lower()
    assert "τ²-bench" in joined or "tau" in joined or "distractor" in joined
    assert "bfcl" in joined
    assert "ifbench" in joined or "distraction" in joined
    assert "junior" in joined or "operator" in joined
    # meta required-prose fields populated and reflect the cluttered-
    # API framing.
    rwm = pack.meta.real_world_meaning.lower()
    assert "cluttered" in rwm or "distractor" in rwm or "irrelevant" in rwm
    assert "robot" in pack.meta.robotics_analogue.lower() \
        or "operator" in pack.meta.robotics_analogue.lower()
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def test_full_9_tool_api_surface_on_every_level():
    """Cluttered-API contract: every level exposes the full 9-tool
    spec'd API, NO `forbidden_tools` — the test is outcome-graded,
    not tool-name graded. The model must self-select the relevant
    subset (move_units + observe) from the 9 available tools."""
    pack = load_pack(PACK_PATH)
    expected = {
        "move_units", "attack_unit", "attack_move", "build",
        "place_building", "harvest", "set_stance", "set_rally_point",
        "observe",
    }
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        tools = set(c.scenario.tools or [])
        assert tools == expected, (
            f"{lvl}: tools allowlist must be the full 9-tool API spec; "
            f"got {sorted(tools)}, missing {sorted(expected - tools)}, "
            f"extra {sorted(tools - expected)}"
        )
        # No forbidden_tools — the discriminator is OUTCOME, not name.
        assert c.forbidden_tools == [], (
            f"{lvl}: this pack must NOT use forbidden_tools (the "
            f"discriminator is reach + zero-loss + within_ticks, NOT "
            f"a tool_violations counter); got {c.forbidden_tools}"
        )


def test_win_predicate_requires_reach_zero_loss_within_ticks():
    """The win predicate must enforce the advertised capability: a
    jeep reaches the observation point, zero units lost, before the
    tick deadline. Each clause is load-bearing."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        in_region = [(110, 20), (109, 19), (110, 21)]
        spawn_only = [(8, 5), (8, 20), (8, 35)]
        # Intended win state — clean arrival in time.
        assert evaluate(c.win_condition, _ctx(in_region, tick=900, lost=0)), \
            f"{lvl}: clean arrival within deadline must WIN"
        # Sitting at spawn — never wins regardless of tick / loss.
        assert not evaluate(c.win_condition, _ctx(spawn_only, tick=900, lost=0)), \
            f"{lvl}: sitting at spawn must NOT win"
        # ONE loss — wins predicate has units_lost_lte:0, so any
        # casualty invalidates the win clause.
        assert not evaluate(c.win_condition, _ctx(in_region, tick=900, lost=1)), \
            f"{lvl}: even a single casualty must invalidate the win"


def test_fail_predicate_reachable_loss_on_every_level():
    """Fail must be a real reachable LOSS, not a draw. The after_ticks
    fail clause must fit inside max_turns (~90 ticks/turn ⇒
    93 + 90·(max_turns-1)). Casualty (units_lost_lte:0 violated) and
    full force-wipe (not own_units_gte:1) likewise fail."""
    pack = load_pack(PACK_PATH)
    in_region = [(110, 20)]
    expected_deadline = {"easy": 2001, "medium": 1501, "hard": 1501}
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        deadline = expected_deadline[lvl]
        # after_ticks fits inside max_turns (no draw degeneracy)
        assert deadline <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks {deadline} must be reachable within "
            f"max_turns {c.max_turns} (else timeout draws instead of "
            f"losing). Max reachable tick "
            f"{93 + 90 * (c.max_turns - 1)}."
        )
        # Past deadline → fail
        assert evaluate(c.fail_condition, _ctx(in_region, tick=deadline + 1, lost=0)), \
            f"{lvl}: past after_ticks must fail"
        # Casualty → fail (units_lost_lte:0 is the cap on every tier)
        assert evaluate(c.fail_condition, _ctx(in_region, tick=900, lost=1)), \
            f"{lvl}: any unit loss must fail (cap is 0)"


def test_intended_predicate_wins_only_when_actually_at_observation_point():
    """Sanity: the reach_region(110,20,r=6) clause must require an
    actual arrival; the spawn area and the SW base must not satisfy
    it. Generous radius admits any reasonable approach footprint."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # SW base coords — not in the observation region.
        sw_base = [(14, 20), (18, 20)]
        assert not evaluate(c.win_condition, _ctx(sw_base, tick=900, lost=0)), \
            f"{lvl}: SW base must NOT count as arrival at (110,20)"
        # Adjacent (within radius 6) — wins.
        near = [(108, 22)]
        assert evaluate(c.win_condition, _ctx(near, tick=900, lost=0)), \
            f"{lvl}: arrival within radius 6 must win"


def test_hard_has_two_spawn_groups_under_upgraded():
    """The hard tier is in tests/test_hard_tier.py::UPGRADED, so it
    MUST declare ≥2 distinct agent spawn_point groups (north lane /
    south lane round-robined by seed)."""
    pack = load_pack(PACK_PATH)
    c = compile_level(pack, "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert groups == {0, 1}, (
        f"hard must define exactly two spawn_point groups {{0, 1}} "
        f"(north lane / south lane); got {sorted(groups)}"
    )


def test_pack_is_in_upgraded_set():
    """Round-trip check that the hard-tier curation matrix records
    this pack as UPGRADED (spawn-varied), not NOT_APPLICABLE."""
    from tests.test_hard_tier import NOT_APPLICABLE, UPGRADED

    assert "proc-tool-use-multi-distractor" in UPGRADED, (
        "pack must be registered in UPGRADED (≥2 spawn groups on hard)"
    )
    assert "proc-tool-use-multi-distractor" not in NOT_APPLICABLE, (
        "pack is spawn-varied; should not appear in NOT_APPLICABLE"
    )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Idle: only observe. Never moves; misses the within_ticks
    deadline ⇒ after_ticks LOSS."""
    return [Command.observe()]


def _intended_direct_policy(rs, Command):
    """Movement-only beeline east to the observation point. Issues
    `move_units` once per unit per turn — ignores every other tool
    in the 9-tool API. On easy/medium the spawn is on y=20 so a
    straight east drive arrives cleanly."""
    units = [
        u for u in (rs.get("units_summary") or [])
        if not u.get("is_building", False)
    ]
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=110, target_y=20)
        for u in units
    ]


def _intended_lane_policy(rs, Command):
    """Hard-tier version of the intended policy: stay on the spawn
    lane (y≈5 north or y≈35 south, picked from current unit y) to
    avoid the central sentry stack at (60,18..22) and the off-path
    north garrison at (45,8). Drops south to (110,20) only at the
    final approach (ux ≥ 100). Uses move_units only."""
    units = [
        u for u in (rs.get("units_summary") or [])
        if not u.get("is_building", False)
    ]
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if ux < 100:
            ly = 5 if uy < 20 else 35
            cmds.append(
                Command.move_units([str(u["id"])], target_x=105, target_y=ly)
            )
        else:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=110, target_y=20)
            )
    return cmds


def _use_multiple_distractors_policy(rs, Command):
    """The "wastes-turns-on-distractor-tools" failure mode: every
    turn issue MULTIPLE non-move tool calls (set_stance, attack_move
    on the units, build, harvest, set_rally_point) — each of which
    REPLACES the units' active move order so the column never makes
    monotonic east progress. The within_ticks deadline then bites.

    This is the τ²-bench / BFCL V4 cluttered-API failure mode: a
    model that "uses every tool the API exposes" because it sees a
    salient distractor for each, instead of self-selecting the
    relevant subset (move_units + observe)."""
    units = [
        u for u in (rs.get("units_summary") or [])
        if not u.get("is_building", False)
    ]
    if not units:
        return [Command.observe()]
    cmds = []
    # Set stance (re-issues an order to the units — replaces any
    # prior move).
    cmds.append(
        Command.set_stance([str(u["id"]) for u in units], 3)
    )
    # attack_move to the off-path north garrison cell — pulls the
    # column off-course.
    for u in units:
        cmds.append(
            Command.attack_move([str(u["id"])], target_x=45, target_y=8)
        )
    # Build something irrelevant (cash is on hand, pbox is buildable).
    cmds.append(Command.build("pbox"))
    # No-op-tool calls: harvest with no harvester, set_rally_point
    # with no producer.
    cmds.append(
        Command.harvest([str(units[0]["id"])], target_x=20, target_y=24)
    )
    cmds.append(
        Command.set_rally_point([str(units[0]["id"])], target_x=15, target_y=15)
    )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_policy_wins(level):
    """The intended movement-only policy WINS on every level. On
    easy/medium the column starts on y=20 and a direct beeline
    suffices; on hard the column starts on y=5 or y=35 (round-robin
    by seed) and the lane-respecting variant routes around the
    central sentry stack and the off-path garrison."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    pol = _intended_lane_policy if level == "hard" else _intended_direct_policy
    for s in seeds:
        res = run_level(c, pol, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended movement-only policy should "
            f"WIN, got {res.outcome} after {res.turns} turns "
            f"(lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # Sanity: only move_units is invoked (the intended subset).
        called = set(res.signals.tools_called.keys())
        assert called <= {"move_units"}, (
            f"{level} seed={s}: intended policy must call only "
            f"move_units; got {dict(res.signals.tools_called)}"
        )
        # No tool_violations (the pack has no forbidden_tools, so
        # this stays at 0 regardless — but check it as a contract).
        assert res.signals.tool_violations == 0


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses_on_deadline(level):
    """Stall (only observe) LOSES on every level/seed because the
    within_ticks deadline bites — the column never moves so the
    fail_condition's after_ticks clause fires."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE on the deadline, got "
            f"{res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick})"
        )
        # Stall causes the LOSS via the deadline, not via casualty.
        assert res.signals.units_lost == 0, (
            f"{level} seed={s}: stall must not lose any units "
            f"(no casualty trigger); got {res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_use_multiple_distractors_policy_loses(level):
    """The "use-multiple-distractors-wastes-turns" failure mode LOSES
    on every level/seed: the distractor tool calls each replace the
    units' active move order so the column never makes monotonic
    east progress, and the within_ticks deadline bites.

    This IS the τ²-bench / BFCL V4 cluttered-API failure mode (a
    model that fires every salient tool instead of picking only the
    relevant subset)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4)
    for s in seeds:
        res = run_level(c, _use_multiple_distractors_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: use-multiple-distractors must LOSE, "
            f"got {res.outcome} after {res.turns} turns "
            f"(lost={res.signals.units_lost}, "
            f"tick={res.signals.game_tick}, "
            f"tools_called={dict(res.signals.tools_called)})"
        )
        # The distractor policy actively uses multiple non-move tools
        # (the whole point of this failure mode). Sanity-check that
        # it really did call ≥3 distinct distractor tools — otherwise
        # the test isn't exercising the cluttered-API discriminator.
        called = set(res.signals.tools_called.keys())
        distractor_subset = called & {
            "set_stance", "attack_move", "build",
            "harvest", "set_rally_point",
        }
        assert len(distractor_subset) >= 3, (
            f"{level} seed={s}: distractor policy must invoke ≥3 "
            f"distinct distractor tools (the test of the cluttered-"
            f"API failure mode); got distractors {sorted(distractor_subset)}"
        )
