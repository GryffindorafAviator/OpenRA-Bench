"""coord-relay-attack — rocket-then-tank relay strike.

The bar: intended Squad A (e3 rockets) engages FIRST → softens enemy
armour → THEN Squad B (2tnk) follows up. WINS on every level and
every hard seed (1-4). The wrong-policy lattice (stall, both-attack-
at-once, B-only) must LOSE on medium and hard. Easy is the bare-skill
tier (smaller enemy cluster, looser caps) so some wrong policies may
squeak by there per the SCENARIO_REVIEW_CHECKLIST inert-easy-teeth
convention.

The Wave-2 `then:` happened-before composite enforces the ordering
predicate: `units_killed_gte: K1` must latch before `units_killed_gte:
K2`. Wrong policies fail either on attrition (`units_lost_lte`) or
the clock (`after_ticks`).

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "coord-relay-attack.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "coord-relay-attack"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "smac", "overlapping fires", "bound-and-bound"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, kills=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        power_provided=0,
        power_drained=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        explored_percent=0.0,
        then_progress={},
        seq_progress={},
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n, type_="e3"):
    return [
        {"cell_x": 15, "cell_y": 20, "type": type_, "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: relay completes (need K1 then K2 — same ctx satisfies
    # both clauses in order on first eval), in time, under cap → WIN.
    ctx = _ctx(units=_alive(5), tick=2500, kills=4, lost=2)
    assert evaluate(c.win_condition, ctx)
    # K1 unmet → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=2500, kills=1, lost=0))
    # K2 unmet (3 kills, K2 is 4) → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=2500, kills=3, lost=0))
    # Attrition cap busted (5 > 4) → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(1), tick=2500, kills=4, lost=5))
    # Force-wipe → fail.
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2500, kills=4, lost=6))
    # Timeout with bar unmet → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(5), tick=3002, kills=2, lost=1))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 7 kills, 5 lost, in time → WIN.
    ctx = _ctx(units=_alive(3), tick=3500, kills=7, lost=5)
    assert evaluate(c.win_condition, ctx)
    # K1 unmet (only 2 kills) → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=3500, kills=2, lost=0))
    # K2 unmet (6 kills) → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=3500, kills=6, lost=0))
    # Attrition cap busted (6 > 5) → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(2), tick=3500, kills=7, lost=6))
    # Force-wipe → fail.
    assert evaluate(c.fail_condition, _ctx(units=[], tick=3500, kills=7, lost=7))
    # Timeout → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(6), tick=4002, kills=6, lost=2))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 10 kills, ≤5 lost, in time → WIN.
    ctx = _ctx(units=_alive(3), tick=4000, kills=10, lost=5)
    assert evaluate(c.win_condition, ctx)
    # K1 unmet (3 kills) → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=4000, kills=3, lost=0))
    # K2 unmet (9 kills) → not a win.
    assert not evaluate(c.win_condition, _ctx(units=_alive(6), tick=4000, kills=9, lost=0))
    # Attrition cap busted → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(2), tick=4000, kills=10, lost=6))
    # Force-wipe → fail.
    assert evaluate(c.fail_condition, _ctx(units=[], tick=4000, kills=10, lost=8))
    # Timeout → fail.
    assert evaluate(c.fail_condition, _ctx(units=_alive(6), tick=4502, kills=9, lost=2))


def test_then_clause_enforces_ordering_under_consistent_eval():
    """The then-clause latch is per-id and persists across evals via
    `signals.then_progress`. Using a fresh ctx (and so fresh
    then_progress) tests the leaf semantics; persistent-eval semantics
    are tested in tests/test_then_composite.py."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    # New ctx, but K2 already satisfied: greedy advance latches both
    # clauses in the same eval (matches waypoint_sequence semantics
    # validated in test_then_composite.py::test_then_late_a_then_b_…).
    ctx = _ctx(units=_alive(3), tick=3500, kills=7, lost=0)
    assert evaluate(c.win_condition, ctx)
    # But: if K1 is never met (and K2 also not met), no progress.
    ctx2 = _ctx(units=_alive(3), tick=3500, kills=2, lost=0)
    assert not evaluate(c.win_condition, ctx2)


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: the after_ticks fail trigger must be
    reachable within max_turns (engine advances ~90 ticks per turn,
    so max tick ≈ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    expectations = {"easy": 3001, "medium": 4001, "hard": 4501}
    for lvl, after in expectations.items():
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert after <= max_tick, (
            f"{lvl}: after_ticks {after} > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the west-edge staging latitude (north / south)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_squads_are_e3_then_2tnk_on_every_level():
    """The relay only has teeth if Squad A is rocket soldiers (e3,
    anti-vehicle) and Squad B is medium tanks (2tnk, anti-infantry):
    A's rockets soften the enemy 2tnk horde; B mops up the e1s."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        agent_types = [a.type for a in c.scenario.actors if a.owner == "agent"]
        assert "e3" in agent_types, f"{lvl}: Squad A must be e3 (rocket); got {agent_types}"
        assert "2tnk" in agent_types, f"{lvl}: Squad B must be 2tnk; got {agent_types}"
        enemy_types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "2tnk" in enemy_types, f"{lvl}: enemy must include 2tnk (the threat A softens)"
        assert "e1" in enemy_types, f"{lvl}: enemy must include e1 (what B mops up)"
        # Persistent far enemy marker (engine auto-done mitigation).
        assert "proc" in enemy_types, f"{lvl}: needs a persistent enemy survivor marker"


# ── engine-driven scripted policies ─────────────────────────────────


def _split_squads(rs):
    """Return (a_ids, b_ids): Squad A = e3 (rocket); Squad B = 2tnk."""
    a_ids, b_ids = [], []
    for u in (rs.get("units_summary") or []):
        t = str(u.get("type", "")).lower()
        if t == "e3":
            a_ids.append(str(u["id"]))
        elif t == "2tnk":
            b_ids.append(str(u["id"]))
    return a_ids, b_ids


def _enemy_centre(rs):
    en = rs.get("enemy_summary") or []
    if not en:
        return (60, 20)  # fallback to known cluster centre
    cx = sum(int(e["cell_x"]) for e in en) / len(en)
    cy = sum(int(e["cell_y"]) for e in en) / len(en)
    return (int(cx), int(cy))


def _stall(rs, Command):
    """Pure observe — enemies hold (stance:2), agent never engages →
    kill bar unmet → after_ticks LOSS."""
    return [Command.observe()]


def _both_attack_at_once(rs, Command):
    """Both squads attack_move the enemy cluster simultaneously.
    B's tanks reach the cluster around the same instant A's rockets do
    and get focus-fired by the enemy 4× 2tnk → units_lost cap busts on
    medium/hard."""
    a_ids, b_ids = _split_squads(rs)
    if not a_ids and not b_ids:
        return [Command.observe()]
    ex, ey = _enemy_centre(rs)
    cmds = []
    if a_ids:
        cmds.append(Command.attack_move(a_ids, ex, ey))
    if b_ids:
        cmds.append(Command.attack_move(b_ids, ex, ey))
    return cmds or [Command.observe()]


def _b_only(rs, Command):
    """Squad B (tanks) advances first; Squad A (rockets) holds. B
    faces the un-softened enemy 2tnk horde alone → bleeds → attrition
    cap busts before the kill bar is met."""
    a_ids, b_ids = _split_squads(rs)
    if not b_ids and not a_ids:
        return [Command.observe()]
    ex, ey = _enemy_centre(rs)
    cmds = []
    if b_ids:
        cmds.append(Command.attack_move(b_ids, ex, ey))
    if a_ids:
        cmds.append(Command.stop(a_ids))
    return cmds or [Command.observe()]


def _make_intended_a_then_b():
    """Factory: per-episode closure (turn counter + soften-state
    tracker). Relay doctrine:
      • Squad A (rockets) commits to the enemy cluster from turn 1
        and attempts to soften the heavy 2tnk line.
      • Squad B (tanks) HOLDS until either (a) the enemy heavy line
        is at least half-suppressed (≤50% of original 2tnk left),
        OR (b) enough wall-clock turns have elapsed that A has had
        the chance to do its work (fallback for the case where A
        gets melted before completing the soften — B still needs
        to advance and contribute kills).
    The two-trigger design models real fire-and-maneuver: the relay
    fires on suppression OR on a timed bound, not just on a perfect
    suppression latch. (Without the timed fallback, a wipe of A
    means B never advances and the kill bar is never met.)
    """
    state = {"initial_tanks": None, "turn": 0, "released": False}
    # Turn at which B is released regardless of suppression status.
    # ~16 turns ≈ 1440 ticks = enough for A to walk from x=15 to
    # x=60 (45 cells) and engage for a few seconds. Tuned so the
    # B-only / both-at-once paths don't accidentally land before
    # this trigger (they advance from turn 1, so they hit the
    # enemy faster than the gated release).
    RELEASE_TURN = 16

    def _policy(rs, Command):
        state["turn"] += 1
        a_ids, b_ids = _split_squads(rs)
        if not a_ids and not b_ids:
            return [Command.observe()]
        ex, ey = _enemy_centre(rs)
        enemy_tanks = [
            e for e in (rs.get("enemy_summary") or [])
            if str(e.get("type", "")).lower() == "2tnk"
        ]
        if state["initial_tanks"] is None and len(enemy_tanks) > 0:
            state["initial_tanks"] = len(enemy_tanks)
        cmds = []
        if a_ids:
            cmds.append(Command.attack_move(a_ids, ex, ey))
        if b_ids:
            if state["released"]:
                cmds.append(Command.attack_move(b_ids, ex, ey))
            else:
                init = state["initial_tanks"]
                suppressed = (
                    init is not None
                    and init > 0
                    and len(enemy_tanks) * 2 <= init
                )
                if suppressed or state["turn"] >= RELEASE_TURN:
                    state["released"] = True
                    cmds.append(Command.attack_move(b_ids, ex, ey))
                else:
                    cmds.append(Command.stop(b_ids))
        return cmds or [Command.observe()]

    return _policy


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_a_then_b_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _make_intended_a_then_b(), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended A-then-B relay should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no engagement → kill bar unmet), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_both_attack_at_once_loses(level, seed):
    """Sending both squads in simultaneously exposes B's tanks to the
    un-softened enemy 4× 2tnk before A's rockets can suppress them.
    Attrition cap busts on medium/hard. Easy is excluded (forgiving
    bare-skill tier with smaller cluster and cap=4)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _both_attack_at_once, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: both-at-once must LOSE (B tanks bleed "
        f"to enemy 2tnk before A softens), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_b_only_loses(level, seed):
    """Squad B (tanks) charging alone faces the full enemy 2tnk + e1
    cluster — outgunned by the enemy tanks, attrition cap busts before
    the K2 bar is met. Easy excluded (forgiving cluster)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _b_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: B-only must LOSE (B tanks alone vs "
        f"enemy 4× 2tnk + 4× e1), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )
