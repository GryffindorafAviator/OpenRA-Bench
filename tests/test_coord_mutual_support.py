"""coord-mutual-support — advance a 6-tank squad as a TIGHT BALL so
every unit stays inside mutual-support range; a strung-out advance
loses stragglers to focused fire (military mutual support / SC2 ball
micro / SMAC squad coherence).

Bar: the intended tight-ball advance (regroup before each contact so
the whole squad concentrates reciprocal fire) is the load-bearing
decision. The strict engine-driven LOSS bar holds for the lazy / brute
policies:

  • stall (only observe)              → LOSS (region bar never met; the
    harassers are stance:2 and never reach the idle squad, so the
    within_ticks deadline elapses → after_ticks LOSS).
  • brute strung-out column (each     → LOSS (the lead tanks enter each
    tank attack_move'd far east,         harasser cluster ALONE and are
    no regroup)                          focus-fired down with no
    supporting fire in range → ≥3 tanks lost → own_units_gte:5 busts /
    the en-route checkpoint latch never advances → LOSS).

The intended tight-ball policy WINS on every level and every hard
seed (1-4): the whole squad enters each cluster's fire envelope
together, concentrates its cannon fire, and erases the cluster before
it can finish a tank — ≥5 of 6 survive, K kills, checkpoint cleared.

Verified 2026-05-20:
  easy   seed1 : stall LOSS / brute LOSS lost=3 / tightball WIN k=5 l=0
  medium seed1 : stall LOSS / brute LOSS lost=3 / tightball WIN k=7 l=0
  hard   seed1-4: stall LOSS / brute LOSS lost≥3 / tightball WIN k=9 l=0
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "coord-mutual-support.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(units_xy=(), tick=1000, killed=0, lost=0, seq=None):
    """Synthesize a WinContext for predicate-level checks. `seq` lets a
    test pre-seed the `then:` latch progress so the objective clause can
    be exercised in isolation."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        seq_progress=dict(seq or {}),
        then_progress=dict(seq or {}),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # 5 tanks at the objective (within radius 6 of 60,20) + 1 elsewhere.
    at_obj = [(58, 19), (60, 20), (62, 21), (60, 18), (61, 20), (6, 17)]
    # 5 tanks at objective, never visited the checkpoint → the `then:`
    # latch must advance through clause 0 first (checkpoint) — but the
    # objective coords already satisfy clause 0? No: clause 0 region is
    # (40,20) r5 — none of these are within it, so the latch is stuck.
    # Evaluating with a FRESH (unseeded) ctx: then-latch idx 0, clause 0
    # not satisfied → `then` returns False even though objective is met.
    fresh = _ctx(at_obj, tick=2000, killed=5)
    assert not evaluate(c.win_condition, fresh), (
        "objective reached but checkpoint never latched → win must NOT fire"
    )
    # Checkpoint THEN objective in order: first satisfy the checkpoint
    # region (5 tanks near 40,20), then the objective. We exercise the
    # latch by evaluating twice on the same signals object.
    chk = [(39, 19), (40, 20), (41, 21), (40, 18), (41, 20), (6, 17)]
    ctx = _ctx(chk, tick=1500, killed=5)
    evaluate(c.win_condition, ctx)  # advances the latch past clause 0
    # Now move the squad to the objective on the SAME signals (latch
    # persists on signals.then_progress).
    ctx.render_state["units_summary"] = [
        {"cell_x": x, "cell_y": y} for x, y in at_obj
    ]
    assert evaluate(c.win_condition, ctx), (
        "checkpoint then objective, ≥5 kills, ≥5 alive, in time → WIN"
    )
    # Same route but only 4 tanks at the objective → region clause fails.
    ctx.render_state["units_summary"] = [
        {"cell_x": x, "cell_y": y}
        for x, y in [(58, 19), (60, 20), (62, 21), (60, 18)]
    ]
    assert not evaluate(c.win_condition, ctx)


def test_fail_condition_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # 2 tanks remaining → fail clause fires (not own_units_gte:3).
    two_left = [(60, 20), (61, 20)]
    assert evaluate(c.fail_condition, _ctx(two_left, tick=2000))
    # 3 tanks remaining → NOT yet a fail on the survivor clause.
    three_left = [(60, 20), (61, 20), (59, 20)]
    assert not evaluate(c.fail_condition, _ctx(three_left, tick=2000))
    # Past the deadline → real loss, reachable within max_turns.
    assert evaluate(c.fail_condition, _ctx(three_left, tick=4502))


def test_kill_bar_scales_by_level():
    """The kill bar tightens easy(5) → medium(7) → hard(9)."""
    bars = {"easy": 5, "medium": 7, "hard": 9}
    chk = [(39, 19), (40, 20), (41, 21), (40, 18), (41, 20), (6, 17)]
    obj = [(58, 19), (60, 20), (62, 21), (60, 18), (61, 20), (6, 17)]
    for lvl, k in bars.items():
        c = compile_level(load_pack(PACK_PATH), lvl)
        # checkpoint coords differ on hard (43,20) — use a wide block.
        chk_l = [(42, 19), (43, 20), (44, 21), (43, 18), (44, 20), (6, 17)] \
            if lvl == "hard" else chk
        ctx = _ctx(chk_l, tick=1500, killed=k)
        evaluate(c.win_condition, ctx)
        ctx.render_state["units_summary"] = [
            {"cell_x": x, "cell_y": y} for x, y in obj
        ]
        assert evaluate(c.win_condition, ctx), f"{lvl}: k={k} should WIN"
        # One kill short → predicate fails.
        ctx2 = _ctx(chk_l, tick=1500, killed=k - 1)
        evaluate(c.win_condition, ctx2)
        ctx2.render_state["units_summary"] = [
            {"cell_x": x, "cell_y": y} for x, y in obj
        ]
        assert not evaluate(c.win_condition, ctx2), (
            f"{lvl}: k={k - 1} (one short) must NOT win"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 not reachable within max_turns"
        )


def test_hard_has_two_spawn_groups():
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "action"
    assert pack.meta.id == "coord-mutual-support"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "mutual support" in joined
    assert "sc2" in joined or "ball" in joined or "smac" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


# ── engine-driven scripted policies ──────────────────────────────────


def _enemy_targets(rs):
    enemies = rs.get("enemy_summary", []) or []
    return [e for e in enemies
            if (e.get("type") or "").lower() in ("e3", "e1", "1tnk")
            and not e.get("is_building")]


def _stall_policy(rs, Command):
    """Stall: only observe. The region/checkpoint bars are never met
    (the squad never moves) → after_ticks LOSS."""
    return [Command.observe()]


def _brute_strung_policy(rs, Command):
    """Brute: each tank attack_move'd straight to the objective,
    independently — the squad strings out into a single-file column;
    the lead tanks enter each harasser cluster alone and are
    focus-fired down with no supporting fire in range."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move([str(u["id"])], target_x=60, target_y=20)
        for u in units
    ]


def _tight_ball_policy(rs, Command):
    """Intended tight-ball advance: hold the squad in a ~3-cell clump;
    on contact the WHOLE ball focus-fires the closest enemy; with no
    contact, advance the ball only while it is cohesive (regroup in
    place if a tank has strayed)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cx = sum(u["cell_x"] for u in units) / len(units)
    cy = sum(u["cell_y"] for u in units) / len(units)
    targs = _enemy_targets(rs)
    near = [e for e in targs
            if abs(e["cell_x"] - cx) + abs(e["cell_y"] - cy) <= 9]
    if near:
        focus = min(near, key=lambda e: (e["cell_x"] - cx) ** 2
                    + (e["cell_y"] - cy) ** 2)
        return [Command.attack_unit([str(u["id"])], str(focus["id"]))
                for u in units]
    spread = max(abs(u["cell_x"] - cx) + abs(u["cell_y"] - cy) for u in units)
    ax = int(round(cx)) if spread > 4 else min(60, int(round(cx)) + 6)
    slots = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1)]
    sorted_units = sorted(units, key=lambda u: (u["cell_y"], u["cell_x"]))
    return [
        Command.move_units([str(u["id"])], target_x=ax + dx, target_y=20 + dy)
        for u, (dx, dy) in zip(sorted_units, slots)
    ]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level — the squad never advances, the
    harassers are stance:2 and never reach it, so the within_ticks
    deadline elapses → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_strung_column_loses(level):
    """Brute strung-out column must LOSE — the lead tanks enter each
    harasser cluster alone and are focus-fired down; ≥3 tanks lost →
    own_units_gte:5 busts / the checkpoint latch never advances."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_strung_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute strung-out column must LOSE; "
            f"got {res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_tight_ball_wins(level):
    """Intended tight-ball advance WINS on easy and medium seed=1: the
    whole squad concentrates reciprocal fire, erases each cluster
    before it can finish a tank, ≥5 of 6 survive."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _tight_ball_policy, seed=1)
    assert res.outcome == "win", (
        f"{level} seed=1: tight ball should WIN, got {res.outcome} "
        f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
    )


def test_tight_ball_wins_hard_all_seeds():
    """Intended tight-ball advance WINS on hard for every seed (1-4):
    the harasser belt and objective cluster are symmetric across y=20
    so the doctrine generalises regardless of the seed-driven spawn
    latitude."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "hard")
    for s in (1, 2, 3, 4):
        res = run_level(c, _tight_ball_policy, seed=s)
        assert res.outcome == "win", (
            f"hard seed={s}: tight ball should WIN, got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )
