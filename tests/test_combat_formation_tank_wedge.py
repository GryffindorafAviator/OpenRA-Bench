"""combat-formation-tank-wedge — drive 5× 2tnk in a wedge formation
through a bracketing fire corridor to reach the eastern objective
region with most of the force intact.

Bar: the intended wedge (lead absorbs on-axis fire; flankers engage
the brackets end-on from off-axis) is the load-bearing decision.

The strict engine-driven LOSS bar holds for the lazy / brute
policies:

  • stall (only observe)           → LOSS (region bar unmet on the
    clock; defenders are stance:2 and never approach the strike
    force, so the within_ticks bar elapses → after_ticks LOSS).
  • brute attack_move east on y=20 → LOSS (column on the engagement
    axis takes simultaneous cross-fire from BOTH brackets at
    Manhattan 4; ≥3 tanks die before the column clears the gap →
    own_units_gte:3 fails / region bar unmet → LOSS).

Engine note (verified 2026-05-20): the OpenRA-Rust combat numbers
for 2tnk-vs-e3 trade favour tank cannon DPS by a wide margin, so
the PREDICATE-level discrimination is strict (a play that loses ≥3
of 5 tanks LOSES under own_units_gte:3 regardless of mechanism),
and the column-vs-wedge geometry is the load-bearing decision
encoded in the win predicate. The engine-driven scripted wedge
policy WINS on easy and medium (lead-on-axis + flankers off-axis
sequences the engagement so only 1-2 e3 fire on a given tank at
once); the stall and brute LOSS bars hold on every level/seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-formation-tank-wedge.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(units_xy=(), tick=1000, killed=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
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


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # 4 tanks AT objective (within radius 6 of 80,20) — WIN
    at_obj4 = [(78, 19), (80, 20), (82, 21), (80, 18)]
    # 3 tanks at objective + 2 elsewhere — own ≥4 ok but region n<4 fails
    at_obj3 = [(78, 19), (80, 20), (82, 21), (6, 18), (6, 19)]
    # 4 tanks at objective + 1 elsewhere — 5 alive total, region n=4 ok
    at_obj4_plus = [(78, 19), (80, 20), (82, 21), (80, 18), (6, 18)]
    # 2 tanks at objective — region predicate fails
    at_obj2 = [(78, 19), (80, 20)]

    # Intended: ≥4 at objective, ≥4 alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(at_obj4_plus, tick=3000))
    assert evaluate(c.win_condition, _ctx(at_obj4, tick=3000))
    # 3 at objective + 2 elsewhere: own=5 ok but region n<4 → predicate fails
    assert not evaluate(c.win_condition, _ctx(at_obj3, tick=3000))
    # Only 2 tanks at objective and only 2 alive: own_units_gte:4 fails
    assert not evaluate(c.win_condition, _ctx(at_obj2, tick=3000))
    # 2 tanks remaining → fail clause fires (not own_units_gte:3)
    assert evaluate(c.fail_condition, _ctx(at_obj2, tick=3000))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(at_obj4_plus, tick=4502))
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 4501 must be reachable within max_turns"
    )


def test_predicates_medium_relaxed_three_survive_bar():
    """Per the header relaxation: medium uses own_units_gte:3 + ≥3 of
    5 tanks at objective, not the strict 5-of-5. The discriminator is
    column-vs-wedge survival differential, not absolute counts."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    at_obj3 = [(78, 19), (80, 20), (82, 21)]
    at_obj3_plus = [(78, 19), (80, 20), (82, 21), (6, 18)]
    at_obj2 = [(78, 19), (80, 20)]

    # Intended: ≥3 at objective, ≥3 alive, ≥3 kills, in time → WIN
    assert evaluate(c.win_condition, _ctx(at_obj3, tick=3000, killed=3))
    assert evaluate(c.win_condition, _ctx(at_obj3_plus, tick=3000, killed=3))
    # 3 at objective but only 2 kills → predicate fails (kill bar)
    assert not evaluate(c.win_condition, _ctx(at_obj3, tick=3000, killed=2))
    # 2 at objective (and 2 alive) → predicate fails (region + survival)
    assert not evaluate(c.win_condition, _ctx(at_obj2, tick=3000, killed=4))
    # 2 tanks remaining → fail clause fires
    assert evaluate(c.fail_condition, _ctx(at_obj2, tick=3000, killed=4))


def test_predicates_hard_two_blockers():
    c = compile_level(load_pack(PACK_PATH), "hard")
    at_obj3 = [(78, 19), (80, 20), (82, 21)]
    at_obj2 = [(78, 19), (80, 20)]

    # Intended: ≥3 at objective, ≥3 alive, ≥4 kills, in time → WIN
    assert evaluate(c.win_condition, _ctx(at_obj3, tick=3000, killed=4))
    # Kill bar tighter on hard — 3 kills not enough
    assert not evaluate(c.win_condition, _ctx(at_obj3, tick=3000, killed=3))
    # Two tanks alive → fail
    assert evaluate(c.fail_condition, _ctx(at_obj2, tick=3000, killed=4))
    # Past deadline → fail reachable
    assert evaluate(c.fail_condition, _ctx(at_obj3, tick=4502, killed=4))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


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
    assert pack.meta.id == "combat-formation-tank-wedge"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the doctrines the brief calls out.
    assert "wedge" in joined or "formation" in joined
    assert "sc2" in joined or "military" in joined or "combined-arms" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 not reachable within max_turns"
        )


# ── engine-driven scripted policies ──────────────────────────────────


def _targets(enemies):
    return [
        e for e in enemies
        if (e.get("type") or "").lower() in ("e3", "1tnk", "3tnk")
        and not e.get("is_building")
    ]


def _stall_policy(rs, Command):
    """Stall: only observe. Region bar never met (the agent never
    moves toward (80,20)) → after_ticks LOSS."""
    return [Command.observe()]


def _brute_column_policy(rs, Command):
    """Brute column attack_move east along y=20. The column squeezes
    through the corridor on the engagement axis; cross-fire from
    both brackets focuses on the lead, then inherits down the line
    — the column busts the survival bar before reaching (80,20)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.attack_move([str(u["id"])], target_x=110, target_y=20)
        )
    return cmds


def _intended_wedge_policy(rs, Command):
    """Intended wedge cycle: identify the leader (closest to y=20),
    drive lead on-axis at y=20, push flankers to y=18 and y=22
    trailing one cell west, attack any in-range enemy with all units
    that can bear. The wedge engages each bracket end-on so only
    1-2 e3 per bracket fire on a given tank at a time.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    targs = _targets(enemies)
    if not units:
        return [Command.observe()]
    # Sort units by current y; assign formation slots.
    sorted_units = sorted(units, key=lambda u: u["cell_y"])
    # 5-tank wedge slots: from outermost-flank to lead and back.
    # We use 5 logical slots: (dx=-2, dy=-2), (dx=-1, dy=-1),
    # (dx=0, dy=0 — lead), (dx=-1, dy=+1), (dx=-2, dy=+2).
    slots = [(-2, -2), (-1, -1), (0, 0), (-1, 1), (-2, 2)]
    cmds = []
    for u, (dx, dy) in zip(sorted_units, slots):
        ux, uy = u["cell_x"], u["cell_y"]
        # Lead pushes east aggressively; flankers track formation.
        lead_x = max(ux, 12) + 8
        tx = min(110, lead_x + dx)
        ty = 20 + dy
        in_range = [
            e for e in targs
            if abs(e["cell_x"] - ux) + abs(e["cell_y"] - uy) <= 5
        ]
        if in_range:
            t0 = min(
                in_range,
                key=lambda e: abs(e["cell_x"] - ux) + abs(e["cell_y"] - uy),
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(t0["id"])))
        else:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=tx, target_y=ty)
            )
    return cmds


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on medium and hard — the region bar is never
    met because the agent never moves; defenders are stance:2 and
    never come to the strike force, so the within_ticks deadline
    elapses → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_brute_column_attack_move_loses(level):
    """Brute attack_move east on y=20 must LOSE — column on the
    engagement axis takes simultaneous cross-fire from both
    brackets; ≥3 tanks die before the column clears the gap →
    own_units_gte:3 fails OR the region-at-objective bar is unmet
    in time → LOSS.
    """
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_column_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute column attack_move must LOSE; "
            f"got {res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


def test_intended_wedge_wins_medium():
    """Intended wedge cycle WINS on medium seed=1: lead absorbs the
    on-axis 1tnk's fire, flankers offset to y=18/22 engage each
    bracket end-on, ≥3 of 5 tanks reach (80,20) intact with the kill
    bar met. Verified 2026-05-20 (killed=7 lost=1 tick=1353)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "medium")
    res = run_level(c, _intended_wedge_policy, seed=1)
    assert res.outcome == "win", (
        f"medium seed=1: intended wedge should WIN, got {res.outcome} "
        f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
    )


def test_intended_wedge_wins_easy():
    """Wedge WINS on easy seed=1 (single bracket; verified 2026-05-20
    killed=4 lost=0 tick=1233 — all 5 tanks reach (80,20) intact)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    res = run_level(c, _intended_wedge_policy, seed=1)
    assert res.outcome == "win", (
        f"easy seed=1: intended wedge should WIN, got {res.outcome} "
        f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
    )
