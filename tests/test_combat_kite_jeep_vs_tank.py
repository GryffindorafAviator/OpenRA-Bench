"""combat-kite-jeep-vs-tank — kite a slow heavy unit with fast raiders.

Bar (recalibrated 2026-05-20 after the OpenRA-Rust engine movement
fixes — moving units now fire AND take fire en route, and attack_unit
on an out-of-sight target paths normally instead of teleporting).
Those fixes (a) made the old 1tnk-light-tank easy variant fully
degenerate — every policy, pure stall included, killed the weak light
tank for free — and (b) shifted the close-range trade so a static
attack_unit stand now trades exactly ONE raider for the kill. The
recalibration:

  • easy now uses the same 3tnk Soviet heavy as medium/hard, staged
    CLOSER (x≈70) so the kite-band is widest (the easy difficulty
    axis is band width, not a weaker enemy);
  • the survival bar is own_units_gte:3 on EVERY tier — kiting kills
    the heavy losing ZERO raiders, every non-kite policy loses ≥1, so
    "lose no raider" is the load-bearing teeth.

The four script-policy proxies, every level, seeds 1-4:

  • stall (observe only)        → LOSS — the hunt-bot heavy closes and
    grinds a raider down (killed 0, lost 1).
  • stand-and-shoot             → LOSS — attack_unit the heavy without
    disengaging; the cannon out-trades raider weapons at close range
    and one raider falls (killed 0-1, lost 1).
  • brute attack_move east      → LOSS — no disengage; the column
    meets the heavy point-blank and loses a raider (killed 0, lost
    1-2).
  • intended kite cycle         → WIN — when the heavy is within ~5
    cells move the raiders away along the lane, else attack_unit it;
    repeat. Kills the heavy keeping all three raiders (killed 1,
    lost 0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-kite-jeep-vs-tank.yaml"


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
    raiders3 = [(28, 9), (30, 10), (28, 11)]

    # Intended: 1 kill (the heavy), all 3 raiders alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(raiders3, tick=3000, killed=1, lost=0))
    # 1 loss (only 2 raiders) → win fails (the bar is own_units_gte:3)
    assert not evaluate(
        c.win_condition, _ctx(raiders3[:2], tick=3000, killed=1, lost=1)
    )
    # 0 kills → win fails
    assert not evaluate(c.win_condition, _ctx(raiders3, tick=3000, killed=0, lost=0))
    # 1 raider lost → fail clause fires (own_units_gte:3 busts)
    assert evaluate(c.fail_condition, _ctx(raiders3[:2], tick=3000, killed=1, lost=1))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(raiders3, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 4501 must be reachable within max_turns"
    )


def test_predicates_medium_force_preservation_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    raiders3 = [(28, 9), (30, 10), (28, 11)]
    raiders2 = raiders3[:2]

    # Intended: 1 kill, all 3 raiders alive → WIN
    assert evaluate(c.win_condition, _ctx(raiders3, tick=3000, killed=1, lost=0))
    # 1 raider lost → predicate fails (need ≥3)
    assert not evaluate(c.win_condition, _ctx(raiders2, tick=3000, killed=1, lost=1))
    # 0 kills → predicate fails
    assert not evaluate(c.win_condition, _ctx(raiders3, tick=3000, killed=0, lost=0))
    # 1 raider lost → fail clause fires
    assert evaluate(c.fail_condition, _ctx(raiders2, tick=3000, killed=1, lost=1))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(raiders3, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_tighter_deadline_and_survival_bar():
    c = compile_level(load_pack(PACK_PATH), "hard")
    raiders3 = [(28, 9), (30, 10), (28, 11)]

    # Intended: 1 kill, all 3 alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(raiders3, tick=3000, killed=1, lost=0))
    # 1 raider lost → predicate fails (need ≥3)
    assert not evaluate(
        c.win_condition, _ctx(raiders3[:2], tick=3000, killed=1, lost=1)
    )
    # Past the tighter deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(raiders3, tick=3602, killed=0, lost=0))
    assert 3601 <= 93 + 90 * (c.max_turns - 1), (
        "hard after_ticks 3601 must be reachable within tightened max_turns"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the raider start latitude."""
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
    assert pack.meta.id == "combat-kite-jeep-vs-tank"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the doctrines the brief calls out: SC2 kiting +
    # cavalry maneuver + military fire-and-maneuver.
    assert "kit" in joined  # kiting
    assert "cavalry" in joined or "fire-and-maneuver" in joined or "skirmish" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_force_preservation_bar_is_lose_zero_on_every_tier():
    """The recalibrated bar is own_units_gte:3 on every level — a
    policy that loses even ONE of the three raiders fails the win and
    trips the fail clause. This is what makes the kite cycle (lose
    zero) load-bearing against a stand-and-shoot (lose one)."""
    pack = load_pack(PACK_PATH)
    raiders3 = [(28, 9), (30, 10), (28, 11)]
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # all 3 alive + a kill → WIN
        assert evaluate(c.win_condition, _ctx(raiders3, tick=2000, killed=1))
        # exactly one raider lost → NOT a win, and fail clause fires
        assert not evaluate(
            c.win_condition, _ctx(raiders3[:2], tick=2000, killed=1, lost=1)
        )
        assert evaluate(
            c.fail_condition, _ctx(raiders3[:2], tick=2000, killed=1, lost=1)
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    bounds = {"easy": 4501, "medium": 4501, "hard": 3601}
    for lvl, bound in bounds.items():
        c = compile_level(pack, lvl)
        assert bound <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks {bound} not reachable within max_turns"
        )


# ── engine-driven scripted policies ──────────────────────────────────
#
# Intended kite policy: each turn, if the heavy tank is within
# KITE_TRIGGER cells of a raider, move that raider RETREAT_DIST cells
# AWAY from the tank along the lane; otherwise attack_unit the
# nearest tank. The cycle is purely reactive — no memory, derived
# each turn from geometry. KITE_TRIGGER=5 is the proximity threshold
# that generalises across all tiers and both hard spawn corridors
# (the recalibrated kite band).

KITE_TRIGGER = 5
RETREAT_DIST = 12


def _tanks(enemies):
    return [
        e for e in enemies
        if (e.get("type") or "").lower() in ("1tnk", "3tnk")
        and not e.get("is_building")
    ]


def _stall_policy(rs, Command):
    """Stall: only observe. The hunt-bot heavy closes on the idle
    raider stack and grinds a raider down → own_units_gte:3 busts."""
    return [Command.observe()]


def _stand_and_shoot_policy(rs, Command):
    """Stand at staging, attack_unit the heavy when visible — never
    disengage. The cannon out-trades raider weapons head-on; one
    raider falls and the own_units_gte:3 survival bar fails."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tanks = _tanks(enemies)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        if tanks:
            cmds.append(Command.attack_unit([str(u["id"])], str(tanks[0]["id"])))
        else:
            # No vision yet — march east to the engagement axis.
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(60, u["cell_x"] + 12),
                    target_y=u["cell_y"],
                )
            )
    return cmds


def _brute_attack_move_policy(rs, Command):
    """Brute: one attack_move order eastward. No disengage; the
    column meets the hunt-bot heavy and loses a raider in the same
    close-range trade as stand-and-shoot."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.attack_move([str(u["id"])], target_x=92, target_y=u["cell_y"])
        )
    return cmds


def _intended_kite_policy(rs, Command):
    """Intended kite cycle (the spec's required policy):
    each turn, if the heavy is within KITE_TRIGGER cells, move
    RETREAT_DIST cells AWAY along the lane; otherwise attack_unit
    the nearest tank.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tanks = _tanks(enemies)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if tanks:
            t0 = min(
                tanks,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            dx = t0["cell_x"] - ux
            dy = t0["cell_y"] - uy
            d = (dx * dx + dy * dy) ** 0.5
            if d <= KITE_TRIGGER:
                # Retreat: opposite of tank along x (the lane).
                sign = 1 if dx <= 0 else -1
                tx = max(2, min(93, ux + sign * RETREAT_DIST))
                cmds.append(
                    Command.move_units([str(u["id"])], target_x=tx, target_y=uy)
                )
            else:
                cmds.append(
                    Command.attack_unit([str(u["id"])], str(t0["id"]))
                )
        else:
            # No vision yet — march east to the engagement axis.
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(55, ux + 10), target_y=uy
                )
            )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level/seed — the hunt-bot heavy closes
    on the idle raider stack and grinds a raider down (own_units_gte:3
    busts)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_move_loses(level):
    """Brute attack_move east must LOSE — no disengage; the hunt-bot
    heavy closes to point-blank and out-trades the column, costing a
    raider before the survival bar is met."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _brute_attack_move_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute attack_move must LOSE; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stand_and_shoot_loses(level):
    """Stand-and-shoot must LOSE on every level/seed — the heavy tank's
    cannon out-trades raider weapons head-on. With the recalibrated
    own_units_gte:3 bar a stand trades exactly one raider for the kill
    and busts the survival cap (was an xfail on the pre-fix engine,
    now a strict LOSS)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _stand_and_shoot_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stand-and-shoot must LOSE; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_kite_wins(level):
    """Intended kite cycle (reactive move-away + attack_unit) — the
    spec's load-bearing decision: each turn, if the heavy is within
    KITE_TRIGGER (~5) cells, retreat along the lane; otherwise
    attack_unit the nearest tank. Verified WINNING on every level and
    every seed (1..4) — kills the heavy keeping all three raiders."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _intended_kite_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended kite should WIN, got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )
