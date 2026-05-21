"""combat-hold-chokepoint — hold a narrow corridor so a larger enemy
force can only feed a 3-abreast trickle through the choke.

Capability (action): positioning the squad IN a chokepoint lets a
small force defeat a larger one — the terrain caps how many attackers
can bring weapons to bear at once. Anchored in a 3-cell silo-walled
corridor, 4 medium tanks grind a 12-16 strong light-tank force down
piecemeal; the same squad fighting in the open is surrounded and
focus-fired down.

Bar (recalibrated 2026-05-20 after the OpenRA-Rust engine movement
fixes — moving units now fire and take fire en route, and attack_unit
on an out-of-sight target paths normally. Those fixes made a static
auto-firing squad strong enough that the old 12-tank / quota-8 easy
tier was beatable by a pure stall; the recalibration bumps easy to 14
attackers + quota 9 and adds a corridor-geometry win clause):

  • stall (only observe)        -> LOSS. The pre-placed squad
    auto-fires (stance:2) but SPREADS its fire across the 3-abreast
    front instead of concentrating it, kills too slowly (~7-8 < the
    quota of 9), and is worn down -> own_units_gte:2 busts AND the
    kill quota is unmet (killed 7-8, lost 3-4).
  • brute attack_move east      -> LOSS. The squad charges out of the
    corridor into the open; the whole force converges and focus-fires
    all 4 tanks -> loss cap busted before the kill quota (killed 0-1,
    lost 3-4), and the squad is no longer in the choke region.
  • wrong-path retreat WEST     -> LOSS. Pulling the squad west out of
    the corridor into the open: even when it scrapes the kill quota
    the survival cap busts and the units_in_region geometry clause
    fails (the squad is at x~20, not in the lane).
  • intended hold-the-choke     -> WIN. The squad stays anchored in
    the corridor and focus-fires the frontmost enemy; the 3-abreast
    funnel is ground down; the squad keeps >=3 tanks in the choke
    region and hits the kill quota (killed 9-10, lost 0-1).

Engine note: `silo` walls block PATHING but not weapons fire, so the
chokepoint works by CONFINING the attacker to the 3-row lane (capping
how many enemies are within weapon range at once), not by blocking
line of sight — this is why the corridor brackets the lane along its
whole length.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-hold-chokepoint.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(n_units, tick=1000, killed=0, fact=1, pos=(52, 20)):
    """Synthesize a WinContext for predicate-level checks. `pos` is the
    cell every own unit sits at — defaults to the choke (52,20) so the
    units_in_region win clause is satisfied; pass an out-of-corridor
    cell to exercise the geometry teeth."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=0,
        # own_buildings is a list of (type, cell_x, cell_y) tuples.
        own_buildings=[("fact", 10, 20)] * fact,
        own_building_types={"fact"} if fact else set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": pos[0], "cell_y": pos[1]}
            ] * n_units,
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # 3 tanks alive (in the choke region), 9 kills, fact alive, in
    # time -> WIN
    assert evaluate(c.win_condition, _ctx(3, tick=3000, killed=9))
    assert evaluate(c.win_condition, _ctx(4, tick=3000, killed=12))
    # only 2 tanks alive -> own_units_gte:3 fails
    assert not evaluate(c.win_condition, _ctx(2, tick=3000, killed=12))
    # kill quota unmet (8 < 9) -> win fails
    assert not evaluate(c.win_condition, _ctx(4, tick=3000, killed=8))
    # fact razed -> win fails
    assert not evaluate(c.win_condition, _ctx(4, tick=3000, killed=12, fact=0))
    # 1 tank left -> fail clause fires (not own_units_gte:2)
    assert evaluate(c.fail_condition, _ctx(1, tick=3000, killed=12))
    # fact razed -> not a fail clause here (fail is unit-count / deadline),
    # but win is unmet so the episode is a non-win.
    # past deadline -> real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(4, tick=5402, killed=12))
    assert 5401 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 5401 must be reachable within max_turns"
    )


def test_predicates_medium_and_hard():
    for lvl, quota in (("medium", 9), ("hard", 9)):
        c = compile_level(load_pack(PACK_PATH), lvl)
        # meeting the quota with >=3 tanks in the choke and fact alive
        # -> WIN
        assert evaluate(c.win_condition, _ctx(3, tick=3000, killed=quota))
        # one short of the quota -> win fails
        assert not evaluate(
            c.win_condition, _ctx(4, tick=3000, killed=quota - 1)
        )
        # 1 tank left -> fail fires
        assert evaluate(c.fail_condition, _ctx(1, tick=3000, killed=quota))
        # deadline reachable
        assert evaluate(c.fail_condition, _ctx(4, tick=5402, killed=quota))


def test_corridor_geometry_clause_bites():
    """The units_in_region win clause makes the hold load-bearing: a
    squad that met the kill quota and survival cap but abandoned the
    corridor (charged east or pulled west into the open) still fails
    the win — only a squad anchored in the choke satisfies it."""
    for lvl, quota in (("easy", 9), ("medium", 9), ("hard", 9)):
        c = compile_level(load_pack(PACK_PATH), lvl)
        # Anchored in the choke (52,20) -> WIN
        assert evaluate(
            c.win_condition, _ctx(4, tick=3000, killed=quota, pos=(52, 20))
        )
        # Pulled WEST into the open (x=20) -> geometry clause fails
        assert not evaluate(
            c.win_condition, _ctx(4, tick=3000, killed=quota, pos=(20, 20))
        )
        # Charged EAST out of the corridor (x=100) -> geometry fails
        assert not evaluate(
            c.win_condition, _ctx(4, tick=3000, killed=quota, pos=(100, 20))
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks 5401 fits inside max_turns on
    every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 5401 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 5401 not reachable within max_turns"
        )


def test_hard_has_two_spawn_groups():
    """Hard defines two ENEMY-side spawn_point groups (NORTH / SOUTH
    cluster) round-robined by seed — enemy-side activation per Wave-9
    per-owner spawn_point semantics."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    enemy_sps = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert len(enemy_sps) >= 2, (
        f"hard needs ≥2 enemy spawn_point groups, got {enemy_sps}"
    )


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "action"
    assert pack.meta.id == "combat-hold-chokepoint"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "chokepoint" in joined
    assert "thermopylae" in joined
    assert "ramp" in joined or "sc2" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Stall: only observe. The pre-placed squad auto-fires (stance:2)
    but spreads its fire — kills too slowly, is worn down, and the
    own_units_gte:2 fail clause busts."""
    return [Command.observe()]


def _brute_east_policy(rs, Command):
    """Brute: attack_move every tank east out of the corridor into the
    open. The whole enemy force converges and focus-fires the squad
    down before the kill quota is met."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move([str(u["id"])], target_x=120, target_y=20)
        for u in units
    ]


def _wrong_path_west_policy(rs, Command):
    """Wrong-path: pull the squad WEST out of the corridor into the
    open ground behind the choke. Even when the retreating squad
    auto-fires down enough pursuers to scrape the kill quota, it ends
    outside the choke region (geometry clause fails) and the survival
    cap busts -> LOSS on every level/seed."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=20, target_y=u["cell_y"])
        for u in units
    ]


def _intended_hold_policy(rs, Command):
    """Intended hold-the-choke: keep the squad anchored in the corridor
    and FOCUS-FIRE the frontmost (lowest cell_x) enemy with every tank.
    The bracketing walls confine the attackers to a 3-abreast trickle;
    concentrated fire grinds them down piecemeal."""
    units = rs.get("units_summary", []) or []
    enemies = [
        e
        for e in (rs.get("enemy_summary", []) or [])
        if not e.get("is_building")
        and (e.get("type") or "").lower() == "1tnk"
    ]
    if not units or not enemies:
        return [Command.observe()]
    front = min(enemies, key=lambda e: e["cell_x"])
    return [
        Command.attack_unit([str(u["id"])], str(front["id"]))
        for u in units
    ]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level/seed — the pre-placed squad
    auto-fires but spreads fire instead of concentrating it, so it
    kills too slowly (below the quota of 9) and is worn down before
    the quota is met -> own_units_gte:2 busts AND the kill quota is
    unmet (recalibrated 2026-05-20: 14-tank easy / quota 9)."""
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
def test_wrong_path_west_policy_loses(level):
    """Wrong-path retreat WEST out of the corridor must LOSE on every
    level/seed — the squad abandons the choke; the survival cap busts
    and/or the units_in_region geometry win clause fails (the squad
    sits in the open at x~20, not in the lane)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _wrong_path_west_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: wrong-path west must LOSE; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_east_policy_loses(level):
    """Brute attack_move east out of the corridor must LOSE — the squad
    abandons the choke and meets the full force in the open; it is
    surrounded and focus-fired down before any kills register."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _brute_east_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute east must LOSE; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_hold_wins(level):
    """The intended hold-the-choke policy (anchor in the corridor +
    focus-fire the frontmost enemy) WINS on every level/seed — the
    3-cell corridor caps the larger force to a 3-abreast trickle the
    squad grinds down, keeping >=3 tanks anchored in the choke
    region (recalibrated 2026-05-20: killed 9-10, lost 0-1)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    for s in (1, 2, 3, 4):
        res = run_level(c, _intended_hold_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended hold should WIN; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )
