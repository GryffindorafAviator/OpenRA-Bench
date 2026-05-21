"""combat-harass-balanced-hit-and-run — BALANCED pulsed harass.

Bar: the intended hit-and-run cycle WINS on every level and every hard
seed; stall (only observe), retreat-only (never engage), brute
(attack-nearest, which switches onto the un-killable guard tank), and
commit-and-stay (charge deep into the guarded zone and hold) all LOSE
on every level. Non-win is a real reachable timeout LOSS (not a draw).

Recalibrated 2026-05-20 after the engine balance fixes (armour-class
weapon selection, stance semantics). The previous design used
harvesters as the score targets; post-fix the jeep's anti-infantry
M60mg cannot dent a harvester's heavy armour (and harvesters crush
jeeps), so the kill bar became unsatisfiable. The score targets are
now soft e1 rifle workers (the jeep MG one-shots them, stance:0 so
they are passive) and the leashed defender is a 3tnk heavy tank — the
genuine un-killable threat (jeep MG cannot scratch its heavy armour;
its cannon one-shots a jeep). The load-bearing decision: strike the
soft workers from MG range and disengage before the leashed tank
bears; never charge deep into the guarded zone, and never waste fire
on the un-killable tank.

Validation is scripted (no model / network): the policies below are
the exhaustive proxies for the real strategies and exercise the
predicate teeth directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-harass-balanced-hit-and-run.yaml"


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
    jeeps = [(6, 20), (6, 20)]

    # Intended: 2 kills, no losses, in time → WIN
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=0))
    # 1 loss still wins on easy (cap is 1)
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=1))
    # 2 losses → fail (cap busted)
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=3000, killed=2, lost=2))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 5401 must be reachable within max_turns"
    )


def test_predicates_medium_zero_loss_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    jeeps = [(6, 20), (6, 20)]

    # Intended: 3 kills, ZERO losses → WIN (BALANCED bar)
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=3, lost=0))
    # 3 kills but lost a jeep → fail (units_lost_lte:0)
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=3000, killed=3, lost=1))
    # 2 kills, 0 losses → predicate fails (need ≥3)
    assert not evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=0))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_three_kills_and_zero_loss():
    c = compile_level(load_pack(PACK_PATH), "hard")
    jeeps = [(6, 10), (6, 10)]

    # Intended: 3 kills, ZERO losses → WIN
    assert evaluate(c.win_condition, _ctx(jeeps, tick=4000, killed=3, lost=0))
    # 2 kills only → predicate fails (need ≥3)
    assert not evaluate(c.win_condition, _ctx(jeeps, tick=4000, killed=2, lost=0))
    # Any loss → fail
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=4000, killed=3, lost=1))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


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
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "combat-harass-balanced-hit-and-run"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the two doctrines the brief calls out: SC2 pulsed
    # harass and military guerrilla / hit-and-run.
    assert "muta" in joined or "harass" in joined or "sentry" in joined
    assert "guerrilla" in joined or "hit-and-run" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks 5401 fits inside max_turns on
    every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 5401 <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ──────────────────────────────────
#
# Constants from CLAUDE.md / openra-sim/src/scripted_bot.rs:
#   GUARD_AGGRO = 16 cells (post-radius for lunge)
#   GUARD_LEASH = 18 cells (post-radius for snap-back)
# The intended hit-run-cycle retreats the instant the leashed tank is
# within RETREAT_TRIGGER cells — the jeep MG out-ranges nothing on the
# 3tnk and the cannon one-shots a jeep, so a single decision spent in
# the tank's envelope is fatal.
RETREAT_TRIGGER = 16


def _workers(enemies):
    """Soft e1 score targets (jeep MG one-shots them)."""
    return [e for e in enemies if (e.get("type") or "").lower() == "e1"]


def _guards(enemies):
    """The leashed defender is a 3tnk heavy tank (un-killable by the
    jeep's anti-infantry MG; one-shots a jeep with its cannon)."""
    return [
        e for e in enemies
        if (e.get("type") or "").lower() == "3tnk"
        and not e.get("is_building")
    ]


def _hit_run_cycle_policy(rs, Command):
    """Pulsed hit-and-run — the intended BALANCED strategy.

    Per-jeep, memoryless, derived each turn from geometry:
      - Track the cluster latitude (the y of the nearest visible
        worker) so the jeep advances and retreats along the worker
        lane rather than drifting off it.
      - If the leashed tank is within RETREAT_TRIGGER cells → RETREAT
        west ~26 cells along the lane (cross the leash; the snap-back
        triggers as the tank loses the jeep past LEASH=18).
      - Else if a worker is visible → STRIKE the nearest one
        (attack_unit; the jeep MG one-shots a soft e1 worker).
      - Else (fog from the far-west start) → ADVANCE east along the
        lane to acquire vision of the cluster.

    The cycle is self-sustaining: strike → tank lunges into range →
    retreat past the leash → tank snaps back → re-advance → strike
    the next worker. The jeep never wastes fire on the un-killable
    tank and never lingers in its cannon envelope.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    if not units:
        return [Command.observe()]
    workers = _workers(enemies)
    guards = _guards(enemies)
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        # Lane = latitude of the nearest visible worker (else hold y).
        if workers:
            nearest_w = min(
                workers,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            lane = nearest_w["cell_y"]
        else:
            nearest_w = None
            lane = uy
        # Nearest leashed tank — the thing we must not linger near.
        gd2 = 10 ** 9
        if guards:
            gd2 = min(
                (g["cell_x"] - ux) ** 2 + (g["cell_y"] - uy) ** 2
                for g in guards
            )
        # Inside the tank's reach → RETREAT west along the lane.
        if gd2 <= RETREAT_TRIGGER ** 2:
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=max(2, ux - 26), target_y=lane
                )
            )
            continue
        # Safe: strike the nearest soft worker.
        if nearest_w is not None:
            cmds.append(
                Command.attack_unit([str(u["id"])], str(nearest_w["id"]))
            )
            continue
        # No worker in sight (fog): advance east along the lane.
        cmds.append(
            Command.move_units(
                [str(u["id"])], target_x=min(64, ux + 10), target_y=lane
            )
        )
    return cmds


def _stall_policy(rs, Command):
    """Stall: only observe. Kill bar never met → fail on the clock."""
    return [Command.observe()]


def _commit_and_stay_policy(rs, Command):
    """Charge deep into the guarded cluster and HOLD — attack the
    DEEPEST worker (max x, nearest the tank post) and never retreat.
    The 3tnk's cannon one-shots a jeep, so committing inside its
    envelope until the workers fall loses both jeeps before the kill
    bar is met → LOSS. (When fog hides the cluster from the western
    start, advance east to acquire vision, then commit.)"""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    workers = _workers(enemies)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if workers:
            # Push onto the deepest worker — into the tank's envelope.
            target = max(workers, key=lambda e: e["cell_x"])
            cmds.append(Command.attack_unit([str(u["id"])], str(target["id"])))
        else:
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(90, ux + 10), target_y=uy
                )
            )
    return cmds


def _brute_policy(rs, Command):
    """Brute: attack the NEAREST enemy unit, no retreat. When the
    leashed tank lunges it becomes the nearest enemy and the brute
    switches onto it — wasting fire on the un-killable 3tnk while its
    cannon one-shots the jeep → LOSS."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    targets = [
        e for e in enemies
        if (e.get("type") or "").lower() in ("e1", "3tnk")
        and not e.get("is_building")
    ]
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if targets:
            t = min(
                targets,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(t["id"])))
        else:
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(115, ux + 10), target_y=uy
                )
            )
    return cmds


def _retreat_only_policy(rs, Command):
    """Sit at the west staging line; never approach. Kill bar unmet →
    fail on the clock."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        # Park at the far-west wall (well outside any guard's AGGRO).
        cmds.append(
            Command.move_units([str(u["id"])], target_x=4, target_y=u["cell_y"])
        )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_hit_run_cycle_wins(level, seed):
    """Intended hit-and-run WINS on every level and every hard seed:
    strike the soft workers from MG range, retreat past the leash when
    the tank closes, re-engage — kill bar met with ZERO jeeps lost
    (medium/hard) / ≤1 lost (easy)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _hit_run_cycle_policy, seed=seed)
    assert res.outcome == "win", (
        f"{level} seed={seed}: intended hit-run-cycle should WIN, got "
        f"{res.outcome} after {res.turns} turns (killed="
        f"{res.signals.units_killed}, lost={res.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_policy_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall_policy, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE (kill bar unmet, clock), "
        f"got {res.outcome} after {res.turns} turns; "
        f"killed={res.signals.units_killed}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_commit_and_stay_loses(level, seed):
    """Charge deep into the guarded zone and hold → the 3tnk cannon
    one-shots a jeep before the kill bar is met → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _commit_and_stay_policy, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: commit-and-stay must LOSE (jeep dies to "
        f"the 3tnk cannon), got {res.outcome} after {res.turns} turns; "
        f"killed={res.signals.units_killed}, lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_attack_nearest_loses(level, seed):
    """Attack-nearest (which switches onto the lunging un-killable
    3tnk) → a jeep is lost to the cannon → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _brute_policy, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: brute attack-nearest must LOSE (jeep lost "
        f"to the un-killable 3tnk), got {res.outcome} after {res.turns} "
        f"turns; killed={res.signals.units_killed}, "
        f"lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_retreat_only_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _retreat_only_policy, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: retreat-only must LOSE (kill bar unmet), "
        f"got {res.outcome}; killed={res.signals.units_killed}"
    )
