"""combat-heli-flank — air-mobility lateral-flank no-cheat bar.

F1 cheat-WIN audit fix (2026-05-25): a brute `attack_move` straight
east on the y=20 centre lane used to WIN the kill bar — the "flank"
verb was cosmetic, since the wall of pillboxes was impassable only to
GROUND units and the helis flew straight over on the centre line.

Fix: an `agun` (anti-aircraft) is placed in the airspace at (45,20).
Any heli that crosses the y≈20 centre lane near x=45 is shot down
before reaching the cluster. A real lateral flank (move north past
y≈10 or south past y≈30 first, then east past x≈55, then engage
from outside the AA's disc) clears the kill bar; the brute
straight-east attack_move dies to the AA.

The bar pinned here:
  • stall (only observe) LOSES on every level + every hard seed
  • brute east attack_move LOSES on every level + every hard seed
  • intended split-flank WINS on every level + every hard seed
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-heli-flank.yaml"
LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── structural checks ────────────────────────────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-heli-flank"
    assert pack.meta.capability == "action"
    rwm = pack.meta.real_world_meaning.lower()
    assert "flank" in rwm or "lateral" in rwm or "agun" in rwm or "anti-aircraft" in rwm
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_central_aa_tower_present_on_every_level():
    """The agun on the centre airspace lane is the load-bearing
    obstacle that turns this from a straight strike into a real
    lateral flank. If it's missing, the brute straight-east attack
    cheat is back."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        aguns = [
            a for a in c.scenario.actors
            if a.type == "agun" and a.owner == "enemy"
        ]
        assert aguns, f"{lvl}: no enemy agun present"
        # At least one agun must sit near the centre lane (y≈20) to
        # actually block the y=19..21 heli straight-line approach.
        on_centre = [a for a in aguns if 18 <= a.position[1] <= 22]
        assert on_centre, (
            f"{lvl}: no centre-lane agun (y∈[18..22]); got "
            f"{[a.position for a in aguns]}"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks fits inside max_turns
    (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        deadline = None
        for clause in fc.get("any_of", []) or []:
            if "after_ticks" in clause:
                deadline = int(clause["after_ticks"])
        assert deadline is not None, f"{lvl}: no after_ticks fail clause"
        reachable = 93 + 90 * (c.max_turns - 1)
        assert deadline < reachable, (
            f"{lvl}: deadline {deadline} unreachable within "
            f"{c.max_turns} turns (max tick {reachable})"
        )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    return [Command.observe()]


def _brute_east_attack_move(rs, Command):
    """Brute: attack_move all helis straight east to the cluster
    centroid (60,20). Crosses the centre AA lane at x=45 and the
    helis die before reaching the cluster ⇒ LOSS."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "heli"
    ]
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    return [Command.attack_move(ids, target_x=60, target_y=20)]


def _intended_flank_policy(rs, Command):
    """Intended: split-flank one heli north (y=4 band), one south
    (y=36 band), then move east past x=55, then engage the cluster.
    The flank bands keep the helis outside the centre AA's effective
    range; on arrival the auto-engage scan picks up the e1 cluster
    and gun them down."""
    units = [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "heli"
    ]
    enemies = rs.get("enemy_summary", []) or []
    targets = [
        e for e in enemies
        if (e.get("type") or "").lower() == "e1"
        and not e.get("is_building")
    ]
    if not units:
        return [Command.observe()]
    units = sorted(units, key=lambda u: int(u["id"]))
    cmds = []
    for i, u in enumerate(units):
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        flank_y = 4 if i % 2 == 0 else 36
        # Phase 1: get to the flank band (still west of AA).
        if abs(uy - flank_y) > 3 and ux < 50:
            cmds.append(Command.move_units([uid], target_x=ux, target_y=flank_y))
        # Phase 2: move east past the AA disc.
        elif ux < 55:
            cmds.append(Command.move_units([uid], target_x=55, target_y=flank_y))
        # Phase 3: engage the cluster.
        else:
            if targets:
                t = min(
                    targets,
                    key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
                )
                cmds.append(Command.attack_unit([uid], str(t["id"])))
            else:
                cmds.append(Command.move_units([uid], target_x=60, target_y=20))
    return cmds


@pytest.mark.parametrize("level", LEVELS)
def test_stall_policy_loses_every_level_and_seed(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (no kills, deadline bites); "
            f"got {res.outcome} killed={res.signals.units_killed}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_brute_east_attack_move_loses(level):
    """The cheat that the audit caught: straight east attack_move
    used to WIN. With the centre AA tower added it MUST LOSE."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_east_attack_move, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute east attack_move must LOSE — the "
            f"centre AA is supposed to shoot the helis down on the "
            f"y=20 lane; got {res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_intended_flank_policy_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = HARD_SEEDS if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_flank_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended split-flank must WIN; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost} turns={res.turns}"
        )
