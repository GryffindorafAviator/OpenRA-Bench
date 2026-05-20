"""adv-asymmetric-weaker-must-win — underdog kite-and-pick.

REASONING capability: 2× medium tanks must defeat a stronger garrison
(4× e1 rifle infantry + 1× heavy 3tnk on medium/hard) by refusing
decisive head-on engagement and committing to a flank approach that
stays OUTSIDE the heavy's `guard`-bot aggro envelope (≈16 cells from
its spawn cell at the east face of the cluster).

Four-policy bar:
  • stall (observe only)                       → LOSS (kill bar unmet,
    clock LOSES — enemy `guard` bot holds post).
  • head-on charge (attack_move at y=20 lane)  → LOSS on medium/hard
    (the e1 wall fixes the column, then the 3tnk lunges and out-
    DPSs the medium pair).
  • brute attack_move deep east                → LOSS on medium/hard
    (any path that closes into the 3tnk aggro envelope dies before
    the kill bar is met).
  • intended off-axis kite-and-pick            → WIN on medium and
    every hard seed (1..4): stage off-axis, attack_unit the nearest
    e1 from a flank cell that stays outside heavy aggro, retreat
    west if the heavy lunges inside ~14 cells.

On easy there is NO heavy (2× e1 only) and head-on works — the easy
tier is the bare commit-and-kill skill with no asymmetry pressure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "adv-asymmetric-weaker-must-win.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(units_xy=(), tick=1000, killed=0, lost=0, buildings=("fact",)):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types={b.lower() for b in buildings},
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
    tanks = [(8, 20), (8, 22)]

    # Intended: 2 e1 kills, ≥1 tank alive, fact retained → WIN
    assert evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=2, lost=0))
    # 1 tank lost still wins (own_units_gte:1)
    assert evaluate(c.win_condition, _ctx(tanks[:1], tick=3000, killed=2, lost=1))
    # Lost the fact → fail (has_building:fact missing)
    assert not evaluate(
        c.win_condition, _ctx(tanks, tick=3000, killed=2, lost=0, buildings=())
    )
    assert evaluate(
        c.fail_condition, _ctx(tanks, tick=3000, killed=2, lost=0, buildings=())
    )
    # All tanks lost → fail
    assert evaluate(c.fail_condition, _ctx([], tick=3000, killed=2, lost=2))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(tanks, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 5401 must be reachable within max_turns"
    )


def test_predicates_medium_four_kill_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    tanks = [(8, 10), (8, 12)]

    # Intended: 4 kills (all e1), ≥1 tank alive, fact retained → WIN
    assert evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=4, lost=0))
    # 1 tank lost still satisfies own_units_gte:1
    assert evaluate(c.win_condition, _ctx(tanks[:1], tick=3000, killed=4, lost=1))
    # 3 kills (e.g. killed 3 of 4 e1) → not enough
    assert not evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=3, lost=0))
    # 0 kills → not enough (head-on charge case)
    assert not evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=0, lost=0))
    # Both tanks lost → fail clause fires
    assert evaluate(c.fail_condition, _ctx([], tick=3000, killed=0, lost=2))
    # Fact lost → fail
    assert evaluate(
        c.fail_condition, _ctx(tanks, tick=3000, killed=4, lost=0, buildings=())
    )
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(tanks, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_tighter_deadline():
    c = compile_level(load_pack(PACK_PATH), "hard")
    tanks = [(8, 11), (8, 13)]

    # Intended: 4 kills, ≥1 alive, in time, fact retained → WIN
    assert evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=4, lost=0))
    # 3 kills → predicate fails
    assert not evaluate(c.win_condition, _ctx(tanks, tick=3000, killed=3, lost=0))
    # Past the tighter deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(tanks, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "hard after_ticks 4501 must be reachable within tightened max_turns"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the staging corridor latitude."""
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
    assert pack.meta.id == "adv-asymmetric-weaker-must-win"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Required real-world anchors per spec.
    assert "sc2 asymmetric" in joined
    assert "asymmetric warfare" in joined
    assert "guerrilla" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    bounds = {"easy": 5401, "medium": 5401, "hard": 4501}
    for lvl, bound in bounds.items():
        c = compile_level(pack, lvl)
        assert bound <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks {bound} not reachable within max_turns"
        )


def test_garrison_composition_per_tier():
    """Spec contract: easy = 2× e1 (no heavy); medium = 4× e1 + 1× 3tnk;
    hard = same garrison as medium (only the agent spawn varies)."""
    pack = load_pack(PACK_PATH)
    for lvl, expect_e1, expect_3tnk in (
        ("easy", 2, 0),
        ("medium", 4, 1),
        ("hard", 4, 1),
    ):
        c = compile_level(pack, lvl)
        e1s = [a for a in c.scenario.actors if a.owner == "enemy" and a.type == "e1"]
        heavies = [a for a in c.scenario.actors if a.owner == "enemy" and a.type == "3tnk"]
        assert len(e1s) == expect_e1, f"{lvl}: expected {expect_e1} e1, got {len(e1s)}"
        assert len(heavies) == expect_3tnk, (
            f"{lvl}: expected {expect_3tnk} 3tnk, got {len(heavies)}"
        )


# ── engine-driven scripted policies ──────────────────────────────────
#
# Off-axis kite-and-pick policy (the spec's load-bearing decision):
#  - Stage off-axis; the natural flank-y corresponds to the agent's
#    starting corridor (north < y=20 ⇒ flank_y=14; otherwise 26).
#  - If the 3tnk is visible AND closer than HEAVY_DANGER cells,
#    retreat WEST along the flank corridor (avoid the heavy's lethal
#    close-range trade).
#  - Otherwise attack_unit the nearest e1 (auto-finishes infantry
#    one at a time; rifle weapons barely scratch medium-tank armour).
#  - When no enemies are visible yet, move to the canonical flank
#    cell at (74, flank_y).

HEAVY_DANGER = 14
RETREAT_DIST = 10


def _stall_policy(rs, Command):
    """Stall: only observe. Kill bar never met → fail on the clock
    (the `guard` enemies hold post, so it is a pure clock loss)."""
    return [Command.observe()]


def _headon_policy(rs, Command):
    """Head-on: drive into the e1 wall on the y=20 latitude. The
    column is fixed by the infantry wall long enough for the 3tnk
    to lunge inside aggro 16 and out-DPS the medium pair."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        if (u.get("type") or "").lower() == "2tnk":
            cmds.append(
                Command.attack_move([str(u["id"])], target_x=80, target_y=20)
            )
    return cmds or [Command.observe()]


def _brute_attack_move_policy(rs, Command):
    """Brute: attack_move deep east on the unit's starting latitude.
    On medium/hard the path closes into the 3tnk's aggro envelope
    and the column is destroyed before the kill bar is met."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        if (u.get("type") or "").lower() == "2tnk":
            cmds.append(
                Command.attack_move(
                    [str(u["id"])], target_x=110, target_y=u["cell_y"]
                )
            )
    return cmds or [Command.observe()]


def _intended_flank_pick_policy(rs, Command):
    """Intended off-axis kite-and-pick (the spec's load-bearing play):
    flank approach + retreat if heavy closes + attack_unit nearest e1.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    if not units:
        return [Command.observe()]
    e1s = [
        e for e in enemies
        if (e.get("type") or "").lower() == "e1" and not e.get("is_building")
    ]
    heavies = [
        e for e in enemies
        if (e.get("type") or "").lower() == "3tnk" and not e.get("is_building")
    ]
    cmds = []
    for u in units:
        if (u.get("type") or "").lower() != "2tnk":
            continue
        ux, uy = u["cell_x"], u["cell_y"]
        flank_y = 14 if uy < 20 else 26
        if heavies:
            h = heavies[0]
            dh = ((h["cell_x"] - ux) ** 2 + (h["cell_y"] - uy) ** 2) ** 0.5
            if dh <= HEAVY_DANGER:
                tx = max(4, ux - RETREAT_DIST)
                cmds.append(
                    Command.move_units([str(u["id"])], target_x=tx, target_y=flank_y)
                )
                continue
        if e1s:
            e = min(
                e1s,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(e["id"])))
        else:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=74, target_y=flank_y)
            )
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level (kill bar unmet → after_ticks)."""
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
def test_headon_charge_loses(level):
    """Head-on charge on y=20 must LOSE on medium/hard — the e1 wall
    fixes the column long enough for the 3tnk to lunge and out-DPS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _headon_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: head-on must LOSE; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_brute_attack_move_loses(level):
    """Brute attack_move deep east must LOSE on medium/hard — any path
    that closes into the 3tnk aggro envelope dies."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_attack_move_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute attack_move must LOSE; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_intended_flank_pick_wins(level):
    """Intended off-axis flank-pick — the spec's load-bearing decision.
    Stage off-axis, attack_unit the nearest e1 from a flank cell that
    stays outside the heavy's aggro envelope, retreat west if the
    heavy lunges inside ~14 cells. Verified WINNING on medium (seed=1)
    and on every hard seed (1..4)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_flank_pick_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended flank-pick should WIN, got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )
