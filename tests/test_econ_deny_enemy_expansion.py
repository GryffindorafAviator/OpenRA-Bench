"""econ-deny-enemy-expansion — economic denial / expansion denial.

Bar: the intended play is the RAID — drive the 4-tank strike force
forward to the lightly-defended enemy expansion `proc` at mid-map and
raze it before the deadline, keeping ≥2 tanks and the home `fact`
alive. The degenerate plays (stall, turtle) must LOSE.

Discrimination bar verified by engine-driven scripted policies:

  • stall (only observe)             → LOSS (the expansion proc is
    never razed; enemy_key_buildings_destroyed_in_region unmet →
    after_ticks fail).
  • turtle (keep the army home)      → LOSS (same — guarding the
    home fact scores nothing; the proc-destruction clause is the
    only path to a win).
  • naive all-in attack-move into the patch (medium / hard) → LOSS
    (the thicker garrison's anti-tank e3 trades the column below
    the ≥2-tank survival floor; the raid must focus the e3 first).
  • intended raid (advance, focus the anti-tank defender, raze the
    proc, keep ≥2 tanks + the home fact)  → WIN.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "econ-deny-enemy-expansion.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(
    units_xy=(),
    tick=1000,
    lost=0,
    destroyed_records=(),
    own_buildings=(),
):
    """Synthesize a WinContext for predicate-level checks.

    destroyed_records: iterable of (type, x, y) for buildings the agent
    destroyed. own_buildings: iterable of (type, x, y) for agent-owned
    buildings still standing.
    """
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        own_buildings=list(own_buildings),
        own_building_types={t for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        enemy_buildings_destroyed_records=list(destroyed_records),
        enemy_buildings_destroyed_types={},
        enemy_buildings_destroyed=len(destroyed_records),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [{"cell_x": x, "cell_y": y} for x, y in units_xy]
        },
    )


# The intended terminal state: proc razed, ≥2 tanks alive, home fact
# still standing, before the deadline.
_PROC_RAZED = [("proc", 60, 20)]
_HOME_FACT = [("fact", 6, 20)]
_TANKS4 = [(60, 20), (60, 21), (60, 19), (60, 22)]


@pytest.mark.parametrize(
    "level,deadline", [("easy", 5400), ("medium", 4500), ("hard", 4500)]
)
def test_predicates_win_and_fail(level, deadline):
    c = compile_level(load_pack(PACK_PATH), level)

    # Intended: proc razed, ≥2 tanks, home fact alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(_TANKS4, tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=_HOME_FACT),
    )
    assert evaluate(
        c.win_condition,
        _ctx(_TANKS4[:2], tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=_HOME_FACT),
    )
    # Proc NOT razed → win predicate unmet (the only path to a win)
    assert not evaluate(
        c.win_condition,
        _ctx(_TANKS4, tick=deadline - 500, destroyed_records=(),
             own_buildings=_HOME_FACT),
    )
    # Only 1 tank survives → win unmet, fail fires (not own_units_gte:2)
    assert not evaluate(
        c.win_condition,
        _ctx(_TANKS4[:1], tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=_HOME_FACT),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(_TANKS4[:1], tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=_HOME_FACT),
    )
    # Home fact lost → win unmet, fail fires (not building_count_gte:fact)
    assert not evaluate(
        c.win_condition,
        _ctx(_TANKS4, tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=()),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(_TANKS4, tick=deadline - 500, destroyed_records=_PROC_RAZED,
             own_buildings=()),
    )
    # Past the deadline → real timeout LOSS, reachable within max_turns
    assert evaluate(
        c.fail_condition,
        _ctx(_TANKS4, tick=deadline + 2, destroyed_records=(),
             own_buildings=_HOME_FACT),
    )
    assert deadline + 1 <= 93 + 90 * (c.max_turns - 1), (
        f"{level}: after_ticks {deadline + 1} must be reachable within max_turns"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_sentinel_and_main_do_not_satisfy_win(level):
    """Region scoping: razing the far-corner SENTINEL fact (124,4) or
    the enemy MAIN buildings (108,20) does NOT satisfy the win — only
    the EXPANSION proc at (60,20) counts."""
    c = compile_level(load_pack(PACK_PATH), level)
    for stray in (
        [("fact", 124, 4)],
        [("fact", 108, 20)],
        [("powr", 108, 24)],
    ):
        assert not evaluate(
            c.win_condition,
            _ctx(_TANKS4, tick=1000, destroyed_records=stray,
                 own_buildings=_HOME_FACT),
        ), f"{level}: {stray} must not satisfy the proc-region win clause"


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the strike-force start latitude."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_each_spawn_group_has_strike_force_and_home_fact():
    """Both hard spawn groups must carry the full kit (4 tanks + a
    home fact) so the raid + the building_count_gte:fact:1 clause are
    well-posed regardless of which spawn the seed selects."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    for grp in (0, 1):
        actors = [
            a for a in c.scenario.actors
            if a.owner == "agent"
            and (a.spawn_point if a.spawn_point is not None else 0) == grp
        ]
        tanks = sum(1 for a in actors if a.type == "2tnk")
        facts = sum(1 for a in actors if a.type == "fact")
        assert tanks == 4, f"hard spawn {grp}: expected 4 tanks, got {tanks}"
        assert facts == 1, f"hard spawn {grp}: expected 1 home fact, got {facts}"


def test_expansion_proc_present_and_garrison_grows():
    """Structural: every tier has the enemy expansion proc at (60,20)
    and a harvester; the garrison around it grows easy → medium=hard."""
    pack = load_pack(PACK_PATH)
    sizes = {}
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        procs = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "proc"
            and tuple(a.position) == (60, 20)
        ]
        assert procs, f"{lvl}: enemy expansion proc must sit at (60,20)"
        harvs = [a for a in c.scenario.actors
                 if a.owner == "enemy" and a.type == "harv"]
        assert harvs, f"{lvl}: enemy expansion needs a harvester"
        garrison = [
            a for a in c.scenario.actors
            if a.owner == "enemy"
            and a.type in ("e1", "e3")
            and abs(a.position[0] - 60) + abs(a.position[1] - 20) <= 8
        ]
        sizes[lvl] = len(garrison)
    assert sizes["easy"] < sizes["medium"], (
        f"expansion garrison must thicken easy→medium: {sizes}"
    )
    assert sizes["medium"] == sizes["hard"], (
        f"medium and hard share the garrison structure: {sizes}"
    )


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "econ-deny-enemy-expansion"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "expansion denial" in joined
    assert "economic warfare" in joined
    assert "resource interdiction" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    expected = {"easy": 5401, "medium": 4501, "hard": 4501}
    for lvl, deadline in expected.items():
        c = compile_level(pack, lvl)
        assert deadline <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks {deadline} not reachable within max_turns"
        )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Stall: only observe. The expansion proc is never razed → win
    predicate unmet → after_ticks LOSS."""
    return [Command.observe()]


def _turtle_policy(rs, Command):
    """Turtle: keep the army home near the construction yard. The
    proc stands; guarding the home fact scores nothing → LOSS."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units([str(u["id"])], target_x=12, target_y=u["cell_y"])
        for u in units
    ]


def _naive_allin_policy(rs, Command):
    """Naive all-in: undirected attack-move into the patch. On the
    harder tiers the anti-tank e3 trades the column below the ≥2-tank
    survival floor (the raid must focus the e3 first)."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move([str(u["id"])], target_x=60, target_y=20)
        for u in units
    ]


def _intended_raid_policy(rs, Command):
    """Intended expansion raid: advance to the patch, focus the
    anti-tank rocket soldier first, then raze the proc. Keeps the
    column intact and razes the refinery well before the deadline."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    if not units:
        return [Command.observe()]
    e3 = [
        e for e in enemies
        if (e.get("type") or "").lower() == "e3"
        and not e.get("is_building", False)
    ]
    proc = [e for e in enemies if (e.get("type") or "").lower() == "proc"]
    cmds = []
    for u in units:
        if u["cell_x"] < 50:
            cmds.append(
                Command.attack_move([str(u["id"])], target_x=58, target_y=20)
            )
        elif e3:
            cmds.append(Command.attack_unit([str(u["id"])], str(e3[0]["id"])))
        elif proc:
            cmds.append(Command.attack_unit([str(u["id"])], str(proc[0]["id"])))
        else:
            cmds.append(
                Command.attack_move([str(u["id"])], target_x=60, target_y=20)
            )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level (the expansion proc is never
    razed → after_ticks LOSS)."""
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
def test_turtle_policy_loses(level):
    """Turtle must LOSE on every level (army stays home; the proc
    stands; guarding the home fact scores nothing)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _turtle_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: turtle must LOSE; got {res.outcome}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_naive_allin_loses_on_harder_tiers(level):
    """Naive undirected all-in must LOSE on medium / hard — the
    anti-tank e3 trades the column below the ≥2-tank survival floor."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _naive_allin_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: naive all-in must LOSE; got {res.outcome} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_raid_wins(level):
    """The intended expansion raid must WIN on every level and every
    hard seed — advance, focus the anti-tank defender, raze the proc,
    keep ≥2 tanks + the home fact."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_raid_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended raid must WIN; got {res.outcome} "
            f"lost={res.signals.units_lost}"
        )
