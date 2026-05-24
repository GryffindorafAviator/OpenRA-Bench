"""scout-discover-hidden-base — perception pack with OFF-AXIS recon.

The map presents an obvious "enemy presence" in the east (decoy `e1`
pickets) but the REAL enemy `fact` is in an off-axis corner (far NW
on easy/medium, either far NW or far SW on hard). Capability under
test: searching OFF the salient compass axis when the obvious axis
returns no high-value finding.

Scripted policies cover the four bar-defining outcomes per
CLAUDE.md "no defect, no cheat":

  * stall                       → LOSS (clock; nothing discovered)
  * east-only-sweep             → LOSS (clock; decoys aren't a base)
  * attack-decoys               → LOSS (clock; engaging decoys
                                  burns the budget without ever
                                  steering toward the real corner)
  * intended-off-axis-scout     → WIN (every level, seeds 1..4)

Hard tier rotates the agent spawn between NORTH-leaning (15, 16)
and SOUTH-leaning (15, 24); both hidden bases (NW and SW corners)
spawn every seed because enemy actors don't honour `spawn_point`
(see CLAUDE.md), so the spawn rotation flips which off-axis corner
is the SHORTER reach but EITHER hidden-base discovery satisfies
the win.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "scout-discover-hidden-base.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-discover-hidden-base"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: the anchors must call out the
    CICERO / ERQA / intelligence-ops framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("CICERO" in a for a in anchors), anchors
    assert any("ERQA off-axis" in a for a in anchors), anchors
    assert any("intelligence ops" in a for a in anchors), anchors
    assert any("SC2 hidden-base" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout
    or scout-attrition."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (binding
    contract from tests/test_hard_tier.py::UPGRADED)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside
    max_turns (engine advances ~90 ticks/turn → reachable max =
    93 + 90·(N-1)). A `within_ticks` above reachable is inert ⇒
    the timeout LOSS never fires and a stall is a DRAW."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list[int] = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


def test_discovery_clause_present_in_win():
    """The defining feature: the win must enforce
    buildings_discovered_gte (the hidden fact must be SEEN)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "buildings_discovered_gte" in node:
                    found.append(node["buildings_discovered_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found, f"{lvl}: missing buildings_discovered_gte clause"


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _east_only_sweep(rs, Command):
    """Push every jeep due east toward the obvious decoy band. The
    decoys are passive (stance:0) so the jeeps don't die in combat
    — they just spend the entire clock in the wrong region and
    never reveal the hidden NW (or SW) base ⇒ LOSS on the
    deadline."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=60, target_y=u["cell_y"])
        )
    return cmds


def _attack_decoys(rs, Command):
    """Engage the obvious east decoys with the attack_unit tool.
    The decoys are stance:0 so they don't return fire effectively,
    but committing to the east engagement means the jeeps never
    look at the off-axis corners ⇒ LOSS on the clock. If no enemy
    is yet visible, march east to acquire one."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    enemies = rs.get("enemies_summary", []) or []
    cmds = []
    if enemies:
        eid = str(enemies[0]["id"])
        for u in units:
            cmds.append(Command.attack_unit([str(u["id"])], target_id=eid))
    else:
        for u in units:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=56, target_y=u["cell_y"])
            )
    return cmds


def _intended_off_axis_scout(rs, Command):
    """Push the scouts to the NEAR off-axis corner (NW for an
    N-leaning spawn, SW for an S-leaning spawn). For easy/medium
    the spawn is always (10, 20) so NW is always the answer; for
    hard the choice flips per seed but EITHER corner satisfies
    the win clause — picking the NEARER one is the model's job.
    """
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        # Pick the nearer off-axis corner relative to the unit's
        # current row. The two hidden facts sit around y=5 (NW)
        # and y=35 (SW); a unit at y<21 is closer to NW, y>21
        # closer to SW.
        if u["cell_y"] > 21:
            tx, ty = 5, 35
        else:
            tx, ty = 5, 5
        cmds.append(Command.move_units([str(u["id"])], target_x=tx, target_y=ty))
    return cmds


# ── Solvency: intended WINS every (level, seed) ───────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_off_axis_scout_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_off_axis_scout, seed=seed)
    assert res.outcome == "win", (
        f"intended off-axis scout must WIN on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


# ── Stability: every non-win pattern is a real reachable LOSS ─────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_east_only_sweep_loses(level, seed):
    """Sweeping the obvious east axis finds only decoys ⇒ LOSS
    on the deadline, never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _east_only_sweep, seed=seed)
    assert res.outcome == "loss", (
        f"east-only sweep must LOSE on {level} s={seed}; got {res.outcome} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_attack_decoys_loses(level, seed):
    """Engaging the obvious east decoys never steers the scouts
    to the off-axis corner ⇒ LOSS on the clock."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _attack_decoys, seed=seed)
    assert res.outcome == "loss", (
        f"attack-decoys must LOSE on {level} s={seed}; got {res.outcome} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


# ── Spawn variation contract (hard) ───────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    a stall completes deterministically per seed and seeds 1..4
    collectively produce ≥2 distinct starts (the cross-seed
    contract is enforced by tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
