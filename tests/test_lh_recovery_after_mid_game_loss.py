"""No-cheat + solvency proof for `lh-recovery-after-mid-game-loss`
(Wave-11 REASONING: PlanBench replanning under exogenous failure /
disaster recovery / SC2 comeback anchor).

The pack tests disaster recovery: the agent inherits a working
production base (fact + proc + powr + weap + harv) plus a forward
column of heavy tanks and an offensive objective (clear the eastern
`e1` garrison). Mid-episode a `scheduled_events` `destroy_actors`
event at tick 1500 wipes every agent actor inside a circular blast
region — the Ore Refinery (`proc`) and the forward tanks caught in it
(the "disaster"). The deadline does NOT move. The agent must
(1) rebuild the refinery (`build('proc') + place_building`), AND
(2) continue the offensive with the surviving + replacement force.

The win clause is `units_killed_gte` (the objective) AND
`building_count_gte:{proc,1}` (the refinery rebuilt) AND a
`after_ticks:1600` gate (so the win cannot latch before the disaster
fires) AND `within_ticks` (the deadline). The `after_ticks` gate is
SAFE here: the agent only ever kills the `e1` garrison, never the
persistent enemy `fact`, so the engine's conquest auto-`done` cannot
pre-empt the window (CLAUDE.md footgun does not apply).

For every level + every hard seed (1-4):
  * INTENDED (rebuild the refinery AND push the force east) WINS;
  * STALL / give-up-after-loss (only `observe`) LOSES;
  * ATTACK-ONLY (clear the garrison but never rebuild the refinery)
    LOSES — the proc clause is false at the after_ticks gate;
  * REBUILD-ONLY (rebuild the refinery but never attack) LOSES — the
    kill bar is never met.

The 3 lazy plays + 1 intended × 3 levels × (1 or 4) seeds gives the
full no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-recovery-after-mid-game-loss.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _tanks(rs):
    return [u for u in (rs.get("units_summary") or [])
            if str(u.get("type", "")).lower() == "3tnk"]


def _has_proc(rs):
    return any(str(b.get("type", "")).lower() == "proc"
              for b in (rs.get("own_buildings") or []))


def _fact(rs):
    return next((b for b in (rs.get("own_buildings") or [])
                 if str(b.get("type", "")).lower() == "fact"), None)


def _prod_str(rs):
    return " ".join(str(p).lower() for p in (rs.get("production") or []))


def _enemy_target(rs):
    """Nearest surviving enemy unit cell, else the garrison cell."""
    units = [e for e in (rs.get("enemy_summary") or [])
             if e.get("cell_x") is not None]
    if units:
        return units[0]["cell_x"], units[0]["cell_y"]
    return 82, 20


def _stall(rs, Command):
    """Give up after the loss: only observe. Never rebuilds, never
    attacks → both win clauses unmet → reachable timeout LOSS."""
    return [Command.observe()]


def _attack_only(rs, Command):
    """Continue the objective but never rebuild the refinery. May
    clear the garrison, but `building_count_gte:proc` is false at the
    after_ticks gate (the disaster razed the refinery) → LOSS."""
    tanks = _tanks(rs)
    if not tanks:
        return [Command.observe()]
    tx, ty = _enemy_target(rs)
    return [Command.attack_move([str(u["id"]) for u in tanks],
                                target_x=tx, target_y=ty)]


def _rebuild_only(rs, Command):
    """Rebuild the refinery but never push east. The proc clause
    latches but the kill bar is never met → LOSS."""
    if _has_proc(rs):
        return [Command.observe()]
    ps = _prod_str(rs)
    fact = _fact(rs)
    if "proc" in ps and fact:
        return [Command.place_building("proc",
                                       target_x=fact["cell_x"] + 4,
                                       target_y=fact["cell_y"])]
    if "proc" not in ps:
        return [Command.build("proc")]
    return [Command.observe()]


def _intended(rs, Command):
    """Disaster recovery: rebuild the refinery, fund replacement
    tanks from the restored economy, and push the surviving +
    replacement force east to clear the garrison. WINS every level ×
    every seed inside the deadline."""
    cmds = []
    ps = _prod_str(rs)
    fact = _fact(rs)
    cash = rs.get("cash", 0) or 0
    if not _has_proc(rs):
        if "proc" in ps and fact:
            cmds.append(Command.place_building("proc",
                                               target_x=fact["cell_x"] + 4,
                                               target_y=fact["cell_y"]))
        elif "proc" not in ps:
            cmds.append(Command.build("proc"))
    tanks = _tanks(rs)
    if len(tanks) < 4 and cash >= 1150 and "3tnk" not in ps:
        cmds.append(Command.build("3tnk"))
    if tanks:
        tx, ty = _enemy_target(rs)
        cmds.append(Command.attack_move([str(u["id"]) for u in tanks],
                                        target_x=tx, target_y=ty))
    return cmds or [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-recovery-after-mid-game-loss"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue


def test_benchmark_anchor_lists_planbench_disaster_and_comeback():
    """meta.benchmark_anchor must name PlanBench replanning, disaster
    recovery, and the SC2 comeback (Wave-11 spec anchors)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "planbench" in blob, anchors
    assert "disaster recovery" in blob, anchors
    assert "comeback" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_disaster_is_a_destroy_actors_scheduled_event(level):
    """The disaster premise: each tier declares a `scheduled_events`
    `destroy_actors` event filtered to an agent-owned circular region
    that fires mid-episode (CLAUDE.md Wave-9 feature). Hard declares
    one per spawn latitude."""
    c = compile_level(load_pack(PACK), level)
    sched = c.scheduled_events or []
    destroys = [e for e in sched if e.get("type") == "destroy_actors"]
    assert destroys, f"{level}: must declare a destroy_actors disaster event"
    for e in destroys:
        assert e.get("tick", 0) >= 1, f"{level}: destroy event needs a tick"
        filt = e.get("filter") or {}
        assert filt.get("owner") == "agent", (
            f"{level}: the disaster must target the AGENT's own actors"
        )
        region = filt.get("region") or {}
        assert region.get("radius", 0) > 0, (
            f"{level}: the disaster must be a bounded circular region"
        )
    if level == "hard":
        assert len(destroys) >= 2, (
            "hard: one destroy_actors region per spawn latitude"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_win_requires_kill_bar_and_rebuilt_refinery_after_disaster(level):
    """Structural: the win clause must require `units_killed_gte` (the
    objective) AND `building_count_gte:{proc,1}` (the refinery rebuilt)
    AND an `after_ticks` gate that fires AFTER the disaster tick (so
    the win cannot latch before the disaster bites)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    killed = next((int(x["units_killed_gte"]) for x in all_of
                   if "units_killed_gte" in x), None)
    assert killed and killed >= 1, f"{level}: win must require units_killed_gte"
    proc = next((x["building_count_gte"] for x in all_of
                 if "building_count_gte" in x
                 and (x["building_count_gte"] or {}).get("type") == "proc"),
                None)
    assert proc is not None and int(proc.get("n", 0)) >= 1, (
        f"{level}: win must require building_count_gte proc≥1 (rebuilt)"
    )
    win_after = next((int(x["after_ticks"]) for x in all_of
                      if "after_ticks" in x), None)
    assert win_after is not None, (
        f"{level}: win must be gated behind an after_ticks (post-disaster)"
    )
    sched = c.scheduled_events or []
    disaster_tick = min(
        e.get("tick", 0) for e in sched if e.get("type") == "destroy_actors"
    )
    assert win_after > disaster_tick, (
        f"{level}: win after_ticks {win_after} must be AFTER the disaster "
        f"tick {disaster_tick} so a pre-disaster assault cannot win early"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: fail after_ticks must
    be reachable within max_turns (tick ≈ 93 + 90·(max_turns − 1));
    within_ticks + 1 == fail after_ticks so a non-finisher LOSES on the
    very next tick after the win window."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fail = c.fail_condition.model_dump()
    fail_after = next(
        int(x["after_ticks"]) for x in fail["any_of"] if "after_ticks" in x
    )
    reachable = 93 + 90 * (c.max_turns - 1)
    assert fail_after <= reachable, (
        f"{level}: fail after_ticks {fail_after} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within if "within_ticks" in x)
    assert fail_after == wt + 1, (
        f"{level}: fail after_ticks {fail_after} must equal within_ticks+1"
    )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups so the engine round-robins the
    start by seed."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN ───────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_recover_and_continue_wins(level, seed):
    """The intended capability — rebuild the refinery after the
    disaster AND continue the offensive — WINS every level × every
    hard seed inside the deadline."""
    c, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended recover-and-continue should WIN, "
        f"got {r.outcome}; killed={getattr(r.signals, 'units_killed', '?')}, "
        f"tick={getattr(r.signals, 'game_tick', '?')}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall / give-up-after-loss must LOSE every level × every seed —
    never rebuilds, never attacks → reachable timeout LOSS."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_attack_only_without_rebuild_loses(level, seed):
    """Attack-only (clear the garrison but never rebuild the refinery)
    must LOSE — `building_count_gte:proc` is false at the after_ticks
    gate, so the win never latches → reachable timeout LOSS. This is
    the canonical 'continued the objective but ignored the disaster'
    failure mode."""
    c, r = _run(level, _attack_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} attack-only must LOSE; got {r.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_rebuild_only_without_attack_loses(level, seed):
    """Rebuild-only (rebuild the refinery but never push east) must
    LOSE — the kill bar is never met → reachable timeout LOSS."""
    c, r = _run(level, _rebuild_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} rebuild-only must LOSE; got {r.outcome}"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
