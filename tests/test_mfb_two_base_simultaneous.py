"""mfb-two-base-simultaneous — concurrent two-base ramp.

The bar: the intended CONCURRENT-PIPELINED policy (queue BOTH `proc`
orders on turn 1, then place each as soon as the engine reports it
ready) WINS on every level and every hard seed. Three failure modes
all LOSE on every level + seed:
  • STALL  — only `observe()` issued; nothing is built; after_ticks
              fires.
  • ONE-BASE-ONLY — build & place ONE proc only at the WEST base
              (ignore the EAST base). The EAST `building_in_region:
              proc` clause is never satisfied; after_ticks fires.
  • SERIALIZE-WITH-IDLE-OBSERVE — naive build → observe-and-wait →
              place → observe → build → observe → place. Adds ≥1
              extra throttled turn between every productive command,
              pushing the second `proc` placement to ≥ tick 1893 ≫
              medium/hard within_ticks 1850 → after_ticks 1851 fires.

Validation is scripted (no model / network) — uses
`openra_bench.eval_core.run_level`.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "mfb-two-base-simultaneous.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(own_buildings=(), tick=200):
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_easy_predicates_require_proc_at_each_base():
    c = compile_level(load_pack(PACK_PATH), "easy")
    west_fact = ("fact", 15, 20)
    east_fact = ("fact", 85, 20)
    west_proc = ("proc", 15, 20)
    east_proc = ("proc", 85, 20)
    # WIN: facts alive + a proc at each base inside the budget.
    assert evaluate(
        c.win_condition,
        _ctx([west_fact, east_fact, west_proc, east_proc], tick=500),
    )
    # FAIL clauses don't fire on the winning state.
    assert not evaluate(
        c.fail_condition,
        _ctx([west_fact, east_fact, west_proc, east_proc], tick=500),
    )
    # FAIL: only WEST proc (east side never satisfied).
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, east_fact, west_proc], tick=500),
    )
    # FAIL: only EAST proc.
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, east_fact, east_proc], tick=500),
    )
    # FAIL: a third proc at the middle but neither edge — still fails.
    middle_proc = ("proc", 50, 20)
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, east_fact, middle_proc], tick=500),
    )
    # Past deadline ⇒ fail.
    assert evaluate(
        c.fail_condition,
        _ctx([west_fact, east_fact, west_proc, east_proc], tick=2702),
    )
    # Lost every fact ⇒ fail (the not has_building:fact clause).
    assert evaluate(c.fail_condition, _ctx([], tick=500))
    # Deadline reachable inside max_turns (∼90 ticks/turn).
    assert 2701 <= 93 + 90 * (c.max_turns - 1)


def test_medium_tight_clock_bites():
    c = compile_level(load_pack(PACK_PATH), "medium")
    west_fact = ("fact", 15, 20)
    east_fact = ("fact", 85, 20)
    west_proc = ("proc", 15, 20)
    east_proc = ("proc", 85, 20)
    # WIN before the tight deadline.
    assert evaluate(
        c.win_condition,
        _ctx([west_fact, east_fact, west_proc, east_proc], tick=500),
    )
    # Past tick 1851 ⇒ fail.
    assert evaluate(
        c.fail_condition,
        _ctx([west_fact, east_fact, west_proc, east_proc], tick=1852),
    )
    # Deadline reachable inside max_turns.
    assert 1851 <= 93 + 90 * (c.max_turns - 1)


def test_hard_either_latitude_wins():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # NORTH layout: facts at y=12.
    n_w_fact = ("fact", 15, 12)
    n_e_fact = ("fact", 85, 12)
    n_w_proc = ("proc", 15, 12)
    n_e_proc = ("proc", 85, 12)
    assert evaluate(
        c.win_condition,
        _ctx([n_w_fact, n_e_fact, n_w_proc, n_e_proc], tick=500),
    )
    # SOUTH layout: facts at y=28.
    s_w_fact = ("fact", 15, 28)
    s_e_fact = ("fact", 85, 28)
    s_w_proc = ("proc", 15, 28)
    s_e_proc = ("proc", 85, 28)
    assert evaluate(
        c.win_condition,
        _ctx([s_w_fact, s_e_fact, s_w_proc, s_e_proc], tick=500),
    )
    # MISMATCH: NORTH facts but a SOUTH-latitude proc on east — the
    # NORTH clause needs east at y=12 (south proc at y=28 is too
    # far); the SOUTH clause needs west AT y=28 (the NORTH proc is
    # too far). Neither full-layout clause is satisfied.
    assert not evaluate(
        c.win_condition,
        _ctx([n_w_fact, n_e_fact, n_w_proc, s_e_proc], tick=500),
    )
    # Past tick 1851 ⇒ fail.
    assert evaluate(
        c.fail_condition,
        _ctx([n_w_fact, n_e_fact, n_w_proc, n_e_proc], tick=1852),
    )
    # Deadline reachable inside max_turns.
    assert 1851 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the BASE LATITUDE (NORTH y=12
    vs SOUTH y=28); the chosen target regions flip accordingly."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mfb-two-base-simultaneous"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Spec requires the SC2 / multi-plant / distributed anchors.
    assert "sc2" in joined
    assert "multi-plant" in joined or "manufacturing" in joined
    assert "distributed" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        # Tools must include build + place_building + observe (the
        # whole task is build & place; deploy/move are unnecessary).
        tools = set(c.scenario.tools or [])
        for t in ("observe", "build", "place_building"):
            assert t in tools, f"{lvl}: missing tool {t}"


def test_timeout_loss_reachable_every_level():
    """No draw degeneracy: the level's after_ticks fail must fit
    inside max_turns on every level."""
    pack = load_pack(PACK_PATH)
    bars = {"easy": 2701, "medium": 1851, "hard": 1851}
    for lvl, bar in bars.items():
        c = compile_level(pack, lvl)
        assert bar <= 93 + 90 * (c.max_turns - 1), lvl


def test_starting_cash_exactly_two_procs():
    """Cash sized 2 × proc cost (1400 each) — exactly two refineries;
    a third would not fit so the bar cannot be brute-forced by
    over-building."""
    pack = load_pack(PACK_PATH)
    assert pack.starting_cash == 2800


# ── engine-driven scripted policies ──────────────────────────────────


def _agent_own_buildings(rs):
    """Read agent-owned buildings from the render state.

    The pack also includes an UNARMED enemy `fact` marker at (152,52)
    to inhibit engine auto-`done`; the engine surfaces only
    agent-owned buildings in `own_buildings`, so a simple type filter
    is sufficient. (We still belt-and-suspenders strip any building
    whose cell is the SE marker location.)"""
    out = []
    for b in (rs.get("own_buildings", []) or []):
        cx = int(b.get("cell_x", 0))
        cy = int(b.get("cell_y", 0))
        # Skip the SE marker cell.
        if cx == 152 and cy == 52:
            continue
        out.append(b)
    return out


def _bases(rs):
    """Return [(label, cx, cy)] for WEST / EAST pre-placed facts,
    sorted by cell_x ascending."""
    facts = [b for b in _agent_own_buildings(rs) if str(b.get("type", "")).lower() == "fact"]
    facts_sorted = sorted(facts, key=lambda f: int(f.get("cell_x", 0)))
    out = []
    for label, f in zip(("WEST", "EAST"), facts_sorted[:2]):
        out.append((label, int(f["cell_x"]), int(f["cell_y"])))
    return out


def _placed_proc_cells(rs):
    return {
        (int(b["cell_x"]), int(b["cell_y"]))
        for b in _agent_own_buildings(rs)
        if str(b.get("type", "")).lower() == "proc"
    }


def _proc_pending(rs):
    """Count `proc` items currently in the agent's production list."""
    prod = rs.get("production", []) or []
    return sum(1 for item in prod if str(item).lower() == "proc")


def _intended_concurrent_policy(rs, Command):
    """Pipelined two-base ramp: keep the shared Building queue
    saturated by queueing the NEXT `build proc` whenever any base
    still needs a proc and the queue + already-placed procs together
    won't cover them. Spam `place_building` at each base on every
    turn (the engine just no-ops with PLACE BLOCKED until the build
    actually completes — same idiom as
    `mfb-third-base-against-clock`).

    The placement is at (fact_x+5, fact_y) so the proc footprint
    doesn't collide with the fact/powr footprints, and the cell sits
    well inside the radius-8 region centred at (fact_x, fact_y).
    """
    cmds: list = []
    bases = _bases(rs)
    if len(bases) < 2:
        return [Command.observe()]

    cash = int(rs.get("cash", 0) or 0)
    placed_cells = _placed_proc_cells(rs)
    in_queue = _proc_pending(rs)

    # Which bases still need a proc inside their radius-8 region?
    needed = []
    for label, bx, by in bases:
        if not any((px - bx) ** 2 + (py - by) ** 2 <= 64 for (px, py) in placed_cells):
            needed.append((label, bx, by))

    # Saturate the queue. Each `proc` costs 1400 ⇒ we can afford up
    # to floor(cash/1400) more in the queue, capped by `needed`.
    affordable = cash // 1400
    short = max(0, len(needed) - in_queue)
    to_queue = min(short, affordable)
    for _ in range(to_queue):
        cmds.append(Command.build("proc"))

    # Spam place_building at each needed base.
    for label, bx, by in needed:
        cmds.append(Command.place_building("proc", bx + 5, by))

    if not cmds:
        return [Command.observe()]
    return cmds


def _stall_policy(rs, Command):
    return [Command.observe()]


def _one_base_only_policy(rs, Command):
    """Build and place a proc only at the WEST base — ignore the
    EAST base entirely. The EAST region predicate is never
    satisfied → after_ticks LOSS."""
    bases = _bases(rs)
    if not bases:
        return [Command.observe()]
    _, bx, by = bases[0]
    placed_cells = _placed_proc_cells(rs)
    cash = int(rs.get("cash", 0) or 0)
    placed_west = any(
        (px - bx) ** 2 + (py - by) ** 2 <= 64 for (px, py) in placed_cells
    )
    in_queue = _proc_pending(rs)
    cmds: list = []
    if not placed_west:
        if in_queue == 0 and cash >= 1400:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", bx + 5, by))
    if not cmds:
        return [Command.observe()]
    return cmds


def _serialize_with_idle_observes_policy(rs, Command):
    """Naive sequential policy with idle observe gaps between every
    productive command. The agent acts only on every OTHER turn (a
    90-tick observe pause) and STRICTLY serializes the two bases:
    build west → wait until placed → build east → wait until placed.
    Total wall-clock at least ~tick 1900 ≫ medium/hard within_ticks
    1850 → after_ticks 1851 fires → LOSS."""

    tick = int(rs.get("game_tick", 0) or 0)
    # Idle gap: act only when the turn index is even.
    if (tick // 90) % 2 == 1:
        return [Command.observe()]

    bases = _bases(rs)
    if len(bases) < 2:
        return [Command.observe()]

    cash = int(rs.get("cash", 0) or 0)
    placed_cells = _placed_proc_cells(rs)
    in_queue = _proc_pending(rs)

    w_label, wx, wy = bases[0]
    e_label, ex, ey = bases[1]
    has_proc_west = any(
        (px - wx) ** 2 + (py - wy) ** 2 <= 64 for (px, py) in placed_cells
    )
    has_proc_east = any(
        (px - ex) ** 2 + (py - ey) ** 2 <= 64 for (px, py) in placed_cells
    )

    cmds: list = []
    if not has_proc_west:
        # Strictly one item in the queue at a time.
        if in_queue == 0 and cash >= 1400:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", wx + 5, wy))
    elif not has_proc_east:
        if in_queue == 0 and cash >= 1400:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", ex + 5, ey))

    if not cmds:
        return [Command.observe()]
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_concurrent_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_concurrent_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended concurrent should WIN, got "
            f"{res.outcome} after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE, got {res.outcome} "
            f"after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_one_base_only_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _one_base_only_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: one-base-only must LOSE (east region "
            f"never covered), got {res.outcome} after {res.turns} "
            f"turns; buildings={res.signals.own_buildings}"
        )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_serialize_with_idle_observes_loses_on_tight_clocks(level):
    """The TIGHT-clock tiers (medium + hard) must distinguish
    pipelined-concurrent from serialize-with-idle-observe; this
    failure mode is allowed to WIN on easy (its generous clock is
    deliberate — easy only filters out STALL)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _serialize_with_idle_observes_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: serialize-with-idle-observes must "
            f"LOSE on tight clock, got {res.outcome} after "
            f"{res.turns} turns; buildings={res.signals.own_buildings}"
        )
