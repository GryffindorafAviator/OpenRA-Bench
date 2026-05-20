"""mfb-third-base-against-clock — greedy 3-base macro under deadline.

The bar (binding): the intended EXPAND-3RD-BASE policy (build a 3rd
`proc` and `place_building` at the FAR-EAST target region) WINS on
every level and every hard seed. Three failure modes all LOSE on
every level + seed:
  • STALL                — only `observe()`; never builds anything →
                           clock LOSS (`after_ticks`).
  • ARMY-ONLY            — `build('e1')` on a loop; cash burns on
                           infantry, the 3rd `proc` is never funded
                           → `building_in_region`/`building_count_gte`
                           clauses never fire → clock LOSS.
  • PLACE-PROC-AT-WEST   — build `proc` but place it adjacent to the
                           WEST base (outside the east region) →
                           `building_count_gte:{type:proc, n:3}` fires
                           but `building_in_region:(90,*,r8)` does NOT
                           → clock LOSS.

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
PACK_PATH = PACKS / "mfb-third-base-against-clock.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS_HARD = (1, 2, 3, 4)


# ── unit-level predicate checks (no engine) ────────────────────────


def _ctx(own_buildings=(), tick=1000, n_units=1):
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    units = [
        {"id": str(i), "type": "harv", "cell_x": 22, "cell_y": 20, "owner": "agent"}
        for i in range(n_units)
    ]
    return WinContext(signals=sig, render_state={"units_summary": units})


def test_predicates_easy_win_requires_3_procs_and_one_in_east():
    c = compile_level(load_pack(PACK_PATH), "easy")
    west_fact = ("fact", 15, 20)
    mid_fact = ("fact", 50, 20)
    west_proc = ("proc", 18, 20)
    mid_proc = ("proc", 53, 20)
    east_proc = ("proc", 90, 20)
    # WIN: 3 procs (one inside east region), fact present, within time.
    assert evaluate(
        c.win_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc, east_proc], tick=4000),
    )
    # FAIL: only 2 procs (3rd never funded).
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc], tick=4000),
    )
    # FAIL: 3 procs but the 3rd is at the WEST (not in east region) —
    # building_in_region clause fails.
    proc_at_west = ("proc", 22, 20)  # adjacent to west base
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc, proc_at_west], tick=4000),
    )
    # FAIL: clock past deadline.
    assert evaluate(
        c.fail_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc, east_proc], tick=6302),
    )
    # FAIL: last fact lost.
    assert evaluate(
        c.fail_condition,
        _ctx([west_proc, mid_proc, east_proc], tick=4000),
    )
    # Deadline reachable inside max_turns (~90 ticks/turn).
    assert 6301 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_medium_radius_8_boundary():
    c = compile_level(load_pack(PACK_PATH), "medium")
    west_fact = ("fact", 15, 20)
    mid_fact = ("fact", 50, 20)
    west_proc = ("proc", 18, 20)
    mid_proc = ("proc", 53, 20)
    # Inside: (92, 22) — dist² = 4+4 = 8 ≤ 64 ✓
    inside = ("proc", 92, 22)
    assert evaluate(
        c.win_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc, inside], tick=3000),
    )
    # Outside: (99, 20) — dist 9 > 8.
    outside = ("proc", 99, 20)
    assert not evaluate(
        c.win_condition,
        _ctx([west_fact, mid_fact, west_proc, mid_proc, outside], tick=3000),
    )
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_either_latitude_wins():
    c = compile_level(load_pack(PACK_PATH), "hard")
    fact = ("fact", 15, 20)
    p1 = ("proc", 18, 20)
    p2 = ("proc", 53, 20)
    north = ("proc", 90, 20)
    south = ("proc", 90, 50)
    # Either candidate region wins (the any_of branch).
    assert evaluate(c.win_condition, _ctx([fact, p1, p2, north], tick=3000))
    assert evaluate(c.win_condition, _ctx([fact, p1, p2, south], tick=3000))
    # Mid-latitude (y=35) is outside BOTH candidate regions → LOSS.
    mid_lat = ("proc", 90, 35)
    assert not evaluate(c.win_condition, _ctx([fact, p1, p2, mid_lat], tick=3000))
    # Past deadline ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([fact, p1, p2, north], tick=5402))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract (tests/test_hard_tier.py): hard
    must define ≥2 agent spawn_point groups so the seed round-robins
    the base latitude (NORTH vs SOUTH)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mfb-third-base-against-clock"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor or []
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # The spec demands these three real-world / benchmark anchors.
    assert "sc2 3-base macro" in joined, anchors
    assert "microrts expansion" in joined, anchors
    assert "industrial site expansion" in joined, anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        tools = set(c.scenario.tools or [])
        # Required tools for the intended capability play.
        for t in ("observe", "build", "place_building", "move_units", "stop"):
            assert t in tools, f"{lvl}: missing tool {t}"


def test_timeout_loss_reachable_every_level():
    """No DRAW degeneracy: every level's after_ticks fail must fit
    inside max_turns."""
    pack = load_pack(PACK_PATH)
    bars = {"easy": 6301, "medium": 4501, "hard": 5401}
    for lvl, bar in bars.items():
        c = compile_level(pack, lvl)
        assert bar <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ────────────────────────────────


def _own_buildings(rs):
    return rs.get("own_buildings", []) or []


def _own_b_types(rs):
    return [b["type"] for b in _own_buildings(rs)]


def _own_facts(rs):
    return [b for b in _own_buildings(rs) if b.get("type") == "fact"]


def _target_for(level: str, rs):
    """Pick the FAR-EAST target cell appropriate to the spawn.

    easy/medium have a single target at (90,20). Hard has TWO
    candidate regions ((90,20) and (90,50)); pick the one whose
    latitude matches the agent's actual bases (read the first own
    fact's y-coord)."""
    if level != "hard":
        return (90, 20)
    facts = _own_facts(rs)
    if not facts:
        return (90, 20)
    fy = facts[0].get("cell_y", 20)
    return (90, 20) if abs(fy - 20) <= abs(fy - 50) else (90, 50)


def _intended_expand_policy_for(level: str):
    """Build a 3rd proc and place it inside the eastern target
    region. Place_building is retried every turn until the building
    surfaces in own_buildings (engine emits 'PLACE BLOCKED' until the
    build clock finishes; same idiom as
    mfb-tech-base-vs-economy-base, rob-cash-depletion-recovery)."""

    def _policy(rs, Command):
        own_b = _own_b_types(rs)
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        cmds = []
        n_proc = sum(1 for t in own_b if t == "proc")
        tx, ty = _target_for(level, rs)
        # If we don't yet have 3 procs, queue a new one and spam
        # place_building at the target cell until it lands.
        if n_proc < 3:
            if "proc" not in prod and cash >= 1400:
                cmds.append(Command.build("proc"))
            cmds.append(Command.place_building("proc", tx, ty))
        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return _policy


def _stall_policy(rs, Command):
    return [Command.observe()]


def _army_only_policy(rs, Command):
    """Spend the cash on infantry instead of on the 3rd refinery.
    The proc is never queued → the building_in_region clause never
    fires → clock LOSS."""
    prod = rs.get("production", []) or []
    cash = rs.get("cash", 0)
    if cash >= 100 and "e1" not in prod:
        return [Command.build("e1")]
    return [Command.observe()]


def _place_proc_at_west_policy_for(level: str):
    """Build a 3rd proc but place it ADJACENT TO THE WEST BASE
    (cell (22,20) — outside the (90,20) east region). The
    building_count_gte:proc,n:3 clause fires but the
    building_in_region clause does NOT → clock LOSS."""
    del level  # west-base placement is the same on every level

    def _policy(rs, Command):
        own_b = _own_b_types(rs)
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        cmds = []
        n_proc = sum(1 for t in own_b if t == "proc")
        if n_proc < 3:
            if "proc" not in prod and cash >= 1400:
                cmds.append(Command.build("proc"))
            # Pick a west-adjacent cell that's NEVER inside either
            # east region (radius 8 around x=90).
            facts = _own_facts(rs)
            wx = facts[0].get("cell_x", 15) + 7 if facts else 22
            wy = facts[0].get("cell_y", 20) if facts else 20
            cmds.append(Command.place_building("proc", wx, wy))
        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return _policy


@pytest.mark.parametrize("level", LEVELS)
def test_intended_expand_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = SEEDS_HARD if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_expand_policy_for(level), seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended expand-3rd-base should WIN, "
            f"got {res.outcome} after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}, "
            f"cash={res.signals.cash}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = SEEDS_HARD if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE, got {res.outcome} "
            f"after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_army_only_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = SEEDS_HARD if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _army_only_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: army-only must LOSE (3rd proc never "
            f"funded), got {res.outcome} after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_place_proc_at_west_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = SEEDS_HARD if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _place_proc_at_west_policy_for(level), seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: place-proc-at-west must LOSE "
            f"(building_in_region clause never fires), got {res.outcome} "
            f"after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )
