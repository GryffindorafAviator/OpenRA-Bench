"""scout-far-frontier — long-range reconnaissance with chassis-by-speed selection.

The agent base has TWO pre-placed scout assets — a fast jeep (~12-13
cells/turn) and a slow e1 rifle infantry (~4-5 cells/turn) — and a
frontier objective at the far edge of the map. Only the jeep is fast
enough to reach the frontier inside the tick budget. The chassis
selection IS the test.

Win = `units_in_region_gte:{frontier, n:1} AND own_units_gte:1 AND
within_ticks:K AND building_count_gte:{fact, n:1}`. The clock budget
across tiers (easy 1530 / medium 1170 / hard 1080) is sized to admit
the jeep traverse (~9 turns) but reject the e1 traverse (~24 turns).

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall            → LOSS (clock; nothing dispatched)
  * e1-only          → LOSS (slow chassis; cannot reach in budget)
  * jeep             → WIN  (every level, every seed)

Hard tier additionally rotates the agent base latitude between NORTH
(y=8) and SOUTH (y=32) by seed; the frontier objective tracks the
base latitude (NE (120, 5) for NORTH, SE (120, 35) for SOUTH) so a
memorised frontier coordinate cannot generalise. The pack's hard win
clause is `any_of` over the two frontier regions so a correct dispatch
to either matching frontier satisfies it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "scout-far-frontier.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-far-frontier"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: anchors must call out the ERQA recon /
    military long-range reconnaissance / SC2 scout framing the
    chassis-by-speed selection rests on."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("ERQA recon" in a for a in anchors), anchors
    assert any("military long-range reconnaissance" in a for a in anchors), anchors
    assert any("SC2 scout" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout,
    force-wipe, or base-collapse."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (the binding
    contract from tests/test_hard_tier.py::UPGRADED)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks/after_ticks must be reachable inside max_turns.
    Engine advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
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


def test_within_ticks_actually_bites_on_each_tier():
    """The chassis-by-speed discrimination requires the clock to be
    TIGHT enough that the slow chassis (e1, ~4-5 cells/turn) cannot
    cover the ~110-cell traverse. The e1 needs ~24 turns ⇒ ~2160 ticks;
    every tier's within_ticks must be strictly below ~2000 to make the
    slow-chassis play time out."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        wts: list[int] = []

        def _collect(node):
            if isinstance(node, dict):
                if "within_ticks" in node:
                    wts.append(node["within_ticks"])
                for v in node.values():
                    _collect(v)
            elif isinstance(node, list):
                for v in node:
                    _collect(v)

        _collect(win)
        for wt in wts:
            assert wt < 2000, (
                f"{lvl} within_ticks={wt} is too loose — the slow e1 "
                f"can reach the frontier within ~2160 ticks; the test "
                f"is the chassis pick, so the clock must reject e1."
            )


def test_units_in_region_clause_targets_far_frontier():
    """The win must require a unit within radius of a FAR-FRONTIER
    cell (x≥100) — the whole point is long-range reconnaissance,
    not a short hop."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "units_in_region_gte" in node:
                    found.append(node["units_in_region_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found, f"{lvl}: missing units_in_region_gte clause"
        for spec in found:
            assert int(spec.get("x", 0)) >= 100, (
                f"{lvl}: units_in_region_gte must target a FAR-FRONTIER "
                f"cell (x≥100), got {spec}"
            )


def test_building_count_fact_clause_in_win():
    """Win must require building_count_gte:fact:1 — keeps the base
    fact alive as the agent's home anchor and grounds the
    own_buildings signal."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "building_count_gte" in node:
                    found.append(node["building_count_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found, f"{lvl}: missing building_count_gte clause"
        assert any(
            str(spec.get("type", "")).lower() == "fact" for spec in found
        ), f"{lvl}: building_count_gte must include type:fact, got {found}"


def test_both_jeep_and_e1_pre_placed_on_each_tier():
    """The chassis-pick test requires BOTH options on the field at
    start — agent has ≥1 jeep AND ≥1 e1 across the merged actors of
    every level. (Hard duplicates across spawn_point groups; pre-
    spawn filter applies after compile.)"""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {
            str(a.type).lower() for a in c.scenario.actors if a.owner == "agent"
        }
        assert "jeep" in agent_types, f"{lvl}: agent must have a jeep, got {agent_types}"
        assert "e1" in agent_types, f"{lvl}: agent must have an e1, got {agent_types}"
        assert "fact" in agent_types, f"{lvl}: agent must have a fact, got {agent_types}"


# ── Scripted policies ─────────────────────────────────────────────


def _frontier_for(uy: int) -> tuple[int, int]:
    """Pick the spawn-matched frontier coords for a unit at row uy.
    NORTH base (y<16) → NE (120, 5). SOUTH base (y>24) → SE (120, 35).
    Centre / easy / medium base (y∈[16..24]) → (120, 30)."""
    if uy < 16:
        return (120, 5)
    if uy > 24:
        return (120, 35)
    return (120, 30)


def _stall(_rs, Command):
    return [Command.observe()]


def _e1_only(rs, Command):
    """Dispatch ONLY the slow rifleman to the frontier. The e1 needs
    ~24 turns to cross the map but every tier's clock cuts off at or
    before turn 17 ⇒ LOSS on the within_ticks teeth."""
    units = [u for u in (rs.get("units_summary", []) or []) if u.get("type") == "e1"]
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        tx, ty = _frontier_for(u["cell_y"])
        cmds.append(Command.move_units([str(u["id"])], target_x=tx, target_y=ty))
    return cmds


def _jeep_dispatch(rs, Command):
    """Dispatch the FAST chassis to the spawn-matched frontier. The
    jeep covers ~12-13 cells/turn so a 110-cell traverse completes
    in ~9 turns ⇒ WIN well inside every tier's clock."""
    units = [u for u in (rs.get("units_summary", []) or []) if u.get("type") == "jeep"]
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        tx, ty = _frontier_for(u["cell_y"])
        cmds.append(Command.move_units([str(u["id"])], target_x=tx, target_y=ty))
    return cmds


# ── Solvency: jeep-dispatch WINS every (level, seed) ──────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_jeep_dispatch_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _jeep_dispatch, seed=seed)
    assert res.outcome == "win", (
        f"jeep dispatch must WIN on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} turns={res.turns}"
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
def test_e1_only_loses(level, seed):
    """The slow-chassis-only play: e1 dispatched, jeep idle. The e1
    cannot cross ~110 cells in the tight tick budget ⇒ clock LOSS,
    never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _e1_only, seed=seed)
    assert res.outcome == "loss", (
        f"e1-only must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} turns={res.turns}"
    )


# ── Spawn variation contract (hard) ───────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    seeds 1..4 collectively must produce ≥2 distinct starts. The
    spawn-variation contract is enforced cross-seed by
    tests/test_hard_tier.py::test_curated_hard_still_compiles_and_runs."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
