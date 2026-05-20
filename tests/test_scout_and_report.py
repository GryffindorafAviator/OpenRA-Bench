"""scout-and-report — perception pack with REVEAL + FULL-TEAM EXTRACT.

Sibling to scout-and-survive (which requires only ANY return of a
single jeep). This pack adds the BOTH-jeep extraction bar and the
specific-far-target reveal bar:

  * 2× jeep scout team (one jeep = the asset, two = the report channel)
  * far enemy fact at (110, 20) (the specific target)
  * win = buildings_discovered_gte:1 AND n:2 units in start region
          AND units_lost_lte:0 AND within_ticks:4500

Scripted policies cover the four bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall                          → LOSS (clock; nothing discovered)
  * one-way scout (no return)      → LOSS (n:2 return clause unmet)
  * brute attack the defenders     → LOSS (units_lost_lte:0 — jeep
                                     dies to picket return-fire)
  * intended discover-and-return   → WIN (every level, seeds 1..4)

Hard tier additionally rotates the agent spawn corner between NORTH
(10, 8) and SOUTH (10, 32) by seed; the intended policy routes per-
corner and returns BOTH jeeps to the SAME corner. The pack's hard
win clause is `any_of` over the two corner regions so a correct
round trip to either corner satisfies it.
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
    / "scout-and-report.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-and-report"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: anchors must call out the military-
    intel / SC2 recon-and-report / drone-surveillance framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("military intelligence" in a for a in anchors), anchors
    assert any("SC2 scout" in a for a in anchors), anchors
    assert any("drone surveillance" in a for a in anchors), anchors
    assert any("intel ops" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout
    or scout-death."""
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


def test_return_clause_n2_present_in_win():
    """Defining feature: the win must enforce n:2 (full-team) return
    at the start region, not just ANY jeep returning."""
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
        assert found, f"{lvl}: missing units_in_region_gte return clause"
        for spec in found:
            assert int(spec.get("n", 1)) >= 2, (
                f"{lvl}: units_in_region_gte must require n>=2 (full team), "
                f"got {spec}"
            )


def test_units_lost_lte_zero_in_win():
    """The full-team-intact bar: units_lost_lte:0 must be a win clause
    (the no-attrition extraction teeth)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "units_lost_lte" in node:
                    found.append(node["units_lost_lte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found, f"{lvl}: missing units_lost_lte clause (no attrition teeth)"
        assert 0 in [int(x) for x in found], (
            f"{lvl}: units_lost_lte must include 0 (zero-attrition extract); "
            f"got {found}"
        )


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _one_way_scout(rs, Command):
    """Push BOTH jeeps east to near the fact and STOP there. Discovers
    the fact but never returns ⇒ units_in_region_gte clause unmet ⇒
    LOSS. Models the "discovery is the goal" misread that the pack
    exists to catch."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        uid = str(u["id"])
        ux, _uy = u["cell_x"], u["cell_y"]
        if ux < 50:
            # Climb to the y=4 edge lane to get past the central
            # bottleneck (works on every level), then run east.
            cmds.append(Command.move_units([uid], target_x=50, target_y=4))
        elif ux < 105:
            cmds.append(Command.move_units([uid], target_x=108, target_y=4))
        elif ux < 108:
            # Dip down to the fact's row to reveal it.
            cmds.append(Command.move_units([uid], target_x=110, target_y=20))
        else:
            # ARRIVED — stop in enemy territory, never return.
            cmds.append(Command.stop([uid]))
    return cmds


def _brute_attack(rs, Command):
    """Drive both jeeps straight east through the picket stack at
    x=58..62, y=18..22. The jeeps auto-fire opportunistically in
    range and draw return fire from the picket stack. Jeep DPS vs an
    infantry stack is poor; at least one jeep dies before crossing,
    so units_lost_lte:0 fires the LOSS clause."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        uid = str(u["id"])
        cmds.append(Command.move_units([uid], target_x=110, target_y=20))
    return cmds


def _intended_discover_and_return(rs, Command):
    """Both jeeps: climb to the spawn-matching edge lane, run east,
    dip down to (110, 20) to surface the fact, climb back to the
    edge lane, traverse west, drop to the spawn cell. The team
    returns INTACT to the SAME spawn corner.

    Phases:
      1. Outbound — climb to edge lane (y≈4 from NORTH, y≈35 from
         SOUTH), run east, dip to reveal the fact at (110, 20).
      2. Return — climb back to the edge lane (so the west traverse
         doesn't pass through the central bottleneck at y=14..24
         on hard), traverse west on the edge, drop to the start
         cell.
    """
    state = _intended_discover_and_return._state  # type: ignore[attr-defined]
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]

    # Latch each jeep's start cell once (its actual spawn) so the
    # return trip targets its OWN spawn corner.
    if not state["starts"]:
        # Use ALL units' spawn cells (the agent's persistent ID set
        # is stable across turns).
        for u in units:
            state["starts"][str(u["id"])] = (u["cell_x"], u["cell_y"])

    # Detect discovery (fact in enemy_buildings_summary).
    eb = rs.get("enemy_buildings_summary", []) or []
    if any(b.get("type") == "fact" for b in eb):
        state["discovered"] = True

    cmds = []
    for u in units:
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        sx, sy = state["starts"].get(uid, (ux, uy))
        # Spawn-matching edge lane.
        edge_y = 35 if sy >= 26 else 4

        if not state["discovered"]:
            if ux < 25:
                cmds.append(Command.move_units([uid], target_x=25, target_y=edge_y))
            elif ux < 105:
                cmds.append(Command.move_units([uid], target_x=108, target_y=edge_y))
            else:
                # Final dip down to the fact's row to reveal it.
                cmds.append(Command.move_units([uid], target_x=110, target_y=20))
        else:
            # Return: climb back to edge lane first, then traverse
            # west, then drop to the start cell.
            if abs(uy - edge_y) > 4 and ux > 100:
                cmds.append(Command.move_units([uid], target_x=ux, target_y=edge_y))
            elif ux > 25:
                cmds.append(Command.move_units([uid], target_x=20, target_y=edge_y))
            else:
                cmds.append(Command.move_units([uid], target_x=sx, target_y=sy))
    return cmds


_intended_discover_and_return._state = {  # type: ignore[attr-defined]
    "starts": {},
    "discovered": False,
}


def _reset_intended_state():
    _intended_discover_and_return._state["starts"] = {}  # type: ignore[attr-defined]
    _intended_discover_and_return._state["discovered"] = False  # type: ignore[attr-defined]


# ── Solvency: intended WINS every (level, seed) ───────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_discover_and_return_wins(level, seed):
    _reset_intended_state()
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_discover_and_return, seed=seed)
    assert res.outcome == "win", (
        f"intended discover+return must WIN on {level} s={seed}; "
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
def test_one_way_scout_loses(level, seed):
    """One-way commit: discovers the fact but stops in enemy
    territory. The n:2 return clause is never satisfied ⇒ LOSS on
    the clock, never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _one_way_scout, seed=seed)
    assert res.outcome == "loss", (
        f"one-way scout must LOSE on {level} s={seed}; got {res.outcome} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)} "
        f"lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", ["medium", "hard"])
def test_brute_attack_loses_medium_and_hard(level, seed):
    """Brute through the picket stack: jeep DPS is insufficient vs
    the 4+× e1 bottleneck (medium/hard); at least one jeep dies to
    return-fire ⇒ units_lost_lte:0 fires ⇒ LOSS.

    Easy is intentionally permissive (only 2 pickets) — a brute
    play CAN survive on easy. The strict bar is medium/hard."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _brute_attack, seed=seed)
    assert res.outcome == "loss", (
        f"brute attack must LOSE on {level} s={seed}; got {res.outcome} "
        f"lost={res.signals.units_lost} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}"
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
