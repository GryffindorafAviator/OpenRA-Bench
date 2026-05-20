"""scout-and-survive — perception pack with mandatory return.

The perception cell whose win clause is `discover AND return alive`
rather than `discover only`. Scripted policies cover the four
bar-defining outcomes per CLAUDE.md "no defect, no cheat":

  * stall              → LOSS (clock; nothing discovered)
  * one-way scout      → LOSS (return clause; jeep strands east)
  * brute attack       → LOSS (own_units_gte clause; jeep dies to
                         the picket stack; jeep DPS vs infantry is
                         poor and there is no attack tool in any
                         case — the auto-fire engagement still
                         loses on attrition)
  * intended detour    → WIN (every level, seeds 1..4)

Hard tier additionally rotates the agent spawn corner between
NORTH (10, 8) and SOUTH (10, 32) by seed; the intended policy
routes per-corner and returns to the SAME corner. The pack's
hard win clause is `any_of` over the two corner regions so a
correct round trip to either corner satisfies it.
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
    / "scout-and-survive.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-and-survive"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: the anchors must call out the
    SC2 worker-scout / ERQA preservation framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2 worker-scout" in a for a in anchors), anchors
    assert any("ERQA partial-observability" in a for a in anchors), anchors
    assert any("UAV reconnaissance" in a for a in anchors), anchors


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


def test_return_clause_present_in_win():
    """The defining feature of this pack: the win must enforce
    units_in_region_gte at the start region (not just discovery)."""
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


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _start_corner_for(rs) -> tuple[int, int]:
    """Hard rotates the spawn between (10, 8) and (10, 32). Return
    the cell closest to the jeep's CURRENT row so the intended
    policy can target the same corner for the return trip."""
    units = rs.get("units_summary", []) or []
    if not units:
        return (10, 20)
    j = units[0]
    if j["cell_y"] < 18:
        return (10, 8)
    if j["cell_y"] > 26:
        return (10, 32)
    return (10, 20)


def _one_way_scout(rs, Command):
    """Push the jeep east to the fact and STOP there. Discovers the
    fact but never returns ⇒ units_in_region_gte clause unmet ⇒ LOSS.
    Models the "discovery is the goal" misread that the pack exists
    to catch."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        if ux < 110:
            # Detour around the central bottleneck via the top edge
            # (y=4 row is clear on every level) — the goal here is
            # to ACTUALLY reach the fact and then stop, not to
            # brute the bottleneck (that's a different failure mode
            # covered by _brute_attack below).
            if ux < 50:
                cmds.append(Command.move_units([uid], target_x=50, target_y=4))
            elif ux < 110:
                cmds.append(Command.move_units([uid], target_x=115, target_y=20))
        else:
            cmds.append(Command.stop([uid]))
    return cmds


def _brute_attack(rs, Command):
    """Drive the jeep straight east through the picket stack at
    x=58..62, y=18..22. The jeep has no attack tool, but
    move_units auto-fires opportunistically in range — the jeep
    fires AND draws return fire from the picket stack. Jeep DPS
    vs an infantry stack is poor; the jeep dies before crossing.
    Either own_units_gte:1 or the timeout fires the LOSS clause."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        uid = str(u["id"])
        # Beeline EAST at the bottleneck centreline — into the stack.
        cmds.append(Command.move_units([uid], target_x=115, target_y=20))
    return cmds


def _intended_detour_return(rs, Command):
    """Detour AROUND the central picket stack (and any forward
    picket on hard), reveal the fact, and return to the start
    corner. The route uses the y=6 (top) edge or y=33 (bottom)
    edge — chosen by spawn corner — to avoid the bottleneck that
    spans y=16..22 on every level. On hard the forward picket at
    (50, 10) sits 4 rows below the y=6 lane (out of the
    react/aggro radius of an `e1` at stance:1, ≈4); the south
    counterpart at (50, 30) is 3 rows above the y=33 lane.

    Phases:
      1. Outbound — climb to the spawn-matching edge lane, run
         east, then dip down to (115, 20) to surface the fact.
      2. Return — climb BACK up to the edge lane (so the return
         west doesn't pass through the central bottleneck at
         y≈18..22), traverse west on the edge, then drop to the
         start cell.
    """
    state = _intended_detour_return._state  # type: ignore[attr-defined]
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    u = units[0]
    uid = str(u["id"])
    ux, uy = u["cell_x"], u["cell_y"]

    # Latch the start corner once on turn 1 (the jeep's actual
    # spawn cell), so the return trip targets the SAME corner.
    if state["start"] is None:
        state["start"] = (ux, uy)

    sx, sy = state["start"]
    # Spawn-matching edge lane. The map's playable y-range is
    # roughly [2, 37] (cordon=4 on h=40). The thickened hard
    # bottleneck (y=14..24) means only the FAR edges of the map
    # safely clear it in e1 range (~5 cells): y=4 from the north,
    # y=35 from the south. Easy/medium have a thinner bottleneck
    # (y=16..22) so y=4 / y=35 work there too — same policy
    # for every level.
    edge_y = 35 if sy >= 26 else 4

    eb = rs.get("enemy_buildings_summary", []) or []
    if any(b.get("type") == "fact" for b in eb):
        state["discovered"] = True

    if not state["discovered"]:
        if ux < 25:
            # Climb from spawn row to the edge lane.
            return [Command.move_units([uid], target_x=25, target_y=edge_y)]
        if ux < 105:
            # Hold the edge lane east until clear of the pickets.
            return [Command.move_units([uid], target_x=110, target_y=edge_y)]
        # Final dip down to the fact's row to reveal it.
        return [Command.move_units([uid], target_x=115, target_y=20)]

    # Return: climb back up to the edge lane before traversing
    # west — a straight west from (115, 20) would walk straight
    # back through the central bottleneck.
    if abs(uy - edge_y) > 4 and ux > 100:
        return [Command.move_units([uid], target_x=ux, target_y=edge_y)]
    if ux > 25:
        return [Command.move_units([uid], target_x=20, target_y=edge_y)]
    return [Command.move_units([uid], target_x=sx, target_y=sy)]


_intended_detour_return._state = {  # type: ignore[attr-defined]
    "start": None,
    "discovered": False,
}


def _reset_intended_state():
    _intended_detour_return._state["start"] = None  # type: ignore[attr-defined]
    _intended_detour_return._state["discovered"] = False  # type: ignore[attr-defined]


# ── Solvency: intended WINS every (level, seed) ───────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_detour_return_wins(level, seed):
    _reset_intended_state()
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_detour_return, seed=seed)
    assert res.outcome == "win", (
        f"intended detour+return must WIN on {level} s={seed}; "
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
    territory. The return clause (units_in_region_gte at start)
    is never satisfied ⇒ LOSS on the clock, never a draw."""
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
    """Brute through the picket stack: jeep DPS is insufficient
    vs the 4× e1 bottleneck (medium/hard); the jeep dies to
    return-fire ⇒ own_units_gte:1 clause fires ⇒ LOSS.

    Easy is intentionally permissive (only 2 pickets) — a brute
    play CAN survive on easy (jeep auto-fires in range and may
    walk past the two-cell pocket). The bar is medium/hard."""
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
    seeds 1..4 collectively must produce ≥2 distinct starts. We
    just confirm a stall completes deterministically; the
    spawn-variation contract is enforced cross-seed by
    tests/test_hard_tier.py::test_curated_hard_still_compiles_and_runs."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
