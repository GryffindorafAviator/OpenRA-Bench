"""scout-map-reveal-percent-target — perception coverage pack.

The perception cell whose win clause is `reveal X% of the map within
a tick budget` (with `units_lost_lte:0` making the preservation
contract explicit — there is no enemy in the pack, so units_lost is
structurally zero). Distinct from the perception-frontier-reading
family, which scores discrete `buildings_discovered_gte` of hidden
landmarks; this pack scores the continuous `explored_pct_gte`
coverage signal — the agent must spread the scout force across
disjoint regions so the union of sight cones meets the percentage in
time. Anchors: SC2 map-control / vision objectives, ERQA coverage,
military area dominance, drone-search coverage planning.

Scripted policies cover the four bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall                       → LOSS (clock; ~4% revealed)
  * single-jeep-only (others    → LOSS (asymptote ≈38% covers neither
    idle)                              medium nor hard targets)
  * bunched (all jeeps one      → LOSS (one-swath asymptote caps at
    target)                            ≈50.4% on easy/medium and
                                       ≈68.6% on hard — under the
                                       70% hard bar and outside the
                                       easy/medium clock)
  * intended split-and-         → WIN (every level, seeds 1..4)
    distribute (jeeps fan to
    distinct quadrant corners,
    multi-stage on hard)

Hard tier additionally rotates the agent spawn between SW (5,20..36)
and NW (5,4..20) by seed; the intended policy fans the 5 jeeps to
the four distant corners + a centre diagonal regardless of which
column they spawn in, then redirects to a second-stage target once
the first is reached.
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
    / "scout-map-reveal-percent-target.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-map-reveal-percent-target"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Seed-taxonomy contract: the anchors must call out the
    SC2 map-control / ERQA / drone-coverage framing."""
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2 map-control" in a for a in anchors), anchors
    assert any("ERQA coverage" in a for a in anchors), anchors
    assert any("area dominance" in a or "drone search" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout."""
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
    """within_ticks / after_ticks must be reachable inside max_turns.
    Engine advances ~90 ticks/turn → reachable max = 93 + 90·(N-1).
    A within_ticks above the reachable tick is INERT (no anti-stall
    teeth ⇒ draw degeneracy)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        fail = compile_level(pack, lvl).fail_condition.model_dump(exclude_none=True)

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

        ats: list[int] = []
        _collect(fail, "after_ticks", ats)
        assert ats, f"{lvl} has no after_ticks fail leaf"
        for at in ats:
            assert at <= reachable, (
                f"{lvl} after_ticks={at} > reachable={reachable} "
                f"(max_turns={max_turns}) — fail never bites ⇒ draw"
            )


def test_explored_pct_clause_present():
    """The defining feature of this pack: the win must enforce
    explored_pct_gte (the coverage objective)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        found: list = []

        def _walk(node):
            if isinstance(node, dict):
                if "explored_pct_gte" in node:
                    found.append(node["explored_pct_gte"])
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(win)
        assert found, f"{lvl}: missing explored_pct_gte coverage clause"
    # Difficulty axis: easy < medium < hard target.
    easy_pct = compile_level(pack, "easy").win_condition.model_dump(
        exclude_none=True
    )["all_of"][0]["explored_pct_gte"]
    med_pct = compile_level(pack, "medium").win_condition.model_dump(
        exclude_none=True
    )["all_of"][0]["explored_pct_gte"]
    hard_pct = compile_level(pack, "hard").win_condition.model_dump(
        exclude_none=True
    )["all_of"][0]["explored_pct_gte"]
    assert easy_pct < med_pct < hard_pct, (easy_pct, med_pct, hard_pct)


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _bunched(rs, Command):
    """All jeeps to ONE target — the union footprint is one swath,
    not the map. With the pack map at 32x48 the previous x=125
    targets were silently OOB (jeeps stuck at spawn). Use a near
    in-bounds cell so bunched stays well under the (now-tighter)
    easy bar of 48%."""
    units = sorted(rs.get("units_summary", []) or [], key=lambda x: x["id"])
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=8, target_y=24)
        )
    return cmds


def _single_jeep_only(rs, Command):
    """One jeep scouts to a far corner; the rest are stopped/idle.
    With the easy bar raised to 48% the single-jeep asymptote stays
    strictly under every tier's bar."""
    units = sorted(rs.get("units_summary", []) or [], key=lambda x: x["id"])
    if not units:
        return [Command.observe()]
    cmds = [
        Command.move_units([str(units[0]["id"])], target_x=28, target_y=5)
    ]
    for u in units[1:]:
        cmds.append(Command.stop([str(u["id"])]))
    return cmds


def _intended_split_distribute(rs, Command):
    """The intended policy: fan the jeeps to as many distinct
    quadrant corners as we have, with a multi-stage redirect on
    hard's 5-jeep roster (so the second-stage target sweeps a
    disjoint quadrant from the first). Map is 32×48 — quadrant
    corners pinned inside the cordon.
    """
    units = sorted(rs.get("units_summary", []) or [], key=lambda x: x["id"])
    n = len(units)
    if n == 0:
        return [Command.observe()]
    plan4 = [(28, 4), (28, 44), (16, 4), (16, 44)]
    plan5_stage1 = [(28, 4), (28, 44), (16, 4), (16, 44), (22, 24)]
    plan5_stage2 = [(16, 44), (16, 4), (28, 44), (28, 4), (22, 4)]
    cmds = []
    for i, u in enumerate(units):
        ux, uy = u["cell_x"], u["cell_y"]
        uid = str(u["id"])
        if n <= 4:
            tx, ty = plan4[i % 4]
        else:
            wp1 = plan5_stage1[i % 5]
            wp2 = plan5_stage2[i % 5]
            d1 = ((ux - wp1[0]) ** 2 + (uy - wp1[1]) ** 2) ** 0.5
            tx, ty = wp2 if d1 < 6 else wp1
        cmds.append(Command.move_units([uid], target_x=tx, target_y=ty))
    return cmds


# ── Solvency: intended WINS every (level, seed) ───────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_split_distribute_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_split_distribute, seed=seed)
    assert res.outcome == "win", (
        f"intended split-distribute must WIN on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick} "
        f"pct={res.signals.explored_percent:.1f}% "
        f"lost={res.signals.units_lost}"
    )


# ── Stability: every laziest play is a real reachable LOSS ────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"pct={res.signals.explored_percent:.1f}%"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_single_jeep_only_loses(level, seed):
    """A single jeep cannot cover the map alone — the asymptote
    is ≈38-40%, strictly under medium (50%) and hard (70%). On
    easy (30%) the budget is tight enough that the lone jeep
    can't reach 30% by tick 453 (it crosses 30% only at tick 723,
    past the deadline) ⇒ LOSS."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _single_jeep_only, seed=seed)
    assert res.outcome == "loss", (
        f"single-jeep-only must LOSE on {level} s={seed}; "
        f"got {res.outcome} pct={res.signals.explored_percent:.1f}%"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_bunched_loses(level, seed):
    """All jeeps to one destination — they reveal one swath, not
    the map. Asymptote caps at ≈50.4% (4-jeep easy/medium) or
    ≈68.6% (5-jeep hard). On medium the bunched policy reaches
    50.4% only at tick 903, past the within_ticks=825 deadline.
    On hard the bunched asymptote (68.6%) is strictly under the
    70% target ⇒ LOSS on every level."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _bunched, seed=seed)
    assert res.outcome == "loss", (
        f"bunched must LOSE on {level} s={seed}; got {res.outcome} "
        f"pct={res.signals.explored_percent:.1f}% "
        f"tick={res.signals.game_tick}"
    )


# ── Spawn variation contract (hard) ───────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_run_deterministically(seed):
    """Hard's two spawn_point groups (SW vs NW columns) must
    round-robin by seed. We confirm a stall completes
    deterministically; the cross-seed start-distinctness contract
    is enforced by
    tests/test_hard_tier.py::test_curated_hard_still_compiles_and_runs."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
