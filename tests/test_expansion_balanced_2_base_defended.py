"""expansion-balanced-2-base-defended — Wave-4 Group B BALANCED 2-base expansion.

Scripted-policy no-defect / no-cheat validation. The pack hands the
agent a NW starter base (fact+tent+powr) plus a spare MCV at map
centre. The BALANCED bar is: (1) deploy the MCV at the eastern
target region for the 2nd fact, (2) build a pbox at EACH base, and
(3) train ≥3 defenders. Light patrols at BOTH bases enforce the
defend-BOTH discrimination.

The intended bar (CLAUDE.md): every lazy / greedy / turtle / stall
policy LOSES on every level and every hard seed (1–4); only the
balanced-expand-and-defend policy WINS.

  - stall                          (only Command.observe())     → LOSS
  - greedy-2-facts-no-defence      (deploy + skip pbox/units)   → LOSS
  - turtle-1-fact-fortified        (pbox+units, skip deploy)    → LOSS
  - intended-2-facts-defended      (deploy + pbox/base + 3 e1)  → WIN
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "expansion-balanced-2-base-defended.yaml"

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)

# Per-level deploy targets (MCV must reach (cx+1, cy+1) so the fact
# lands at the region centre). Hard accepts EITHER NE or SE.
TARGETS = {
    "easy":   [(131, 16)],
    "medium": [(131, 16)],
    "hard":   [(131, 16), (131, 46)],
}

# Pbox placement coords near NW base (always) and near each
# candidate target region (chosen per seed on hard).
PBOX_NW = (17, 22)
PBOX_NW_2 = (17, 24)
PBOX_NEAR_NE = (128, 18)
PBOX_NEAR_SE = (128, 48)

# Required pbox count per tier (matches the win predicate).
PBOX_BAR = {"easy": 1, "medium": 2, "hard": 3}

# Required own_units (e1) for the units_gte clause.
UNIT_BAR = {"easy": 3, "medium": 3, "hard": 4}


def _pbox_cells_for(level: str, deploy_target):
    """Pick pbox cells: 1 at NW + 1 at the chosen new base for
    easy/medium; for hard add a second NW pbox to hit the n:3 bar."""
    near = PBOX_NEAR_NE if deploy_target == (131, 16) else PBOX_NEAR_SE
    if level == "easy":
        return [PBOX_NW]
    if level == "medium":
        return [PBOX_NW, near]
    # hard: 1 at NW + 1 near new base + 1 second NW
    return [PBOX_NW, near, PBOX_NW_2]


# ── scripted policies ─────────────────────────────────────────────────


def stall(rs, Command):
    """Do nothing. No fact #2, no pbox, no extra units ⇒ LOSS on
    the deadline. Persistent enemy `fact` marker keeps the episode
    alive past any patrol-elimination auto-`done`."""
    return [Command.observe()]


def _mcv_id(rs):
    for u in rs.get("units_summary") or []:
        if str(u.get("type", "")).lower() == "mcv":
            return str(u["id"]), int(u["cell_x"]), int(u["cell_y"])
    return None


def _pbox_count(rs):
    return sum(
        1 for b in (rs.get("own_buildings") or [])
        if str(b.get("type", "")).lower() == "pbox"
    )


def _fact_count(rs):
    return sum(
        1 for b in (rs.get("own_buildings") or [])
        if str(b.get("type", "")).lower() == "fact"
    )


def _own_unit_count(rs):
    return len(rs.get("units_summary") or [])


def _e1_count(rs):
    return sum(
        1 for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "e1"
    )


def make_greedy_2_facts_no_defence(level: str):
    """Greedy expand-only: drive the MCV to the (nearest) target,
    deploy, then idle. No pbox, no extra units — `pbox` and `units`
    clauses never satisfy ⇒ LOSS."""
    state = {"target": None, "deployed": False}

    def fn(rs, Command):
        m = _mcv_id(rs)
        if m is None:
            return [Command.observe()]
        mid, mx, my = m
        if state["target"] is None:
            targets = TARGETS[level]
            state["target"] = min(
                targets,
                key=lambda t: (t[0] - mx) ** 2 + (t[1] - my) ** 2,
            )
        tx, ty = state["target"]
        if abs(mx - tx) <= 1 and abs(my - ty) <= 1:
            if not state["deployed"]:
                state["deployed"] = True
                return [Command.deploy([mid])]
            return [Command.observe()]
        return [Command.move_units([mid], tx, ty)]

    return fn


def make_turtle_1_fact_fortified(level: str):
    """Turtle: keep the MCV parked (never deploy) and fortify NW with
    every pbox + e1 the budget allows. Satisfies pbox + units bars
    but `building_count_gte:fact,n:2` never satisfies ⇒ LOSS on
    every tier and every seed."""
    bar_units = UNIT_BAR[level]
    bar_pbox = PBOX_BAR[level]
    # Turtle places ALL pbox at base #1 cluster (no spread to base #2).
    turtle_cells = [(17, 22), (17, 24), (12, 22), (12, 24)][: bar_pbox]

    def fn(rs, Command):
        cmds = []
        # Train e1 until we hit the unit bar.
        if _e1_count(rs) < bar_units:
            prod = rs.get("production", []) or []
            if "e1" not in prod:
                cmds.append(Command.build("e1"))
        # Build/place pbox until we hit the pbox bar.
        n_placed = _pbox_count(rs)
        if n_placed < bar_pbox:
            prod = rs.get("production", []) or []
            if "pbox" not in prod:
                cmds.append(Command.build("pbox"))
            tx, ty = turtle_cells[n_placed]
            cmds.append(Command.place_building("pbox", tx, ty))
        return cmds or [Command.observe()]

    return fn


def make_intended(level: str):
    """Intended BALANCED play: drive the MCV to the (nearest) target
    region and deploy ⇒ 2 facts; in parallel build pbox at the NW
    base AND near the chosen new base, and train ≥N e1. All clauses
    satisfied ⇒ WIN."""
    bar_units = UNIT_BAR[level]
    state = {"target": None, "deployed": False, "pbox_cells": None}

    def fn(rs, Command):
        cmds = []
        # ── (1) MCV → target → deploy ──
        m = _mcv_id(rs)
        if m is not None:
            mid, mx, my = m
            if state["target"] is None:
                targets = TARGETS[level]
                state["target"] = min(
                    targets,
                    key=lambda t: (t[0] - mx) ** 2 + (t[1] - my) ** 2,
                )
                state["pbox_cells"] = _pbox_cells_for(level, state["target"])
            tx, ty = state["target"]
            if abs(mx - tx) <= 1 and abs(my - ty) <= 1:
                if not state["deployed"]:
                    state["deployed"] = True
                    cmds.append(Command.deploy([mid]))
            else:
                cmds.append(Command.move_units([mid], tx, ty))
        # If target not yet resolved (MCV gone) default to NE cells.
        pbox_cells = state["pbox_cells"] or _pbox_cells_for(level, (131, 16))
        # ── (2) train e1 in parallel ──
        if _e1_count(rs) < bar_units:
            prod = rs.get("production", []) or []
            if "e1" not in prod:
                cmds.append(Command.build("e1"))
        # ── (3) pbox at each base site in parallel ──
        n_placed = _pbox_count(rs)
        if n_placed < len(pbox_cells):
            prod = rs.get("production", []) or []
            if "pbox" not in prod:
                cmds.append(Command.build("pbox"))
            tx, ty = pbox_cells[n_placed]
            cmds.append(Command.place_building("pbox", tx, ty))
        return cmds or [Command.observe()]

    return fn


# ── tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_intended_balanced_wins(level, seed):
    """The intended BALANCED play (deploy + pbox at each base +
    3 e1) must WIN on every level and every hard seed (1–4)."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_intended(level), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed={seed}: intended must WIN; got {res.outcome} "
        f"(tick={res.signals.game_tick} buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_stall_loses(level, seed):
    """Stall: no deploy, no pbox, no e1 ⇒ deadline LOSS."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE; got {res.outcome}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_greedy_2_facts_no_defence_loses(level, seed):
    """Greedy: deploy then idle. `pbox` / `units` clauses never
    satisfy ⇒ LOSS, every tier, every seed."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_greedy_2_facts_no_defence(level), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: greedy-2-facts-no-defence must LOSE; "
        f"got {res.outcome} (buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_turtle_1_fact_fortified_loses(level, seed):
    """Turtle: fortify NW only, never deploy. `fact,n:2` clause
    never satisfies ⇒ LOSS, every tier, every seed."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_turtle_1_fact_fortified(level), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: turtle-1-fact-fortified must LOSE; "
        f"got {res.outcome} (buildings={res.signals.own_buildings})"
    )


def test_within_ticks_is_reachable_per_tier():
    """Tick/turn alignment (CLAUDE.md): the within_ticks deadline
    must be inside max_turns (tick ≤ 93 + 90·(max_turns − 1)) —
    otherwise a staller DRAWS instead of LOSING."""
    pack = load_pack(PACK)
    for lvl, deadline in (("easy", 5400), ("medium", 5400), ("hard", 4800)):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert deadline <= max_tick, (
            f"{lvl}: within_ticks={deadline} > reachable max_tick="
            f"{max_tick} (would draw instead of losing)"
        )


def test_hard_has_two_spawn_groups_with_distinct_starts():
    """Hard-tier curation contract: ≥2 agent spawn_point groups and
    seeds (1, 2, 3, 4) produce ≥2 distinct starts on the live engine.
    Mirrors `tests/test_hard_tier.py` for early signal."""
    from pathlib import Path

    from openra_bench.eval_core import _scenario_to_tmp_yaml, RustEnvPool
    from openra_bench.rust_adapter import RustObsAdapter

    pack = load_pack(PACK)
    c = compile_level(pack, "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )

    starts = set()
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        for seed in HARD_SEEDS:
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            mcvs = sorted(
                (x["cell_x"], x["cell_y"])
                for x in u
                if str(x.get("type", "")).lower() == "mcv"
            )
            starts.add(tuple(mcvs))
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds {HARD_SEEDS} produced identical MCV starts "
        f"{starts}; spawn_point round-robin not taking effect"
    )


def test_meta_benchmark_anchor_carries_required_strings():
    """The pack must declare the four benchmark / real-world anchors
    the brief required."""
    pack = load_pack(PACK)
    anchors = set(pack.meta.benchmark_anchor)
    required = {
        "SC2 standard 2-base macro / safe-expand timing",
        "sustainable growth: expand AND defend",
        "product expansion-with-quality: new market + retention",
        "balanced macro: growth+defense",
    }
    assert required.issubset(anchors), (
        f"missing benchmark anchors: {required - anchors}"
    )


def test_pack_compiles_to_arena():
    """All three levels materialise the 160x60 arena via overrides.base_map."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported, f"{lvl}: arena base_map must resolve"
