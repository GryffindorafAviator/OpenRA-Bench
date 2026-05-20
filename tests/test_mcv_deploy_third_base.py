"""mcv-deploy-third-base — Group A seed (Mid-Late: 3rd MCV → 3rd base).

Tri-region operational footprint scripted-policy validation. The pack
declares THREE MCVs and a single existing west `fact`; the win
predicate requires ≥3 facts AND a fact INSIDE each of three distinct
target regions (NE / SE / S-CENTER) before tick 6300. The discriminator
is committing each MCV to a DISTINCT region — bunching all three at
the base satisfies `building_count_gte` but fails all three region
predicates; deploying only 1-2 leaves at least one region uncovered;
stalling produces no new facts at all.

This file proves the no-defect / no-cheat bar at scripted level (no
model, no network) across seeds 1–4 and across easy / medium / hard:

  - stall  (only Command.observe())                          → LOSS
  - bunch-all-at-base (deploy in place; on medium/hard MCVs
    start away from the target regions, so the resulting facts
    land in the west cluster, NOT in any target region)     → LOSS
  - deploy-only-two-regions (cover NE+SE, abandon S-CENTER)  → LOSS
  - intended-3-regions (move each MCV to a distinct region
    and deploy)                                              → WIN
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mcv-deploy-third-base.yaml"

# Three target region centers (see pack YAML). Deploy converts an MCV
# at (mx, my) into a fact at (mx-1, my-1), so to land a fact exactly
# at (cx, cy) the MCV must be at (cx+1, cy+1).
REGIONS = [(130, 15), (130, 65), (75, 70)]
MCV_TARGETS = [(cx + 1, cy + 1) for (cx, cy) in REGIONS]

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def stall(rs, Command):
    """Do nothing — the deadline (and the early pre-fail clause at
    tick 4000 with <2 facts) emits a real LOSS."""
    return [Command.observe()]


def bunch_in_place(rs, Command):
    """Deploy every MCV at its current cell. On EASY the MCVs already
    sit inside their target regions (so this also wins easy — easy's
    decision is solely "deploy all three, not zero"). On MEDIUM and
    HARD the MCVs cluster at the west (or east) starter base and the
    resulting facts land OUTSIDE every target region, so the pack's
    three `building_in_region` predicates all fail → LOSS."""
    cmds = []
    for u in rs.get("units_summary") or []:
        if str(u.get("type", "")).lower() == "mcv":
            cmds.append(Command.deploy([str(u["id"])]))
    return cmds or [Command.observe()]


def make_deploy_only_two():
    """Send MCVs to the first TWO regions, never the third. The
    third region's `building_in_region` predicate is never satisfied
    → LOSS on the deadline. Works on all tiers."""
    state = {"assigned": {}, "deployed": set()}
    # Only two of the three targets are used.
    targets_pool = [MCV_TARGETS[0], MCV_TARGETS[1]]

    def fn(rs, Command):
        units = rs.get("units_summary") or []
        mcvs = sorted(
            [u for u in units if str(u.get("type", "")).lower() == "mcv"],
            key=lambda u: u["id"],
        )
        # Stable id→target assignment, nearest-unused-target.
        for u in mcvs:
            if u["id"] not in state["assigned"] and len(state["assigned"]) < len(
                targets_pool
            ):
                used = set(state["assigned"].values())
                cands = [t for t in targets_pool if t not in used]
                if not cands:
                    continue
                t = min(
                    cands,
                    key=lambda t: (t[0] - u["cell_x"]) ** 2
                    + (t[1] - u["cell_y"]) ** 2,
                )
                state["assigned"][u["id"]] = t
        cmds = []
        for u in mcvs:
            tgt = state["assigned"].get(u["id"])
            if tgt is None:
                continue
            tx, ty = tgt
            if abs(u["cell_x"] - tx) <= 1 and abs(u["cell_y"] - ty) <= 1:
                if u["id"] not in state["deployed"]:
                    cmds.append(Command.deploy([str(u["id"])]))
                    state["deployed"].add(u["id"])
            else:
                cmds.append(Command.move_units([str(u["id"])], tx, ty))
        return cmds or [Command.observe()]

    return fn


def make_intended_three_regions():
    """The intended policy: assign each MCV to a DISTINCT target
    region (nearest-unused), move it there, deploy. After all three
    MCVs deploy, the agent owns ≥3 facts (existing west + 3 new)
    AND each target region holds one fact → WIN."""
    state = {"assigned": {}, "deployed": set()}

    def fn(rs, Command):
        units = rs.get("units_summary") or []
        mcvs = sorted(
            [u for u in units if str(u.get("type", "")).lower() == "mcv"],
            key=lambda u: u["id"],
        )
        for u in mcvs:
            if u["id"] in state["assigned"]:
                continue
            used = set(state["assigned"].values())
            cands = [t for t in MCV_TARGETS if t not in used]
            if not cands:
                continue
            t = min(
                cands,
                key=lambda t: (t[0] - u["cell_x"]) ** 2
                + (t[1] - u["cell_y"]) ** 2,
            )
            state["assigned"][u["id"]] = t
        cmds = []
        for u in mcvs:
            tgt = state["assigned"].get(u["id"])
            if tgt is None:
                continue
            tx, ty = tgt
            if abs(u["cell_x"] - tx) <= 1 and abs(u["cell_y"] - ty) <= 1:
                if u["id"] not in state["deployed"]:
                    cmds.append(Command.deploy([str(u["id"])]))
                    state["deployed"].add(u["id"])
            else:
                cmds.append(Command.move_units([str(u["id"])], tx, ty))
        return cmds or [Command.observe()]

    return fn


# ── tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_intended_three_regions_wins_every_level_every_seed(level, seed):
    """The intended commit-each-MCV-to-its-own-region policy must WIN
    on every level and every hard seed (1–4) — the no-defect bar."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_intended_three_regions(), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed={seed}: intended-3-regions must WIN; got "
        f"{res.outcome} (tick={res.signals.game_tick} "
        f"buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_stall_loses_every_level_every_seed(level, seed):
    """The do-nothing policy: no MCV ever deploys → no new facts →
    the early pre-fail (`after_ticks: 4000` AND <2 facts) bites well
    before the deadline. Real reachable LOSS, never a DRAW."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_bunch_at_base_loses_on_medium_and_hard(level, seed):
    """On medium and hard the MCVs cluster at the west (or east, on
    hard's seed-rotated group) starter base — deploying in place
    lands every resulting fact OUTSIDE all three target regions, so
    every `building_in_region` predicate fails and the run LOSES on
    the deadline. (Easy MCVs already sit in their regions by design,
    so bunch-in-place wins easy — the easy decision is solely "do
    you deploy at all"; only medium/hard test the distribution
    capability.)"""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, bunch_in_place, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: bunch-in-place must LOSE; got "
        f"{res.outcome} (buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_deploy_only_two_regions_loses(level, seed):
    """Partial coverage policy: only the FIRST TWO target regions get
    a fact; the THIRD is abandoned. The third region's
    `building_in_region` predicate is never satisfied → LOSS, on
    every tier and every seed."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_deploy_only_two(), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: deploy-only-two-regions must LOSE; "
        f"got {res.outcome} (buildings={res.signals.own_buildings})"
    )


def test_within_ticks_is_reachable_per_tier():
    """Tick/turn alignment: the within_ticks deadline must be inside
    max_turns (tick ≤ 93 + 90·(max_turns − 1)) — otherwise a staller
    DRAWS instead of LOSING. Re-derived per tier."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 6300 <= max_tick, (
            f"{lvl}: within_ticks=6300 > reachable max_tick="
            f"{max_tick} (would draw instead of losing)"
        )


def test_hard_has_two_spawn_groups_with_distinct_starts():
    """Hard-tier curation contract: ≥2 agent spawn_point groups, and
    the seeds (1, 2, 3, 4) must actually produce ≥2 distinct starts
    on the live engine (the whole MCV CLUSTER rotates between WEST
    and EAST). Mirrors `tests/test_hard_tier.py` for early signal."""
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
    the brief required (SC2LE / MicroRTS / distribution-network /
    tri-region operational footprint)."""
    pack = load_pack(PACK)
    anchors = set(pack.meta.benchmark_anchor)
    required = {
        "SC2LE 3-base macro",
        "MicroRTS multi-front expansion",
        "distribution-network multi-warehouse planning",
        "tri-region operational footprint",
    }
    assert required.issubset(anchors), (
        f"missing benchmark anchors: {required - anchors}"
    )


def test_pack_compiles_with_arena_base_map():
    """The pack-level base_map is a known-supported sentinel; the
    real generator-spec arena is materialised per-level via
    `overrides.base_map` (the documented idiom — a Level-level
    `base_map` outside `overrides:` is silently ignored). All three
    levels must compile onto the generated arena."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.scenario.base_map == "mcv-deploy-third-base-arena", (
            f"{lvl}: expected mcv-deploy-third-base-arena, got "
            f"{c.scenario.base_map}"
        )
        assert c.map_supported, (
            f"{lvl}: mcv-deploy-third-base-arena must resolve to a "
            "real .oramap (mapgen materialise should have written it)"
        )
