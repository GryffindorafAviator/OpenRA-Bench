"""expansion-aggro-3-base-greedy — Wave-4 Group B AGGRO triple expansion.

Scripted-policy no-defect / no-cheat validation. The pack hands the
agent THREE MCVs at a NW staging zone (no starter base) and asks the
agent to commit ALL three to three DISTINCT eastern target regions
(NE / E / SE). The discrimination is whether the model embraces the
GREEDY commit — holding one MCV back as "defence reserve" forfeits a
region and LOSES; stalling produces no facts and LOSES via the early
pre-fail.

The intended bar (CLAUDE.md): every lazy / hedged / stall policy
LOSES on every level and every hard seed (1–4); only the
deploy-all-distinct policy WINS.

  - stall                              (only Command.observe())  → LOSS
  - deploy-1-only                      (commit one MCV, hold two) → LOSS
  - deploy-all-bunched-at-NW           (deploy in place; medium /
    hard MCVs sit at NW so facts land at NW, every region predicate
    fails)                                                         → LOSS
  - intended-deploy-all-distinct       (one MCV per region)        → WIN
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "expansion-aggro-3-base-greedy.yaml"

# Medium / hard target region centres (see pack YAML). Deploy at (mx,
# my) converts to a fact at (mx-1, my-1), so to land a fact at (cx,
# cy) the MCV must be driven to (cx+1, cy+1).
REGIONS_MED = [(165, 15), (165, 40), (165, 65)]
MCV_TARGETS_MED = [(cx + 1, cy + 1) for (cx, cy) in REGIONS_MED]
# Easy uses only 2 regions (degenerate-to-balanced variant).
REGIONS_EASY = [(165, 15), (165, 65)]
MCV_TARGETS_EASY = [(cx + 1, cy + 1) for (cx, cy) in REGIONS_EASY]

LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


def _targets_for(level: str):
    return MCV_TARGETS_EASY if level == "easy" else MCV_TARGETS_MED


# ── scripted policies ─────────────────────────────────────────────────


def stall(rs, Command):
    """Do nothing — the early pre-fail at tick 3500 (medium) / 3200
    (hard) with <1 fact bites; easy stalls all the way to the
    after_ticks deadline. Real reachable LOSS, never a DRAW."""
    return [Command.observe()]


def bunch_in_place(rs, Command):
    """Deploy every MCV at its current cell. On EASY the MCVs sit
    inside their target regions by design, so this WINS easy — easy's
    decision is solely "deploy at all". On MEDIUM and HARD the MCVs
    cluster at NW (or SW on hard's spawn 1), so the resulting facts
    land in the staging zone, OUTSIDE every target region → all three
    `building_in_region` predicates fail → LOSS."""
    cmds = []
    for u in rs.get("units_summary") or []:
        if str(u.get("type", "")).lower() == "mcv":
            cmds.append(Command.deploy([str(u["id"])]))
    return cmds or [Command.observe()]


def make_deploy_one_only(level: str):
    """Hedged-commit policy: send the FIRST MCV to the FIRST target,
    leave the rest parked. The other region predicates never satisfy
    → LOSS on the deadline, on every tier and every seed."""
    targets = _targets_for(level)
    state = {"assigned_to": None, "deployed": False}

    def fn(rs, Command):
        units = rs.get("units_summary") or []
        mcvs = sorted(
            [u for u in units if str(u.get("type", "")).lower() == "mcv"],
            key=lambda u: u["id"],
        )
        if not mcvs:
            return [Command.observe()]
        chosen = mcvs[0]
        if state["assigned_to"] is None:
            state["assigned_to"] = targets[0]
        tx, ty = state["assigned_to"]
        if abs(chosen["cell_x"] - tx) <= 1 and abs(chosen["cell_y"] - ty) <= 1:
            if not state["deployed"]:
                state["deployed"] = True
                return [Command.deploy([str(chosen["id"])])]
            return [Command.observe()]
        return [Command.move_units([str(chosen["id"])], tx, ty)]

    return fn


def make_intended(level: str):
    """The intended policy: assign each MCV to a DISTINCT target
    region (nearest-unused), move it there, deploy. Easy uses 2
    regions; medium / hard use all 3. After all deploys the agent
    owns the required number of facts AND each target region holds
    one → WIN."""
    targets = _targets_for(level)
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
            cands = [t for t in targets if t not in used]
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
def test_intended_deploy_all_distinct_wins(level, seed):
    """The intended GREEDY commit: one MCV per region. Must WIN on
    every level and every hard seed (1–4) — the no-defect bar."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_intended(level), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed={seed}: intended-deploy-all-distinct must WIN; "
        f"got {res.outcome} (tick={res.signals.game_tick} "
        f"buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_stall_loses(level, seed):
    """Do-nothing: no MCV ever deploys. easy times out on the
    deadline; medium/hard hit the pre-fail at tick 3500/3200. Real
    reachable LOSS, never a DRAW."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_deploy_one_only_loses(level, seed):
    """Hedged-commit: deploy only the FIRST MCV. Remaining regions
    are never covered → at least one `building_in_region` predicate
    fails → LOSS, on every tier and every seed."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_deploy_one_only(level), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: deploy-one-only must LOSE; got "
        f"{res.outcome} (buildings={res.signals.own_buildings})"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_bunch_at_staging_loses_on_medium_and_hard(level, seed):
    """On medium/hard the MCVs cluster at NW (or SW on hard's spawn 1)
    — deploying in place lands every resulting fact in the staging
    zone, OUTSIDE all three target regions, so every
    `building_in_region` predicate fails → LOSS. (Easy MCVs sit in
    their regions by design, so bunch-in-place wins easy; easy tests
    only the "deploy at all" decision, not the distribution
    capability.)"""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, bunch_in_place, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: bunch-in-place must LOSE; got "
        f"{res.outcome} (buildings={res.signals.own_buildings})"
    )


def test_within_ticks_is_reachable_per_tier():
    """Tick/turn alignment (CLAUDE.md): the within_ticks deadline must
    be inside max_turns (tick ≤ 93 + 90·(max_turns − 1)) — otherwise a
    staller DRAWS instead of LOSING. Re-derived per tier."""
    pack = load_pack(PACK)
    for lvl, deadline in (("easy", 5400), ("medium", 5400), ("hard", 4800)):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert deadline <= max_tick, (
            f"{lvl}: within_ticks={deadline} > reachable max_tick="
            f"{max_tick} (would draw instead of losing)"
        )


def test_hard_has_two_spawn_groups_with_distinct_starts():
    """Hard-tier curation contract: ≥2 agent spawn_point groups, and
    the seeds (1, 2, 3, 4) must actually produce ≥2 distinct starts
    on the live engine (MCV cluster rotates NW ↔ SW). Mirrors
    `tests/test_hard_tier.py` for early signal."""
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
        "SC2 greedy 3-base macro / fastest expansion timing",
        "startup blitzscale: skip margin for growth",
        "aggressive market-share land-grab",
        "growth-vs-defence tradeoff",
    }
    assert required.issubset(anchors), (
        f"missing benchmark anchors: {required - anchors}"
    )


def test_pack_compiles_with_arena_base_map():
    """The pack-level base_map is a known-supported sentinel; the
    real generator-spec arena is materialised per-level via
    `overrides.base_map`. All three levels must compile onto the
    generated arena."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.scenario.base_map == "expansion-aggro-3-base-greedy-arena", (
            f"{lvl}: expected expansion-aggro-3-base-greedy-arena, got "
            f"{c.scenario.base_map}"
        )
        assert c.map_supported, (
            f"{lvl}: expansion-aggro-3-base-greedy-arena must resolve "
            "to a real .oramap (mapgen materialise should have written it)"
        )
