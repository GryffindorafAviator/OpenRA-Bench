"""mcv-deploy-second-base — multi-site ramp-up via 2nd MCV.

The bar: the intended drive-MCV-to-eastern-target-and-deploy policy
WINS on every level and every hard seed; stall (only observe),
deploy-next-to-existing-base (just deploy in place at the center,
keeping the new fact OUTSIDE the eastern region), and on hard,
deploy-in-the-wrong-corner (cross the map to the FAR candidate
region — fails the tick budget and/or eats the wrong-corner patrol)
all LOSE on every level. Non-win is a real reachable timeout LOSS.

Validation is scripted (no model / network).
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
PACK_PATH = PACKS / "mcv-deploy-second-base.yaml"


# ── unit-level predicate checks ──────────────────────────────────────

def _ctx(own_buildings=(), tick=1000):
    """Synthesize a WinContext from a building list.
    own_buildings: iterable of (type, cell_x, cell_y).
    """
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


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    base1 = ("fact", 15, 10)  # primary base in NW lobe per YAML
    # WIN: 2 facts, one in the SE expansion disk centred at (95, 30)
    # radius 8 — in time.
    east = ("fact", 95, 30)
    assert evaluate(c.win_condition, _ctx([base1, east], tick=3000))
    # FAIL: only 1 fact (never deployed)
    assert not evaluate(c.win_condition, _ctx([base1], tick=3000))
    # FAIL: 2 facts but second one is in the CENTER (next to MCV start
    # ~ (60,20)) — well outside the east region radius-8 around (95,30).
    next_to_mcv = ("fact", 60, 20)
    assert not evaluate(c.win_condition, _ctx([base1, next_to_mcv], tick=3000))
    # FAIL: 2 facts but second one is in NW corner (wrong region).
    nw = ("fact", 20, 5)
    assert not evaluate(c.win_condition, _ctx([base1, nw], tick=3000))
    # Past deadline ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([base1, east], tick=5402))
    # No fact at all ⇒ fail (lost base #1 AND never deployed).
    assert evaluate(c.fail_condition, _ctx([], tick=3000))
    # Deadline reachable inside max_turns (∼90 ticks/turn).
    assert 5401 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 5401 must be reachable within max_turns"
    )


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    base1 = ("fact", 15, 10)
    east = ("fact", 92, 28)  # inside radius-8 around (95,30)
    assert evaluate(c.win_condition, _ctx([base1, east], tick=2500))
    # Just outside the radius
    outside = ("fact", 95, 39)  # distance 9 > 8 from (95,30)
    assert not evaluate(c.win_condition, _ctx([base1, outside], tick=2500))
    # Timeout reachable
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_either_corner_wins():
    c = compile_level(load_pack(PACK_PATH), "hard")
    base1 = ("fact", 15, 20)
    ne = ("fact", 95, 10)
    se = ("fact", 95, 30)
    # WIN: deploy at NE candidate region
    assert evaluate(c.win_condition, _ctx([base1, ne], tick=2000))
    # WIN: deploy at SE candidate region
    assert evaluate(c.win_condition, _ctx([base1, se], tick=2000))
    # FAIL: deploy at center (neither candidate)
    center = ("fact", 60, 20)
    assert not evaluate(c.win_condition, _ctx([base1, center], tick=2000))
    # FAIL: deploy next to base #1 (still outside both candidate regions)
    near_base = ("fact", 22, 20)
    assert not evaluate(c.win_condition, _ctx([base1, near_base], tick=2000))
    # Past deadline ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([base1, ne], tick=3802))
    # Deadline reachable
    assert 3801 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the staging latitude (NE vs SE
    MCV)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "mcv-deploy-second-base"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "sc2le" in joined and "microrts" in joined
    assert "geographic" in joined or "multi-site" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


# ── engine-driven scripted policies ──────────────────────────────────


def _find_mcv(rs):
    for u in rs.get("units_summary", []) or []:
        if str(u.get("type", "")).lower() == "mcv":
            return u
    return None


def _nearest_target(mcv, level):
    """Return (tx, ty) the deploy-region center the MCV is nearer to.
    Easy/medium win on building_in_region centered at (95,30) radius
    8; hard offers two candidate regions (NE/SE), pick by proximity."""
    if level != "hard":
        return (95, 30)
    if mcv is None:
        return (95, 10)
    # On hard, NE (95,10) vs SE (95,30); pick nearer to MCV start.
    dx_n = (mcv["cell_x"] - 95) ** 2 + (mcv["cell_y"] - 10) ** 2
    dx_s = (mcv["cell_x"] - 95) ** 2 + (mcv["cell_y"] - 30) ** 2
    return (95, 10) if dx_n <= dx_s else (95, 30)


def _intended_policy_for(level):
    """Drive the MCV to the (nearest) target region; deploy when in
    range. After deploy the MCV is consumed and the episode just
    observes."""

    def _policy(rs, Command):
        mcv = _find_mcv(rs)
        if mcv is None:
            return [Command.observe()]
        tx, ty = _nearest_target(mcv, level)
        dx = mcv["cell_x"] - tx
        dy = mcv["cell_y"] - ty
        dist2 = dx * dx + dy * dy
        # Within deploy region (radius 8 in YAML; deploy a bit inside
        # to be safe).
        if dist2 <= 36:
            return [Command.deploy([str(mcv["id"])])]
        return [Command.move_units([str(mcv["id"])], target_x=tx, target_y=ty)]

    return _policy


def _stall_policy(rs, Command):
    return [Command.observe()]


def _deploy_in_place_policy(rs, Command):
    """Deploy MCV immediately wherever it stands (~(60,20) for
    easy/medium, ~(60,10) or (60,30) for hard). The new fact ends up
    well outside the east target region → win predicate fails."""
    mcv = _find_mcv(rs)
    if mcv is None:
        return [Command.observe()]
    return [Command.deploy([str(mcv["id"])])]


def _deploy_next_to_existing_policy(rs, Command):
    """Move MCV BACK toward base #1 (west) and deploy there. The new
    fact ends up near (18,20) — clearly outside the east region."""
    mcv = _find_mcv(rs)
    if mcv is None:
        return [Command.observe()]
    # Once we're nearly on top of the existing west base, deploy.
    if abs(mcv["cell_x"] - 22) <= 1 and abs(mcv["cell_y"] - 20) <= 2:
        return [Command.deploy([str(mcv["id"])])]
    return [Command.move_units([str(mcv["id"])], target_x=22, target_y=20)]


def _deploy_wrong_corner_policy(rs, Command):
    """Hard-only: drive MCV to the OPPOSITE corner (the candidate
    region that is NOT the nearest one) and deploy. The cross-map
    haul + wrong-corner patrol busts the tick budget or kills the
    MCV en route."""

    def _policy(rs, Command):
        mcv = _find_mcv(rs)
        if mcv is None:
            return [Command.observe()]
        # Pick the FARTHER corner.
        dx_n = (mcv["cell_x"] - 95) ** 2 + (mcv["cell_y"] - 10) ** 2
        dx_s = (mcv["cell_x"] - 95) ** 2 + (mcv["cell_y"] - 30) ** 2
        tx, ty = (95, 10) if dx_n > dx_s else (95, 30)
        dx = mcv["cell_x"] - tx
        dy = mcv["cell_y"] - ty
        if dx * dx + dy * dy <= 36:
            return [Command.deploy([str(mcv["id"])])]
        return [Command.move_units([str(mcv["id"])], target_x=tx, target_y=ty)]

    return _policy(rs, Command)


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_policy_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_policy_for(level), seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended deploy-MCV-at-target-region "
            f"should WIN, got {res.outcome} after {res.turns} turns "
            f"(buildings={sorted(res.signals.own_building_types)}, "
            f"own_buildings={res.signals.own_buildings})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE (no 2nd fact ever), "
            f"got {res.outcome} after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_deploy_next_to_existing_base_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _deploy_next_to_existing_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: deploy-next-to-existing-base must "
            f"LOSE (2nd fact not in east region), got {res.outcome} "
            f"after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}"
        )


def test_deploy_wrong_corner_loses_hard():
    """Hard-only: deploying at the FAR candidate region (cross-map
    diagonal) must LOSE — tick budget + wrong-corner patrol attrition
    is the discrimination axis vs the intended NEAREST-corner play."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "hard")
    for s in (1, 2, 3, 4):
        res = run_level(c, _deploy_wrong_corner_policy, seed=s)
        assert res.outcome == "loss", (
            f"hard seed={s}: deploy-in-wrong-corner must LOSE "
            f"(tick budget blown or MCV killed by far-corner patrol), "
            f"got {res.outcome} after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the level's after_ticks fail must fit
    inside max_turns on every level (∼90 ticks/turn)."""
    pack = load_pack(PACK_PATH)
    bars = {"easy": 5401, "medium": 4501, "hard": 3801}
    for lvl, bar in bars.items():
        c = compile_level(pack, lvl)
        assert bar <= 93 + 90 * (c.max_turns - 1), lvl
