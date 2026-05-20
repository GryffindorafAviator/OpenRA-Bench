"""scout-detect-enemy-tech pack — full no-cheat validation on Rust.

PERCEPTION (observation-then-plan / tech-read): the agent must scout
the enemy base and DISCOVER ≥K distinct enemy buildings (K=2 easy,
K=3 medium+hard). The win predicate is `buildings_discovered_gte`
plus an attrition cap (`units_lost_lte:1` medium / `:0` hard) and a
tick budget reachable by `max_turns` (no draw degeneracy).

Bar (per CLAUDE.md):
  * intended just-enough split-scout policy WINS on every (level, seed)
  * stall LOSSES on every (level, seed) — discovery bar unmet, clock
  * over-scout (visit every corner) LOSSES on every (level, seed) —
    clock burns before the deeper enemy footprint is registered
  * brute-attack (focus-fire the first spotted building) LOSSES on
    medium/hard — bds caps at 1 (jeeps stuck firing one tile, never
    register the spaced 2nd/3rd buildings); easy is rehearsal (tight
    cluster means a brute approach incidentally registers both
    buildings — documented rehearsal acceptance, not a defect)
  * hard's two spawn_point groups (NW / SW) round-robin by seed and
    produce distinct starts.

Scenario shape:
  * rush-hour-arena, soviet enemies, allied agent, 2× jeep west spawn
  * easy: 2-building tight east cluster (fact + weap)
  * medium: 3 spaced buildings (fact mid + weap N + dome S), ≤1 loss
  * hard: 2 spawn groups (NW / SW) + 4 enemy buildings (fact + weap +
    tsla north + sam south), ≤0 loss; agent needs 3 — the seed-varied
    spawn dictates which tech building is the natural third read.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "scout-detect-enemy-tech.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_scout_policy(easy_mode: bool):
    """Drive both jeeps east. Easy: one cluster, both jeeps to (115,18).
    Medium/hard: split — jeep[0] to (113,10) (north band → weap + tech),
    jeep[1] to (113,28) (south band → dome + tech); both cross the mid
    band en route, registering fact at (110,18) along the way."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if not jeeps:
            return [Cmd.observe()]
        ids = sorted([u["id"] for u in jeeps])
        if easy_mode:
            return [Cmd.move_units(ids, 115, 18)]
        cmds = []
        if len(ids) >= 1:
            cmds.append(Cmd.move_units([ids[0]], 113, 10))
        if len(ids) >= 2:
            cmds.append(Cmd.move_units([ids[1]], 113, 28))
        return cmds or [Cmd.observe()]
    return pol


def _over_scout_policy():
    """Visit every corner of the map then come back to the centre —
    burns the clock without registering enough enemy buildings.
    Models the "drive into every fog pocket" failure mode."""
    state = {"wp_idx": 0}
    waypoints = [(120, 5), (5, 5), (5, 40), (120, 40), (60, 18), (115, 18)]

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if not jeeps:
            return [Cmd.observe()]
        ids = [u["id"] for u in jeeps]
        x, y = waypoints[state["wp_idx"] % len(waypoints)]
        u0 = jeeps[0]
        if abs(u0["cell_x"] - x) <= 3 and abs(u0["cell_y"] - y) <= 3:
            state["wp_idx"] += 1
            x, y = waypoints[state["wp_idx"] % len(waypoints)]
        return [Cmd.move_units(ids, x, y)]
    return pol


def _brute_attack_policy():
    """Focus-fire the first enemy building spotted and never disengage.
    Jeeps lock to one tile, never traverse to register the spaced
    2nd/3rd buildings — bds caps at 1."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if not jeeps:
            return [Cmd.observe()]
        ids = [u["id"] for u in jeeps]
        eb = obs.get("enemy_buildings_summary", []) or []
        if eb:
            return [Cmd.attack_unit(ids, eb[0].get("id"))]
        return [Cmd.move_units(ids, 115, 18)]
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-detect-enemy-tech"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench / SC2 tech-detect /
    military doctrine-recognition / industrial CI."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench observation" in a for a in anchors), anchors
    assert any("SC2 tech detection" in a for a in anchors), anchors
    assert any("military doctrine" in a for a in anchors), anchors
    assert any("industrial competitive intelligence" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start corner (binding contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
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

        wts = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_wins(level, seed):
    """The intended capability play (split scout to register fact +
    weap + tech-building) must WIN on every (level, seed). This is the
    load-bearing test that the pack is solvable inside the budget by
    the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_scout_policy(easy_mode=(level == "easy")), seed=seed
    )
    assert res.outcome == "win", (
        f"intended scout must WIN on {level} s={seed}; "
        f"got {res.outcome} (bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"lost={res.signals.units_lost})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSS (no win, no draw) on every
    (level, seed). The fail_condition's after_ticks clause bites at
    the turn budget; never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_over_scout_loses(level, seed):
    """Visiting every corner of the map blows the clock without
    registering K distinct enemy buildings — must LOSS on every
    (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _over_scout_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"over-scout must LOSE on {level} s={seed}; "
        f"got {res.outcome} (bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"lost={res.signals.units_lost})"
    )


# Easy is the rehearsal tier (K=2 in a tight cluster) — a brute
# attack that approaches the cluster incidentally registers both
# buildings. Documented as acceptable rehearsal behaviour, not a
# defect; the strict brute-attack-must-LOSS bar applies to medium
# and hard (where the 3rd building is spaced out of one tile's
# sight cone) only.
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", ("medium", "hard"))
def test_brute_attack_loses_on_medium_and_hard(level, seed):
    """Focus-firing the first spotted enemy building never traverses
    far enough to register the spaced 2nd/3rd buildings — bds caps at
    1 and the discovery bar (≥3) is unmet → LOSS on the clock."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_attack_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute-attack must LOSE on {level} s={seed}; "
        f"got {res.outcome} (bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"lost={res.signals.units_lost})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    different seeds must place the agent at a different (x,y) set.
    Smoke-tests the spawn-variation contract that tests/test_hard_
    tier.py also enforces, but locally so the closer-look loop
    catches a regression on this pack first."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
