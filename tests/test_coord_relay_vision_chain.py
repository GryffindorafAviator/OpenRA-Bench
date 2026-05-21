"""coord-relay-vision-chain — ACTION vision-relay-chain pack.

The pack tests SCOUT CHAIN SPACING (military relay chain / sensor-
network coverage / communications-relay anchor): the agent holds 4
jeep scouts and a construction yard and must keep a distant objective
OBSERVED by spacing the jeeps into a relay chain — one jeep parked on
each intermediate relay region plus one parked on the far objective.
The win is an `all_of` of `units_in_region_gte` clauses (each relay
leg + the far leg) plus `enemies_discovered_gte` (the far objective is
genuinely seen) — evaluated every tick, so every region must be
occupied SIMULTANEOUSLY.

Bar (binding):
- intended SPACED RELAY CHAIN (one jeep per relay region + one on the
  far objective) WINS on every level + every hard seed (1..4);
- STALL (observe-only) LOSES on every level + every seed — no region
  is ever occupied, the deadline expires;
- ONE-SCOUT-FAR (race one jeep to the far objective, others idle)
  LOSES on every level + every seed — the intermediate relay regions
  stay empty;
- BUNCHED-SCOUTS (drive all 4 jeeps together to one spot) LOSES on
  every level + every seed — they occupy at most one region at a
  time, never the relays AND the objective together;
- non-win is a real reachable timeout LOSS (the fail `after_ticks` =
  `within_ticks` + 1, both below the per-tier tick ceiling) — never a
  DRAW;
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.eval_core import run_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "coord-relay-vision-chain.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── 1) declarative / schema invariants (no engine needed) ──────────────


def test_pack_loads_and_three_levels_compile():
    p = load_pack(PACK)
    assert p.meta.id == "coord-relay-vision-chain"
    assert p.meta.capability == "action"
    assert p.meta.real_world_meaning, "real_world_meaning required"
    assert p.meta.robotics_analogue, "robotics_analogue required"
    anchors = " | ".join(p.meta.benchmark_anchor).lower()
    for needed in ("relay chain", "sensor-network coverage",
                   "communications relay"):
        assert needed in anchors, f"benchmark_anchor missing {needed!r}: {anchors}"
    for lv in LEVELS:
        c = compile_level(p, lv)
        assert c.map_supported, f"{lv}: rush-hour-arena must be Rust-loadable"
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _win_clause_keys(c):
    """Flatten the win predicate's clause keys (handles all_of nested
    in an any_of, as the hard tier uses)."""
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    keys: set[str] = set()
    branches = []
    if "all_of" in wc:
        branches = [wc["all_of"]]
    elif "any_of" in wc:
        for b in wc["any_of"]:
            branches.append(dict(b).get("all_of", [b]))
    for clauses in branches:
        for cl in clauses:
            keys |= set(cl)
    return keys


def test_win_requires_region_chain_discovery_clock_and_fact():
    """Every level's win must require: ≥3 `units_in_region_gte`
    (≥2 relay legs + the far leg), `enemies_discovered_gte` (the far
    objective observed), `within_ticks` (the clock), and the standing
    `fact`."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        keys = _win_clause_keys(c)
        assert "units_in_region_gte" in keys, f"{lv}: win needs region clauses"
        assert "enemies_discovered_gte" in keys, f"{lv}: win needs discovery"
        assert "within_ticks" in keys, f"{lv}: win needs the clock"
        assert "building_count_gte" in keys, f"{lv}: win needs the fact"


def test_win_has_at_least_three_region_clauses_per_branch():
    """The chain teeth: each win branch carries ≥3 `units_in_region_gte`
    clauses (≥2 intermediate relays + the far objective). With 4 jeeps
    they can only all be satisfied by SPACING the scouts."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        branches = []
        if "all_of" in wc:
            branches = [wc["all_of"]]
        elif "any_of" in wc:
            branches = [dict(b).get("all_of", [b]) for b in wc["any_of"]]
        for clauses in branches:
            n_region = sum(1 for cl in clauses if "units_in_region_gte" in cl)
            assert n_region >= 3, (
                f"{lv}: each win branch needs ≥3 region clauses "
                f"(≥2 relays + far), got {n_region}"
            )


@pytest.mark.parametrize("lv", LEVELS)
def test_fail_reachable_no_draw(lv):
    """`within_ticks` and the fail `after_ticks` (= within+1) must both
    be below the tick ceiling (93 + 90·(max_turns−1)) so a non-win run
    crosses the fail clause — a real LOSS, never a silent DRAW."""
    p = load_pack(PACK)
    c = compile_level(p, lv)
    ceiling = 93 + 90 * (c.max_turns - 1)
    keys = _win_clause_keys(c)
    assert "within_ticks" in keys
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    # pull a within_ticks value from the first branch
    if "all_of" in wc:
        clauses = wc["all_of"]
    else:
        clauses = dict(wc["any_of"][0]).get("all_of", wc["any_of"])
    wt = next(cl["within_ticks"] for cl in clauses if "within_ticks" in cl)
    fc = dict(c.fail_condition.__pydantic_extra__ or {})
    aft = next(cl["after_ticks"] for cl in fc["any_of"] if "after_ticks" in cl)
    assert wt < ceiling, f"{lv}: within_ticks {wt} ≥ ceiling {ceiling} ⇒ DRAW"
    assert aft == wt + 1, f"{lv}: fail after_ticks should be within_ticks+1"
    assert aft <= ceiling, f"{lv}: fail after_ticks {aft} > ceiling {ceiling}"


def test_fail_has_timeout_and_fact_clauses():
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        keys: set[str] = set()
        for cl in fc["any_of"]:
            keys |= set(cl)
            if "not" in cl:
                keys |= set(cl["not"])
        assert "after_ticks" in keys, f"{lv}: fail must include the timeout"
        assert "building_count_gte" in keys, f"{lv}: fail must catch a razed fact"


def test_four_jeeps_one_fact_per_spawn_group_no_combat_tools():
    """Each spawn group fields exactly 4 jeeps + 1 fact, and the pack
    grants NO attack/build tools — this is a pure positioning test."""
    p = load_pack(PACK)
    tools = set(p.base.get("tools") or [])
    assert not (tools & {"attack_unit", "attack_move", "build",
                         "place_building", "harvest"}), (
        f"vision-relay pack must not grant attack/build tools; got {sorted(tools)}"
    )
    for lv in LEVELS:
        c = compile_level(p, lv)
        groups: dict[int, list] = {}
        for a in c.scenario.actors:
            if a.owner == "agent":
                groups.setdefault(a.spawn_point or 0, []).append(a)
        for g, actors in groups.items():
            jeeps = [a for a in actors if a.type == "jeep"]
            facts = [a for a in actors if a.type == "fact"]
            assert len(jeeps) == 4, f"{lv} group {g}: need 4 jeeps, got {len(jeeps)}"
            assert len(facts) == 1, f"{lv} group {g}: need 1 fact"


def test_hidden_cluster_is_holdfire_e3():
    """The hidden objective cluster is `e3` (surfaces in
    enemy_positions; CLAUDE.md `e1` footgun) and stance:0 HoldFire so
    it HOLDS the far objective instead of advancing on the base."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        clusters = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "e3"
        ]
        assert clusters, f"{lv}: missing the hidden e3 objective cluster"
        for a in clusters:
            assert a.stance == 0, f"{lv}: hidden cluster must be stance:0 HoldFire"


def test_persistent_inert_enemy_fact_marker():
    """An unarmed enemy `fact` far east keeps the episode alive past
    win/fail evaluation (anti-DRAW; engine auto-`done`s on enemy
    elimination)."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        far = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact" and a.position[0] >= 110
        ]
        assert far, f"{lv}: missing the far-east anti-DRAW enemy fact marker"


def test_hard_has_multiple_spawn_point_groups():
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    sp = {
        a.spawn_point if a.spawn_point is not None else 0
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


# ── 2) engine-required scripted-policy discrimination sweep ────────────


def _jeeps(rs):
    us = rs.get("units_summary") or []
    return sorted([u for u in us if u.get("type") == "jeep"], key=lambda u: u["id"])


def _latitude(jeeps):
    """Infer the corridor latitude from the jeep cluster (hard tier
    flips between NORTH y=13 and SOUTH y=27; easy/medium are y=20)."""
    if not jeeps:
        return 20
    ym = sorted(j["cell_y"] for j in jeeps)[len(jeeps) // 2]
    return 13 if ym < 20 else (27 if ym > 22 else 20)


def _relay_xs(level):
    """Intermediate relay x-coords by tier (medium/hard add a 3rd)."""
    return [45, 75] if level == "easy" else [45, 60, 75]


def _stall(rs, C):
    """Observe-only — no jeep moves, no region occupied → LOSS."""
    return [C.observe()]


def _spaced_chain(level):
    """Intended capability: space the 4 jeeps — one per relay region
    plus one on the far objective (108). The 4th jeep doubles up on a
    leg. Every region clause holds simultaneously → WIN."""
    relays = _relay_xs(level)

    def policy(rs, C):
        jeeps = _jeeps(rs)
        if not jeeps:
            return [C.observe()]
        lat = _latitude(jeeps)
        targets = [(x, lat) for x in relays] + [(108, lat)]
        cmds = []
        for i, j in enumerate(jeeps):
            tx, ty = targets[i] if i < len(targets) else targets[-1]
            cmds.append(C.move_units([str(j["id"])], tx, ty))
        return cmds or [C.observe()]

    return policy


def _one_scout_far(rs, C):
    """Race ONE jeep to the far objective; the other 3 idle at base.
    The intermediate relay regions stay empty → LOSS."""
    jeeps = _jeeps(rs)
    if not jeeps:
        return [C.observe()]
    lat = _latitude(jeeps)
    return [C.move_units([str(jeeps[0]["id"])], 108, lat)]


def _bunched_far(rs, C):
    """Drive ALL 4 jeeps together to the far objective. They occupy
    only the far region; the relay regions are empty → LOSS."""
    jeeps = _jeeps(rs)
    if not jeeps:
        return [C.observe()]
    lat = _latitude(jeeps)
    return [C.move_units([str(j["id"]) for j in jeeps], 108, lat)]


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_spaced_chain_wins(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _spaced_chain(lv), seed=seed)
    assert res.outcome == "win", (
        f"{lv} seed{seed}: intended spaced relay chain must WIN, got "
        f"{res.outcome} (turns={res.turns}, lost={res.signals.units_lost})"
    )


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{lv} seed{seed}: stall must be a real timeout LOSS, got {res.outcome}"
    )


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_one_scout_far_loses(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _one_scout_far, seed=seed)
    assert res.outcome == "loss", (
        f"{lv} seed{seed}: one-scout-far must LOSE (relays empty), "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_bunched_scouts_lose(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _bunched_far, seed=seed)
    assert res.outcome == "loss", (
        f"{lv} seed{seed}: bunched scouts must LOSE (one region only), "
        f"got {res.outcome}"
    )
