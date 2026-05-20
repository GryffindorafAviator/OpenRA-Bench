"""build-defensive-tower-cluster scenario family, full loop on Rust.

The pack tests CONCENTRATED-DEFENSE topology: when one high-value
building (the agent fact) is the protected asset and a heavy rush
focuses on it, the right architecture is a TIGHT CLUSTER of pillboxes
WRAPPING the fact (overlapping fields of fire on the protected cell,
military strongpoint doctrine), NOT a thin LINE spread across the
attack lane far from the fact. The win predicate makes the topology
decision load-bearing — total pbox count alone is not enough:

* `building_count_gte:{pbox, n:4}` ⇒ the agent built the budget worth
  of defences (exactly 4 on every level — cash is tight enough that
  spending on other things blocks the count);
* `building_in_region:{pbox, x:fact_x, y:fact_y, radius:4, count:3}`
  ⇒ ≥3 of those pbox sit INSIDE the radius-4 disc around the fact —
  a pbox-LINE layout (pboxes strung along the attack lane far from the
  fact) satisfies the count but NOT the region;
* `building_count_gte:{fact,1}` ⇒ the fact must still STAND (the
  PRESENT-TENSE predicate, not `has_building:fact` which is a one-shot
  ever-seen set — CLAUDE.md footgun);
* `units_killed_gte:K` ⇒ the cluster has to actually engage the rush
  band (a cluster that never sees combat because the rush band misses
  it does not satisfy the bar);
* `within_ticks` paired with `after_ticks` ⇒ a non-finisher is a real
  reachable timeout LOSS (no interrupts on this pack ⇒ each step is
  exactly 90 ticks, so max_turns is a hard tick budget that the
  `after_ticks` deadline reliably bites in).

The scripted-policy validations prove deterministically that:

* the intended adaptive CLUSTER policy (4 pbox WRAPPING the active
  fact within radius 4) WINS every level + every hard seed (1..4);
* stall / pbox-line (pboxes strung along the attack lane far from the
  fact) / pure-army (no pbox) all LOSE every level + every hard seed —
  a real LOSS, not a draw;
* the hard tier defines ≥2 spawn_point groups (north fact y=14 /
  south fact y=26) so a memorised "build cluster at (10,20)"
  placement that worked on easy/medium FAILS the region clause on
  hard (the fact never sits at y=20 on hard).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-defensive-tower-cluster.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. The rush razes the fact
    and/or the clock runs out."""
    return [C.observe()]


def _build_and_place(rs, C, cells):
    """Common build-place loop: at each turn, place the next pbox in
    `cells` if the previous one finished; queue the next build."""
    own_b = rs.get("own_buildings") or []
    n = sum(1 for b in own_b if b.get("type") == "pbox")
    if n >= len(cells):
        return [C.observe()]
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    cmds = []
    if "pbox" not in prod_items:
        cmds.append(C.build("pbox"))
    cmds.append(C.place_building("pbox", cells[n][0], cells[n][1]))
    return cmds or [C.observe()]


def make_adaptive_cluster():
    """Intended CLUSTER topology: read the fact's cell from the
    observation on turn 1, then place 4 pboxes inside the radius-4
    disc around it. This is the policy the pack rewards: the cluster
    must follow the fact, which on hard flips between y=14 and y=26
    by seed."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            # 4 cells inside the radius-4 disc around (10, fy), avoiding
            # the fact's 2x2 footprint and the pre-placed tent/powr.
            state["cells"] = [
                (9, fy - 2), (12, fy - 1), (12, fy + 2), (9, fy + 3),
            ]
        return _build_and_place(rs, C, state["cells"])

    return policy


def make_pbox_line():
    """The "build-defensive-tower-line" counterfactual: 4 pboxes strung
    along the east-west attack lane (x=20..35, y≈18..21) — well outside
    the radius-4 disc around the fact at (10, fact_y). Meets the count
    clause but FAILS the region clause (0 of 4 inside the disc) AND
    cannot mass enough firepower to blunt the rush, which slips through
    the gaps to reach the fact."""
    cells = [(20, 18), (25, 19), (30, 20), (35, 21)]

    def policy(rs, C):
        return _build_and_place(rs, C, cells)

    return policy


def make_wrong_centre_cluster():
    """A cluster centred on the OLD (10,20) location — wins easy/medium
    (where the fact IS at (10,20)) but FAILS the region clause on hard
    (where the fact is at (10,14) or (10,26) per seed, so a cluster
    at (10,20) lands 0 of 4 inside the radius-4 disc around the fact).
    Demonstrates the spawn-driven discrimination: a memorised cell list
    that worked at lower tiers does NOT generalise to the hard fact-
    flip."""
    cells = [(9, 18), (12, 19), (12, 21), (9, 22)]

    def policy(rs, C):
        return _build_and_place(rs, C, cells)

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a pbox. FAILS the
    `building_count_gte:pbox` clause; the rifle wall alone cannot stop
    the heavier rush band either, so the fact often falls on hard."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-defensive-tower-cluster"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (CLAUDE.md / pack spec).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("erqa" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("strongpoint" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("asset protection" in a for a in anchors), pack.meta.benchmark_anchor
    # rusher bot wired through (charges agent centroid → the rush
    # converges on the fact regardless of seed).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_exact_pbox_budget():
    """Cash is intentionally tight (4 pbox at 600 each = 2400, zero
    slack). A model that wastes cash on extra units cannot complete
    the pbox count clause."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == 2400, (lvl, c.starting_cash)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns. No interrupts on
    this pack ⇒ each step is exactly 90 ticks (max tick = 93+90·(N-1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


def test_fact_alive_clause_uses_present_tense_predicate():
    """The fact-survival clause must use the PRESENT-TENSE predicate
    (`building_count_gte:{type:fact,n:1}`) rather than `has_building`,
    which is a one-shot "ever seen" set that stays true after the fact
    is destroyed (a documented CLAUDE.md footgun)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        fact_clauses = [
            clause for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type") == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive fail clause"


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so the
    fact (and therefore the cluster centre) flips by seed. The two
    groups must define the NORTH (y=14) and SOUTH (y=26) fact pair."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # The fact at each spawn group sits at the NORTH/SOUTH latitudes.
    fact_ys = sorted({
        a.position[1] for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    })
    assert fact_ys == [14, 26], fact_ys
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended CLUSTER wins every level + every hard seed ────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_adaptive_cluster_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_adaptive_cluster(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended adaptive cluster must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / wrong-topology policy LOSES (not draws) ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall",     lambda: stall),
        ("pbox_line", make_pbox_line),
        ("pure_army", lambda: pure_army),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (rush razes fact OR clock), pbox-line (region clause
    unmet — the "build-defensive-tower-line" counterfactual the pack
    is named against), and pure-army (count clause unmet) must ALL
    LOSE on every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_wrong_centre_cluster_loses_on_hard_because_fact_flips():
    """A non-adaptive cluster centred on the OLD (10,20) cell (the
    fact's lower-tier location) FAILS the region clause on every hard
    seed — the fact never sits at y=20 on hard, so 0 of 4 pboxes land
    inside the radius-4 disc around the active fact. This is the
    spawn-driven discrimination the pack is curated for: a memorised
    cell list that worked on easy/medium does NOT generalise."""
    c = compile_level(load_pack(PACK), "hard")
    fn = make_wrong_centre_cluster()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"hard seed{seed} wrong-centre cluster: must LOSE; "
            f"got {r.outcome} buildings={r.signals.own_buildings}"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_adaptive_cluster(), seed=3)
    b = run_level(c, make_adaptive_cluster(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
