"""def-tower-line-vs-cluster scenario family, full loop on Rust.

The pack tests DEFENSE TOPOLOGY: the agent must match the topology of
its pillbox layout (LINE across rows OR CLUSTER at one cell) to the
attacker's forcing geometry. Each tier presents a different geometry
and therefore demands a different topology:

* easy  — open arena 112x40, 4 attacker squads on 4 separate rows
          (y=4/12/28/36) → LINE wins, CLUSTER fails (the line rung
          clauses are all missed by a y=20 cluster).
* medium— chokepoint-arena 112x40, one 5-cell corridor at y=18..22
          x=60 → CLUSTER wins, LINE fails (the off-corridor rows are
          impassable wall water, so a line attempt at y=8/16/24/32
          either has no place to go or is rejected outright).
* hard  — open arena 112x40 with TWO enemy spawn_point groups: a
          CONCENTRATED-thrust composition (9×e1 + 2×e3 stacked at
          y=20) and a WIDE-FRONT composition (4 stance:0 squads on
          y=4/12/28/36). The win predicate is `any_of` over the
          cluster bar and the line bar; the kill quota
          `units_killed_gte:8` is the load-bearing clause — a
          memorised CLUSTER vs the wide-front seed (or a memorised
          LINE vs the concentrated seed) cannot fire on the active
          threat and so cannot satisfy the kill quota.

The win predicate makes the topology decision load-bearing — total
pbox count alone is not sufficient:

* `building_count_gte:{pbox, n:4}` ⇒ the agent built the full budget;
* `building_in_region` row/disc clauses ⇒ the pbox layout MATCHES the
  forcing geometry (LINE rungs on easy, CLUSTER disc on medium, EITHER
  on hard);
* `units_killed_gte:N` ⇒ the pbox layout actively KILLS the attackers
  (a stall / wrong-topology layout that satisfies the region clause
  by luck still fails to kill enough);
* `building_count_gte:{fact,n:1}` (present-tense — `has_building` is
  the one-shot "ever-seen" set, see CLAUDE.md footgun);
* `within_ticks` paired with `after_ticks` in the fail clause ⇒ a
  non-finisher is a real reachable timeout LOSS (no interrupts on this
  pack ⇒ each step is exactly 90 ticks, so max_turns is a hard tick
  budget that the `after_ticks` deadline reliably bites in).

The scripted-policy validations prove deterministically that:

* the intended TOPOLOGY (line on easy / cluster on medium / matched
  topology per seed on hard) WINS every level + every applicable seed;
* stall, pure-army, and the WRONG-TOPOLOGY (cluster on easy / line on
  medium) all LOSE on every level + every seed — a real LOSS, not a
  draw (the count or region or kill clause is unmet AND the
  `after_ticks` deadline reliably bites);
* the hard tier defines ≥2 distinct enemy spawn_point groups (Wave-9
  per-owner spawn_point activation) so a single memorised topology
  cannot generalise across seeds.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-tower-line-vs-cluster.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Pbox cells for each topology. Outer rows are placed FIRST so they
# are up by the time the (slowest-arriving) flank squads reach x=60.
LINE_CELLS    = [(60, 4), (60, 36), (60, 12), (60, 28)]
CLUSTER_CELLS = [(58, 20), (60, 19), (60, 21), (62, 20)]


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. The win predicate
    requires `building_count_gte:pbox,4` AND `units_killed_gte`, so
    the win never latches and the episode times out via
    `after_ticks`. (On every tier the attackers are configured so a
    pure stall also cannot satisfy the fact-alive clause.)"""
    return [C.observe()]


def pure_army(rs, C):
    """PURE-ARMY: only train e1 — never build a pbox. FAILS the
    `building_count_gte:pbox,4` clause regardless of how many kills
    the home-trained infantry rack up."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


def _make_topology(cells):
    """Build one pbox at each cell, in order, then idle. The
    place_building order is rejected by the engine if the cell is
    impassable terrain — which is how the medium tier's LINE attempt
    silently fails the count clause (only the one in-corridor cell
    succeeds)."""
    cells = list(cells)

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        if n >= len(cells):
            return [C.observe()]
        cmds = []
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", cells[n][0], cells[n][1]))
        return cmds or [C.observe()]

    return policy


def make_line():
    return _make_topology(LINE_CELLS)


def make_cluster():
    return _make_topology(CLUSTER_CELLS)


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-tower-line-vs-cluster"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors — both the chokepoint /
    # min-cut anchor (cluster doctrine) AND the perimeter / wide-front
    # anchor (line doctrine) must be declared, since the pack
    # discriminates both.
    anchors = pack.meta.benchmark_anchor
    assert any("min-cut" in a.lower() or "chokepoint" in a.lower() for a in anchors), anchors
    assert any("bunker" in a.lower() for a in anchors), anchors
    assert any("perimeter" in a.lower() for a in anchors), anchors
    # All tiers compile + the per-tier base_map is Rust-loadable.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported, (lvl, c.scenario.base_map)


def test_starting_cash_is_exact_pbox_budget():
    """The cash is tight (exactly 4 pbox at 600 each, zero slack on
    every tier). A model that spends on units OR extra power runs out
    before the count clause is satisfied — the topology decision is
    the spend."""
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


def test_hard_has_two_seed_driven_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so a memorised topology cannot generalise. This pack uses the
    Wave-9 ENEMY-side spawn_point axis (the agent base is fixed
    across all seeds; only the enemy composition flips)."""
    c = compile_level(load_pack(PACK), "hard")
    enemy_groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert enemy_groups == {0, 1}, enemy_groups
    # In-bounds check (per-tier playable area; the maps are 112x40
    # with cordon=2, so x in 2..109 and y in 2..37).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 109 and 2 <= y <= 37, (a.type, a.position)


# ── solvency: intended topology wins every level + every seed ────────


def test_easy_intended_line_wins_every_seed():
    c = compile_level(load_pack(PACK), "easy")
    for seed in SEEDS:
        r = run_level(c, make_line(), seed=seed)
        assert r.outcome == "win", (
            f"easy seed{seed}: intended LINE topology must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_medium_intended_cluster_wins_every_seed():
    c = compile_level(load_pack(PACK), "medium")
    for seed in SEEDS:
        r = run_level(c, make_cluster(), seed=seed)
        assert r.outcome == "win", (
            f"medium seed{seed}: intended CLUSTER topology must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_hard_matched_topology_wins_every_seed():
    """Hard rotates enemy spawn_point per seed: even-indexed
    spawn_point = 0 (concentrated) ⇒ CLUSTER wins; spawn_point = 1
    (wide-front) ⇒ LINE wins. The seed→spawn_point round-robin in the
    env is seed%2 on enemy-side, so odd seeds → spawn_point 1 and
    even seeds → spawn_point 0."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        # Try both topologies; at least one must WIN on this seed
        # (the matched one). The pack's no-cheat bar (below) checks
        # the unmatched one LOSES.
        r_cluster = run_level(c, make_cluster(), seed=seed)
        r_line = run_level(c, make_line(), seed=seed)
        wins = [n for n, r in (("cluster", r_cluster), ("line", r_line))
                if r.outcome == "win"]
        assert wins, (
            f"hard seed{seed}: neither topology won (cluster={r_cluster.outcome}, "
            f"line={r_line.outcome}) — at least one must match the active "
            f"enemy composition"
        )


# ── no-cheat: every lazy / wrong-topology policy LOSES (not draws) ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall",     lambda lvl: stall),
        ("pure_army", lambda lvl: pure_army),
    ],
)
def test_lazy_policies_lose_every_level_and_seed(level, policy_name, policy_factory):
    """Stall (no pbox built, attackers raze fact or clock runs out)
    and pure-army (never builds a pbox, fails the count clause) must
    BOTH LOSE on every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory(level)
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_easy_wrong_topology_cluster_loses_every_seed():
    """On easy, the LINE topology is intended; a CLUSTER at (60,20)
    fails the four LINE rung clauses (the cluster's pbox cells are all
    far from y=4/12/28/36) — the win never latches and the episode
    times out → LOSS."""
    c = compile_level(load_pack(PACK), "easy")
    for seed in SEEDS:
        r = run_level(c, make_cluster(), seed=seed)
        assert r.outcome == "loss", (
            f"easy seed{seed}: wrong-topology CLUSTER must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_medium_wrong_topology_line_loses_every_seed():
    """On medium, the CLUSTER topology is intended; a LINE attempt at
    rows y=4/12/28/36 lands on the wall water (the chokepoint-arena's
    corridor is only y=18..22 at x≈60). Only the in-corridor cell
    actually places; the rest are rejected, so the count clause
    fails."""
    c = compile_level(load_pack(PACK), "medium")
    for seed in SEEDS:
        r = run_level(c, make_line(), seed=seed)
        assert r.outcome == "loss", (
            f"medium seed{seed}: wrong-topology LINE must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_hard_unmatched_topology_loses_on_at_least_one_seed():
    """On hard, the enemy composition flips by seed. A memorised
    topology that satisfies the region clause (cluster OR line)
    cannot satisfy the kill quota on the unmatched seed — at least
    one of the four seeds must be a LOSS for EACH topology
    individually."""
    c = compile_level(load_pack(PACK), "hard")
    cluster_losses = sum(
        1 for seed in SEEDS
        if run_level(c, make_cluster(), seed=seed).outcome == "loss"
    )
    line_losses = sum(
        1 for seed in SEEDS
        if run_level(c, make_line(), seed=seed).outcome == "loss"
    )
    assert cluster_losses >= 1, (
        f"hard: CLUSTER must LOSE on at least one of seeds {SEEDS} "
        f"(the wide-front seed); got 0 losses"
    )
    assert line_losses >= 1, (
        f"hard: LINE must LOSE on at least one of seeds {SEEDS} "
        f"(the concentrated seed); got 0 losses"
    )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_line(), seed=3)
    b = run_level(c, make_line(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
