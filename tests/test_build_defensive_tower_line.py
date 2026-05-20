"""build-defensive-tower-line scenario family, full loop on Rust.

The pack tests DEFENSIVE PERIMETER TOPOLOGY: when the threat is funnelled
through a known corridor whose WIDTH matters (y=18..22 at x=60), the right
architecture is one pbox per row across the FULL corridor width (a LINE),
NOT a cluster on the centre row and NOT a scatter near the base. This is
the sibling/inverse of `def-tower-line-vs-cluster` (which forces a
CLUSTER at a single bottleneck cell); together the two packs discriminate
whether the model understands the FORCING GEOMETRY (single-cell chokepoint
vs corridor-width approach).

Anchors: ERQA spatial commit / MicroRTS defense placement / military
perimeter (firewall rule placement).

The win predicate makes the LINE topology load-bearing — total pbox
count alone is not enough:

* `building_count_gte:{pbox, n:4}` ⇒ the agent built the full budget;
* `building_in_region:{pbox, x:60, y:Y, radius:0.5, count:1}` for each
  of the four corridor rungs Y ∈ {18,19,21,22} ⇒ exactly one pbox per
  row across the corridor (a tiny radius 0.5 means only the exact cell
  counts, so a cluster on (60,20) misses ALL FOUR rungs and a scatter
  near the base misses all four);
* `building_count_gte:{fact,n:1}` (present-tense — `has_building` is
  the one-shot "ever-seen" set, see CLAUDE.md footgun);
* `within_ticks` paired with `after_ticks` in the fail clause ⇒ a
  non-finisher is a real reachable timeout LOSS (no interrupts on this
  pack ⇒ each step is exactly 90 ticks, so max_turns is a hard tick
  budget that the `after_ticks` deadline reliably bites in).

The scripted-policy validations prove deterministically that:

* the intended LINE policy (one pbox at each of the four corridor rung
  cells) WINS every level + every hard seed (1..4);
* stall / random-4-pbox (4 pboxes placed near the base, away from the
  corridor) both LOSE every level + every hard seed — a real LOSS,
  not a draw;
* the hard tier defines ≥2 spawn_point groups (NORTH base y=12 / SOUTH
  base y=28) so a memorised base-relative placement cannot generalise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-defensive-tower-line.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Required corridor rung cells (the four "rungs" spanning y=18..22 at
# the choke column x=60; y=20 is the middle that the LINE topology
# leaves open by construction — placing on y=20 won't satisfy any rung).
RUNGS = [(60, 18), (60, 19), (60, 21), (60, 22)]

# Cells used by the "random-4-pbox" wrong-topology policy: 4 pboxes
# clustered near the base rather than along the corridor. None of these
# lie inside ANY rung region (radius 0.5 around the rung cells), so the
# region clauses are all unsatisfied.
RANDOM_CELLS_NEAR_BASE = [(20, 18), (22, 20), (24, 22), (26, 19)]


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. Fact gets razed by the
    rush AND the count/region clauses are never satisfied."""
    return [C.observe()]


def make_line():
    """Intended LINE topology: one pbox at EACH of the four corridor
    rung cells (60,18) (60,19) (60,21) (60,22)."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        # Once 4 pboxes are up, idle (the win clause re-evaluates each turn).
        if n >= len(RUNGS):
            return [C.observe()]
        cmds = []
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", RUNGS[n][0], RUNGS[n][1]))
        return cmds

    return policy


def make_random_4_pbox():
    """WRONG TOPOLOGY: 4 pboxes placed near the base (not at the
    corridor rungs). Satisfies `building_count_gte:{pbox,n:4}` but
    FAILS every rung region (none of the cells lie in any rung's
    radius-0.5 disk), so the win predicate cannot fire."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        if n >= len(RANDOM_CELLS_NEAR_BASE):
            return [C.observe()]
        cmds = []
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(
            C.place_building(
                "pbox",
                RANDOM_CELLS_NEAR_BASE[n][0],
                RANDOM_CELLS_NEAR_BASE[n][1],
            )
        )
        return cmds

    return policy


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-defensive-tower-line"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert "ERQA" in anchors, anchors
    assert "MicroRTS defense" in anchors, anchors
    assert "military perimeter" in anchors, anchors
    # Rusher bot wired through (charges agent centroid → forces the
    # rush path through the corridor on every seed).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_exact_pbox_budget():
    """The cash is intentionally tight (4 pbox at 600 each = 2400 on
    every level, zero slack). A model that spends on units OR extra
    power runs out before the count clause is satisfied."""
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
    is destroyed (a documented CLAUDE.md footgun). Otherwise the rush
    razing the fact would not trigger a LOSS."""
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


def test_win_requires_one_pbox_per_corridor_rung():
    """The LINE-enforcement contract: every level's win clause requires
    exactly one pbox in EACH of the four corridor rungs at x=60
    y∈{18,19,21,22}. A cluster on the centre row (y=20) misses all four
    rungs because each rung region has radius 0.5 (cell-exact)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        rungs_seen = set()
        for clause in wc.get("all_of", []) or []:
            br = clause.get("building_in_region")
            if (
                isinstance(br, dict)
                and br.get("type") == "pbox"
                and int(br.get("x", -1)) == 60
                and int(br.get("count", 0)) == 1
                and float(br.get("radius", 0)) <= 1.0
            ):
                rungs_seen.add(int(br["y"]))
        assert rungs_seen == {18, 19, 21, 22}, (
            f"{lvl}: corridor rungs y∈{{18,19,21,22}} required, got {sorted(rungs_seen)}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so a
    memorised relative-to-base placement that lands in the same world
    cell on every seed cannot generalise."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126):
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended LINE wins every level + every hard seed ───────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_line_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_line(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended LINE topology must WIN; "
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
        ("stall",         lambda: stall),
        ("random_4_pbox", lambda: make_random_4_pbox()),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (rush razes fact AND clock runs out with no pbox) and
    random-4-pbox (count satisfied but every rung region unsatisfied,
    so the win never fires and the clock runs out) must ALL LOSE on
    every level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
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
