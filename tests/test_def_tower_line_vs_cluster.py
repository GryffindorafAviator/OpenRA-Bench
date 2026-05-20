"""def-tower-line-vs-cluster scenario family, full loop on Rust.

The pack tests DEFENSE TOPOLOGY: when the threat is forced through a
known chokepoint, the right architecture is dense cluster AT the choke
(graph min-cut / military bunker placement doctrine), NOT a thin spread
along the perimeter. The win predicate makes the topology decision
load-bearing — total pbox count alone is not sufficient:

* `building_count_gte:{pbox, n}` ⇒ the agent actually built the budget
  worth of defences (3 on easy, 4 on medium, 5 on hard);
* `building_in_region:{pbox, x:60, y:20, radius:5, count}` ⇒ ≥3 (easy),
  ≥3 (medium), or ≥4 (hard) of those pbox sit INSIDE the choke region —
  a spread-line layout (one pbox at the choke, the rest along the
  perimeter) satisfies the count but NOT the region;
* `building_count_gte:{fact,1}` ⇒ the fact must still stand (rather than
  `has_building:fact` which is a one-shot "ever seen" set — see CLAUDE.md
  footgun);
* `within_ticks` paired with `after_ticks` ⇒ a non-finisher is a real
  reachable timeout LOSS (no interrupts on this pack ⇒ each step is
  exactly 90 ticks, so max_turns is a hard tick budget that the
  `after_ticks` deadline reliably bites in).

The scripted-policy validations prove deterministically that:

* the intended CLUSTER policy (all pbox built INSIDE the choke region)
  WINS every level + every hard seed (1..4);
* stall / spread-line (1 at choke + rest perimeter) / pure-army
  (no pbox) all LOSE every level + every hard seed — a real LOSS, not
  a draw;
* the hard tier defines ≥2 spawn_point groups (north y=10 / south y=30)
  so a memorised relative-to-base placement that lands in the same
  world cell on every seed cannot solve the pack.
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

# Per-level total pbox budget (exactly `starting_cash / 600`).
_N_PBOX = {"easy": 3, "medium": 4, "hard": 5}


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. Fact gets razed and/or
    the clock runs out."""
    return [C.observe()]


def make_cluster(choke=(60, 20)):
    """Intended CLUSTER topology: every pbox is placed INSIDE the choke
    region (within radius 5 of the choke cell)."""
    cx0, cy0 = choke
    # Eight pre-chosen cells around the choke; the policy uses as many
    # as the budget needs (easy:3, medium:4, hard:5).
    cells = [
        (cx0 - 2, cy0 - 1), (cx0, cy0 - 1), (cx0 + 2, cy0 - 1),
        (cx0 - 2, cy0 + 1), (cx0, cy0 + 1), (cx0 + 2, cy0 + 1),
        (cx0 - 1, cy0),     (cx0 + 1, cy0),
    ]

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        cmds = []
        # Once enough pboxes are up, idle (the win clause counts the
        # current building list each turn).
        if n >= len(cells):
            return [C.observe()]
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", cells[n][0], cells[n][1]))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_spread_line(n_pbox):
    """SPREAD-LINE topology: one pbox at the choke + the rest along the
    perimeter near the base. Satisfies `building_count_gte` but FAILS
    `building_in_region` (only 1 of N at the choke, not the required
    3-of-3 / 3-of-4 / 4-of-5)."""
    cells = [(60, 20), (20, 18), (24, 18), (28, 18), (32, 18)][:n_pbox]

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        n = sum(1 for b in own_b if b.get("type") == "pbox")
        prod = rs.get("production") or []
        prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
        cmds = []
        if n >= len(cells):
            return [C.observe()]
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", cells[n][0], cells[n][1]))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a pbox. FAILS the
    `building_count_gte:pbox` clause AND lets the rush eventually reach
    the fact (or runs out the clock with no pbox)."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-tower-line-vs-cluster"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert any("min-cut" in a.lower() or "chokepoint" in a.lower() for a in anchors), anchors
    assert any("bunker" in a.lower() for a in anchors), anchors
    # Rusher bot wired through (charges agent centroid → forces the
    # rush path through the mid-map choke on every seed).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_exact_pbox_budget():
    """The cash is intentionally tight (3/4/5 pbox at 600 each, zero
    slack). A model that spends on units OR extra power runs out before
    the count clause is satisfied — the topology decision is the spend."""
    pack = load_pack(PACK)
    for lvl, expected in (("easy", 1800), ("medium", 2400), ("hard", 3000)):
        c = compile_level(pack, lvl)
        assert c.starting_cash == expected, (lvl, c.starting_cash)


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


# ── solvency: intended CLUSTER wins every level + every hard seed ────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_cluster_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_cluster(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended cluster topology must WIN; "
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
        ("stall",       lambda lvl: stall),
        ("spread_line", lambda lvl: make_spread_line(_N_PBOX[lvl])),
        ("pure_army",   lambda lvl: pure_army),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (rush razes fact OR clock), spread-line (region clause
    unmet), and pure-army (count clause unmet) must ALL LOSE on every
    level + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory(level)
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
    a = run_level(c, make_cluster(), seed=3)
    b = run_level(c, make_cluster(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
