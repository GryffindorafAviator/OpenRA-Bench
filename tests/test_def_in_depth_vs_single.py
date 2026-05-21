"""def-in-depth-vs-single scenario family, full loop on Rust.

The pack tests DEFENSE-IN-DEPTH topology: against a heavy enemy wave
that drives straight at the construction yard, the same finite pillbox
budget split into a FRONT band plus a REAR band at greater depth beats
a single thick wall. The discriminator is layout topology — total pbox
count alone is not enough.

* `building_count_gte:{pbox, n:4}` — the agent built the budget worth of
  defences (exactly 4 on every level — cash is exactly 4·600cr).
* `building_in_region:{pbox, x:35, y:fact_y, radius:4, count:2}` AND
  `building_in_region:{pbox, x:20, y:fact_y, radius:4, count:2}` — two
  REGION clauses with non-overlapping radii (FRONT and REAR centres are
  15 cells apart, > 2·radius=8) so the model MUST physically split the
  4 pbox across two depths; a single thick wall massed at one band
  satisfies at most ONE region clause.
* `building_count_gte:{fact,1}` — the fact must still STAND (the
  PRESENT-TENSE predicate, not `has_building:fact` which is a one-shot
  ever-seen set — CLAUDE.md footgun).
* `units_killed_gte:K` — the depth has to actually destroy the wave.
* `within_ticks` paired with `after_ticks` — a non-finisher is a real
  reachable timeout LOSS (no interrupts ⇒ each step is exactly 90
  ticks, so max_turns is a hard tick budget the `after_ticks` deadline
  reliably bites in).

The scripted-policy validations prove deterministically (no model, no
network) that:

* the intended TWO-BAND DEPTH policy (2 pbox FRONT + 2 pbox REAR,
  following the active fact latitude) WINS every level + every hard
  seed (1..4);
* stall / single-thick-wall (all 4 pbox in ONE band, front-only OR
  rear-only) / pure-army (no pbox) all LOSE every level + every seed —
  a real LOSS, not a draw;
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 agent spawn_point groups (north base y=14 /
  south base y=26) so a memorised "build bands at y=20" placement that
  worked on easy/medium FAILS the region clauses on hard.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-in-depth-vs-single.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. The clock runs out (and/or
    the wave razes the fact)."""
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


def make_two_band_depth():
    """Intended DEFENSE-IN-DEPTH topology: read the fact's cell from the
    observation on turn 1, then place 2 pbox in the FRONT band (x=35)
    and 2 pbox in the REAR band (x=20), centred on the fact's latitude.
    This is the policy the pack rewards: both bands must follow the
    fact, which on hard flips between y=14 and y=26 by seed."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            # 2 FRONT (x=35) + 2 REAR (x=20), straddling the fact lane.
            state["cells"] = [
                (35, fy - 1), (35, fy + 1), (20, fy - 1), (20, fy + 1),
            ]
        return _build_and_place(rs, C, state["cells"])

    return policy


def make_single_wall(band_x):
    """SINGLE-THICK-WALL counterfactual: all 4 pboxes massed into ONE
    band at `band_x` (front-only at x=35, or rear-only at x=20). Meets
    the count clause and ONE region clause but FAILS the other region
    clause — the wall has no depth."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            state["cells"] = [
                (band_x, fy - 2), (band_x, fy - 1),
                (band_x, fy + 1), (band_x, fy + 2),
            ]
        return _build_and_place(rs, C, state["cells"])

    return policy


def make_wrong_latitude_depth():
    """A two-band split memorised at the OLD y=20 latitude — wins
    easy/medium (where the fact IS at y=20) but FAILS the region
    clauses on hard (where the fact is at y=14 or y=26 per seed, so
    bands at y=20 land 0 pbox inside the radius-4 discs around the
    active fact's bands). Demonstrates the spawn-driven discrimination:
    a memorised cell list that worked at lower tiers does NOT
    generalise to the hard base-flip."""
    cells = [(35, 19), (35, 21), (20, 19), (20, 21)]

    def policy(rs, C):
        return _build_and_place(rs, C, cells)

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a pbox. FAILS the
    `building_count_gte:pbox` clause and both region clauses."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-in-depth-vs-single"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (defense-in-depth / layered).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("defense-in-depth" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("layered defense" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("microrts" in a for a in anchors), pack.meta.benchmark_anchor
    # rusher bot wired through (charges the agent fact → the wave
    # converges down the lane through both depth bands).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_exact_pbox_budget():
    """Cash is exactly 4 pbox at 600 each = 2400 on every level — no
    slack for extra units; a model that overspends cannot complete the
    pbox count clause."""
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


def test_easy_medium_have_two_non_overlapping_depth_regions():
    """The FRONT and REAR band regions MUST NOT overlap, or a single
    tall wall could satisfy both region predicates and the depth
    topology would not be enforced."""
    for lvl in ("easy", "medium"):
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        regions = [
            clause["building_in_region"]
            for clause in win.get("all_of", [])
            if "building_in_region" in clause
        ]
        assert len(regions) == 2, (lvl, regions)
        a, b = regions
        dx = abs(int(a["x"]) - int(b["x"]))
        assert dx > int(a["radius"]) + int(b["radius"]), (
            f"{lvl}: depth regions overlap: dx={dx}, "
            f"rA+rB={a['radius'] + b['radius']}"
        )


def test_win_requires_a_kill_bar():
    """Every level's win must include a `units_killed_gte` clause — the
    depth has to actually destroy the wave, not merely soak it."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        kill = [
            clause for clause in win.get("all_of", [])
            if "units_killed_gte" in clause
        ]
        assert kill, f"{lvl}: missing units_killed_gte clause"
        assert int(kill[0]["units_killed_gte"]) >= 1


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so the
    base (and therefore both depth bands) flips by seed. The two groups
    define the NORTH (y=14) and SOUTH (y=26) fact pair."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    fact_ys = sorted({
        a.position[1] for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    })
    assert fact_ys == [14, 26], fact_ys
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended TWO-BAND DEPTH wins every level + every seed ──


@pytest.mark.parametrize("level", LEVELS)
def test_intended_two_band_depth_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_two_band_depth(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended two-band depth must WIN; "
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
        ("single_front",  lambda: make_single_wall(35)),
        ("single_rear",   lambda: make_single_wall(20)),
        ("pure_army",     lambda: pure_army),
    ],
)
def test_lazy_and_single_wall_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (clock OR fact razed), single-thick-wall (one region
    clause unmet — the single-line counterfactual the pack is named
    against), and pure-army (count clause unmet) must ALL LOSE on every
    level + every seed — a real reachable LOSS, not a draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_wrong_latitude_depth_loses_on_hard_because_base_flips():
    """A two-band split memorised at the OLD y=20 latitude FAILS the
    region clauses on every hard seed — the base never sits at y=20 on
    hard, so 0 pboxes land inside the radius-4 discs around the active
    fact's depth bands. This is the spawn-driven discrimination the
    pack is curated for: a memorised cell list that worked on
    easy/medium does NOT generalise."""
    c = compile_level(load_pack(PACK), "hard")
    fn = make_wrong_latitude_depth()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"hard seed{seed} wrong-latitude depth: must LOSE; "
            f"got {r.outcome} buildings={r.signals.own_buildings}"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_two_band_depth(), seed=3)
    b = run_level(c, make_two_band_depth(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
