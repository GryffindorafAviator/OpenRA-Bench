"""build-defensive-skirt-corners scenario family, full loop on Rust.

The pack tests DISTRIBUTED-DEFENSE topology — the inverse of
`build-defensive-tower-cluster`. One high-value building (the agent
fact) sits at the centre of the map and is rushed CONCURRENTLY from
ALL FOUR diagonal corners (NE / NW / SE / SW). Massing every pillbox
on a single corner holds that corner but lets the other three waves
stride untouched into the fact and raze it. The right doctrine is a
SKIRT: one pillbox planted in EACH of the four corner approaches.
The win predicate makes the four-corner topology load-bearing — total
pbox count alone is not enough:

* `building_count_gte:{pbox, n:4}` ⇒ the agent built the budget worth
  of defences (exactly 4 on every level — cash is tight enough that
  spending on other things blocks the count);
* FOUR `building_in_region:{pbox, radius:4, count:1}` clauses, one
  per corner ⇒ one pbox must sit inside the radius-4 disc of EACH
  corner region — a concentrate-at-one-corner layout satisfies the
  count and exactly ONE region clause, failing the other three;
* `building_count_gte:{fact,1}` ⇒ the fact must still STAND (the
  PRESENT-TENSE predicate, not `has_building:fact` which is a one-shot
  ever-seen set that stays true after the fact is razed — CLAUDE.md
  footgun);
* `units_killed_gte:K` ⇒ the skirt has to actually engage the rush;
  the pbox is the load-bearing weapon (engine pbox-weapon fix), and
  with no pre-placed agent defenders the pbox skirt is the SOLE kill
  source — a stall / pure-army layout kills 0;
* `within_ticks` paired with `after_ticks` ⇒ a non-finisher is a real
  reachable timeout LOSS (no interrupts ⇒ each step is exactly 90
  ticks, so max_turns is a hard tick budget the `after_ticks` deadline
  reliably bites in).

The rush arrives as a `scheduled_events: spawn_actors` wave at tick
1800 — AFTER the skirt has had time to assemble — with one band sitting
ON each corner region so a pbox planted there engages it immediately.

The scripted-policy validations prove deterministically that:

* the intended adaptive SKIRT policy (one pbox in EACH of the four
  corner regions around the active fact) WINS every level + every hard
  seed (1..4);
* stall / concentrate (all 4 pbox massed at one corner) / pure-army
  (no pbox) all LOSE every level + every hard seed — a real LOSS,
  not a draw;
* the hard tier defines >=2 spawn_point groups (west fact x=50 / east
  fact x=78) so a memorised "skirt the corners of (64,20)" placement
  that worked on easy/medium FAILS the region clause on hard (the
  fact never sits at x=64 on hard).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-defensive-skirt-corners.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# -- scripted policies -------------------------------------------------


def stall(rs, C):
    """Observe-only — the agent never spends. The clock runs out (and/or
    the rush razes the fact)."""
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


def _fact_cell(rs):
    """Read the agent fact's cell from the observation, or None."""
    own_b = rs.get("own_buildings") or []
    facts = [b for b in own_b if b.get("type") == "fact"]
    if not facts:
        return None
    return facts[0].get("cell_x"), facts[0].get("cell_y")


# Corner-region offsets from the fact (NE, NW, SE, SW). The four
# corner discs sit at fact + (+/-18, +/-11). The scheduled rush bands
# spawn ON these discs, so a pbox planted in a disc engages its band
# the moment the wave arrives.
SKIRT_OFFSETS = [(18, -11), (-18, -11), (18, 11), (-18, 11)]


def make_adaptive_skirt():
    """Intended SKIRT topology: read the fact's cell from the observation
    on turn 1, then place one pbox in EACH of the four corner regions
    around it (offsets +/-18 in x, +/-11 in y). This is the policy the
    pack rewards: the skirt must follow the fact, which on hard flips
    between x=50 and x=78 by seed."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            fc = _fact_cell(rs)
            if fc is None:
                return [C.observe()]
            fx, fy = fc
            state["cells"] = [(fx + dx, fy + dy) for dx, dy in SKIRT_OFFSETS]
        return _build_and_place(rs, C, state["cells"])

    return policy


def make_concentrate():
    """The concentrate counterfactual: all 4 pboxes massed at a SINGLE
    corner (the NE corner of the active fact). Meets the count clause
    and satisfies exactly ONE region clause but FAILS the other three
    AND lets the three uncovered corner waves walk into the fact."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            fc = _fact_cell(rs)
            if fc is None:
                return [C.observe()]
            fx, fy = fc
            cx, cy = fx + 18, fy - 11  # NE corner
            # Four pboxes in a 1-cell-spaced row at the NE corner.
            state["cells"] = [(cx - i, cy) for i in range(4)]
        return _build_and_place(rs, C, state["cells"])

    return policy


def make_wrong_centre_skirt():
    """A skirt centred on the OLD (64,20) location — the easy/medium
    fact cell — but applied on HARD where the fact is at (50,20) or
    (78,20) per seed. A skirt around (64,20) lands every pbox >=10 cells
    outside the active corner discs (radius 4), failing the region
    clauses. Demonstrates the spawn-driven discrimination: a memorised
    cell list that worked at lower tiers does NOT generalise to the hard
    fact-flip."""
    cells = [(64 + dx, 20 + dy) for dx, dy in SKIRT_OFFSETS]

    def policy(rs, C):
        return _build_and_place(rs, C, cells)

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a pbox. FAILS the
    `building_count_gte:pbox` clause and all four region clauses."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
        return [C.build("e1")]
    return [C.observe()]


# -- scenario-shape invariants ----------------------------------------


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-defensive-skirt-corners"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (CLAUDE.md / pack spec).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("microrts" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("distributed defense" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("quadrant" in a for a in anchors), pack.meta.benchmark_anchor
    # rusher bot wired through (charges agent centroid -> every corner
    # wave converges on the central fact regardless of seed).
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


def test_win_requires_four_corner_region_clauses():
    """Each level's win predicate must require one pbox in EACH of four
    distinct corner regions (directly on easy/medium, via an `any_of`
    over two candidate four-corner layouts on hard)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        regions = [
            clause["building_in_region"]
            for clause in wc.get("all_of", []) or []
            if isinstance(clause, dict) and "building_in_region" in clause
        ]
        assert len(regions) == 4, (lvl, regions)
        centres = {(r["x"], r["y"]) for r in regions}
        assert len(centres) == 4, f"{lvl}: corner regions must be distinct"
    # hard: the four-corner bar lives inside an any_of over two layouts.
    c = compile_level(pack, "hard")
    wc = c.win_condition.model_dump(exclude_none=True)
    any_of = [
        clause["any_of"] for clause in wc.get("all_of", []) or []
        if isinstance(clause, dict) and "any_of" in clause
    ]
    assert any_of, "hard: missing any_of over candidate skirt layouts"
    for layout in any_of[0]:
        regions = [
            cl["building_in_region"] for cl in layout.get("all_of", []) or []
            if isinstance(cl, dict) and "building_in_region" in cl
        ]
        assert len(regions) == 4, layout


def test_win_requires_a_kill_quota():
    """The pbox skirt must actively KILL the rush: every level's win
    clause carries a `units_killed_gte` quota. With no pre-placed agent
    defenders the pbox skirt is the sole kill source, so this clause
    makes the pbox weapon load-bearing."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        kill = [
            cl for cl in wc.get("all_of", []) or []
            if isinstance(cl, dict) and "units_killed_gte" in cl
        ]
        assert kill, f"{lvl}: missing units_killed_gte kill quota"
        assert int(kill[0]["units_killed_gte"]) >= 8, (lvl, kill)


def test_rush_arrives_as_a_scheduled_event():
    """The four-corner rush is injected via `scheduled_events:
    spawn_actors` AFTER the skirt has time to assemble — there is no
    t=0 enemy band racing the build."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        raw = pack.levels[lvl]
        ov = getattr(raw, "overrides", None) or {}
        if hasattr(ov, "model_dump"):
            ov = ov.model_dump(exclude_none=True)
        evts = ov.get("scheduled_events") or []
        assert evts, f"{lvl}: expected a scheduled rush wave"
        assert any(e.get("type") == "spawn_actors" for e in evts), (lvl, evts)


def test_no_pre_placed_agent_combat_screen():
    """The pbox skirt must be the sole kill source — there is no
    pre-placed agent combat screen ringing the fact. Only ONE
    non-combatant agent e1 per active spawn group is parked in a far
    map corner (so units_summary is non-empty for the env-reset check);
    it never fights."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        agent_units = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "e1"
        ]
        assert len(agent_units) <= 2, (lvl, [a.position for a in agent_units])
        for a in agent_units:
            x, y = a.position
            assert x <= 6 and (y <= 6 or y >= 34), (lvl, a.position)


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
    """Hard-tier contract: >=2 distinct agent spawn_point groups so the
    fact (and therefore the skirt centre) flips by seed. The two groups
    must define the WEST (x=50) and EAST (x=78) fact pair."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # The fact at each spawn group sits at the WEST/EAST longitudes.
    fact_xs = sorted({
        a.position[0] for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    })
    assert fact_xs == [50, 78], fact_xs
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# -- solvency: intended SKIRT wins every level + every hard seed ------


@pytest.mark.parametrize("level", LEVELS)
def test_intended_adaptive_skirt_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_adaptive_skirt(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended adaptive skirt must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# -- no-cheat: every lazy / wrong-topology policy LOSES (not draws) ---


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall",       lambda: stall),
        ("concentrate", make_concentrate),
        ("pure_army",   lambda: pure_army),
    ],
)
def test_lazy_and_wrong_topology_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (clock runs out), concentrate (all 4 pbox at one corner —
    three region clauses unmet, three uncovered waves walk into the
    fact), and pure-army (count clause unmet) must ALL LOSE on every
    level + every seed — a real LOSS, no draw."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_wrong_centre_skirt_loses_on_hard_because_fact_flips():
    """A non-adaptive skirt centred on the OLD (64,20) cell (the fact's
    lower-tier location) FAILS the region clauses on every hard seed —
    the fact never sits at x=64 on hard, so every pbox lands >=10 cells
    outside the active corner discs (radius 4). This is the spawn-driven
    discrimination the pack is curated for: a memorised cell list that
    worked on easy/medium does NOT generalise to the hard fact-flip."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        r = run_level(c, make_wrong_centre_skirt(), seed=seed)
        assert r.outcome == "loss", (
            f"hard seed{seed} wrong-centre skirt: must LOSE; "
            f"got {r.outcome} buildings={r.signals.own_buildings}"
        )


# -- determinism ------------------------------------------------------


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_adaptive_skirt(), seed=3)
    b = run_level(c, make_adaptive_skirt(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
