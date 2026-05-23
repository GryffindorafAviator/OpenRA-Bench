"""def-walls-vs-towers scenario family, full loop on Rust.

The pack tests TIER-DIFFERENTIATED defensive DOCTRINE selection: the
agent must choose between PASSIVE OBSTACLE (`brik` concrete walls —
200cr, HP 40000, NO Armament) and ACTIVE TURRET (`pbox` pillbox —
600cr, M60mg burst — anti-infantry direct fire). The correct posture
DIFFERS PER TIER:

* EASY    — frontal rush in the open. 4 pbox cluster wraps the fact
            and shreds the wave. Walls kill nothing → LOSS.
* MEDIUM  — overwhelming horde funnels on one lane. Win predicate
            requires BOTH a wall belt AND a pbox at the choke AND a
            kill quota: a wall barrier funnels the horde to a single
            pass where the pbox serially shreds the queue. Pure-walls
            (0 kills) and pure-pbox (no brik clause) both LOSE.
* HARD    — two simultaneous waves + seed-driven fact flip. The agent
            must allocate budget across both postures (2 pbox + 8
            brik). Pure-walls and pure-pbox both LOSE.

The scripted-policy validations prove deterministically that:

* the intended PER-TIER doctrine WINS every seed (1..4);
* stall, pure-walls (brik-only), pure-pbox (no brik on medium/hard),
  and pure-army all LOSE every seed — a real LOSS, not a draw;
* the hard tier defines >=2 spawn_point groups (north fact y=14 /
  south fact y=26) so the defence cluster must follow the fact.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-walls-vs-towers.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. The rush razes the fact
    and/or the clock runs out."""
    return [C.observe()]


def _prod_items(rs):
    """The Defense queue surfaces as a flat list of strings — handle
    that AND the legacy dict-with-item shape robustly."""
    return [
        (p if isinstance(p, str) else p.get("item"))
        for p in (rs.get("production") or [])
    ]


def _build_and_place(rs, C, kind, cells):
    """Place the next `kind` building in `cells` if the previous one
    finished; queue the next build only if not already in the queue."""
    own_b = rs.get("own_buildings") or []
    n = sum(1 for b in own_b if b.get("type") == kind)
    if n >= len(cells):
        return [C.observe()]
    cmds = []
    if kind not in _prod_items(rs):
        cmds.append(C.build(kind))
    cmds.append(C.place_building(kind, cells[n][0], cells[n][1]))
    return cmds or [C.observe()]


def make_pbox_only(cells_fn):
    """Pure-ACTIVE policy: build N pbox at `cells_fn(fy)`. Wraps the
    fact with active turrets that shoot the rush. Wins on easy
    (predicate matches) and LOSES on medium/hard (no brik clause)."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            state["cells"] = cells_fn(fy)
        return _build_and_place(rs, C, "pbox", state["cells"])

    return policy


def make_brik_only(cells_fn):
    """Pure-PASSIVE policy: build N brik (inert walls). The brik clause
    may pass on medium/hard but `units_killed_gte` never does — walls
    have no weapon. LOSES every tier."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            state["cells"] = cells_fn(fy)
        return _build_and_place(rs, C, "brik", state["cells"])

    return policy


def make_intended_easy():
    """EASY doctrine: 4 pbox wrapping the fact on the east approach."""
    return make_pbox_only(
        lambda fy: [(12, fy - 2), (12, fy), (12, fy + 2), (13, fy)]
    )


def _build_combo(rs, C, plan):
    """Two-phase build: complete all pbox first, then the brik belt.
    Pbox first because they kill — building the wall belt before the
    pbox lets the horde reach the fact before kill output is online."""
    own_b = rs.get("own_buildings") or []
    n_pbox = sum(1 for b in own_b if b.get("type") == "pbox")
    n_brik = sum(1 for b in own_b if b.get("type") == "brik")
    prod_items = _prod_items(rs)
    cmds = []
    if n_pbox < len(plan["pbox"]):
        if "pbox" not in prod_items:
            cmds.append(C.build("pbox"))
        cmds.append(C.place_building("pbox", plan["pbox"][n_pbox][0], plan["pbox"][n_pbox][1]))
    elif n_brik < len(plan["brik"]):
        if "brik" not in prod_items:
            cmds.append(C.build("brik"))
        cmds.append(C.place_building("brik", plan["brik"][n_brik][0], plan["brik"][n_brik][1]))
    else:
        cmds.append(C.observe())
    return cmds


def make_intended_medium():
    """MEDIUM doctrine: 1 pbox at the choke + 8 brik wall belt that
    leaves a single-cell pass at the fact's y, funnelling the horde
    onto the pbox burst."""
    state = {"plan": None}

    def policy(rs, C):
        if state["plan"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            state["plan"] = {
                "pbox": [(12, fy)],
                "brik": [
                    (14, fy - 4), (14, fy - 3), (14, fy - 2), (14, fy - 1),
                    (14, fy + 1), (14, fy + 2), (14, fy + 3), (14, fy + 4),
                ],
            }
        return _build_combo(rs, C, state["plan"])

    return policy


def make_intended_hard():
    """HARD doctrine: 2 pbox at the choke + 8 brik full-lane wall belt
    that blocks BOTH attack vectors converging on the fact. The pbox
    cluster sits behind the wall belt where it can shred whatever
    funnels through the single-cell gap left by the wall."""
    state = {"plan": None}

    def policy(rs, C):
        if state["plan"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            # 2nd pbox is placed on the FACT side (north for the north
            # fact, south for the south fact) to cover the off-lane
            # approach as it walks past.
            second_y = fy - 2 if fy == 14 else fy + 2
            state["plan"] = {
                "pbox": [(12, fy), (12, second_y)],
                "brik": [
                    (13, fy - 3), (13, fy - 2), (13, fy - 1),
                    (13, fy + 1), (13, fy + 2), (13, fy + 3),
                    (13, fy + 4), (13, fy - 4),
                ],
            }
        return _build_combo(rs, C, state["plan"])

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a defensive
    building. FAILS the `building_count_gte:pbox` clause; a thin
    home-trained rifle screen cannot out-trade the heavier rush band
    either, so the fact often falls → LOSS."""
    if "e1" not in _prod_items(rs):
        return [C.build("e1")]
    return [C.observe()]


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-walls-vs-towers"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (CLAUDE.md / pack spec).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("security architecture" in a for a in anchors), pack.meta.benchmark_anchor
    assert any(
        "passive vs active mitigation" in a for a in anchors
    ), pack.meta.benchmark_anchor
    assert any(
        "military fortification" in a for a in anchors
    ), pack.meta.benchmark_anchor
    # rusher bot wired through (charges agent centroid → the rush
    # converges on the fact regardless of seed).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_starting_cash_is_per_tier_doctrine_budget():
    """Cash is sized per tier so the intended doctrine fits and the
    wrong doctrine does not: easy/medium fund either 4 pbox (4×600) or
    12 brik (12×200) within 2400cr; hard funds 2 pbox + 8 brik = 2800
    within 3000cr."""
    pack = load_pack(PACK)
    cash_by_level = {lvl: compile_level(pack, lvl).starting_cash for lvl in LEVELS}
    assert cash_by_level["easy"] == 2400, cash_by_level
    assert cash_by_level["medium"] == 2400, cash_by_level
    assert cash_by_level["hard"] == 3000, cash_by_level


def test_no_preplaced_combat_units_near_the_base():
    """The agent's BUILT defences must be the sole source of kill
    output. The only pre-placed agent unit is a single non-combatant
    e1 parked in a far map corner (x<=3, far from every y-lane the
    rush uses) so it never reaches combat — a walls-only / stall play
    cannot pass the kill clause off a pre-placed defender."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_units = [
            a
            for a in c.scenario.actors
            if a.owner == "agent" and a.type == "e1"
        ]
        for a in agent_units:
            x, y = a.position
            assert x <= 3 and (y <= 6 or y >= 34), (lvl, a.position)


def test_uses_scheduled_wave_event():
    """The rush arrives via a `scheduled_events: spawn_actors` wave so
    the build race is fair (the wave lands after the intended build
    can complete)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        evs = list(c.scheduled_events or [])
        assert evs, f"{lvl}: missing scheduled rush wave"
        assert any(
            e.get("type") == "spawn_actors" for e in evs
        ), f"{lvl}: scheduled event is not a spawn_actors wave"


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
            clause
            for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type") == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive fail clause"


def test_win_requires_a_kill_quota_per_tier():
    """The load-bearing clause: every level's win requires
    `units_killed_gte` — the predicate a walls-only (0-kill) spend
    cannot satisfy. Without it, inert walls would win."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        kill_clauses = [
            clause
            for clause in wc.get("all_of", []) or []
            if isinstance(clause, dict) and "units_killed_gte" in clause
        ]
        assert kill_clauses, f"{lvl}: win condition has no units_killed_gte clause"
        assert int(kill_clauses[0]["units_killed_gte"]) >= 4, lvl


def test_medium_and_hard_require_wall_plus_tower_combo():
    """Tier-differentiated doctrine: medium/hard predicates must require
    BOTH a pbox AND a brik clause. Without the brik clause a pure-pbox
    spend would win medium/hard and the tier doctrine would collapse to
    "always build towers"."""
    for lvl in ("medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        all_of = wc.get("all_of") or []
        pbox_clauses = [
            cl for cl in all_of
            if isinstance(cl, dict)
            and (cl.get("building_count_gte") or {}).get("type") == "pbox"
        ]
        brik_clauses = [
            cl for cl in all_of
            if isinstance(cl, dict)
            and (cl.get("building_count_gte") or {}).get("type") == "brik"
        ]
        assert pbox_clauses, f"{lvl}: missing pbox count clause"
        assert brik_clauses, f"{lvl}: missing brik count clause"
        # The brik count must require a genuine wall belt, not 1-2
        # incidental walls — otherwise a 4-pbox spend that happens to
        # toss out a couple of walls passes the predicate.
        assert int(brik_clauses[0]["building_count_gte"]["n"]) >= 6, lvl


def test_easy_requires_pbox_not_brik():
    """Easy tier is the pure-active-doctrine tier: the predicate must
    require pbox and NOT require any brik (a 4-pbox spend is the
    intended easy win, and forcing brik on easy would collapse the
    tier differentiation)."""
    c = compile_level(load_pack(PACK), "easy")
    wc = c.win_condition.model_dump(exclude_none=True)
    all_of = wc.get("all_of") or []
    pbox_clauses = [
        cl for cl in all_of
        if isinstance(cl, dict)
        and (cl.get("building_count_gte") or {}).get("type") == "pbox"
    ]
    brik_clauses = [
        cl for cl in all_of
        if isinstance(cl, dict)
        and (cl.get("building_count_gte") or {}).get("type") == "brik"
    ]
    assert pbox_clauses, "easy: missing pbox count clause"
    assert not brik_clauses, "easy: must not require brik (pure-active doctrine)"


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: >=2 distinct agent spawn_point groups so the
    fact (and therefore the defence cluster) flips by seed. The two
    groups must define the NORTH (y=14) and SOUTH (y=26) fact pair."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    fact_ys = sorted(
        {
            a.position[1]
            for a in c.scenario.actors
            if a.owner == "agent" and a.type == "fact"
        }
    )
    assert fact_ys == [14, 26], fact_ys
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended per-tier doctrine wins every seed ─────────────


def test_intended_easy_pbox_wins_every_seed():
    c = compile_level(load_pack(PACK), "easy")
    for seed in SEEDS:
        r = run_level(c, make_intended_easy(), seed=seed)
        assert r.outcome == "win", (
            f"easy seed{seed}: intended 4-pbox cluster must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_intended_medium_walls_plus_tower_wins_every_seed():
    c = compile_level(load_pack(PACK), "medium")
    for seed in SEEDS:
        r = run_level(c, make_intended_medium(), seed=seed)
        assert r.outcome == "win", (
            f"medium seed{seed}: intended 1 pbox + 8 brik choke must "
            f"WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_intended_hard_mixed_doctrine_wins_every_seed():
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        r = run_level(c, make_intended_hard(), seed=seed)
        assert r.outcome == "win", (
            f"hard seed{seed}: intended 2 pbox + 8 brik mixed doctrine "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: stall + wrong doctrine all LOSE every tier × seed ──────


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Stall (observe-only): the rush razes the fact OR the clock runs
    out → real LOSS, not a draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE; got {r.outcome}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_pure_walls_loses_every_level_and_seed(level):
    """Pure-walls (brik-only): walls have no weapon, so `units_killed`
    stays at zero on every tier — the kill clause is never satisfied
    and the episode times out → LOSS. The mechanism-level proof that
    passive obstruction alone is never the right doctrine for ANY
    tier (even where walls are PART of the right doctrine)."""
    c = compile_level(load_pack(PACK), level)
    pol = make_brik_only(lambda fy: [(14, fy - 6 + i) for i in range(12) if fy - 6 + i != fy])
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} pure-walls: must LOSE; got {r.outcome} "
            f"(kills={r.signals.units_killed})"
        )
        # Mechanism check: pure-walls always kills zero, on every tier.
        assert r.signals.units_killed == 0, (
            f"{level} seed{seed}: pure-walls must kill 0 (inert "
            f"obstacle); got {r.signals.units_killed}"
        )


def test_pure_pbox_wins_easy_but_loses_medium_and_hard():
    """The TIER-DIFFERENTIATION proof: a 4-pbox cluster (the easy
    doctrine, hard-coded as always-towers) WINS easy but LOSES
    medium and hard — proving the predicate / geometry genuinely
    forces a different doctrine per tier. An "always build towers"
    play that memorises easy fails the higher tiers."""
    pack = load_pack(PACK)

    def cells(fy):
        return [(12, fy - 2), (12, fy), (12, fy + 2), (13, fy)]

    # Easy — pure-pbox is the correct doctrine, must WIN.
    c_easy = compile_level(pack, "easy")
    for seed in SEEDS:
        r = run_level(c_easy, make_pbox_only(cells), seed=seed)
        assert r.outcome == "win", (
            f"easy seed{seed} pure-pbox: must WIN (the easy doctrine); "
            f"got {r.outcome}"
        )

    # Medium + hard — pure-pbox fails the brik clause, must LOSE.
    for lvl in ("medium", "hard"):
        c = compile_level(pack, lvl)
        for seed in SEEDS:
            r = run_level(c, make_pbox_only(cells), seed=seed)
            assert r.outcome == "loss", (
                f"{lvl} seed{seed} pure-pbox: must LOSE (wrong "
                f"doctrine — no wall belt); got {r.outcome} "
                f"(kills={r.signals.units_killed})"
            )


@pytest.mark.parametrize("level", LEVELS)
def test_pure_army_loses_every_level_and_seed(level):
    """Pure-army (only train e1): fails the pbox count clause and a
    thin home-trained rifle screen cannot out-trade the heavier rush
    band, so the fact often falls → LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, pure_army, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} pure-army: must LOSE; got {r.outcome}"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended_easy(), seed=3)
    b = run_level(c, make_intended_easy(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
