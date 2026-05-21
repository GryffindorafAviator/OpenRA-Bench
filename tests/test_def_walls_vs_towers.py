"""def-walls-vs-towers scenario family, full loop on Rust.

The pack tests PASSIVE-OBSTACLE vs ACTIVE-DEFENSE mitigation: the agent
has ONE budget (3200cr) and must choose how to spend it against an
incoming rush. The budget funds EITHER ~16 inert `brik` concrete walls
(200cr, HP 40000, NO Armament — they channel and delay but kill
nothing) OR 4 active `gun` turrets (800cr, Armament Weapon TurretGun —
the engine's auto-target loop fires them at the rush). The win
predicate makes the choice load-bearing:

* `building_count_gte:{gun, n:4}` ⇒ the agent built the budget worth of
  ACTIVE turrets (4 — cash is tight enough that the wall option is
  mutually exclusive);
* `units_killed_gte:K` ⇒ the load-bearing clause. A walls-only spend
  kills 0 — walls have no weapon — so the clause is never satisfied; an
  active gun cluster shreds the rush and clears it. This is the
  predicate that makes ACTIVE defence (not passive obstruction) the
  only winning play;
* `building_count_gte:{fact,1}` ⇒ the fact must still STAND (the
  PRESENT-TENSE predicate, not `has_building:fact` which is a one-shot
  ever-seen set — CLAUDE.md footgun);
* `within_ticks` paired with `after_ticks` ⇒ a non-finisher (stall,
  walls-only) is a real reachable timeout LOSS (no interrupts ⇒ each
  step is exactly 90 ticks, so max_turns is a hard tick budget the
  `after_ticks` deadline reliably bites in).

The rush arrives as a `scheduled_events: spawn_actors` wave at a fixed
tick (1800) — after the agent has had time to build 4 gun turrets
serially — so the race is fair: the intended build completes before
the wave, while a staller / walls-only spend still cannot satisfy the
kill quota.

The scripted-policy validations prove deterministically that:

* the intended TOWERS policy (4 gun turrets wrapping the active fact)
  WINS every level + every hard seed (1..4);
* walls-only (16 brik — 0 kills) / stall / pure-army all LOSE every
  level + every hard seed — a real LOSS, not a draw;
* the hard tier defines >=2 spawn_point groups (north fact y=14 /
  south fact y=26) so the turret cluster must follow the fact.
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


def _build_and_place(rs, C, kind, cells):
    """Common build-place loop: at each turn, place the next `kind`
    building in `cells` if the previous one finished; queue the next
    build."""
    own_b = rs.get("own_buildings") or []
    n = sum(1 for b in own_b if b.get("type") == kind)
    if n >= len(cells):
        return [C.observe()]
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    cmds = []
    if kind not in prod_items:
        cmds.append(C.build(kind))
    cmds.append(C.place_building(kind, cells[n][0], cells[n][1]))
    return cmds or [C.observe()]


def make_adaptive_towers():
    """Intended TOWERS policy: read the fact's cell from the
    observation on turn 1, then build 4 ACTIVE gun turrets wrapping it.
    On hard the fact flips between y=14 and y=26 by seed, so the turret
    cluster must follow it."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            # 4 cells wrapping the fact at (10, fy), east of it (the
            # rush approaches from the east).
            state["cells"] = [(12, fy - 2), (12, fy), (12, fy + 2), (13, fy)]
        return _build_and_place(rs, C, "gun", state["cells"])

    return policy


def make_walls_only():
    """The PASSIVE counterfactual: spend the whole budget on 16 inert
    `brik` concrete walls (a double barrier east of the fact). Walls
    channel and delay the rush but have NO weapon — they kill nothing,
    so the `units_killed_gte` clause is never satisfied and the win
    never latches; the episode times out → LOSS."""
    state = {"cells": None}

    def policy(rs, C):
        if state["cells"] is None:
            own_b = rs.get("own_buildings") or []
            facts = [b for b in own_b if b.get("type") == "fact"]
            if not facts:
                return [C.observe()]
            fy = facts[0].get("cell_y", facts[0].get("y"))
            state["cells"] = [(14, fy - 7 + i) for i in range(8)] + [
                (16, fy - 7 + i) for i in range(8)
            ]
        return _build_and_place(rs, C, "brik", state["cells"])

    return policy


def pure_army(rs, C):
    """PURE-ARMY: only ever train e1 — never builds a defensive
    building. FAILS the `building_count_gte:gun` clause; a thin
    home-trained rifle screen cannot out-trade the heavier rush band
    either, so the fact often falls → LOSS."""
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    if "e1" not in prod_items:
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


def test_starting_cash_is_the_either_or_budget():
    """Cash is intentionally tight: 3200cr funds EITHER 16 brik walls
    (16×200) OR 4 gun turrets (4×800) — never both. The two options
    are mutually exclusive so the agent must commit to ACTIVE defence."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == 3200, (lvl, c.starting_cash)


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
        # Every pre-placed agent e1 is a corner non-combatant.
        for a in agent_units:
            x, y = a.position
            assert x <= 3 and (y <= 6 or y >= 34), (lvl, a.position)


def test_uses_scheduled_wave_event():
    """The rush arrives via a `scheduled_events: spawn_actors` wave so
    the build race is fair (the wave lands after the intended 4-gun
    build can complete)."""
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


def test_win_requires_a_kill_quota():
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


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: >=2 distinct agent spawn_point groups so the
    fact (and therefore the turret cluster) flips by seed. The two
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


# ── solvency: intended TOWERS wins every level + every hard seed ─────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_towers_win_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_adaptive_towers(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended gun-turret build must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / passive policy LOSES (not draws) ──────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("walls_only", make_walls_only),
        ("pure_army", lambda: pure_army),
    ],
)
def test_passive_and_lazy_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (rush razes fact OR clock), walls-only (inert walls — 0
    kills, kill clause never met, clock runs out), and pure-army (gun
    count clause unmet) must ALL LOSE on every level + every seed — no
    draw. walls-only LOSING is the load-bearing discrimination: a
    passive-obstacle spend, however large, never produces kill output."""
    c = compile_level(load_pack(PACK), level)
    fn = policy_factory()
    for seed in SEEDS:
        r = run_level(c, fn, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


def test_walls_only_produces_zero_kills():
    """The mechanism check: an all-`brik` spend kills NOTHING (walls
    have no weapon). This is why walls LOSE the kill clause and active
    gun turrets WIN it."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        for seed in SEEDS:
            r = run_level(c, make_walls_only(), seed=seed)
            assert r.signals.units_killed == 0, (
                f"{lvl} seed{seed}: walls-only must kill 0 (inert "
                f"obstacle); got {r.signals.units_killed}"
            )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_adaptive_towers(), seed=3)
    b = run_level(c, make_adaptive_towers(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome,
        b.turns,
        b.signals.units_killed,
    ), "same seed must be deterministic"
