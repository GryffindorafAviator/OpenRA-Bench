"""lh-tech-rush-vs-army-rush scenario family, full loop on Rust.

REASONING capability — strategic commitment under a fast incoming
rush: CAPEX (tech investment) vs OPEX (buy army NOW). The agent's tech
is already standing (fact + proc + powr + tent + weap), so both
production queues are open from turn 1; a fast rusher band charges the
construction yard. The fixed 1800-credit budget buys EITHER an
immediate army (cheap units fielded now) OR a tech investment (a 2nd
war factory, an upgrade) that cannot amortise before the rush razes
the base.

The win predicate makes the commitment decision load-bearing:

* `units_killed_gte:K` ⇒ the agent must field a real fighting force
  and blunt the rush (a tech investment fields no army → ~2 kills at
  most from the lone spawn-witness rifleman);
* `building_count_gte:{fact,1}` ⇒ the fact must STILL stand (present-
  tense predicate, not the one-shot `has_building` set — CLAUDE.md
  footgun); stalling or teching up loses the fact to the rush;
* `within_ticks` paired with `after_ticks: T+1` in the fail clause ⇒
  a non-finisher is a real reachable timeout LOSS, never a draw.

The scripted-policy validations prove deterministically that:

* the intended ARMY-RUSH (commit the budget to cheap units NOW and
  defend the fact) WINS every level + every hard seed (1..4);
* stall (observe only) and tech-rush (buy a 2nd war factory) both
  LOSE every level + every hard seed — a real fact-razed / timeout
  LOSS, not a draw;
* the hard tier defines 2 agent spawn_point groups (NORTH y=12 /
  SOUTH y=28) so a memorised army-placement cell cannot generalise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-tech-rush-vs-army-rush.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — the agent never spends. The rusher band razes the
    fact (fail clause `not building_count_gte:{fact,1}`)."""
    return [C.observe()]


def tech_rush(rs, C):
    """CAPEX trap: pour the whole budget into a SECOND war factory
    (throughput investment). No army fields, the rush razes the fact
    before the investment could ever amortise → LOSS."""
    own_b = rs.get("own_buildings") or []
    n_weap = sum(1 for b in own_b if b.get("type") == "weap")
    items = [
        p.get("item") for p in (rs.get("production") or []) if isinstance(p, dict)
    ]
    fb = [b for b in own_b if b.get("type") == "fact"]
    if n_weap < 2 and fb:
        cmds = []
        if "weap" not in items:
            cmds.append(C.build("weap"))
        cmds.append(C.place_building("weap", fb[0]["cell_x"] + 8, fb[0]["cell_y"]))
        return cmds
    return [C.observe()]


def army_rush(rs, C):
    """Intended OPEX play: commit the whole budget to cheap units NOW
    (mass infantry, e1 $100 — the fastest, highest-body-count buy) and
    defend the fact. The army stands up in ~3-5 turns and blunts the
    rush before it reaches the construction yard → WIN."""
    items = [
        p.get("item") for p in (rs.get("production") or []) if isinstance(p, dict)
    ]
    cmds = []
    if "e1" not in items:
        cmds.append(C.build("e1"))
    fb = [b for b in (rs.get("own_buildings") or []) if b.get("type") == "fact"]
    ids = [
        u.get("id")
        for u in (rs.get("units_summary") or [])
        if u.get("id") is not None
    ]
    if ids and fb:
        # Hold ~12 cells east of the actual fact (derived from the
        # seed-driven base position, not a hardcoded cell).
        cmds.append(C.attack_move(ids, fb[0]["cell_x"] + 12, fb[0]["cell_y"]))
    if not cmds:
        cmds.append(C.observe())
    return cmds


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-tech-rush-vs-army-rush"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("tech-vs-army" in a for a in anchors), anchors
    assert any("capex" in a for a in anchors), anchors
    assert any("commitment" in a for a in anchors), anchors
    # The fast-rush regime is driven by the scripted rusher bot.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_tech_already_pre_placed_each_level():
    """fact + proc + powr + tent + weap must all be pre-placed (the
    tech is up — the decision is the SPEND, not the build chain)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {
            a.type for a in c.scenario.actors if a.owner == "agent"
        }
        for t in ("fact", "proc", "powr", "tent", "weap"):
            assert t in agent_types, (lvl, t, agent_types)


def test_starting_cash_is_1800_each_level():
    """1800 funds an army NOW but is SHORT of a 2nd war factory
    ($2000) — the tech-rush is a structural dead-end."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == 1800, (lvl, c.starting_cash)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns."""
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
    """The fact-survival fail clause must use the PRESENT-TENSE
    predicate (`building_count_gte:{type:fact,n:1}`) not `has_building`
    (a one-shot 'ever seen' set — CLAUDE.md footgun) so the rush
    razing the fact triggers a real LOSS."""
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


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so a
    memorised army-placement cell cannot generalise."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended ARMY-RUSH wins every level + every hard seed ──


@pytest.mark.parametrize("level", LEVELS)
def test_intended_army_rush_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, army_rush, seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended army-rush must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: stall and tech-rush LOSE (not draw) every level/seed ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy",
    [("stall", stall), ("tech_rush", tech_rush)],
)
def test_lazy_and_tech_rush_policies_lose_every_level_and_seed(
    level, policy_name, policy
):
    """Stall (rush razes the fact) and tech-rush (a 2nd war factory
    drains the budget with no army) must BOTH LOSE on every level +
    every seed — a real fact-razed / timeout LOSS, never a draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed})"
        )
