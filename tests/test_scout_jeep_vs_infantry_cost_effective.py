"""scout-jeep-vs-infantry-cost-effective — REASONING capability validation.

Cost-effective reconnaissance under an intelligence budget. Starting
cash $900 funds EITHER:

  (A) BUILD 1× jeep ($600, wheeled, sight 7c) — the cost-effective
      optimum. Fast chassis + fast queue + widest sight = fastest
      eyes at the far frontier. INTENDED — WIN every tier / seed.

  (B) BUILD 1× 1tnk ($700, tracked, sight 6c) — affordable on the
      budget but TRACKED is slower than wheeled AND the queue is
      slower (build ≈ 630 ticks vs jeep's 540). On medium / hard
      the tank arrives ~tick 1290, past within_ticks 1200 ⇒ LOSS.

  (C) BUILD 9× e1 ($100 ea = $900, foot, sight 4c) — cheapest body
      but the slowest mover; foot speed ~15 t/cell × 96 cells ≈
      1440 ticks just for travel after a quick 90-tick build ⇒
      first e1 arrives ~tick 1530 on medium / hard, past
      within_ticks 1200 ⇒ LOSS.

  (D) STALL (observe only) — no unit reaches the frontier; after_ticks
      bites ⇒ LOSS, never DRAW.

Anchor: SC2 Reaper / worker-scout economy; OR cost-effective recon
(intelligence budget). The capability is cost-vs-speed reasoning:
cheaper-per-body is NOT cheaper-per-info when the metric is "eyes at
the frontier inside the clock", and more-expensive-per-body buys
negative marginal speed when the dedicated scout chassis is cheaper.

Bar (CLAUDE.md "no defect, no cheat, no draw"):
  • stall (observe only) LOSES every tier / every hard seed —
    no unit enters the frontier region ⇒ after_ticks bites.
  • build-only-1tnk LOSES on medium / hard — tracked + slow queue
    arrives past within_ticks 1200.
  • build-only-e1 LOSES on medium / hard — foot speed too slow to
    traverse 96 cells inside the deadline.
  • walk-the-pre-placed-e1-east LOSES on medium / hard (hard's
    inert seed-witness e1 has foot speed too slow).
  • intended build-jeep WINS every tier / every hard seed —
    arrives ~tick 1100, well inside the deadline.
  • hard tier defines ≥2 agent spawn_point groups (NORTH base
    y=14 / SOUTH base y=26) round-robined by seed — the cost-vs-
    speed reasoning is symmetric across spawns, the frontier
    marker is equidistant, a memorised "always start at (10,20)"
    opening cannot generalise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "scout-jeep-vs-infantry-cost-effective.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _stall(rs, C):
    """Observe-only — no commit, no movement. No unit enters the
    frontier region ⇒ after_ticks bites ⇒ LOSS."""
    return [C.observe()]


def _build_jeep(rs, C):
    """The intended capability: queue a jeep from the pre-placed
    war factory, then send any built jeeps east to the frontier
    region centred at (110, 20). Wheeled chassis + jeep queue is
    fast enough to comfortably clear within_ticks on every tier."""
    cmds = [C.build("jeep")]
    own_u = rs.get("units_summary") or []
    jeep_ids = [u["id"] for u in own_u if str(u.get("type", "")).lower() == "jeep"]
    if jeep_ids:
        # attack_move not in this pack's tool palette; move_units
        # is sufficient (jeep has no interesting auto-fire targets
        # — the only enemy actor is the unarmed frontier `fact`).
        cmds.append(C.move_units(jeep_ids, 110, 20))
    return cmds


def _build_1tnk(rs, C):
    """Wrong commit (over-spend): queue a single 1tnk light tank from
    the pre-placed war factory and send it east. Tracked + slower
    queue ⇒ arrives past within_ticks 1200 on medium / hard ⇒ LOSS."""
    cmds = [C.build("1tnk")]
    own_u = rs.get("units_summary") or []
    tank_ids = [u["id"] for u in own_u if str(u.get("type", "")).lower() == "1tnk"]
    if tank_ids:
        cmds.append(C.move_units(tank_ids, 110, 20))
    return cmds


def _build_e1(rs, C):
    """Wrong commit (under-spend): spam riflemen from the pre-placed
    barracks and march them east. Foot speed ~15 t/cell × 96 cells
    ≈ 1440 ticks just for travel ⇒ arrives past within_ticks 1200
    on medium / hard ⇒ LOSS."""
    cmds = [C.build("e1")]
    own_u = rs.get("units_summary") or []
    e1_ids = [u["id"] for u in own_u if str(u.get("type", "")).lower() == "e1"]
    if e1_ids:
        cmds.append(C.move_units(e1_ids, 110, 20))
    return cmds


# ── structural / metadata tests ─────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-jeep-vs-infantry-cost-effective"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    # Anchored to the SC2 Reaper / OR cost-effective recon /
    # intelligence budget framing per the Wave-8 spec.
    assert "sc2 reaper" in anchors, anchors
    assert "or cost-effective recon" in anchors, anchors
    assert "intelligence budget" in anchors, anchors


def test_starting_cash_is_900():
    """Cash $900 funds EITHER 1× jeep ($600) OR 1× 1tnk ($700) OR
    9× e1 ($900). Heavier tanks (2tnk $850) need the `fix` allied
    service depot which is intentionally NOT pre-placed, so the
    decision is among the three buildable options the brief
    enumerates."""
    pack = load_pack(PACK)
    assert pack.starting_cash == 900, (
        f"starting_cash must be 900 (the intelligence-budget pivot); "
        f"got {pack.starting_cash}"
    )
    # Sanity on the budget arithmetic referenced by the brief.
    assert 1 * 600 == 600          # 1× jeep
    assert 1 * 700 == 700          # 1× 1tnk
    assert 9 * 100 == 900          # 9× e1


def test_tools_are_recon_only_no_attack_no_place_building():
    """Pure-recon tool surface: [build, move_units, stop]. NO attack
    tools (the mission is OBSERVE, not engage) and NO
    place_building (the base is fully pre-placed)."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    assert tools == {"build", "move_units", "stop"}, (
        f"tools must be exactly {{build, move_units, stop}}; got {sorted(tools)}"
    )


def test_preplaced_base_has_fact_proc_powr_tent_weap_on_every_level():
    """Pre-placed on every level: fact (loss-critical), proc (weap
    prereq + economy), powr (queue power), tent (e1 production —
    keeps the under-spend option actionable), weap (jeep / 1tnk
    production). Each candidate option must be ACTIONABLE on turn
    1 so the model is graded on PICKING the right scout, not on
    which queues are open."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        groups: dict[int, set[str]] = {}
        for a in c.scenario.actors:
            if a.owner != "agent":
                continue
            g = a.spawn_point if a.spawn_point is not None else 0
            groups.setdefault(g, set()).add(a.type)
        assert groups, f"{lvl}: no agent actors found"
        for g, ts in groups.items():
            for need in ("fact", "proc", "powr", "tent", "weap"):
                assert need in ts, (
                    f"{lvl}: spawn group {g} missing {need!r}; got {sorted(ts)}"
                )


def test_no_fix_so_heavy_tanks_are_unbuildable():
    """2tnk needs the allied service depot `fix` as a tech prereq.
    `fix` is intentionally NOT pre-placed so the heavy-armour
    temptation (save cash for a $850 medium tank) is mechanically
    blocked, keeping the comparison crisp on the three options
    the brief enumerates (jeep / 1tnk / e1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {a.type for a in c.scenario.actors if a.owner == "agent"}
        assert "fix" not in agent_types, (
            f"{lvl}: `fix` must NOT be pre-placed (heavy tanks must be "
            f"unbuildable to keep the cost-vs-speed comparison crisp); "
            f"got {sorted(agent_types)}"
        )


def test_frontier_marker_present_and_far():
    """The persistent unarmed enemy `fact` marker at the frontier
    keeps the engine auto-`done` path gated AND serves as the in-
    universe landmark the brief names. It must be at (110, 20) on
    every level (the win region's centre)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        enemy_facts = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert len(enemy_facts) == 1, (
            f"{lvl}: need exactly 1 enemy `fact` marker (got "
            f"{len(enemy_facts)})"
        )
        x, y = enemy_facts[0].position
        assert (x, y) == (110, 20), (
            f"{lvl}: enemy fact must be at the frontier (110, 20); "
            f"got ({x}, {y})"
        )


def test_every_level_has_reachable_timeout_fail():
    """`after_ticks` fail must bite WITHIN max_turns (so stall is a
    real reachable LOSS, not a draw). within_ticks + 1 == after_ticks
    so a non-finisher fails the very next tick after the window
    closes."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        wt = next(
            int(c["within_ticks"])
            for c in L.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt < ceiling, f"{lvl}: within_ticks {wt} >= ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            "(boundary non-finisher must LOSE, not draw — fail one "
            "tick past win)"
        )


def test_within_ticks_tier_axis():
    """Easy is LOOSE (within_ticks 1800) — bare 'commit to go east'
    skill; ALL of jeep / 1tnk / e1 wins. Medium / hard are TIGHT
    (within_ticks 1200) — only the jeep makes it inside the
    deadline. CLAUDE.md tier cadence: one new controlled variable
    per tier."""
    pack = load_pack(PACK)
    wts = {}
    for lvl in LEVELS:
        L = pack.levels[lvl]
        wts[lvl] = next(
            int(c["within_ticks"])
            for c in L.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
    assert wts["easy"] == 1800, wts
    assert wts["medium"] == 1200, wts
    assert wts["hard"] == 1200, wts
    assert wts["easy"] > wts["medium"] == wts["hard"], wts


def test_every_level_has_a_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )
    assert sp == {0, 1}, f"expected exactly {{0, 1}}; got {sorted(sp)}"


def test_in_bounds_actors_on_every_level():
    """rush-hour-arena playable bounds ≈ x:2..126, y:2..38."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level checks (no engine) ──────────────────────────────


def _ctx(*, tick=0, units=(), own_buildings=(("fact", 10, 20),)):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=0,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": list(units)})


def _at_frontier(unit_type="jeep"):
    return [{"cell_x": 110, "cell_y": 20, "type": unit_type, "id": "1001"}]


def _at_base(unit_type="jeep"):
    return [{"cell_x": 15, "cell_y": 20, "type": unit_type, "id": "1001"}]


def test_predicates_enforce_capability():
    """Win requires (unit in frontier region AND fact alive AND in
    time); fail fires on timeout OR fact destroyed."""
    c = compile_level(load_pack(PACK), "medium")

    # Intended: jeep at frontier, fact alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(tick=1100, units=_at_frontier("jeep")))
    # Unit type doesn't matter — any unit in the region wins (the
    # cost-vs-speed tooth is the CLOCK, not the type filter).
    assert evaluate(c.win_condition, _ctx(tick=1100, units=_at_frontier("1tnk")))
    assert evaluate(c.win_condition, _ctx(tick=1100, units=_at_frontier("e1")))
    # No unit at frontier → not win
    assert not evaluate(c.win_condition, _ctx(tick=1100, units=_at_base("jeep")))
    # Jeep at frontier but past within_ticks → not win
    assert not evaluate(
        c.win_condition, _ctx(tick=1201, units=_at_frontier("jeep"))
    )
    # Jeep at frontier but fact destroyed → not win
    assert not evaluate(
        c.win_condition,
        _ctx(tick=1100, units=_at_frontier("jeep"), own_buildings=()),
    )
    # Fact destroyed → fail
    assert evaluate(c.fail_condition, _ctx(tick=1100, own_buildings=()))
    # Past after_ticks deadline → fail
    assert evaluate(c.fail_condition, _ctx(tick=1201))
    # Within deadline, fact alive → not fail
    assert not evaluate(c.fail_condition, _ctx(tick=1100))


# ── engine-driven: every lazy / wrong policy LOSES, intended WINS ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """Observe-only ⇒ no unit reaches the frontier ⇒ real LOSS,
    not a draw. The `after_ticks` clause is reachable inside
    max_turns on every tier."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (no unit at frontier ⇒ "
        f"after_ticks bites); got {r.outcome} "
        f"tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_build_only_1tnk_loses_on_tight_tiers(level, seed):
    """Build a single light tank and drive it east ⇒ tracked +
    slower queue ⇒ arrives past within_ticks 1200 on medium / hard
    ⇒ LOSS. (Easy admits it — loose clock by design; that's the
    deliberate easy-tier reduction to bare 'commit to go east'.)"""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _build_1tnk, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: build-1tnk must LOSE on tight tiers "
        f"(tracked too slow); got {r.outcome} "
        f"tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_build_only_e1_loses_on_tight_tiers(level, seed):
    """Spam riflemen and march them east ⇒ foot speed too slow ⇒
    LOSS on medium / hard. (Easy admits it — loose clock by design.)"""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _build_e1, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: build-e1 must LOSE on tight tiers "
        f"(foot too slow); got {r.outcome} "
        f"tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_jeep_wins_every_tier_and_seed(level, seed):
    """The intended capability — queue a jeep from the pre-placed
    war factory and drive it east to the frontier. Wheeled chassis
    + fastest queue ⇒ arrives ~tick 1100, comfortably inside
    within_ticks on every tier."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _build_jeep, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: build-jeep must WIN; got {r.outcome} "
        f"tick={r.signals.game_tick}"
    )


# ── determinism ─────────────────────────────────────────────────────


def test_build_jeep_run_is_deterministic_per_seed():
    """Same seed, same policy → identical outcome / turns."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _build_jeep, seed=2)
    b = run_level(c, _build_jeep, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
