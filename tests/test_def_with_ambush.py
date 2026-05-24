"""def-with-ambush scenario pack — Rust engine full loop.

REASONING capability — FOG-AMBUSH doctrine: 2tnk flankers pre-placed
in concealment OFF the direct enemy axis on stance:0 (HoldFire) so
they sit silent. The agent's load-bearing decision is WHEN to
TRIGGER the ambush — flip the flankers from HoldFire (stance:0) to
AttackAnything (stance:3) once the enemy column has walked into the
L-ambush crossfire envelope. A stall / observe-only policy keeps the
flankers silent → no engagement, no kills, the band rolls over the
rifleman screen and razes the construction yard. The intended play
is a single reactive `set_stance(<flanker_ids>, 3)` once the band
closes into the kill zone.

Wrong-play taxonomy (every level + every hard seed must LOSE):

* STALL — observe-only. stance:0 flankers stay silent forever; the
  band reaches the base unopposed → kill bar missed, fact razed or
  deadline bites. LOSS.
* CHARGE — attack-move the flankers EAST out of the crossfire
  positions to meet the band in the open. With no
  sprint-invincibility the charging tanks take fire while closing,
  string out, and meet the heavier band head-on with no crossfire →
  out-attritioned, kill bar missed. LOSS.
* PILE-ON-FACT — move the flankers OUT of the hidden crossfire
  positions and stack them on the construction yard. The crossfire
  envelope at the lane is abandoned; the thin force is crowded
  against the building and ground down → kill bar missed / fact
  razed / attrition cap busted. LOSS.

The bar:

* `building_count_gte:{type: fact, n: 1}` — fact must survive.
* `units_killed_gte: K` — real engagement bar matched per tier
  (6 easy, 8 medium, 8 hard) so a triggered crossfire clears it
  while stall / charge / pile-on cannot.
* `units_lost_lte: N` — the attrition cap (2 easy, 3 medium/hard)
  that bites CHARGE and PILE-ON.
* `within_ticks: 2400` paired with `after_ticks: 2401` — a non-
  finisher is a real reachable timeout LOSS, never a draw.

These tests prove deterministically (no model / no network):

* the intended TRIGGER play WINS every level + every hard seed;
* stall, charge, and pile-on-fact each LOSE every level and every
  hard seed (a real LOSS, not a draw);
* the hard tier defines ≥2 spawn_point groups so a memorised
  absolute crossfire cell cannot generalise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-with-ambush.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """STALL — observe-only. stance:0 flankers stay silent forever;
    the band advances unopposed, razes the rifleman screen, and runs
    out the kill-bar / attrition / deadline clock. LOSS every level +
    every seed."""
    return [C.observe()]


def intended_ambush(rs, C):
    """Trigger the ambush: flip every 2tnk flanker from stance:0
    (HoldFire) to stance:3 (AttackAnything) once any visible enemy
    has closed within ~10 cells of any flanker (i.e. into the
    crossfire envelope). The engine then auto-engages the band from
    both flanks. WINS every level + every hard seed."""
    units = rs.get("units_summary") or []
    flankers = [u for u in units if (u.get("type") or "").lower() == "2tnk"]
    if not flankers:
        return [C.observe()]
    enemies = rs.get("enemy_summary") or []
    for f in flankers:
        for e in enemies:
            if abs(f["cell_x"] - e["cell_x"]) + abs(
                f["cell_y"] - e["cell_y"]
            ) <= 10:
                return [
                    C.set_stance([str(u["id"]) for u in flankers], 3)
                ]
    return [C.observe()]


def charge(rs, C):
    """Charge the flankers OUT of the ambush — attack-move them
    EAST at the band. The crossfire envelope is abandoned on turn
    1; with the engine no-sprint-invincibility fix the charging
    tanks take fire while closing, string out along the lane, and
    meet the heavier band head-on with no crossfire support. LOSS
    every level + every hard seed."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    east_target_x = max((u["cell_x"] for u in units), default=40) + 40
    flank_y = units[0]["cell_y"]
    enemies = rs.get("enemy_summary", []) or []
    target_y = enemies[0]["cell_y"] if enemies else flank_y
    return [C.attack_move(ids, target_x=east_target_x, target_y=target_y)]


def pile_on_fact(rs, C):
    """Pile the flankers back onto the construction yard. The
    crossfire envelope at the lane is abandoned; the thin force is
    stacked against the building and ground down by the band. LOSS
    every level + every hard seed."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    facts = [
        b for b in (rs.get("own_buildings", []) or [])
        if (b.get("type") or "").lower() == "fact"
    ]
    if facts:
        tx, ty = int(facts[0]["cell_x"]), int(facts[0]["cell_y"])
    else:
        tx, ty = 10, 20
    return [C.move_units(ids, target_x=tx, target_y=ty)]


# ── structural checks (no engine) ────────────────────────────────────


def test_pack_loads_and_metadata_is_complete():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-with-ambush"
    assert pack.meta.capability == "reasoning"
    anchors = [str(a).lower() for a in (pack.meta.benchmark_anchor or [])]
    assert anchors, "benchmark_anchor must be non-empty"
    assert any("sc2" in a and "hidden" in a for a in anchors), anchors
    assert any("ambush" in a for a in anchors), anchors
    assert any("fog" in a for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = (
            getattr(enemy, "bot_type", None)
            or getattr(enemy, "bot", None)
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


def test_set_stance_is_in_base_tools():
    """The agent must be able to TRIGGER the ambush via set_stance —
    that's the load-bearing verb. Without it, no policy can flip the
    stance:0 flankers to stance:3, so no policy can win."""
    pack = load_pack(PACK)
    base = pack.base if isinstance(pack.base, dict) else {}
    tools = set(base.get("tools", []) or [])
    assert "set_stance" in tools, f"set_stance must be in base tools; got {tools}"


def test_flankers_start_on_holdfire():
    """The 2tnk flankers must be pre-placed at stance:0 (HoldFire) so
    a stall / observe-only policy collects ZERO kills. If they ship
    on stance:2 (Defend) the engine auto-fires for free and the
    no-cheat bar collapses (the original PR #46 defect)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        flankers = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert flankers, f"{lvl}: no agent 2tnk flankers found"
        for a in flankers:
            assert a.stance == 0, (
                f"{lvl}: flanker {a.type}@{a.position} must be stance:0 "
                f"(HoldFire); got stance:{a.stance}. A stance:2 flanker "
                f"auto-fires for free and lets stall WIN — defect."
            )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
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


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_fact_survival_in_fail_clause(level):
    c = compile_level(load_pack(PACK), level)
    fc = c.fail_condition.model_dump(exclude_none=True)
    flat = str(fc)
    assert "building_count_gte" in flat or "has_building" in flat, fc


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_attrition_cap_in_win_and_fail(level):
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    fail = c.fail_condition.model_dump(exclude_none=True)
    win_cap = next(
        (clause["units_lost_lte"] for clause in win.get("all_of", [])
         if "units_lost_lte" in clause),
        None,
    )
    assert win_cap is not None, win
    has_fail_cap = any(
        (clause.get("not") or {}).get("units_lost_lte") == win_cap
        for clause in fail.get("any_of", []) or []
    )
    assert has_fail_cap, fail


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_enforces_fact_survival_in_win(level):
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "building_count_gte" in flat, win
    assert "fact" in flat.lower(), win


def test_hard_has_two_spawn_point_groups():
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    enemy_groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert enemy_groups == {0, 1}, enemy_groups
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended TRIGGER WINS every level + every hard seed ────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_ambush_trigger_wins(level):
    """The intended capability — a reactive set_stance from stance:0
    (HoldFire) to stance:3 (AttackAnything) once the band closes
    into the crossfire envelope — WINS every level + every hard
    seed. Load-bearing solvency check for the ambush-trigger
    capability."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, intended_ambush, seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended ambush-trigger must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every lazy / wrong policy LOSES every level + seed ─────


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_tier_and_seed(level):
    """STALL (observe-only) MUST LOSE every level + every hard seed.
    With flankers at stance:0 the engine never auto-engages on its
    own; a stall policy collects zero kills and the band reaches the
    base. This is the PR #46 defect inversion: a stance:2 flanker
    auto-fired for free and let stall WIN — the no-cheat bar
    collapsed. The fix (stance:0) restores the bar."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed}: stall must LOSE (real fail, not "
            f"draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_charge_loses_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, charge, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} charge: must LOSE (real fail, "
            f"not draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_pile_on_fact_loses_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, pile_on_fact, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} pile-on-fact: must LOSE "
            f"(real fail, not draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_medium():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, intended_ambush, seed=2)
    b = run_level(c, intended_ambush, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
