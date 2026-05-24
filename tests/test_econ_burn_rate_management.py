"""econ-burn-rate-management — REASONING capability validation.

Real-world anchor: startup financial runway / corporate burn-rate
discipline / SC2 mineral-bank overflow. The agent runs a productive
base (fact + proc + powr + tent + weap + fix; 2 harvs on 2 near
mines; 3 pre-placed medium tanks) with starting cash $1500, and must
defeat a central enemy garrison WHILE landing cash inside a target
BAND [MIN, MAX] at the latch tick. The two ends of the band give the
test concrete teeth:

  * upper bound (MAX) — encoded `not: {cash_gte: MAX+1}` because no
    `cash_lte` predicate exists. A stalling / pure-hoarding / tank-only
    play accumulates cash above MAX → LOSS.
  * lower bound (MIN, `cash_gte: MIN`) — a burn-everything play drops
    cash below MIN → LOSS.
  * kill bar (`units_killed_gte: K`) ensures the cash spend translates
    to MILITARY capability, not arbitrary buildings.

Bar (CLAUDE.md "no defect, no cheat, no draw"):
   * stall / save-only / burn-all / tank-only LOSE every tier and
     every hard seed (each fails a distinct bar — kill, upper-cash,
     lower-cash, upper-cash respectively).
   * intended LEAN (attack-move pre-placed tanks east + queue 2× 2tnk
     on turn 1, harvest both harvs) WINS every tier and every hard
     seed (cash settles in the band as the kill bar fires).
   * hard tier defines ≥2 agent spawn_point groups (NORTH / SOUTH
     base) round-robined by seed; the burn-rate decision generalises
     across spawns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "econ-burn-rate-management.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, C):
    """No-op every turn. Pre-placed tanks sit idle → kills=0 → LOSS
    every tier on the kill bar (regardless of cash band)."""
    return [C.observe()]


def _save_only(rs, C):
    """Harvest only, never engage, never build. Cash accumulates
    above MAX (~3500 by turn 12) AND kills=0 → LOSS on both the
    upper-cash bound and the kill bar."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    if not harvs:
        return [C.observe()]
    return [C.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs]


def _engage(rs, C, cmds):
    """Append an explicit attack order driving every pre-placed 2tnk
    onto the nearest garrison e1. The pre-placed tanks are stance:0
    (HoldFire) — they never auto-engage, so an explicit attack_unit is
    REQUIRED to score kills (the kill bar stays load-bearing after the
    engine stance fix). Falls back to attack_move toward the patch
    column while no garrison unit is yet visible."""
    units = rs.get("units_summary", []) or []
    tanks = [u for u in units if u.get("type") == "2tnk"]
    if not tanks:
        return cmds
    tank_ids = [str(u["id"]) for u in tanks]
    targets = [e for e in (rs.get("enemy_summary") or [])
               if e.get("type") == "e1"]
    if targets:
        cmds.append(C.attack_unit(tank_ids, str(targets[0]["id"])))
    else:
        cmds.append(C.attack_move(tank_ids, 40, int(tanks[0]["cell_y"])))
    return cmds


def _burn_all(rs, C):
    """Chain 2tnk + e1 + place pbox + place powr until cash → 0 while
    also engaging the garrison. Cash drops below MIN (~0-44) at the
    latch tick → LOSS on the lower-cash bound. On easy / medium / hard
    the burn-rate is sufficient to blow the floor before income can
    refill above MIN."""
    units = rs.get("units_summary", []) or []
    own_b = rs.get("own_buildings") or []
    fy = 22
    for b in own_b:
        if b.get("type") == "fact":
            fy = int(b["cell_y"])
            break
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = [C.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs]
    cmds.append(C.build("2tnk"))
    cmds.append(C.build("e1"))
    cmds.append(C.build("e1"))
    cmds.append(C.build("pbox"))
    cmds.append(C.place_building("pbox", 24, fy))
    cmds.append(C.build("powr"))
    cmds.append(C.place_building("powr", 24, fy + 2))
    return _engage(rs, C, cmds)


def _tank_only(rs, C):
    """Use the pre-placed tanks WITHOUT building anything more.
    Harvest income lifts cash above MAX while kills clear the bar →
    LOSS on the upper-cash bound on EVERY tier (easy ≤ 1800; medium
    ≤ 1500; hard ≤ 1499). This is the "save while attacking" play —
    the BURN-RATE teeth catch it."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = [C.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs]
    return _engage(rs, C, cmds)


def _intended_lean(rs, C):
    """The intended burn-rate capability: drive the pre-placed tanks
    onto the garrison (attack_unit — the stance:0 tanks need an
    explicit order) AND queue 2× 2tnk from the war factory on turn 1
    to burn down cash at the operating rate, AND harvest both harvs.
    Cash settles inside the band as the kill bar fires — wins every
    tier and every hard seed (verified scripted run_level)."""
    # Per-call turn counter via attribute on the function itself
    # (avoids module-level state leaking across pytest runs).
    n = getattr(_intended_lean, "_n", 0) + 1
    _intended_lean._n = n

    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = [C.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs]
    if n == 1:
        cmds.append(C.build("2tnk"))
        cmds.append(C.build("2tnk"))
    return _engage(rs, C, cmds)


def _reset_lean():
    """Reset the per-episode counter on _intended_lean so each test
    starts on turn 1. Tests must call this in their setup."""
    _intended_lean._n = 0


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "bespoke 56x40 arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-burn-rate-management"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "startup" in anchors and "runway" in anchors, anchors
    assert "burn rate" in anchors or "burn-rate" in anchors, anchors
    assert "financial" in anchors or "sc2" in anchors, anchors


def test_tools_include_required_set():
    """Pack must expose [observe, build, place_building, harvest,
    move_units, attack_unit, attack_move, stop] — the burn-rate
    decision needs BOTH production levers (build/place_building) and
    combat levers (attack_move) AND the harvest tool to keep income
    flowing."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in (
        "observe", "build", "place_building", "harvest",
        "move_units", "attack_unit", "attack_move", "stop",
    ):
        assert required in tools, f"missing tool: {required!r}"


def test_starting_cash_is_1500():
    """Starting cash is the load-bearing knob — verify it stays at
    the calibrated 1500 (the band thresholds are derived against it)."""
    pack = load_pack(PACK)
    assert pack.starting_cash == 1500, (
        f"starting_cash must be 1500 (the band calibration); got "
        f"{pack.starting_cash}"
    )


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks + 1 == after_ticks (so a
    non-finisher LOSES, not draws)."""
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
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            "(non-finisher must LOSE, not draw — fail clause one tick"
            " past win clause)"
        )


def test_every_level_has_a_cash_band():
    """Every tier's win predicate must encode BOTH a cash floor
    (`cash_gte`) AND a cash ceiling (`not: {cash_gte: MAX+1}`). The
    BAND is the burn-rate teeth — without both ends the capability
    is not enforced."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        win = pack.levels[lvl].win_condition.model_dump()
        clauses = win.get("all_of") or []
        has_floor = any("cash_gte" in c for c in clauses)
        has_ceiling = any(
            "not" in c
            and isinstance(c["not"], dict)
            and "cash_gte" in c["not"]
            for c in clauses
        )
        assert has_floor, f"{lvl}: missing cash_gte floor"
        assert has_ceiling, f"{lvl}: missing not:cash_gte ceiling"


def test_every_level_has_a_kill_bar():
    """The kill bar (`units_killed_gte`) ensures cash spend converts
    to MILITARY capability, not just any build action."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        win = pack.levels[lvl].win_condition.model_dump()
        clauses = win.get("all_of") or []
        assert any("units_killed_gte" in c for c in clauses), (
            f"{lvl}: missing units_killed_gte kill bar"
        )


def test_band_tightens_from_easy_to_medium():
    """Easy is the loose tier (wide band); medium tightens it (the
    canonical "+1 controlled variable per tier" idiom). Both ends
    move INWARD on medium vs easy (or one end at minimum)."""
    pack = load_pack(PACK)
    def _band(lvl):
        clauses = pack.levels[lvl].win_condition.model_dump()["all_of"]
        floor = next(int(c["cash_gte"]) for c in clauses if "cash_gte" in c)
        ceil = next(
            int(c["not"]["cash_gte"]) - 1
            for c in clauses
            if "not" in c
            and isinstance(c["not"], dict)
            and "cash_gte" in c["not"]
        )
        return floor, ceil
    e_lo, e_hi = _band("easy")
    m_lo, m_hi = _band("medium")
    assert (m_hi - m_lo) < (e_hi - e_lo), (
        f"medium band ({m_lo},{m_hi}) must be tighter than easy "
        f"({e_lo},{e_hi})"
    )


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 distinct agent spawn_point groups so engine
    round-robins start by seed."""
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
    """Bespoke 56x40 arena (cordon 2) playable bounds: x:2..52, y:2..36."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 52 and 2 <= y <= 36, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


def test_preplaced_base_has_full_production_stack():
    """Every tier must pre-place fact + proc + powr + tent + weap +
    fix on the agent side (the burn-rate decision needs ALL the
    spend levers actionable on turn 1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {a.type for a in c.scenario.actors if a.owner == "agent"}
        for needed in ("fact", "proc", "powr", "tent", "weap", "fix"):
            assert needed in agent_types, (
                f"{lvl}: pre-placed base missing {needed!r}; got "
                f"{sorted(agent_types)}"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, cash=0, kills=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=0,
        cash=cash,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (kill bar AND cash floor AND cash ceiling AND
    fact alive AND in time); fail fires on timeout OR fact destroyed."""
    c = compile_level(load_pack(PACK), "medium")
    base_b = [("fact", 10, 22), ("proc", 12, 18)]

    # Intended: cash=1000 (in [400, 1500]), kills=3, fact alive,
    # in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(tick=600, cash=1000, kills=3, own_buildings=base_b),
    )
    # Cash one above ceiling (1500 max — `not cash_gte 1501`) → not win
    assert not evaluate(
        c.win_condition,
        _ctx(tick=600, cash=1501, kills=3, own_buildings=base_b),
    )
    # Cash one below floor → not win
    assert not evaluate(
        c.win_condition,
        _ctx(tick=600, cash=399, kills=3, own_buildings=base_b),
    )
    # Kills under bar → not win
    assert not evaluate(
        c.win_condition,
        _ctx(tick=600, cash=1000, kills=2, own_buildings=base_b),
    )
    # Past within_ticks → not win
    assert not evaluate(
        c.win_condition,
        _ctx(tick=1081, cash=1000, kills=3, own_buildings=base_b),
    )
    # Fact destroyed → not win, AND fail
    assert not evaluate(
        c.win_condition,
        _ctx(tick=600, cash=1000, kills=3, own_buildings=base_b[1:]),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(tick=600, cash=1000, kills=3, own_buildings=base_b[1:]),
    )
    # Past after_ticks → fail
    assert evaluate(
        c.fail_condition,
        _ctx(tick=1081, cash=1000, kills=0, own_buildings=base_b),
    )
    # Within deadline, fact alive → not fail
    assert not evaluate(
        c.fail_condition,
        _ctx(tick=600, cash=1000, kills=0, own_buildings=base_b),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """No-op → tanks idle → kills=0 → kill bar unmet → LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (no kills); "
        f"got {r.outcome} cash={r.signals.cash} kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_save_only_loses_every_tier_and_seed(level, seed):
    """Pure harvest, never engage → cash overflows MAX AND kills=0
    → LOSS on both the upper-cash bound and the kill bar."""
    _, r = _run(level, _save_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: save-only must LOSE (cash overflow + 0 "
        f"kills); got {r.outcome} cash={r.signals.cash} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_burn_all_loses_every_tier_and_seed(level, seed):
    """Chain build/place until cash → 0 → cash falls below MIN at
    the latch tick → LOSS on the lower-cash bound."""
    _, r = _run(level, _burn_all, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: burn-all must LOSE (cash under floor); "
        f"got {r.outcome} cash={r.signals.cash} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_tank_only_loses_every_tier_and_seed(level, seed):
    """Use pre-placed tanks WITHOUT building anything → kills clear
    the bar but cash overflows MAX → LOSS on the upper-cash bound.
    The BURN-RATE teeth in action — saving while attacking still
    LOSES."""
    _, r = _run(level, _tank_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: tank-only must LOSE (cash overflow despite "
        f"kills); got {r.outcome} cash={r.signals.cash} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_lean_wins_every_tier_and_seed(level, seed):
    """The intended capability — engage pre-placed tanks east AND
    queue a modest production burn (2× 2tnk) to keep cash inside
    the band — WINS every tier and every hard seed. Cash settles
    inside the band at the kill-bar latch tick."""
    _reset_lean()
    _, r = _run(level, _intended_lean, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended burn-rate policy must WIN; "
        f"got {r.outcome} cash={r.signals.cash} "
        f"kills={r.signals.units_killed}"
    )


# ── determinism ─────────────────────────────────────────────────────


def test_outcome_is_deterministic_per_seed():
    """Same seed, same policy → identical outcome / cash / turns /
    kills (no engine non-determinism leaks into the burn-rate test)."""
    c = compile_level(load_pack(PACK), "medium")
    _reset_lean()
    a = run_level(c, _intended_lean, seed=2)
    _reset_lean()
    b = run_level(c, _intended_lean, seed=2)
    assert (a.outcome, a.turns, a.signals.cash, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.cash, b.signals.units_killed
    )
