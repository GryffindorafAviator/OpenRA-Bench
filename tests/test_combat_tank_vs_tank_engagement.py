"""combat-tank-vs-tank-engagement — tank trade: a controlled
focus-fire `attack_unit` engagement WINS; STALL and a BRUTE
`attack_move` drive-in LOSE.

The bar: the intended FOCUS-fire engagement (close to cannon range,
hold, concentrate `attack_unit` fire on one target at a time) WINS on
every level and every hard seed (1-4); STALL (pure observe) and a
BRUTE `attack_move` drive straight INTO the enemy position LOSE on
every level and every hard seed. Non-win is a real reachable timeout
LOSS via the `after_ticks` fail clause (within_ticks 2400 +
after_ticks 2401 on easy/medium with max_turns 30; within_ticks 1500
+ after_ticks 1501 on hard with max_turns 20).

Recalibrated after the engine movement fixes (moving units take fire
en route; `attack_unit` on out-of-sight targets paths normally at
real Mobile speed; no sprint-invincibility). Finding from this
recalibration: with the post-fix combat model a SYMMETRIC 3-vs-3
tank mirror is a flat meat-grinder — whatever the target assignment
(focus one target, or each tank its own nearest), the agent loses
exactly two tanks closing the distance. The symmetric-mirror
focus-vs-spread SURVIVOR delta the pack originally relied on no
longer exists in the engine (a `spread_closest` policy ends
identically to focus). The load-bearing discrimination is therefore
CONTROLLED ENGAGEMENT vs BRUTE drive-in, and the difficulty axis is
re-tuned:
  * EASY — 3-vs-3. Focus `attack_unit` closes to cannon range and
    clears the line (≥1 survivor); a brute `attack_move` onto the
    enemy cell bunches the column in melee and force-wipes.
  * MEDIUM — 4-vs-3 (a fourth enemy tank, the agent is
    numerically out-gunned). A controlled focus engagement clears
    ≥3 of the 4 enemy tanks while keeping ≥2 of its own; a brute
    drive-in eats 4-tank crossfire and wipes before killing 3.
  * HARD — 3-vs-3 with a tight kill-speed deadline (within_ticks
    1500) and two seed-driven spawn corridors (NORTH / SOUTH).

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-tank-vs-tank-engagement.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-tank-vs-tank-engagement"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 3, (
        f"benchmark_anchor must list all 3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "lanchester", "concentration"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, kills=0, lost=0, buildings=("fact",)):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[{"type": b} for b in buildings],
        own_building_types=set(buildings),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": list(units),
            "buildings": [{"type": b} for b in buildings],
        },
    )


def _alive(n):
    return [
        {"cell_x": 30, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended focus: kills 3, 3 alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=3, lost=0))
    # Spread allowed to squeak by on easy (own_units_gte:1, 1 survivor)
    assert evaluate(c.win_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # Kill bar unmet (only 2 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=2, lost=0))
    # Force wipe → fail (own_units_gte:1 trips via fail clause)
    assert evaluate(c.fail_condition, _ctx(units=[], tick=900, kills=3, lost=3))
    # Timeout with bar unmet → fail (after_ticks 2401)
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=2402, kills=2, lost=0))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended focus: kills 3, 3 alive (cap met at ≥2) → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=3, lost=0))
    # ≥2 alive (cap met) with 3 kills → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(2), tick=900, kills=3, lost=1))
    # Spread-fire: only 1 alive < 2 → not a win (cap busted)
    assert not evaluate(c.win_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # And the fail clause trips when own_units_gte:2 fails
    assert evaluate(c.fail_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=900, kills=3, lost=3))
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=2402, kills=2, lost=0))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended focus: kills 3, 3 alive, within tight tick budget → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=3, lost=0))
    # 2 survivors (own_units_gte:2) with full kill bar → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(2), tick=900, kills=3, lost=1))
    # 1 survivor (cap busted on hard) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # And the fail clause trips when own_units_gte:2 fails (this is
    # the load-bearing fix that turns the SOUTH-spawn brute drive-in
    # from a DRAW into a real LOSS).
    assert evaluate(c.fail_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # Kill bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=2, lost=0))
    # Outside tight tick budget (kills met but slow) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=1600, kills=3, lost=0))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=900, kills=3, lost=3))
    # Timeout → fail (tight after_ticks 1501)
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=1502, kills=2, lost=0))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl, want_after in [("easy", 2401), ("medium", 2401), ("hard", 1501)]:
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert want_after <= max_tick, (
            f"{lvl}: after_ticks {want_after} > max reachable tick "
            f"{max_tick} (max_turns={c.max_turns}); deadline never bites"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging corridor (NORTH y=11..13 / SOUTH
    y=27..29). Engine-roundtrip is asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_enemy_line_is_a_spread_tank_line():
    """The enemy line MUST be a spread tank line on distinct
    latitudes (each enemy independently targetable): 3 tanks on
    easy/hard, 4 on medium (the 4-vs-3 over-match). The (50,20)
    silent-fail cell must not be used."""
    pack = load_pack(PACK_PATH)
    expected = {"easy": 3, "medium": 4, "hard": 3}
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        enemy_tanks = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "2tnk"
        ]
        assert len(enemy_tanks) == expected[lvl], (
            f"{lvl}: must have exactly {expected[lvl]} enemy tanks, "
            f"got {len(enemy_tanks)}"
        )
        ys = sorted(a.position[1] for a in enemy_tanks)
        assert len(set(ys)) == expected[lvl], (
            f"{lvl}: enemy tanks must be on {expected[lvl]} distinct "
            f"latitudes (spread line), got ys={ys}"
        )
        # Verify the (50,20) silent-fail cell is NOT used.
        positions = [tuple(a.position) for a in enemy_tanks]
        assert (50, 20) not in positions, (
            f"{lvl}: (50,20) is a CLAUDE.md-documented silent-fail "
            f"cell. Got {positions}"
        )
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "fact" in types, f"{lvl}: needs a persistent enemy fact"


def test_agent_strike_force_is_3_tanks_bunched():
    """The agent strike force MUST be 3 medium tanks bunched on
    adjacent rows at a single column (one centroid) — that bunched-
    vs-spread asymmetry is what makes focus-fire load-bearing."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        agent_tanks = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert len(agent_tanks) == 3, (
            f"{lvl}: must have exactly 3 agent tanks, got {len(agent_tanks)}"
        )
        xs = {a.position[0] for a in agent_tanks}
        assert len(xs) == 1, (
            f"{lvl}: agent tanks must be bunched on ONE column (same x), "
            f"got xs={xs}"
        )
    # Hard: per-spawn-group bunching (each spawn_point has 3 tanks
    # bunched on one column at a single latitude band).
    c = compile_level(pack, "hard")
    for sp in (0, 1):
        agent_tanks = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk" and a.spawn_point == sp
        ]
        assert len(agent_tanks) == 3, (
            f"hard sp={sp}: must have 3 agent tanks, got {len(agent_tanks)}"
        )
        xs = {a.position[0] for a in agent_tanks}
        assert len(xs) == 1, (
            f"hard sp={sp}: agent tanks bunched on ONE column, got xs={xs}"
        )


# ── engine-driven scripted policies ─────────────────────────────────


def _own_ids(rs):
    return [str(u["id"]) for u in (rs.get("units_summary", []) or [])]


def _enemy_tanks(rs):
    out = []
    for e in (rs.get("enemy_summary") or []):
        t = (e.get("type") or e.get("actor_type") or "").lower()
        if t == "2tnk":
            out.append(e)
    return out


def _stall(rs, Command):
    """Pure observe — tanks never close to firing range; nothing dies
    either way ⇒ kill bar unmet ⇒ after_ticks LOSS."""
    return [Command.observe()]


def _brute_attack_move(rs, Command):
    """Brute: every tank attack_moves straight onto the enemy line.
    The `attack_move` drives the bunched column INTO the enemy
    position (rather than holding at cannon range) — the stack is
    enveloped in the enemy crossfire and force-wipes before clearing
    the line ⇒ LOSS (force-wipe / kill-bar unmet)."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    return [Command.attack_move(own, 51, 20)]


def _focus_fire(rs, Command):
    """Focus-fire: ALL agent tanks attack_unit the SAME target each
    turn — the closest enemy to the agent centroid. Once that enemy
    falls, the policy automatically re-targets the next-closest. This
    is the Lanchester-square-law optimal: concentrate N output on 1
    target, remove that target's DPS, then repeat — combat-power
    surplus grows quadratically with each kill."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    es = _enemy_tanks(rs)
    if not es:
        return [Command.attack_move(own, 51, 20)]
    us = rs.get("units_summary") or []
    cx = sum(u["cell_x"] for u in us) / len(us)
    cy = sum(u["cell_y"] for u in us) / len(us)
    es.sort(key=lambda e: (e["cell_x"] - cx) ** 2 + (e["cell_y"] - cy) ** 2)
    tid = es[0].get("id")
    if tid is not None:
        return [Command.attack_unit(own, str(tid))]
    return [Command.attack_move(own, 51, 20)]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_focus_fire_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _focus_fire, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended focus-fire should WIN, got "
        f"{r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no engagement → kill bar unmet → after_ticks fires), got "
        f"{r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_attack_move_loses(level, seed):
    """Brute attack_move into the centre crosses through the 3-tank
    crossfire and force-wipes the agent strike force on EASY (3 of 3
    lost) ⇒ LOSS via not own_units_gte:1; on MEDIUM (2 lost) ⇒ LOSS
    via not own_units_gte:2; on HARD (2-3 lost) ⇒ LOSS via not
    own_units_gte:2 (the SOUTH-spawn variant leaves 1 survivor with
    only 1 kill — kill bar unmet AND survival cap busted)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_attack_move, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute attack_move must LOSE (drive-into-"
        f"crossfire force-wipe / tick budget bust), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_outnumbered_needs_controlled_engagement(level, seed):
    """The medium-tier 4-vs-3 over-match is the load-bearing
    discrimination: the intended controlled focus-fire engagement
    clears ≥3 of the 4 enemy tanks while keeping ≥2 of its own (WIN),
    whereas the brute `attack_move` drive-in is enveloped in the
    4-tank crossfire and force-wipes before killing 3 (LOSS). This
    re-asserts the focus-WIN / brute-LOSS bar across every level —
    the per-policy tests above already cover it, this is the
    aggregate invariant pinned by the recalibration."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    win = run_level(c, _focus_fire, seed=seed)
    lose = run_level(c, _brute_attack_move, seed=seed)
    assert win.outcome == "win", (
        f"{level} seed={seed}: controlled focus engagement must WIN, "
        f"got {win.outcome} (kills={win.signals.units_killed}, "
        f"losses={win.signals.units_lost})"
    )
    assert lose.outcome == "loss", (
        f"{level} seed={seed}: brute drive-in must LOSE, got "
        f"{lose.outcome} (kills={lose.signals.units_killed}, "
        f"losses={lose.signals.units_lost})"
    )
