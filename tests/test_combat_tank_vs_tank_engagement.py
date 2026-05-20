"""combat-tank-vs-tank-engagement — Mirror tank trade: focus-fire WINS,
spread-fire (and brute attack_move, and stall) LOSE.

The bar: intended FOCUS-fire WINS on every level and every hard seed
(1-4); STALL and BRUTE attack_move LOSE on every level and every hard
seed. SPREAD-fire (each tank picks its own closest enemy) LOSES on
MEDIUM (the load-bearing discrimination: survival cap own_units_gte:2
trips because spread bleeds 2 tanks in the asymmetric flank chase) —
SPREAD is permitted to squeak by on EASY (own_units_gte:1, forgiving
bare-skill tier per the SCENARIO_REVIEW_CHECKLIST inert-easy-teeth
convention) and on HARD (the asymmetric geometry collapses spread to
focus when the agent stack starts on a flank latitude — spread ≡
focus when there's a unique closest enemy from a flank perspective;
the hard discrimination is kill-speed + spawn-variation, not
spread-vs-focus survivor count).

Non-win is a real reachable timeout LOSS via the `after_ticks` fail
clause (within_ticks 2400 + after_ticks 2401 on easy/medium with
max_turns 30; within_ticks 1200 + after_ticks 1201 on hard with
max_turns 15).

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
    # 1 survivor (own_units_gte:1) with full kill bar → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(1), tick=900, kills=3, lost=2))
    # Kill bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=900, kills=2, lost=0))
    # Outside tight tick budget (kills met but slow) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=1300, kills=3, lost=0))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=900, kills=3, lost=3))
    # Timeout → fail (tight after_ticks 1201)
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=1202, kills=2, lost=0))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl, want_after in [("easy", 2401), ("medium", 2401), ("hard", 1201)]:
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


def test_enemy_line_is_3_tanks_asymmetric_spread():
    """The asymmetric geometry is the load-bearing physics — the
    enemy line MUST be 3 tanks spread across three distinct
    latitudes (the spread vs focus discrimination depends on each
    enemy being independently targetable). Centre enemy at x=51 (not
    x=50) per the CLAUDE.md silent-fail-cell note for (50,20)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        enemy_tanks = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "2tnk"
        ]
        assert len(enemy_tanks) == 3, (
            f"{lvl}: must have exactly 3 enemy tanks, got {len(enemy_tanks)}"
        )
        ys = sorted(a.position[1] for a in enemy_tanks)
        assert len(set(ys)) == 3, (
            f"{lvl}: enemy tanks must be on 3 distinct latitudes "
            f"(asymmetric spread), got ys={ys}"
        )
        # Verify the (50,20) silent-fail cell is NOT used.
        positions = [tuple(a.position) for a in enemy_tanks]
        assert (50, 20) not in positions, (
            f"{lvl}: (50,20) is a CLAUDE.md-documented silent-fail "
            f"cell — centre enemy must be at (51,20). Got {positions}"
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
    """Brute: every tank attack_moves toward the centre enemy. The
    bunched stack drives into the 3-tank crossfire at the engagement
    line; concentrated incoming fire kills ≥2 agent tanks ⇒ LOSS."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    return [Command.attack_move(own, 51, 20)]


def _spread_attack_closest(rs, Command):
    """Spread: each agent tank attack_units ITS OWN nearest visible
    enemy tank. With the asymmetric spread (3 enemies on three rows),
    once the centre dies the surviving agent tanks chase different
    flank enemies in 1-vs-1 duels — Lanchester linear law collapses
    the trade to mutual annihilation, ending with 1-of-3 alive. On
    MEDIUM (own_units_gte:2) this busts the survival cap ⇒ LOSS."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    es = _enemy_tanks(rs)
    if not es:
        # No targets in sight — advance to contact.
        return [Command.attack_move(own, 51, 20)]
    cmds = []
    for u in (rs.get("units_summary") or []):
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        es_sorted = sorted(
            es, key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2
        )
        tid = es_sorted[0].get("id")
        if tid is not None:
            cmds.append(Command.attack_unit([uid], str(tid)))
    return cmds or [Command.observe()]


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
    via not own_units_gte:2; on HARD (1-3 lost, but timing slow) ⇒
    LOSS via the tight within_ticks 1200 + force-wipe."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_attack_move, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute attack_move must LOSE (drive-into-"
        f"crossfire force-wipe / tick budget bust), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["medium"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_spread_attack_closest_loses_on_medium(level, seed):
    """Spread-attack-closest must LOSE on MEDIUM — the asymmetric
    flank chase ends with 1-of-3 agent tanks alive (2 lost), busting
    the survival cap own_units_gte:2. EASY is excluded as the bare-
    skill tier (own_units_gte:1 lets the 1 survivor squeak by — the
    documented SCENARIO_REVIEW_CHECKLIST inert-easy-teeth pattern).
    HARD is excluded because the asymmetric geometry collapses
    spread to focus when the agent stack starts on a flank latitude
    (NORTH or SOUTH) — from a flank there is a unique closest enemy
    that all 3 agent tanks naturally target (spread ≡ focus); the
    hard discrimination is kill-speed + spawn-variation, not the
    survivor-count delta."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _spread_attack_closest, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: spread-attack-closest must LOSE on "
        f"medium (flank chase bleeds 2 tanks, own_units_gte:2 fails), "
        f"got {r.outcome} (kills={r.signals.units_killed}, "
        f"losses={r.signals.units_lost})"
    )
