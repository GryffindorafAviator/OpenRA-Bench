"""combat-kite-and-pull — ACTION capability validation.

Kiting micro: a fast light strike force must hit-and-PULL a slow
heavy enemy — strike at weapon range, retreat out of the heavy's
lethal close-range window before it can fire back, repeat. Standing
and fighting LOSES (the heavy cannon out-trades the kiter stack
head-on); only the move-away + attack_unit kite cycle WINS.

Per-tier escalation in kiter count + chaser toughness:

  easy   — 1 kiter vs 1 chaser at 35% HP   (bare kite skill;
           survival bar ≥1)
  medium — 2 kiters vs 1 chaser at 40% HP  (paired kite +
           both-survive bar)
  hard   — 3 kiters vs 1 chaser at 70% HP + seed-spawn flip
           (full-formation kite + all-three-survive bar +
           tighter clock + spawn variation)

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed — a
    passive ReturnFire stack that never kites is overrun by the
    hunting heavy → the survival bar fails / the deadline bites.
  * stand-and-fight (attack_move onto the heavy, never retreat)
    LOSES every tier / seed — the heavy cannon collapses the stack
    head-on.
  * brute / wrong-path (one attack_move far east, no disengage)
    LOSES every tier / seed — same close-range trade.
  * intended kite-and-pull (retreat when the heavy closes within
    ~6 cells, else attack_unit) WINS every tier / every hard seed,
    preserving the survival bar (≥1 easy, ≥2 medium, ≥3 hard).
  * hard tier defines ≥2 agent spawn_point groups (NORTH y=10 /
    SOUTH y=30 corridor) round-robined by seed so a memorised
    opening cannot generalise.
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

PACK = PACKS_DIR / "combat-kite-and-pull.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-tier expected survival bar (own_units_gte). Mirrors the YAML.
SURVIVAL_BAR = {"easy": 1, "medium": 2, "hard": 3}


# ── scripted policies ───────────────────────────────────────────────


def _kiters(rs):
    return [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]


def _stall(rs, C):
    """Observe-only. A passive ReturnFire kiter that never kites is
    overrun by the hunting heavy → LOSS."""
    return [C.observe()]


def _stand(rs, C):
    """Stand-and-fight: attack_move straight onto the heavy and never
    retreat. The heavy cannon out-trades the stack head-on → LOSS."""
    own = _kiters(rs)
    if not own:
        return [C.observe()]
    return [C.attack_move([str(u["id"]) for u in own], target_x=70, target_y=20)]


def _brute(rs, C):
    """Brute / wrong-path: one attack_move far east, no disengage.
    Same close-range trade as stand-and-fight → LOSS."""
    own = _kiters(rs)
    if not own:
        return [C.observe()]
    return [
        C.attack_move(
            [str(u["id"]) for u in own], target_x=92, target_y=own[0]["cell_y"]
        )
    ]


def _kite(rs, C):
    """Intended kite-and-pull: each turn, if the heavy has closed
    within ~7 cells of a kiter (Manhattan), MOVE that kiter ~8 cells
    AWAY along its lane (the PULL); otherwise attack_unit the heavy
    from range (the STRIKE). When no heavy is yet visible, advance
    east to draw the hunting chaser into vision (capped at x=50 —
    far enough to contact, not so far as to march into the heavy's
    lethal close range without warning). The cycle is purely
    reactive — derived each turn from geometry, no memory.

    The retreat-threshold ≤7 and retreat-distance 8 are tuned for
    the 96x40 arena (kiters stage at x=20, heavy at x=80): on the
    diagonal-lag geometry (kiter y=9/11, heavy y=20), the chaser
    enters vision at distance ~7 (Manhattan), so a ≤6 trigger
    leaves the kiter inside cannon range one extra turn — costing
    a kiter on medium's 2-of-2-must-survive bar."""
    own = _kiters(rs)
    if not own:
        return [C.observe()]
    enemies = rs.get("enemy_summary") or []
    heavies = [e for e in enemies if (e.get("type") or "").lower() == "3tnk"]
    cmds = []
    if heavies:
        for u in own:
            t = min(
                heavies,
                key=lambda e: abs(e["cell_x"] - u["cell_x"])
                + abs(e["cell_y"] - u["cell_y"]),
            )
            d = abs(u["cell_x"] - t["cell_x"]) + abs(u["cell_y"] - t["cell_y"])
            if d <= 7:
                cmds.append(
                    C.move_units(
                        [str(u["id"])],
                        target_x=max(4, u["cell_x"] - 8),
                        target_y=u["cell_y"],
                    )
                )
            else:
                cmds.append(C.attack_unit([str(u["id"])], str(t["id"])))
    else:
        # No vision yet — march east on the staging lane until the
        # hunting heavy comes into sight (cap at x=38 so the kiter
        # does not blind-march into the heavy's lethal close range
        # on the 96x40 arena; gives the kiter ~36 cells of retreat
        # space before the cordon).
        cmds.append(
            C.move_units(
                [str(u["id"]) for u in own],
                target_x=min(50, own[0]["cell_x"] + 5),
                target_y=own[0]["cell_y"],
            )
        )
    return cmds


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_action():
    pack = load_pack(PACK)
    assert pack.meta.id == "combat-kite-and-pull"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2 kiting micro" in anchors, anchors
    assert "cavalry skirmish doctrine" in anchors, anchors


def test_enemy_uses_hunt_bot_on_every_level():
    """The heavy must HUNT — a stance:2 heavy idle in fog would never
    be discoverable; the hunt advance brings it into vision."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported, f"{lvl}: tailored arena terrain required"
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", f"{lvl}: enemy bot must be 'hunt'; got {bot}"


def test_tools_are_combat_only():
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("move_units", "attack_unit", "attack_move", "stop"):
        assert required in tools, f"missing tool: {required!r}"
    assert "build" not in tools, "this is a combat-micro pack — no build tool"


def test_every_level_has_reachable_timeout_fail():
    """`after_ticks` fail must bite within max_turns; within_ticks+1
    == after_ticks so a boundary non-finisher LOSES, not draws."""
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
        assert wt + 1 == ft, f"{lvl}: within/after mismatch {wt}/{ft}"


def test_every_level_has_a_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_survival_bar_scales_with_tier():
    """Per-tier escalation: own_units_gte rises from 1 (easy 1v1) to
    2 (medium 2v1) to 3 (hard 3v1). A kite that loses a kiter on
    easy passes easy but fails medium/hard — the bar tightens as the
    formation grows."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        bar = next(
            int(c["own_units_gte"])
            for c in L.win_condition.model_dump()["all_of"]
            if "own_units_gte" in c
        )
        assert bar == SURVIVAL_BAR[lvl], (
            f"{lvl}: survival bar must be {SURVIVAL_BAR[lvl]}; got {bar}"
        )


def test_kiter_count_matches_survival_bar():
    """The number of pre-placed 2tnk kiters per tier equals the
    survival bar — easy has 1, medium has 2, hard has 3 per
    spawn_point group. This is the load-bearing "preserve every
    kiter" check the bar enforces."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_2tnks = [a for a in c.scenario.actors if a.owner == "agent" and a.type == "2tnk"]
        if lvl == "hard":
            # Hard has 2 spawn_point groups; each group has SURVIVAL_BAR["hard"] kiters
            by_sp: dict[int, int] = {}
            for a in agent_2tnks:
                sp = a.spawn_point if a.spawn_point is not None else 0
                by_sp[sp] = by_sp.get(sp, 0) + 1
            assert set(by_sp) == {0, 1}, f"hard must have spawn groups 0+1; got {sorted(by_sp)}"
            for sp, n in by_sp.items():
                assert n == SURVIVAL_BAR["hard"], (
                    f"hard spawn {sp}: expected {SURVIVAL_BAR['hard']} kiters; got {n}"
                )
        else:
            assert len(agent_2tnks) == SURVIVAL_BAR[lvl], (
                f"{lvl}: expected {SURVIVAL_BAR[lvl]} kiters; got {len(agent_2tnks)}"
            )


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert sp == {0, 1}, f"hard must define spawn_point groups {{0,1}}; got {sorted(sp)}"


def test_in_bounds_actors_on_every_level():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 93 and 2 <= y <= 37, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, killed=0, n_units=2, start=2):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=start - n_units,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": 28, "cell_y": 10} for _ in range(n_units)
            ]
        },
    )


def test_predicates_enforce_kill_and_survival():
    pe = compile_level(load_pack(PACK), "easy")
    # easy: kill 1, ≥1 alive, in time → WIN
    assert evaluate(pe.win_condition, _ctx(tick=1000, killed=1, n_units=1, start=1))
    # easy: kill 0 → not win
    assert not evaluate(pe.win_condition, _ctx(tick=1000, killed=0, n_units=1, start=1))
    # easy: 0 kiters left → fail (need ≥1)
    assert evaluate(pe.fail_condition, _ctx(tick=1000, killed=1, n_units=0, start=1))

    pm = compile_level(load_pack(PACK), "medium")
    # medium: both kiters alive + kill → WIN
    assert evaluate(pm.win_condition, _ctx(tick=1000, killed=1, n_units=2, start=2))
    # medium: only 1 kiter alive → not win, and fail fires
    assert not evaluate(pm.win_condition, _ctx(tick=1000, killed=1, n_units=1, start=2))
    assert evaluate(pm.fail_condition, _ctx(tick=1000, killed=1, n_units=1, start=2))
    # medium: past deadline → fail
    assert evaluate(pm.fail_condition, _ctx(tick=4502, killed=0, n_units=2, start=2))

    ph = compile_level(load_pack(PACK), "hard")
    # hard: all 3 kiters alive + kill → WIN
    assert evaluate(ph.win_condition, _ctx(tick=1000, killed=1, n_units=3, start=3))
    # hard: only 2 kiters alive → fail (need ≥3)
    assert evaluate(ph.fail_condition, _ctx(tick=1000, killed=1, n_units=2, start=3))


# ── engine-driven: every lazy/wrong policy LOSES, intended WINS ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stand_and_fight_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stand, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stand-and-fight must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_brute_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _brute, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: brute attack_move must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_kite_wins_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _kite, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: kite-and-pull must WIN; got {r.outcome} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost}"
    )


def test_kite_run_is_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _kite, seed=2)
    b = run_level(c, _kite, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
