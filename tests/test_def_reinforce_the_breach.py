"""def-reinforce-the-breach — ACTION capability validation.

Reactive reserve commitment: a two-lane defence with a mobile
reserve held at the centre. Mid-episode a HEAVY breach wave is
injected on the SOUTH lane via `scheduled_events: spawn_actors`.
The agent must read which lane was breached and shift the reserve
FORWARD into that lane's intercept zone.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed — the
    reserve sits at the centre; the forward-zone clause is never
    satisfied and the un-reinforced south garrison is overrun.
  * brute (one attack_move straight east) LOSES every tier / seed
    — the reserve drives off the breached lane's intercept zone.
  * reinforce-the-wrong-lane (commit the reserve to the NORTH
    lane) LOSES on medium/hard — the south forward zone is left
    two tanks short of the n=4 bar.
  * intended reinforce-the-breach (shift the reserve into the
    SOUTH lane's intercept zone) WINS every tier / every hard
    seed.
  * the win carries an `after_ticks: 720` gate so it cannot latch
    before the breach (tick 450) has developed — no premature win.
  * hard tier defines ≥2 agent spawn_point groups (reserve start
    column x=40 / x=50) round-robined by seed.
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

PACK = PACKS_DIR / "def-reinforce-the-breach.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _tanks(rs):
    return [u for u in (rs.get("units_summary") or []) if u.get("type") == "2tnk"]


def _reserve(own):
    """The centre reserve: tanks on the mid latitude, x ≥ 37."""
    return [u for u in own if 17 <= u["cell_y"] <= 23 and u["cell_x"] >= 37]


def _stall(rs, C):
    """Observe-only — the reserve never moves; the forward-zone clause
    is never satisfied and the south garrison is overrun → LOSS."""
    return [C.observe()]


def _brute(rs, C):
    """One attack_move straight east — the reserve drives off the
    breached lane's intercept zone → the forward-zone clause fails →
    LOSS."""
    own = _tanks(rs)
    if not own:
        return [C.observe()]
    return [
        C.attack_move([str(u["id"]) for u in own], target_x=110, target_y=own[0]["cell_y"])
    ]


def _make_reinforce(lane_y):
    """Shift the reserve forward into the lane at `lane_y` (28 = the
    breached SOUTH lane; 12 = the un-breached NORTH lane). The
    garrisons attack the nearest visible enemy."""

    def p(rs, C):
        own = _tanks(rs)
        if not own:
            return [C.observe()]
        enemies = rs.get("enemy_summary") or []
        reserve = _reserve(own)
        cmds = []
        for u in reserve:
            if enemies and lane_y == 28:
                t = min(
                    enemies,
                    key=lambda e: abs(e["cell_x"] - u["cell_x"])
                    + abs(e["cell_y"] - u["cell_y"]),
                )
                cmds.append(C.attack_unit([str(u["id"])], str(t["id"])))
            else:
                cmds.append(
                    C.attack_move([str(u["id"])], target_x=58, target_y=lane_y)
                )
        for u in own:
            if u not in reserve and enemies:
                t = min(
                    enemies,
                    key=lambda e: abs(e["cell_x"] - u["cell_x"])
                    + abs(e["cell_y"] - u["cell_y"]),
                )
                cmds.append(C.attack_unit([str(u["id"])], str(t["id"])))
        return cmds or [C.observe()]

    return p


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_action():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-reinforce-the-breach"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "military reserve commitment" in anchors, anchors
    assert "reinforce the breach doctrine" in anchors, anchors
    assert "incident surge response" in anchors, anchors


def test_enemy_uses_hunt_bot():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", f"{lvl}: enemy bot must be 'hunt'; got {bot}"


def test_every_level_has_a_scheduled_breach_wave():
    """Every level injects the breach via scheduled_events:
    spawn_actors — the breach must develop MID-episode."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        ev = getattr(c, "scheduled_events", None)
        assert ev, f"{lvl}: must declare scheduled_events"
        spawns = [e for e in ev if e.get("type") == "spawn_actors"]
        assert spawns, f"{lvl}: must have a spawn_actors breach wave"
        # The breach wave fires after t=0 (mid-episode).
        for e in spawns:
            assert int(e["tick"]) > 0, f"{lvl}: breach tick must be > 0"


def test_win_has_after_ticks_gate_before_the_breach_resolves():
    """The win carries an `after_ticks` gate so it cannot latch
    before the breach wave (scheduled) has developed."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump()["all_of"]
        gate = next((int(x["after_ticks"]) for x in win if "after_ticks" in x), None)
        assert gate is not None, f"{lvl}: win must carry an after_ticks gate"
        breach_tick = min(
            int(e["tick"])
            for e in c.scheduled_events
            if e.get("type") == "spawn_actors"
        )
        assert gate > breach_tick, (
            f"{lvl}: win gate {gate} must be after the breach tick {breach_tick}"
        )


def test_forward_zone_clause_is_load_bearing():
    """Every level's win requires a units_in_region_gte forward-zone
    clause at the south lane intercept point."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump()["all_of"]
        region = next(
            (x["units_in_region_gte"] for x in win if "units_in_region_gte" in x),
            None,
        )
        assert region is not None, f"{lvl}: win needs a units_in_region_gte clause"
        assert region["y"] == 28, f"{lvl}: forward zone must be the SOUTH lane"


def test_every_level_has_reachable_timeout_fail():
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
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=1000, units_xy=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [{"cell_x": x, "cell_y": y} for x, y in units_xy]
        },
    )


def test_predicates_enforce_forward_zone_and_gate():
    pm = compile_level(load_pack(PACK), "medium")
    # 4 tanks in the south zone, after the gate, in time → WIN
    south4 = [(62, 28), (61, 27), (63, 29), (60, 28)]
    assert evaluate(pm.win_condition, _ctx(tick=1000, units_xy=south4))
    # Same 4 tanks but BEFORE the after_ticks gate → not win
    assert not evaluate(pm.win_condition, _ctx(tick=300, units_xy=south4))
    # Only 2 tanks in the south zone (wrong-lane) → not win (need 4)
    assert not evaluate(pm.win_condition, _ctx(tick=1000, units_xy=south4[:2]))
    # Past the deadline → fail
    assert evaluate(pm.fail_condition, _ctx(tick=2402, units_xy=south4))


# ── engine-driven: every lazy/wrong policy LOSES, intended WINS ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"lost={r.signals.units_lost} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_brute_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _brute, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: brute must LOSE; got {r.outcome} "
        f"lost={r.signals.units_lost} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_wrong_lane_loses_on_multi_tank_zone_tiers(level, seed):
    """Reinforcing the un-breached NORTH lane leaves the south
    forward zone two tanks short of the n=4 bar → LOSS on
    medium/hard."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _make_reinforce(12), seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: reinforce-wrong-lane must LOSE; got {r.outcome} "
        f"lost={r.signals.units_lost} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_reinforce_the_breach_wins_every_tier_and_seed(level, seed):
    """The intended capability — shift the reserve into the breached
    SOUTH lane — WINS every tier and every hard seed."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _make_reinforce(28), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: reinforce-the-breach must WIN; got {r.outcome} "
        f"lost={r.signals.units_lost} tick={r.signals.game_tick}"
    )


def test_reinforce_run_is_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_reinforce(28), seed=2)
    b = run_level(c, _make_reinforce(28), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
