"""def-counter-battery — kill the artillery FIRST (counter-battery).

The bar (binding, applied seeds 1..4 on every level):
- intended counter-battery strike (kill the artillery, THEN mop the
  screen) WINS;
- stall / brute attack_move / screen-first all LOSE on every level
  and every seed — a wrong-priority play that grinds the infantry
  screen lets the artillery raze the construction yard;
- non-win is a real reachable LOSS (fact razed, or after_ticks 2701
  ≤ 93 + 90·(max_turns − 1)) — no DRAW degeneracy;
- hard ships ≥2 `spawn_point` groups (seed-driven base latitude).

Validation is scripted (no model / no network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "def-counter-battery.yaml"


# ── declarative / schema invariants (no engine needed) ─────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "def-counter-battery"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 3, (
        f"benchmark_anchor must list all 3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("counter-battery", "threat prioritization",
                   "siege-tank"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, buildings=(), tick=1000, kills=0, units=1):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=0,
        cash=0,
        resources=0,
        own_buildings=list(buildings),
        own_building_types={t for (t, _, _) in buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"cell_x": 10, "cell_y": 20, "type": "2tnk", "id": str(i)}
            for i in range(units)
        ]},
    )


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_win_requires_kills_and_fact_alive(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    K = 12 if lvl == "easy" else 14
    fact = [("fact", 20, 21)]
    # Kill bar met, fact alive, in time → WIN.
    assert evaluate(c.win_condition,
                    _ctx(buildings=fact, tick=2000, kills=K))
    # Kill bar met but fact razed → not a WIN.
    assert not evaluate(c.win_condition,
                        _ctx(buildings=[], tick=2000, kills=K))
    # Fact alive but kill bar short → not a WIN.
    assert not evaluate(c.win_condition,
                        _ctx(buildings=fact, tick=2000, kills=K - 1))
    # Everything met but past the deadline → not a WIN.
    assert not evaluate(c.win_condition,
                        _ctx(buildings=fact, tick=2800, kills=K))


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_fail_clauses_are_reachable_losses(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    # Fact razed (the artillery did its job) → fail.
    assert evaluate(c.fail_condition,
                    _ctx(buildings=[], tick=2000, kills=5))
    # Past the deadline → fail.
    assert evaluate(c.fail_condition,
                    _ctx(buildings=[("fact", 20, 21)], tick=2800))
    # Force wiped → fail.
    assert evaluate(c.fail_condition,
                    _ctx(buildings=[("fact", 20, 21)], tick=2000, units=0))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 2701 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 2701 <= max_tick, (
            f"{lvl}: after_ticks 2701 > max reachable tick {max_tick}"
        )
        assert 2700 <= max_tick, (
            f"{lvl}: within_ticks 2700 > max reachable tick {max_tick}"
        )


def test_each_level_has_artillery_outranging_the_pillboxes():
    """The load-bearing threat is `arty` (range 7c) which out-ranges
    the agent's `pbox` (range 5c). Without artillery the pack has no
    counter-battery decision; without pillboxes the 'out-ranges the
    base defences' framing is empty."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        artys = [a for a in c.scenario.actors
                 if a.owner == "enemy" and a.type == "arty"]
        pboxes = [a for a in c.scenario.actors
                  if a.owner == "agent" and a.type == "pbox"]
        n_arty = 2 if lvl == "easy" else 3
        # hard duplicates the battery across 2 spawn groups
        if lvl == "hard":
            n_arty *= 2
        assert len(artys) == n_arty, (
            f"{lvl}: expected {n_arty} arty, got {len(artys)}"
        )
        assert pboxes, f"{lvl}: must have ≥1 pbox (base defence)"


def test_screen_is_passive_holdfire_infantry():
    """The infantry screen is `e1` on stance:0 (HoldFire) — a passive
    picket line that neither chips the fact nor shoots the strike
    force, so the artillery's razing clock is deterministic. A
    stance:3 screen near the fact would accelerate razing AND killing
    it would slow razing, collapsing the discrimination."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        screen = [a for a in c.scenario.actors
                  if a.owner == "enemy" and a.type == "e1"]
        assert screen, f"{lvl}: must have an e1 screen"
        for e in screen:
            assert e.stance == 0, (
                f"{lvl}: screen e1 must be stance:0 (passive), "
                f"got stance={e.stance}"
            )


def test_artillery_is_stance_two_defend():
    """The artillery is stance:2 (Defend — auto-fires the in-range
    fact, never advances). A stationary battery keeps the razing
    window a stable clock independent of the agent's play."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        artys = [a for a in c.scenario.actors
                 if a.owner == "enemy" and a.type == "arty"]
        for a in artys:
            assert a.stance == 2, (
                f"{lvl}: arty must be stance:2 (Defend), "
                f"got stance={a.stance}"
            )


def test_hard_has_two_spawn_point_groups():
    """Hard curation: ≥2 distinct agent spawn_point groups so the seed
    round-robins the base latitude (NORTH y≈14 / SOUTH y≈28)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_hard_enemy_battery_tagged_per_spawn_group():
    """The enemy screen+battery is tagged with the SAME spawn_point as
    the matching base so the per-owner spawn filter instantiates one
    self-consistent base+battery pair per seed (the persistent far
    anti-DRAW fact is the only un-tagged enemy actor)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    untagged_enemy = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is None
    ]
    assert len(untagged_enemy) == 1, (
        "exactly one un-tagged enemy actor expected (the far anti-DRAW "
        f"fact marker), got {[(a.type) for a in untagged_enemy]}"
    )
    assert untagged_enemy[0].type == "fact"


# ── engine-driven scripted policies ────────────────────────────────


def _tanks(rs):
    return [u for u in (rs.get("units_summary", []) or [])
            if str(u.get("type", "")).lower() == "2tnk"]


def _stall(rs, Command):
    """Pure observe — the tanks never move; the artillery razes the
    fact unopposed → `not building_count_gte:fact:1` LOSS."""
    return [Command.observe()]


def _brute_attack_move(rs, Command):
    """attack_move east — the engine auto-targets the NEAREST hostile
    (the infantry screen), so the tanks halt and trade with the
    screen while the rear artillery shells the fact → LOSS."""
    ts = _tanks(rs)
    if not ts:
        return [Command.observe()]
    ay = int(sum(t["cell_y"] for t in ts) / len(ts))
    return [Command.attack_move([str(t["id"]) for t in ts], 27, ay)]


def _screen_first(rs, Command):
    """Wrong priority — explicitly focus-fire the infantry screen
    FIRST, only pivoting to the artillery once the screen is clear.
    The screen is deep enough that the fact is razed before the
    pivot → LOSS."""
    ts = _tanks(rs)
    if not ts:
        return [Command.observe()]
    es = rs.get("enemy_summary", []) or []
    ids = [str(t["id"]) for t in ts]
    e1s = [e for e in es if str(e.get("type", "")).lower() == "e1"]
    if e1s:
        return [Command.attack_unit(ids, str(e1s[0]["id"]))]
    artys = [e for e in es if str(e.get("type", "")).lower() == "arty"]
    if artys:
        return [Command.attack_unit(ids, str(artys[0]["id"]))]
    ay = int(sum(t["cell_y"] for t in ts) / len(ts))
    return [Command.move_units(ids, 24, ay)]


def _counter_battery_strike(rs, Command):
    """The intended play — drive the tanks toward the artillery and
    attack_unit the guns FIRST (highest-impact threat); only once the
    battery is silent mop up the infantry screen. Saving the fact
    leaves the rest of the budget to clear the kill bar → WIN."""
    ts = _tanks(rs)
    if not ts:
        return [Command.observe()]
    es = rs.get("enemy_summary", []) or []
    ids = [str(t["id"]) for t in ts]
    artys = [e for e in es if str(e.get("type", "")).lower() == "arty"]
    if artys:
        return [Command.attack_unit(ids, str(artys[0]["id"]))]
    e1s = [e for e in es if str(e.get("type", "")).lower() == "e1"]
    if e1s:
        return [Command.attack_unit(ids, str(e1s[0]["id"]))]
    ay = int(sum(t["cell_y"] for t in ts) / len(ts))
    return [Command.move_units(ids, 27, ay)]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_counter_battery_strike_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _counter_battery_strike, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: counter-battery strike should WIN, got "
        f"{r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real LOSS (the "
        f"artillery razes the fact), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_attack_move_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_attack_move, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute attack_move must LOSE (auto-"
        f"targets the screen while the artillery shells the fact), "
        f"got {r.outcome} (kills={r.signals.units_killed})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_screen_first_loses(level, seed):
    """The wrong-priority play — grinding the infantry screen before
    the artillery — must LOSE on every level and seed: the fact is
    razed before the strike pivots to the guns."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _screen_first, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: screen-first must LOSE (the artillery "
        f"razes the fact before the screen is cleared), got "
        f"{r.outcome} (kills={r.signals.units_killed})"
    )
