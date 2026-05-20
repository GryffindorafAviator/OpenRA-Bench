"""combat-rocket-soldier-anti-vehicle — REASONING / RPS hard-counter.

The bar (no-cheat + solvency, four scripted policies):
  • stall (only observe)           → LOSS (kill bar + e3 count never
    met; after_ticks deadline bites)
  • build-1tnk (light tanks)       → LOSS (light cannons lose attrition
    to heavy armour; the e3-type bar never latches; timeout fires)
  • build-e1 (rifle infantry)      → LOSS (rifles have no anti-armour
    weapon; even if they accidentally score a kill, the e3-type bar
    never latches; timeout fires)
  • intended build-e3 + advance    → WIN on every level and every hard
    seed (rocket soldiers shred heavy armour at range; the kill bar
    and unit-type count bar both latch well inside the clock)

Non-win is a real reachable timeout LOSS via the `after_ticks` fail
clause (verified inside max_turns ⇒ tick ≤ 93 + 90·(max_turns-1)).

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
PACK_PATH = PACKS / "combat-rocket-soldier-anti-vehicle.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-rocket-soldier-anti-vehicle"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list ≥3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2 hard-counter", "anti-armor procurement", "military rps"):
        assert needle in joined, f"missing required anchor keyword: {needle!r}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported, f"{lvl}: base_map must resolve"
        assert c.win_condition is not None
        assert c.fail_condition is not None
        assert c.starting_cash == 1800, (
            f"{lvl}: starting_cash should be 1800 (funds exactly one "
            f"composition: 6× e3); got {c.starting_cash}"
        )


def _ctx(*, e3_count=0, tick=1000, kills=0, lost=0, has_fact=True):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types={"fact"} if has_fact else set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    units = [
        {"cell_x": 10, "cell_y": 20, "type": "e3", "id": str(1000 + i)}
        for i in range(e3_count)
    ]
    return WinContext(
        signals=sig,
        render_state={"units_summary": units},
    )


def test_easy_predicates_enforce_e3_type_and_kill_and_fact():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 6 e3, 1 kill, fact alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(e3_count=6, tick=2000, kills=1))
    # Only 5 e3 → predicate fails (need 6)
    assert not evaluate(c.win_condition, _ctx(e3_count=5, tick=2000, kills=1))
    # 6 e3 but 0 kills → predicate fails
    assert not evaluate(c.win_condition, _ctx(e3_count=6, tick=2000, kills=0))
    # Lost fact → fail clause fires
    assert evaluate(c.fail_condition, _ctx(e3_count=6, tick=2000, kills=1, has_fact=False))
    # Timeout with bar unmet → fail (after_ticks 4501)
    assert evaluate(c.fail_condition, _ctx(e3_count=2, tick=4502, kills=0))


def test_medium_predicates_tighter_kill_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 6 e3, 2 kills → WIN
    assert evaluate(c.win_condition, _ctx(e3_count=6, tick=2500, kills=2))
    # 1 kill (medium needs 2) → not a win
    assert not evaluate(c.win_condition, _ctx(e3_count=6, tick=2500, kills=1))
    # Timeout with bar unmet → fail
    assert evaluate(c.fail_condition, _ctx(e3_count=6, tick=4502, kills=1))
    # Fact down → fail
    assert evaluate(c.fail_condition, _ctx(e3_count=6, tick=2500, kills=2, has_fact=False))


def test_hard_predicates_tighter_deadline():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 6 e3, 2 kills → WIN
    assert evaluate(c.win_condition, _ctx(e3_count=6, tick=2500, kills=2))
    # Just past the hard deadline → fail
    assert evaluate(c.fail_condition, _ctx(e3_count=6, tick=3602, kills=1))
    # Bar unmet (5 e3) → not a win
    assert not evaluate(c.win_condition, _ctx(e3_count=5, tick=2500, kills=2))


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    bounds = {"easy": 4501, "medium": 4501, "hard": 3601}
    for lvl, bound in bounds.items():
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert bound <= max_tick, (
            f"{lvl}: after_ticks {bound} > max reachable tick "
            f"{max_tick}; deadline never bites → draw degeneracy"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract (tests/test_hard_tier.py::UPGRADED):
    ≥2 distinct agent spawn_point groups so the seed round-robins the
    base latitude (NORTH y=12 / SOUTH y=28)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_enemy_band_is_heavy_armour_and_persistent_fact_present():
    """Every level must include heavy armour (3tnk or 4tnk) as the
    enemy band — that is what makes the e3 (anti-vehicle) the RPS
    hard-counter. A persistent far-east enemy fact prevents engine
    auto-done on enemy unit wipe."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        n_heavy = sum(1 for t in types if t in ("3tnk", "4tnk"))
        assert n_heavy >= 2, (
            f"{lvl}: must include ≥2 heavy tanks (3tnk/4tnk); got {types}"
        )
        assert "fact" in types, (
            f"{lvl}: needs a persistent enemy fact (engine auto-done mitigation)"
        )


def test_agent_base_has_both_counter_prereqs():
    """Both counters (e3 via tent / 1tnk via weap) must be PRODUCIBLE
    from the pre-placed base — the RPS choice must not be gated by
    missing prereqs, so the load-bearing decision is purely which unit
    TYPE counters heavy armour."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "agent"]
        # tent enables e3 (the intended counter), weap enables 1tnk
        # (the wrong tank counter). Both must be present.
        assert "tent" in types, f"{lvl}: needs tent (barracks) to build e3"
        assert "weap" in types, f"{lvl}: needs weap (war factory) to build 1tnk"
        assert "fact" in types, f"{lvl}: needs fact (construction yard)"
        assert "powr" in types, f"{lvl}: needs powr (power plant)"


# ── engine-driven scripted policies ─────────────────────────────────


def _own_units(rs, *, type_filter=None):
    out = []
    for u in (rs.get("units_summary", []) or []):
        if type_filter and (u.get("type") or "").lower() not in type_filter:
            continue
        out.append(u)
    return out


def _heavy_tanks(rs):
    return [
        e for e in (rs.get("enemy_summary") or [])
        if (e.get("type") or "").lower() in ("3tnk", "4tnk")
        and not e.get("is_building")
    ]


def _stall(rs, Command):
    """Pure observe — no production, no advance. The kill bar and the
    unit_type_count_gte:e3:6 bar never latch → after_ticks LOSS."""
    return [Command.observe()]


def _build_1tnk_policy(rs, Command):
    """WRONG counter — light tanks. The $1800 budget buys only ~2
    1tnks; light cannons lose attrition to heavy armour. The e3-type
    bar never latches either way → LOSS."""
    cmds = []
    cash = rs.get("cash", 0)
    if cash >= 700:
        cmds.append(Command.build("1tnk"))
    fighters = _own_units(rs, type_filter={"1tnk"})
    targets = _heavy_tanks(rs)
    for u in fighters:
        if targets:
            cmds.append(Command.attack_unit([str(u["id"])], str(targets[0]["id"])))
        else:
            cmds.append(Command.attack_move([str(u["id"])], 68, u["cell_y"]))
    return cmds if cmds else [Command.observe()]


def _build_e1_policy(rs, Command):
    """WRONG counter — rifle infantry. e1 has no anti-armour weapon;
    rifle volleys cannot scratch heavy tank armour. Even if a stray
    shot somehow lands, the unit-type bar (e3:6) never latches → LOSS."""
    cmds = []
    cash = rs.get("cash", 0)
    if cash >= 100:
        cmds.append(Command.build("e1"))
    fighters = _own_units(rs, type_filter={"e1"})
    targets = _heavy_tanks(rs)
    for u in fighters:
        if targets:
            cmds.append(Command.attack_unit([str(u["id"])], str(targets[0]["id"])))
        else:
            cmds.append(Command.attack_move([str(u["id"])], 68, u["cell_y"]))
    return cmds if cmds else [Command.observe()]


def _intended_build_e3_policy(rs, Command):
    """INTENDED counter — rocket soldiers (anti-vehicle Dragon launcher).
    Queue e3 every turn the budget allows; once produced, attack_unit
    the visible heavy tank (engine auto-targets work too but explicit
    attack_unit is cleaner). $1800 buys exactly 6× e3, which is the
    win bar (unit_type_count_gte:e3:6 AND units_killed_gte:K)."""
    cmds = []
    cash = rs.get("cash", 0)
    if cash >= 300:
        cmds.append(Command.build("e3"))
    e3s = _own_units(rs, type_filter={"e3"})
    targets = _heavy_tanks(rs)
    for u in e3s:
        if targets:
            cmds.append(Command.attack_unit([str(u["id"])], str(targets[0]["id"])))
        else:
            cmds.append(Command.attack_move([str(u["id"])], 68, u["cell_y"]))
    return cmds if cmds else [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _stall, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: stall must be a real timeout LOSS; got "
            f"{r.outcome} (kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_1tnk_wrong_counter_loses(level):
    """Light tanks lose attrition to heavy armour AND the e3-type bar
    never latches → real LOSS on every level."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _build_1tnk_policy, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: 1tnk wrong-counter must LOSE; got "
            f"{r.outcome} (kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_e1_wrong_counter_loses(level):
    """Rifle infantry has no anti-armour weapon AND the e3-type bar
    never latches → real LOSS on every level."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _build_e1_policy, seed=s)
        assert r.outcome == "loss", (
            f"{level} seed={s}: e1 wrong-counter must LOSE; got "
            f"{r.outcome} (kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_build_e3_wins(level):
    """The RPS counter pick: train 6× e3 (anti-vehicle rocket soldiers)
    and engage. Wins on every level; hard wins on every seed 1..4."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        r = run_level(c, _intended_build_e3_policy, seed=s)
        assert r.outcome == "win", (
            f"{level} seed={s}: intended e3 must WIN; got {r.outcome} "
            f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
        )
