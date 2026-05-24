"""econ-contention-with-enemy — REASONING capability validation.

Common-pool contested mining. A shared central ore patch sits between
the agent and an enemy that fields its own (flavor) harvesters AND a
worker-priority `raider` bot that drives at the agent's harvesters.

The defenders are pre-placed at the WEST base edge (x=6) — well west
of the harv path (x=14..22). Base sight does NOT reach the patch, so
a stance:2 defender at the base will NOT auto-engage a raider that
attacks the harv at the patch. The agent must explicitly attack-move
the defenders forward to intercept.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / seed (no defender movement → raider
     reaches the harvs and kills them → harv-count fail fires).
   - pure-defend LOSES (defenders advance, harvs explicitly stopped
     → no income → EV bar unmet → timeout LOSS).
   - pure-mine-no-defense LOSES (defenders moved off-vector → raider
     kills harvs → fail by harv count).
   - intended-balance WINS (defenders attack-move / attack-unit the
     raider; harvs auto-route to the patch).
   - hard tier defines ≥2 agent spawn_point groups (NORTH y=14 /
     SOUTH y=26) so a memorised opening cannot generalise.

Anchors: SC2 contested expansion / split-mining; game theory common-
pool resource (Hardin tragedy of commons); competitive market entry /
TAM contention; fishing common-resource dynamics.
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

PACK = PACKS_DIR / "econ-contention-with-enemy.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    return [Command.observe()]


def _pure_mine_no_defense(rs, Command):
    """Harvest only — move defenders FAR to the NW corner so the
    raider's worker-priority attack on the harv lane runs unopposed.
    The raider's 3tnk overruns the auto-routed harvs at the patch.
    """
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    tanks = [u for u in units if u.get("type") in ("2tnk", "3tnk", "1tnk")]
    cmds = []
    for h in harvs:
        my = int(h.get("cell_y", 20))
        cmds.append(Command.harvest([str(h["id"])], 22, my))
    for t in tanks:
        cmds.append(Command.move_units([str(t["id"])], 4, 4))
    return cmds or [Command.observe()]


def _pure_defend(rs, Command):
    """Defenders advance east to intercept the raider; harvs are
    explicitly stopped (no income). The EV bar is unmet → LOSS."""
    units = rs.get("units_summary", []) or []
    cmds = []
    for u in units:
        if u.get("type") in ("2tnk", "3tnk"):
            cmds.append(Command.attack_move([str(u["id"])], 30, u.get("cell_y", 20)))
        elif u.get("type") == "harv":
            cmds.append(Command.stop([str(u["id"])]))
    return cmds or [Command.observe()]


def _intended(rs, Command):
    """Committed balance: defenders explicitly attack the raider(s);
    harvs auto-route to the patch (the engine auto-routes pre-placed
    harvs once a proc exists, so no explicit harvest order needed).
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tanks = [u for u in units if u.get("type") in ("2tnk", "3tnk", "1tnk")]
    raiders = [
        e for e in enemies
        if e.get("type") in ("3tnk", "2tnk", "1tnk")
        and not e.get("is_building", False)
    ]
    cmds = []
    if raiders:
        for i, t in enumerate(tanks):
            r = raiders[i % len(raiders)]
            cmds.append(Command.attack_unit([str(t["id"])], str(r["id"])))
    else:
        for t in tanks:
            cmds.append(Command.attack_move([str(t["id"])], 25, t.get("cell_y", 20)))
    return cmds or [Command.observe()]


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


def _ev(res):
    return res.signals.cash + res.signals.resources


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-contention-with-enemy"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = pack.meta.benchmark_anchor
    joined = " ".join(anchors).lower()
    assert "sc2" in joined
    assert "common-pool" in joined or "commons" in joined
    assert "tam" in joined or "market entry" in joined
    assert "fishing" in joined


def test_uses_raider_bot():
    """The pack must declare the Wave-2 `raider` bot — the worker-
    priority idiom is what binds the contestation cost to the harv
    line (not generic combat)."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "raider", f"expected raider bot, got {bot!r}"


def test_all_tiers_have_reachable_deadlines():
    """Tick-alignment idiom: within_ticks ≤ ceiling AND after_ticks
    ≤ ceiling AND within_ticks + 1 == after_ticks (non-finisher
    LOSES, not draws)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
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
            "(non-finisher must LOSE, not draw)"
        )


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


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, cash=0, resources=0, killed=0):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=0,
        cash=cash,
        resources=resources,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (EV bar AND ≥2 harvs AND ≥1 kill) AND in-time;
    fail fires on timeout OR all-harvs-dead."""
    c = compile_level(load_pack(PACK), "medium")
    two_harvs = [
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
        {"cell_x": 14, "cell_y": 20, "type": "harv"},
    ]

    # Intended: bar met, 2 harvs, ≥2 kills (med bar), in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=2000, cash=2200, killed=2),
    )
    # Bar one short of 2200 → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=2000, cash=2199, killed=2),
    )
    # No kills — auto-harvest alone doesn't satisfy
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=2000, cash=10000, killed=0),
    )
    # Only 1 harv (raider got one) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs[:1], tick=2000, cash=10000, killed=2),
    )
    # All harvs dead → real fail (capability collapses)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, cash=10000, killed=2),
    )
    # Timeout (tick past after_ticks): bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=two_harvs, tick=4502, cash=0, killed=0),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No defender movement → raider reaches the harvs and kills
    them → harv-count fail fires."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns} units_lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_mine_no_defense_loses(level, seed):
    """Harvest + abandon the defenders (move them to NW corner). The
    raider's worker-priority attack runs unopposed and kills the harvs
    at the contested patch → fail by harv count."""
    _, r = _run(level, _pure_mine_no_defense, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-mine-no-defense must LOSE; got "
        f"{r.outcome} ev={_ev(r)} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_defend_loses(level, seed):
    """Defenders engage, harvs explicitly stopped → no income → bar
    unmet → timeout LOSS via after_ticks."""
    _, r = _run(level, _pure_defend, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-defend must LOSE; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_balance_wins(level, seed):
    """The intended capability — defenders attack-unit the raider while
    harvs auto-mine — WINS every tier and every hard seed."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended-balance should WIN; got "
        f"{r.outcome} ev={_ev(r)} turns={r.turns} "
        f"killed={r.signals.units_killed} lost={r.signals.units_lost}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and EV."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
