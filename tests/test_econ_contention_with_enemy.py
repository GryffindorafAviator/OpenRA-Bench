"""econ-contention-with-enemy — REASONING capability validation.

Common-pool contested mining. A shared central ore patch sits between
the agent and an enemy that fields its own (flavor) harvesters AND a
worker-priority `raider` bot that drives at the agent's harvesters.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / seed (no harvest cmd → EV stuck at the
     starting_cash floor → bar unmet → timeout LOSS).
   - pure-mine-no-defense LOSES (harvest but move defenders FAR off-
     vector → raider's worker-priority attack runs unopposed → harvs
     die → fail by harv count).
   - pure-defend LOSES (defenders engage, NO harvest orders → no
     income → bar unmet → timeout LOSS).
   - intended-balance WINS (harvest + leave the pre-ringed defenders
     on station — they auto-engage on Defend stance; throughput climbs
     uninterrupted).
   - hard tier defines ≥2 agent spawn_point groups (NORTH y=14 /
     SOUTH y=26 base orientation, rear-guard tank marks the spawn)
     so a memorised opening cannot generalise.

Recalibrated 2026-05 after the engine movement fixes ((A) attack_unit
on out-of-sight targets paths normally, (B) moving units fire and take
fire en route, stance-respecting): a lone weak raider could no longer
overrun an undefended harv lane, so easy's pure-mine-no-defense leaked
into a win. The easy raider strike is now TWO 3tnk and the easy
defender ring FOUR 3tnk so pure-mine LOSES and intended still WINS.

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
    """Harvest only — move defenders to the FAR NW corner so the
    raider's worker-priority attack on the harv lane runs unopposed.

    Movement uses `move_units` (not `attack_move`) so defenders don't
    sweep-engage the raider en route; they leave the harv lane and
    don't come back. The patch column at x=22 works for every spawn
    (the harv geometry is shared north/south on hard)."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    tanks = [u for u in units if u.get("type") in ("1tnk", "3tnk")]
    cmds = []
    for h in harvs:
        cmds.append(Command.harvest([str(h["id"])], 22, 20))
    for t in tanks:
        cmds.append(Command.move_units([str(t["id"])], 5, 5))
    return cmds or [Command.observe()]


def _pure_defend(rs, Command):
    """Defenders attack-move east, NO harvest orders → no income."""
    units = rs.get("units_summary", []) or []
    tanks = [u for u in units if u.get("type") in ("1tnk", "3tnk")]
    cmds = []
    for t in tanks:
        cmds.append(Command.attack_move([str(t["id"])], 60, 20))
    return cmds or [Command.observe()]


def _intended(rs, Command):
    """Committed balance: every harv in `harvest` mode at the contested
    patch (22,20); defenders stay ringed and auto-engage the raider on
    default Defend stance — no explicit defender orders needed."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = []
    for h in harvs:
        cmds.append(Command.harvest([str(h["id"])], 22, 20))
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
            "(non-finisher must LOSE, not draw — fail clause one tick"
            " past win clause)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 distinct agent spawn_point groups so engine
    round-robins start by seed. The committed-balance task is the same
    per spawn but base orientation flips (NORTH y=14 vs SOUTH y=26),
    so a memorised opening cannot generalise."""
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


def _ctx(*, units=(), tick=1000, cash=0, resources=0):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
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
    """Win requires (EV bar AND ≥2 harvs) AND in-time; fail fires on
    timeout OR all-harvs-dead."""
    c = compile_level(load_pack(PACK), "medium")
    two_harvs = [
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
        {"cell_x": 14, "cell_y": 20, "type": "harv"},
    ]

    # Intended: bar met, 2 harvs, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=2000, cash=3500),
    )
    # Bar one short of 3500 → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=2000, cash=3499),
    )
    # Only 1 harv (raider got one) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs[:1], tick=2000, cash=10000),
    )
    # All harvs dead → real fail (capability collapses)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, cash=10000),
    )
    # Timeout (tick past after_ticks): bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=two_harvs, tick=5402, cash=0),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No harvest order → harvs stay idle → EV stuck at starting_cash
    (500) → bar unmet → timeout LOSS via after_ticks."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_mine_no_defense_loses(level, seed):
    """Harvest + abandon the defender ring (move defenders to NW
    corner). The raider's worker-priority attack runs unopposed and
    kills the harvs at the contested patch → fail by harv count."""
    _, r = _run(level, _pure_mine_no_defense, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-mine-no-defense must LOSE; got "
        f"{r.outcome} ev={_ev(r)} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_defend_loses(level, seed):
    """Defenders engage, NO harvest orders → no income → bar unmet
    → timeout LOSS via after_ticks."""
    _, r = _run(level, _pure_defend, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-defend must LOSE; got {r.outcome} "
        f"ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_balance_wins(level, seed):
    """The intended capability — harvest + ringed defenders (auto-
    engage) — WINS every tier and every hard seed comfortably inside
    the tick budget."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended-balance should WIN; got "
        f"{r.outcome} ev={_ev(r)} turns={r.turns} "
        f"lost={r.signals.units_lost}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and EV."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
