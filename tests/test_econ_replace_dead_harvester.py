"""econ-replace-dead-harvester — REASONING capability validation.

Real-world anchor: PlanBench replanning under exogenous loss; SC2
replace-killed-workers reflex; ScienceWorld error recovery; disaster
recovery worker replacement. The agent starts with 3 working harvs +
a small reserve cash (exactly enough for ONE replacement at easy/med,
TWO at hard). An enemy 4tnk strike via the `raider` worker-priority
bot kills 1 (or 2) harvs early in the episode; the agent must
   1. notice the loss (units_summary shows < 3 harvs),
   2. build('harv') to replace the dead one with the indivisible
      reserve cash, AND
   3. re-issue `harvest` to every harvester (the death event
      interrupts the surviving harvs' harvest loop),
so throughput recovers before the deadline.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no harvest cmd → EV
     stuck at starting_cash → bar unmet → timeout LOSS).
   - over-build-army LOSES every tier (cash spent on combat units
     instead of the replacement harv → no extra throughput → bar
     unmet → timeout LOSS).
   - intended build-harv + harvest-redirect WINS every tier / seed.
   - hard tier defines ≥2 agent spawn_point groups (NORTH / SOUTH
     base) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-replace-dead-harvester.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    return [Command.observe()]


def _over_build_army(rs, Command):
    """Spend the reserve on combat units instead of the replacement
    harv — the canonical "wrong commit" failure mode."""
    return [Command.build("1tnk")]


def _replace_only_no_redirect(rs, Command):
    """Build replacement harv but never issue harvest to ANY harv
    (existing or new). NB the engine auto-harvests a freshly produced
    harv when ore is nearby, so this policy CAN succeed at easy/med
    bars — it's recorded for documentation, not as a strict LOSS
    proxy."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    if len(harvs) < 3:
        return [Command.build("harv")]
    return [Command.observe()]


def _intended(rs, Command):
    """Replan: re-issue `harvest` to every surviving harv each turn
    (their loop is interrupted by the kill event); if harv count is
    below 3, build a replacement. The patches at (22,18) and (22,22)
    work for every spawn (central geometry on hard)."""
    units = rs.get("units_summary", []) or []
    cmds = []
    harvs = [u for u in units if u.get("type") == "harv"]
    patches = [(22, 18), (22, 22)]
    for i, h in enumerate(harvs):
        mx, my = patches[i % len(patches)]
        cmds.append(Command.harvest([str(h["id"])], mx, my))
    if len(harvs) < 3:
        cmds.append(Command.build("harv"))
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
    assert pack.meta.id == "econ-replace-dead-harvester"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = pack.meta.benchmark_anchor
    joined = " ".join(anchors).lower()
    assert "planbench" in joined
    assert "scienceworld" in joined
    assert "sc2" in joined
    assert "disaster" in joined or "recovery" in joined


def test_uses_raider_bot():
    """The pack must declare the Wave-2 `raider` bot — the worker-
    priority idiom is what makes the harv-kill load-bearing."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "raider", f"expected raider bot, got {bot!r}"


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks == after_ticks (so a
    non-finisher LOSES, not draws)."""
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
    round-robins start by seed. The replacement-replan task is the
    same per spawn but base orientation flips (NORTH y=14 vs SOUTH
    y=26), so a memorised opening cannot generalise."""
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


def _ctx(*, units=(), tick=1000, cash=0, resources=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=resources,
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
    """Win requires (EV bar AND ≥3 harvs AND proc) AND in-time; fail
    fires on timeout OR all-harvs-dead OR proc destroyed."""
    c = compile_level(load_pack(PACK), "medium")
    three_harvs = [
        {"cell_x": 22, "cell_y": 18, "type": "harv"},
        {"cell_x": 22, "cell_y": 20, "type": "harv"},
        {"cell_x": 22, "cell_y": 22, "type": "harv"},
    ]
    proc = [("proc", 12, 18)]

    # Intended: bar met, 3 harvs, proc alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=2000, cash=2500, own_buildings=proc),
    )
    # Bar one short of 2500 → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=2000, cash=2499, own_buildings=proc),
    )
    # Only 2 harvs (didn't replace) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs[:2], tick=2000, cash=4000, own_buildings=proc),
    )
    # All harvs dead → real fail (capability collapses)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, cash=4000, own_buildings=proc),
    )
    # Timeout (tick past after_ticks): bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=three_harvs, tick=5402, cash=0, own_buildings=proc),
    )
    # Proc destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=three_harvs, tick=2000, cash=4000, own_buildings=[]),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No harvest order → existing harvs stay idle → EV stuck at
    starting_cash → bar unmet → timeout LOSS via after_ticks."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (EV stuck at "
        f"starting_cash, bar unmet); got {r.outcome} ev={_ev(r)} "
        f"turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_over_build_army_loses(level, seed):
    """Spending the indivisible reserve on combat units (1tnk) leaves
    no cash for the harv replacement; harv count stays below 3 (the
    win clause's hard constraint) and throughput never recovers →
    timeout LOSS."""
    _, r = _run(level, _over_build_army, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: over-build-army must LOSE; got "
        f"{r.outcome} ev={_ev(r)} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_replan_wins(level, seed):
    """The intended capability — replace the dead harv AND re-issue
    `harvest` to every survivor — WINS every tier and every hard seed
    well inside the tick budget."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended replan should WIN; got "
        f"{r.outcome} ev={_ev(r)} turns={r.turns} losses={r.signals.units_lost}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and EV."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b))
