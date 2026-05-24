"""econ-quantitative-vs-qualitative-spend — REASONING capability validation.

Real-world anchor: SC2 mass-vs-elite army composition; military
quantity-vs-quality doctrine; OR unit-composition optimisation under
a fixed cash budget.

A fixed cash budget ($4500) funds either:

  (A) SWARM (mass cheap): 8× rifle infantry e1 @ $100 = $800.
  (B) ELITE (concentrated armour): 5× medium tank 2tnk @ $800 = $4000.

The pre-placed kit (fact + powr + tent + weap + fix) makes BOTH
compositions actionable on turn 1 — `e1` needs tent (allied infantry
trainer); `2tnk` needs weap + fix (allied service depot — verified
gamerules.rs prereq table) — and the budget can satisfy either
purely. The decision is composition fit to the garrison, not a tech
chain.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe/stop-only) LOSES every tier / every hard seed:
    no production, no kills → kill-bar unmet → real reachable
    timeout LOSS.
  * intended SWARM (build e1 + attack-move) WINS every tier / seed:
    8 rifles overwhelm the single garrison on easy/medium and the
    anti-vehicle garrison on hard (rocket soldiers overkill cheap
    infantry).
  * intended ELITE-ARMOUR (build 2tnk, hold until 3+ tanks, then
    attack-move) WINS on easy/medium: 3-5 medium tanks burn through
    the centre garrison's mixed rifles + rockets cleanly. (On hard
    the south-spawn faces anti-vehicle rockets which shred tanks;
    the north-spawn faces anti-infantry which favours tanks — the
    counter-pick is the load-bearing capability there.)
  * hard tier defines ≥2 agent spawn_point groups (NORTH y=10 /
    SOUTH y=30) so the right composition flips per seed.
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

PACK = PACKS_DIR / "econ-quantitative-vs-qualitative-spend.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _stall(rs, C):
    """Observe/stop-only — no production, no kills → real reachable
    timeout LOSS."""
    return [C.stop([])]


def _swarm(rs, C):
    """Mass cheap composition — keep queueing e1 and attack-move them
    at the centre garrison. The swarm overwhelms the single-garrison
    tiers and the anti-vehicle garrison on hard."""
    cash = rs.get("cash", 0)
    if isinstance(cash, dict):
        cash = cash.get("value", 0)
    prod = [
        x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
    ]
    units = rs.get("units_summary", []) or []
    my_inf = [u for u in units if u.get("type") == "e1"]
    cmds = []
    if cash >= 100 and "e1" not in prod:
        cmds.append(C.build("e1"))
    for u in my_inf:
        cmds.append(C.attack_move([str(u["id"])], target_x=40, target_y=20))
    return cmds if cmds else [C.stop([])]


def _make_armor():
    """Elite armour composition — build 2tnks; HOLD them at base
    until we have 3+, then commit (a piecemeal commit gets
    shredded). The grouped advance breaks the centre garrison on
    easy/medium."""
    state = {"attacks_started": False}

    def policy(rs, C):
        cash = rs.get("cash", 0)
        if isinstance(cash, dict):
            cash = cash.get("value", 0)
        prod = [
            x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
        ]
        units = rs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        if len(tanks) < 5 and cash >= 800 and "2tnk" not in prod:
            cmds.append(C.build("2tnk"))
        if len(tanks) >= 3:
            state["attacks_started"] = True
        if state["attacks_started"]:
            for u in tanks:
                cmds.append(C.attack_move([str(u["id"])], target_x=40, target_y=20))
        return cmds if cmds else [C.stop([])]

    return policy


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_reasoning():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-quantitative-vs-qualitative-spend"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors or "mass" in anchors
    assert "quantity" in anchors or "quality" in anchors


def test_starting_cash_fits_both_compositions():
    """Budget $4500 — fits 8× e1 ($800) or 5× 2tnk ($4000) with margin."""
    pack = load_pack(PACK)
    assert pack.starting_cash >= 4000, (
        f"starting_cash {pack.starting_cash} must fund an elite armour fist"
    )
    assert pack.starting_cash <= 6000, (
        f"starting_cash {pack.starting_cash} not so high that any composition is trivial"
    )


def test_tools_include_build_and_attack_surface():
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("build", "place_building", "move_units",
                     "attack_unit", "attack_move", "stop"):
        assert required in tools, f"missing tool: {required!r}"


def test_preplaced_base_supports_both_compositions():
    """Pre-placed: fact + powr + tent (e1 trainer) + weap (vehicle
    trainer) + fix (2tnk allied tech gate). Both 8-e1 swarm AND
    5-2tnk armour are buildable on turn 1 with no further tech step."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        for sp in {a.spawn_point for a in agent_actors}:
            grp = [a for a in agent_actors if a.spawn_point == sp]
            types = [a.type for a in grp]
            for needed in ("fact", "powr", "tent", "weap", "fix"):
                assert needed in types, (
                    f"{lvl} spawn {sp}: missing {needed!r}; got {types}"
                )


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
        # after_ticks fires inside max_turns so the timeout-LOSS is
        # reachable.
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"


def test_every_level_has_a_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_win_clauses_combine_kills_and_survival():
    """Spec: win = units_killed_gte:K AND units_lost_lte:L AND
    within_ticks:T. The loss cap tightens easy→medium so a brittle
    mini-spend bleeds past it on the tighter tier."""
    pack = load_pack(PACK)
    caps = {}
    for lvl in LEVELS:
        L = pack.levels[lvl]
        clauses = L.win_condition.model_dump()["all_of"]
        has_kills = any("units_killed_gte" in c for c in clauses)
        has_lost = any("units_lost_lte" in c for c in clauses)
        assert has_kills, f"{lvl}: win missing units_killed_gte"
        assert has_lost, f"{lvl}: win missing units_lost_lte"
        caps[lvl] = next(
            int(c["units_lost_lte"]) for c in clauses if "units_lost_lte" in c
        )
    assert caps["medium"] <= caps["easy"], (
        f"medium loss cap {caps['medium']} must be ≤ easy {caps['easy']}"
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
            assert 2 <= x <= 54 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, kills=0, lost=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_kill_and_loss_bars():
    c = compile_level(load_pack(PACK), "easy")
    base = [("fact", 10, 20)]
    # Intended: kills above bar, losses under cap, in time → WIN
    assert evaluate(c.win_condition, _ctx(tick=2000, kills=6, lost=0, own_buildings=base))
    # 5 kills (under bar) → not win
    assert not evaluate(c.win_condition, _ctx(tick=2000, kills=5, lost=0, own_buildings=base))
    # 6 kills but loss cap busted → not win
    assert not evaluate(
        c.win_condition, _ctx(tick=2000, kills=6, lost=10, own_buildings=base)
    )


# ── engine-driven: stall LOSES, both viable paths WIN ───────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """No production → kill bar never met → real reachable LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"kills={r.signals.units_killed} lost={r.signals.units_lost} "
        f"turns={r.turns}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_swarm_wins_every_tier_and_seed(level, seed):
    """Mass cheap composition — WINS every tier and every seed.
    Eight rifles overwhelm the centre garrison on easy/medium; on
    hard, e3 anti-tank rockets overkill cheap rifles so swarm
    counters the south anti-vehicle garrison (and north anti-infantry
    garrison's e1+dog struggles to scale damage against many cheap
    targets in the time window)."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _swarm, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: SWARM (mass cheap) must WIN; got "
        f"{r.outcome} kills={r.signals.units_killed} "
        f"lost={r.signals.units_lost} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ("easy", "medium"))
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_armor_wins_on_easy_medium(level, seed):
    """Elite armour composition (3-5× 2tnk, hold until grouped, then
    attack) — WINS on easy/medium. The centre garrison's mixed
    rifles + rockets dies to a focused tank wedge before the loss
    cap bites. Hard tier flips the right answer per seed (the
    anti-vehicle south garrison shreds tanks); the counter-pick is
    the load-bearing capability tested by the swarm path there."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _make_armor(), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: ARMOR (elite 2tnk wedge) must WIN; got "
        f"{r.outcome} kills={r.signals.units_killed} "
        f"lost={r.signals.units_lost} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _swarm, seed=2)
    b = run_level(c, _swarm, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
