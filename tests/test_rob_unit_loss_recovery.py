"""rob-unit-loss-recovery — REASONING capability validation.

Real-world anchor: PlanBench replanning under exogenous loss (gold
standard); SC2 rebuild-after-trade; military force-regen / unit
replacement; ScienceWorld error recovery. The agent starts with 4
combat heavy tanks (3tnk) + a production base (fact + powr + weap +
fix) and a discrete reserve cash budget; the standing column is BELOW
the force structure the mission requires. The agent must
   1. notice the establishment gap (units_summary shows fewer 3tnk
      than the required quota N),
   2. build('3tnk') to commission the missing tank(s) from the war
      factory with the reserve cash, AND
   3. continue the eastward offensive to clear the e1 garrison
      defending the enemy fact at x=120,
so the kill bar (5 enemy units) + the establishment bar (≥N 3tnk) +
the construction-yard intact bar all pass before the tick deadline.

Recalibrated (2026-05) after the engine movement fixes — the earlier
design relied on the opening 4tnk strike shaving exactly one 3tnk off
the column, but bench-wide combat rebalancing made that casualty
unreliable (a moving heavy-tank column now out-trades the strikers
without loss). The discriminator is now DETERMINISTIC and combat-RNG-
free: the agent's starting column (4 3tnk) is BELOW the establishment
quota N, so fielding the missing tank(s) is a mandatory build verb.
The quota is 5 (easy/medium, one build) and 6 (hard, two builds).

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no attack, no kills →
     kill bar unmet, and the stationary column is ground down by the
     strikers → LOSS).
   - no_rebuild LOSES every tier (the moving column out-trades the
     strikers and clears the garrison so kills ≥5, but the starting
     column is below the quota and a policy that never builds a 3tnk
     can never satisfy `unit_type_count_gte: {type: 3tnk, n: N}` →
     timeout LOSS even with kills ≥5).
   - intended-build-and-attack WINS every tier / seed.
   - hard tier defines ≥2 agent spawn_point groups (NORTH-flank scout
     vs SOUTH-flank scout) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "rob-unit-loss-recovery.yaml"

# The establishment quota (`unit_type_count_gte: {type: 3tnk, n: N}`)
# per tier. The agent starts a 4-tank column on every tier, so the
# quota is the number of mandatory 3tnk builds + 4.
_QUOTA = {"easy": 5, "medium": 5, "hard": 6}


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """Pure observe — no orders → no kills → kill bar unmet, and the
    stationary column is ground down by the strikers → LOSS."""
    return [Command.observe()]


def _no_rebuild(rs, Command):
    """Attack-move east WITHOUT ever issuing build. The moving column
    out-trades the strikers and clears the garrison so kills ≥5, BUT
    the starting 4-tank column is below the establishment quota and a
    policy that never builds a 3tnk can never satisfy
    `unit_type_count_gte: {type: 3tnk, n: N}` → timeout LOSS even
    though kills ≥ 5."""
    units = rs.get("units_summary", []) or []
    tanks = [u for u in units if u.get("type") == "3tnk"]
    if not tanks:
        return [Command.observe()]
    ids = [str(u["id"]) for u in tanks]
    return [Command.attack_move(ids, 80, 20)]


def _make_intended(quota):
    """Build the 3tnk establishment UP to the quota, and always
    attack_move the live 3tnk force toward the eastern garrison at
    (80,20). The garrison falls; the type-count + kill-bar +
    construction-yard bars all pass → WIN."""

    def _intended(rs, Command):
        units = rs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "3tnk"]
        cmds = []
        if len(tanks) < quota:
            cmds.append(Command.build("3tnk"))
        ids = [str(u["id"]) for u in tanks]
        if ids:
            cmds.append(Command.attack_move(ids, 80, 20))
        return cmds or [Command.observe()]

    return _intended


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-unit-loss-recovery"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = pack.meta.benchmark_anchor
    joined = " ".join(anchors).lower()
    assert "planbench" in joined
    assert "scienceworld" in joined
    assert "sc2" in joined
    assert "force-regen" in joined or "force regen" in joined or "rebuild" in joined


def test_uses_turtle_bot():
    """The pack must declare the Wave-2 `turtle` bot — the holding-
    in-place idiom is what isolates the loss event to the pre-placed
    strikers and keeps the kill bar reachable from a single eastbound
    assault (a surging garrison would either overrun the base or be
    killed without the agent moving, breaking both bars)."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "turtle", f"expected turtle bot, got {bot!r}"


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks + 1 == after_ticks (so a
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
    round-robins start by seed. The central combat force + base
    buildings are SHARED across both groups so the strike geometry is
    symmetric, but the spawn-distinguishing scout (NORTH (16,14) vs
    SOUTH (16,28)) reveals which seed the engine picked — a memorised
    opening cannot generalise."""
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


def test_tools_match_spec():
    """The advertised toolset is exactly the rebuild-and-assault kit:
    observe + build + place_building + move_units + attack_unit +
    attack_move + stop. No `harvest` (this is a no-economy scenario —
    the rebuild budget is the starting_cash reserve)."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []))
    expected = {
        "observe", "build", "place_building", "move_units",
        "attack_unit", "attack_move", "stop",
    }
    assert tools == expected, f"tools mismatch: got {sorted(tools)}"


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, kills=0, lost=0, own_buildings=()):
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
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (≥N 3tnk AND ≥5 kills AND fact alive) AND in-time;
    fail fires on timeout OR all-units-dead OR fact destroyed. N is
    the per-tier establishment quota (5 for medium)."""
    c = compile_level(load_pack(PACK), "medium")
    quota = _QUOTA["medium"]

    def _tk(n):
        return [
            {"cell_x": 22, "cell_y": 18 + 2 * i, "type": "3tnk"}
            for i in range(n)
        ]

    fact = [("fact", 8, 18)]

    # Intended: N 3tnk, 5 kills, fact alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_tk(quota), tick=2000, kills=5, own_buildings=fact),
    )
    # Only N-1 3tnk (the starting column, never built up) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_tk(quota - 1), tick=2000, kills=5, own_buildings=fact),
    )
    # Only 4 kills (didn't clear the garrison) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_tk(quota), tick=2000, kills=4, own_buildings=fact),
    )
    # All units dead → real fail (capability collapses)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, kills=5, own_buildings=fact),
    )
    # Timeout (tick past after_ticks): bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_tk(quota), tick=5402, kills=0, own_buildings=fact),
    )
    # Construction yard destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_tk(quota), tick=2000, kills=5, own_buildings=[]),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No orders → no kills → kill bar unmet → timeout LOSS via
    after_ticks. The strike force kills 1-4 agent tanks during the
    grind, but the agent neither rebuilds nor advances on the
    garrison."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (no kills, no rebuild); "
        f"got {r.outcome} kills={r.signals.units_killed} "
        f"losses={r.signals.units_lost} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_no_rebuild_loses(level, seed):
    """Assault-without-build: the moving column clears the garrison so
    kills ≥ 5, but the starting 4-tank column is below the
    establishment quota and the type-count clause is never satisfied
    because no 3tnk is ever built → timeout LOSS even with a cleared
    kill bar. This is the canonical 'never built up to establishment'
    failure mode the scenario is designed to catch."""
    _, r = _run(level, _no_rebuild, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: no-rebuild must LOSE (establishment "
        f"quota never reached); got {r.outcome} "
        f"kills={r.signals.units_killed} losses={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_rebuild_and_attack_wins(level, seed):
    """The intended capability — build the 3tnk establishment up to
    the quota AND attack_move the live force east — WINS every tier
    and every hard seed well inside the tick budget."""
    _, r = _run(level, _make_intended(_QUOTA[level]), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended build-and-attack should WIN; "
        f"got {r.outcome} kills={r.signals.units_killed} "
        f"losses={r.signals.units_lost} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    pol = _make_intended(_QUOTA["medium"])
    a = run_level(c, pol, seed=2)
    b = run_level(c, pol, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed, a.signals.units_lost) == (
        b.outcome, b.turns, b.signals.units_killed, b.signals.units_lost
    )
