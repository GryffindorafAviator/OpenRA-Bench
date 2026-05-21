"""rob-partial-base-loss-continue — REASONING capability validation.

Graceful degradation / incident triage. A mid-episode
`scheduled_events: destroy_actors` hook razes a NON-CRITICAL slice of
the agent's base — an outer pillbox and one of two harvesters — at
tick ~1200, while the war-critical core (fact + proc + powr + weap +
fix) is untouched. The intended policy TRIAGES the loss: it
recognises the razed assets are not on the win path and PRESSES ON
with the core objective — field the heavy-armour establishment
(>=4 2tnk) and clear the eastern e1 garrison defending the enemy
construction yard — rather than panic-rebuilding the lost
non-essentials.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall (only `observe`) LOSES every tier / every hard seed (no
     attack, no kills -> kill bar unmet -> after_ticks 5401 fires ->
     timeout LOSS).
   - no_rebuild (attack east but never replace the opening tank
     casualty) LOSES every tier: the 1tnk strike kills ~1 agent 2tnk;
     the survivors clear the kill bar but `unit_type_count_gte:
     {type: 2tnk, n: 4}` busts because the casualty was never
     replaced -> LOSS even with kills >= 5.
   - rebuild_nonessentials (after the destroy event, build pbox +
     harv to restore the razed slice) LOSES every tier: the
     indivisible reserve is spent on assets NOT in the win predicate,
     so the tank casualty goes unreplaced and the type-count clause
     busts -> LOSS.
   - intended press-on (attack_move east + rebuild a 2tnk only when
     the establishment drops below 4) WINS every tier / every seed.
   - hard tier defines >=2 agent spawn_point groups (NORTH-flank
     scout vs SOUTH-flank scout) so a memorised opening cannot
     generalise; the destroy event fires on every seed.
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

PACK = PACKS_DIR / "rob-partial-base-loss-continue.yaml"


# ── policies ────────────────────────────────────────────────────────


def _tanks(rs):
    return [
        u
        for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]


def _stall(rs, Command):
    """Pure observe — no orders -> no kills -> kill bar unmet ->
    after_ticks 5401 fires -> timeout LOSS."""
    return [Command.observe()]


def _no_rebuild(rs, Command):
    """Attack-move east WITHOUT ever issuing build. The 1tnk strike
    kills ~1 2tnk; the survivors clear the kill bar BUT the
    `unit_type_count_gte: {type: 2tnk, n: 4}` clause is busted by the
    unreplaced casualty -> LOSS even though kills >= 5."""
    ids = [str(u["id"]) for u in _tanks(rs)]
    return [Command.attack_move(ids, 80, 20)] if ids else [Command.observe()]


def _rebuild_nonessentials(rs, Command):
    """Panic-rebuild the razed pbox + harv (the non-critical slice the
    destroy event removed). The indivisible reserve is spent on assets
    NOT in the win predicate, so the tank casualty goes unreplaced and
    the type-count clause busts -> LOSS. This is the canonical failure
    mode the graceful-degradation scenario is designed to catch."""
    cmds = [
        Command.build("pbox"),
        Command.build("harv"),
        Command.place_building("pbox", 12, 30),
        Command.place_building("harv", 10, 30),
    ]
    ids = [str(u["id"]) for u in _tanks(rs)]
    if ids:
        cmds.append(Command.attack_move(ids, 80, 20))
    return cmds


def _intended(rs, Command):
    """Triage the loss: IGNORE the razed non-essentials. If the 2tnk
    establishment drops below 4, commission a replacement via
    build('2tnk'); always attack_move the live 2tnk column toward the
    eastern garrison at (80,20). The reserve covers the opening
    casualty; the garrison falls; type-count + kill + fact-intact bars
    all pass -> WIN."""
    t = _tanks(rs)
    cmds = []
    if len(t) < 4:
        cmds.append(Command.build("2tnk"))
    ids = [str(u["id"]) for u in t]
    if ids:
        cmds.append(Command.attack_move(ids, 80, 20))
    return cmds or [Command.observe()]


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-partial-base-loss-continue"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = pack.meta.benchmark_anchor
    joined = " ".join(anchors).lower()
    assert "graceful degradation" in joined
    assert "incident triage" in joined
    assert "critical-vs-noncritical" in joined or "critical" in joined


def test_uses_turtle_bot():
    """The pack declares the Wave-2 `turtle` bot — the holding-in-place
    idiom keeps the eastern garrison at x=80 and isolates the loss
    event to the scripted `destroy_actors` hook (a roaming enemy would
    confound the triage discrimination)."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "turtle", f"expected turtle bot, got {bot!r}"


def test_every_tier_has_a_destroy_actors_event():
    """The load-bearing graceful-degradation mechanism: every tier
    must declare a `scheduled_events: destroy_actors` hook so a
    non-critical slice of the base is razed mid-episode."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        evs = c.scheduled_events
        assert evs, f"{lvl}: must declare scheduled_events"
        kinds = {e.get("type") for e in evs}
        assert "destroy_actors" in kinds, (
            f"{lvl}: must declare a destroy_actors event; got {kinds}"
        )
        # The destroy event must fire mid-episode (after the agent's
        # first decision turns) so the loss is observed and the triage
        # decision is live.
        for e in evs:
            if e.get("type") == "destroy_actors":
                assert 90 < int(e["tick"]) < 5400, (
                    f"{lvl}: destroy tick {e['tick']} must be mid-episode"
                )
                assert e["filter"]["owner"] == "agent"
                assert "region" in e["filter"], (
                    f"{lvl}: destroy filter must use a region to isolate "
                    "the blast to the south outpost"
                )


def test_destroy_event_razes_only_the_south_outpost():
    """Engine-driven: the destroy event at tick ~1200 removes the
    outer pbox(es) + south harvester but leaves the 5-building core
    (fact + proc + powr + weap + fix) and the north harvester
    intact — the loss must be NON-critical."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        snaps = []

        def _probe(rs, Command, _snaps=snaps):
            bs = rs.get("buildings_summary") or rs.get("own_buildings") or []
            btypes = sorted(str(b.get("type", "")).lower() for b in bs)
            nharv = sum(
                1
                for u in (rs.get("units_summary") or [])
                if str(u.get("type", "")).lower() == "harv"
            )
            _snaps.append((rs.get("game_tick"), btypes, nharv))
            return [Command.observe()]

        run_level(c, _probe, seed=1)
        pre = next(
            (s for s in snaps if isinstance(s[0], int) and s[0] < 1100),
            snaps[0],
        )
        post = next(
            (s for s in snaps if isinstance(s[0], int) and s[0] > 1300),
            snaps[-1],
        )
        core = {"fact", "proc", "powr", "weap", "fix"}
        post_set = set(post[1])
        assert core <= post_set, (
            f"{lvl}: core base must survive the destroy event; "
            f"post-destroy buildings = {post[1]}"
        )
        assert "pbox" not in post_set, (
            f"{lvl}: the outer pbox(es) must be razed; post = {post[1]}"
        )
        assert post[2] < pre[2], (
            f"{lvl}: one harvester must be razed (pre {pre[2]} -> "
            f"post {post[2]})"
        )
        assert post[2] >= 1, (
            f"{lvl}: the surviving harvester must remain (post {post[2]})"
        )


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks <= ceiling AND
    after_ticks <= ceiling AND within_ticks + 1 == after_ticks (so a
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
            "(non-finisher must LOSE, not draw)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: >=2 distinct agent spawn_point groups so the engine
    round-robins start by seed. The core base + 2tnk column + south
    outpost are SHARED across both groups at identical cells so the
    strike and destroy geometry is symmetric, but the
    spawn-distinguishing scout (NORTH (18,12) vs SOUTH (18,32))
    reveals which seed the engine picked."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define >=2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_tools_match_spec():
    """The advertised toolset is exactly the press-on kit: observe +
    build + place_building + move_units + attack_unit + attack_move +
    stop."""
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
    """Win requires (>=4 2tnk AND >=5 kills AND fact alive) AND
    in-time; fail fires on timeout OR all-units-dead OR fact
    destroyed."""
    c = compile_level(load_pack(PACK), "medium")
    four_tanks = [
        {"cell_x": 22, "cell_y": 18 + 2 * i, "type": "2tnk"} for i in range(4)
    ]
    fact = [("fact", 8, 18)]

    # Intended: 4 2tnk, 5 kills, fact alive, in time -> WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=four_tanks, tick=2000, kills=5, own_buildings=fact),
    )
    # Only 3 2tnk (didn't replace the casualty) -> not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=four_tanks[:3], tick=2000, kills=5, own_buildings=fact),
    )
    # Only 4 kills (didn't clear the garrison) -> not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=four_tanks, tick=2000, kills=4, own_buildings=fact),
    )
    # All units dead -> real fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, kills=5, own_buildings=fact),
    )
    # Timeout -> fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=four_tanks, tick=5402, kills=0, own_buildings=fact),
    )
    # Construction yard destroyed -> fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=four_tanks, tick=2000, kills=5, own_buildings=[]),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No orders -> no kills -> kill bar unmet -> timeout LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"kills={r.signals.units_killed} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_no_rebuild_loses(level, seed):
    """Assault-without-rebuild: the 1tnk strike kills ~1 2tnk; the
    survivors clear the garrison so kills >= 5, but the type-count
    clause busts because the casualty was never replaced -> LOSS."""
    _, r = _run(level, _no_rebuild, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: no-rebuild must LOSE (type-count busted "
        f"by unreplaced casualty); got {r.outcome} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_rebuild_nonessentials_loses(level, seed):
    """Panic-rebuild the razed pbox + harv: the indivisible reserve is
    spent on assets NOT in the win predicate, so the tank casualty
    goes unreplaced and the type-count clause busts -> LOSS. This is
    the graceful-degradation anti-pattern the scenario is designed to
    catch — rebuilding the non-essentials wastes the budget and the
    clock."""
    _, r = _run(level, _rebuild_nonessentials, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: rebuild-the-non-essentials must LOSE "
        f"(budget wasted on assets off the win path); got {r.outcome} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_press_on_wins(level, seed):
    """The intended capability — triage the non-critical loss, press
    on with the core objective (attack_move east + replace only the
    tank casualties) — WINS every tier and every hard seed well
    inside the tick budget."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended press-on should WIN; got "
        f"{r.outcome} kills={r.signals.units_killed} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy -> identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
