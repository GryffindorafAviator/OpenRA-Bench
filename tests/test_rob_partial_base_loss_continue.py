"""rob-partial-base-loss-continue — REASONING capability validation
(two-base strategic-retreat template, task #81).

Continue operating after partial base loss. The agent starts with TWO
bases: a FORWARD outpost (doomed: a series of scheduled `destroy_actors`
pulses razes everything agent-owned inside the forward region every
~900-1200 ticks) and a HOME base (safe: the destroy region never
reaches it). HOME is seeded with fact + powr only — the agent must
RECONSTITUTE the production chain (proc + weap) at HOME to continue
operating.

The win predicate is STATE-BASED at the deadline:
  - building_in_region:{HOME, type: proc, count: 1}  AND
  - building_in_region:{HOME, type: weap, count: 1}  AND
  - has_building: fact                               AND
  - within_ticks: 5400

Both `preemptive redundancy` (build at HOME BEFORE the first pulse,
while the forward base is still alive) and `reactive rebuild` (build
at HOME AFTER the first pulse) satisfy the predicate equally.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall (only `observe`) LOSES every tier/seed: the FORWARD proc +
     weap are razed by the first pulse, the HOME base never had them,
     both region clauses never latch -> timeout LOSS.
   - build-at-forward (rebuild proc + weap inside the FORWARD region
     after every pulse) LOSES every tier/seed: the next destroy pulse
     razes the replacements; the agent burns cash and decision turns
     on assets that don't survive -> region clauses still busted ->
     timeout LOSS. This is the canonical strategic-retreat failure
     the scenario is designed to catch.
   - build-at-home (queue proc + weap and place each inside the HOME
     region after the first pulse) WINS every tier/seed.
   - do-both (preemptive redundancy: build proc + weap at HOME BEFORE
     the first pulse) WINS every tier/seed — same WIN as the reactive
     path.
   - hard tier defines >=2 agent spawn_point groups (NORTH-flank scout
     vs SOUTH-flank scout) so a memorised opening cannot generalise;
     the destroy pulses fire on every seed.
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

# Region centres / radius shared across all tiers.
HOME_X, HOME_Y, HOME_R = 100, 18, 10
FORWARD_X, FORWARD_Y, FORWARD_R = 20, 18, 10


# ── policies ────────────────────────────────────────────────────────


def _own_buildings(rs):
    return rs.get("own_buildings") or rs.get("buildings_summary") or []


def _has_building_in_region(rs, btype, cx, cy, r):
    for b in _own_buildings(rs):
        if str(b.get("type", "")).lower() != btype:
            continue
        x = int(b.get("cell_x", b.get("x", 0)))
        y = int(b.get("cell_y", b.get("y", 0)))
        if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
            return True
    return False


def _stall(rs, Command):
    """Pure observe — no proc or weap is ever built. The FORWARD pair
    is razed by the first destroy pulse; both region clauses stay
    busted -> timeout LOSS."""
    return [Command.observe()]


def _make_build_at_forward():
    """Rebuild proc + weap inside the FORWARD region after every pulse.
    The next destroy pulse razes the replacements; the agent never
    satisfies the HOME region clauses -> timeout LOSS. This is the
    strategic-retreat failure the scenario is built to catch — fighting
    the attacker at the doomed site instead of retreating to safety."""
    s = {"queued_proc": False, "queued_weap": False, "placed_proc": False,
         "placed_weap": False, "last_proc_seen": False,
         "last_weap_seen": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        cmds = []
        has_proc_fwd = _has_building_in_region(
            rs, "proc", FORWARD_X, FORWARD_Y, FORWARD_R
        )
        has_weap_fwd = _has_building_in_region(
            rs, "weap", FORWARD_X, FORWARD_Y, FORWARD_R
        )
        # Re-trigger queue + place every time the forward copy dies
        # (was previously alive but now isn't).
        if s["last_proc_seen"] and not has_proc_fwd:
            s["queued_proc"] = False
            s["placed_proc"] = False
        if s["last_weap_seen"] and not has_weap_fwd:
            s["queued_weap"] = False
            s["placed_weap"] = False
        s["last_proc_seen"] = has_proc_fwd
        s["last_weap_seen"] = has_weap_fwd

        if not has_proc_fwd and not s["queued_proc"]:
            cmds.append(Command.build("proc"))
            s["queued_proc"] = True
        if not has_weap_fwd and not s["queued_weap"]:
            cmds.append(Command.build("weap"))
            s["queued_weap"] = True
        # Place a few turns after queueing — the shared Building queue
        # serialises items so placement attempts before the item is
        # ready are no-ops, harmless.
        if s["queued_proc"] and not s["placed_proc"]:
            cmds.append(Command.place_building("proc", 18, 20))
            s["placed_proc"] = True
        if s["queued_weap"] and not s["placed_weap"] and s["turn"] > 12:
            cmds.append(Command.place_building("weap", 22, 20))
            s["placed_weap"] = True
        return cmds or [Command.observe()]

    return policy


def _make_build_at_home():
    """Reactive rebuild at HOME. Queue proc + weap and place each
    inside the HOME region. Pulses never reach HOME -> both region
    clauses latch -> WIN."""
    s = {"queued_proc": False, "queued_weap": False,
         "placed_proc": False, "placed_weap": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        cmds = []
        if not s["queued_proc"]:
            cmds.append(Command.build("proc"))
            s["queued_proc"] = True
        # Place proc when it's likely ready (proc cost 1400 -> 840
        # ticks ~= turn 10).
        if s["queued_proc"] and not s["placed_proc"] and s["turn"] >= 11:
            cmds.append(Command.place_building("proc", HOME_X, HOME_Y - 2))
            s["placed_proc"] = True
        # Queue weap after proc lands so the shared Building queue
        # serialises them deterministically.
        if s["placed_proc"] and not s["queued_weap"]:
            cmds.append(Command.build("weap"))
            s["queued_weap"] = True
        # weap cost 2000 -> 1200 ticks ~= 14 turns after queue.
        if s["queued_weap"] and not s["placed_weap"] and s["turn"] >= 26:
            cmds.append(Command.place_building("weap", HOME_X + 2, HOME_Y))
            s["placed_weap"] = True
        return cmds or [Command.observe()]

    return policy


def _make_do_both():
    """Preemptive redundancy: queue + place proc and weap at HOME
    BEFORE the first destroy pulse (tick 600 = ~turn 7) so the
    redundancy is already in place when the forward outpost falls.
    Same WIN as the reactive path — the predicate credits both."""
    s = {"queued_proc": False, "queued_weap": False,
         "placed_proc": False, "placed_weap": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        cmds = []
        # Queue both on turn 1 — the shared Building queue serialises
        # them but both are committed from t=0.
        if not s["queued_proc"]:
            cmds.append(Command.build("proc"))
            s["queued_proc"] = True
        if s["queued_proc"] and not s["queued_weap"]:
            cmds.append(Command.build("weap"))
            s["queued_weap"] = True
        if s["queued_proc"] and not s["placed_proc"] and s["turn"] >= 11:
            cmds.append(Command.place_building("proc", HOME_X, HOME_Y - 2))
            s["placed_proc"] = True
        if s["placed_proc"] and not s["placed_weap"] and s["turn"] >= 25:
            cmds.append(Command.place_building("weap", HOME_X + 2, HOME_Y))
            s["placed_weap"] = True
        return cmds or [Command.observe()]

    return policy


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
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "graceful degradation" in anchors
    assert "two-base" in anchors or "strategic retreat" in anchors


def test_uses_turtle_bot():
    """The pack declares the Wave-2 `turtle` bot — there must be no
    roaming offence that could confound the destroy-pulse
    discrimination. The only threat is the scripted pulses."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "turtle", f"expected turtle bot, got {bot!r}"


def test_every_tier_has_repeated_destroy_pulses():
    """The load-bearing strategic-retreat mechanism: every tier must
    declare a SERIES of destroy_actors pulses (>=2) targeting the
    forward region. A single pulse would let a forward-rebuild policy
    win after it lands; the series is what makes the doomed site
    genuinely doomed."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        evs = c.scheduled_events or []
        assert evs, f"{lvl}: must declare scheduled_events"
        destroy = [e for e in evs if e.get("type") == "destroy_actors"]
        assert len(destroy) >= 2, (
            f"{lvl}: must declare >=2 destroy_actors pulses (the doomed-"
            f"forward template requires rebuilds also die); got "
            f"{len(destroy)}"
        )
        for e in destroy:
            assert 90 < int(e["tick"]) < 5400, (
                f"{lvl}: destroy tick {e['tick']} must be mid-episode"
            )
            f = e["filter"]
            assert f["owner"] == "agent"
            region = f["region"]
            assert region["x"] == FORWARD_X and region["y"] == FORWARD_Y, (
                f"{lvl}: destroy region must be centred on FORWARD "
                f"(20,18); got ({region['x']},{region['y']})"
            )


def test_home_region_is_safe_from_destroy_pulses():
    """The HOME region (centred on (100,18), radius 10) must lie
    entirely outside every destroy radius — otherwise the
    strategic-retreat path is unsafe and the template breaks. Distance
    between centres ~80, with both radii 10, gives ~60 cells of clear
    separation per pulse."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        for e in (c.scheduled_events or []):
            if e.get("type") != "destroy_actors":
                continue
            r = e["filter"]["region"]
            dx = abs(r["x"] - HOME_X)
            dy = abs(r["y"] - HOME_Y)
            # Conservative gap: pulse radius + HOME radius < centre dist
            gap = (dx * dx + dy * dy) ** 0.5
            need = int(r["radius"]) + HOME_R
            assert gap > need, (
                f"{lvl}: destroy region ({r['x']},{r['y']},r={r['radius']}) "
                f"can clip HOME (r={HOME_R} around ({HOME_X},{HOME_Y})); "
                f"distance {gap:.1f} <= radii sum {need}"
            )


def test_win_predicate_is_state_based_at_home():
    """The win predicate must require BOTH a proc AND a weap inside
    the HOME region (the strategic-retreat template — state-based at
    deadline). A predicate that only required one of them would
    trivialise the build budget."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        clauses = pack.levels[lvl].win_condition.model_dump()["all_of"]
        regions = [c.get("building_in_region") for c in clauses
                   if "building_in_region" in c]
        types = {(r.get("type"),) for r in regions
                 if int(r.get("x", 0)) == HOME_X
                 and int(r.get("y", 0)) == HOME_Y}
        assert ("proc",) in types, (
            f"{lvl}: win must require a proc inside the HOME region; "
            f"got {types}"
        )
        assert ("weap",) in types, (
            f"{lvl}: win must require a weap inside the HOME region; "
            f"got {types}"
        )


def test_home_base_seeds_lack_proc_and_weap():
    """HOME must be seeded with fact + powr ONLY — no proc and no weap.
    Otherwise stall would WIN trivially (the home proc/weap satisfy
    the predicate without any build verb)."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        # Per-tier check uses the first spawn group (hard tier
        # duplicates the home base across both groups, identical
        # cells).
        spawns_seen = set()
        for a in c.scenario.actors:
            if a.owner != "agent":
                continue
            sp = a.spawn_point if a.spawn_point is not None else 0
            if sp not in spawns_seen:
                spawns_seen.add(sp)
            if sp != next(iter(spawns_seen)):
                continue
            # Is this actor inside the HOME region?
            x, y = int(a.position[0]), int(a.position[1])
            if (x - HOME_X) ** 2 + (y - HOME_Y) ** 2 <= HOME_R ** 2:
                assert a.type.lower() not in ("proc", "weap"), (
                    f"{lvl}: HOME region must NOT be pre-seeded with "
                    f"{a.type} (stall would WIN trivially); "
                    f"found at ({x},{y})"
                )


def test_forward_base_is_seeded_with_full_chain():
    """FORWARD outpost must start with proc + weap + fact + powr so
    a stall policy briefly has them — the partial-loss event is
    OBSERVED in-game (pre-pulse the chain exists; post-pulse it
    doesn't). Without this, the 'continue after partial loss' framing
    collapses to a flat economy task."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        spawns_seen = set()
        fwd_types = set()
        for a in c.scenario.actors:
            if a.owner != "agent":
                continue
            sp = a.spawn_point if a.spawn_point is not None else 0
            if sp not in spawns_seen:
                spawns_seen.add(sp)
            if sp != next(iter(spawns_seen)):
                continue
            x, y = int(a.position[0]), int(a.position[1])
            if (x - FORWARD_X) ** 2 + (y - FORWARD_Y) ** 2 <= FORWARD_R ** 2:
                fwd_types.add(a.type.lower())
        for required in ("fact", "proc", "powr", "weap"):
            assert required in fwd_types, (
                f"{lvl}: FORWARD must be seeded with {required}; got "
                f"{sorted(fwd_types)}"
            )


def test_destroy_event_razes_forward_outpost():
    """Engine-driven: after the first pulse fires the FORWARD proc +
    weap are gone, while the HOME fact + powr are intact."""
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        snaps = []

        def _probe(rs, Command, _snaps=snaps):
            bs = _own_buildings(rs)
            tick = rs.get("game_tick")
            fwd_proc = _has_building_in_region(
                rs, "proc", FORWARD_X, FORWARD_Y, FORWARD_R
            )
            fwd_weap = _has_building_in_region(
                rs, "weap", FORWARD_X, FORWARD_Y, FORWARD_R
            )
            home_fact = _has_building_in_region(
                rs, "fact", HOME_X, HOME_Y, HOME_R
            )
            home_powr = _has_building_in_region(
                rs, "powr", HOME_X, HOME_Y, HOME_R
            )
            _snaps.append((tick, fwd_proc, fwd_weap, home_fact, home_powr))
            return [Command.observe()]

        run_level(c, _probe, seed=1)
        pre = next((s for s in snaps if isinstance(s[0], int) and s[0] < 500),
                   snaps[0])
        post = next((s for s in snaps if isinstance(s[0], int) and s[0] > 800),
                    snaps[-1])
        assert pre[1] and pre[2], (
            f"{lvl}: FORWARD proc + weap must be alive PRE-pulse; "
            f"snap={pre}"
        )
        assert not post[1] and not post[2], (
            f"{lvl}: FORWARD proc + weap must be RAZED post-pulse; "
            f"snap={post}"
        )
        assert post[3] and post[4], (
            f"{lvl}: HOME fact + powr must SURVIVE; snap={post}"
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
    round-robins start by seed. The FORWARD + HOME bases are SHARED
    across both groups at identical cells so the destroy geometry is
    symmetric, but the spawn-distinguishing scout (NORTH (50,10) vs
    SOUTH (50,30)) reveals which seed the engine picked."""
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
    """The advertised toolset is the build + control kit: observe +
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


def _ctx(*, tick=1000, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=0,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={},
    )


def test_predicates_enforce_two_base_template():
    """Win requires proc-in-HOME AND weap-in-HOME AND has_building:fact
    AND within_ticks; fail fires on timeout OR no fact."""
    c = compile_level(load_pack(PACK), "medium")
    # HOME-region proc + weap + fact alive, in time -> WIN.
    home_full = [
        ("proc", HOME_X, HOME_Y - 2),
        ("weap", HOME_X + 2, HOME_Y),
        ("fact", HOME_X, HOME_Y),
    ]
    assert evaluate(c.win_condition, _ctx(tick=4000, own_buildings=home_full))
    # FORWARD-region proc + weap (NOT home) + fact -> not a win.
    fwd_only = [
        ("proc", FORWARD_X, FORWARD_Y),
        ("weap", FORWARD_X + 2, FORWARD_Y),
        ("fact", HOME_X, HOME_Y),
    ]
    assert not evaluate(c.win_condition, _ctx(tick=4000, own_buildings=fwd_only))
    # Only proc-in-HOME, no weap -> not a win.
    half = [
        ("proc", HOME_X, HOME_Y - 2),
        ("fact", HOME_X, HOME_Y),
    ]
    assert not evaluate(c.win_condition, _ctx(tick=4000, own_buildings=half))
    # No fact -> fail.
    assert evaluate(c.fail_condition, _ctx(tick=4000, own_buildings=[]))
    # Timeout -> fail.
    assert evaluate(
        c.fail_condition, _ctx(tick=5402, own_buildings=home_full)
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No build orders -> proc + weap razed at FORWARD by the first
    pulse, never built at HOME -> region clauses busted -> timeout
    LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_build_at_forward_loses(level, seed):
    """Rebuild proc + weap inside the FORWARD region after every
    pulse: the next pulse razes the replacements; region clauses at
    HOME never latch -> timeout LOSS. This is the canonical
    strategic-retreat failure the scenario is designed to catch."""
    _, r = _run(level, _make_build_at_forward(), seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: build-at-forward must LOSE; got "
        f"{r.outcome} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_build_at_home_wins(level, seed):
    """Reactive rebuild at HOME — queue proc + weap and place each
    inside the HOME region. Pulses never reach HOME, both region
    clauses latch -> WIN."""
    _, r = _run(level, _make_build_at_home(), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: build-at-home must WIN; got "
        f"{r.outcome} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_do_both_wins(level, seed):
    """Preemptive redundancy — build proc + weap at HOME BEFORE the
    first pulse. The HOME copies survive every pulse; same WIN as
    the reactive path (the predicate credits both)."""
    _, r = _run(level, _make_do_both(), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: do-both (preemptive) must WIN; got "
        f"{r.outcome} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy -> identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    pol = _make_build_at_home()
    a = run_level(c, pol, seed=2)
    pol = _make_build_at_home()
    b = run_level(c, pol, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
