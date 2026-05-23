"""econ-contested-expansion — REASONING capability validation.

Mid-game expansion under genuine geographic contention. The pack ships
a custom 128×40 arena (`econ-contested-corridor-128x40`) with two
water walls that frame a horizontal corridor (y=15..24) at x=46..81.
A rich contested ore patch sits in the middle of the corridor at
(66, 20), guarded by patrol e3 (rocket infantry). The agent's
mini-base sits in the WEST lobe with a small home patch + two
starter Light Tanks.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / seed (no commit → home-patch single-
     harv income ceiling stays below the cash bar → timeout LOSS).
   - home-proc LOSES (2nd refinery placed at home → auto-harv joins
     the already-dry home patch → same income ceiling as stall).
   - patch-no-escort LOSES (2nd refinery placed at the contested
     patch WITHOUT moving the starter tanks → the patrol e3
     destroys the auto-spawned harv in a few turns → cash bar
     unmet).
   - intended WINS (escort the starter tanks to the corridor FIRST
     to clear the patrol, then build + place the 2nd refinery at
     the patch → auto-harv mines the rich patch and deposits at
     the new closer proc → bar cleared comfortably before tick
     4000 on every tier and every hard seed).
   - hard tier defines ≥2 agent spawn_point groups (NW y=8 /
     SW y=32 base orientation) so a memorised opening cannot
     generalise (the corridor / patch / patrol stay fixed; only
     the agent's starting latitude varies).

Anchors: SC2 contested expansion (take a 2nd resource under harass),
facility-siting under adversarial contention, convoy security
(forward asset commission + escort), RTS second-refinery idiom
(free-worker on-build, at the new proc).
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

PACK = PACKS_DIR / "econ-contested-expansion.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No action — single home harv vs the home patch only."""
    return [Command.observe()]


def _make_home_proc():
    """Build + place 2nd proc AT HOME (next to the existing proc).
    The auto-harv lands on the home patch; both harvs compete for
    the same already-dry patch → no net throughput gain."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 17:
            # Place near the home harv's current latitude (works for
            # both hard spawn_point groups: harv is at y=8 (NW) or
            # y=32 (SW)).
            units = rs.get("units_summary", []) or []
            harv = next((u for u in units if u.get("type") == "harv"), None)
            y = int(harv.get("cell_y", 30)) if harv else 30
            s["placed"] = True
            return [Command.place_building("proc", 20, y)]
        return [Command.observe()]

    return policy


def _make_patch_no_escort():
    """Build + place 2nd proc AT THE CONTESTED PATCH but DON'T move
    the starter tanks. The auto-harv lands on the patch in range of
    the e3 patrol; the patrol shreds it in a few ticks → 2nd proc
    generates ~0 income → cash bar unmet."""
    s = {"queued": False, "placed": False, "turn": 0}

    def policy(rs, Command):
        s["turn"] += 1
        if not s["queued"]:
            s["queued"] = True
            return [Command.build("proc")]
        if not s["placed"] and s["turn"] >= 17:
            s["placed"] = True
            return [Command.place_building("proc", 66, 20)]
        return [Command.observe()]

    return policy


def _make_intended():
    """ESCORT first: `attack_move` the two Light Tanks to the
    contested patch (66, 20) so the patrol is engaged before the
    auto-spawned harv arrives. Then build + place the 2nd proc at
    the patch. The auto-harv mines the rich patch and deposits at
    the new (closer) refinery → bar clears."""
    s = {"queued": False, "placed": False, "turn": 0, "escorted": False}

    def policy(rs, Command):
        s["turn"] += 1
        units = rs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "1tnk"]
        cmds = []
        if not s["escorted"] and tanks:
            ids = [str(t["id"]) for t in tanks]
            cmds.append(Command.attack_move(ids, 66, 20))
            s["escorted"] = True
        if not s["queued"]:
            cmds.append(Command.build("proc"))
            s["queued"] = True
            return cmds
        if not s["placed"] and s["turn"] >= 17:
            cmds.append(Command.place_building("proc", 66, 20))
            s["placed"] = True
        return cmds or [Command.observe()]

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "corridor map must be Rust-loadable"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-contested-expansion"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors
    assert "contested" in anchors or "contention" in anchors


def test_uses_corridor_map_generator():
    """The pack must declare its custom corridor map (the geographic
    contention substrate), not the generic shared rush-hour arena.
    The two water walls are what force the single-corridor approach
    that makes the patrol's denial position load-bearing."""
    pack = load_pack(PACK)
    bm = pack.base_map
    assert isinstance(bm, dict), (
        f"base_map must be a generator spec dict, got {type(bm).__name__}: "
        "the contested-expansion task needs the corridor obstacles, "
        "not the generic rush-hour-arena."
    )
    assert bm.get("generator") == "arena"
    assert bm.get("name") == "econ-contested-corridor-128x40"
    obs = bm.get("obstacles") or []
    assert len(obs) >= 2, "must paint the corridor walls as obstacles"


def test_uses_patrol_bot():
    """`patrol` bot keeps the e3 patrol oscillating around the
    contested patch (PATROL_RADIUS=8) — the corridor stays denied
    by enemies that don't leave to hunt elsewhere."""
    pack = load_pack(PACK)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "patrol", f"expected patrol bot, got {bot!r}"


def test_all_tiers_have_reachable_deadlines():
    """Tick-alignment idiom: within_ticks ≤ ceiling AND after_ticks
    ≤ ceiling AND within_ticks + 1 == after_ticks (non-finisher
    LOSES, not draws). Ceiling = 93 + 90·(max_turns-1)."""
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


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 distinct agent spawn_point groups so the engine
    round-robins start by seed. The corridor / patch / patrol are
    fixed; only the agent's base latitude (NW y=8 vs SW y=32)
    varies."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else None)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    sp.discard(None)
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_persistent_enemy_sentinel_anti_draw():
    """An unarmed enemy `fact` sentinel keeps
    `ConquestVictoryConditions` from auto-`done`-ing on
    enemy-elimination before the win/fail evaluates. It is pinned
    at stance:0 (HoldFire) so it never auto-engages and never tries
    to chase. On hard the sentinel is duplicated per spawn_point
    group."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        sentinels = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert sentinels, f"{lvl}: missing enemy `fact` sentinel (anti-DRAW)"
        for s in sentinels:
            assert s.stance == 0, (
                f"{lvl}: sentinel must be stance:0 (HoldFire), got {s.stance}"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, building_types=(), tick=1000, cash=0):
    """Build a WinContext with N agent buildings of the given types.
    `signals.own_buildings` is a list of `(type, x, y)` tuples (per
    win_conditions.py); `signals.own_building_types` is the distinct
    set of types."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=0,
        own_buildings=[(t, 0, 0) for t in building_types],
        own_building_types=set(building_types),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_capability():
    """Win requires (has_building:proc AND ≥2 procs AND cash bar) AND
    in time; fail fires on timeout OR fact gone."""
    c = compile_level(load_pack(PACK), "easy")

    # All clauses met → WIN
    assert evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr", "proc", "proc"), tick=2000, cash=5800),
    )
    # Only 1 proc → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr", "proc"), tick=2000, cash=99999),
    )
    # 2 procs but cash below bar → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(building_types=("fact", "powr", "proc", "proc"), tick=2000, cash=5799),
    )
    # Timeout: bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(building_types=("fact", "powr", "proc"), tick=4002, cash=0),
    )
    # Fact lost → fail
    assert evaluate(
        c.fail_condition,
        _ctx(building_types=("powr", "proc"), tick=2000, cash=99999),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No build/place order → home patch single-harv income ceiling
    stays below the cash bar → after_ticks LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_home_proc_loses(level, seed):
    """2nd proc placed at home → auto-harv lands on the already-dry
    home patch → no net throughput gain → same ceiling as stall →
    after_ticks LOSS."""
    _, r = _run(level, _make_home_proc(), seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: home-proc must LOSE; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_patch_no_escort_loses(level, seed):
    """2nd proc placed at the contested patch WITHOUT moving the
    tanks → the patrol e3 destroys the auto-spawned harv in a few
    ticks → 2nd proc generates ~0 income → cash bar unmet → LOSS."""
    _, r = _run(level, _make_patch_no_escort(), seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: patch-no-escort must LOSE; got "
        f"{r.outcome} cash={r.signals.cash} lost={r.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_escort_plus_2nd_proc_wins(level, seed):
    """The intended capability — escort tanks into the corridor +
    place the 2nd proc on the contested patch — WINS every tier and
    every hard seed comfortably inside the tick budget."""
    _, r = _run(level, _make_intended(), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended escort+2nd-proc should WIN; got "
        f"{r.outcome} cash={r.signals.cash} turns={r.turns} "
        f"lost={r.signals.units_lost}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome and cash."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_intended(), seed=2)
    b = run_level(c, _make_intended(), seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash,
    )
