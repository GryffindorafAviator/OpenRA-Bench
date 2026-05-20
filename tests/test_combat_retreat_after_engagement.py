"""combat-retreat-after-engagement — disengage to preserve the force.

Bar (four script-policy proxies):
  • stall (observe only)                    → LOSS (kill bar unmet)
  • brute attack-until-death                → LOSS (loses too many tanks)
  • never-engage (sit at safe zone)         → LOSS (kill bar unmet)
  • intended engage-then-retreat            → WIN

The "intended" policy is the spec's load-bearing decision: march east
into engagement range, focus-fire e3 rocket infantry (the cheap kill-
bar targets), and the instant the kill bar is met OR a tank is lost
pull back to the safe-zone radius. The retreat trigger is the
capability under test — too early ⇒ kill bar fails; too late ⇒
attrition busts the survival bar.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-retreat-after-engagement.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(units_xy=(), tick=1000, killed=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # 3 tanks back in safe zone (5,20,r=6), killed 2 enemies, 1 lost, in time → WIN
    home3 = [(5, 18), (5, 20), (5, 21)]
    assert evaluate(c.win_condition, _ctx(home3, tick=3000, killed=2, lost=1))
    # Kill bar unmet (only 1 killed) → not WIN
    assert not evaluate(c.win_condition, _ctx(home3, tick=3000, killed=1, lost=1))
    # Tanks elsewhere (not at safe zone) → not WIN
    assert not evaluate(
        c.win_condition, _ctx([(55, 20), (55, 21), (55, 22)], tick=3000, killed=3, lost=1)
    )
    # 3 tanks lost (only 1 alive) → fail clause own_units_gte:2 fires
    assert evaluate(c.fail_condition, _ctx([(5, 20)], tick=3000, killed=3, lost=3))
    # Past deadline → real LOSS reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(home3, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "easy after_ticks 4501 must be reachable within max_turns"
    )


def test_predicates_medium_force_preservation_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    home3 = [(5, 18), (5, 20), (5, 21)]
    home2 = home3[:2]
    # Intended: 3 kills, ≥3 tanks at safe zone, ≤1 lost → WIN
    assert evaluate(c.win_condition, _ctx(home3, tick=3000, killed=3, lost=1))
    # Same kills but only 2 tanks left → predicate fails (need ≥3)
    assert not evaluate(c.win_condition, _ctx(home2, tick=3000, killed=3, lost=2))
    # 3 tanks alive but only 2 in safe zone → fails
    assert not evaluate(
        c.win_condition,
        _ctx([(5, 20), (5, 21), (55, 20)], tick=3000, killed=3, lost=1),
    )
    # 2 tanks alive ⇒ fail clause fires (preservation cap)
    assert evaluate(c.fail_condition, _ctx(home2, tick=3000, killed=3, lost=2))
    # Past deadline ⇒ real LOSS reachable
    assert evaluate(c.fail_condition, _ctx(home3, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_two_safe_zones():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # NORTH safe zone (5,10) satisfies the any_of geometry
    home_north = [(5, 9), (5, 10), (5, 11)]
    assert evaluate(c.win_condition, _ctx(home_north, tick=3000, killed=3, lost=1))
    # SOUTH safe zone (5,30) also satisfies the any_of geometry
    home_south = [(5, 29), (5, 30), (5, 31)]
    assert evaluate(c.win_condition, _ctx(home_south, tick=3000, killed=3, lost=1))
    # Tanks at the WRONG centre (5,20) — outside BOTH safe zones at r=6
    # ((5,20)-(5,10)=10>6 and (5,20)-(5,30)=10>6) → fails the geometry
    assert not evaluate(
        c.win_condition,
        _ctx([(5, 20), (5, 19), (5, 21)], tick=3000, killed=3, lost=1),
    )
    # Past tighter deadline → real LOSS reachable
    assert evaluate(c.fail_condition, _ctx(home_north, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 agent spawn_point groups so
    the seed flips the safe-zone corridor."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "combat-retreat-after-engagement"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the doctrines the brief calls out: SC2 retreat
    # micro + military tactical withdrawal + preservation of force.
    assert "retreat" in joined
    assert "withdrawal" in joined or "preservation" in joined or "pullback" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 4501 not reachable within max_turns"
        )


# ── engine-driven scripted policies ──────────────────────────────────
#
# The four-policy bar. All engine-driven tests guard on the Rust env
# wheel; predicate-level tests above run without it.


def _stall_policy(rs, Command):
    """Stall: only observe. Kill bar never met → after_ticks LOSS."""
    return [Command.observe()]


def _brute_attack_until_death_policy(rs, Command):
    """Brute: attack_move toward the enemy centre and never retreat.
    The 2× 3tnk + 5× e3 mass alpha out-trades 4× 2tnk in melee; the
    column dies before clearing the squad → own_units_gte:N fails."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move([str(u["id"])], target_x=60, target_y=20)
        for u in units
    ]


def _never_engage_policy(rs, Command):
    """Never-engage: each turn re-issue a move-in-place at the spawn.
    The agent has ≥3 tanks in the safe zone the entire episode, but
    units_killed_gte:N is never met → after_ticks LOSS. This is the
    "preserve everything, score nothing" trap the kill bar punishes."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.move_units(
            [str(u["id"])], target_x=u["cell_x"], target_y=u["cell_y"]
        )
        for u in units
    ]


def _make_intended_engage_then_retreat():
    """Intended policy (the spec's load-bearing decision): march to
    the engagement axis, focus-fire e3 rocket infantry, and the
    instant a tank is lost OR ≥3 enemies are observed killed pull
    back to the safe-zone radius (detected from the agent's spawn
    median-y latched on first observation). Stateful — uses a
    closure to track the peak number of visible killables (so we
    can infer kills from the shrink without reading
    signals.units_killed)."""
    state = {"peak_visible": 0, "retreat_latched": False, "home_y": None}

    def pol(rs, Command):
        units = rs.get("units_summary", []) or []
        enemies = rs.get("enemy_summary", []) or []
        if not units:
            return [Command.observe()]
        # Latch the home Y on first observation. The agent's spawn
        # cell median y resolves to one of the three safe-zone
        # corridors (north y=10, centre y=20, south y=30).
        if state["home_y"] is None:
            ys = sorted(u["cell_y"] for u in units)
            hy_med = ys[len(ys) // 2]
            if hy_med < 15:
                state["home_y"] = 10
            elif hy_med > 25:
                state["home_y"] = 30
            else:
                state["home_y"] = 20
        hy = state["home_y"]
        n_alive = len(units)
        killable = [
            e
            for e in enemies
            if not e.get("is_building")
            and (e.get("type") or "").lower() != "fact"
        ]
        visible = len(killable)
        if visible > state["peak_visible"]:
            state["peak_visible"] = visible
        killed_observed = state["peak_visible"] - visible
        # RETREAT TRIGGER: latched, or any tank lost, or ≥3 enemies
        # observed killed. Once retreating, stay retreating (a re-
        # engagement would re-expose the survivors to attrition).
        if state["retreat_latched"] or n_alive < 4 or killed_observed >= 3:
            state["retreat_latched"] = True
            return [
                Command.move_units([str(u["id"])], target_x=5, target_y=hy)
                for u in units
            ]
        # ENGAGE: pick the closest e3 to home and focus-fire it with
        # ALL tanks (4-vs-1 ends a Dragon-soldier in 1-2 decision turns).
        e3s = [e for e in killable if (e.get("type") or "").lower() == "e3"]
        if e3s:
            e3s.sort(
                key=lambda e: (e["cell_x"] - 5) ** 2 + (e["cell_y"] - hy) ** 2
            )
            t = e3s[0]
            return [
                Command.attack_unit([str(u["id"])], str(t["id"])) for u in units
            ]
        # APPROACH: advance toward the engagement axis (50, 20) so
        # the spawn corridor (y=10 or y=30 on hard) closes onto the
        # mid-latitude line where the e3s will come into view.
        return [
            Command.move_units(
                [str(u["id"])],
                target_x=min(50, u["cell_x"] + 12),
                target_y=20,
            )
            for u in units
        ]

    return pol


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level — kill bar unmet → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome} "
            f"k={res.signals.units_killed} l={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_until_death_loses(level):
    """Brute attack-until-death must LOSE — the mass alpha at the
    enemy line out-trades the column before the bar is met."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_attack_until_death_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute must LOSE; got {res.outcome} "
            f"k={res.signals.units_killed} l={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_never_engage_policy_loses(level):
    """Never-engage must LOSE — ≥3 tanks survive in the safe zone,
    but units_killed_gte:N is never met → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _never_engage_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: never-engage must LOSE; got {res.outcome} "
            f"k={res.signals.units_killed} l={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_engage_then_retreat_wins(level):
    """Intended engage-then-retreat must WIN on every level and every
    hard seed (1..4): focus-fire e3s, retreat the instant a tank is
    lost or ≥3 kills observed, end with ≥3 tanks in the safe zone."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        pol = _make_intended_engage_then_retreat()
        res = run_level(c, pol, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended engage-then-retreat must WIN; "
            f"got {res.outcome} k={res.signals.units_killed} "
            f"l={res.signals.units_lost}"
        )
