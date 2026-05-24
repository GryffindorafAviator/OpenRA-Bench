"""combat-tanya-vs-rush — hero commando engagement (Tanya vs e1 rush).

Bar: the intended doctrine is "actively engage the hero asset with a
stance flip + per-turn attack_unit on the closest visible enemy".
Tanya's Colt45 one-shots an e1 (10000 dmg vs 5000 HP) and reloads in 7
ticks; without an active engagement she dies to the rush.

Strict engine-driven LOSS bars (all seeds, all levels) for the lazy
policies:

  • stall (only observe)            → LOSS (Tanya is stance:0 HoldFire
    by scenario default; she never auto-fires and is dropped by the
    rush — `not own_units_gte:1` fail clause fires)
  • brute attack_move toward the fact, NO stance flip → LOSS (the
    HoldFire stance gates auto-engage during attack_move, so Tanya
    walks east and dies without firing)

Strict engine-driven WIN bar for the intended doctrine:

  • intended (`set_stance(tanya, 2)` Defend on turn 1 + `attack_unit`
    on the closest visible e1 every subsequent turn) WINS on every
    level + every hard seed. Tanya kills the rush in time, alive at
    the end. Stance:2 (not :3) is the load-bearing choice — :3
    AttackAnything hunts the persistent fact marker at (28,20) the
    moment all e1 are cleared, ending the episode prematurely
    before scheduled-event wave B arrives (medium).

Engine roots fixed in the same push (see
`openra-sim/tests/test_tanya_alias.rs`):

  1. `GameRules::from_ruleset` now registers a `tanya` alias cloning
     the canonical `e7` ActorStats so a scenario `type: tanya` actor
     gets the real Allied-commando combat numbers (HP 10000, Colt45
     weapon) instead of the fallback (max_hp=50000, weapons=[], →
     100-dps "default" weapon). The bench scenarios use the canonical
     name `tanya`; without the alias every Tanya scenario was
     unsolvable in production (only the `defaults()` test path knew
     about "tanya").
  2. `from_ruleset` now filters `Armament@GARRISONED` from the
     actor's weapon list. C# OpenRA's GARRISONED weapon (e1's Vulcan,
     e2's Grenade, …) only fires while the unit is loaded into an
     AttackGarrisoned cargo slot (APC / pillbox) — a non-garrisoned
     unit must fire its PRIMARY armament. Before the filter,
     `best_weapon_against` would pick the GARRISONED weapon when it
     had higher effective damage (e1's Vulcan at 1000×Versus 200% =
     2000 eff beat M1Carbine at 1000×Versus 150% = 1500 eff),
     inflating every e1's anti-infantry DPS by ~33% across the bench
     and collapsing the Tanya doctrine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-tanya-vs-rush.yaml"


# ── unit-level predicate checks ─────────────────────────────────────


def _ctx(units=(), tick=1000, killed=0, lost=0):
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
        render_state={"units_summary": list(units)},
    )


def _tanya(x=12, y=20):
    return {"cell_x": x, "cell_y": y, "type": "tanya", "hp": 1.0}


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    alive = [_tanya()]
    dead = []

    # Intended: 4 kills, Tanya alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(alive, tick=2000, killed=4))
    # 3 kills → predicate fails
    assert not evaluate(c.win_condition, _ctx(alive, tick=2000, killed=3))
    # Tanya dead → fail clause fires regardless of kills
    assert evaluate(c.fail_condition, _ctx(dead, tick=2000, killed=4))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(alive, tick=2702, killed=0))
    assert 2701 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 2701 must be reachable within max_turns"
    )


def test_predicates_medium_six_kill_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    alive = [_tanya()]
    dead = []

    # Intended: 6 kills, Tanya alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(alive, tick=2000, killed=6))
    # 5 kills → predicate fails
    assert not evaluate(c.win_condition, _ctx(alive, tick=2000, killed=5))
    # Tanya dead → fail clause fires
    assert evaluate(c.fail_condition, _ctx(dead, tick=2000, killed=6))
    # Past deadline → real loss
    assert evaluate(c.fail_condition, _ctx(alive, tick=2702, killed=0))
    assert 2701 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_six_kill_bar():
    c = compile_level(load_pack(PACK_PATH), "hard")
    alive = [_tanya()]
    dead = []

    assert evaluate(c.win_condition, _ctx(alive, tick=2000, killed=6))
    assert not evaluate(c.win_condition, _ctx(alive, tick=2000, killed=5))
    assert evaluate(c.fail_condition, _ctx(dead, tick=2000, killed=6))
    assert evaluate(c.fail_condition, _ctx(alive, tick=2702, killed=0))
    assert 2701 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct enemy spawn_point
    groups so the seed round-robins which corridor (NORTH y=15..17 OR
    SOUTH y=23..25) the rush comes from, forcing the policy to read
    the rush direction at runtime."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "enemy"
    }
    assert len(groups) >= 2, f"hard needs ≥2 enemy spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "action"
    assert pack.meta.id == "combat-tanya-vs-rush"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "tanya" in joined or "hero" in joined or "commando" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: the after_ticks deadline fits inside
    max_turns on every level (~90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 2701 <= 93 + 90 * (c.max_turns - 1), (
            f"{lvl}: after_ticks 2701 not reachable within max_turns"
        )


# ── engine-driven scripted policies ──────────────────────────────────


def _stall_policy(rs, Command):
    """Stall: only observe. Tanya is stance:0 HoldFire by scenario
    default — she never auto-fires even when shot, so the rush walks
    in and drops her. `not own_units_gte:1` fail clause fires ⇒ LOSS."""
    return [Command.observe()]


def _brute_attack_move_policy(rs, Command):
    """Brute attack_move toward the persistent fact at (28,20)
    WITHOUT a stance flip. attack_move auto-targets only when stance
    allows; stance:0 HoldFire suppresses it, so Tanya walks east
    silently and dies under fire ⇒ LOSS."""
    units = rs.get("units_summary", []) or []
    own_tanya = [u for u in units if (u.get("type") or "").lower() == "tanya"]
    if not own_tanya:
        return [Command.observe()]
    u = own_tanya[0]
    return [Command.attack_move([str(u["id"])], target_x=28, target_y=20)]


def _intended_doctrine_policy_factory():
    """Closure-state factory so `stance_set` resets per episode."""
    stance_set = [False]

    def intended(rs, Command):
        units = rs.get("units_summary", []) or []
        enemies = rs.get("enemy_summary", []) or []
        own_tanya = [
            u for u in units if (u.get("type") or "").lower() == "tanya"
        ]
        rifle = [
            e
            for e in enemies
            if (e.get("type") or "").lower() == "e1" and not e.get("is_building")
        ]
        if not own_tanya:
            return [Command.observe()]
        u = own_tanya[0]
        ux, uy = u["cell_x"], u["cell_y"]
        cmds = []
        # Stance:2 Defend on turn 1 — auto-fires on the closest in-range
        # enemy without abandoning post to hunt the persistent fact
        # marker (stance:3 AttackAnything hunts the fact at (28,20)
        # once all e1 are cleared, ending the episode prematurely
        # before medium's scheduled-event wave B arrives at tick 300).
        if not stance_set[0]:
            cmds.append(Command.set_stance([str(u["id"])], 2))
            stance_set[0] = True
        if rifle:
            t0 = min(
                rifle,
                key=lambda e: max(abs(e["cell_x"] - ux), abs(e["cell_y"] - uy)),
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(t0["id"])))
        return cmds or [Command.observe()]

    return intended


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    """Stall must LOSE on every level and every hard seed — Tanya
    is stance:0 HoldFire and never fires; the rush wipes her."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE; got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_attack_move_loses(level):
    """Brute attack_move east toward the fact, NO stance flip, must
    LOSE — HoldFire gates auto-engage during attack_move, so Tanya
    walks silently into the rush and dies without firing."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _brute_attack_move_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: brute attack_move must LOSE; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_doctrine_wins(level):
    """Intended doctrine (stance:2 Defend + per-turn attack_unit on
    closest e1) must WIN on every level and every hard seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        intended = _intended_doctrine_policy_factory()
        res = run_level(c, intended, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended doctrine should WIN; got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )
