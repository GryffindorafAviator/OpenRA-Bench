"""combat-skirmish-then-disengage — ONE coordinated strike-then-pull-back.

Bar: the intended skirmish-then-disengage policy WINS on every level
and every hard seed; stall (only observe), never-engage (park at
start), and commit-until-overwhelmed (charge east and never retreat)
LOSE on every level. Non-win is a real reachable timeout LOSS (not a
draw).

Validation is scripted (no model / network): the four policies below
are the exhaustive proxies for the four real strategies and exercise
the predicate teeth directly. The load-bearing decision under test is
"stop attacking after the kill bar is met and order the disengage
back to the recovery zone before the deadline".
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-skirmish-then-disengage.yaml"


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


def test_predicates_easy_recovery_clause():
    c = compile_level(load_pack(PACK_PATH), "easy")
    home = [(5, 20), (5, 20), (5, 20), (5, 20)]
    cluster = [(50, 20), (50, 20), (50, 20), (50, 20)]
    mixed_3_home = [(5, 20), (5, 20), (5, 20), (50, 20)]

    # Intended: 3+ kills, ≥3 alive, ≥3 in recovery → WIN
    assert evaluate(c.win_condition, _ctx(home, tick=2000, killed=3, lost=0))
    assert evaluate(c.win_condition, _ctx(mixed_3_home, tick=2000, killed=4, lost=0))
    # Kill bar met but all units still at the kill site → fail region clause
    assert not evaluate(c.win_condition, _ctx(cluster, tick=2000, killed=4, lost=0))
    # 3 kills but only 2 own_units → predicate fails
    assert not evaluate(c.win_condition, _ctx(home[:2], tick=2000, killed=3, lost=2))
    # 0 kills → predicate fails even if everyone is at home
    assert not evaluate(c.win_condition, _ctx(home, tick=2000, killed=0, lost=0))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(home, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 4501 must be reachable within max_turns"
    )


def test_predicates_medium_same_bar_six_enemies():
    c = compile_level(load_pack(PACK_PATH), "medium")
    home = [(5, 20), (5, 20), (5, 20), (5, 20)]
    cluster = [(50, 20), (50, 20), (50, 20), (50, 20)]

    # Intended: 3+ kills, ≥3 alive, ≥3 in recovery → WIN
    assert evaluate(c.win_condition, _ctx(home, tick=3000, killed=3, lost=0))
    # Commit-and-stay: kill bar met but jeeps at cluster, not home → fail
    assert not evaluate(c.win_condition, _ctx(cluster, tick=3000, killed=6, lost=0))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(home, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_any_of_spawn_corner_recovery():
    c = compile_level(load_pack(PACK_PATH), "hard")
    home_north = [(5, 10), (5, 10), (5, 10), (5, 10)]
    home_south = [(5, 30), (5, 30), (5, 30), (5, 30)]
    mid_lat = [(5, 20), (5, 20), (5, 20), (5, 20)]  # neither corner
    cluster = [(50, 20), (50, 20), (50, 20), (50, 20)]

    # Either spawn corner satisfies the any_of recovery clause.
    assert evaluate(c.win_condition, _ctx(home_north, tick=3000, killed=3, lost=0))
    assert evaluate(c.win_condition, _ctx(home_south, tick=3000, killed=3, lost=0))
    # Mid-latitude (y=20) is outside BOTH spawn-corner radii (radius=6
    # from (5,10) ⇒ y=20 is 10 cells away; same from (5,30)) → fail.
    assert not evaluate(c.win_condition, _ctx(mid_lat, tick=3000, killed=3, lost=0))
    # Commit-and-stay at cluster → fail region clause.
    assert not evaluate(c.win_condition, _ctx(cluster, tick=3000, killed=6, lost=0))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(home_north, tick=4502, killed=0, lost=0))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the raider start corner."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "action"
    assert pack.meta.id == "combat-skirmish-then-disengage"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the doctrines the brief calls out: SC2 skirmisher +
    # military reconnaissance-by-fire / cavalry doctrine.
    assert "skirmish" in joined
    assert "recon" in joined or "cavalry" in joined or "disengage" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks 4501 fits inside max_turns on
    every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ──────────────────────────────────
#
# Stage thresholds for the intended policy:
#   PHASE_STRIKE_UNTIL_TICK = 1300 — drive east, attack_unit any
#     visible e1 until this tick (~14 turns @ ~90 ticks/turn). At
#     this point the kill bar (≥3) is comfortably met and the
#     disengage budget (4500 - 1300 = ~3200 ticks ≈ 35 turns) is
#     more than enough to march back to (5,20)-ish.
#   APPROACH_STEP = 15  cells / turn east toward the cluster axis
#     (jeep ~50% faster than infantry).

PHASE_STRIKE_UNTIL_TICK = 1300
APPROACH_STEP = 15
APPROACH_LIMIT_X = 46  # don't overrun the cluster (~x=50)


def _e1_foes(enemies):
    return [
        e for e in enemies
        if (e.get("type") or "").lower() == "e1"
        and not e.get("is_building")
    ]


def _stall_policy(rs, Command):
    """Stall: only observe. Kill bar never met (jeeps are stance:0;
    no auto-return-fire) → LOSS on the clock; on hard the hunt-bot
    e1 close on the idle stack and wipe it → LOSS on
    `not own_units_gte:1`."""
    return [Command.observe()]


def _never_engage_policy(rs, Command):
    """Park at the start; never move east, never fire. Recovery
    region clause is trivially satisfied but the kill bar is unmet
    → LOSS on the clock (easy/medium) or LOSS on hard when hunt-bot
    e1 wipe the idle stack."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units(
                [str(u["id"])], target_x=u["cell_x"], target_y=u["cell_y"]
            )
        )
    return cmds


def _commit_until_overwhelmed_policy(rs, Command):
    """Charge east; attack_unit any visible foe; never retreat. The
    kill bar IS met (4× jeep MG vs stance:0 rifles), but the jeeps
    end the run sitting at the kill site (~x=50), not in the
    recovery region. The region clause fails → after_ticks LOSS.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    if not units:
        return [Command.observe()]
    foes = _e1_foes(enemies)
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if foes:
            t = min(
                foes,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(t["id"])))
        else:
            # March east to the cluster axis but STOP at the cluster
            # (don't overrun to the far-east fact and trip auto-done).
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(50, ux + 12), target_y=uy
                )
            )
    return cmds


def _intended_skirmish_then_disengage_policy(rs, Command):
    """Intended skirmisher cycle:
      - PHASE 1 (tick < PHASE_STRIKE_UNTIL_TICK): drive east, attack_unit
        any visible e1.
      - PHASE 2 (tick >= PHASE_STRIKE_UNTIL_TICK): stop attacking; order
        move_units back to the nearest spawn corner — the RECOVERY zone.
    The phase switch is the spec's load-bearing decision: "stop
    fighting and pull back" before the deadline.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tick = rs.get("game_tick") or 0
    if not units:
        return [Command.observe()]
    foes = _e1_foes(enemies)
    # Pick the nearest spawn-corner candidate as the recovery target
    # (stateless — works for both single-corner and any_of-corner
    # recovery clauses).
    candidates = [(5, 20), (5, 10), (5, 30)]
    cx = sum(u["cell_x"] for u in units) / len(units)
    cy = sum(u["cell_y"] for u in units) / len(units)
    home = min(
        candidates, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2
    )
    cmds = []
    if tick < PHASE_STRIKE_UNTIL_TICK:
        if foes:
            for u in units:
                ux, uy = u["cell_x"], u["cell_y"]
                t = min(
                    foes,
                    key=lambda e: (e["cell_x"] - ux) ** 2
                    + (e["cell_y"] - uy) ** 2,
                )
                cmds.append(
                    Command.attack_unit([str(u["id"])], str(t["id"]))
                )
        else:
            # No foes in sight yet — drive east toward the cluster
            # axis. Cap at APPROACH_LIMIT_X so the strike force
            # doesn't overrun past the cluster.
            for u in units:
                ux, uy = u["cell_x"], u["cell_y"]
                cmds.append(
                    Command.move_units(
                        [str(u["id"])],
                        target_x=min(APPROACH_LIMIT_X, ux + APPROACH_STEP),
                        target_y=uy,
                    )
                )
    else:
        # PHASE 2: PULL BACK. Stop fighting; drive home.
        for u in units:
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=home[0], target_y=home[1]
                )
            )
    return cmds


# ── policy bar tests ────────────────────────────────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    """Stall must LOSE on every level: jeeps are stance:0 so they
    never return fire (kill bar unmet); on hard the hunt-bot e1
    close on the idle stack and trip `not own_units_gte:1`."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE, got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_never_engage_loses(level):
    """Park-at-start must LOSE: kill bar unmet; on hard hunt-bot e1
    wipe the idle stack."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _never_engage_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: never-engage must LOSE, got {res.outcome} "
            f"killed={res.signals.units_killed} lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_commit_until_overwhelmed_loses(level):
    """Commit-and-stay at the cluster must LOSE on every level: the
    kill bar IS met but the jeeps end the run at the kill site
    (~x=50), not the recovery region around the start. The region
    clause fails → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _commit_until_overwhelmed_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: commit-and-stay must LOSE, got "
            f"{res.outcome} killed={res.signals.units_killed} "
            f"lost={res.signals.units_lost}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_skirmish_then_disengage_wins(level):
    """Intended skirmisher (strike phase → disengage phase) must
    WIN on every level and every hard seed: kill bar met, ≥3 jeeps
    alive, ≥3 jeeps inside the spawn-corner recovery region, all
    inside the 4500-tick budget."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(
            c, _intended_skirmish_then_disengage_policy, seed=s
        )
        assert res.outcome == "win", (
            f"{level} seed={s}: intended skirmish-then-disengage should "
            f"WIN, got {res.outcome} after {res.turns} turns "
            f"(killed={res.signals.units_killed}, "
            f"lost={res.signals.units_lost})"
        )
