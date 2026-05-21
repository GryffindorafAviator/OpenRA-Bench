"""harass-response-preserve — perimeter alert under preservation cap.

The bar: intended fire-support-on-home-turf WINS on every level and
every hard seed; stall (only observe), brute-chase (move defenders
east), and yield (pull defenders back to map edge) LOSE on every
level. Non-win is a real reachable timeout LOSS.

Validation is scripted (no model / network) — the policies below are
exhaustive proxies for the four real strategies, and they exercise
the predicate teeth directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "harass-response-preserve.yaml"


# ── unit-level predicate checks ──────────────────────────────────────

def _ctx(
    units_xy=(),
    enemies_seen=0,
    tick=1000,
    killed=0,
    lost=0,
    own_buildings=(),
):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(range(enemies_seen)),
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
    units = [(14, 18), (14, 20), (14, 22), (13, 19), (13, 21)]
    procs = [("proc", 16, 20)]

    # Intended: 1 kill, no losses, proc alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units, tick=3000, killed=1, lost=0, own_buildings=procs),
    )
    # Stall analogue: proc razed → fail, regardless of kills
    assert not evaluate(
        c.win_condition,
        _ctx(units, tick=3000, killed=1, lost=0, own_buildings=[]),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=0, lost=0, own_buildings=[]),
    )
    # Attrition cap busted (>2 lost) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=1, lost=3, own_buildings=procs),
    )
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=4502, killed=0, lost=0, own_buildings=procs),
    )
    assert 4501 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 4501 must be reachable within max_turns "
        "(else timeout draws instead of losing)"
    )


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    units = [(14, 18), (14, 20), (14, 22), (13, 19), (13, 21)]
    procs = [("proc", 16, 20)]

    # Intended: 2 kills, no losses, proc alive → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units, tick=3000, killed=2, lost=0, own_buildings=procs),
    )
    # 5 alive but proc razed → fail (preserve clause)
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=2, lost=0, own_buildings=[]),
    )
    # Lost 2 (cap is 1) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=2, lost=2, own_buildings=procs),
    )
    # Only 1 kill — predicate fails (need ≥2)
    assert not evaluate(
        c.win_condition,
        _ctx(units, tick=3000, killed=1, lost=0, own_buildings=procs),
    )
    # own_units_gte:5 — drop to 4 alive → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units[:4], tick=3000, killed=2, lost=1, own_buildings=procs),
    )
    # Timeout reachable
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_two_buildings_and_zero_loss():
    c = compile_level(load_pack(PACK_PATH), "hard")
    units = [(14, 18), (14, 20), (14, 22), (13, 19), (13, 21)]
    both = [("proc", 16, 20), ("tent", 16, 12)]

    # Intended: 3 kills, no losses, BOTH buildings alive → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units, tick=3000, killed=3, lost=0, own_buildings=both),
    )
    # Lost 1 (cap is 0) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=3, lost=1, own_buildings=both),
    )
    # proc gone (tent alive) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=3, lost=0,
             own_buildings=[("tent", 16, 12)]),
    )
    # tent gone (proc alive) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units, tick=3000, killed=3, lost=0,
             own_buildings=[("proc", 16, 20)]),
    )
    # Timeout reachable
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the staging latitude (anti-
    memorisation). Verified at compile time; the engine-roundtrip is
    asserted by tests/test_hard_tier.py::test_curated_hard_still_…"""
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
    assert pack.meta.id == "harass-response-preserve"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "τ²-bench" in joined or "tau" in joined or "distractor" in joined
    assert "soc" in joined or "perimeter" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


# ── engine-driven scripted policies ──────────────────────────────────

pytestmark_engine = pytest.mark.skipif(
    pytest.importorskip is None, reason="never"
)


# The defended asset footprint — defenders regroup here whenever no
# probe is visible so they stay ON the refinery (16,20) / barracks and
# are in range the moment the patrol bounces a probe back west.
_ANCHOR = (15, 20)


def _intended_policy(rs, Command):
    """Fire-support-on-home-turf: order every defender onto the nearest
    visible enemy unit each turn; when no probe is in sight (it has
    oscillated east into fog) regroup ON the asset footprint. The
    patroller's own oscillation script brings it back into range
    repeatedly — re-engaging each swing kills every probe in turn
    while the defenders never leave the asset."""
    units = rs.get("units_summary", []) or []
    enemies = [e for e in (rs.get("enemy_summary", []) or [])
               if not e.get("is_building")]
    if not units:
        return [Command.observe()]
    cmds = []
    if enemies:
        for u in units:
            # Pick the nearest visible enemy unit (not building).
            ux, uy = u["cell_x"], u["cell_y"]
            target = min(enemies, key=lambda e:
                         (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2)
            cmds.append(
                Command.attack_unit([str(u["id"])], str(target["id"]))
            )
    else:
        # No probe in sight — hold the asset footprint, do not wander.
        for u in units:
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=_ANCHOR[0], target_y=_ANCHOR[1]
                )
            )
    return cmds


def _stall_policy(rs, Command):
    return [Command.observe()]


def _brute_chase_policy(rs, Command):
    """Hunt the probe off-map east. Defenders abandon the asset and
    march to x=120 (where the enemy `fact` marker sits). Even if they
    catch the bouncing probes, they overextend; on hard the second
    spawn_point's probe (parallel arc) eats them; in every tier the
    abandoned proc is chipped to death."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=120, target_y=20)
        )
    return cmds


def _yield_policy(rs, Command):
    """Pull defenders back to the map edge (the SW corner), conceding
    the proc footprint entirely. proc gets razed → fail."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(
            Command.move_units([str(u["id"])], target_x=2, target_y=38)
        )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_policy_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended fire-support should WIN, "
            f"got {res.outcome} after {res.turns} turns "
            f"(killed={res.signals.units_killed}, "
            f"lost={res.signals.units_lost}, "
            f"buildings={sorted(res.signals.own_building_types)})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: stall must LOSE (proc razed under unopposed chip), "
        f"got {res.outcome} after {res.turns} turns; "
        f"buildings={sorted(res.signals.own_building_types)}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_brute_chase_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _brute_chase_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: brute-chase east must LOSE (proc undefended OR "
        f"team caught out), got {res.outcome} after {res.turns} turns; "
        f"buildings={sorted(res.signals.own_building_types)}, "
        f"lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_yield_policy_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _yield_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: yield/retreat must LOSE (proc razed), got "
        f"{res.outcome}; buildings={sorted(res.signals.own_building_types)}"
    )


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks 4501 must fit inside max_turns
    on every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl
