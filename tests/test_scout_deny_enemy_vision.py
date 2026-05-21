"""scout-deny-enemy-vision — REASONING capability validation.

Counter-reconnaissance: enemy scouts hold observation posts watching
the agent's base; left alone they complete their report. The agent
must intercept and KILL every scout before the report window closes
(`within_ticks`). Sitting still lets the report complete; covering
only ONE recon vector leaves the other scout's report intact.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed — the
    HoldFire strike force never engages; 0 kills; the scouts
    survive the window → the `after_ticks` deadline bites.
  * wrong-path (sweep away from the scouts) LOSES every tier /
    seed — no scout is reached → 0 kills.
  * one-lane (commit the whole force to a single recon vector)
    LOSES on medium/hard — kills one scout, the other completes
    its report → the N=2 kill bar is unmet.
  * intended counter-recon (split the force to cover EVERY recon
    vector, attack_unit each scout) WINS every tier / every hard
    seed.
  * hard tier defines ≥2 agent spawn_point groups (NORTH / SOUTH
    base) round-robined by seed so the interception geometry
    cannot be memorised.
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

PACK = PACKS_DIR / "scout-deny-enemy-vision.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _tanks(rs):
    return [u for u in (rs.get("units_summary") or []) if u.get("type") == "2tnk"]


def _scouts(rs):
    return [e for e in (rs.get("enemy_summary") or []) if (e.get("type") or "").lower() == "e3"]


def _stall(rs, C):
    """Observe-only — the HoldFire force never engages → 0 kills →
    the scouts survive the report window → LOSS."""
    return [C.observe()]


def _wrong_path(rs, C):
    """Sweep toward the far enemy marker, away from the scouts. No
    scout is ever reached → 0 kills → LOSS."""
    own = _tanks(rs)
    if not own:
        return [C.observe()]
    return [C.attack_move([str(u["id"]) for u in own], target_x=110, target_y=20)]


def _one_lane(rs, C):
    """Commit the WHOLE force to a single recon vector (the north
    scout). On medium/hard the south scout completes its report →
    the N=2 kill bar is unmet → LOSS."""
    own = _tanks(rs)
    if not own:
        return [C.observe()]
    scouts = sorted(_scouts(rs), key=lambda e: e["cell_y"])
    if scouts:
        return [C.attack_unit([str(u["id"]) for u in own], str(scouts[0]["id"]))]
    return [C.attack_move([str(u["id"]) for u in own], target_x=55, target_y=12)]


def _counter_recon(rs, C):
    """The intended capability — cover EVERY recon vector. With
    vision, every tank attacks the nearest scout; without vision,
    split the force (north half sweeps the north lane, south half
    the south lane) so both observation posts are reached."""
    own = _tanks(rs)
    if not own:
        return [C.observe()]
    scouts = _scouts(rs)
    cmds = []
    if scouts:
        for u in own:
            s = min(
                scouts,
                key=lambda e: abs(e["cell_x"] - u["cell_x"])
                + abs(e["cell_y"] - u["cell_y"]),
            )
            cmds.append(C.attack_unit([str(u["id"])], str(s["id"])))
        return cmds
    sown = sorted(own, key=lambda u: u["cell_y"])
    half = len(sown) // 2
    for u in sown[:half] or sown[:1]:
        cmds.append(C.attack_move([str(u["id"])], target_x=55, target_y=12))
    for u in sown[half:]:
        cmds.append(C.attack_move([str(u["id"])], target_x=55, target_y=28))
    return cmds


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_reasoning():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-deny-enemy-vision"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "counter-reconnaissance" in anchors, anchors
    assert "scout denial" in anchors, anchors
    assert "intrusion prevention" in anchors, anchors


def test_tools_are_combat_only():
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("move_units", "attack_unit", "attack_move", "stop"):
        assert required in tools, f"missing tool: {required!r}"
    assert "build" not in tools, "this is a counter-recon combat pack — no build tool"


def test_scouts_are_holdfire_e3():
    """Every level's enemy scouts are e3 at stance:0 (HoldFire) — they
    hold an observation post and never advance or fight."""
    pack = load_pack(PACK)
    counts = {"easy": 1, "medium": 2, "hard": 2}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        scouts = [
            a for a in c.scenario.actors if a.owner == "enemy" and a.type == "e3"
        ]
        assert len(scouts) == counts[lvl], (
            f"{lvl}: expected {counts[lvl]} e3 scouts; got {len(scouts)}"
        )
        for s in scouts:
            assert s.stance == 0, f"{lvl}: scout must be stance:0 HoldFire"


def test_agent_force_is_holdfire():
    """The agent strike force is stance:0 so it never auto-hunts —
    a stall cannot clear the scouts for free."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tanks = [
            a for a in c.scenario.actors if a.owner == "agent" and a.type == "2tnk"
        ]
        assert tanks, f"{lvl}: agent must have a 2tnk strike force"
        for t in tanks:
            assert t.stance == 0, f"{lvl}: agent tank must be stance:0 HoldFire"


def test_kill_bar_matches_scout_count():
    """The win kill bar must equal the number of scouts — every
    recon vector must be denied."""
    pack = load_pack(PACK)
    counts = {"easy": 1, "medium": 2, "hard": 2}
    for lvl in LEVELS:
        L = pack.levels[lvl]
        bar = next(
            int(c["units_killed_gte"])
            for c in L.win_condition.model_dump()["all_of"]
            if "units_killed_gte" in c
        )
        assert bar == counts[lvl], f"{lvl}: kill bar {bar} != scout count {counts[lvl]}"


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
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt < ceiling, f"{lvl}: within_ticks {wt} >= ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, f"{lvl}: within/after mismatch {wt}/{ft}"


def test_every_level_has_a_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


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
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, killed=0):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_kill_before_window():
    pe = compile_level(load_pack(PACK), "easy")
    # kill 1 scout in time → WIN
    assert evaluate(pe.win_condition, _ctx(tick=600, killed=1))
    # 0 kills → not win
    assert not evaluate(pe.win_condition, _ctx(tick=600, killed=0))
    # kill but past the window → not win
    assert not evaluate(pe.win_condition, _ctx(tick=1101, killed=1))
    # past after_ticks → fail
    assert evaluate(pe.fail_condition, _ctx(tick=1101, killed=0))
    # in-window → not fail
    assert not evaluate(pe.fail_condition, _ctx(tick=600, killed=0))

    pm = compile_level(load_pack(PACK), "medium")
    # only 1 of 2 scouts killed → not win, and the deadline still fails
    assert not evaluate(pm.win_condition, _ctx(tick=600, killed=1))
    assert evaluate(pm.win_condition, _ctx(tick=600, killed=2))
    assert evaluate(pm.fail_condition, _ctx(tick=1302, killed=1))


# ── engine-driven: every lazy/wrong policy LOSES, intended WINS ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_wrong_path_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _wrong_path, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: wrong-path must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_one_lane_loses_on_multi_vector_tiers(level, seed):
    """Committing the whole force to one recon vector kills only one
    scout; the other completes its report → LOSS on medium/hard."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _one_lane, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: one-lane must LOSE; got {r.outcome} "
        f"killed={r.signals.units_killed} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_counter_recon_wins_every_tier_and_seed(level, seed):
    """The intended capability — cover EVERY recon vector — WINS
    every tier and every hard seed."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _counter_recon, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: counter-recon must WIN; got {r.outcome} "
        f"killed={r.signals.units_killed} tick={r.signals.game_tick}"
    )


def test_counter_recon_run_is_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _counter_recon, seed=2)
    b = run_level(c, _counter_recon, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
