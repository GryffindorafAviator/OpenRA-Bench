"""build-tech-skip-decision — REASONING capability validation.

Skip an unnecessary tech tier (PlanBench unnecessary-step pruning /
lean process / YAGNI). The objective only needs BASIC infantry: the
agent starts with a pre-placed Construction Yard (fact) + Allied
barracks (tent), so rifle infantry (e1, $100) are trainable from turn
1 with no prior tech step. A light enemy garrison of rifle infantry
(stance 3, AttackAnything) advances on the base. Training e1 and
rallying them at the base front clears the kill bar comfortably.

The trap is to climb the full tech chain to medium tanks
(powr -> proc -> weap + fix -> 2tnk) — a whole tech tier the
objective never asked for. It burns the clock: by the deadline the
tech play has only powr+proc standing and ZERO army.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed —
    0 kills, the `after_ticks` deadline bites.
  * tech-full-chain (build powr -> proc -> weap -> fix, then 2tnk)
    LOSES every tier / every hard seed — the unneeded tier burns the
    clock; no tank fields before the deadline, kill bar unmet.
  * intended skip-to-e1 (train e1 from the pre-placed tent from turn
    1, rally them at the base front, let them auto-fire on the
    closing garrison) WINS every tier / every hard seed — the kill
    bar is met AND the fact survives, well inside the deadline.
  * hard tier defines >=2 agent spawn_point groups (NORTH base y=14
    / SOUTH base y=26) round-robined by seed; the garrison is
    duplicated at both latitudes (enemy actors don't honour
    spawn_point — CLAUDE.md), so a memorised base-latitude opening
    cannot generalise.
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

PACK = PACKS_DIR / "build-tech-skip-decision.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _stall(rs, C):
    """Observe-only — no production. 0 kills, the after_ticks
    deadline bites -> LOSS."""
    return [C.observe()]


def _tech_full_chain(rs, C):
    """The wrong call: climb the full tech chain (powr -> proc ->
    weap -> fix -> 2tnk) instead of training basic infantry. The
    unneeded tier burns the clock — no tank fields before the
    deadline -> the kill bar is unmet -> LOSS."""
    ob = rs.get("own_buildings") or []
    bt = {b.get("type") for b in ob}
    fy = 20
    for b in ob:
        if b.get("type") == "fact":
            fy = int(b["cell_y"])
    prod = [p.get("item") for p in (rs.get("production") or []) if isinstance(p, dict)]
    cmds = []
    if "powr" not in bt:
        if "powr" not in prod:
            cmds.append(C.build("powr"))
        cmds.append(C.place_building("powr", 17, fy))
    elif "proc" not in bt:
        if "proc" not in prod:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", 20, fy))
    elif "weap" not in bt:
        if "weap" not in prod:
            cmds.append(C.build("weap"))
        cmds.append(C.place_building("weap", 23, fy))
    elif "fix" not in bt:
        if "fix" not in prod:
            cmds.append(C.build("fix"))
        cmds.append(C.place_building("fix", 23, fy + 4))
    else:
        cmds.append(C.build("2tnk"))
    return cmds or [C.observe()]


def _skip_to_e1(rs, C):
    """The intended capability — skip the tech tier. Train e1 from
    the pre-placed tent from turn 1 and rally them at the base front;
    they auto-fire on the closing garrison and clear the kill bar."""
    units = rs.get("units_summary") or []
    ob = rs.get("own_buildings") or []
    fy = 20
    for b in ob:
        if b.get("type") == "tent":
            fy = int(b["cell_y"])
    ids = [u.get("id") for u in units if u.get("type") == "e1"]
    cmds = [C.build("e1")]
    if ids:
        cmds.append(C.move_units(ids, 18, fy))
    return cmds


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-tech-skip-decision"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "planbench" in anchors, anchors
    assert "pruning" in anchors, anchors
    assert "yagni" in anchors, anchors
    assert "lean process" in anchors, anchors


def test_tools_include_build_and_combat_surface():
    """Pack must expose [observe, build, place_building, move_units,
    attack_move, stop] — the build-and-engage interaction surface."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("observe", "build", "place_building", "move_units",
                     "attack_move", "stop"):
        assert required in tools, f"missing tool: {required!r}"


def test_preplaced_base_is_fact_plus_tent_no_higher_tech():
    """Every level pre-places fact + tent for the agent — the cheap
    e1 path must be actionable on turn 1 (tent present) with NO
    higher-tech building (powr / proc / weap / fix) handed to the
    agent: building the tech tier is what the agent must AVOID. The
    hard tier additionally carries an inert HoldFire e1 spawn-witness
    per spawn group (units_summary spawn-variation contract)."""
    pack = load_pack(PACK)
    higher_tech = {"powr", "proc", "weap", "fix", "dome", "apwr"}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {a.type for a in c.scenario.actors if a.owner == "agent"}
        assert "fact" in agent_types, f"{lvl}: agent base missing fact"
        assert "tent" in agent_types, f"{lvl}: agent base missing tent"
        assert not (agent_types & higher_tech), (
            f"{lvl}: agent must NOT start with a higher-tech building; "
            f"got {sorted(agent_types & higher_tech)}"
        )
        # only fact / tent / (hard) the inert e1 witness
        assert agent_types <= {"fact", "tent", "e1"}, (
            f"{lvl}: unexpected agent actor types {sorted(agent_types)}"
        )


def test_garrison_is_light_infantry_stance3():
    """The enemy garrison is basic rifle infantry (e1) at stance 3
    (AttackAnything — advances on the agent base). No tanks / no
    higher-tech enemy units."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        garrison = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "e1"
        ]
        assert garrison, f"{lvl}: must have an e1 garrison"
        for a in garrison:
            assert a.stance == 3, (
                f"{lvl}: garrison e1 must be stance 3 (AttackAnything); "
                f"got {a.stance}"
            )
        # only e1 + the far fact marker on the enemy side
        enemy_types = {a.type for a in c.scenario.actors if a.owner == "enemy"}
        assert enemy_types == {"e1", "fact"}, (
            f"{lvl}: enemy actors must be {{e1, fact}}; got {sorted(enemy_types)}"
        )


def test_far_enemy_fact_marker_present():
    """A persistent unarmed enemy fact marker far east keeps the
    episode alive past the last garrison death (CLAUDE.md auto-done
    footgun) so a non-winner reaches the deadline as a real LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        far = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact" and a.position[0] >= 60
        ]
        assert far, f"{lvl}: missing far enemy fact marker (anti-DRAW)"


def test_every_level_has_reachable_timeout_fail():
    """`after_ticks` fail must bite WITHIN max_turns (so stall / tech
    are a real reachable LOSS, not a draw). within_ticks + 1 ==
    after_ticks so a non-finisher LOSES one tick past the window."""
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
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            "(boundary non-finisher must LOSE, not draw)"
        )


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
    assert sp == {0, 1}, f"hard must define exactly {{0, 1}}; got {sorted(sp)}"


def test_in_bounds_actors_on_every_level():
    """rush-hour-arena playable bounds ~ x:2..126, y:2..38."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, kills=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=0,
        cash=0,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_capability():
    """Win requires (kill bar AND fact alive AND in time); fail fires
    on timeout OR fact destroyed."""
    c = compile_level(load_pack(PACK), "easy")
    base_b = [("fact", 10, 20), ("tent", 13, 20)]

    # Intended: 4 kills, fact alive, in time -> WIN
    assert evaluate(c.win_condition, _ctx(tick=900, kills=4, own_buildings=base_b))
    # 3 kills (under bar) -> not win
    assert not evaluate(
        c.win_condition, _ctx(tick=900, kills=3, own_buildings=base_b)
    )
    # 4 kills but past within_ticks (easy within_ticks 1600) -> not win
    assert not evaluate(
        c.win_condition, _ctx(tick=1601, kills=4, own_buildings=base_b)
    )
    # 4 kills but fact destroyed -> not win
    assert not evaluate(
        c.win_condition, _ctx(tick=900, kills=4, own_buildings=base_b[1:])
    )
    # Past after_ticks deadline (easy after_ticks 1601) -> fail
    assert evaluate(
        c.fail_condition, _ctx(tick=1700, kills=0, own_buildings=base_b)
    )
    # Fact destroyed -> fail
    assert evaluate(
        c.fail_condition, _ctx(tick=900, kills=4, own_buildings=base_b[1:])
    )
    # Within deadline, fact alive -> not fail
    assert not evaluate(
        c.fail_condition, _ctx(tick=900, kills=0, own_buildings=base_b)
    )


# ── engine-driven: every lazy / wrong policy LOSES, intended WINS ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """Observe-only -> 0 kills + the after_ticks deadline bites ->
    a real reachable LOSS, not a draw."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"tick={r.signals.game_tick} kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_tech_full_chain_loses_every_tier_and_seed(level, seed):
    """Climbing the full tech chain (powr -> proc -> weap -> fix ->
    2tnk) burns the clock on a tier the objective never required -> no
    tank fields before the deadline -> kill bar unmet -> LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _tech_full_chain, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: tech-full-chain must LOSE (clock); "
        f"got {r.outcome} tick={r.signals.game_tick} "
        f"kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_skip_to_e1_wins_every_tier_and_seed(level, seed):
    """The intended capability — skip the unneeded tech tier, train
    e1 from the pre-placed tent and rally them at the base front.
    Wins every tier and every hard seed, well inside the deadline."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _skip_to_e1, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: skip-to-e1 must WIN; got {r.outcome} "
        f"tick={r.signals.game_tick} kills={r.signals.units_killed}"
    )


# ── determinism ─────────────────────────────────────────────────────


def test_skip_to_e1_run_is_deterministic_per_seed():
    """Same seed, same policy -> identical outcome / kills / turns."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _skip_to_e1, seed=3)
    b = run_level(c, _skip_to_e1, seed=3)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
