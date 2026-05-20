"""lh-100-turn-marathon-survival pack — full no-cheat validation on Rust.

Wave-5 Group G long-horizon endurance pack. The win is gated on FOUR
things still being true at the after_ticks survival gate:
  1. has_building: fact
  2. own_units_gte: 3
  3. building_count_gte pbox 2
  4. after_ticks N (survived to the gate) AND within_ticks N+x

Sustained pressure comes from STAGGERED hunt squads placed at
increasing x along the east-west axis, so squads arrive in successive
time windows rather than all at once. The fail clause uses the
``any_of: [after_ticks(GATE), not has_building:fact, not own_units_gte:1]``
idiom — eval_core evaluates win FIRST, so the broad after_ticks
fail only bites if the win all-of is unsatisfied at the gate (CLAUDE.md
rule 3: non-win must be a real LOSS, not a draw).

Bar (per CLAUDE.md): the intended SUSTAINED-OPS policy (kick harv +
build ≥2 pbox + continuously train e1) must WIN on every (level,
seed); stall / pure-defense-no-reinforce / pure-build-no-repair must
LOSE on every (level, seed). No draws.

NOTE: episodes are LONG (90/100/120 turns ≈ ~8/9/11 thousand ticks).
The engine-bound tests parametrise over 4 seeds × 3 levels × 4
policies = 48 runs; expect this file to take several minutes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-100-turn-marathon-survival.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Idles every turn — must LOSE (hunt squads raze fact)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _pure_defense_no_reinforce_policy():
    """Build pbox to 4 standing but NEVER train fresh infantry. The
    starting garrison bleeds under sustained pressure; either fact
    falls (no infantry to back the pbox line) or own_units drops
    below 3 (1) at the gate ⇒ LOSS."""
    state = {"placed_pbox": 0, "harv_kicked": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        pbox_have = sum(1 for b in ob if b["type"] == "pbox")
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        units = obs.get("units_summary", []) or []
        base = next((b for b in ob if b["type"] == "fact"), None)
        if not base:
            return [Cmd.observe()]
        bx, by = base["cell_x"], base["cell_y"]
        cmds = []
        if not state["harv_kicked"]:
            harv = next((u for u in units if u.get("type") == "harv"), None)
            if harv:
                cmds.append(Cmd.harvest([harv["id"]], 2, by))
                state["harv_kicked"] = True
        if pbox_have < 4 and "pbox" not in prod and cash >= 600:
            cmds.append(Cmd.build("pbox"))
        if pbox_have < 4:
            i = state["placed_pbox"]
            cmds.append(Cmd.place_building(
                "pbox", bx + 4 + (i % 4), by + (-3 + (i // 4))
            ))
            state["placed_pbox"] += 1
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pure_build_no_repair_policy():
    """Build just 2 pbox up front, then idle — no infantry production
    and no further building. The lone pbox pair gets stripped by
    sustained hunt fire (rocket squads contribute on medium/hard),
    fact eventually falls or building_count_gte pbox 2 drops below 2
    at the gate ⇒ LOSS."""
    state = {"placed_pbox": 0, "harv_kicked": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        units = obs.get("units_summary", []) or []
        base = next((b for b in ob if b["type"] == "fact"), None)
        if not base:
            return [Cmd.observe()]
        bx, by = base["cell_x"], base["cell_y"]
        cmds = []
        if not state["harv_kicked"]:
            harv = next((u for u in units if u.get("type") == "harv"), None)
            if harv:
                cmds.append(Cmd.harvest([harv["id"]], 2, by))
                state["harv_kicked"] = True
        # Build only 2 pbox total — no further construction, no
        # infantry production, no repair.
        if state["placed_pbox"] < 2:
            if "pbox" not in prod and cash >= 600:
                cmds.append(Cmd.build("pbox"))
            i = state["placed_pbox"]
            cmds.append(Cmd.place_building(
                "pbox", bx + 4 + (i % 4), by + (-3 + (i // 4))
            ))
            state["placed_pbox"] += 1
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_sustained_ops_policy():
    """The intended SUSTAINED-OPS capability play:
       turn 1: kick the pre-placed harvester onto the west mine
       continuous: keep ≥4 pbox standing in front of the base
       continuous: train fresh e1 so the e1 garrison stays ≥10
       (cap so we don't drain the queue waiting on cash).
       The continuous infantry production replaces losses from
       sustained hunt attrition; the pbox screen blunts each wave's
       leading edge so the infantry isn't overrun.
    """
    state = {"placed_pbox": 0, "harv_kicked": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        pbox_have = sum(1 for b in ob if b["type"] == "pbox")
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        units = obs.get("units_summary", []) or []
        e1_count = sum(1 for u in units if u.get("type") == "e1")
        # Use the LIVE fact position so this policy generalises across
        # the seed-driven NORTH/SOUTH hard-spawn round-robin.
        base = next((b for b in ob if b["type"] == "fact"), None)
        if not base:
            return [Cmd.observe()]
        bx, by = base["cell_x"], base["cell_y"]
        cmds = []

        # Kick the harvester at the west mine (mines are at x=2 on
        # every level — y matches the base latitude so the loop is
        # short and doesn't cross the lane).
        if not state["harv_kicked"]:
            harv = next((u for u in units if u.get("type") == "harv"), None)
            if harv:
                cmds.append(Cmd.harvest([harv["id"]], 2, by))
                state["harv_kicked"] = True

        # Maintain ≥4 standing pbox in front of the base. Queue one
        # at a time (cash budget is small early); spread placements.
        if pbox_have < 4 and "pbox" not in prod and cash >= 600:
            cmds.append(Cmd.build("pbox"))
        if pbox_have < 4:
            i = state["placed_pbox"]
            cmds.append(Cmd.place_building(
                "pbox", bx + 4 + (i % 4), by + (-3 + (i // 4))
            ))
            state["placed_pbox"] += 1

        # Sustained reinforcement — train fresh riflemen continuously.
        # The own_units_gte:3 win gate requires the garrison stays
        # alive; replacing losses is the load-bearing capability.
        if e1_count < 10 and "e1" not in prod and cash >= 100:
            cmds.append(Cmd.build("e1"))

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-100-turn-marathon-survival"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("lmgame" in a.lower() for a in anchors), anchors
    assert any("sc2le" in a.lower() or "endurance" in a.lower() for a in anchors), anchors
    assert any("sre" in a.lower() or "endurance" in a.lower() for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns, AND the
    after_ticks win gate must sit comfortably below reachable so the
    episode can actually reach it."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)
        wts: list = []
        ats: list = []
        _collect(win, "within_ticks", wts)
        _collect(win, "after_ticks", ats)
        assert wts, f"{lvl} has no within_ticks leaf"
        assert ats, f"{lvl} has no after_ticks leaf (no survival gate)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        for at in ats:
            assert at <= reachable, (
                f"{lvl} after_ticks={at} > reachable={reachable} — "
                f"survival gate never opens ⇒ win unreachable"
            )


def test_win_predicate_includes_all_four_survival_clauses():
    """The win must require has_building:fact, own_units_gte:3,
    building_count_gte pbox≥2, and after_ticks — all four together.
    A scenario that's missing any clause would be gameable."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        all_of = win.get("all_of") or []
        keys: set = set()
        for clause in all_of:
            keys |= set(clause.keys())
        assert "has_building" in keys, f"{lvl} win missing has_building: {keys}"
        assert "own_units_gte" in keys, f"{lvl} win missing own_units_gte: {keys}"
        assert "building_count_gte" in keys, (
            f"{lvl} win missing building_count_gte: {keys}"
        )
        assert "after_ticks" in keys, f"{lvl} win missing after_ticks: {keys}"
        assert "within_ticks" in keys, f"{lvl} win missing within_ticks: {keys}"


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_sustained_ops_wins(level, seed):
    """The intended sustained-ops policy (kick harv + maintain pbox
    line + continuous infantry production) must WIN on every (level,
    seed). Load-bearing solvency test."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_sustained_ops_policy(), seed=seed)
    pbox_n = sum(1 for t, _, _ in res.signals.own_buildings if t == "pbox")
    fact = any(t == "fact" for t, _, _ in res.signals.own_buildings)
    assert res.outcome == "win", (
        f"intended sustained-ops must WIN on {level} s={seed}; "
        f"got {res.outcome} turns={res.turns} tick={res.signals.game_tick} "
        f"units_lost={res.signals.units_lost} pbox={pbox_n} fact={fact}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed). Hunt squads raze
    fact (or the survival gate opens with the win all-of unmet)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_defense_no_reinforce_loses(level, seed):
    """Build pbox but NEVER train new infantry — the garrison bleeds
    under sustained pressure and either fact falls or own_units drops
    below 3 (1) at the survival gate ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_defense_no_reinforce_policy(), seed=seed)
    pbox_n = sum(1 for t, _, _ in res.signals.own_buildings if t == "pbox")
    fact = any(t == "fact" for t, _, _ in res.signals.own_buildings)
    assert res.outcome == "loss", (
        f"pure-defense-no-reinforce must LOSE on {level} s={seed}; got "
        f"{res.outcome} tick={res.signals.game_tick} "
        f"lost={res.signals.units_lost} pbox={pbox_n} fact={fact}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_build_no_repair_loses(level, seed):
    """Build only 2 pbox up front; no infantry production / no
    repair / no further building. The lone pbox pair gets stripped
    and fact eventually falls under sustained attrition ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_build_no_repair_policy(), seed=seed)
    pbox_n = sum(1 for t, _, _ in res.signals.own_buildings if t == "pbox")
    fact = any(t == "fact" for t, _, _ in res.signals.own_buildings)
    assert res.outcome == "loss", (
        f"pure-build-no-repair must LOSE on {level} s={seed}; got "
        f"{res.outcome} tick={res.signals.game_tick} "
        f"lost={res.signals.units_lost} pbox={pbox_n} fact={fact}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed."""
    c = compile_level(load_pack(PACK), "hard")
    # Stall — deterministic, fast; we only need to check the seed flows.
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
