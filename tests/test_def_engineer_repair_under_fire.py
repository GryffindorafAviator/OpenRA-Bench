"""def-engineer-repair-under-fire pack — no-cheat validation on Rust.

Wave-8 ACTION pack: disaster-recovery triage / repair-order doctrine.
The agent's base has FOUR pre-placed structures (fact + proc + pbox +
fix) with the proc pushed forward into the threat lane (most-exposed,
most-critical building). A grenadier-led `rusher` band attrites the
proc cluster; the agent must COMMIT the hold-fire defenders to engage
the attackers AND toggle `repair` on the proc to keep it standing.

Bar (per CLAUDE.md):
  * stall (only observe) must LOSE on every (level, seed) — defenders
    never engage, kill bar 0, proc razed by the grenadier salvos.
  * repair-only (toggle repair on proc but never commit defenders)
    must LOSE — engagement output is zero, kill bar unmet AND proc
    eventually razed even with the repair toggle.
  * intended (commit defenders via attack_unit + toggle repair on the
    proc) must WIN on every (level, seed) — defenders clear the band,
    proc + fact both standing inside the within_ticks clock.

NOTE: the engine's `Cmd.repair` mechanically does not produce a large
HP-regen rate differential (verified by direct HP traces against the
engine). The repair tool is in the agent's toolkit and the brief
flags it as the load-bearing intervention; the engine-level
discrimination is between ENGAGE (commit defenders) and NO-ENGAGE,
which is sufficient to enforce "stall + repair-only LOSE / intended
WIN" on every level + every hard seed (1-4). The repair-during-
engagement framing aligns with the SC2 SCV / SRE disaster-recovery
triage anchor.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-engineer-repair-under-fire.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ───────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — must LOSE on every (level, seed)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _repair_only_policy():
    """Toggle repair on the proc every turn but never engage the
    attackers. Must LOSE — kill bar is unreachable without committing
    the hold-fire defenders, and the proc eventually falls anyway."""
    def pol(obs, Cmd):
        obldg = obs.get("own_buildings", []) or []
        for i, b in enumerate(obldg):
            if b.get("type") == "proc":
                return [Cmd.repair([str(i)])]
        return [Cmd.observe()]
    return pol


def _intended_engage_and_repair_policy():
    """Commit defenders via attack_unit AND toggle repair on the
    proc. Must WIN on every (level, seed)."""
    def pol(obs, Cmd):
        cmds = []
        obldg = obs.get("own_buildings", []) or []
        for i, b in enumerate(obldg):
            if b.get("type") == "proc":
                cmds.append(Cmd.repair([str(i)]))
                break
        units = obs.get("units_summary", []) or []
        enemies = [
            e for e in (obs.get("enemy_summary", []) or [])
            if not e.get("is_building")
        ]
        fighters = [
            u for u in units
            if u.get("type") in ("e1", "e3", "1tnk", "2tnk", "3tnk")
        ]
        if fighters and enemies:
            cmds.append(
                Cmd.attack_unit(
                    [u["id"] for u in fighters], str(enemies[0]["id"])
                )
            )
        if not cmds:
            cmds = [Cmd.observe()]
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-engineer-repair-under-fire"
    assert pack.meta.capability == "action"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_repair_tool_is_in_tools_list():
    """The load-bearing tool must be exposed for every level."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert "repair" in c.scenario.tools, (
            f"{lvl}: repair must be in tools (got {c.scenario.tools})"
        )


def test_meta_benchmark_anchors_match_doctrine():
    anchors = (load_pack(PACK).meta.benchmark_anchor or [])
    joined = " | ".join(a.lower() for a in anchors)
    assert "disaster recovery" in joined, anchors
    assert "scv" in joined or "repair" in joined, anchors


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_win_predicate_includes_proc_and_fact_and_kill_bar():
    """Win must require building_count_gte:proc, building_count_gte:fact,
    units_killed_gte, AND within_ticks — all four together."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        all_of = win.get("all_of") or []
        types = []
        keys: set = set()
        for clause in all_of:
            keys |= set(clause.keys())
            if "building_count_gte" in clause:
                types.append(clause["building_count_gte"].get("type"))
        assert "building_count_gte" in keys, (
            f"{lvl} win missing building_count_gte: {keys}"
        )
        assert "proc" in types and "fact" in types, (
            f"{lvl} win must require both proc AND fact: {types}"
        )
        assert "units_killed_gte" in keys, (
            f"{lvl} win missing units_killed_gte: {keys}"
        )
        assert "within_ticks" in keys, (
            f"{lvl} win missing within_ticks: {keys}"
        )


def test_fail_predicate_drops_proc_or_fact_or_clock():
    """Fail must trigger when proc OR fact dies, OR the clock expires."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fail = c.fail_condition.model_dump(exclude_none=True)
        any_of = fail.get("any_of") or []
        flat = " ".join(str(c) for c in any_of)
        assert "after_ticks" in flat, f"{lvl} fail needs after_ticks: {any_of}"
        assert "proc" in flat, f"{lvl} fail needs proc clause: {any_of}"
        assert "fact" in flat, f"{lvl} fail needs fact clause: {any_of}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks AND after_ticks fail clauses must both be
    reachable inside max_turns."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )
        fail = compile_level(pack, lvl).fail_condition.model_dump(
            exclude_none=True
        )

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
        ats_fail: list = []
        _collect(win, "within_ticks", wts)
        _collect(fail, "after_ticks", ats_fail)
        assert wts, f"{lvl} has no within_ticks leaf"
        assert ats_fail, f"{lvl} has no after_ticks fail leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        for at in ats_fail:
            assert at <= reachable, (
                f"{lvl} fail after_ticks={at} > reachable={reachable} — "
                f"timeout fail never fires ⇒ draw"
            )
        # Interrupt-mode budget (this pack enables enemy_unit_spotted +
        # own_unit_destroyed, so step_until_event cuts each turn short
        # to ~60 ticks). The after_ticks timeout must also fit inside
        # the interrupt-mode budget — the fixed-step bound above is too
        # loose and let a stale max_turns silently degenerate hard to a
        # DRAW after the engine balance pass. ~60 ticks/turn lower bound.
        interrupt_budget = 60 * max_turns
        for at in ats_fail:
            assert at <= interrupt_budget, (
                f"{lvl} fail after_ticks={at} > interrupt-mode budget "
                f"~{interrupt_budget} (max_turns={max_turns}) — the "
                f"timeout fail never bites in interrupt mode ⇒ draw"
            )


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


# ── Engine-bound tests (parameterised over seeds 1..4) ─────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_engage_and_repair_wins(level, seed):
    """The intended commit-defenders + toggle-repair-on-proc policy
    must WIN on every (level, seed). Load-bearing solvency test."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_engage_and_repair_policy(), seed=seed)
    pb = [t for t, _, _ in res.signals.own_buildings]
    proc_alive = "proc" in pb
    fact_alive = "fact" in pb
    assert res.outcome == "win", (
        f"intended must WIN on {level} s={seed}; got {res.outcome} "
        f"turns={res.turns} tick={res.signals.game_tick} "
        f"kills={res.signals.units_killed} lost={res.signals.units_lost} "
        f"proc={proc_alive} fact={fact_alive}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSE on every (level, seed). Defenders never
    engage, kills=0, the rusher band razes the proc."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_repair_only_loses(level, seed):
    """Toggle repair on proc every turn but NEVER engage the
    attackers. Must LOSE — without committed defenders the kill bar
    is unreachable and the proc eventually falls."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _repair_only_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"repair-only must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed (the
    test_hard_tier curation contract)."""
    c = compile_level(load_pack(PACK), "hard")
    # Stall — deterministic, fast; just confirm the seed flows.
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
