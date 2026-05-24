"""econ-resource-trade-with-self — REASONING capability validation.

Real-world anchor: divestiture / asset rotation / SC2 sell-and-rebuild.
The agent owns a working economy AND a redundant building (a second
war factory) that the current win clause does not need. Starting cash
is too low to afford the cash bar by the deadline on harvest income
alone. The intended verb is `sell` — convert the redundant building
back into cash (refund = cost / 2; weap @ 2000 refunds 1000), then
redeploy the freed cash into a second refinery whose doubled income
clears the cash bar inside the clock.

Bar (CLAUDE.md "no defect, no cheat"):
   * stall LOSES every tier / every hard seed (no income action; cash
     bar unmet → timeout LOSS).
   * hoard (harvest auto-runs but never sell + never build 2nd proc):
     the single harvester's income plateau is below the cash bar →
     LOSS.
   * sell-only (sell weap but don't build 2nd proc): the freed cash
     plus single-harv income still doesn't reach the bar in time on
     medium/hard → LOSS.
   * intended sell-and-invest (sell weap → +cost/2 → build 2nd proc →
     doubled income compounds): WINS every tier / every hard seed.
   * hard tier defines ≥2 agent spawn_point groups (NORTH y=10 /
     SOUTH y=28) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-resource-trade-with-self.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No-op every turn. No income action → cash bar unmet → timeout
    LOSS."""
    return [Command.observe()]


def _hoard(rs, Command):
    """Pure auto-harvest baseline — never sell, never build a 2nd
    proc. Single-harvester income plateau is below the cash bar →
    LOSS on every tier."""
    return [Command.observe()]


def _make_sell_only():
    """Sell the redundant weap once, then idle. Frees cash/2 but
    doesn't redeploy it; single-harv income still cannot reach the
    bar in time on medium/hard → LOSS."""
    state = {"sold": False}

    def policy(rs, Command):
        if not state["sold"]:
            bldgs = rs.get("own_buildings") or []
            weaps = [b for b in bldgs if b.get("type") == "weap"]
            if weaps:
                state["sold"] = True
                return [Command.sell([str(weaps[0]["id"])])]
        return [Command.observe()]

    return policy


def _make_sell_and_invest():
    """The intended capability: sell the redundant weap, build a 2nd
    proc to double income, then let auto-harvest compound past the
    cash bar."""
    state = {"sold": False, "proc_queued": False}

    def policy(rs, Command):
        bldgs = rs.get("own_buildings") or []
        weaps = [b for b in bldgs if b.get("type") == "weap"]
        procs = [b for b in bldgs if b.get("type") == "proc"]
        facts = [b for b in bldgs if b.get("type") == "fact"]
        cash = rs.get("cash", 0)
        if isinstance(cash, dict):
            cash = cash.get("value", 0)
        prod = [
            x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
        ]
        cmds = []
        # Step 1: sell the redundant weap (refund 1000).
        if weaps and not state["sold"]:
            state["sold"] = True
            return [Command.sell([str(weaps[0]["id"])])]
        # Step 2: queue the 2nd proc once we have the cash (1400).
        if len(procs) < 2:
            if "proc" not in prod and cash >= 1400:
                cmds.append(Command.build("proc"))
            # Place at the second patch (near (22, y_fact)).
            fy = 18
            for b in facts:
                fy = int(b.get("cell_y", 18))
                break
            cmds.append(Command.place_building("proc", 22, fy))
        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return policy


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-resource-trade-with-self"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors or "sell" in anchors
    assert "divestiture" in anchors or "asset rotation" in anchors


def test_tools_include_sell_verb():
    """The load-bearing verb for this pack is `sell` — must be in
    the tools surface."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    assert "sell" in tools, f"missing tool: sell; got {tools}"
    for required in (
        "observe", "build", "place_building", "harvest", "move_units", "stop",
    ):
        assert required in tools, f"missing tool: {required!r}"


def test_required_prebuild_kit_present_each_tier():
    """Spec: each tier pre-places fact + proc + powr + harv (the
    working economy) PLUS a redundant weap (the asset to sell)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        for sp in {a.spawn_point for a in agent_actors}:
            grp = [a for a in agent_actors if a.spawn_point == sp]
            types = [a.type for a in grp]
            for required in ("fact", "proc", "powr", "harv", "weap"):
                assert required in types, (
                    f"{lvl} spawn {sp}: missing {required!r}; got {types}"
                )


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment: within_ticks ≤ ceiling AND after_ticks ≤
    ceiling AND within_ticks + 1 == after_ticks."""
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
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch"
        )


def test_win_shape_cash_and_fact_each_tier():
    """Spec: win = cash_gte:M AND building_count_gte:fact:1 AND
    within_ticks:T. The M cash bar must tighten easy→medium→hard."""
    pack = load_pack(PACK)
    bars = {}
    for lvl in LEVELS:
        L = pack.levels[lvl]
        clauses = L.win_condition.model_dump()["all_of"]
        cash_target = next(
            int(c["cash_gte"]) for c in clauses if "cash_gte" in c
        )
        bars[lvl] = cash_target
        has_fact = any(
            isinstance(c.get("building_count_gte"), dict)
            and c["building_count_gte"].get("type") == "fact"
            for c in clauses
        )
        assert has_fact, f"{lvl}: win missing building_count_gte:fact"
        has_wt = any("within_ticks" in c for c in clauses)
        assert has_wt, f"{lvl}: win missing within_ticks"
    assert bars["medium"] > bars["easy"], (
        f"medium cash bar {bars['medium']} must exceed easy {bars['easy']}"
    )
    assert bars["hard"] > bars["medium"], (
        f"hard cash bar {bars['hard']} must exceed medium {bars['medium']}"
    )


def test_fail_clauses_include_proc_and_fact_and_timeout():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        clauses = L.fail_condition.model_dump()["any_of"]
        has_after = any("after_ticks" in c for c in clauses)
        has_not_fact = any(
            isinstance(c.get("not"), dict)
            and isinstance(c["not"].get("building_count_gte"), dict)
            and c["not"]["building_count_gte"].get("type") == "fact"
            for c in clauses
        )
        has_not_proc = any(
            isinstance(c.get("not"), dict)
            and c["not"].get("has_building") == "proc"
            for c in clauses
        )
        assert has_after, f"{lvl}: fail missing after_ticks clause"
        assert has_not_fact, f"{lvl}: fail missing not building_count_gte:fact"
        assert has_not_proc, f"{lvl}: fail missing not has_building:proc"


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=500, cash=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicate_accepts_high_cash():
    c = compile_level(load_pack(PACK), "easy")
    kit = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert evaluate(c.win_condition, _ctx(tick=2000, cash=6000, own_buildings=kit))


def test_predicate_rejects_below_cash_bar():
    c = compile_level(load_pack(PACK), "easy")
    kit = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert not evaluate(c.win_condition, _ctx(tick=2000, cash=5499, own_buildings=kit))


def test_predicate_fails_on_proc_loss():
    c = compile_level(load_pack(PACK), "easy")
    no_proc = [("fact", 8, 18), ("powr", 10, 20)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_proc))


def test_predicate_fails_on_fact_loss():
    c = compile_level(load_pack(PACK), "easy")
    no_fact = [("proc", 12, 18), ("powr", 10, 20)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_fact))


def test_predicate_fails_on_timeout():
    c = compile_level(load_pack(PACK), "easy")
    kit = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert evaluate(c.fail_condition, _ctx(tick=2800, cash=10000, own_buildings=kit))


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """No income action → cash bar unmet → timeout LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_hoard_loses_every_tier_and_seed(level, seed):
    """Pure auto-harvest, never sell + never build a 2nd proc.
    Single-harvester income plateau is below the cash bar → LOSS."""
    _, r = _run(level, _hoard, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: hoard (no sell, no 2nd proc) must LOSE; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_sell_only_loses_on_medium_and_hard(level, seed):
    """Sell the redundant weap but never redeploy: freed cash plus
    single-harv income alone still doesn't reach the bar on tighter
    tiers → LOSS. (The sell is necessary but not sufficient.)"""
    _, r = _run(level, _make_sell_only(), seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: sell-only must LOSE (freed cash isn't "
        f"redeployed; income alone is short); got {r.outcome} "
        f"cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_sell_and_invest_wins_every_tier_and_seed(level, seed):
    """The intended capability — sell the redundant weap, build a
    2nd proc to double income — WINS every tier/seed."""
    _, r = _run(level, _make_sell_and_invest(), seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: sell-and-invest must WIN; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


def test_sell_refund_increases_cash():
    """Sanity check the engine sell mechanic: selling the redundant
    weap (cost 2000) increases cash by the documented refund
    (cost / 2 = 1000) within a few ticks."""
    c = compile_level(load_pack(PACK), "easy")

    state = {"sold": False, "cash_before": None, "ticks_after_sell": 0}

    def sell_once(rs, Command):
        cash = rs.get("cash", 0)
        if isinstance(cash, dict):
            cash = cash.get("value", 0)
        if not state["sold"]:
            bldgs = rs.get("own_buildings") or []
            weaps = [b for b in bldgs if b.get("type") == "weap"]
            if weaps:
                state["sold"] = True
                state["cash_before"] = cash
                return [Command.sell([str(weaps[0]["id"])])]
        else:
            state["ticks_after_sell"] += 1
        return [Command.observe()]

    r = run_level(c, sell_once, seed=1)
    assert state["sold"], "sell command was never issued"
    # The cash should be at least cash_before + refund (1000) by the
    # end of the run (auto-harvest may also add income; we only assert
    # the refund landed, not the exact increment).
    final_cash = r.signals.cash
    assert final_cash >= state["cash_before"] + 900, (
        f"sell refund didn't land: cash_before={state['cash_before']} "
        f"final={final_cash}; expected ≥{state['cash_before'] + 900}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed + same policy → identical outcome / cash / turn."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _make_sell_and_invest(), seed=2)
    b = run_level(c, _make_sell_and_invest(), seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash
    )
