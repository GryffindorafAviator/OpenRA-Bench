"""econ-overflow-to-silos — REASONING capability validation.

Real-world anchor: capacity planning / distributed-systems back-
pressure / FIFO perishable inventory cap. The agent owns a high-
throughput economy (fact + proc + powr + 7× harv on a dense adjacent
ore-patch column) whose producer rate exceeds the refinery's 2000-ore
storage cap. The agent must BUILD SILOS (cost 150, +3000 cap each) to
absorb the overflow or deposits beyond cap are silently lost on
`(stored+value).min(cap)`, leaving the EV bar unreachable inside the
tight within_ticks deadline.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no income, EV bar
     unmet → timeout LOSS).
   - no-silo (harvest-only) LOSES on medium and hard (overflow-bound
     accrual cannot reach the 18000 EV bar inside the 1803-tick
     deadline; empirically reaches only ~17840 EV by turn 20) →
     timeout LOSS.
   - intended build-silo policy WINS every tier / seed (silo absorbs
     the throughput, EV bar lands ~600 EV / 1 turn earlier).
   - hard tier defines ≥2 agent spawn_point groups (NORTH base y=10 /
     SOUTH base y=30) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-overflow-to-silos.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No-op every turn. No income, no silo → EV bar unmet →
    timeout LOSS on every tier."""
    return [Command.observe()]


def _btype(b):
    if isinstance(b, dict):
        return b.get("type")
    if isinstance(b, (list, tuple)) and b:
        return b[0]
    return None


def _harvest_only(rs, Command):
    """Pure harvest, no silo. Each harvester targets the mine adjacent
    to its starting cell (the harv's own (cell_x, cell_y) is the mine
    cell — mines are placed AT the harv positions so deposits
    saturate the proc cap on medium/hard). On easy this clears the
    EV bar (single harv, cap not binding); on medium/hard the 7-harv
    income saturates the 2000 proc cap and the EV bar is unreachable
    inside the deadline → LOSS."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    if not harvs:
        return [Command.observe()]
    cmds = []
    for h in harvs:
        cmds.append(
            Command.harvest([str(h["id"])], int(h["cell_x"]), int(h["cell_y"]))
        )
    return cmds


def _intended_silo(rs, Command):
    """Harvest AND build silos near the fact so the cap grows
    (2000 → 5000 with 1 silo; 8000 with 2; 11000 with 3) and high
    throughput is absorbed. The EV bar lands inside the deadline on
    every tier (~600 EV / 1 turn earlier than the no-silo run)."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    bldgs = rs.get("own_buildings") or []
    silos = [b for b in bldgs if _btype(b) == "silo"]
    facts = [b for b in bldgs if _btype(b) == "fact"]
    cmds = []
    for h in harvs:
        cmds.append(
            Command.harvest([str(h["id"])], int(h["cell_x"]), int(h["cell_y"]))
        )
    # Build ONE silo (+3000 cap → total cap 5000). One silo is
    # sufficient to absorb the high-throughput deposits over the
    # within_ticks budget; additional silos add cap headroom but
    # each extra build queues a turn cycle that slightly delays
    # the EV climb, so the minimal-effort intended-capability
    # policy is exactly one silo.
    if len(silos) < 1:
        cmds.append(Command.build("silo"))
        if facts:
            f = facts[0]
            fx = f.get("cell_x") if isinstance(f, dict) else f[1]
            fy = f.get("cell_y") if isinstance(f, dict) else f[2]
            # Stagger placement so silos don't collide.
            offset = 3 + len(silos)
            cmds.append(
                Command.place_building(
                    "silo", int(fx) + 2, int(fy) + offset
                )
            )
        else:
            cmds.append(Command.place_building("silo", 10, 21))
    return cmds


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-overflow-to-silos"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    # Anchor set required by Wave-8 spec.
    assert "capacity planning" in anchors
    assert "distributed-systems backpressure" in anchors
    assert "fifo inventory" in anchors


def test_tools_include_required_set():
    """Pack must declare the silo-build decision surface."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in (
        "observe", "build", "place_building", "harvest", "move_units", "stop",
    ):
        assert required in tools, f"missing tool: {required!r}"


def test_starting_cash_low():
    """Starting cash is LOW (not near the cap as in econ-silo-vs-spend);
    income must do the work, not the starting buffer."""
    pack = load_pack(PACK)
    assert pack.starting_cash <= 1000
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        assert L.starting_cash <= 1000


def test_required_prebuild_kit_present_each_tier():
    """Spec: pre-place fact + proc + powr + harvs (high throughput).
    No silo pre-placed (the agent must build them on medium/hard)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        # Count by type across each spawn group (hard tier duplicates
        # the kit per spawn). Per-spawn invariant: at least 1 fact /
        # proc / powr / harv.
        for sp in {a.spawn_point for a in agent_actors}:
            grp = [a for a in agent_actors if a.spawn_point == sp]
            types = [a.type for a in grp]
            for required in ("fact", "proc", "powr", "harv"):
                assert required in types, (
                    f"{lvl} spawn {sp}: missing {required!r}; got {types}"
                )
            assert "silo" not in types, (
                f"{lvl} spawn {sp}: silo must NOT be pre-placed"
            )
        # Medium / hard need ≥3 harvs in each spawn (high throughput
        # is the load-bearing setup — the cap is only binding under
        # sustained parallel deposits from many harvs on dense patches).
        if lvl in ("medium", "hard"):
            for sp in {a.spawn_point for a in agent_actors}:
                grp = [a for a in agent_actors if a.spawn_point == sp]
                harvs = [a for a in grp if a.type == "harv"]
                assert len(harvs) >= 3, (
                    f"{lvl} spawn {sp}: need ≥3 harvs (high throughput); "
                    f"got {len(harvs)}"
                )


def test_all_tiers_have_reachable_deadlines():
    """tick-alignment idiom: within_ticks ≤ ceiling AND
    after_ticks ≤ ceiling AND within_ticks + 1 == after_ticks (so a
    non-finisher LOSES, not draws)."""
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
            "(non-finisher must LOSE one tick past the win deadline)"
        )


def test_win_has_economy_value_gte_and_fact_each_tier():
    """Spec: win = economy_value_gte:M AND has_building:fact AND
    within_ticks:T. Medium / hard EV bar must be strictly higher
    than easy (the medium silo-required bar is the discrimination
    teeth; the load-bearing no-cheat assertion that no-silo LOSES
    is engine-validated by `test_no_silo_loses_on_medium_and_hard`,
    not by a static theoretical ceiling — the engine income model
    is bursty and the binding constraint is the within_ticks
    deadline, not a steady-state ceiling)."""
    pack = load_pack(PACK)
    bars = {}
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        clauses = L.win_condition.model_dump()["all_of"]
        ev_target = next(
            int(c["economy_value_gte"]) for c in clauses if "economy_value_gte" in c
        )
        bars[lvl] = ev_target
        has_fact = any(c.get("has_building") == "fact" for c in clauses)
        assert has_fact, f"{lvl}: win missing has_building:fact"
    assert bars["medium"] > bars["easy"], (
        f"medium EV bar {bars['medium']} must exceed easy {bars['easy']}"
    )
    assert bars["hard"] >= bars["medium"], (
        f"hard EV bar {bars['hard']} must be ≥ medium {bars['medium']}"
    )


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_fail_clauses_include_proc_and_fact_and_timeout():
    """Spec: fail = after_ticks T+1 OR not has_building:fact OR
    not has_building:proc."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        clauses = L.fail_condition.model_dump()["any_of"]
        has_after = any("after_ticks" in c for c in clauses)
        has_not_fact = any(
            isinstance(c.get("not"), dict)
            and c["not"].get("has_building") == "fact"
            for c in clauses
        )
        has_not_proc = any(
            isinstance(c.get("not"), dict)
            and c["not"].get("has_building") == "proc"
            for c in clauses
        )
        assert has_after, f"{lvl}: fail missing after_ticks clause"
        assert has_not_fact, f"{lvl}: fail missing not has_building:fact"
        assert has_not_proc, f"{lvl}: fail missing not has_building:proc"


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 distinct agent spawn_point groups (per spec:
    'Hard ≥2 spawn_point groups')."""
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


def _ctx(*, units=(), tick=500, cash=0, resources=0, own_buildings=()):
    import types
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=resources,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": list(units)})


def test_predicates_accept_intended_silo_with_high_ev():
    """A run that owns a silo and has high EV satisfies the medium
    tier win predicate."""
    c = compile_level(load_pack(PACK), "medium")
    kit = [
        ("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20),
        ("silo", 14, 21),
    ]
    assert evaluate(
        c.win_condition,
        _ctx(tick=1500, cash=15000, resources=5000, own_buildings=kit),
    )


def test_predicates_reject_no_silo_low_ev():
    """A no-silo run below the EV bar — the win predicate must NOT
    fire while EV is below the bar (the engine-policy assertion
    `test_no_silo_loses_on_medium_and_hard` is the load-bearing
    no-cheat teeth; this is the static-predicate sanity check)."""
    c = compile_level(load_pack(PACK), "medium")
    kit_no_silo = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert not evaluate(
        c.win_condition,
        _ctx(tick=1000, cash=15000, resources=2000, own_buildings=kit_no_silo),
    )


def test_predicates_fail_on_proc_loss():
    """proc destroyed → fail (per spec)."""
    c = compile_level(load_pack(PACK), "medium")
    no_proc = [("fact", 8, 18), ("powr", 10, 20)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_proc))


def test_predicates_fail_on_fact_loss():
    """fact destroyed → fail (per spec)."""
    c = compile_level(load_pack(PACK), "medium")
    no_fact = [("proc", 12, 18), ("powr", 10, 20)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_fact))


def test_predicates_fail_on_timeout():
    """tick past after_ticks → fail."""
    c = compile_level(load_pack(PACK), "medium")
    full_kit = [
        ("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20),
        ("silo", 14, 21),
    ]
    assert evaluate(
        c.fail_condition,
        _ctx(tick=1900, cash=3000, own_buildings=full_kit),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No income, no silo → EV bar unmet → timeout LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_no_silo_loses_on_medium_and_hard(level, seed):
    """Harvest-only (no silo) on medium/hard: 7-harv throughput
    saturates the 2000 cap; overflow is lost; the EV bar is
    unreachable inside the within_ticks deadline → timeout LOSS.
    This is the load-bearing capability assertion."""
    _, r = _run(level, _harvest_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: no-silo (harvest-only) must LOSE "
        f"(overflow → EV unreachable in time); got {r.outcome} "
        f"cash={r.signals.cash} resources={r.signals.resources} "
        f"turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_silo_wins_every_tier_and_seed(level, seed):
    """The intended capability — recognise overflow and build a silo
    to absorb high-throughput income — WINS every tier and every hard
    seed."""
    _, r = _run(level, _intended_silo, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: build-silo discipline should WIN; "
        f"got {r.outcome} cash={r.signals.cash} "
        f"resources={r.signals.resources} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed + same policy → identical outcome / cash / turn."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended_silo, seed=2)
    b = run_level(c, _intended_silo, seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash
    )
