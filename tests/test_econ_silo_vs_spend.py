"""econ-silo-vs-spend — REASONING capability validation.

Real-world anchor: SC2 silo management; working-capital allocation;
FIFO perishable inventory cap. The agent starts with a minimum
revenue kit (fact + proc + powr + harv on an ore patch) and a starting
cash buffer NEAR the proc-only storage cap (1800 vs 2000). Income
accrues. The any_of clause has three legitimate spend-or-build paths:

* SILO: build('silo') (cost 150, prereq proc) — has_building:silo
* KILLS: build tent (400, prereq powr — pre-placed) + e1×K, attack
  the pre-placed enemy garrison — units_killed_gte:K
* PBOX-LINE: build tent + 2× pbox (600 each) —
  building_count_gte:{type:pbox,n:2}

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no silo, no kills, no
     pbox → any_of unmet → timeout LOSS).
   - pure-hold (harvest only, never build / never engage) LOSES every
     tier / seed (same: any_of unmet, cash hoards into cap, ore
     overflow spoils).
   - intended silo policy WINS every tier / seed (silo is the cheap
     floor at cost 150, latches has_building:silo).
   - hard tier defines ≥2 agent spawn_point groups (NORTH base y=14 /
     SOUTH base y=26) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-silo-vs-spend.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No-op every turn. No silo, no kills, no pbox → any_of unmet →
    timeout LOSS."""
    return [Command.observe()]


def _pure_hold(rs, Command):
    """Harvest only, never build / engage. Cash and stored ore
    accumulate; cap overflow spoils harvested ore; the any_of clause
    never latches (no silo, no kills, no pbox) → timeout LOSS."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    if not harvs:
        return [Command.observe()]
    return [
        Command.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs
    ]


def _intended_silo(rs, Command):
    """Build a silo (once) and keep harvesting. has_building:silo
    latches the any_of clause; the silo doubles the cap to 5000 so
    no income spoils; passive income clears the economy_value bar."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    bldgs = rs.get("own_buildings") or []

    def _btype(b):
        if isinstance(b, dict):
            return b.get("type")
        if isinstance(b, (list, tuple)) and b:
            return b[0]
        return None

    has_silo = any(_btype(b) == "silo" for b in bldgs)
    facts = [b for b in bldgs if _btype(b) == "fact"]
    cmds = []
    for h in harvs:
        cmds.append(Command.harvest([str(h["id"])], 22, int(h["cell_y"])))
    if not has_silo:
        cmds.append(Command.build("silo"))
        if facts:
            f = facts[0]
            fx = f.get("cell_x") if isinstance(f, dict) else f[1]
            fy = f.get("cell_y") if isinstance(f, dict) else f[2]
            cmds.append(Command.place_building("silo", int(fx) + 2, int(fy) + 3))
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
    assert pack.meta.id == "econ-silo-vs-spend"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    # Anchor set required by Wave-7 spec.
    assert "sc2 silo management" in anchors
    assert "working-capital allocation" in anchors
    assert "inventory cap" in anchors


def test_tools_include_required_set():
    """Pack must declare the [observe, build, place_building, harvest,
    move_units, attack_unit, attack_move, stop] toolset (the
    silo-or-spend decision surface)."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in (
        "observe", "build", "place_building", "harvest",
        "move_units", "attack_unit", "attack_move", "stop",
    ):
        assert required in tools, f"missing tool: {required!r}"


def test_starting_cash_near_storage_cap():
    """Starting cash must be NEAR the proc-only storage cap (2000) so
    overflow pressure bites quickly. The exact value is 1800."""
    pack = load_pack(PACK)
    # Pack-level default (overridden per-level to the same value).
    assert pack.starting_cash == 1800
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        # Cash is NEAR cap (within $300 of the 2000 proc cap).
        assert L.starting_cash >= 1500
        assert L.starting_cash <= 2000


def test_required_prebuild_kit_present_each_tier():
    """Spec: pre-place 1× fact, 1× proc, 1× powr, 1× harv (no silos)."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        types = [a.type for a in agent_actors]
        for required in ("fact", "proc", "powr", "harv"):
            assert required in types, (
                f"{lvl}: missing pre-placed {required!r} (got types {types})"
            )
        # No silo pre-placed (the agent must build it).
        assert "silo" not in types, f"{lvl}: silo must NOT be pre-placed"


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


def test_win_any_of_has_all_three_paths():
    """Spec: win any_of must offer silo, kills, AND pbox paths."""
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        clauses = L.win_condition.model_dump()["all_of"]
        any_of = next(c["any_of"] for c in clauses if "any_of" in c)
        keys = set()
        for clause in any_of:
            keys |= set(clause.keys())
        assert keys == {
            "has_building", "units_killed_gte", "building_count_gte"
        }, f"{lvl}: any_of must have silo / kills / pbox paths; got {keys}"


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


def _ctx(*, units=(), tick=500, cash=0, resources=0, own_buildings=(), killed=0):
    import types
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=0,
        cash=cash,
        resources=resources,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": list(units)})


def test_predicates_enforce_capability_silo_path():
    """SILO path: has_building:silo latches the any_of clause."""
    c = compile_level(load_pack(PACK), "medium")
    kit_with_silo = [
        ("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20),
        ("silo", 10, 21),
    ]
    assert evaluate(
        c.win_condition,
        _ctx(tick=600, cash=2300, resources=0, own_buildings=kit_with_silo),
    )


def test_predicates_enforce_capability_kills_path():
    """KILLS path: units_killed_gte:2 latches the any_of clause."""
    c = compile_level(load_pack(PACK), "medium")
    kit_no_silo = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert evaluate(
        c.win_condition,
        _ctx(
            tick=600, cash=2300, resources=0,
            own_buildings=kit_no_silo, killed=2,
        ),
    )


def test_predicates_enforce_capability_pbox_path():
    """PBOX-LINE path: building_count_gte:{type:pbox,n:2} latches."""
    c = compile_level(load_pack(PACK), "medium")
    kit_with_pbox = [
        ("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20),
        ("tent", 14, 22), ("pbox", 16, 22), ("pbox", 16, 23),
    ]
    assert evaluate(
        c.win_condition,
        _ctx(tick=600, cash=2300, resources=0, own_buildings=kit_with_pbox),
    )


def test_predicates_reject_hoard():
    """HOARD (no silo, no kills, no pbox) — even with cash and EV
    above the bar, the any_of clause is unmet → not a win."""
    c = compile_level(load_pack(PACK), "medium")
    kit_no_action = [("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20)]
    assert not evaluate(
        c.win_condition,
        _ctx(tick=600, cash=5000, resources=0, own_buildings=kit_no_action),
    )


def test_predicates_fail_on_proc_loss():
    """proc destroyed → fail (per spec fail clause)."""
    c = compile_level(load_pack(PACK), "medium")
    no_proc = [("fact", 8, 18), ("powr", 10, 20), ("silo", 10, 21)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_proc))


def test_predicates_fail_on_fact_loss():
    """fact destroyed → fail (per spec fail clause)."""
    c = compile_level(load_pack(PACK), "medium")
    no_fact = [("proc", 12, 18), ("powr", 10, 20), ("silo", 10, 21)]
    assert evaluate(c.fail_condition, _ctx(tick=600, own_buildings=no_fact))


def test_predicates_fail_on_timeout():
    """tick past after_ticks → fail."""
    c = compile_level(load_pack(PACK), "medium")
    full_kit = [
        ("fact", 8, 18), ("proc", 12, 18), ("powr", 10, 20),
        ("silo", 10, 21),
    ]
    assert evaluate(
        c.fail_condition,
        _ctx(tick=1262, cash=3000, own_buildings=full_kit),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No silo, no kills, no pbox → any_of clause never latches →
    timeout LOSS (hoarding policy must lose)."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (hoarder); "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_hold_loses_every_tier_and_seed(level, seed):
    """Harvest-only (no silo, no kills, no pbox) → any_of clause
    never latches; stored ore overflows the proc cap as income piles
    in; timeout LOSS."""
    _, r = _run(level, _pure_hold, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-hold must LOSE (still hoarding); "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_silo_wins_every_tier_and_seed(level, seed):
    """The intended capability — build a silo (cheap floor at $150)
    while harvesting — WINS every tier and every hard seed."""
    _, r = _run(level, _intended_silo, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: silo discipline should WIN; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


def _btype_helper(b):
    if isinstance(b, dict):
        return b.get("type")
    if isinstance(b, (list, tuple)) and b:
        return b[0]
    return None


def _intended_kills(rs, Command):
    """KILLS path (alternative to silo): build tent (400, prereq powr
    pre-placed) + e1 infantry (100), attack-move them at the pre-placed
    enemy garrison. units_killed_gte:1 latches the any_of clause.
    Validates that the KILLS spend lane is a real second viable path."""
    bldgs = rs.get("own_buildings") or []
    tent = any(_btype_helper(b) == "tent" for b in bldgs)
    cash = rs.get("cash", 0)
    if isinstance(cash, dict):
        cash = cash.get("value", 0)
    prod = [
        x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
    ]
    units = rs.get("units_summary", []) or []
    infs = [u for u in units if u.get("type") == "e1"]
    facts = [b for b in bldgs if _btype_helper(b) == "fact"]
    fy = 18
    if facts:
        f = facts[0]
        fy = (f.get("cell_y") if isinstance(f, dict) else f[2])
    cmds = []
    if not tent:
        if "tent" not in prod and cash >= 400:
            cmds.append(Command.build("tent"))
        cmds.append(Command.place_building("tent", 14, int(fy) + 3))
    elif len(infs) < 3:
        if "e1" not in prod and cash >= 100:
            cmds.append(Command.build("e1"))
    # Push infantry into the enemy garrison cluster.
    for u in infs:
        cmds.append(Command.attack_move([str(u["id"])], target_x=40, target_y=18))
    return cmds if cmds else [Command.observe()]


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_kills_wins_on_easy(seed):
    """KILLS path WINS on easy — second viable strategic choice (alt
    to silo). Build tent + e1 + attack the pre-placed garrison; the
    units_killed_gte clause latches the any_of and the harvest income
    clears the EV bar inside the loose easy deadline. The other tiers
    have a tighter within_ticks budget so this slower path is not
    expected to win there — silo remains the cheap floor everywhere."""
    _, r = _run("easy", _intended_kills, seed=seed)
    assert r.outcome == "win", (
        f"easy/seed{seed}: KILLS path (build tent + e1 + attack) must "
        f"WIN; got {r.outcome} cash={r.signals.cash} "
        f"kills={r.signals.units_killed} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed + same policy → identical outcome / cash / turn."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended_silo, seed=2)
    b = run_level(c, _intended_silo, seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash
    )
