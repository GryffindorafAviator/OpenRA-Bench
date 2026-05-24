"""econ-cash-reserve-management — REASONING capability validation.

Real-world anchor: SC2 cash overflow management (spend down while
keeping reserve); corporate treasury / operational-reserve doctrine;
financial-runway management. The agent starts with a productive base
(4 buildings: fact + proc + tent + powr; 2 harvs on 2 near mines)
and $1500. The win predicate requires SPEND (build more buildings)
AND maintaining a CASH RESERVE simultaneously — the "cash management
loop" (operate while preserving working capital).

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no build → bldg_total
     stuck at 4 → bar unmet → timeout LOSS).
   - pure-hold (harvest only, never build) LOSES every tier / seed
     (same: bldg_total stuck at 4).
   - over-spend (chain proc 1400 + pbox 600 → cash dips to 0) LOSES
     on medium and hard (income cannot refill above the reserve bar
     before the tight tick deadline). Easy permits over-spend because
     the deadline is generous — that's the "loose-bar" tier by
     construction.
   - intended steady-build-with-reserve WINS every tier / seed.
   - hard tier defines ≥2 agent spawn_point groups (NORTH / SOUTH
     base) so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-cash-reserve-management.yaml"


# ── policies ────────────────────────────────────────────────────────


def _stall(rs, Command):
    """No-op every turn. Bldg_total never grows past 4 → bar unmet → LOSS."""
    return [Command.observe()]


def _pure_hold(rs, Command):
    """Harvest only, never build. Income accumulates but bldg_total
    stays at 4 → bar unmet → LOSS on building bar."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    if not harvs:
        return [Command.observe()]
    return [Command.harvest([str(h["id"])], 22, int(h["cell_y"])) for h in harvs]


def _over_spend(rs, Command):
    """Chain a proc (1400) + 2× pbox (600 each) immediately. Cash dips
    to 0 the moment bldg_total reaches 6/7 — must wait for income to
    refill above the 300/500 reserve bar. On medium/hard the tick
    budget runs out before recovery → LOSS."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = []
    for h in harvs:
        cmds.append(Command.harvest([str(h["id"])], 22, int(h["cell_y"])))
    # Queue + place orders spammed each turn until accepted.
    cmds.append(Command.build("proc"))
    cmds.append(Command.place_building("proc", 14, 24))
    cmds.append(Command.build("pbox"))
    cmds.append(Command.place_building("pbox", 16, 24))
    cmds.append(Command.build("pbox"))
    cmds.append(Command.place_building("pbox", 18, 24))
    return cmds


def _intended(rs, Command):
    """Build 3× powr ($900 total) while keeping both harvs harvesting.
    Reaches bldg_total 7 at cash ~600 (well above the medium 300 and
    hard 500 reserve). Wins easy/medium (bldg_total only needs 5/6)
    and hard (bldg_total 7). The cash-management capability: spend
    cheap, keep reserve."""
    units = rs.get("units_summary", []) or []
    harvs = [u for u in units if u.get("type") == "harv"]
    cmds = []
    for h in harvs:
        cmds.append(Command.harvest([str(h["id"])], 22, int(h["cell_y"])))
    cmds.append(Command.build("powr"))
    cmds.append(Command.place_building("powr", 18, 20))
    cmds.append(Command.build("powr"))
    cmds.append(Command.place_building("powr", 18, 22))
    cmds.append(Command.build("powr"))
    cmds.append(Command.place_building("powr", 18, 24))
    return cmds


# ── helpers ─────────────────────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "bespoke 48x40 arena terrain must be present"
    return c, run_level(c, policy, seed=seed)


# ── structural ──────────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-cash-reserve-management"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2" in anchors
    assert "treasury" in anchors or "reserve" in anchors or "runway" in anchors


def test_tools_include_required_set():
    """Pack must declare the [observe, build, place_building, harvest,
    move_units, stop] toolset (the cash-management interaction surface)."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("observe", "build", "place_building", "harvest",
                     "move_units", "stop"):
        assert required in tools, f"missing tool: {required!r}"


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
            "(non-finisher must LOSE, not draw — fail clause one tick"
            " past win clause)"
        )


def test_hard_has_two_seed_driven_spawn_groups():
    """Hard tier: ≥2 distinct agent spawn_point groups so engine
    round-robins start by seed."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_fail_condition_present_on_every_tier():
    pack = load_pack(PACK)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, cash=0, resources=0, own_buildings=()):
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
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (bldg_total bar AND cash reserve AND ≥2 harvs)
    AND in-time; fail fires on timeout OR fact destroyed."""
    c = compile_level(load_pack(PACK), "medium")
    two_harvs = [
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
        {"cell_x": 14, "cell_y": 20, "type": "harv"},
    ]
    six_bldgs = [
        ("fact", 10, 22),
        ("proc", 12, 18),
        ("tent", 10, 18),
        ("powr", 14, 22),
        ("powr", 18, 22),
        ("powr", 18, 24),
    ]

    # Intended: bldg_total≥6, cash≥300, 2 harvs, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=600, cash=900, own_buildings=six_bldgs),
    )
    # Cash one short of 300 → not a win (reserve discipline)
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=600, cash=299, own_buildings=six_bldgs),
    )
    # Only 5 buildings → not a win (growth bar)
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs, tick=600, cash=2000, own_buildings=six_bldgs[:5]),
    )
    # Only 1 harv (the other died) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=two_harvs[:1], tick=600, cash=900, own_buildings=six_bldgs),
    )
    # Timeout (tick past after_ticks): bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=two_harvs, tick=812, cash=0, own_buildings=six_bldgs),
    )
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=two_harvs, tick=600, cash=900, own_buildings=six_bldgs[1:]),
    )


# ── engine-driven policies (the no-cheat bar) ───────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses_every_tier_and_seed(level, seed):
    """No build → bldg_total stays at 4 → bar unmet → timeout LOSS."""
    _, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (no growth, bar unmet); "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pure_hold_loses_every_tier_and_seed(level, seed):
    """Pure harvest, never build → bldg_total stays at 4 → bar unmet
    → timeout LOSS. The cash hoards but the growth axis is the bar."""
    _, r = _run(level, _pure_hold, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: pure-hold must LOSE (no buildings built); "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_over_spend_loses_on_medium_and_hard(level, seed):
    """Chain proc + 2× pbox → cash drains to 0; income cannot refill
    above the reserve bar inside the tight medium/hard tick budget →
    timeout LOSS. (Easy is intentionally loose — over-spend permitted
    there; the discrimination tier is medium/hard.)"""
    _, r = _run(level, _over_spend, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: over-spend must LOSE on tight tier; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_wins_every_tier_and_seed(level, seed):
    """The intended capability — build cheap (3× powr = $900) while
    harvesting — WINS every tier and every hard seed. Reaches
    bldg_total 7 at cash ~600 (above the medium 300 and hard 500
    reserve) well inside the deadline."""
    _, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: intended cash-discipline should WIN; "
        f"got {r.outcome} cash={r.signals.cash} turns={r.turns}"
    )


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same policy → identical outcome / cash / turn count."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome, b.turns, b.signals.cash
    )
