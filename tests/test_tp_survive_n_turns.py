"""tp-survive-n-turns — REASONING pure-survival pack.

The pack tests PURE SURVIVAL under sustained attrition (SC2 survival /
military hold-the-line / sustained-ops anchor): the agent holds a
fixed force (6× 2tnk + fact + powr, no economy tools) and must keep
≥N of those units AND its construction yard (`fact`) ALIVE until the
survival tick T elapses. Relentless rusher waves arrive throughout —
an opening wave at t=0 plus scheduled-event reinforcement waves.
There is NO offensive objective.

Bar (binding):
- intended HOLD-FOCUS (keep the 6 tanks clustered on the fact, focus-
  fire each arriving wave, never chase) WINS on every level + every
  hard seed (1..4);
- STALL (observe-only) LOSES on every level + every seed — spread
  Defend-stance fire is too slow, the sustained waves raze the fact;
- AGGRESSIVE-CHARGE (attack_move the column at the rushers) LOSES on
  every level + every seed — the column is ground down piecemeal and
  own_units drops below N;
- non-win is a real reachable LOSS (the fail tree fires on a razed
  fact / an own-force below N, and the broad after_ticks ≤
  93 + 90·(max_turns − 1) catches a limp to the deadline) — never a
  DRAW;
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.eval_core import run_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK = PACKS / "tp-survive-n-turns.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)
N = 4  # the own_units floor (lose at most 2 of the 6 starting tanks)


# ── 1) declarative / schema invariants (no engine needed) ──────────────


def test_pack_loads_and_three_levels_compile():
    p = load_pack(PACK)
    assert p.meta.id == "tp-survive-n-turns"
    assert p.meta.capability == "reasoning"
    anchors = " | ".join(p.meta.benchmark_anchor)
    for needed in ("SC2 survival", "military hold-the-line", "sustained ops"):
        assert needed in anchors, f"benchmark_anchor missing {needed!r}: {anchors}"
    for lv in LEVELS:
        c = compile_level(p, lv)
        assert c.map_supported, f"{lv}: rush-hour-arena must be Rust-loadable"


def test_win_is_survival_band_fact_and_units():
    """Win = `all_of` of building_count_gte(fact) + own_units_gte(N) +
    [after_ticks T, within_ticks Tw]. The `after_ticks` clause is the
    load-bearing SURVIVAL GATE — without it the win (fact + units)
    would be satisfied on turn 1 and a staller would win for free."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        assert "all_of" in wc, f"{lv}: win must be `all_of`"
        clauses = wc["all_of"]
        keys = set()
        for cl in clauses:
            keys |= set(cl)
        assert "building_count_gte" in keys, f"{lv}: win must require the fact"
        assert "own_units_gte" in keys, f"{lv}: win must require ≥N own units"
        assert "after_ticks" in keys, (
            f"{lv}: win must have the after_ticks survival gate "
            f"(else a turn-1 staller wins for free)"
        )
        assert "within_ticks" in keys, f"{lv}: win must close with within_ticks"
        fact_cl = next(c for c in clauses if "building_count_gte" in c)
        assert str(fact_cl["building_count_gte"]["type"]).lower() == "fact"
        units_cl = next(c for c in clauses if "own_units_gte" in c)
        assert int(units_cl["own_units_gte"]) == N, f"{lv}: own_units floor must be {N}"


@pytest.mark.parametrize("lv", LEVELS)
def test_survival_gate_and_fail_reachable_no_draw(lv):
    """The after_ticks survival gate (win) and the fail after_ticks
    must both be reachable within `max_turns` (tick ≤
    93 + 90·(max_turns − 1)); the fail after_ticks = within_ticks+1."""
    p = load_pack(PACK)
    c = compile_level(p, lv)
    ceiling = 93 + 90 * (c.max_turns - 1)
    wc = dict(c.win_condition.__pydantic_extra__ or {})
    aft_win = next(cl["after_ticks"] for cl in wc["all_of"] if "after_ticks" in cl)
    wt = next(cl["within_ticks"] for cl in wc["all_of"] if "within_ticks" in cl)
    assert aft_win < wt < ceiling, (
        f"{lv}: need after_ticks {aft_win} < within_ticks {wt} < ceiling {ceiling}"
    )
    fc = dict(c.fail_condition.__pydantic_extra__ or {})
    aft_fail = next(cl["after_ticks"] for cl in fc["any_of"] if "after_ticks" in cl)
    assert aft_fail <= ceiling, (
        f"{lv}: fail after_ticks {aft_fail} > ceiling {ceiling} ⇒ DRAW"
    )
    assert aft_fail == wt + 1, f"{lv}: fail after_ticks should be within_ticks+1"


def test_fail_has_timeout_fact_and_units_clauses():
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        keys: set[str] = set()
        for cl in fc["any_of"]:
            keys |= set(cl)
            if "not" in cl:
                keys |= set(cl["not"])
        assert "after_ticks" in keys, f"{lv}: fail must include the timeout clause"
        assert "building_count_gte" in keys, f"{lv}: fail must catch a razed fact"
        assert "own_units_gte" in keys, f"{lv}: fail must catch an own-force below N"


def test_survival_window_grows_easy_to_medium_to_hard():
    """Each tier holds longer than the previous (easy < medium < hard)."""
    p = load_pack(PACK)
    gates: dict[str, int] = {}
    for lv in LEVELS:
        c = compile_level(p, lv)
        wc = dict(c.win_condition.__pydantic_extra__ or {})
        gates[lv] = next(
            cl["after_ticks"] for cl in wc["all_of"] if "after_ticks" in cl
        )
    assert gates["easy"] < gates["medium"] < gates["hard"], (
        f"survival window must lengthen each tier: {gates}"
    )


def test_enemy_is_rusher_bot():
    """`rusher` concentrates the whole band on the agent mass — the
    canonical sustained-attrition opponent for a survival hold."""
    p = load_pack(PACK)
    assert p.base.get("enemy", {}).get("bot_type") == "rusher"


def test_fixed_force_six_tanks_no_economy_tools():
    """The agent fields exactly 6× 2tnk + fact + powr per spawn group
    and has NO build/harvest tools — this is a pure positional hold."""
    p = load_pack(PACK)
    tools = set(p.base.get("tools") or [])
    assert not (tools & {"build", "place_building", "harvest", "sell"}), (
        f"pure-survival pack must not grant economy tools; got {sorted(tools)}"
    )
    for lv in LEVELS:
        c = compile_level(p, lv)
        agent = [a for a in c.scenario.actors if a.owner == "agent"]
        # group actors by spawn_point (None ⇒ group 0)
        groups: dict[int, list] = {}
        for a in agent:
            groups.setdefault(a.spawn_point or 0, []).append(a)
        for g, actors in groups.items():
            tanks = [a for a in actors if a.type == "2tnk"]
            facts = [a for a in actors if a.type == "fact"]
            powrs = [a for a in actors if a.type == "powr"]
            assert len(tanks) == 6, f"{lv} group {g}: need 6 tanks, got {len(tanks)}"
            assert len(facts) == 1, f"{lv} group {g}: need 1 fact"
            assert len(powrs) == 1, f"{lv} group {g}: need 1 powr"


def test_scheduled_reinforcement_waves_present():
    """Sustained pressure: each level injects ≥2 scheduled
    `spawn_actors` reinforcement waves on top of the opening wave."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        sched = c.scheduled_events
        spawns = [e for e in sched if e.get("type") == "spawn_actors"]
        assert len(spawns) >= 2, (
            f"{lv}: need ≥2 scheduled reinforcement waves; got {len(spawns)}"
        )
        # `tick`/`type` must precede `actors` (engine parser reads order)
        for e in spawns:
            ks = list(e.keys())
            assert ks.index("tick") < ks.index("actors"), (
                f"{lv}: scheduled-event keys must list tick before actors"
            )


def test_persistent_inert_enemy_fact_marker():
    """An unarmed enemy `fact` far east keeps the episode alive past
    the last rusher death so the after_ticks survival gate is
    evaluated (anti-DRAW)."""
    p = load_pack(PACK)
    for lv in LEVELS:
        c = compile_level(p, lv)
        far = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact" and a.position[0] >= 100
        ]
        assert far, f"{lv}: missing the far-east anti-DRAW enemy fact marker"


def test_hard_has_multiple_spawn_point_groups():
    p = load_pack(PACK)
    c = compile_level(p, "hard")
    sp = {
        a.spawn_point if a.spawn_point is not None else 0
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sorted(sp)}"


# ── 2) engine-required scripted-policy discrimination sweep ────────────


def _enemy_xy(e):
    return e.get("cell_x", e.get("x", 999)), e.get("cell_y", e.get("y", 999))


def _stall(rs, C):
    """Observe-only — must LOSE (spread Defend fire too slow)."""
    return [C.observe()]


def _hold_focus(rs, C):
    """Intended capability: keep the 6 tanks clustered on the fact and
    focus-fire the enemy nearest the force centroid — concentrate all
    guns so each wave is finished fast. Must WIN."""
    us = rs.get("units_summary") or []
    if not us:
        return [C.observe()]
    ids = [str(u["id"]) for u in us]
    en = rs.get("enemy_positions") or []
    if not en:
        return [C.stop(ids)]
    cx = sum(u["cell_x"] for u in us) / len(us)
    cy = sum(u["cell_y"] for u in us) / len(us)
    tgt = min(en, key=lambda e: (_enemy_xy(e)[0] - cx) ** 2
              + (_enemy_xy(e)[1] - cy) ** 2)
    tid = tgt.get("id")
    if tid is None:
        return [C.stop(ids)]
    return [C.attack_unit(ids, str(tid))]


def _charge(rs, C):
    """Aggressive: drive the column at the rushers (the trap). Must
    LOSE — met in the open and ground down piecemeal."""
    us = rs.get("units_summary") or []
    if not us:
        return [C.observe()]
    ids = [str(u["id"]) for u in us]
    en = rs.get("enemy_positions") or []
    if not en:
        return [C.attack_move(ids, target_x=64, target_y=20)]
    ex, ey = _enemy_xy(en[0])
    return [C.attack_move(ids, target_x=ex, target_y=ey)]


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_hold_focus_wins(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _hold_focus, seed=seed)
    assert res.outcome == "win", (
        f"{lv} seed{seed}: intended hold-focus must WIN, got {res.outcome} "
        f"(units_lost={res.signals.units_lost})"
    )


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{lv} seed{seed}: stall must LOSE (real reachable LOSS), "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("lv", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_aggressive_charge_loses(lv, seed):
    c = compile_level(load_pack(PACK), lv)
    res = run_level(c, _charge, seed=seed)
    assert res.outcome == "loss", (
        f"{lv} seed{seed}: aggressive-charge must LOSE (units below N), "
        f"got {res.outcome}"
    )
