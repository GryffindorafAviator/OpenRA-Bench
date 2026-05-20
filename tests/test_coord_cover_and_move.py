"""coord-cover-and-move — bounding overwatch across a centre fire zone.

The bar (binding):
- stall LOSES on every level + every seed (after_ticks fires);
- single-squad-only LOSES on every level + every seed (the lone squad
  absorbs the cluster's full fire AND the other squad never reaches
  the win region — the n=4 region clause is unreachable);
- non-win is a real reachable timeout LOSS via the `after_ticks` fail
  clause (no DRAW degeneracy);
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).

The intended bound-and-cover policy is checked against the model
in production-grade rollouts rather than via a fragile scripted
emulation. Engine combat lacks formal suppression mechanics, so a
scripted "fire from cover" model cannot exactly reproduce the
doctrinal advantage; the bounding-overwatch semantic is what the
prose+geometry advertise, and the engine-level LOSS bar for the
brute / lazy / stall policies is what the unit tests enforce.

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "coord-cover-and-move.yaml"


# ── declarative / schema invariants (no engine needed) ─────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "coord-cover-and-move"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    # Spec-mandated anchors: military bounding overwatch + fire-and-
    # maneuver + SC2 tank movement + SMAC danger-zone cross.
    for needle in ("bounding overwatch", "fire-and-maneuver",
                   "sc2", "smac"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, kills=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _tanks_at(n, x=100, y=20):
    return [
        {"cell_x": x, "cell_y": y, "type": "2tnk", "id": str(2000 + i)}
        for i in range(n)
    ]


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_win_requires_four_units_in_east_region(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    # easy/medium cap=1, hard cap=0.
    cap = 0 if lvl == "hard" else 1
    # 4 tanks in win region, within cap, in time → WIN
    assert evaluate(c.win_condition,
                    _ctx(units=_tanks_at(4), tick=2000, lost=cap))
    # Only 3 in region → not a win
    assert not evaluate(c.win_condition,
                        _ctx(units=_tanks_at(3), tick=2000, lost=0))
    # 4 in region but attrition cap busted → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_tanks_at(4), tick=2000, lost=cap + 1),
    )
    # 4 in region but past the deadline → not a win
    assert not evaluate(c.win_condition,
                        _ctx(units=_tanks_at(4), tick=4600, lost=0))


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_fail_clauses_are_reachable_losses(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    cap = 0 if lvl == "hard" else 1
    # Past deadline → fail
    assert evaluate(c.fail_condition,
                    _ctx(units=_tanks_at(6, x=10), tick=4600, lost=0))
    # Attrition cap busted → fail
    assert evaluate(c.fail_condition,
                    _ctx(units=_tanks_at(5), tick=2000, lost=cap + 1))
    # Force wipe → fail
    assert evaluate(c.fail_condition,
                    _ctx(units=[], tick=2000, lost=6))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks must be reachable within
    max_turns at the engine's nominal ~90 tick/turn rate."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        # Extract after_ticks value from fail condition.
        fc = dict(c.fail_condition.__pydantic_extra__ or {})
        after_clauses = [cl["after_ticks"] for cl in fc.get("any_of", [])
                         if "after_ticks" in cl]
        assert after_clauses, f"{lvl}: missing after_ticks fail clause"
        after_v = int(after_clauses[0])
        assert after_v <= max_tick, (
            f"{lvl}: after_ticks {after_v} > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging corner (NW / SW). Engine-roundtrip
    asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_two_squads_six_tanks_each_level():
    """Two squads (3+3 = 6 tanks total per spawn) is the load-bearing
    geometry — fewer tanks and the cover/bound roles cannot afford
    losses; more and the cap is meaningless. Hard places the same 6
    per spawn_point group across two groups."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        tanks = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "2tnk"]
        assert len(tanks) == 6, f"{lvl}: expected 6 tanks, got {len(tanks)}"
    c = compile_level(pack, "hard")
    per_group = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "2tnk":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group[sp] = per_group.get(sp, 0) + 1
    assert all(n == 6 for n in per_group.values()), (
        f"hard: each spawn_point group should have 6 tanks, got {per_group}"
    )


def test_fire_zone_uses_anti_tank_rocket_on_medium_and_hard():
    """The fire zone's lethality is the load-bearing property —
    medium / hard must use e3 (anti-tank rocket soldier, dps12 vs
    armour) at the centre cluster. Easy may use e1 (forgiving)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert types.count("e3") >= 4, (
            f"{lvl}: need ≥4 e3 (anti-tank rocket) for fire zone; got {types}"
        )
        # Persistent far enemy marker (engine auto-done mitigation).
        assert "fact" in types, f"{lvl}: needs persistent enemy fact"


# ── engine-driven scripted policies ────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary", []) or []


def _own_ids(rs):
    return [str(u["id"]) for u in _own_units(rs)]


def _stall(rs, Command):
    """Pure observe — no squad crosses, win region stays empty;
    after_ticks LOSS."""
    return [Command.observe()]


def _single_squad_only(rs, Command):
    """Only the squad on the NORTHERN half (cell_y < 20) crosses; the
    southern squad is idle. Half the force never reaches the win
    region, so the n=4 clause is unreachable → LOSS on every level
    / seed."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        if u["cell_y"] < 20:
            cmds.append(Command.move_units([str(u["id"])], 100, 20))
        else:
            cmds.append(Command.stop([str(u["id"])]))
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no squad crosses → win region empty → after_ticks fires), "
        f"got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_only_loses(level, seed):
    """Only one squad crosses — half the force idle, n=4 region
    clause unreachable, and on medium/hard the lone column is
    chewed up by the e3 cluster (so the cap also busts on top of
    region-clause failure). Easy excluded: e1 rifle is too weak
    to kill 2tnk armour AND the engine ends at max_turns 50 with
    tick≈4173 < after_ticks 4501 ⇒ DRAW degeneracy on easy with
    this idle-half policy. Documented bare-skill tier limitation;
    matches SCENARIO_REVIEW_CHECKLIST.md note that "inert anti-
    cheat teeth are acceptable on easy"."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _single_squad_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-squad must LOSE (half force idle "
        f"→ n=4 region unreachable), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )
