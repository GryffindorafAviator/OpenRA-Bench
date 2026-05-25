"""coord-cover-and-move — bounding overwatch across a centre fire zone.

The bar (binding):
- stall LOSES on every level + every seed (after_ticks timeout);
- both-charge-through-the-centre LOSES on every level + every seed
  (the rocket fire + heavy-tank brawl bust the attrition cap);
- single-squad-only LOSES on every level + every seed (half the force
  idle → the n=4 region clause is unreachable);
- the intended bound-and-cover (periphery route around the fire zone)
  WINS on every level + every seed (the pack is solvable);
- non-win is a real reachable timeout LOSS via the `after_ticks` fail
  clause (no DRAW degeneracy);
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).

Recalibrated after the engine movement fixes ((A) attack_unit on
out-of-sight targets paths normally — no teleport; (B) a moving unit
fires AND takes fire en route — no sprint-invincibility). A 6-tank
column that attack_moves through the centre now genuinely trades fire
while crossing. Probing showed the e1 rifle cluster on the old easy
tier could no longer punish the charge at all (0 losses): the
load-bearing punisher is e3 ANTI-TANK ROCKET soldiers (the real
anti-armour DPS) PLUS two 4tnk HEAVY-TANK anchors — e3 rockets alone,
or a single 4tnk, only cost a charge ONE tank; four e3 plus two 4tnk
cost it TWO. Every level now fields an e3 rocket cluster anchored by
TWO 4tnk (easy a smaller 3× e3 cluster, medium/hard a 4× e3 cluster,
hard adds e1 rifle screens). All enemy units are stance:2 Defend
(auto-fire in range, STATIONARY) so a staller is a clean timeout
LOSS, not chased down by hunters.

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
    # easy cap=0 (tightened post engine PR #15 — vendor HP lets a
    # bound-and-cover policy take zero losses, and the in-lane e3
    # screens make the lone-column single-squad lose ≥1 tank;
    # cap=1 would degenerate single-squad into a DRAW); medium
    # cap=1 (middle tier); hard cap=0 (the tightest bar).
    cap = 0 if lvl in ("easy", "hard") else 1
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


def test_fire_zone_uses_anti_tank_rocket_every_level():
    """The fire zone's lethality is the load-bearing property — post
    engine movement fixes the e3 (anti-tank rocket soldier) cluster is
    the real anti-armour DPS, so EVERY level uses e3 at the centre.
    Easy fields a smaller cluster (≥3 e3, the forgiving tier);
    medium / hard a denser one (≥4 e3)."""
    pack = load_pack(PACK_PATH)
    for lvl, want_e3 in (("easy", 3), ("medium", 4), ("hard", 4)):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert types.count("e3") >= want_e3, (
            f"{lvl}: need ≥{want_e3} e3 (anti-tank rocket) for fire "
            f"zone; got {types}"
        )
        # Persistent far enemy marker (engine auto-done mitigation).
        assert "fact" in types, f"{lvl}: needs persistent enemy fact"


def test_fire_zone_anchored_by_two_heavy_tanks():
    """Recalibration invariant: probing showed the e3 rockets alone or
    a single 4tnk only cost a 6-tank charge ONE tank — it takes TWO
    4tnk anchors alongside the rocket cluster to reliably cost the
    charge TWO. Every level therefore anchors the fire zone with two
    enemy 4tnk heavy tanks."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        n4tnk = sum(
            1 for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "4tnk"
        )
        assert n4tnk >= 2, (
            f"{lvl}: fire zone must be anchored by ≥2 enemy 4tnk "
            f"(heavy-tank brawl), got {n4tnk}"
        )


def test_enemy_fire_zone_is_stationary_defend():
    """All enemy fire-zone units are stance:2 Defend (auto-fire in
    range but never advance) so a staller is a clean reachable
    timeout LOSS rather than being hunted down — and the periphery
    bounding lane stays open (a stance:3 hunter would chase the
    bounding squad off its route)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            if a.owner != "enemy" or a.type == "fact":
                continue
            assert a.stance == 2, (
                f"{lvl}: enemy fire-zone actor {a.type} at {a.position} "
                f"must be stance:2 Defend, got stance={a.stance}"
            )


# ── engine-driven scripted policies ────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary", []) or []


def _own_ids(rs):
    return [str(u["id"]) for u in _own_units(rs)]


def _stall(rs, Command):
    """Pure observe — no squad crosses, win region stays empty;
    after_ticks LOSS."""
    return [Command.observe()]


def _both_charge(rs, Command):
    """Brute: all 6 tanks attack_move straight through the centre of
    the map toward the eastern win region. The column drives into the
    centre fire zone (the heavy-tank brawl + rocket fire) and busts
    the attrition cap → LOSS on every level / seed."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    return [Command.attack_move(_own_ids(rs), 100, 20)]


def _single_squad_only(rs, Command):
    """Wrong-path: only ONE squad crosses; the other is idle. The
    crossing squad is the UPPER HALF of the force by the agent's own
    median latitude (spawn-aware so it is genuinely half the force
    whichever spawn group the seed selects). Half the force never
    reaches the win region, so the n=4 region clause is unreachable
    → LOSS on every level / seed."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    ys = sorted(u["cell_y"] for u in units)
    mid = ys[len(ys) // 2]
    cmds = []
    for u in units:
        if u["cell_y"] < mid:
            cmds.append(Command.move_units([str(u["id"])], 100, 20))
        else:
            cmds.append(Command.stop([str(u["id"])]))
    return cmds


def _bound_and_cover(rs, Command):
    """Intended bounding-overwatch proxy: route every tank through the
    FAR periphery (rise to y≈6 north / y≈34 south WEST of the fire
    zone, traverse the periphery, then converge on the eastern win
    region). The squads cross OUTSIDE the centre fire zone entirely
    and take zero losses → WIN on every level / seed. This confirms
    the pack is solvable; the doctrinal cover/move role-alternation
    is what the prose + geometry advertise."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        uid = str(u["id"])
        ux, uy = u["cell_x"], u["cell_y"]
        peri_y = 6 if uy < 20 else 34
        if ux < 30:
            # rise to the periphery WEST of the fire zone longitude
            cmds.append(Command.move_units([uid], 35, peri_y))
        elif ux < 80:
            # traverse the periphery past the fire zone
            cmds.append(Command.move_units([uid], 85, peri_y))
        else:
            # converge on the eastern win region
            cmds.append(Command.move_units([uid], 100, 20))
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


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_both_charge_loses(level, seed):
    """Brute both-charge through the centre fire zone must LOSE on
    every level / seed: the column drives into the heavy-tank brawl
    and busts the attrition cap (≥2 lost easy/medium, ≥1 lost hard)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _both_charge, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: both-charge through the centre must LOSE "
        f"(heavy-tank brawl busts the attrition cap), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_only_loses(level, seed):
    """Only one squad crosses — half the force idle, n=4 region
    clause unreachable; the lone column also takes losses crossing
    the centre fire zone. LOSS on every level / seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _single_squad_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-squad must LOSE (half force idle "
        f"→ n=4 region unreachable), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_bound_and_cover_wins(level, seed):
    """The intended bounding-overwatch route (cross the fire zone via
    the far periphery, not through the centre) must WIN on every level
    / seed — confirms the pack is solvable and the no-cheat bar is not
    achieved by making the scenario unwinnable."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _bound_and_cover, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended bound-and-cover (periphery "
        f"route) must WIN, got {r.outcome} after {r.turns} turns "
        f"(losses={r.signals.units_lost})"
    )
