"""coord-cover-and-move — bounding-overwatch across a centre fire zone.

The bar (binding):
- intended bound-and-cover policy WINS on every level + every hard
  seed (1..4);
- stall / both-charge-together / single-squad-only LOSE on every level
  + every hard seed (with one documented exception: easy uses e1 rifle
  infantry which does limited damage to 2tnk armour, so both-charge
  and single-squad may squeak through — bare-skill tier; matches the
  pack design comment and SCENARIO_REVIEW_CHECKLIST.md note that
  "inert anti-cheat teeth are acceptable on easy");
- non-win is a real reachable timeout LOSS via the `after_ticks`
  fail clause (no DRAW degeneracy: 4501 ≤ 93 + 90·(max_turns − 1));
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).

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
    # 4 tanks in win region, 0 lost, in time → WIN
    cap = 0 if lvl == "hard" else 1
    assert evaluate(c.win_condition,
                    _ctx(units=_tanks_at(4), tick=2000, lost=0))
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
    """No draw degeneracy: after_ticks 4501 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 4501 <= max_tick, (
            f"{lvl}: after_ticks 4501 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )
        assert 4500 <= max_tick, (
            f"{lvl}: within_ticks 4500 > max reachable tick {max_tick}"
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
    """Pure observe — no squad crosses, win region stays empty,
    after_ticks LOSS."""
    return [Command.observe()]


def _both_charge_together(rs, Command):
    """Both squads sprint east through the centre on the same y-band.
    Engine spreads fire across the dense column → on medium/hard the
    4× e3 anti-tank dps stacks enough to bust units_lost_lte:1 → LOSS."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        # Everyone barrels east toward the win region (100,20).
        cmds.append(Command.move_units([str(u["id"])], 100, 20))
    return cmds


def _single_squad_only(rs, Command):
    """Only the NORTHERN squad (y < 20 at spawn) crosses; the southern
    squad is idle. The lone column absorbs ALL the centre cluster's
    fire and busts the attrition cap → LOSS on medium/hard."""
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


def _intended_bound_and_cover(rs, Command):
    """Bounding overwatch:

      Phase 1 — cover squad (northern at spawn, by y) advances to its
        overwatch post at (~45, ~spawn_y) at the EDGE of the e3 range
        (tank rng 4.75 vs e3 rng 4 — tanks fire from dist ~5);
        bounding squad (southern) takes the FAR-NORTH wide-flank route
        outside the fire envelope: detour to (~50, ~6 or ~34) then
        east toward (~100, ~20).
      Phase 2 — bounding squad pushed past the cluster onto the east
        side; switch — original cover squad now relocates to the far
        wide-flank route, then east.

    The policy doesn't need ticks-aware role-flipping: it routes the
    SOUTH squad through y≈6 (or y≈34 if south spawn) — the wide flank
    — and the NORTH squad through y≈6 as well after a stagger. The
    cover team's fire SUPPRESSES the cluster while the bounding team
    is in transit. A simple implementation:
      - sort own units by y; northern half = COVER, southern half = MOVE
      - for the NORTH spawn (median y < 20): MOVE goes via y=8 then
        east; COVER drives to (45, 15) and attack_moves on cluster.
      - for the SOUTH spawn (median y > 20): mirror — MOVE goes via
        y=32 then east; COVER drives to (45, 25) and attack_moves.
      - once MOVE has cleared the fire zone (cell_x > 70), COVER also
        starts the wide-flank route through the same outside-sight
        band and follows east.
    """
    units = _own_units(rs)
    if not units:
        return [Command.observe()]

    # Identify spawn geometry by current median y. Squads start with
    # y ∈ {11..17} (NORTH spawn) or {23..29} (SOUTH spawn) in hard.
    # Easy/medium always use y ∈ {14..16, 24..26}.
    ys = sorted(u["cell_y"] for u in units)
    median_y = ys[len(ys) // 2]

    if median_y < 20:
        # NORTH spawn: COVER on the south-of-spawn flank near the
        # cluster's NORTH edge; MOVE detours through y ≈ 6..8.
        flank_y = 8
        cover_y = 15
    else:
        # SOUTH spawn (hard only): mirror across y=20.
        flank_y = 32
        cover_y = 25

    # Split units: the half closer to the cluster's y axis (i.e.
    # closer to y=20) becomes COVER; the half farther from y=20
    # becomes MOVE (these are the ones we'll route via the wide flank).
    units_sorted = sorted(units, key=lambda u: abs(u["cell_y"] - 20))
    half = len(units_sorted) // 2
    cover_team = units_sorted[:half]
    move_team = units_sorted[half:]

    cmds = []
    # Check whether MOVE team has cleared the fire zone (any move-team
    # tank with x ≥ 75). When cleared, COVER also begins its bound.
    move_cleared = any(u["cell_x"] >= 75 for u in move_team)

    # COVER team: drive to overwatch post and attack_move onto the
    # cluster centre (the engine auto-targets nearest hostile in
    # range; with cover at (45, cover_y) and cluster at (50, 20), the
    # e3s are the visible targets and tanks fire dps22 each).
    for u in cover_team:
        if not move_cleared:
            if u["cell_x"] < 45:
                # Still approaching the overwatch post — drive into
                # firing range of the cluster.
                cmds.append(Command.attack_move([str(u["id"])], 45, cover_y))
            else:
                # Posted: keep firing. attack_move onto cluster cell.
                cmds.append(Command.attack_move([str(u["id"])], 50, 20))
        else:
            # MOVE team has crossed; COVER bounds through the wide
            # flank too (same y-band the MOVE team used).
            if u["cell_x"] < 60:
                # Detour north/south away from the cluster first.
                cmds.append(Command.move_units([str(u["id"])], 50, flank_y))
            else:
                cmds.append(Command.move_units([str(u["id"])], 100, 20))

    # MOVE team: wide-flank route. Stage 1 detour to (50, flank_y) so
    # the column stays outside the cluster's sight envelope; stage 2
    # east toward the win region.
    for u in move_team:
        if u["cell_x"] < 60:
            cmds.append(Command.move_units([str(u["id"])], 50, flank_y))
        elif u["cell_x"] < 90:
            cmds.append(Command.move_units([str(u["id"])], 90, 20))
        else:
            cmds.append(Command.move_units([str(u["id"])], 100, 20))
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_bound_and_cover_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_bound_and_cover, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: bound-and-cover should WIN, got "
        f"{r.outcome} after {r.turns} turns "
        f"(losses={r.signals.units_lost})"
    )


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
def test_both_charge_together_loses(level, seed):
    """Both squads sprint together → fire stacks on dense column →
    busts units_lost_lte cap on medium and hard. Easy excluded
    (e1 rifle does limited damage to armour; forgiving bare-skill
    tier; matches SCENARIO_REVIEW_CHECKLIST.md note that "inert
    anti-cheat teeth are acceptable on easy")."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _both_charge_together, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: both-charge must LOSE (stacked e3 fire "
        f"busts attrition cap), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_only_loses(level, seed):
    """Only one squad crosses → lone column absorbs ALL cluster fire
    → busts attrition cap on medium/hard. Easy excluded (same
    forgiving-bare-skill reasoning as both-charge above)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _single_squad_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-squad must LOSE (lone column "
        f"absorbs full cluster fire), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )
