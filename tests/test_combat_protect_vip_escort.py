"""combat-protect-vip-escort — escort a fragile VIP through hazards.

The bar (binding, applied seeds 1..4 on every level):
- intended escort-clears-path policy WINS;
- stall / VIP-alone / VIP-leads-charge LOSE on every level + every
  hard seed;
- non-win is a real reachable timeout / VIP-died LOSS (no DRAW
  degeneracy: after_ticks 4501 ≤ 93 + 90·(max_turns − 1));
- hard ships ≥2 `spawn_point` groups (seed-driven start variation).

The route threat is a `guard` bot interception force built around
the Soviet heavy tank (3tnk): easy/medium carry ONE full-HP 3tnk
on the route axis, hard carries TWO at health:55 — one aligned to
each seed-chosen staging lane — plus e1 rifle filler. The VIP
itself starts fragile at health:70: a harv crossing the hot route
in the column eats live heavy fire (engine move-fire fix) and
dies. The escort must destroy the interceptors AHEAD of the VIP
and hold the VIP back until the route is clear (cold), then sprint
it across.

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
PACK_PATH = PACKS / "combat-protect-vip-escort.yaml"


# ── declarative / schema invariants (no engine needed) ─────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-protect-vip-escort"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    # Spec-mandated anchors: SC2 VIP-survival, military VIP doctrine,
    # diplomatic convoy escort, fragile-target preservation.
    for needle in ("sc2 vip", "vip protection",
                   "diplomatic", "fragile-target"):
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


def _harv_at(x=100, y=20):
    return {"cell_x": x, "cell_y": y, "type": "harv", "id": "9001"}


def _tank_at(i, x=100, y=20):
    return {"cell_x": x, "cell_y": y, "type": "2tnk", "id": str(2000 + i)}


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_win_requires_harv_in_east_region_and_alive(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    # VIP in region, alive, in time → WIN.
    assert evaluate(
        c.win_condition,
        _ctx(units=[_harv_at(100, 20), _tank_at(0, 99, 20)], tick=2000),
    )
    # Tanks arrive but VIP didn't → not a WIN (VIP missing).
    assert not evaluate(
        c.win_condition,
        _ctx(units=[_tank_at(0, 100, 20), _tank_at(1, 100, 21)], tick=2000),
    )
    # VIP elsewhere on the map → not a WIN (not in region).
    assert not evaluate(
        c.win_condition,
        _ctx(units=[_harv_at(50, 20)], tick=2000),
    )
    # VIP in region but past the deadline → not a WIN.
    assert not evaluate(
        c.win_condition,
        _ctx(units=[_harv_at(100, 20)], tick=4600),
    )


@pytest.mark.parametrize("lvl", ["easy", "medium", "hard"])
def test_fail_clauses_are_reachable_losses(lvl):
    c = compile_level(load_pack(PACK_PATH), lvl)
    # Past deadline → fail (the VIP is still in the west).
    assert evaluate(
        c.fail_condition,
        _ctx(units=[_harv_at(8, 20)], tick=4600),
    )
    # VIP killed (no harv left, even if escorts survive) → fail.
    assert evaluate(
        c.fail_condition,
        _ctx(units=[_tank_at(0, 50, 20)], tick=2000),
    )
    # Force wipe (no harv among the wipe) → fail.
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000))


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
    """Hard curation: ≥2 distinct agent spawn_point groups so the seed
    round-robins the staging row (NORTH y≈14 / SOUTH y≈26). Engine-
    roundtrip asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_exactly_one_harv_per_spawn_group():
    """The VIP is a UNIQUE unit type — exactly one harv per spawn
    group (so `unit_type_count_gte:{harv,1}` collapses to a single
    "VIP alive" check; multiple harvs would let the win predicate
    accept partial loss)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        harvs = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "harv"
        ]
        assert len(harvs) == 1, f"{lvl}: must have exactly 1 harv VIP, got {len(harvs)}"
    c = compile_level(pack, "hard")
    per_group = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "harv":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group[sp] = per_group.get(sp, 0) + 1
    assert all(n == 1 for n in per_group.values()), (
        f"hard: each spawn group should have exactly 1 harv, got {per_group}"
    )


def test_escort_is_four_medium_tanks_per_group():
    """The escort is 4× 2tnk per spawn group (sized so the escort can
    actually clear the route's rifle walls in budget — fewer tanks
    and the route doesn't clean in time)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        tanks = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "2tnk"]
        assert len(tanks) == 4, f"{lvl}: expected 4 escort tanks, got {len(tanks)}"
    c = compile_level(pack, "hard")
    per_group = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "2tnk":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group[sp] = per_group.get(sp, 0) + 1
    assert all(n == 4 for n in per_group.values()), (
        f"hard: each spawn group should have 4 escort tanks, got {per_group}"
    )


def test_route_threat_is_a_guard_bot():
    """The route interception force is a `guard` bot — it holds post
    but lunges at the nearest foe within GUARD_AGGRO. A static
    defender (omitted/Defend stance) would let the heavily-armoured
    harv simply sprint past untouched; the mobile guard interceptor
    is what runs down an unescorted VIP."""
    pack = load_pack(PACK_PATH)
    enemy = (
        pack.base.get("enemy")
        if isinstance(pack.base, dict)
        else getattr(pack.base, "enemy", {})
    ) or {}
    assert enemy.get("bot_type") == "guard", (
        f"enemy must be a `guard` bot interception force; "
        f"got bot_type={enemy.get('bot_type')!r}"
    )


def test_guard_heavy_interceptor_count_and_health():
    """The load-bearing anti-VIP threat is the guard Soviet heavy
    tank (3tnk).

    easy / medium carry exactly ONE full-HP 3tnk on the y=20 route
    axis — the harv's lane — so a harv-led column crossing eats its
    focus and the 70%-HP VIP dies, while a 4× 2tnk escort can still
    focus that one heavy down.

    hard carries TWO 3tnk — one aligned to each seed-chosen staging
    lane — because post the engine no-teleport attack-chase fix a
    single y=20-axis heavy can no longer catch a harv-led column
    that funnels off-axis. Both hard heavies are at health:55 so the
    4-tank escort can still clear BOTH within budget; two FULL-HP
    heavies would out-trade the escort and break intended-WINS."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        heavies = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "3tnk"
        ]
        assert len(heavies) == 1, (
            f"{lvl}: must have exactly ONE guard 3tnk interceptor; "
            f"got {len(heavies)}"
        )
        assert heavies[0].health == 100, (
            f"{lvl}: the single route 3tnk is a full-HP heavy; "
            f"got health={heavies[0].health}"
        )
    c = compile_level(pack, "hard")
    heavies = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.type == "3tnk"
    ]
    assert len(heavies) == 2, (
        f"hard: must have TWO lane-aligned guard 3tnk interceptors; "
        f"got {len(heavies)}"
    )
    for h in heavies:
        assert h.health == 55, (
            f"hard: each lane 3tnk must be at health:55 so the escort "
            f"can clear both in budget; got health={h.health}"
        )


def test_vip_starts_fragile():
    """The VIP starts at health:70 — a genuinely fragile principal.
    Post the engine move-fire fix this is the load-bearing lever: a
    harv crossing the hot route WITH the column eats live heavy fire
    in motion and a 70%-HP harv does not survive it. The intended
    escort holds the VIP back so it only ever crosses a cleared cold
    route — arriving untouched at 70%, alive."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        harvs = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "harv"
        ]
        assert harvs, f"{lvl}: no VIP harv found"
        for h in harvs:
            assert h.health == 70, (
                f"{lvl}: the VIP must start fragile at health:70; "
                f"got health={h.health}"
            )


def test_interrupts_disabled_so_time_budget_is_real():
    """enemy_unit_spotted via step_until_event advances only ~5 ticks
    per interrupt, so on a route with 10+ enemies the per-turn tick
    rate collapses and even the intended escort cannot reach the
    extraction inside the 4500-tick budget. With interrupts disabled,
    each turn advances ~90 ticks uniformly and the budget is real."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        ints = c.scenario.interrupts or {}
        enabled = [k for k, v in ints.items() if v]
        assert not enabled, (
            f"{lvl}: interrupts must be disabled (the route has 10+ "
            f"enemies; step_until_event would collapse the tick "
            f"budget); got enabled={enabled}"
        )


# ── engine-driven scripted policies ────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary", []) or []


def _harv(rs):
    for u in _own_units(rs):
        if str(u.get("type", "")).lower() == "harv":
            return u
    return None


def _escorts(rs):
    return [u for u in _own_units(rs)
            if str(u.get("type", "")).lower() == "2tnk"]


def _stall(rs, Command):
    """Pure observe — no unit moves; VIP never reaches the
    extraction → after_ticks LOSS."""
    return [Command.observe()]


def _vip_alone(rs, Command):
    """Only the harvester is moved east; escorts stay idle in the
    west. The unarmed VIP enters the route unaccompanied → the
    guard heavy tank lunges at it and runs it down → `not
    unit_type_count_gte:{harv,1}` fail clause fires → LOSS."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        if str(u.get("type", "")).lower() == "harv":
            cmds.append(Command.move_units([str(u["id"])], 100, 20))
        else:
            cmds.append(Command.stop([str(u["id"])]))
    return cmds


def _vip_leads_charge(rs, Command):
    """Every unit (harv + tanks) charges east on the same y-band
    with the harv at the front. The guard interceptor + the e1
    corridor focus the leading harvester and kill it before the
    tanks finish clearing the route → LOSS."""
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        cmds.append(Command.move_units([str(u["id"])], 100, 20))
    return cmds


def _killable_enemies(rs):
    """Live, non-building, non-marker enemies (the interception
    force the escort must clear)."""
    return [
        e for e in (rs.get("enemy_summary") or [])
        if not e.get("is_building")
        and str(e.get("type", "")).lower() != "fact"
    ]


def _intended_escort_clears_path(rs, Command):
    """Escort-clears-path:

      1. The escort tanks attack the interception force AHEAD of
         the VIP — each tank focus-fires (attack_unit) its nearest
         live interceptor; once the route is clear they attack_move
         on to the extraction.
      2. The VIP HOLDS at the staging zone — it does NOT move while
         any interceptor is alive OR the escort has not yet reached
         the extraction latitude (rear-most tank past x≥82).
      3. Once the route is confirmed clear (no live interceptors)
         and the escort has pushed through, the VIP sprints across
         to (100,20) uncontested.
    """
    units = _own_units(rs)
    if not units:
        return [Command.observe()]
    h = _harv(rs)
    tanks = _escorts(rs)
    enemies = _killable_enemies(rs)
    cmds = []

    # Escort tanks: focus-fire the nearest interceptor; sweep to
    # the extraction once nothing is left.
    for t in tanks:
        if enemies:
            enemies.sort(
                key=lambda e: (e["cell_x"] - t["cell_x"]) ** 2
                + (e["cell_y"] - t["cell_y"]) ** 2
            )
            cmds.append(
                Command.attack_unit([str(t["id"])], str(enemies[0]["id"]))
            )
        else:
            cmds.append(Command.attack_move([str(t["id"])], 100, 20))

    if h is not None:
        rear_x = min((t["cell_x"] for t in tanks), default=0)
        if tanks and rear_x >= 82 and not enemies:
            # Route confirmed clear and the escort is through —
            # sprint the VIP across.
            cmds.append(Command.move_units([str(h["id"])], 100, 20))
        else:
            # Hold the VIP at the staging zone.
            cmds.append(Command.stop([str(h["id"])]))
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_escort_clears_path_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_escort_clears_path, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: escort-clears-path should WIN, got "
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
        f"(VIP never reaches extraction → after_ticks fires), "
        f"got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_vip_alone_loses(level, seed):
    """VIP sent alone enters the route unaccompanied; the guard
    heavy tank lunges at the lone harvester and destroys it →
    `not unit_type_count_gte:{harv,1}` fail clause fires → LOSS
    on every level and every seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _vip_alone, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: VIP-alone must LOSE (unarmed harv "
        f"dies in the first wall), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_vip_leads_charge_loses(level, seed):
    """Harv + tanks charge in a single column with the harv at the
    front; the guard interceptor + the e1 corridor focus the
    leading harvester → VIP dies → LOSS on every level and every
    seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _vip_leads_charge, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: VIP-leads-charge must LOSE (harv at "
        f"the front of the column is run down by the interceptors), "
        f"got {r.outcome} (losses={r.signals.units_lost})"
    )
