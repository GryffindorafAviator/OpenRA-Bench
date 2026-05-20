"""coord-diversionary-attack — diversionary / split-attack assault.

The bar: intended diversionary-attack (Squad A jeeps drag the
heavier enemy garrison off the real target by attacking the decoy /
centre cluster; Squad B tanks raze the REAL fact while the south
defenders are committed in pursuit) WINS on every level and every
hard seed (1..4). STALL (only observe), BRUTE-FRONTAL (every unit
attack-moves onto the centre / decoy), LAZY NEAREST (jeeps onto the
near-looking fact, tanks onto the near-looking powr), and BAIT-ONLY
(jeeps slash south but tanks stand still) all LOSE on every level
and every hard seed. Non-win is a real reachable timeout LOSS via
the `after_ticks` fail clause; `units_lost_lte` provides the second
LOSS path (lazy/brute trades the strike force).

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
PACK_PATH = PACKS / "coord-diversionary-attack.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "coord-diversionary-attack"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list ≥3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "diversion", "cicero"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, lost=0, destroyed_records=()):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    types_count: dict[str, int] = {}
    for t, _x, _y in destroyed_records:
        types_count[t] = types_count.get(t, 0) + 1
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        cash=0,
        resources=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        enemy_buildings_destroyed=len(destroyed_records),
        enemy_buildings_destroyed_types=types_count,
        enemy_buildings_destroyed_records=list(destroyed_records),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n, kind="2tnk"):
    return [
        {"cell_x": 50, "cell_y": 20, "type": kind, "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: fact at (100,10) destroyed (in region), 3 lost, in budget → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=3000, lost=3,
             destroyed_records=[("fact", 100, 10)]),
    )
    # Razing only the decoy powr at (100,30) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=3000, lost=1,
             destroyed_records=[("powr", 100, 30)]),
    )
    # Razing the sentinel fact at (125,38) is outside the region → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=3000, lost=1,
             destroyed_records=[("fact", 125, 38)]),
    )
    # Loss cap (3) tripped (4 lost) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(2), tick=3000, lost=4,
             destroyed_records=[("fact", 100, 10)]),
    )
    # Past deadline → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=5401, lost=1,
             destroyed_records=[("fact", 100, 10)]),
    )
    # Timeout with bar unmet → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(4), tick=5500, lost=1, destroyed_records=[]),
    )
    # Force-wipe → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=3000, lost=7, destroyed_records=[]),
    )
    # Loss cap (>3) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(1), tick=3000, lost=4,
             destroyed_records=[("fact", 100, 10)]),
    )


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: fact destroyed, 3 lost (3 bait jeeps), within budget → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=4000, lost=3,
             destroyed_records=[("fact", 100, 10)]),
    )
    # 4 lost (one tank lost) → not a win, AND fail
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=4000, lost=4,
             destroyed_records=[("fact", 100, 10)]),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=4000, lost=4,
             destroyed_records=[("fact", 100, 10)]),
    )
    # Only decoy razed → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=4000, lost=1,
             destroyed_records=[("powr", 100, 30)]),
    )


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: NE fact destroyed → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=4000, lost=2,
             destroyed_records=[("fact", 100, 10)]),
    )
    # Intended (mirror): SE fact destroyed → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=4000, lost=2,
             destroyed_records=[("fact", 100, 30)]),
    )
    # Only decoy powr at centre → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(4), tick=4000, lost=1,
             destroyed_records=[("powr", 100, 20)]),
    )
    # 3 lost → not a win, AND fail (cap is 2 on hard)
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=4000, lost=3,
             destroyed_records=[("fact", 100, 10)]),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=4000, lost=3,
             destroyed_records=[("fact", 100, 10)]),
    )


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 5401 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 5401 <= max_tick, (
            f"{lvl}: after_ticks 5401 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging latitude (NORTH y≈10..14 / SOUTH
    y≈26..30). Engine round-trip asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_hunt_bot_and_two_target_kinds():
    """Enemy must be the `hunt` scripted bot (pursues nearest agent
    units — required for the bait to actually pull defenders). Each
    level must include an enemy `fact` (the real / scoring target)
    and an enemy `powr` (the decoy target that does NOT score)."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot") or ""
    assert bot == "hunt", f"expected hunt bot, got {bot!r}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        etypes = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "fact" in etypes, f"{lvl}: must include enemy fact"
        assert "powr" in etypes, f"{lvl}: must include enemy powr decoy"


def test_sentinel_fact_present_per_level():
    """A persistent enemy `fact` placed far from the objective region
    so the engine does not auto-`done` the instant the objective fact
    dies; within_ticks / units_lost_lte then evaluate cleanly on the
    terminal frame (CLAUDE.md auto-terminate footgun)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        facts = [
            (a.position[0], a.position[1])
            for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        # at least one objective candidate near (100,10)/(100,30) AND a
        # far sentinel (>=12 cells from any objective candidate)
        objs = [
            p for p in facts
            if (p[0] - 100) ** 2 + (p[1] - 10) ** 2 <= 6 ** 2
            or (p[0] - 100) ** 2 + (p[1] - 30) ** 2 <= 6 ** 2
        ]
        sentinels = [p for p in facts if p not in objs]
        assert objs, f"{lvl}: must place ≥1 enemy fact in an objective region"
        assert sentinels, (
            f"{lvl}: must place ≥1 sentinel enemy fact outside the "
            f"objective region (engine auto-done guard); got facts={facts}"
        )


def test_squad_composition_per_level():
    """Squad A = 3 jeeps (bait), Squad B = 4 2tnk (strike) per spawn
    group. Each level must place at least one (jeep x3) and one
    (2tnk x4) on the agent side."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        jeep_groups = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "jeep"
        ]
        tank_groups = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert jeep_groups, f"{lvl}: missing agent jeep squad"
        assert tank_groups, f"{lvl}: missing agent 2tnk squad"
        for j in jeep_groups:
            assert (j.count or 1) == 3, f"{lvl}: jeep squad must be 3"
        for t in tank_groups:
            assert (t.count or 1) == 4, f"{lvl}: tank squad must be 4"


# ── engine-driven scripted policies ─────────────────────────────────


def _of_type(rs, want_types):
    out = []
    for u in rs.get("units_summary", []) or []:
        if u.get("type") in want_types:
            out.append(u)
    return out


def _enemies_of_type(rs, want_types):
    out = []
    for e in (rs.get("enemy_summary") or []):
        t = (e.get("type") or e.get("actor_type") or "").lower()
        if t in want_types:
            out.append(e)
    return out


def _enemy_buildings_in_region(rs, cx, cy, r):
    """Enemies of building kind ~ near (cx, cy) within radius r."""
    out = []
    for e in (rs.get("enemy_summary") or []):
        t = (e.get("type") or e.get("actor_type") or "").lower()
        ex = e.get("cell_x")
        ey = e.get("cell_y")
        if ex is None or ey is None:
            continue
        if (ex - cx) ** 2 + (ey - cy) ** 2 <= r * r:
            out.append((t, ex, ey, e.get("id")))
    return out


def _stall(rs, Command):
    """Pure observe — fact never takes damage; clock runs out → LOSS."""
    return [Command.observe()]


def _brute_all_on_decoy(rs, Command):
    """Every unit attack-moves onto the decoy region (powr) — razes
    the wrong target (which doesn't score), and tanks get shredded
    by the heavy e3 anti-tank cluster en route → LOSS."""
    cmds = []
    # Target the decoy powr — easy/medium decoy is at (100,30);
    # hard's decoy is at (100,20). Use centre y=20 for hard, y=30
    # otherwise. Cheap detection: if a powr is at y=20 in vision,
    # pick that; else use (100, 30).
    target_x, target_y = 100, 30
    for e in (rs.get("enemy_summary") or []):
        t = (e.get("type") or "").lower()
        if t == "powr" and e.get("cell_y") is not None:
            target_x, target_y = e["cell_x"], e["cell_y"]
            break
    for u in (rs.get("units_summary", []) or []):
        cmds.append(Command.attack_move([str(u["id"])], target_x, target_y))
    return cmds or [Command.observe()]


def _lazy_nearest(rs, Command):
    """Lazy "send each squad to its nearest visible target":
      • jeeps → nearest fact/powr in vision (typically the near-
        latitude one — for easy/medium that's the fact at (100,10)
        if NW spawn, or the powr at (100,30) if SW spawn).
      • tanks → nearest fact/powr in vision (the OTHER one).
    The lazy assignment puts jeeps on the wrong-strength target and
    tanks on the WRONG type (the decoy). On easy/medium the lazy
    jeeps-on-fact / tanks-on-powr means tanks raze the decoy (which
    doesn't score) and jeeps can't crack the fact garrison fast
    enough → clock expires → LOSS."""
    cmds = []
    jeeps = _of_type(rs, {"jeep"})
    tanks = _of_type(rs, {"2tnk"})
    if not (jeeps or tanks):
        return [Command.observe()]

    # Fixed-coord lazy nearest (model can read positions but lazy
    # means send-to-nearest-visible-target without bait logic):
    # jeeps from NW go to (100,10), tanks from SW go to (100,30).
    # Easy/medium objective: fact (100,10) is the real target;
    # this lazy plan razes only powr at (100,30) — fails.
    for j in jeeps:
        jy = j.get("cell_y", 20)
        # nearest by latitude: jeep at y≈10 → (100,10); else (100,30)
        tx, ty = (100, 10) if jy < 20 else (100, 30)
        cmds.append(Command.attack_move([str(j["id"])], tx, ty))
    for t in tanks:
        ty_unit = t.get("cell_y", 20)
        tx, ty = (100, 30) if ty_unit >= 20 else (100, 10)
        cmds.append(Command.attack_move([str(t["id"])], tx, ty))
    return cmds or [Command.observe()]


def _bait_only(rs, Command):
    """Jeeps slash toward the decoy region (pulls the heavy garrison
    into pursuit), tanks stand still. The fact survives — clock runs
    out → LOSS. Tests that the bait without the counter-attack does
    not score."""
    cmds = []
    for j in _of_type(rs, {"jeep"}):
        # Oscillate around the decoy region (south for easy/medium,
        # centre for hard) so the jeeps stay engaged but moving.
        jy = j.get("cell_y", 20)
        # If on north start, dive into south decoy region; mirror.
        target_y = 30 if jy < 20 else 10
        cmds.append(Command.move_units([str(j["id"])], 100, target_y))
    return cmds or [Command.observe()]


def _intended(rs, Command):
    """Diversion + main strike:
      1. Detect agent staging hemisphere from the unit centroid.
      2. Bait the jeeps INTO the heavier garrison's pursuit region:
         • easy/medium: decoy is south at (100,30) — jeeps slash
           south-east no matter the spawn (NW spawn jeeps must
           cross to south).
         • hard: decoy cluster is at CENTRE (100,20) — jeeps slash
           toward (100,20) to draw the centre e3 cluster.
      3. Main strike: tanks attack-move along the OPPOSITE latitude
         from the decoy to the real target.
         • easy/medium: tanks at SW must cross north to (100,10).
         • hard: real target is whichever fact MATCHES the agent's
           spawn latitude (NORTH spawn → NE fact at (100,10);
           SOUTH spawn → SE fact at (100,30)).
      4. If a fact is in vision and reachable, attack_unit it.
    """
    cmds = []
    jeeps = _of_type(rs, {"jeep"})
    tanks = _of_type(rs, {"2tnk"})
    all_units = jeeps + tanks
    if not all_units:
        return [Command.observe()]

    # Probe: which enemy buildings are present? Hard has a centre
    # powr at (100,20); easy/medium has a powr at (100,30).
    has_centre_decoy = any(
        (e.get("type") or "").lower() == "powr"
        and 16 <= (e.get("cell_y") or 99) <= 24
        for e in (rs.get("enemy_summary") or [])
    )

    # Spawn latitude detection by tank centroid (tanks dominate the
    # mass; jeeps may already have moved).
    if tanks:
        ty_avg = sum(t["cell_y"] for t in tanks) / len(tanks)
    else:
        ty_avg = sum(u["cell_y"] for u in all_units) / len(all_units)
    is_north = ty_avg < 20

    if has_centre_decoy:
        # HARD: bait jeeps INTO the centre decoy cluster.
        bait_x, bait_y = 100, 20
        # Strike: matching-latitude fact (NORTH spawn → NE fact;
        # SOUTH spawn → SE fact).
        strike_x, strike_y = (100, 10) if is_north else (100, 30)
    else:
        # EASY/MEDIUM: decoy is south at (100,30); jeeps slash south
        # regardless of spawn. The intended cross-attack for the
        # tanks is north onto the fact at (100,10).
        bait_x, bait_y = 100, 30
        strike_x, strike_y = 100, 10

    for j in jeeps:
        cmds.append(Command.move_units([str(j["id"])], bait_x, bait_y))

    if not tanks:
        return cmds or [Command.observe()]

    tank_ids = [str(t["id"]) for t in tanks]

    # If the real fact is in vision, attack_unit it for a clean kill;
    # else attack_move onto the cell so the column drives there.
    fact_id = None
    for e in (rs.get("enemy_summary") or []):
        et = (e.get("type") or "").lower()
        if et != "fact":
            continue
        ex, ey = e.get("cell_x"), e.get("cell_y")
        if ex is None or ey is None:
            continue
        if (ex - strike_x) ** 2 + (ey - strike_y) ** 2 <= 6 ** 2:
            fact_id = str(e.get("id"))
            break

    if fact_id is not None:
        cmds.append(Command.attack_unit(tank_ids, fact_id))
    else:
        cmds.append(Command.attack_move(tank_ids, strike_x, strike_y))
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_diversion_then_strike_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended diversion-then-strike should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(lost={r.signals.units_lost}, "
        f"bldgs_destroyed={r.signals.enemy_buildings_destroyed_types}, "
        f"records={getattr(r.signals, 'enemy_buildings_destroyed_records', [])})"
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
        f"(no building takes damage), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_all_on_decoy_loses(level, seed):
    """Brute frontal — every unit attack-moves onto the decoy powr.
    The decoy is razed (which scores zero — `enemy_key_buildings_
    destroyed_in_region` requires a `fact`) and the heavy e3 cluster
    shreds the tanks; the real fact is left standing → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_all_on_decoy, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute-all-on-decoy must LOSE "
        f"(real fact intact / strike force traded), got {r.outcome} "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_lazy_nearest_assignment_loses(level, seed):
    """Lazy "send each squad to its nearest visible target" play.
    On easy/medium this puts jeeps on the close-looking fact (too
    weak to crack) and tanks on the close-looking powr (the wrong
    type — `enemy_key_buildings_destroyed_in_region` requires a
    `fact`), so the real fact survives the clock → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _lazy_nearest, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: lazy nearest assignment must LOSE, "
        f"got {r.outcome} "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_bait_only_loses(level, seed):
    """Bait-only (jeeps slash toward the decoy, tanks stand still)
    must LOSE on every tier — the bait displacement yields ZERO
    objective progress; the real fact survives and the clock runs
    out. Discriminates "bait without strike" from the intended
    "bait AND strike" idiom."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _bait_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: bait-only must LOSE (real fact not "
        f"destroyed), got {r.outcome} "
        f"(bldgs={r.signals.enemy_buildings_destroyed_types})"
    )
