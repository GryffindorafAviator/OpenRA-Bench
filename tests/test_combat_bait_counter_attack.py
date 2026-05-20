"""combat-bait-counter-attack — bait-and-flank counter-attack micro.

The bar: intended bait-then-strike (bait jeep pulls a SIDE of a leashed
GUARD cluster off the enemy fact; strike tanks engage the now-isolated
guards and then destroy the briefly-undefended fact) WINS on every
level and every hard seed (1..4). STALL (only observe), BRUTE FRONTAL
(attack-move all units east at the fact), and BAIT-ONLY (jeep flanks
south but no strike) all LOSE on every level and every hard seed.
Non-win is a real reachable timeout LOSS via the `after_ticks` fail
clause; `units_lost_lte` provides the second LOSS path (brute frontal
trades the strike force).

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-bait-counter-attack.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-bait-counter-attack"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 3, (
        f"benchmark_anchor must list all 3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "feint", "cicero"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, lost=0, fact_destroyed=False):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    types_count = {"fact": 1} if fact_destroyed else {}
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
        enemy_buildings_destroyed=1 if fact_destroyed else 0,
        enemy_buildings_destroyed_types=types_count,
        enemy_buildings_destroyed_records=(
            [("fact", 80, 20)] if fact_destroyed else []
        ),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n):
    return [
        {"cell_x": 10, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: fact destroyed, 1 unit lost (bait), in budget → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=3000, lost=1, fact_destroyed=True),
    )
    # Fact NOT destroyed → not a win even if in budget and few losses
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=3000, lost=1, fact_destroyed=False),
    )
    # Loss cap (3) tripped → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(0), tick=3000, lost=4, fact_destroyed=True),
    )
    # Past deadline → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=5401, lost=1, fact_destroyed=True),
    )
    # Timeout with bar unmet → fail (after_ticks 5401)
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(3), tick=5500, lost=1, fact_destroyed=False),
    )
    # Force-wipe → fail (not own_units_gte:1)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=3000, lost=4, fact_destroyed=False),
    )
    # Loss cap tripped (>3) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(0), tick=3000, lost=4, fact_destroyed=True),
    )


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: fact destroyed, ≤2 lost → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=3000, lost=2, fact_destroyed=True),
    )
    # 3 lost (one too many) → not a win, AND fail
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(2), tick=3000, lost=3, fact_destroyed=True),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(2), tick=3000, lost=3, fact_destroyed=True),
    )
    # Timeout with fact intact → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(4), tick=5500, lost=1, fact_destroyed=False),
    )


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: fact destroyed, ≤2 lost → WIN (cap is 2: bait + ≤1 tank)
    assert evaluate(
        c.win_condition,
        _ctx(units=_alive(3), tick=3500, lost=2, fact_destroyed=True),
    )
    # 3 lost → not a win, AND fail
    assert not evaluate(
        c.win_condition,
        _ctx(units=_alive(1), tick=3500, lost=3, fact_destroyed=True),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units=_alive(1), tick=3500, lost=3, fact_destroyed=True),
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
        assert 5400 <= max_tick, (
            f"{lvl}: within_ticks 5400 > max tick {max_tick}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging latitude (north y=10 / south y=30).
    Engine-roundtrip is asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_guard_bot_and_e3_cluster():
    """The enemy must be the `guard` scripted bot (leashed defender) and
    the cluster must be e3 rocket-infantry (anti-tank Dragon, range 5,
    damage 5000 vs Heavy) — these together make the bait-and-flank idiom
    load-bearing (a hunter or patrol bot would not snap back, and an e1
    cluster would not threaten the strike tanks)."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot") or ""
    assert bot == "guard", f"expected guard bot, got {bot!r}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "e3" in types, f"{lvl}: cluster must be e3 rocket infantry"
        assert "fact" in types, f"{lvl}: objective must include enemy fact"


def test_objective_fact_and_sentinel_present():
    """Two enemy `fact` per level: one objective fact (mid-map, ~80,20)
    and one sentinel fact (far north ~120,4) keeping the episode alive
    past objective-fact destruction so within_ticks / units_lost_lte
    can evaluate on the terminal frame (the CLAUDE.md auto-terminate
    footgun on MustBeDestroyed)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        facts = [
            (a.position[0], a.position[1])
            for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert len(facts) == 2, (
            f"{lvl}: must have 2 enemy facts (objective + sentinel), "
            f"got {len(facts)} at {facts}"
        )
        # One ~mid-map (objective), one far from that one (sentinel).
        # Use the rough centre of the playable arena.
        mid = [p for p in facts if 60 <= p[0] <= 100 and 14 <= p[1] <= 26]
        far = [p for p in facts if not (60 <= p[0] <= 100 and 14 <= p[1] <= 26)]
        assert len(mid) == 1 and len(far) == 1, (
            f"{lvl}: expected 1 objective fact mid-map + 1 sentinel far, "
            f"got mid={mid} far={far}"
        )


def test_guard_count_per_level():
    """Difficulty axis: easy 3 guards, medium 5, hard 6 — one new
    controlled variable per tier (cluster density). The loss cap also
    tightens (3 → 2 → 2 with the hard tier additionally adding spawn
    variation)."""
    pack = load_pack(PACK_PATH)
    expected = {"easy": 3, "medium": 5, "hard": 6}
    for lvl, want in expected.items():
        c = compile_level(pack, lvl)
        n_e3 = sum(
            1 for a in c.scenario.actors if a.owner == "enemy" and a.type == "e3"
        )
        assert n_e3 == want, f"{lvl}: expected {want} guards, got {n_e3}"


# ── engine-driven scripted policies ─────────────────────────────────


def _own_ids(rs):
    return [str(u["id"]) for u in (rs.get("units_summary", []) or [])]


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


def _stall(rs, Command):
    """Pure observe — fact never takes damage; clock runs out → LOSS."""
    return [Command.observe()]


def _brute(rs, Command):
    """Attack-move every unit straight east at the fact. The strike
    column eats anti-tank fire from the whole guard cluster on
    approach, trades 4 units (1 jeep + 3 tanks) before the fact
    falls, trips the units_lost_lte cap → LOSS."""
    cmds = []
    for u in (rs.get("units_summary", []) or []):
        cmds.append(Command.attack_move([str(u["id"])], 80, 20))
    return cmds or [Command.observe()]


def _bait_only(rs, Command):
    """Jeep flanks south (pulls a side of the guards off), strike
    tanks stand still. The fact survives — clock runs out → LOSS.
    Tests that the bait without the counter-attack does not score."""
    cmds = []
    for u in (rs.get("units_summary", []) or []):
        if u.get("type") == "jeep":
            jx, jy = u["cell_x"], u["cell_y"]
            if jy >= 31:
                cmds.append(Command.move_units([str(u["id"])], 80, 28))
            else:
                cmds.append(Command.move_units([str(u["id"])], 80, 33))
    return cmds or [Command.observe()]


def _intended(rs, Command):
    """Bait + counter-attack:
      1. Detect the spawn hemisphere from the agent unit centroid
         (avg_y < 20 = north spawn).
      2. Bait the jeep into the MATCHING hemisphere flank, oscillating
         on the far cell — this stays within GUARD_AGGRO (~16) of the
         matching half of the guard cluster so they lunge off post,
         but outside the e3 anti-tank Dragon range (5 cells) so the
         jeep survives the kite.
      3. Strike: pick the guard NEAREST the tank centroid (the most
         immediate threat: typically one of the matching-side guards
         being pulled OUT of formation toward the bait, so engaging it
         away from its post is a favourable 3-vs-1 trade per kill).
         Cycle until the cluster is broken.
      4. Once no guards are in vision, attack-move onto the fact
         coordinate. If a near fact is in vision, attack_unit it.
    """
    cmds = []
    tanks = _of_type(rs, {"2tnk"})
    jeeps = _of_type(rs, {"jeep"})
    tank_ids = [str(t["id"]) for t in tanks]
    if not (tanks or jeeps):
        return [Command.observe()]

    all_units = tanks + jeeps
    avg_y = sum(u["cell_y"] for u in all_units) / len(all_units)
    is_north = avg_y < 20

    # Bait: oscillate at the matching-hemisphere flank to keep guards
    # engaged without dying.
    for j in jeeps:
        jy = j["cell_y"]
        if is_north:
            target_y = 8 if jy > 9 else 12
        else:
            target_y = 33 if jy < 32 else 28
        cmds.append(Command.move_units([str(j["id"])], 80, target_y))

    if not tanks:
        return cmds or [Command.observe()]

    tx = sum(t["cell_x"] for t in tanks) / len(tanks)
    ty = sum(t["cell_y"] for t in tanks) / len(tanks)

    guards = _enemies_of_type(rs, {"e3"})
    fact_id = None
    for e in (rs.get("enemy_summary", []) or []):
        et = (e.get("type") or "").lower()
        if et == "fact" and e.get("cell_x", 200) < 100:
            fact_id = str(e.get("id"))
            break

    if guards:
        guards.sort(
            key=lambda g: (g["cell_x"] - tx) ** 2 + (g["cell_y"] - ty) ** 2
        )
        cmds.append(Command.attack_unit(tank_ids, str(guards[0]["id"])))
    elif fact_id is not None:
        cmds.append(Command.attack_unit(tank_ids, fact_id))
    else:
        # Cluster broken but fact not yet in vision — push onto its
        # known coordinate (it stays put as a building).
        cmds.append(Command.attack_move(tank_ids, 80, 20))
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_bait_then_strike_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended bait-then-strike should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(lost={r.signals.units_lost}, "
        f"bldgs_destroyed={r.signals.enemy_buildings_destroyed_types})"
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
        f"(fact never takes damage), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_frontal_loses(level, seed):
    """Attack-move all units straight east at the fact must LOSE on
    every tier — the e3 anti-tank Dragons (range 5, damage 5000 vs
    Heavy armour) shred the column on approach, trading the whole
    strike force (4 lost) before the fact can be destroyed; the
    units_lost_lte cap fails."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute frontal must LOSE "
        f"(strike force traded), got {r.outcome} "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_bait_only_loses(level, seed):
    """Bait-only (jeep kites south, strike tanks stand still) must LOSE
    on every tier — the bait displacement yields ZERO objective
    progress; the fact survives and the clock runs out. Discriminates
    "bait without counter-attack" from the intended "bait AND counter-
    attack" idiom."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _bait_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: bait-only must LOSE (fact not destroyed), "
        f"got {r.outcome} (bldgs={r.signals.enemy_buildings_destroyed_types})"
    )
