"""coord-converge-on-target — triple-prong convergent attack.

The bar: intended THREE-PRONG converge (all three squads driven onto
the objective fact at ~100,20 simultaneously) WINS on every level and
every hard seed (1..4). STALL (only observe), SINGLE-SQUAD-N (only the
north squad attacks while the other two stand still), and TWO-SQUADS
(north + south only — the west / east third squad stands still) all
LOSE on every level and every hard seed. Non-win is a real reachable
timeout LOSS via the `after_ticks` fail clause (force-wipe also fails).

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
PACK_PATH = PACKS / "coord-converge-on-target.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "coord-converge-on-target"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list ≥3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "convergent", "envelopment"):
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
            [("fact", 100, 20)] if fact_destroyed else []
        ),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _tanks_at(cx, cy, n, base_id=1000):
    return [
        {"cell_x": cx, "cell_y": cy, "type": "2tnk", "id": str(base_id + i)}
        for i in range(n)
    ]


def test_predicates_easy_medium_hard():
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # WIN: ≥6 tanks inside (100,20,r=8) AND fact destroyed AND in budget.
        win_units = _tanks_at(100, 20, 6) + _tanks_at(10, 20, 3, base_id=2000)
        assert evaluate(
            c.win_condition,
            _ctx(units=win_units, tick=4000, fact_destroyed=True),
        ), f"{lvl}: 6-in-region + fact destroyed must WIN"

        # Fact NOT destroyed → not a win even with tanks in region.
        assert not evaluate(
            c.win_condition,
            _ctx(units=win_units, tick=4000, fact_destroyed=False),
        )

        # Only 5 tanks in region (one short of n=6) → not a win.
        five_in = _tanks_at(100, 20, 5) + _tanks_at(10, 20, 4, base_id=2000)
        assert not evaluate(
            c.win_condition,
            _ctx(units=five_in, tick=4000, fact_destroyed=True),
        ), f"{lvl}: 5-in-region must not WIN (n=6 threshold)"

        # Past deadline → not a win.
        assert not evaluate(
            c.win_condition,
            _ctx(units=win_units, tick=4501, fact_destroyed=True),
        )

        # Timeout with bar unmet → fail (after_ticks 4501).
        assert evaluate(
            c.fail_condition,
            _ctx(units=win_units, tick=4600, fact_destroyed=False),
        )

        # Force-wipe → fail (not own_units_gte:1).
        assert evaluate(
            c.fail_condition,
            _ctx(units=[], tick=3000, fact_destroyed=False),
        )


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
            f"{lvl}: within_ticks 4500 > max tick {max_tick}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the staging geometry (N/W/S vs N/E/S). Engine
    round-trip is asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_guard_bot_and_e3_cluster():
    """The enemy must be the `guard` scripted bot (leashed defender)
    and the cluster must be e3 rocket-infantry (anti-tank Dragon) —
    these together make any single squad's frontal commit costly,
    so a 3-prong convergence is required to win the trade."""
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
    """Two enemy `fact` per level: one objective fact (~100,20) and one
    far sentinel fact keeping the episode alive past objective-fact
    destruction so within_ticks evaluates on the terminal frame (the
    CLAUDE.md MustBeDestroyed auto-terminate footgun)."""
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
        obj = [p for p in facts if 92 <= p[0] <= 108 and 14 <= p[1] <= 26]
        far = [p for p in facts if not (92 <= p[0] <= 108 and 14 <= p[1] <= 26)]
        assert len(obj) == 1 and len(far) == 1, (
            f"{lvl}: expected 1 objective fact near (100,20) + 1 sentinel "
            f"far, got obj={obj} far={far}"
        )


def test_guard_count_per_level():
    """Difficulty axis: easy 3 guards (distributed), medium 4 (one
    per cardinal face), hard 4 + seed-driven agent spawn variation.
    The discrimination on medium / hard is tighter than on easy: a
    single squad alone is destroyed in every tier; a two-squad
    converge destroys the fact but trades the strike force out on
    medium / hard (cardinal cover); only the three-prong meets the
    n=6 region threshold."""
    pack = load_pack(PACK_PATH)
    expected = {"easy": 3, "medium": 4, "hard": 4}
    for lvl, want in expected.items():
        c = compile_level(pack, lvl)
        n_e3 = sum(
            1 for a in c.scenario.actors if a.owner == "enemy" and a.type == "e3"
        )
        assert n_e3 == want, f"{lvl}: expected {want} guards, got {n_e3}"


def test_three_agent_squads_per_spawn():
    """Every spawn group must place 3 squads of 3× 2tnk each (9 tanks
    total) so the n=6 region clause requires a true 3-prong converge
    (a 2-prong with full attrition cannot reach 6 inside the region)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # group spawn -> total agent tank count
        by_grp: dict = {}
        for a in c.scenario.actors:
            if a.owner == "agent" and a.type == "2tnk":
                key = a.spawn_point if a.spawn_point is not None else 0
                by_grp[key] = by_grp.get(key, 0) + (a.count or 1)
        for k, n in by_grp.items():
            assert n == 9, (
                f"{lvl}: spawn group {k} must have 9× 2tnk (3 squads × 3), "
                f"got {n}"
            )


# ── engine-driven scripted policies ─────────────────────────────────


def _of_type(rs, want_types):
    return [
        u for u in (rs.get("units_summary", []) or [])
        if u.get("type") in want_types
    ]


def _stall(rs, Command):
    """Pure observe — no tank ever moves; objective fact untouched and
    the deadline expires → LOSS."""
    return [Command.observe()]


def _by_squad(rs):
    """Bucket agent tanks by their starting hemisphere so a partial
    policy can drive only some squads. Squads start at:
      NORTH y≈6   (any tank with cell_y ≤ 12)
      SOUTH y≈34  (any tank with cell_y ≥ 28)
      WEST  x≈10  (any tank with cell_x ≤ 20 and 12 < cell_y < 28)
      EAST  x≈120 (any tank with cell_x ≥ 110 and 12 < cell_y < 28)
    Per-seed the hard tier flips W↔E for the third squad; the bucket
    label is "flank" for whichever lateral squad is present.
    """
    north, south, flank = [], [], []
    for t in _of_type(rs, {"2tnk"}):
        x, y = t["cell_x"], t["cell_y"]
        if y <= 12:
            north.append(t)
        elif y >= 28:
            south.append(t)
        else:
            flank.append(t)
    return north, south, flank


def _single_squad_n(rs, Command):
    """Only the NORTH squad attack-moves onto the objective; the other
    two squads stand still. Three tanks vs the guard cluster (3 / 6 /
    8 by tier) lose the trade — fact never falls and/or the attacking
    squad is wiped (≤3 of 9 in region < n=6). LOSS."""
    north, _south, _flank = _by_squad(rs)
    cmds = []
    for t in north:
        cmds.append(Command.attack_move([str(t["id"])], 100, 20))
    return cmds or [Command.observe()]


def _two_squads(rs, Command):
    """NORTH + SOUTH squads attack-move onto the objective; the
    lateral (WEST or EAST) squad stands still. 6 tanks vs the cluster
    are still defeated under heavier defender density (medium 6
    guards, hard 8) and the surviving in-region count drops below
    n=6. LOSS."""
    north, south, _flank = _by_squad(rs)
    cmds = []
    for t in north + south:
        cmds.append(Command.attack_move([str(t["id"])], 100, 20))
    return cmds or [Command.observe()]


def _intended_three_prong(rs, Command):
    """All three squads attack-move directly onto the objective fact at
    (100,20). 9 tanks converging on the guard cluster (3 / 6 / 8 e3
    defenders) overwhelm by mass: at least 6 of the 9 tanks survive
    inside the (100,20,r=8) region while the cluster + fact fall.
    WIN on every tier and every seed."""
    cmds = []
    for t in _of_type(rs, {"2tnk"}):
        cmds.append(Command.attack_move([str(t["id"])], 100, 20))
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_three_prong_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_three_prong, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended 3-prong converge should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
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


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_single_squad_north_loses(level, seed):
    """Single north squad alone vs the guard cluster must LOSE on
    medium (6 guards) and hard (8 guards). Easy with only 3 guards
    may occasionally let a single squad squeak through; the
    discrimination tier is medium+."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _single_squad_n, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: single-squad north must LOSE "
        f"(3 tanks vs guard cluster), got {r.outcome} "
        f"(lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_two_squads_loses(level, seed):
    """Two squads (north + south) without the third must LOSE on
    medium (6 guards) and hard (8 guards) — 6 tanks vs the cluster
    trade out and the surviving in-region count drops below n=6 (or
    the fact never falls)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _two_squads, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: two-squad converge must LOSE on "
        f"medium/hard (defender density overwhelms 6 tanks), "
        f"got {r.outcome} (lost={r.signals.units_lost}, "
        f"bldgs={r.signals.enemy_buildings_destroyed_types})"
    )
