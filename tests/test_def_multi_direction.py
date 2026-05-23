"""def-multi-direction scenario pack, full loop on Rust.

REASONING capability — DISTRIBUTED DEFENSE across N concurrent attack
lanes converging on a CENTRAL base (per-tier lane scale: easy=2,
medium=3, hard=4). The load-bearing predicate is per-lane
`units_in_region_gte` defensive-zone clauses, each requiring >=1
tank: any concentration policy satisfies one clause and fails the
others, AND the unopposed lanes' bands raze the construction yard
(`has_building:fact` also bites).

The enemy uses the `hunt` bot (each unit attacks the nearest agent
unit) so a band placed in lane Y engages whatever defenders the
agent allocated to lane Y — the lanes are genuinely independent.
Agent tanks are `stance:2` (Defend): they auto-fire on an in-range
enemy but never advance, so the agent must actively order each
lane's pair to engage its band.

Per-tier MAP geometry (openra_bench.mapgen 'arena' generator):
* easy   — def-multi-direction-easy   128x40, 2 lanes (N+S)
* medium — def-multi-direction-medium 128x40, 3 lanes (N+S+E)
* hard   — def-multi-direction-hard   128x40, 4 lanes (NW+NE+SW+SE)

The bar:

* Per-lane `units_in_region_gte: n: 1` clauses ⇒ even-split is the
  safe solution; concentration policies satisfy at most one zone
  and fail the others.
* `units_killed_gte: N` paired with the band sizes so an active
  even-split defence — the intended policy actively `attack_unit`s
  the band in each lane — clears the bar comfortably while stall /
  wrong-concentration do not.
* `units_lost_lte: 3` (medium + hard) ⇒ attrition cap so a sloppy
  intercept that bleeds the line busts even if the zone clauses
  hold.
* `has_building: fact` ⇒ the fact survival clause stays as an
  always-on safety net.
* `within_ticks: 2700` paired with `after_ticks: 2701` ⇒ a non-
  finisher is a real reachable timeout LOSS, never a draw.
* Hard tier defines TWO agent spawn_point groups (NORTH base y=14 /
  SOUTH base y=26) so the lane geometry flips per seed; the win
  predicate is `any_of` over the two matching four-zone layouts,
  enforcing the doctrine over a memorised single-zone quad.

These tests prove deterministically (no model / no network):

* the intended even-split distributed-defence policy WINS every
  level + every hard seed;
* stall and concentrate-on-one-lane both LOSE (real LOSS, not a
  draw);
* the hard tier defines >=2 spawn_point groups so a memorised
  lane-zone layout cannot generalise across seeds.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-multi-direction.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-tier LANE-zone layouts (excludes the centre safety-net zone).
# Easy: 2 lanes (N+S). Medium: 3 lanes (N+S+E). Hard: 4 lanes
# (NW+NE+SW+SE) — base latitude rotates per spawn so the zone
# quad shifts; the intended policy detects base latitude from the
# tank centroid and picks the matching layout.
_ZONES_BY_LEVEL = {
    "easy":   [(64, 12), (64, 28)],
    "medium": [(64, 12), (64, 28), (108, 20)],
}


def _zones_hard(base_y: int) -> list:
    """Hard-tier zone layout matching the detected base latitude."""
    if base_y <= 18:
        # NORTH base (y=14) ⇒ NW(36,8), NE(92,8), SW(36,20), SE(92,20)
        return [(36, 8), (92, 8), (36, 20), (92, 20)]
    # SOUTH base (y=26) ⇒ NW(36,20), NE(92,20), SW(36,32), SE(92,32)
    return [(36, 20), (92, 20), (36, 32), (92, 32)]


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. Tanks stay clustered at central staging, never
    reach any defensive zone; stance:2 auto-fires only on whatever
    closes into the static knot, but the concurrent `hunt` bands
    overwhelm the cluster and raze the fact → LOSS."""
    return [C.observe()]


def _own_tank_ids(rs):
    own = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
    return sorted(str(u["id"]) for u in own)


def _own_tank_centroid_y(rs):
    own = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
    if not own:
        return 20
    return sum(u["cell_y"] for u in own) // len(own)


def make_concentrate(zone_x, zone_y):
    """Send ALL tanks to a single defensive zone. Satisfies ONE zone
    clause; the other zone clauses (and the unopposed-lane fact-
    survival clause) all fail → LOSS."""

    def policy(rs, C):
        ids = _own_tank_ids(rs)
        if not ids:
            return [C.observe()]
        return [C.attack_move(ids, target_x=zone_x, target_y=zone_y)]

    return policy


def make_intended(level):
    """Lane-aware even-split distributed defence. On the first call,
    sort tank ids, detect base latitude (so the hard-tier spawn
    rotation generalises), and assign tanks evenly across the
    level's lane zones. Each turn: every lane's pair `attack_unit`s
    the nearest band member in range; if no enemy is near it falls
    back to `attack_move` onto the lane's defensive zone. Tanks
    are `stance:2` (Defend, never advance), so the agent must
    issue the engagement orders actively — auto-fire alone does
    not close on the band."""

    state = {"assignments": None}

    def policy(rs, C):
        ids = _own_tank_ids(rs)
        if not ids:
            return [C.observe()]
        if state["assignments"] is None:
            if level == "hard":
                zones = _zones_hard(_own_tank_centroid_y(rs))
            else:
                zones = _ZONES_BY_LEVEL[level]
            n = len(ids)
            per = max(1, n // len(zones))
            assignments = {}
            for i, z in enumerate(zones):
                start = i * per
                end = start + per if i < len(zones) - 1 else n
                assignments[z] = ids[start:end]
            state["assignments"] = assignments
        alive = set(ids)
        enemies = [
            e for e in rs.get("enemy_summary", [])
            if e.get("type") == "e1"
        ]
        cmds = []
        for (zx, zy), tids in state["assignments"].items():
            live = [i for i in tids if i in alive]
            if not live:
                continue
            lane_enemies = [
                e for e in enemies
                if abs(e["cell_x"] - zx) + abs(e["cell_y"] - zy) <= 30
            ]
            if lane_enemies:
                tgt = min(
                    lane_enemies,
                    key=lambda e: abs(e["cell_x"] - zx) + abs(e["cell_y"] - zy),
                )
                cmds.append(C.attack_unit(live, str(tgt["id"])))
            else:
                cmds.append(C.attack_move(live, target_x=zx, target_y=zy))
        return cmds or [C.observe()]

    return policy


# ── structural checks (no engine) ────────────────────────────────────


def test_pack_loads_and_metadata_is_complete():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-multi-direction"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor or []
    # The required anchors from the Wave-7 spec.
    joined = " | ".join(a.lower() for a in anchors)
    assert "distributed-systems load balancing" in joined, anchors
    assert "graph min-cut" in joined, anchors
    assert "military multi-front" in joined, anchors
    # `hunt` bot wired through to the engine for every level: each
    # band attacks the nearest agent unit, so a band in lane Y
    # engages whatever defenders the agent allocated to lane Y —
    # the lanes are genuinely independent threats.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", (lvl, bot)
        # Agent tanks must be stance:2 (Defend) — never auto-advance,
        # so a stall cluster cannot self-deliver across the map.
        tank_stances = {
            a.stance for a in c.scenario.actors
            if a.type == "2tnk" and a.owner == "agent"
        }
        assert tank_stances == {2}, (lvl, tank_stances)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (<=90 ticks/step
    in interrupt mode)."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert deadline < reachable, (
        f"{level}: deadline {deadline} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


@pytest.mark.parametrize("level,min_zones", [("easy", 2), ("medium", 3), ("hard", 4)])
def test_every_level_requires_per_lane_zones(level, min_zones):
    """The distribution axis must be load-bearing: every level's win
    condition must require one `units_in_region_gte` clause per
    concurrent attack lane (easy: 2, medium: 3, hard: 4). Any
    concentration policy fails at least all-but-one of those
    clauses. (Hard counts 8 occurrences in the flat string —
    4 zones x 2 candidate spawn-rotated layouts.)"""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert flat.count("units_in_region_gte") >= min_zones, (level, win)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: >=2 distinct seed-driven spawn_point groups
    so the base latitude (and lane geometry) flips by seed
    (anti-memorisation of an absolute lane quad — the distributed-
    defence doctrine must generalise)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (128x40 cordon 2 ⇒ playable x in [2..125],
    # y in [2..37]).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 125 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_split_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(level), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended even-split distributed defence "
            f"must WIN; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Stall (observe only) — tanks never reach any LANE defensive
    zone; the concurrent waves (especially the unopposed flanks)
    raze the fact → real LOSS on `not has_building:fact` and/or
    LANE zone clauses empty + `after_ticks` deadline."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not draw); "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# Per-tier "concentrate on the northern lane" target.
_CONCENTRATE_NORTH = {
    "easy":   (64, 12),     # N lane defensive zone
    "medium": (64, 12),     # N lane defensive zone
    "hard":   (36, 8),      # NW corner for the NORTH-base layout
}


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_north_loses_every_level_and_seed(level):
    """Concentrate-NORTH: all tanks at the northern lane zone. That
    one zone clause satisfies but the other lane zones fail
    (n=0 < 1), AND the unopposed lane bands raze the fact → real
    LOSS."""
    c = compile_level(load_pack(PACK), level)
    zx, zy = _CONCENTRATE_NORTH[level]
    pol = make_concentrate(zx, zy)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-NORTH @ ({zx},{zy}): "
            f"must LOSE; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# Per-tier "concentrate on the central/MID staging" target — the
# defenders never leave the centre, so NO lane zone is ever held.
_CONCENTRATE_MID = {
    "easy":   (64, 20),     # base centre
    "medium": (64, 20),     # base centre
    "hard":   (64, 14),     # NORTH base centre (NORTH-spawn seeds)
}


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_mid_loses_every_level_and_seed(level):
    """Concentrate-MID (defenders stay at the base centre): no lane's
    defensive zone is reached, ALL lane zone clauses fail, and the
    concurrent waves carve through the cluster from multiple sides
    → LOSS."""
    c = compile_level(load_pack(PACK), level)
    cx, cy = _CONCENTRATE_MID[level]
    pol = make_concentrate(cx, cy)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-MID @ ({cx},{cy}): "
            f"must LOSE; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# Per-tier "concentrate on the southern lane" target.
_CONCENTRATE_SOUTH = {
    "easy":   (64, 28),     # S lane defensive zone
    "medium": (64, 28),     # S lane defensive zone
    "hard":   (36, 32),     # SW corner for the SOUTH-base layout
}


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_south_loses_every_level_and_seed(level):
    """Concentrate-SOUTH: symmetric to concentrate-NORTH — one zone
    holds, the other lane zones fail, the unopposed bands raze the
    fact → LOSS."""
    c = compile_level(load_pack(PACK), level)
    zx, zy = _CONCENTRATE_SOUTH[level]
    pol = make_concentrate(zx, zy)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-SOUTH @ ({zx},{zy}): "
            f"must LOSE; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended("easy"), seed=2)
    b = run_level(c, make_intended("easy"), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
