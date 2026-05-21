"""def-pre-position-mobile-reserve scenario pack, full loop on Rust.

REASONING capability — CENTRAL MOBILE RESERVE that must move OUT of
the centre to intercept rushers on whichever flank materialises. The
load-bearing predicate is `units_in_region_gte` at the FORWARD
intercept zone(s): a passive central reserve auto-engages the rush
when it closes to cannon range (the engine handles the kill economy)
but never SATISFIES the forward-zone clause; only an explicit
forward `attack_move` to the lane mouth puts a tank in the zone.

The bar:

* `units_in_region_gte: {x, y, radius, n: 1..2}` at the lane-mouth
  intercept point(s) ⇒ a stall keeps tanks at the centre → outside
  the zone → LOSS even when the kill bar is met.
* On medium: BOTH a N forward zone AND a S forward zone ⇒ the
  reserve must SPLIT or stagger; committing the whole reserve to one
  flank leaves the opposite zone empty → LOSS.
* On hard: THREE forward zones (N, CENTRE, S) ⇒ the reserve must
  split three ways; any single-flank commit fails ≥2 zones.
* `units_killed_gte: N` ⇒ a real engagement bar (matched to the
  total band size so an active intercept clears it comfortably).
* `units_lost_lte: 2` (hard only) ⇒ attrition cap so a sloppy
  intercept that bleeds the reserve busts even if the position
  clauses hold.
* `within_ticks: 2400` paired with `after_ticks: 2401` ⇒ a non-
  finisher is a real reachable timeout LOSS, never a draw.

These tests prove deterministically (no model / no network):

* the intended forward-intercept play WINS every level + every
  hard seed;
* stall, commit-NORTH, and commit-SOUTH all LOSE on the levels
  where they don't match the intended capability (a real LOSS, not
  a draw);
* the hard tier defines ≥2 spawn_point groups so a memorised
  centre cell cannot generalise.
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

PACK = PACKS_DIR / "def-pre-position-mobile-reserve.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. The stance:2 reserve auto-fires only on what
    closes into the static cluster, but NEVER MOVES OUT of the
    starting cell → the forward-zone clauses are never satisfied and
    the concurrent `hunt` bands overrun the un-forward reserve →
    real LOSS (position clause empty + units lost / clock)."""
    return [C.observe()]


def commit_north(rs, C):
    """`attack_move` the WHOLE reserve at the NORTH intercept zone on
    turn 1 — a single-flank pre-commitment. On medium/hard the S
    (and E) forward zones stay empty → LOSS on the position
    clause(s); this policy is only asserted on the multi-flank
    levels (easy single-direction is covered by the intended-WIN
    test, which uses the active lane-aware intercept)."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    return [C.attack_move(ids, target_x=60, target_y=12)]


def commit_south(rs, C):
    """Move the WHOLE reserve to the SOUTH intercept zone on turn 1.
    On every level S is either wrong (easy: rush is NORTH) or
    insufficient (medium/hard: other zones empty) → LOSS."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    return [C.attack_move(ids, target_x=60, target_y=28)]


# Forward-intercept zone centres per lane label (x, y).
_ZONE_XY = {"N": (60, 12), "C": (65, 20), "S": (60, 28)}


def make_intended():
    """Stable per-tank assignment so each tank's lane doesn't flip
    frame-to-frame. On the FIRST call, sort tank ids and read enemy
    positions to detect which lanes are active; assign tanks to the
    active lanes (NORTH-only on easy, NORTH+SOUTH on medium,
    NORTH+CENTRE+SOUTH on hard). Each turn, every lane's detachment
    actively `attack_unit`s the nearest `hunt` band member in its
    own latitude band — because the tanks are `stance:2` (Defend,
    never advance) the agent must issue the forward engagement
    orders itself; if no enemy is in the lane it falls back to
    `attack_move` onto the lane's forward intercept zone."""

    state = {"assignments": None}

    def policy(rs, C):
        own = [
            u for u in rs.get("units_summary", [])
            if u.get("type") == "2tnk"
        ]
        if not own:
            return [C.observe()]
        ids = sorted(str(u["id"]) for u in own)
        if state["assignments"] is None:
            # Wait until enemies are visible (interrupt mode fires
            # enemy_unit_spotted quickly), then detect active lanes.
            enem = [
                e for e in (rs.get("enemy_summary") or [])
                if e.get("type") == "e1"
            ]
            if not enem:
                # Enemies not yet visible — hold at centre (observe)
                # until interrupt fires.
                return [C.observe()]
            n_band = any(e["cell_y"] < 18 for e in enem)
            s_band = any(e["cell_y"] > 22 for e in enem)
            e_band = any(
                18 <= e["cell_y"] <= 22 and e["cell_x"] > 60
                for e in enem
            )
            active = []
            if n_band:
                active.append("N")
            if e_band:
                active.append("C")
            if s_band:
                active.append("S")
            if not active:
                return [C.observe()]
            n = len(ids)
            if len(active) == 1:
                state["assignments"] = {active[0]: ids}
            elif len(active) == 2:
                half = max(1, n // 2)
                state["assignments"] = {
                    active[0]: ids[:half], active[1]: ids[half:]
                }
            else:
                third = max(1, n // 3)
                state["assignments"] = {
                    active[0]: ids[:third],
                    active[1]: ids[third:2 * third],
                    active[2]: ids[2 * third:],
                }
        a = state["assignments"]
        alive = set(ids)
        enemies = [
            e for e in (rs.get("enemy_summary") or [])
            if e.get("type") == "e1"
        ]
        cmds = []
        for lane, tids in a.items():
            live = [i for i in tids if i in alive]
            if not live:
                continue
            zx, zy = _ZONE_XY[lane]
            lane_enemies = [
                e for e in enemies if abs(e["cell_y"] - zy) <= 10
            ]
            if lane_enemies:
                tgt = min(
                    lane_enemies,
                    key=lambda e: abs(e["cell_x"] - zx)
                    + abs(e["cell_y"] - zy),
                )
                cmds.append(C.attack_unit(live, str(tgt["id"])))
            else:
                cmds.append(
                    C.attack_move(live, target_x=zx, target_y=zy)
                )
        return cmds or [C.observe()]

    return policy


# ── structural checks (no engine) ────────────────────────────────────


def test_pack_loads_and_metadata_is_complete():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-pre-position-mobile-reserve"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor must be non-empty"
    assert any("reserve" in a.lower() for a in anchors), anchors
    assert any(
        "chess" in a.lower() or "central" in a.lower() for a in anchors
    ), anchors
    # `hunt` bot wired through to the engine for every level: each
    # band attacks the nearest agent unit, so a band in lane Y
    # engages the reserve detachment the agent committed to lane Y.
    # (`rusher` pooled all bands onto the centroid; it was retired
    # in the post-engine-balance recalibration — see pack docstring.)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "hunt", (lvl, bot)
        # Agent tanks must be stance:2 (Defend) — never auto-advance,
        # so a stall reserve cannot self-deliver across the map.
        tank_stances = {
            a.stance for a in c.scenario.actors
            if a.type == "2tnk" and a.owner == "agent"
        }
        assert tank_stances == {2}, (lvl, tank_stances)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (≤90 ticks/step
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


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_enforces_forward_zone(level):
    """The position axis must be load-bearing: every level's win
    condition must require at least one `units_in_region_gte` clause
    (the forward-zone discriminator). Medium and hard require ≥2
    zones (must split)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "units_in_region_gte" in flat, win
    # Count zones (one per occurrence in the win dict).
    zone_count = flat.count("units_in_region_gte")
    if level == "easy":
        assert zone_count >= 1, (level, zone_count, win)
    elif level == "medium":
        assert zone_count >= 2, (level, zone_count, win)
    elif level == "hard":
        assert zone_count >= 3, (level, zone_count, win)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the reserve's starting cell rotates by seed (anti-memorisation
    of an absolute cell — the doctrine must generalise)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_split_intercept_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended split/forward-intercept play "
            f"must WIN; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Stall (observe only) — the reserve never reaches the forward
    zone(s) → LOSS even when the auto-defence racks up kills."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not draw); "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ("medium", "hard"))
def test_commit_north_loses_on_multi_flank_levels(level):
    """Committing the WHOLE reserve to NORTH leaves the SOUTH (and on
    hard, also the CENTRE/E) forward zone(s) empty → position-clause
    LOSS. (On EASY commit-NORTH is the intended single-direction
    play and WINS — that case is covered by the intended-WIN test.)"""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, commit_north, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} commit-NORTH: must LOSE on multi-flank "
            f"level; got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_commit_south_loses_every_level_and_seed(level):
    """Committing the WHOLE reserve to SOUTH is always wrong — on
    easy the rush is NORTH (S zone irrelevant); on medium/hard the
    N (and E) forward zones stay empty → LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, commit_south, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} commit-SOUTH: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=2)
    b = run_level(c, make_intended(), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
