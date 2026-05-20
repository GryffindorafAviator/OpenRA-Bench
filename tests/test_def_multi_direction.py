"""def-multi-direction scenario pack, full loop on Rust.

REASONING capability — DISTRIBUTED DEFENSE across three concurrent
attack lanes (NORTH y=8, MID y=20, SOUTH y=32 — or the spawn-rotated
hard-tier equivalents). The load-bearing predicate is THREE
`units_in_region_gte` defensive-zone clauses, each requiring ≥2
tanks: any concentration policy satisfies one clause and fails the
other two, AND the unopposed lanes' rushers raze the construction
yard (`has_building:fact` also bites).

The bar:

* Three `units_in_region_gte: n: 1` clauses ⇒ the 2/2/2 distributed
  allocation is the safe solution (the n=1 floor leaves attrition
  headroom on top of the 2-per-zone doctrine); concentration
  policies satisfy at most one zone and fail the other two.
* `units_killed_gte: N` paired with the band sizes so an active
  2/2/2 defence clears the bar comfortably while stall / wrong-
  concentration do not (kills mostly come from the bot's rusher
  funnel reaching whichever cluster the agent left at the centre,
  but the zone clauses still gate the win).
* `units_lost_lte: 3` (medium + hard) ⇒ attrition cap so a sloppy
  intercept that bleeds the line busts even if the zone clauses
  hold.
* `has_building: fact` ⇒ the fact survival clause stays as an
  always-on safety net.
* `within_ticks: 2700` paired with `after_ticks: 2701` ⇒ a non-
  finisher is a real reachable timeout LOSS, never a draw.
* Hard tier defines TWO agent spawn_point groups (NORTH base y=14 /
  SOUTH base y=26) so the lane geometry flips per seed; the win
  predicate is `any_of` over the two matching three-zone layouts,
  enforcing the doctrine over a memorised single-zone trio.

These tests prove deterministically (no model / no network):

* the intended 2/2/2 split distributed-defence policy WINS every
  level + every hard seed;
* stall, concentrate-NORTH, concentrate-MID, and concentrate-SOUTH
  all LOSE (real LOSS, not a draw);
* the hard tier defines ≥2 spawn_point groups so a memorised
  lane-zone trio cannot generalise across seeds.
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


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. Tanks stay at central staging, never reach the
    (25, *) defensive zones; auto-engagement at stance:3 racks up
    some kills as the MID band arrives, but the N and S waves walk
    untouched to the fact and raze it → LOSS."""
    return [C.observe()]


def _own_tank_ids(rs):
    own = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
    return sorted(str(u["id"]) for u in own)


def _own_tank_centroid_y(rs):
    own = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
    if not own:
        return 20
    return sum(u["cell_y"] for u in own) // len(own)


def make_concentrate(zone_y):
    """Send ALL tanks to the single defensive zone at (25, zone_y).
    Satisfies ONE zone clause; the other two clauses (and the
    unopposed-lane fact-survival clause) both fail → LOSS."""

    def policy(rs, C):
        ids = _own_tank_ids(rs)
        if not ids:
            return [C.observe()]
        return [C.attack_move(ids, target_x=25, target_y=zone_y)]

    return policy


def make_intended():
    """Stable per-tank 2/2/2 assignment across the three matching
    defensive zones (N / MID / S relative to the agent's actual base
    latitude — read from the fact's cell_y so the hard-tier spawn
    rotation generalises). On the first call, sort tank ids, detect
    base latitude from the fact position, and assign 2 tanks per
    lane. Reissue commands only for tanks still alive."""

    state = {"assignments": None}

    def _zone_ys(rs):
        # Detect base latitude from the tank centroid (the tanks
        # always start on the base row; `buildings_summary` is not
        # exposed in `RustObsAdapter.render_state` so we rely on the
        # tank cluster's median y on turn 1).
        base_y = _own_tank_centroid_y(rs)
        # Outer lanes ±6 from base (NORTH base y=14 ⇒ 2/14/26;
        # SOUTH base y=26 ⇒ 14/26/38; default base y=20 ⇒ 8/20/32).
        if base_y <= 16:
            return (2, 14, 26)
        if base_y >= 24:
            return (14, 26, 38)
        return (8, 20, 32)

    def policy(rs, C):
        ids = _own_tank_ids(rs)
        if not ids:
            return [C.observe()]
        if state["assignments"] is None:
            n = len(ids)
            two = max(1, n // 3)
            n_zone, m_zone, s_zone = _zone_ys(rs)
            state["assignments"] = {
                ("N", n_zone): ids[:two],
                ("M", m_zone): ids[two:2 * two],
                ("S", s_zone): ids[2 * two:],
            }
        alive = set(ids)
        cmds = []
        for (_lane, zy), tids in state["assignments"].items():
            live = [i for i in tids if i in alive]
            if live:
                cmds.append(C.attack_move(live, target_x=25, target_y=zy))
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
    # Rusher bot wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        # `rusher`: bands converge on the agent centroid. The
        # distributed-defence test is enforced by the THREE
        # `units_in_region_gte` zone clauses in the win predicate
        # (a concentration satisfies one zone and fails two);
        # `hunt` was smoke-tested but leaves the e1 bands idle at
        # spawn for most of the tick budget so neither the kill bar
        # nor the engagement-mediated zone-coverage attrition fire
        # (see pack docstring).
        assert str(bot).lower() == "rusher", (lvl, bot)


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
def test_every_level_requires_three_defensive_zones(level):
    """The distribution axis must be load-bearing: every level's win
    condition must require ≥3 `units_in_region_gte` clauses (the
    three concurrent-lane defensive-zone discriminators). Any
    concentration policy fails at least two of them."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert flat.count("units_in_region_gte") >= 3, (level, win)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the base latitude (and lane geometry) flips by seed
    (anti-memorisation of an absolute lane trio — the distributed-
    defence doctrine must generalise)."""
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
def test_intended_split_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended 2/2/2 distributed defence "
            f"must WIN; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Stall (observe only) — tanks never reach any defensive zone,
    and the three concurrent waves (especially the unopposed N + S
    bands) raze the fact → real LOSS on `not has_building:fact`
    and/or all three zone clauses empty + `after_ticks` deadline."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not draw); "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_north_loses_every_level_and_seed(level):
    """Concentrate-NORTH: all 6 tanks at the N defensive zone. The
    N zone clause satisfies but MID and SOUTH zone clauses fail
    (0 < 2), AND the unopposed MID + SOUTH waves raze the fact
    → real LOSS."""
    c = compile_level(load_pack(PACK), level)
    # Use the N zone latitude matched to default base (easy/medium
    # base at y=20 ⇒ N=8). On hard the default N (8) is OFF either
    # candidate layout, so concentrate-NORTH at y=8 fails ALL zone
    # clauses on both spawns — strictly the harder failure mode.
    pol = make_concentrate(8)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-NORTH: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_mid_loses_every_level_and_seed(level):
    """Concentrate-MID: all 6 tanks at the MID defensive zone
    (25,20). The N and S zone clauses fail and the N + S waves walk
    around the central cluster to raze the fact → LOSS."""
    c = compile_level(load_pack(PACK), level)
    pol = make_concentrate(20)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-MID: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_concentrate_south_loses_every_level_and_seed(level):
    """Concentrate-SOUTH: all 6 tanks at the S defensive zone. Same
    failure mode as concentrate-NORTH but mirrored — N and MID zone
    clauses fail and the unopposed N + MID waves raze the fact."""
    c = compile_level(load_pack(PACK), level)
    pol = make_concentrate(32)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} concentrate-SOUTH: must LOSE; got "
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
