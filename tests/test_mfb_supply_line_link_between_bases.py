"""mfb-supply-line-link-between-bases scenario pack, full loop on Rust.

ACTION capability — CORRIDOR INTERDICTION / supply-line protection.
Two bases (WEST fact+proc, EAST fact+proc) are joined by a single
supply corridor (y=20, or the spawn-rotated hard-tier latitudes
y=12 / y=28). A four-tank mobile squad must be POSITIONED ON the
corridor midpoint so the `rusher` raider bands — converging on the
agent centroid — are drawn into the corridor and destroyed there.

The load-bearing predicate is `units_in_region_gte:{x:50,y:<corridor>,
radius:6,n:1}`: a base garrison (x=15 / x=85) or a 2/2 split is 35
cells from the corridor midpoint and can never satisfy it. The
`after_ticks` deadline (fixed-step ~90 ticks/turn, no interrupts) is
the teeth — a non-winner emits a real reachable LOSS, not a DRAW.

The bar:

* `units_in_region_gte` corridor clause ⇒ only a squad ON the
  corridor midpoint can win; garrison / split / stall fail it.
* `units_killed_gte: N` (easy 3, medium/hard 6) ⇒ the corridor
  squad must actually interdict the rusher bands.
* `building_count_gte:{type:fact,n:2}` + `{type:proc,n:2}` ⇒ both
  bases must keep their construction yard AND refinery (present-
  tense pair-alive checks, not `has_building`).
* `within_ticks: 2300` paired with `after_ticks: 2301` ⇒ a non-
  finisher is a real reachable timeout LOSS, never a draw.
* Hard tier defines TWO agent spawn_point groups (NORTH corridor
  y=12 / SOUTH corridor y=28); the win `any_of` PAIRS each
  candidate corridor midpoint with a `building_in_region` base-
  position check, so a wrong-latitude (or memorised-(50,20))
  patrol satisfies neither branch.

These tests prove deterministically (no model / no network):

* the intended corridor-patrol policy WINS every level + every
  hard seed;
* stall, garrison-WEST, garrison-EAST, and split-2/2 all LOSE
  (real LOSS, not a draw);
* on hard, a wrong-latitude patrol and a memorised-(50,20) patrol
  both LOSE;
* the hard tier defines ≥2 spawn_point groups so a memorised
  absolute corridor midpoint cannot generalise across seeds.
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

PACK = PACKS_DIR / "mfb-supply-line-link-between-bases.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def _tank_ids(rs):
    own = [u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"]
    return sorted(str(u["id"]) for u in own)


def _corridor_y(rs):
    """The corridor latitude = the WEST fact's cell_y (the bases sit
    on the corridor). Read from `own_buildings` so the hard-tier
    spawn rotation (y=12 / y=28) generalises."""
    facts = [b for b in rs.get("own_buildings", []) if b.get("type") == "fact"]
    return facts[0]["cell_y"] if facts else 20


def stall(rs, C):
    """Observe-only. The stance:2 tanks never advance off the
    shoulder, so the corridor-midpoint clause is never satisfied and
    the `after_ticks` deadline fires → real LOSS."""
    return [C.observe()]


def make_corridor():
    """Intended play: hold the squad on the corridor at its midpoint
    (x=50, matching base latitude) and INTERDICT — engage the raider
    band nearest the corridor midpoint, falling back to attack_move
    onto the midpoint when none is visible. The rusher bands converge
    on the centroid (on the corridor), so the squad engages them in
    the corridor and a tank stays inside the (50, corridor_y) region
    at win time → WIN."""

    def policy(rs, C):
        ids = _tank_ids(rs)
        if not ids:
            return [C.observe()]
        y = _corridor_y(rs)
        raiders = [
            e for e in rs.get("enemy_summary", [])
            if e.get("type") == "e1" and e.get("id") is not None
        ]
        if raiders:
            raiders.sort(
                key=lambda e: (e["cell_x"] - 50) ** 2 + (e["cell_y"] - y) ** 2
            )
            return [C.attack_unit(ids, str(raiders[0]["id"]))]
        return [C.attack_move(ids, target_x=50, target_y=y)]

    return policy


def make_garrison(base_x):
    """Garrison the whole squad at one base (x=15 WEST or x=85 EAST).
    No tank is ever within radius 6 of the corridor midpoint
    (distance 35) → corridor clause empty → cannot win → LOSS."""

    def policy(rs, C):
        ids = _tank_ids(rs)
        if not ids:
            return [C.observe()]
        return [C.attack_move(ids, target_x=base_x, target_y=_corridor_y(rs))]

    return policy


def make_split():
    """Split the squad 2/2 across the two bases. The corridor itself
    is undefended — the midpoint clause stays empty → LOSS."""

    state = {"assign": None}

    def policy(rs, C):
        ids = _tank_ids(rs)
        if not ids:
            return [C.observe()]
        y = _corridor_y(rs)
        if state["assign"] is None:
            half = len(ids) // 2
            state["assign"] = (ids[:half], ids[half:])
        west, east = state["assign"]
        return [
            C.attack_move(west, target_x=15, target_y=y),
            C.attack_move(east, target_x=85, target_y=y),
        ]

    return policy


def make_wrong_latitude():
    """HARD only: patrol the OTHER candidate corridor midpoint (the
    one NOT matching the agent's base latitude). The hard win
    `any_of` pairs each midpoint with a base-position check, so this
    satisfies neither branch → LOSS."""

    def policy(rs, C):
        ids = _tank_ids(rs)
        if not ids:
            return [C.observe()]
        y = _corridor_y(rs)
        wrong = 28 if y <= 16 else 12
        return [C.attack_move(ids, target_x=50, target_y=wrong)]

    return policy


def make_memorised_y20():
    """HARD only: a memorised "always patrol (50,20)" plan. y=20 is
    neither candidate corridor midpoint (y=12 / y=28) → LOSS."""

    def policy(rs, C):
        ids = _tank_ids(rs)
        if not ids:
            return [C.observe()]
        return [C.attack_move(ids, target_x=50, target_y=20)]

    return policy


# ── structural checks (no engine) ────────────────────────────────────


def test_pack_loads_and_metadata_is_complete():
    pack = load_pack(PACK)
    assert pack.meta.id == "mfb-supply-line-link-between-bases"
    assert pack.meta.capability == "action"
    anchors = pack.meta.benchmark_anchor or []
    joined = " | ".join(a.lower() for a in anchors)
    assert "military supply-line protection" in joined, anchors
    assert "logistics route security" in joined, anchors
    assert "sc2 harass-route defense" in joined, anchors
    # `rusher` bot wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "rusher", (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (~90 ticks/step
    fixed-step — the pack declares no interrupts)."""
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
def test_every_level_requires_the_corridor_region(level):
    """The corridor-positioning axis must be load-bearing: every
    level's win condition must require a `units_in_region_gte`
    clause (the corridor-midpoint discriminator)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    assert "units_in_region_gte" in str(win), (level, win)
    # Both bases must be required intact (fact:2 AND proc:2).
    flat = str(win)
    assert "building_count_gte" in flat, (level, win)
    assert "'type': 'fact'" in flat and "'type': 'proc'" in flat, win


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the corridor latitude (and the matching midpoint) flips by
    seed — a memorised absolute corridor midpoint cannot generalise."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (generator arena 160×60, playable x∈[4,156),
    # y∈[4,56)).
    for a in c.scenario.actors:
        x, y = a.position
        assert 4 <= x < 156 and 4 <= y < 56, (a.type, a.position)


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_corridor_patrol_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_corridor(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended corridor patrol must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Stall (observe only) — the stance:2 squad never advances off
    the shoulder, the corridor clause stays empty, and the
    `after_ticks` deadline fires → real LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not "
            f"draw); got {r.outcome} (tick={r.signals.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_garrison_west_loses_every_level_and_seed(level):
    """Garrison-WEST: all four tanks parked at the west base (x=15) —
    35 cells from the corridor midpoint → corridor clause empty →
    cannot win → real LOSS on the deadline."""
    c = compile_level(load_pack(PACK), level)
    pol = make_garrison(15)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} garrison-WEST: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_garrison_east_loses_every_level_and_seed(level):
    """Garrison-EAST: symmetric to garrison-WEST (tanks at x=85)."""
    c = compile_level(load_pack(PACK), level)
    pol = make_garrison(85)
    for seed in SEEDS:
        r = run_level(c, pol, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} garrison-EAST: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_split_two_two_loses_every_level_and_seed(level):
    """Split-2/2: two tanks at each base — the corridor itself is
    undefended → corridor clause empty → real LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_split(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} split-2/2: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick})"
        )


def test_hard_wrong_latitude_patrol_loses_every_seed():
    """HARD: patrolling the OTHER candidate corridor midpoint (not
    the one matching the agent's base latitude) satisfies neither
    `any_of` branch → real LOSS. A memorised absolute midpoint
    cannot generalise across the seed-rotated corridor latitude."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        r = run_level(c, make_wrong_latitude(), seed=seed)
        assert r.outcome == "loss", (
            f"hard seed{seed} wrong-latitude patrol: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick})"
        )


def test_hard_memorised_y20_patrol_loses_every_seed():
    """HARD: a memorised "always patrol (50,20)" plan — y=20 is
    neither candidate corridor midpoint (y=12 / y=28) → real LOSS."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        r = run_level(c, make_memorised_y20(), seed=seed)
        assert r.outcome == "loss", (
            f"hard seed{seed} memorised-(50,20) patrol: must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_corridor(), seed=2)
    b = run_level(c, make_corridor(), seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
