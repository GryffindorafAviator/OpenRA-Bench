"""def-with-ambush scenario pack — Rust engine full loop.

REASONING capability — FOG-AMBUSH doctrine: 2tnk flankers pre-placed
in concealment OFF the direct enemy axis (at (40,16)/(40,24) on
easy/medium; at the corresponding off-row positions for the
seed-chosen base latitude on hard) must HOLD their hidden crossfire
positions and let the rusher band advance into the L-ambush. Breaking
the ambush — charging the flankers forward to meet the band in the
open, or piling them back onto the construction yard — abandons the
crossfire envelope: the thin force is out-attritioned head-on / ground
down against the building.

Engine recalibration note (movement fixes): the engine fixes —
attack_unit on out-of-sight targets paths at real speed (no teleport)
and a moving unit fires AND takes fire en route (no
sprint-invincibility) — shifted combat balance bench-wide. The old
"centralise to the choke" wrong policy inverted: a tank cluster
funnelled at a choke is now genuinely strong play, so it was retired.
The pack now discriminates the held crossfire ambush against the two
wrong plays the engine fixes leave load-bearing:

* CHARGE — attack-move all flankers EAST at the band. With no
  sprint-invincibility the charging tanks take fire while closing,
  string out, and meet the heavier band head-on with no crossfire →
  out-attritioned, kill bar missed.
* PILE-ON-FACT — pull the flankers back and stack them on the
  construction yard. The crossfire envelope is abandoned; the thin
  stacked force is crowded against the building and ground down →
  kill bar missed / fact razed / attrition cap busted.

The load-bearing discriminator is POSITIONAL DISCIPLINE — the intended
policy KEEPS the flankers at their starting cells (do nothing, or
explicitly stop). The flankers carry stance:2 (Defend) — they hold
their post and auto-fire on any enemy that enters weapon range; the
crossfire rows are placed 4 cells off the band's lane (inside the
~4-cell 2tnk weapon range) so the band walks into auto-engage range
as it passes. The model does NOT need to issue attack orders; it ONLY
needs to NOT MOVE the flankers.

The bar:

* `building_count_gte:{type: fact, n: 1}` — fact must survive.
* `units_killed_gte: K` — real engagement bar matched per tier
  (6 easy, 8 medium, 8 hard) so an active-HOLD crossfire clears it
  while a charging / piled-on force cannot.
* `units_lost_lte: N` — the attrition cap (2 easy, 3 medium/hard)
  that bites CHARGE and PILE-ON, both of which trade ≥3/≥4 tanks
  while a clean HOLD trades only ~2.
* `within_ticks: 2400` paired with `after_ticks: 2401` — a non-
  finisher is a real reachable timeout LOSS, never a draw.
* `not building_count_gte:{type:fact,n:1}` and `not units_lost_lte:N`
  in the fail clause — fact-razed and over-attrition are real LOSSES.

These tests prove deterministically (no model / no network):

* the intended HOLD-ambush-positions play WINS every level + every
  hard seed;
* charge and pile-on-fact both LOSE every level and every hard seed
  (a real LOSS, not a draw);
* the hard tier defines ≥2 spawn_point groups so a memorised
  absolute crossfire cell cannot generalise.
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

PACK = PACKS_DIR / "def-with-ambush.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def hold(rs, C):
    """HOLD the ambush positions — observe-only. stance:2 (Defend)
    on the pre-placed 2tnk flankers means each tank holds its post
    and auto-fires on any enemy that enters weapon range; the
    crossfire rows are 4 cells off the band lane, INSIDE the
    ~4-cell 2tnk weapon range, so the engine auto-engages the band
    from BOTH flanks as it walks past. The model only needs to NOT
    MOVE the flankers — the intended capability is positional
    discipline, not active micro. WINS every level + every hard
    seed."""
    return [C.observe()]


def charge(rs, C):
    """Charge the flankers OUT of the ambush — attack-move them
    EAST at the band. The crossfire envelope is abandoned on turn
    1; with the engine no-sprint-invincibility fix the charging
    tanks take fire while closing, string out along the lane, and
    meet the heavier band head-on with no crossfire support. They
    are out-attritioned every level — kill bar missed, ≥3 tanks
    lost → LOSS every level + every hard seed."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    return [C.attack_move(ids, target_x=115, target_y=20)]


def pile_on_fact(rs, C):
    """Pile the flankers back onto the construction yard — move
    every 2tnk to the fact cell (the "defend the high-value asset"
    pattern). The crossfire envelope at the lane is abandoned; the
    thin force is stacked against the building, crowded so it
    cannot bring all its weapons to bear, and is ground down by the
    band — kill bar missed / fact razed / attrition cap busted →
    LOSS every level + every hard seed."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    # Target the agent fact's actual cell (works for both NORTH and
    # SOUTH spawn on hard); fall back to (10,20).
    facts = [
        b for b in (rs.get("own_buildings", []) or [])
        if (b.get("type") or "").lower() == "fact"
    ]
    if facts:
        tx, ty = int(facts[0]["cell_x"]), int(facts[0]["cell_y"])
    else:
        tx, ty = 10, 20
    return [C.move_units(ids, target_x=tx, target_y=ty)]


# ── structural checks (no engine) ────────────────────────────────────


def test_pack_loads_and_metadata_is_complete():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-with-ambush"
    assert pack.meta.capability == "reasoning"
    anchors = [str(a).lower() for a in (pack.meta.benchmark_anchor or [])]
    assert anchors, "benchmark_anchor must be non-empty"
    # The brief's three explicit anchors.
    assert any("sc2" in a and "hidden" in a for a in anchors), anchors
    assert any("ambush" in a for a in anchors), anchors
    assert any("fog" in a for a in anchors), anchors
    # rusher bot wired through to the engine for every level (centroid
    # concentration is what makes the band drive WEST on the base row
    # past the crossfire cells).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = (
            getattr(enemy, "bot_type", None)
            or getattr(enemy, "bot", None)
        )
        assert str(bot).lower() == "rusher", (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns (~90 ticks/step
    fixed-step or ~60 in interrupt mode)."""
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
def test_every_level_has_fact_survival_in_fail_clause(level):
    """The fact-razed condition must be a real LOSS clause (a
    piled-on force on medium/hard lets the band reach and raze the
    fact)."""
    c = compile_level(load_pack(PACK), level)
    fc = c.fail_condition.model_dump(exclude_none=True)
    flat = str(fc)
    assert "building_count_gte" in flat or "has_building" in flat, fc


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_attrition_cap_in_win_and_fail(level):
    """Every tier carries a `units_lost_lte` attrition cap in BOTH
    the win and fail clauses — the cap is the load-bearing
    discriminator that makes CHARGE / PILE-ON a real LOSS (they
    trade more tanks than a clean HOLD)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    fail = c.fail_condition.model_dump(exclude_none=True)
    win_cap = next(
        (clause["units_lost_lte"] for clause in win.get("all_of", [])
         if "units_lost_lte" in clause),
        None,
    )
    assert win_cap is not None, win
    has_fail_cap = any(
        (clause.get("not") or {}).get("units_lost_lte") == win_cap
        for clause in fail.get("any_of", []) or []
    )
    assert has_fail_cap, fail


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_enforces_fact_survival_in_win(level):
    """Win must require the fact to survive — the brief-specified
    `building_count_gte:{type:fact,n:1}` clause."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "building_count_gte" in flat, win
    assert "fact" in flat.lower(), win


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the base latitude (and the matching crossfire cells) flips
    per seed — a memorised absolute "(40,16)/(40,24)" plan cannot
    generalise; the hold-the-hidden-position doctrine must."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # The enemy band declares the matching spawn_point so exactly one
    # concentrated band places at the active base latitude.
    enemy_groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "enemy" and a.spawn_point is not None
    }
    assert enemy_groups == {0, 1}, enemy_groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended HOLD WINS every level + every hard seed ───────


@pytest.mark.parametrize("level", LEVELS)
def test_hold_wins_every_level_and_seed(level):
    """HOLD-ambush (observe-only — stance:2 + crossfire geometry
    does the work) WINS every level on every seed. This is the
    load-bearing solvency check: the intended capability is
    reachable inside the tick budget on every seed."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, hold, seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: HOLD-ambush must WIN; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / break-the-ambush policy LOSES ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_charge_loses_every_level_and_seed(level):
    """Charge attack-moves the flankers EAST out of the crossfire
    positions to meet the band in the open. With no
    sprint-invincibility they take fire while closing, string out,
    and fight the heavier band head-on with no crossfire — out-
    attritioned, kill bar missed. LOSS every level + every seed
    (a real LOSS, not a draw)."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, charge, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} charge: must LOSE (real fail, "
            f"not draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_pile_on_fact_loses_every_level_and_seed(level):
    """Pile-on-fact moves the flankers OUT of the hidden crossfire
    positions and stacks them on the construction yard. The
    crossfire envelope is abandoned; the thin force is crowded
    against the building and ground down by the band (kill bar
    missed / fact razed / attrition cap busted). LOSS every level
    + every seed (a real LOSS, not a draw)."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, pile_on_fact, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} pile-on-fact: must LOSE "
            f"(real fail, not draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_hold_run_is_deterministic_on_medium():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, hold, seed=2)
    b = run_level(c, hold, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
