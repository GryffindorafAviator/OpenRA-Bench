"""def-with-ambush scenario pack — Rust engine full loop.

REASONING capability — FOG-AMBUSH doctrine: defenders pre-placed in
concealment OFF the direct enemy axis (at (40,15)/(40,25) on
easy/medium; at the corresponding off-row positions for the
seed-chosen base latitude on hard) must HOLD their hidden firing
positions and let the rusher band advance into the L-ambush
crossfire. Centralising the ambush force to the fact or to the
choke abandons the flanking fire envelope: the stacked cluster at
the choke is out-attritioned by the band's tank component on
medium/hard, and on easy the ambushers take attrition while moving
WEST through the band's fire axis (y=20).

The load-bearing discriminator is POSITIONAL DISCIPLINE — the
intended policy KEEPS the ambushers at their starting cells (do
nothing, or set_stance to 3, or explicitly stop). stance:3 +
sight_range≥5 makes auto-engagement load-bearing once the band
walks into range — the model does NOT need to issue attack
orders; it ONLY needs to NOT MOVE the ambushers.

The bar:

* `building_count_gte:{type: fact, n: 1}` — fact must survive
  (centralise lets the band raze it).
* `units_killed_gte: K` — real engagement bar matched per tier
  (6 easy, 8 medium, 8 hard) so an active-HOLD ambush clears it
  while a centralise pile-up cannot.
* `units_lost_lte: 0` (easy only) — the attrition cap that bites
  CENTRALISE on easy where the lighter band cannot raze the fact
  fast enough on its own (without this clause, easy's centralise
  also wins on the kill bar because the ambushers fire en route).
* `within_ticks: 2400` paired with `after_ticks: 2401` — a non-
  finisher is a real reachable timeout LOSS, never a draw.
* `not building_count_gte:{type:fact,n:1}` in the fail clause —
  fact-razed is a real LOSS.

These tests prove deterministically (no model / no network):

* the intended HOLD-ambush-positions play WINS every level + every
  hard seed;
* centralise-to-fact and centralise-to-choke both LOSE every level
  and every hard seed (a real LOSS, not a draw);
* the hard tier defines ≥2 spawn_point groups so a memorised
  absolute ambush cell cannot generalise.
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
    """HOLD the ambush positions — observe-only. stance:3 on the
    pre-placed 2tnk ambushers + sight_range≥5 (the y=15/y=25 firing
    positions cover the y=20 band lane at the edge of sight) means
    the engine auto-engages the band as it walks past the ambush
    cells. The model only needs to NOT MOVE the ambushers — the
    intended capability is positional discipline, not active
    micro. WINS every level + every hard seed."""
    return [C.observe()]


def centralise_to_fact(rs, C):
    """Centralise all ambushers to the fact cell (the "defend the
    high-value asset" pattern). Moves the 2tnk ambushers WEST out
    of their (40,15)/(40,25) firing positions; the flanking
    crossfire envelope is abandoned. On easy the move crosses the
    band's fire axis (y=20) and at least 1 tank is hit → attrition
    cap busted → LOSS. On medium/hard the stacked cluster at the
    fact cannot out-attrition the band's tank component → fact
    razed → LOSS."""
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


def centralise_to_choke(rs, C):
    """Centralise the ambushers to the choke (5 cells east of the
    fact, where the static defender sits). Same pathology as
    centralise-to-fact: the flanking crossfire envelope is
    abandoned, the centralised cluster is out-attritioned by the
    band's tank component, fact razed. LOSS every level + every
    hard seed."""
    units = [
        u for u in rs.get("units_summary", []) if u.get("type") == "2tnk"
    ]
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    facts = [
        b for b in (rs.get("own_buildings", []) or [])
        if (b.get("type") or "").lower() == "fact"
    ]
    if facts:
        tx, ty = int(facts[0]["cell_x"]) + 5, int(facts[0]["cell_y"])
    else:
        tx, ty = 15, 20
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
    # past the ambush cells).
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
    """The fact-razed condition must be a real LOSS clause (the
    centralise discriminator on medium/hard is precisely that the
    band reaches and razes the fact)."""
    c = compile_level(load_pack(PACK), level)
    fc = c.fail_condition.model_dump(exclude_none=True)
    flat = str(fc)
    assert "building_count_gte" in flat or "has_building" in flat, fc


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
    so the base latitude (and the matching ambush cells) flips per
    seed — a memorised absolute "(40,15)/(40,25)" plan cannot
    generalise; the hold-the-hidden-position doctrine must."""
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


# ── solvency: intended HOLD WINS every level + every hard seed ───────


@pytest.mark.parametrize("level", LEVELS)
def test_hold_wins_every_level_and_seed(level):
    """HOLD-ambush (observe-only — stance:3 + sight does the work)
    WINS every level on every seed. This is the load-bearing
    solvency check: the intended capability is reachable inside
    the tick budget on every seed."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, hold, seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: HOLD-ambush must WIN; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


# ── no-cheat: every wrong / centralise policy LOSES (real LOSS) ──────


@pytest.mark.parametrize("level", LEVELS)
def test_centralise_to_fact_loses_every_level_and_seed(level):
    """Centralise-to-fact moves the ambushers OUT of the hidden
    firing positions and into the fact. The flanking crossfire
    envelope is abandoned; on easy the move crosses the band's
    fire axis (attrition cap busted), on medium/hard the cluster
    is out-attritioned (fact razed). LOSS every level + every
    seed (a real LOSS, not a draw)."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, centralise_to_fact, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} centralise-to-fact: must LOSE "
            f"(real fail, not draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"kills={r.signals.units_killed}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_centralise_to_choke_loses_every_level_and_seed(level):
    """Centralise-to-choke (move the ambushers to the static
    defender's cell) has the same pathology — the crossfire
    envelope is abandoned and the cluster is out-attritioned at
    close range. LOSS every level + every seed."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, centralise_to_choke, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} centralise-to-choke: must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
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
