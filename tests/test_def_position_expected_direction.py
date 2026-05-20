"""def-position-expected-direction scenario pack, full loop on Rust.

REASONING capability — DEFENSE POSITIONING BASED ON DECLARED THREAT
DIRECTION. The model is TOLD where the enemy will attack from
(the objective brief states the intel: "rushers from the NORTH").
The decision is purely one of SPATIAL COMMITMENT: invest the
defence budget on the CORRECT lane and the rush is blunted; commit
the same budget on the WRONG lane (SOUTH) and the rushers walk past,
the deadline expires (the wrong-direction pbox cannot satisfy the
NORTH region clause), and the run loses. Building no defences at
all also loses (no pbox).

The win predicate makes the directional axis load-bearing:

* `building_count_gte:{pbox, 3}` ⇒ three pillboxes total (forces a
  real defence commitment, not a token pbox);
* `building_in_region:{pbox, NORTH_centre, r=6, 3}` ⇒ ALL THREE must
  sit in the NORTH region — a SOUTH-built triple cannot satisfy it;
* `has_building: fact` ⇒ the construction yard still stands;
* `own_units_gte: 3` ⇒ the throughput SLA (the four pre-placed
  defenders must mostly survive);
* `within_ticks: 5400` paired with `after_ticks: 5401` ⇒ a non-
  finisher is a real reachable timeout LOSS (max_turns 62 reaches
  tick 5583 in interrupt mode), never a draw.

These tests prove deterministically (no model / no network) that:

* the intended NORTH-pbox triple WINS every level + every hard seed;
* stall, build-defences-SOUTH (wrong direction), and build-no-
  defences ALL LOSE every level + every hard seed (a real LOSS,
  never a draw);
* `after_ticks` is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups so the threat axis
  rotates by seed (single-cell memorisation cannot generalise).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-position-expected-direction.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── helpers ──────────────────────────────────────────────────────────


def _fact_xy(rs):
    fact = next(
        (b for b in (rs.get("own_buildings") or [])
         if b.get("type") == "fact"),
        None,
    )
    if fact is None:
        return None
    return int(fact["cell_x"]), int(fact["cell_y"])


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. Never builds anything; never satisfies the pbox
    count or the region clauses → reachable timeout LOSS."""
    return [C.observe()]


def make_intended():
    """Build THREE pbox in the NORTH region of the agent base. The
    NORTH side is `fy - 6` from the base (for easy/medium the base
    is at y=20 so pbox at y≈14; for the hard NORTH spawn at y=12
    pbox at y≈6; for the hard SOUTH spawn at y=28 pbox at y≈34).
    Detects spawn by reading fact y and picks the side AWAY from
    the centre (rushers come from the OUTSIDE on hard; on easy/
    medium they come from the NORTH so the centre IS the rush
    direction)."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        # Pick the defence side: for easy/medium fact at y=20 →
        # NORTH lane (fy-6 = 14). For hard NORTH spawn (fy=12) →
        # NORTH (fy-6 = 6). For hard SOUTH spawn (fy=28) → SOUTH
        # (fy+6 = 34). Heuristic: if fact y is in the SOUTH half
        # (y>20), defend SOUTH; otherwise defend NORTH.
        if fy > 20:
            target_y = fy + 6  # SOUTH of base
        else:
            target_y = fy - 6 if fy >= 12 else fy - 6  # NORTH of base
        cmds = []
        if pbox_count < 3:
            if prod_items.count("pbox") + pbox_count < 3:
                cmds.append(C.build("pbox"))
            # Stagger placement so 3 pbox sit in a tight cluster on
            # the threat lane (all within radius 6 of (fx, target_y)).
            dx = -2 + 2 * pbox_count
            cmds.append(
                C.place_building("pbox", fx + dx, target_y)
            )
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_defenses_south():
    """Build THREE pbox in the WRONG region (the SOUTH side of the
    base when the intel says NORTH, and vice versa for the hard SOUTH
    spawn). Satisfies `building_count_gte:pbox,3` but NEVER the
    `building_in_region` NORTH clause ⇒ deadline LOSS — AND the
    NORTH lane is left naked so the rusher band tears through the
    starter defenders (additional LOSS path on `own_units_gte`)."""

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")
        prod = rs.get("production") or []
        prod_items = [
            p.get("item") for p in prod if isinstance(p, dict)
        ]
        xy = _fact_xy(rs)
        if xy is None:
            return [C.observe()]
        fx, fy = xy
        # WRONG side: invert the intended heuristic.
        if fy > 20:
            wrong_y = fy - 6  # NORTH of a SOUTH base ⇒ wrong
        else:
            wrong_y = fy + 6  # SOUTH of a NORTH base ⇒ wrong
        cmds = []
        if pbox_count < 3:
            if prod_items.count("pbox") + pbox_count < 3:
                cmds.append(C.build("pbox"))
            dx = -2 + 2 * pbox_count
            cmds.append(
                C.place_building("pbox", fx + dx, wrong_y)
            )
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def no_defenses(rs, C):
    """Build no pillboxes ever. Never satisfies `building_count_gte:
    pbox,3` ⇒ reachable timeout LOSS. (Also typically dies on
    `own_units_gte` as the rush eats the defenders.)"""
    # Burn the budget on extra power plants — proves the agent had
    # cash to spend, just spent it on the wrong category.
    own_b = rs.get("own_buildings") or []
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    n_powr = sum(1 for b in own_b if b.get("type") == "powr")
    xy = _fact_xy(rs)
    if xy is None:
        return [C.observe()]
    fx, fy = xy
    cmds = []
    if n_powr < 5 and "powr" not in prod_items:
        cmds.append(C.build("powr"))
        cmds.append(
            C.place_building("powr", fx - 4, fy + 2 + n_powr)
        )
    if not cmds:
        cmds.append(C.observe())
    return cmds


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-position-expected-direction"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert any("ERQA" in a for a in anchors), anchors
    assert any("MicroRTS" in a for a in anchors), anchors
    assert any("military" in a.lower() for a in anchors), anchors
    assert any("intel" in a.lower() for a in anchors), anchors
    # Rusher bot wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
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
def test_every_level_enforces_pbox_count_and_region(level):
    """The directional axis must be load-bearing: the win condition
    must require BOTH ≥3 pbox AND those pbox in the correct region."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "building_count_gte" in flat and "pbox" in flat, win
    assert "building_in_region" in flat, win
    # The placement region must be NORTH-of-base (low y) on
    # easy/medium, and `any_of` over NORTH+SOUTH on hard.
    region_ys = []

    def walk(node):
        if isinstance(node, dict):
            if "building_in_region" in node:
                v = node["building_in_region"]
                if (v or {}).get("type") == "pbox":
                    region_ys.append(int(v["y"]))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(win)
    assert region_ys, win
    if level == "hard":
        # hard must use any_of over the two correct region orientations.
        assert "any_of" in flat, win
        assert set(region_ys) == {6, 34}, region_ys
    else:
        # easy/medium fortify NORTH of base (y=20) ⇒ region centre y=14.
        assert region_ys == [14], region_ys


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the threat axis rotates by seed (anti-memorisation)."""
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
def test_intended_north_pbox_triple_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended NORTH-pbox-triple play "
            f"must WIN; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("defenses_south", make_defenses_south),
        ("no_defenses", lambda: no_defenses),
    ],
)
def test_lazy_and_wrong_direction_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (no pbox), defenses-SOUTH (3 pbox in WRONG region), and
    no-defenses (no pbox, just power) must ALL LOSE on every level +
    every seed — a real reachable timeout LOSS, never a draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=3)
    b = run_level(c, make_intended(), seed=3)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
