"""def-position-expected-direction scenario pack, full loop on Rust.

REASONING capability — DEFENSE POSITIONING BASED ON DECLARED THREAT
DIRECTION. The model is TOLD where the enemy will attack from
(the objective brief states the intel: "rushers from the EAST" on
easy/medium; "from the side facing your base" on hard with a
seed-rotated N/S latitude). The decision is purely one of SPATIAL
COMMITMENT: invest the defence budget on the CORRECT lane and the
rush is blunted; commit the same budget on the WRONG lane and the
rushers walk past, the deadline expires (the wrong-direction pbox
cannot satisfy the region clause), and the run loses. Building no
defences at all also loses (no pbox).

Easy/medium run on a per-tier 96×40 arena with an OBSTACLE FENCE
(water bands flanking the western base on north and south) so the
only inbound corridor is the EAST mouth — the topology cue makes
the threat axis obvious without reading the brief. Hard rotates to
an open 96×40 arena with the agent base seed-varied N/S via
spawn_point.

The win predicate makes the directional axis load-bearing:

* `building_count_gte:{pbox, 3}` ⇒ three pillboxes total (forces a
  real defence commitment, not a token pbox);
* `building_in_region:{pbox, CORRECT_centre, r=5, 3}` ⇒ ALL THREE
  must sit in the correct region — a wrong-side triple cannot
  satisfy it (easy/medium centre is EAST of base at (24, 20); hard
  is `any_of` over (48, 6) and (48, 34));
* `building_count_gte:{fact, 1}` ⇒ the construction yard still
  stands (the live-frame predicate; `has_building:fact` accumulates
  across the episode and is useless as a still-standing gate);
* `own_units_gte: 3` ⇒ the throughput SLA (the four pre-placed
  defenders must mostly survive);
* `within_ticks: 5400` paired with `after_ticks: 5401` ⇒ a non-
  finisher is a real reachable timeout LOSS (max_turns 66 reaches
  tick 5943 in interrupt mode), never a draw.

These tests prove deterministically (no model / no network) that:

* the intended correct-axis pbox triple WINS every level + every
  hard seed;
* stall, build-defences-WRONG-direction, and build-no-defences ALL
  LOSE every level + every hard seed (a real LOSS, never a draw);
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


def _intended_target(fx, fy):
    """Resolve the intended pbox-triple centroid from the fact xy.

    Easy/medium: fact at (12, 20), defend EAST at (24, 20) — the
    open corridor between the obstacle fence and the eastern rush.
    Hard NORTH spawn: fact at (48, 12), defend NORTH at (48, 6).
    Hard SOUTH spawn: fact at (48, 28), defend SOUTH at (48, 34).
    """
    if fx < 30:  # easy / medium WEST base ⇒ EAST defence
        return fx + 12, fy
    if fy < 20:  # hard NORTH spawn ⇒ NORTH defence
        return fx, fy - 6
    return fx, fy + 6  # hard SOUTH spawn ⇒ SOUTH defence


def _wrong_target(fx, fy):
    """Inverse of _intended_target — the WRONG side a wrong-direction
    policy places its pbox at."""
    if fx < 30:  # easy / medium ⇒ wrong = WEST behind the base
        return max(4, fx - 6), fy
    if fy < 20:  # hard NORTH spawn ⇒ wrong = SOUTH (away from threat)
        return fx, fy + 6
    return fx, fy - 6  # hard SOUTH spawn ⇒ wrong = NORTH


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. Never builds anything; never satisfies the pbox
    count or the region clauses → reachable timeout LOSS."""
    return [C.observe()]


def make_intended():
    """Build THREE pbox in the CORRECT region of the agent base.

    Easy/medium fact at (12, 20) ⇒ defend EAST at (24, 20). Hard
    NORTH spawn at y=12 ⇒ defend NORTH at (48, 6). Hard SOUTH spawn
    at y=28 ⇒ defend SOUTH at (48, 34). Picks the side by reading
    fact x (a WEST base means EAST attack) then fact y (which
    latitude on hard)."""

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
        tx, ty = _intended_target(fx, fy)
        cmds = []
        if pbox_count < 3:
            if prod_items.count("pbox") + pbox_count < 3:
                cmds.append(C.build("pbox"))
            # Stagger placement so 3 pbox sit in a tight cluster on
            # the threat lane (all within radius 5 of (tx, ty)).
            # For easy/medium (EAST axis), stagger along y; for hard
            # (N/S axis), stagger along x.
            if fx < 30:
                # EAST axis ⇒ stagger y around ty
                dy = -2 + 2 * pbox_count
                cmds.append(C.place_building("pbox", tx, ty + dy))
            else:
                # N/S axis ⇒ stagger x around tx
                dx = -2 + 2 * pbox_count
                cmds.append(C.place_building("pbox", tx + dx, ty))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_defenses_wrong():
    """Build THREE pbox in the WRONG region (WEST of base on easy/
    medium when the intel says EAST; opposite latitude on hard).
    Satisfies `building_count_gte:pbox,3` but NEVER the
    `building_in_region` correct-axis clause ⇒ deadline LOSS — AND
    the correct lane is left naked so the rusher band tears through
    the starter defenders / fact (additional LOSS path on fact-loss
    or `own_units_gte`)."""

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
        wx, wy = _wrong_target(fx, fy)
        cmds = []
        if pbox_count < 3:
            if prod_items.count("pbox") + pbox_count < 3:
                cmds.append(C.build("pbox"))
            if fx < 30:
                dy = -2 + 2 * pbox_count
                cmds.append(C.place_building("pbox", wx, wy + dy))
            else:
                dx = -2 + 2 * pbox_count
                cmds.append(C.place_building("pbox", wx + dx, wy))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def no_defenses(rs, C):
    """Build no pillboxes ever. Never satisfies `building_count_gte:
    pbox,3` ⇒ reachable timeout LOSS. (Also typically dies on
    `building_count_gte:fact,1` as the unguarded rush razes the
    fact.)"""
    # Burn the budget on extra power plants — proves the agent had
    # cash to spend, just spent it on the wrong category. Place the
    # extra power plants well clear of the base footprint so the
    # placement orders don't collide with the existing structures.
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
        # Stagger placement away from base centroid; for the WEST
        # base (easy/medium) place east of the corridor mouth (x>=20)
        # well clear of fact/tent; for the N/S hard bases place down/up
        # along the spawn-matched lane edge.
        if fx < 30:
            cmds.append(C.place_building("powr", 22, 20 + 2 * n_powr))
        else:
            cmds.append(C.place_building("powr", fx - 6, fy + 2 * n_powr))
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
    must require BOTH ≥3 pbox AND those pbox in the correct region.

    Easy/medium: region centre is EAST of the agent base — (24, 20).
    Hard: `any_of` over (48, 6) and (48, 34) (NORTH/SOUTH per spawn).
    """
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "building_count_gte" in flat and "pbox" in flat, win
    assert "building_in_region" in flat, win
    region_xys = []

    def walk(node):
        if isinstance(node, dict):
            if "building_in_region" in node:
                v = node["building_in_region"]
                if (v or {}).get("type") == "pbox":
                    region_xys.append((int(v["x"]), int(v["y"])))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(win)
    assert region_xys, win
    if level == "hard":
        # hard must use any_of over the two correct region orientations
        # — the two N/S lanes that match the spawn-selected latitude.
        assert "any_of" in flat, win
        assert set(region_xys) == {(48, 6), (48, 34)}, region_xys
    else:
        # easy/medium fortify EAST of base (12, 20) ⇒ region centre
        # (24, 20) at the funnel mouth.
        assert region_xys == [(24, 20)], region_xys


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the threat axis rotates by seed (anti-memorisation)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (96×40 arena, cordon 2 ⇒ playable x∈[2..93],
    # y∈[2..37]).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 93 and 2 <= y <= 37, (a.type, a.position)


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_correct_axis_pbox_triple_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended correct-axis pbox-triple play "
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
        ("defenses_wrong", make_defenses_wrong),
        ("no_defenses", lambda: no_defenses),
    ],
)
def test_lazy_and_wrong_direction_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (no pbox), defenses-WRONG (3 pbox in WRONG region — WEST
    behind base on easy/medium, opposite-latitude on hard), and
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
