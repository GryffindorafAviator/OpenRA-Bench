"""mcv-deploy-defensible-site scenario family, full loop on Rust.

The pack tests OPENING site choice under a KNOWN threat axis: the
agent's MCV starts mid-west, a rusher band is already committed from
the NE corner, and two candidate deploy sites exist:

* the SAFE south-west corner (≈(8,34)) — protected by distance and
  the map edge from the NE rushers; pays travel cost but the bring-up
  chain (powr, tent, first defender) survives;

* the EXPOSED central site (≈(40-50,20)) — closer to mid-map but
  directly in the NE rusher's lane; fact gets razed during bring-up
  AND the spatial win-predicate (`building_in_region` of (8,34) r=5)
  fails even if it were to survive.

Once the MCV deploys, the site is committed — the conyard cannot be
relocated. The win predicate makes both axes load-bearing:

* `building_in_region:{x:8,y:34,r:5,type:fact,count:1}` ⇒ the deploy
  MUST land at the SAFE region — deploying centrally never satisfies
  this, regardless of what else gets built;
* `has_building:powr` AND `has_building:tent` AND `own_units_gte:1`
  ⇒ the full bring-up chain (production queue re-enable after
  deploy, then powr + tent + train one defender) must complete;
* `within_ticks:5400` paired with `after_ticks:5401` ⇒ a non-finisher
  is a real reachable timeout LOSS (60 turns × ≤90 ticks/step reaches
  ≥5400 in interrupt mode), never a draw.

These tests prove with deterministic scripted policies (no model,
no network) that:

* the intended deploy-safe policy WINS every level + every hard seed;
* the stall policy LOSES every level + every hard seed (real LOSS,
  not a draw — MCV never deploys, no fact, after_ticks fires);
* the deploy-exposed policy LOSES every level + every hard seed
  (fact placed at the wrong cell fails the spatial check; on
  medium/hard the rusher also razes it before the bring-up window
  closes);
* the `after_ticks` deadline is reachable inside `max_turns`;
* the hard tier defines ≥2 spawn_point groups (so the agent must
  identify the safe corner from each start, not memorise one path).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mcv-deploy-defensible-site.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Spec coordinates: the SAFE deploy region is the SW corner; the
# EXPOSED candidate is the central cell (mid-map). The win predicate
# only accepts the SAFE region (radius 5 of SAFE_XY).
SAFE_XY = (8, 34)
EXPOSED_XY = (40, 20)


# ── scripted policies ────────────────────────────────────────────────


def _mcv_pos_id(rs):
    """First scenario-declared MCV in agent units, if any.
    Returns (id_str, (cell_x, cell_y)) or (None, None)."""
    for u in rs.get("units_summary") or []:
        if str(u.get("type", "")).lower() == "mcv":
            return str(u["id"]), (int(u["cell_x"]), int(u["cell_y"]))
    return None, None


def stall(rs, C):
    """Observe-only — the MCV never deploys, no fact ever appears.
    The fail clause (after grace) `not building_count_gte:{fact,1}`
    bites at tick 1500+; the episode ends as a real LOSS (no draw)."""
    return [C.observe()]


def _make_site_policy(target_xy: tuple[int, int]):
    """Drive the MCV to `target_xy`, deploy on arrival, then build
    powr + tent + one e1 to satisfy the bring-up chain. The chain
    is identical for safe and exposed sites; only the target cell
    differs — so the test contrast is purely the site choice (the
    spatial win-predicate + survival)."""
    tx, ty = target_xy
    state = {
        "deployed": False,
        "move_issued": False,
        "placed_powr": False,
        "placed_tent": False,
    }

    def policy(rs, C):
        cmds: list = []
        # Stage 1: drive the MCV to the target site.
        if not state["deployed"]:
            mid, pos = _mcv_pos_id(rs)
            if mid is None:
                # MCV gone before deploy ⇒ no recovery (this is what
                # a bad-siting run does on its way to LOSS).
                return [C.observe()]
            if not state["move_issued"]:
                cmds.append(C.move_units([mid], target_x=tx, target_y=ty))
                state["move_issued"] = True
            # Re-issue movement until arrival; deploy once close
            # enough to the target. The deploy transform consumes
            # the MCV and yields a fact at the MCV's CURRENT cell,
            # so we deploy ON ARRIVAL, not en route.
            elif abs(pos[0] - tx) + abs(pos[1] - ty) <= 2:
                cmds.append(C.deploy([mid]))
                state["deployed"] = True
            else:
                cmds.append(C.move_units([mid], target_x=tx, target_y=ty))
            return cmds or [C.observe()]

        # Stage 2: bring-up chain — powr (300) → tent (500) → e1 (100).
        # `render_state()` formats own_buildings as a list of dicts
        # ({type, cell_x, cell_y}), NOT the raw tuples that live on
        # `signals.own_buildings`.
        own_b = rs.get("own_buildings") or []
        types = {str(b.get("type", "")).lower() for b in own_b}
        prod = rs.get("production") or []
        # `production` in render_state is a flat list of item strings.
        prod_items = [
            p if isinstance(p, str)
            else (p.get("item") if isinstance(p, dict) else None)
            for p in prod
        ]
        fact = next(
            ((int(b["cell_x"]), int(b["cell_y"]))
             for b in own_b if str(b.get("type", "")).lower() == "fact"),
            None,
        )
        if fact is None:
            # Deploy issued but the fact's not yet visible in the
            # adapter (transform completes within ticks_per_step) —
            # wait one tick.
            return [C.observe()]
        fx, fy = fact

        if "powr" not in types:
            if "powr" not in prod_items and not state["placed_powr"]:
                cmds.append(C.build("powr"))
            cmds.append(C.place_building("powr", fx + 2, fy + 2))
            state["placed_powr"] = True
        elif "tent" not in types:
            if "tent" not in prod_items and not state["placed_tent"]:
                cmds.append(C.build("tent"))
            cmds.append(C.place_building("tent", fx - 2, fy + 2))
            state["placed_tent"] = True
        else:
            # Train at least one defender to satisfy own_units_gte:1.
            n_units = sum(
                1 for u in (rs.get("units_summary") or [])
                if str(u.get("type", "")).lower() in ("e1", "e3")
            )
            if n_units < 1 and "e1" not in prod_items:
                cmds.append(C.build("e1"))
        return cmds or [C.observe()]

    return policy


def make_intended_safe():
    """Intended capability: drive MCV to the SAFE SW corner, deploy
    there, complete the bring-up chain. Wins every level + every
    seed."""
    return _make_site_policy(SAFE_XY)


def make_deploy_exposed():
    """Defective choice: drive MCV to the EXPOSED central cell, deploy
    there. Even if bring-up completes, the building_in_region check
    fails (fact at wrong cell); on medium/hard the rusher also razes
    the fact during bring-up. LOSES every level + every seed."""
    return _make_site_policy(EXPOSED_XY)


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "mcv-deploy-defensible-site"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = pack.meta.benchmark_anchor
    assert any("MicroRTS" in a for a in anchors), anchors
    assert any("FOB" in a for a in anchors), anchors
    assert any("urban planning" in a for a in anchors), anchors
    assert any("resilience" in a.lower() for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        # The decision tool (deploy) must be in the agent's allowed
        # toolset for every level.
        assert "deploy" in (c.scenario.tools or []), (lvl, c.scenario.tools)


def test_medium_and_hard_use_rusher_bot():
    """Medium/hard apply timing pressure via the scripted `rusher`
    bot (easy uses `turtle` so the bare site-choice skill is
    isolated)."""
    for lvl in ("medium", "hard"):
        c = compile_level(load_pack(PACK), lvl)
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "rusher", (lvl, bot)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: `after_ticks` must be strictly
    below the tick reachable at max_turns (≤90 ticks/step in
    interrupt mode)."""
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


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the agent has to identify the safe corner from each start
    (anti-memorisation of a single MCV path)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126):
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_win_predicate_requires_safe_region_anchor():
    """The spatial check (`building_in_region` of the SAFE cell) is
    in every level's win — site choice is structurally load-bearing,
    not just timing."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        w = c.win_condition.model_dump(exclude_none=True)
        regions = [
            clause["building_in_region"]
            for clause in w.get("all_of", []) or []
            if "building_in_region" in clause
        ]
        assert regions, f"{lvl}: no building_in_region in win"
        r = regions[0]
        assert (int(r["x"]), int(r["y"])) == SAFE_XY, (lvl, r)
        assert str(r.get("type")).lower() == "fact"


# ── solvency: intended WINS every level + every seed ─────────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_deploy_safe_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_intended_safe(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed}: intended deploy-safe must WIN; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings}, "
            f"units_lost={r.signals.units_lost})"
        )


# ── no-cheat: every lazy / mis-sited policy LOSES (not draws) ────────


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """Observe-only: the MCV never deploys → no fact → the
    not-building_count_gte fail bites (after grace) or the deadline
    does. Real LOSS on every level + every seed."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not "
            f"draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_deploy_exposed_loses_every_level_and_seed(level):
    """Deploy at the exposed central cell: the spatial win-check
    rejects the fact (wrong region); on medium/hard the rusher also
    razes the fact during bring-up. Real LOSS on every level + seed."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, make_deploy_exposed(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} deploy-exposed: must LOSE; got "
            f"{r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )
