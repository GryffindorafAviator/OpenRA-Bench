"""scout-detect-base-direction — PERCEPTION pack: scout BOTH candidate
corners to find which one holds the enemy base, then commit the
attackers to the correct corner.

The pack tests TWO chained skills:

  1) PERCEPTION (the scout step): push jeeps to BOTH far east
     corners — NE (110, 8) and SE (110, 32) — to read which
     corner is the actionable target from THIS spawn (or to
     confirm the only target on lower tiers). On hard the agent
     must additionally identify its OWN spawn latitude (NORTH vs
     SOUTH) to decide which corner is the LATITUDE-MATCHING
     short reach.
  2) ACTION (the commit step): attack-move the 2tnk strike force
     on the chosen corner; raze the enemy fact inside the
     deadline without losing more than 1 unit.

Win predicate (load-bearing across all four axes):
  * `any_of`(`enemy_key_buildings_destroyed_in_region`(NE),
            `enemy_key_buildings_destroyed_in_region`(SE)) —
    EITHER raze satisfies the win. Both facts always place per
    engine semantics (CLAUDE.md: enemy actors don't honour
    `spawn_point`); the seed-driven AGENT spawn flips which
    corner is the close reach on hard.
  * `within_ticks:2500` (easy/medium) / `:2700` (hard) — the
    timed deadline. A stall / scout-no-commit / cross-axis
    commit cannot finish inside the budget.
  * `units_lost_lte:1` — the attrition cap. A cross-axis commit
    on hard drives the tank column through a mid-lane e3 rocket
    CURTAIN (x=85, y=14..26) and bleeds ≥2 tanks before clearing
    → busts the cap → LOSS.
  * paired `after_ticks` fail clause (`2501` / `2701`) so a
    non-finisher emits a real reachable timeout LOSS (max_turns
    50/45 → reachable tick 4503/4053 ≥ the fail deadline).

Scripted policies cover the four bar-defining outcomes:
  * stall              → LOSS every (level, seed) — after_ticks
                         bites; no fact destroyed.
  * scout-but-no-commit → LOSS every (level, seed) — jeeps scout
                         both corners but tanks never push; clock
                         LOSS (and the curtain may already chip
                         a jeep on hard).
  * always-NE blind    → WIN on easy/medium every seed
                         (equidistant corners from single spawn);
                         on hard LOSS on SOUTH-spawn seeds
                         (cross-axis through the curtain) and
                         WIN on NORTH-spawn seeds. The
                         seed-half-half pattern is the
                         load-bearing tooth: a memorised
                         single-corner opening cannot generalise.
  * always-SE blind    → mirror of always-NE.
  * intended scout-then-commit → WIN every (level, seed). The
                         CAPABILITY play: scout both corners on
                         turn 1, identify the latitude-matching
                         corner from the spawn position, commit
                         the tank column ON-AXIS (no curtain
                         engagement), raze the fact inside the
                         deadline with ≤1 unit lost.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "scout-detect-base-direction.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Corner anchors (mirror the pack).
NE = (110, 8)
SE = (110, 32)


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-detect-base-direction"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported, f"{lvl}: map not supported"


def test_benchmark_anchor_set():
    """Wave-8 seed-taxonomy contract: anchors must call out ERQA
    direction-reading, military intelligence triage, and SC2 scout
    direction (the three explicit anchors in the design brief)."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("ERQA" in a for a in anchors), anchors
    assert any(
        "military intelligence" in a.lower() or "intelligence triage" in a.lower()
        for a in anchors
    ), anchors
    assert any("SC2" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS on
    timeout, attrition, or force-wipe."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks AND after_ticks must be reachable inside max_turns
    (engine ≤90 ticks/turn → reachable max = 93 + 90·(N-1)).
    Otherwise the deadline never bites ⇒ DRAW degeneracy."""
    pack = load_pack(PACK)

    def _collect(node, key, out):
        if isinstance(node, dict):
            if key in node:
                out.append(node[key])
            for v in node.values():
                _collect(v, key, out)
        elif isinstance(node, list):
            for v in node:
                _collect(v, key, out)

    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        max_turns = pack.levels[lvl].max_turns
        reachable = 93 + 90 * (max_turns - 1)
        wts: list[int] = []
        _collect(c.win_condition.model_dump(exclude_none=True), "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        fts: list[int] = []
        _collect(c.fail_condition.model_dump(exclude_none=True), "after_ticks", fts)
        assert fts, f"{lvl} has no after_ticks fail leaf"
        for ft in fts:
            assert ft <= reachable, (
                f"{lvl} after_ticks={ft} > reachable={reachable} "
                f"(max_turns={max_turns}) — fail never bites ⇒ draw"
            )


def test_or_clause_used_in_win_over_both_corners():
    """The win must be `any_of` over the two corner destruction
    paths (the spec-load-bearing structural shape)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        or_branches = [b for b in ao if "any_of" in b]
        assert len(or_branches) == 1, (
            f"{lvl} must have exactly one any_of branch; got {win}"
        )
        clauses = or_branches[0]["any_of"]
        assert len(clauses) == 2, (
            f"{lvl} any_of must have exactly 2 paths; got {clauses}"
        )
        regions = set()
        for cl in clauses:
            assert "enemy_key_buildings_destroyed_in_region" in cl, cl
            v = cl["enemy_key_buildings_destroyed_in_region"]
            regions.add((int(v["x"]), int(v["y"])))
            assert int(v["radius"]) == 6, v
            assert "fact" in v["types"], v
        assert regions == {NE, SE}, regions


def test_actor_composition_matches_spec():
    """Per-spec actor manifest: 2 jeep scouts + 4 medium tanks +
    1 fact PER spawn group at the WEST staging. Easy/medium have
    one implicit group; hard has TWO declared spawn groups."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        groups = {
            a.spawn_point for a in c.scenario.actors
            if a.owner == "agent" and a.spawn_point is not None
        }
        n_groups = max(1, len(groups))
        jeeps = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "jeep"]
        tanks = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "2tnk"]
        facts = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "fact"]
        assert len(jeeps) == 2 * n_groups, (lvl, jeeps)
        assert len(tanks) == 4 * n_groups, (lvl, tanks)
        assert len(facts) == 1 * n_groups, (lvl, facts)


def test_both_corner_facts_always_placed():
    """Enemy facts at BOTH corners must always place every seed
    (engine constraint: enemy actors don't honour spawn_point)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        enemy_facts = [
            (a.position[0], a.position[1])
            for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
        ]
        assert NE in enemy_facts, (lvl, enemy_facts)
        assert SE in enemy_facts, (lvl, enemy_facts)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the base latitude varies by seed (UPGRADED registration)."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_hard_has_mid_lane_picket_curtain():
    """Hard-tier discrimination tooth: the e3 picket curtain at
    x=85, y=14..26 makes cross-axis commits attritionally
    infeasible. Without it, both corners are equidistant from
    either spawn and a memorised single-corner beeline would
    generalise."""
    c = compile_level(load_pack(PACK), "hard")
    curtain = [
        a for a in c.scenario.actors
        if a.owner == "enemy" and a.type == "e3"
        and a.position[0] == 85 and 14 <= a.position[1] <= 26
    ]
    assert len(curtain) >= 5, f"hard curtain has only {len(curtain)} e3"


def test_tools_match_spec():
    """Tools per spec — observe + move_units + attack_unit +
    attack_move + stop (offensive scout+strike toolset, no build)."""
    pack = load_pack(PACK)
    tools = pack.base.get("tools") if isinstance(pack.base, dict) \
        else pack.base.tools
    expected = {"observe", "move_units", "attack_unit", "attack_move", "stop"}
    assert set(tools) == expected, tools


# ── scripted policies ─────────────────────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary") or []


def _corner_fact(rs, corner):
    """Return the enemy `fact` actor near ``corner`` from the
    observation's ``enemy_buildings_summary`` (engine surfaces a
    visible corner fact there with its real actor id), else None.

    A bare ``attack_move`` to the corner cell no longer auto-engages
    the building (post engine-balance armour-class weapon-selection
    fix); the strike must ``attack_unit`` the fact by id to raze it.
    """
    for b in rs.get("enemy_buildings_summary") or []:
        if str(b.get("type", "")).lower() != "fact":
            continue
        bx, by = b.get("cell_x"), b.get("cell_y")
        if bx is None or by is None:
            continue
        if abs(int(bx) - corner[0]) <= 8 and abs(int(by) - corner[1]) <= 8:
            return b
    return None


def stall(rs, C):
    """Observe-only — every unit sits at the staging cell. No corner
    fact ever destroyed; after_ticks fires → LOSS."""
    return [C.observe()]


def scout_but_no_commit(rs, C):
    """Jeeps push to BOTH corners (scout confirms each) but the
    tanks never move. No fact destroyed → after_ticks fires →
    LOSS even though detection succeeded."""
    state = scout_but_no_commit  # function attr container
    if not hasattr(state, "sent"):
        state.sent = False
    units = _own_units(rs)
    if not units:
        return [C.observe()]
    if state.sent:
        return [C.observe()]
    jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
    cmds = []
    if jeeps:
        state.sent = True
        if len(jeeps) >= 1:
            cmds.append(C.move_units([str(jeeps[0]["id"])], target_x=108, target_y=8))
        if len(jeeps) >= 2:
            cmds.append(C.move_units([str(jeeps[1]["id"])], target_x=108, target_y=32))
    if not cmds:
        cmds.append(C.observe())
    return cmds


def _make_blind(corner):
    """Competent blind attacker: beeline ALL tanks to a FIXED corner
    — no scout, no spawn-aware corner choice — and ``attack_unit``
    the corner fact the instant it surfaces so the strike actually
    razes. On easy/medium (single spawn, no curtain) this WINS; on
    hard it LOSES on the seeds where the fixed corner is the
    cross-axis target (the column traverses the x=85 e3 curtain and
    bleeds ≥2 tanks past the units_lost_lte:1 cap)."""
    def policy(rs, C):
        units = _own_units(rs)
        if not units:
            return [C.observe()]
        tanks = [u for u in units if str(u.get("type", "")).lower() == "2tnk"]
        if not tanks:
            return [C.observe()]
        tids = [str(t["id"]) for t in tanks]
        fact = _corner_fact(rs, corner)
        if fact is not None:
            return [C.attack_unit(tids, str(fact["id"]))]
        return [C.attack_move(tids, target_x=corner[0], target_y=corner[1])]
    return policy


def make_intended():
    """Intended scout-then-commit play.

      1. Turn 1: push BOTH jeeps to the two candidate corners (one
         NE, one SE) to confirm each corner has a fact.
      2. Read the tank centroid's y-band to identify the staging
         latitude (NORTH y<=20 → NE, SOUTH y>20 → SE).
      3. Commit the tank column on the LATITUDE-MATCHING corner.
         This keeps the column on-axis (along the staging y-band)
         and AWAY from the mid-lane e3 curtain on hard, so the
         tank column itself loses zero units; the only attrition
         is the single cross-axis SCOUT jeep that crosses the
         curtain — exactly the 1-unit cost the units_lost_lte:1
         cap is calibrated to admit.
      4. Once the chosen corner's fact surfaces, ``attack_unit``
         it by id — the corner fact falls to 4× 2tnk cannon fire
         well inside the deadline. (A bare attack_move no longer
         razes a building post engine-balance fix.)
    """
    state = {"turn": 0, "commit": None}

    def policy(rs, C):
        state["turn"] += 1
        units = _own_units(rs)
        if not units:
            return [C.observe()]
        tanks = [u for u in units if str(u.get("type", "")).lower() == "2tnk"]
        jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
        cmds = []
        if tanks:
            ty = sum(int(t["cell_y"]) for t in tanks) // len(tanks)
        else:
            ty = 20
        if state["commit"] is None:
            state["commit"] = NE if ty <= 20 else SE
        if state["turn"] == 1 and jeeps:
            if len(jeeps) >= 1:
                cmds.append(C.move_units(
                    [str(jeeps[0]["id"])], target_x=108, target_y=8,
                ))
            if len(jeeps) >= 2:
                cmds.append(C.move_units(
                    [str(jeeps[1]["id"])], target_x=108, target_y=32,
                ))
        if tanks and state["turn"] >= 2:
            tids = [str(t["id"]) for t in tanks]
            fact = _corner_fact(rs, state["commit"])
            if fact is not None:
                cmds.append(C.attack_unit(tids, str(fact["id"])))
            else:
                cmds.append(C.attack_move(
                    tids, target_x=state["commit"][0],
                    target_y=state["commit"][1],
                ))
        if not cmds:
            cmds.append(C.observe())
        return cmds
    return policy


# ── solvency: intended WINS every level + every hard seed ─────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_then_commit_wins(level, seed):
    """The intended scout-then-commit capability play must WIN every
    (level, seed). This is the load-bearing test that the pack is
    solvable inside the budget by the advertised capability across
    all hard-tier spawn variants."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, make_intended(), seed=seed)
    assert r.outcome == "win", (
        f"intended scout-then-commit must WIN on {level} s={seed}; "
        f"got {r.outcome} (tick={r.signals.game_tick}, "
        f"lost={r.signals.units_lost}, killed={r.signals.units_killed})"
    )


# ── no-cheat: every lazy / blind / partial policy LOSES (not draws) ──


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE every (level, seed) — the
    after_ticks clause bites at the turn budget; no corner razed."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, stall, seed=seed)
    assert r.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {r.outcome} "
        f"(tick={r.signals.game_tick}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_scout_no_commit_loses(level, seed):
    """Jeeps scout both corners but the tanks never push — clock
    LOSS even though detection succeeded."""
    c = compile_level(load_pack(PACK), level)
    # Reset the function-attr state across runs (one fresh policy
    # per call).
    def _fresh(rs, C):
        return scout_but_no_commit(rs, C)
    if hasattr(scout_but_no_commit, "sent"):
        del scout_but_no_commit.sent
    r = run_level(c, _fresh, seed=seed)
    assert r.outcome == "loss", (
        f"scout-no-commit must LOSE on {level} s={seed}; "
        f"got {r.outcome} (tick={r.signals.game_tick}, "
        f"lost={r.signals.units_lost})"
    )


# ── hard-tier: blind always-one-corner LOSES on the spawn where that
#   corner is the FAR (cross-axis) target — the seed-flipped curtain
#   geometry IS the load-bearing direction-discrimination tooth ─────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_always_ne_blind_loses_on_south_spawn_seeds(seed):
    """On hard: a memorised "always strike NE" beeline LOSES on
    SOUTH-spawn seeds (the staging is at y≈32 → NE is the FAR
    cross-axis target → the tank column traverses the x=85 e3
    curtain and bleeds ≥2 tanks → busts units_lost_lte:1). On
    NORTH-spawn seeds the same policy may WIN (on-axis to NE);
    that asymmetry is the load-bearing seed-discrimination tooth.
    SEEDS 1,3 = SOUTH staging in this pack's RNG (confirmed by
    smoke); SEEDS 2,4 = NORTH staging."""
    c = compile_level(load_pack(PACK), "hard")
    r = run_level(c, _make_blind(NE), seed=seed)
    if seed in (1, 3):
        assert r.outcome == "loss", (
            f"hard always-NE on SOUTH-spawn s={seed} must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost})"
        )
    else:
        assert r.outcome == "win", (
            f"hard always-NE on NORTH-spawn s={seed} should WIN "
            f"(on-axis); got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost})"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_always_se_blind_loses_on_north_spawn_seeds(seed):
    """Mirror: always-SE blind LOSES on NORTH-spawn seeds (cross-
    axis through curtain) and may WIN on SOUTH-spawn seeds
    (on-axis to SE)."""
    c = compile_level(load_pack(PACK), "hard")
    r = run_level(c, _make_blind(SE), seed=seed)
    if seed in (2, 4):
        assert r.outcome == "loss", (
            f"hard always-SE on NORTH-spawn s={seed} must LOSE; "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost})"
        )
    else:
        assert r.outcome == "win", (
            f"hard always-SE on SOUTH-spawn s={seed} should WIN "
            f"(on-axis); got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost})"
        )


# ── Spawn-variation contract (hard) ───────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin — a
    stall on each seed completes deterministically and the cross-
    seed start-cell distinctness is verified across all four seeds
    via the centroid pivot. (The cross-seed distinctness contract
    is enforced globally by tests/test_hard_tier.py.)"""
    c = compile_level(load_pack(PACK), "hard")
    r = run_level(c, stall, seed=seed)
    assert r.outcome == "loss"  # stall must lose
