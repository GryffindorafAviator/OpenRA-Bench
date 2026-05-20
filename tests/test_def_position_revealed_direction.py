"""def-position-revealed-direction scenario pack — full no-cheat
validation on Rust.

REASONING capability — INTEL-DRIVEN ADAPTIVE DEFENSE. The threat axis
is HIDDEN from the brief (sibling `def-position-expected-direction`
DECLARES the axis; this pack does not). The agent must FIRST drive its
scout jeeps to register the enemy's forward outpost on the actual
approach lane, THEN commit a 3-pbox triple between its base and that
outpost, THEN engage and break the incoming rush. The win predicate is
the Wave-2 `then:` happened-before composite:

    then:
      - {buildings_discovered_gte: 1}         # SCOUT first (axis intel)
      - {building_count_gte: {type: pbox, n: 3}}  # FORTIFY second
      - {units_killed_gte: 3}                 # ENGAGE third

paired with `building_count_gte:{fact, 1}` (the home fact still stands)
and `within_ticks: 5400` (a reachable deadline given max_turns=66, which
caps ticks at 93+90·65=5943). Fail is `after_ticks: 5401` OR
`not building_count_gte:{fact,1}` OR `not own_units_gte:1` — a non-win
is always a real reachable LOSS, never a draw.

`building_count_gte:{fact,1}` is used in place of the more common
`has_building: fact` because the latter checks
`own_building_types` which ACCUMULATES across the episode (once
the agent ever owned a fact the set still contains "fact" after
the fact is razed). The `building_count_gte` predicate reads the
live `own_buildings` list each frame and correctly bites when
the fact is destroyed — without this swap the stall policy on
the hard NORTH spawn ended in DRAW (engine auto-`done` on player
elimination before the deadline) rather than LOSS.

These tests prove deterministically (no model / no network) that:

* the intended scout-then-fortify-then-engage policy WINS every level
  and every hard seed (1..4) — including BOTH the NORTH-base and the
  SOUTH-base spawn groups on hard;
* stall (observe-only) LOSES every level and every seed (real LOSS,
  never draw);
* no-scout-pre-fortify-NORTH (build 3 pbox at y=14 immediately and
  never move the jeeps) WINS only on EASY (where the jeeps start at
  y=8 already in sight of the outpost) and LOSES on every MEDIUM and
  HARD seed (clause 1 of the then-chain never latches);
* scout-then-fortify-WRONG-AXIS (memorised "always defend NORTH") LOSES
  on every hard SOUTH-spawn seed — the south rush walks past the
  wrong-axis pbox line and razes the SOUTH-spawn fact, AND the
  buildings-discovered clause cannot latch on the SOUTH-spawn jeep
  because the north outpost is out of sight;
* the hard tier defines ≥2 spawn_point groups (NORTH base y=16 /
  SOUTH base y=24) so the base latitude rotates by seed (no single-
  cell memorised opening can generalise);
* every level's `after_ticks` is reachable inside `max_turns` so a
  non-finisher is a real timeout LOSS, never a draw.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "def-position-revealed-direction.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ────────────────────────────────────────────────


def stall(rs, C):
    """Observe-only. Never scouts, never builds, never engages.
    Reachable timeout LOSS (or fact-loss LOSS, which is the same
    fail-clause family)."""
    return [C.observe()]


def make_intended():
    """The intended capability play: identify own base latitude
    (NORTH-of-mid → threat-from-NORTH; SOUTH-of-mid → threat-from-
    SOUTH; mid → easy preset NORTH lane), DRIVE the jeeps to the
    matching outpost band, BUILD 3 pbox between base and that band,
    and let the on-axis pbox + base defenders rack up ≥3 kills on
    the incoming rush."""

    state = {"scout_dispatched": False, "queued": 0}

    def policy(rs, C):
        units = rs.get("units_summary") or []
        own_b = rs.get("own_buildings") or []
        fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if fact is None:
            return [C.observe()]
        fx, fy = int(fact["cell_x"]), int(fact["cell_y"])
        # Pick the threat side from base latitude. Hard: y=16 ⇒
        # north lane (scout to y=8, pbox at y=10); y=24 ⇒ south
        # lane (scout to y=32, pbox at y=30). Easy/medium: y=20 ⇒
        # north preset.
        if fy < 20:
            scout_y = max(2, fy - 8)
            pbox_y = max(4, fy - 6)
        elif fy > 20:
            scout_y = min(38, fy + 8)
            pbox_y = min(34, fy + 6)
        else:
            scout_y = 8
            pbox_y = 14

        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")

        cmds = []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if jeeps and not state["scout_dispatched"]:
            cmds.append(
                C.move_units([j["id"] for j in jeeps], fx, scout_y),
            )
            state["scout_dispatched"] = True
        # Queue the 3 pbox up front; one StartProduction per turn
        # (the Defense queue is one-at-a-time, so queueing more
        # than 3 wastes only the StartProduction order).
        if state["queued"] < 3:
            cmds.append(C.build("pbox"))
            state["queued"] += 1
        # Always emit the NEXT slot's place_building — the engine
        # buffers blocked PLACEs until production finishes, then
        # places at the requested cell.
        if pbox_count < 3:
            dx = -2 + 2 * pbox_count
            cmds.append(C.place_building("pbox", fx + dx, pbox_y))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_no_scout_north_pbox():
    """Pre-fortify NORTH-lane WITHOUT ever moving the jeeps. On EASY
    this WINS (the jeeps are pre-staged at y=8, passively in sight of
    the outpost at y=2 — clause 1 latches at turn 1 even without a
    move command). On MEDIUM and HARD it LOSES every seed — the
    jeeps sit at the base, outside outpost vision, so the then-chain
    NEVER reaches clause 1, and the deadline expires."""

    state = {"queued": 0}

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if fact is None:
            return [C.observe()]
        fx = int(fact["cell_x"])
        # Memorised: always build pbox at y=14 (the easy/medium
        # canonical NORTH lane). NEVER moves the jeeps.
        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")
        cmds = []
        if state["queued"] < 3:
            cmds.append(C.build("pbox"))
            state["queued"] += 1
        if pbox_count < 3:
            dx = -2 + 2 * pbox_count
            cmds.append(C.place_building("pbox", fx + dx, 14))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_scout_then_fortify_always_north():
    """Drives the scout NORTH and builds pbox on the NORTH lane —
    the 'memorised: always defend NORTH' policy. On HARD SOUTH-spawn
    seeds the north scout never reaches the south outpost (the
    matching threat), so clause 1 stays unlatched AND the south rush
    razes the SOUTH-spawn fact (the y=18 pbox line on the SOUTH
    spawn is on the wrong axis). LOSS on every hard SOUTH-spawn
    seed; WIN on every hard NORTH-spawn seed (where it accidentally
    happens to be the correct lane)."""

    state = {"scout_dispatched": False, "queued": 0}

    def policy(rs, C):
        units = rs.get("units_summary") or []
        own_b = rs.get("own_buildings") or []
        fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if fact is None:
            return [C.observe()]
        fx, fy = int(fact["cell_x"]), int(fact["cell_y"])
        # ALWAYS scout NORTH (whatever the base latitude); ALWAYS
        # build pbox on the NORTH side of base.
        scout_y = max(2, fy - 8)
        pbox_y = max(4, fy - 6)

        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")
        cmds = []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if jeeps and not state["scout_dispatched"]:
            cmds.append(
                C.move_units([j["id"] for j in jeeps], fx, scout_y),
            )
            state["scout_dispatched"] = True
        if state["queued"] < 3:
            cmds.append(C.build("pbox"))
            state["queued"] += 1
        if pbox_count < 3:
            dx = -2 + 2 * pbox_count
            cmds.append(C.place_building("pbox", fx + dx, pbox_y))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


# ── scenario-shape invariants ────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "def-position-revealed-direction"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (per the seed taxonomy).
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("military" in a.lower() for a in anchors), anchors
    assert any("intel" in a.lower() for a in anchors), anchors
    # Rusher bot wired through to the engine for every level.
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
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns (≤90 ticks per
    step in interrupt mode)."""
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
def test_every_level_uses_then_chain_with_scout_fortify_engage(level):
    """The 3-phase scout→fortify→engage chain MUST be wired through
    as a Wave-2 `then:` happened-before composite (the whole point of
    the pack — ordered intel-driven defense, not a degenerate
    permutation-tolerant `all_of`)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    ao = win.get("all_of") or []
    then_branches = [b for b in ao if "then" in b]
    assert len(then_branches) == 1, (
        f"{level}: expected exactly one then-chain in win; got {win}"
    )
    clauses = then_branches[0]["then"]["clauses"]
    assert len(clauses) == 3, (
        f"{level}: then-chain must be 3 clauses (scout, fortify, "
        f"engage); got {len(clauses)}: {clauses}"
    )
    # Clause 1: scout latch — ≥1 enemy building discovered (the
    # forward outpost the agent's jeep MUST reach by movement on
    # medium/hard; passively visible on easy).
    assert clauses[0].get("buildings_discovered_gte") == 1, clauses[0]
    # Clause 2: fortify — exactly 3 pbox (not a token pbox).
    bcg = clauses[1].get("building_count_gte") or {}
    assert bcg.get("type") == "pbox" and int(bcg.get("n", 0)) == 3, (
        clauses[1]
    )
    # Clause 3: engage — at least 3 kills.
    assert clauses[2].get("units_killed_gte") == 3, clauses[2]


@pytest.mark.parametrize("level", LEVELS)
def test_win_predicate_uses_building_count_gte_for_fact_persistence(
    level,
):
    """The fact-persistence clause MUST be `building_count_gte:
    {type:fact, n:1}` (live count) and NOT `has_building:fact`
    (accumulating set). See module docstring for the footgun."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump(exclude_none=True)
    flat = str(win)
    assert "has_building" not in flat, (
        f"{level}: win must avoid the accumulating `has_building` "
        f"footgun; use `building_count_gte:{{type:fact,n:1}}` "
        f"instead. win={win}"
    )
    # And the live count clause IS present:
    ao = win.get("all_of") or []
    found = False
    for clause in ao:
        bcg = (
            clause.get("building_count_gte")
            if isinstance(clause, dict) else None
        )
        if isinstance(bcg, dict) and bcg.get("type") == "fact":
            assert int(bcg.get("n", 0)) >= 1
            found = True
    assert found, f"{level}: missing fact-alive clause; got {win}"


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the threat axis rotates by seed (anti-memorisation). Both
    bands of enemy outposts always place (CLAUDE.md: enemy actors
    don't honour spawn_point) so the SAME scout-then-fortify
    discipline is tested from a flipped base latitude per seed."""
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


def test_tools_list_matches_spec():
    pack = load_pack(PACK)
    expected = {
        "observe", "build", "place_building", "move_units",
        "attack_unit", "attack_move", "stop",
    }
    tools = (
        pack.base.get("tools")
        if isinstance(pack.base, dict)
        else pack.base.tools
    )
    assert set(tools) == expected, tools


# ── solvency: intended WINS every level + every hard seed ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_fortify_engage_wins_every_level_and_seed(
    level, seed,
):
    """The intended capability play (identify base latitude → scout
    matching outpost → fortify matching lane → engage) MUST WIN on
    every (level, seed). This is the load-bearing solvency test."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, make_intended(), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended scout-fortify-engage must WIN; "
        f"got {r.outcome} (tick={r.signals.game_tick}, "
        f"kills={r.signals.units_killed}, "
        f"then_progress={getattr(r.signals, 'then_progress', {})}, "
        f"buildings_seen="
        f"{len(r.signals.enemy_buildings_seen_ids)})"
    )


# ── no-cheat: every wrong / lazy policy LOSES (real LOSS, not draw) ──


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level, seed):
    """Stall (observe-only) MUST LOSE on every (level, seed). The
    fail clause family (after_ticks: 5401 OR
    not building_count_gte:{fact,1} OR not own_units_gte:1) bites
    deterministically — never a draw."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed}: stall must LOSE (real fail, not a "
        f"draw); got {r.outcome} (tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_scout_pre_fortify_loses_on_medium(seed):
    """Build 3 pbox at the canonical NORTH lane without ever moving
    the jeeps. The jeeps sit at the base, outside outpost vision —
    so clause 1 (`buildings_discovered_gte:1`) NEVER latches and the
    deadline expires. (On EASY the jeeps START forward at y=8 and
    passively see the outpost from turn 1, so this play wins easy —
    that's by design, easy explicitly teaches the chain without
    requiring scout movement.)"""
    c = compile_level(load_pack(PACK), "medium")
    r = run_level(c, make_no_scout_north_pbox(), seed=seed)
    assert r.outcome == "loss", (
        f"medium s={seed}: no-scout pre-fortify must LOSE; got "
        f"{r.outcome} (tick={r.signals.game_tick}, "
        f"then_progress={getattr(r.signals, 'then_progress', {})})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_scout_pre_fortify_loses_on_hard(seed):
    """Same no-scout pre-fortify play on hard. Loses for two
    independent reasons depending on spawn: (i) on NORTH spawn the
    jeeps at y=16 can't see the y=4 outpost without moving — clause
    1 never latches; (ii) on SOUTH spawn the pbox at y=14 is on the
    WRONG axis (south rush comes from y=38), the south rush razes
    the south-spawn fact AND clause 1 never latches."""
    c = compile_level(load_pack(PACK), "hard")
    r = run_level(c, make_no_scout_north_pbox(), seed=seed)
    assert r.outcome == "loss", (
        f"hard s={seed}: no-scout pre-fortify must LOSE; got "
        f"{r.outcome} (tick={r.signals.game_tick}, "
        f"then_progress={getattr(r.signals, 'then_progress', {})})"
    )


def test_scout_then_always_north_fortify_loses_on_hard_south_spawn():
    """The memorised 'always scout/fortify NORTH' policy LOSES on
    every hard SOUTH-spawn seed (s=1, s=3 in the engine's spawn_idx
    round-robin): the north scout doesn't reach the south outpost
    (clause 1 never latches) AND the south-spawn fact at y=24 is
    razed by the y=38 south rush which the NORTH-side pbox line at
    y=18 can't intercept. Demonstrates the spawn-variation
    discrimination — a fixed-axis opening cannot generalise across
    seeds."""
    c = compile_level(load_pack(PACK), "hard")
    # Verified empirically: SOUTH spawn corresponds to seeds 1 & 3
    # in the engine's spawn_idx round-robin; NORTH to 2 & 4.
    south_spawn_seeds = (1, 3)
    for seed in south_spawn_seeds:
        r = run_level(
            c, make_scout_then_fortify_always_north(), seed=seed,
        )
        assert r.outcome == "loss", (
            f"hard SOUTH-spawn s={seed}: scout-and-fortify-always-"
            f"NORTH must LOSE; got {r.outcome} "
            f"(tick={r.signals.game_tick}, "
            f"then_progress={getattr(r.signals, 'then_progress', {})})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=3)
    b = run_level(c, make_intended(), seed=3)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
