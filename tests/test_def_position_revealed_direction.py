"""def-position-revealed-direction scenario pack — full no-cheat
validation on Rust.

REASONING capability — INTEL-DRIVEN ADAPTIVE DEFENSE. The threat axis
is HIDDEN from the brief (sibling `def-position-expected-direction`
DECLARES the axis; this pack does not). The agent must scout the
corridor that holds the enemy outpost, commit a 3-pbox triple at the
matching lane's MOUTH in front of its central base, then engage and
break the incoming rush. The win predicate is the Wave-2 `then:`
happened-before composite:

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
the fact is destroyed.

Topology — a custom 112×40 lane arena (the YAML's `base_map:
generator: arena` materialises with horizontal water walls in the
EAST half splitting the rush into parallel corridors; the WEST half
is open ground for the agent's central base + defender screen). On
EASY a single wall block seals the MID corridor leaving N+S
corridors; the actual rush comes from NORTH only. On MEDIUM two
thinner walls split the east half into N/MID/S corridors; the
actual rush comes from MID only. On HARD the geometry is the same
as easy but the rush LANE is round-robined by enemy-side
`spawn_point:` (Wave-9 CLAUDE.md axis) — spawn_point 0 = NORTH,
spawn_point 1 = SOUTH — so the threat lane flips half the seeds.

These tests prove deterministically (no model / no network) that:

* the intended scout-then-fortify-then-engage policy WINS every level
  and every hard seed (1..4) — including BOTH the NORTH-base and
  SOUTH-base spawn groups on hard;
* stall (observe-only) LOSES every level and every seed (real LOSS,
  never draw);
* scout-then-fortify-WRONG-AXIS (memorised "always defend NORTH" on
  hard) LOSES on at least one hard seed — the SOUTH-base round-
  robin half — because the NORTH pbox at (22, 8) sit silent in the
  open west ground while the south-corridor rushers walk straight
  to the SOUTH-base fact and the kill bar is missed;
* the hard tier defines ≥2 AGENT-side spawn_point groups (NORTH
  base y=14 / SOUTH base y=26) so the threat lane rotates by seed
  (no single-cell memorised opening can generalise);
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
    """The intended capability play on the new lane-arena topology:
    IDENTIFY the threat axis from observation (own fact y AND/OR
    the scouted enemy outpost y). Easy: fact y=20 ⇒ NORTH lane
    (only one corridor has an outpost). Medium: fact y=20 ⇒ MID
    lane. Hard: fact y<20 ⇒ NORTH lane (north scout sees the
    matching outpost); fact y>20 ⇒ SOUTH lane. Then commit 3
    pbox at the matching lane MOUTH (x≈22) so the on-axis pbox
    triple intercepts the funnelled rush as it exits the corridor."""

    state = {"queued": 0, "lane_y": None}

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        enemy_b = rs.get("enemy_buildings_summary") or []
        fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if fact is None:
            return [C.observe()]
        fy = int(fact["cell_y"])

        # Prefer scouted outpost (most direct intel); fall back to
        # own fact latitude (the hard-tier doctrine).
        if state["lane_y"] is None:
            tents = [
                b for b in enemy_b if b.get("type") == "tent"
            ]
            if tents:
                ey = int(tents[0]["cell_y"])
                if ey < 13:
                    state["lane_y"] = 8
                elif ey > 27:
                    state["lane_y"] = 32
                else:
                    state["lane_y"] = 20
            elif fy < 18:
                state["lane_y"] = 8
            elif fy > 22:
                state["lane_y"] = 32
            # else still None — wait for a tent observation.

        cmds = []
        if state["queued"] < 3:
            cmds.append(C.build("pbox"))
            state["queued"] += 1
        if state["lane_y"] is not None:
            types = [b.get("type") for b in own_b]
            pbox_count = sum(1 for t in types if t == "pbox")
            if pbox_count < 3:
                dx = -2 + 2 * pbox_count
                cmds.append(
                    C.place_building("pbox", 22 + dx, state["lane_y"]),
                )
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


def make_scout_then_fortify_always_north():
    """The 'memorised: always defend NORTH' policy — pbox stamped at
    the NORTH lane mouth (38, 8) on every seed regardless of what
    the scout reports. On hard the NORTH scout sees the NORTH
    outpost when the seed picks spawn_point 0 (clause 1 latches
    AND the on-axis pbox catches the rush → WIN). When the seed
    picks spawn_point 1 (SOUTH), the SOUTH scout sees the SOUTH
    outpost so clause 1 still latches passively, BUT the NORTH pbox
    sit silent in the open west ground while the south-funnelled
    rushers walk straight to the central fact at y=20 — the 3-kill
    bar is missed AND the fact is razed → LOSS."""

    state = {"queued": 0}

    def policy(rs, C):
        own_b = rs.get("own_buildings") or []
        fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if fact is None:
            return [C.observe()]
        types = [b.get("type") for b in own_b]
        pbox_count = sum(1 for t in types if t == "pbox")
        cmds = []
        if state["queued"] < 3:
            cmds.append(C.build("pbox"))
            state["queued"] += 1
        if pbox_count < 3:
            dx = -2 + 2 * pbox_count
            cmds.append(C.place_building("pbox", 22 + dx, 8))
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
    # forward outpost in the relevant corridor; passively visible
    # from the matching pre-placed scout jeep).
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
    so the threat axis rotates by seed (anti-memorisation). This
    pack uses the AGENT-side spawn_point axis — spawn_point 0 =
    NORTH base latitude (y=14), spawn_point 1 = SOUTH base
    latitude (y=26). Enemy bands always place (no enemy
    spawn_point declared); the matching-side band is the
    immediate threat because of the rusher's shortest-path
    target seek."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (custom 112x40 arena, cordon 2 ⇒ playable
    # x in [2..109], y in [2..37]).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 109 and 2 <= y <= 37, (a.type, a.position)


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
    """The intended capability play (read enemy_buildings_summary →
    pick lane mouth from outpost y → fortify matching lane → let
    on-axis pbox + central defenders break the rush) MUST WIN on
    every (level, seed). Load-bearing solvency test."""
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


def test_always_north_fortify_loses_on_hard_south_spawn():
    """The memorised 'always defend NORTH' policy — stamps 3 pbox
    at (38, 8) every game. On SOUTH-spawn seeds the NORTH pbox sit
    silent in the open west ground while the south-funnelled rush
    walks to the central fact at y=20; the 3-kill bar is missed
    AND the fact is razed → LOSS. We assert at least one hard seed
    losses so the seed-variation axis bites."""
    c = compile_level(load_pack(PACK), "hard")
    losses = []
    for seed in SEEDS:
        r = run_level(
            c, make_scout_then_fortify_always_north(), seed=seed,
        )
        if r.outcome == "loss":
            losses.append(seed)
    assert losses, (
        "always-NORTH fortify must LOSE on at least one hard seed "
        "(the SOUTH-spawn round-robin half); got WIN on every seed "
        "— seed-variation axis is not biting."
    )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended(), seed=3)
    b = run_level(c, make_intended(), seed=3)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        "same seed must be deterministic"
    )
