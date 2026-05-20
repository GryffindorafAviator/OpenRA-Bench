"""then:[A,B,…] happened-before composite predicate.

Unblocks the reactive-replan / scout-then-counter / strict-ordered-
sequence family: a scenario can require clauses to become true IN
ORDER over the course of an episode, not merely all-of-now.

Real-world anchor: PlanBench replanning, ALFWorld goal-conditional
adaptation, PERT stage dependencies. The bench's existing predicates
let you say "all of these must be true now" (`all_of`) and "any of
these must be true now" (`any_of`); `then:` is the missing temporal
operator: the clauses must have been observed-true in sequence.

State: per-episode `signals.then_progress[id]` (an integer index of
how many clauses have been observed-true so far), so a clause that
stopped being true still counts. Identical pattern to
`waypoint_sequence`'s `seq_progress` latch.
"""

from __future__ import annotations

import types

from openra_bench.scenarios.win_conditions import (
    COMPOSITE_KEYS,
    WinContext,
    evaluate,
)


def _ctx(units_killed: int = 0, buildings_disc: int = 0,
         own_units: int = 0):
    sig = types.SimpleNamespace(
        units_killed=units_killed,
        enemy_buildings_seen_ids=set(range(buildings_disc)),
        game_tick=0,
        units_lost=0,
        cash=0, resources=0, power_provided=0, power_drained=0,
        own_building_types=set(), own_buildings=[],
        enemies_seen_ids=set(),
        explored_percent=0.0,
        then_progress={},
        seq_progress={},
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": [
            {"id": i, "type": "e1", "cell_x": 0, "cell_y": 0}
            for i in range(own_units)
        ]},
    )


def test_then_is_a_composite_key():
    """The new operator must register as a composite (not a leaf), so
    the WinCondition validator accepts {then: …} alongside all_of /
    any_of / not."""
    assert "then" in COMPOSITE_KEYS


def test_then_satisfied_when_clauses_observed_in_order():
    """A → B in order ⇒ true. The clauses don't both need to be true
    NOW; they just need to have been observed-true at some past
    evaluation, in order."""
    cond = {"then": {
        "id": "scout-then-counter",
        "clauses": [
            {"buildings_discovered_gte": 1},  # A
            {"units_killed_gte": 3},           # B
        ],
    }}
    ctx = _ctx(units_killed=0, buildings_disc=0)
    # Initially: neither clause true.
    assert evaluate(cond, ctx) is False

    # A becomes true: scouted 1 building.
    ctx.signals.enemy_buildings_seen_ids = {1}
    assert evaluate(cond, ctx) is False  # A latched, B not yet
    assert ctx.signals.then_progress["scout-then-counter"] == 1

    # A is no longer true (lost vision) — but the latch holds.
    ctx.signals.enemy_buildings_seen_ids = set()
    ctx.signals.units_killed = 3   # B now true
    assert evaluate(cond, ctx) is True
    assert ctx.signals.then_progress["scout-then-counter"] == 2


def test_then_no_credit_for_b_alone_without_a():
    """B true while A is still false ⇒ the chain does NOT advance.
    The latch only counts clauses observed-true-in-order. Once A
    finally becomes true on a later evaluation, the chain advances
    through every consecutive currently-satisfied clause in one pass
    (greedy advance, mirroring `waypoint_sequence`'s semantics). The
    bench-relevant guarantee is that a policy which ONLY ever made B
    true (never A) can never satisfy the chain — keeping the
    happened-before signal honest."""
    cond = {"then": {
        "id": "ordered-x",
        "clauses": [
            {"buildings_discovered_gte": 1},  # A
            {"units_killed_gte": 3},           # B
        ],
    }}
    ctx = _ctx()
    # B true, A false → no progress.
    ctx.signals.units_killed = 5
    assert evaluate(cond, ctx) is False
    assert ctx.signals.then_progress["ordered-x"] == 0
    # Many more evals with B-only must STILL not advance (no false
    # credit for the late-A case).
    for _ in range(20):
        assert evaluate(cond, ctx) is False
        assert ctx.signals.then_progress["ordered-x"] == 0


def test_then_late_a_then_b_credits_both_clauses():
    """Once A finally latches, the greedy advance picks up any
    currently-satisfied later clauses in the same call. This matches
    waypoint_sequence's semantics and is the right semantic: 'I
    already had the counter ready; now that I've also scouted, the
    chain is satisfied'."""
    cond = {"then": {
        "id": "late-a",
        "clauses": [
            {"buildings_discovered_gte": 1},
            {"units_killed_gte": 3},
        ],
    }}
    ctx = _ctx()
    ctx.signals.units_killed = 5  # B pre-satisfied
    assert evaluate(cond, ctx) is False  # but A blocks
    ctx.signals.enemy_buildings_seen_ids = {1}  # A becomes true
    assert evaluate(cond, ctx) is True  # chain completes
    assert ctx.signals.then_progress["late-a"] == 2


def test_then_three_clauses_strict_order():
    """Three-stage chain: A → B → C. Each stage must latch before the
    next can advance. Skipping is impossible by construction."""
    cond = {"then": {
        "id": "abc",
        "clauses": [
            {"buildings_discovered_gte": 1},
            {"units_killed_gte": 2},
            {"own_units_gte": 4},
        ],
    }}
    ctx = _ctx()
    # Skip straight to C: own_units=5 but no scout, no kills.
    ctx.render_state["units_summary"] = [
        {"id": i, "type": "e1", "cell_x": 0, "cell_y": 0} for i in range(5)
    ]
    assert evaluate(cond, ctx) is False
    assert ctx.signals.then_progress["abc"] == 0

    # Now satisfy A: scout 1 building.
    ctx.signals.enemy_buildings_seen_ids = {1}
    assert evaluate(cond, ctx) is False
    # Stage advanced to 1, but B (kills) is still 0.
    assert ctx.signals.then_progress["abc"] == 1

    # Now B: kill 2.
    ctx.signals.units_killed = 2
    # On this evaluation, B advances (idx 1→2), and C is already true
    # (5 own units >= 4) so C also advances (idx 2→3). Returns true.
    assert evaluate(cond, ctx) is True
    assert ctx.signals.then_progress["abc"] == 3


def test_then_empty_clauses_returns_false():
    """Authoring-hygiene: an empty or missing clauses list must not
    accidentally satisfy the predicate (silent always-true would be a
    no-cheat bar violation)."""
    assert evaluate({"then": {"id": "empty", "clauses": []}}, _ctx()) is False


def test_then_phrase_translation():
    """The objective_brief must render `then` as a plain-language
    ordered chain so the model knows the order matters."""
    from openra_bench.game_knowledge import objective_brief

    brief = objective_brief(
        "description",
        {"then": {
            "id": "scout-then-counter",
            "clauses": [
                {"buildings_discovered_gte": 1},
                {"units_killed_gte": 3},
            ],
        }},
        None,
        max_turns=30,
    )
    assert "in this exact order" in brief
    assert "THEN" in brief
    assert "spot ≥1 enemy buildings" in brief
    assert "destroy ≥3 enemy units" in brief


def test_then_id_isolation():
    """Two `then` predicates with different ids must latch
    independently — necessary for a pack that uses multiple ordered
    chains (e.g. a win clause + a fail clause both with their own
    sequencing)."""
    a = {"then": {"id": "A", "clauses": [{"units_killed_gte": 1}]}}
    b = {"then": {"id": "B", "clauses": [{"units_killed_gte": 5}]}}
    ctx = _ctx(units_killed=1)
    assert evaluate(a, ctx) is True
    assert evaluate(b, ctx) is False
    assert ctx.signals.then_progress["A"] == 1
    assert ctx.signals.then_progress["B"] == 0
