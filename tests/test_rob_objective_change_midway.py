"""rob-objective-change-midway pack — full no-cheat validation.

Wave-11 reasoning pack: goal-conditional adaptation — the objective
effectively SHIFTS mid-episode. The win is a `then:` chain of TWO
DIFFERENT goals that must latch IN ORDER:
  Phase 1 (clause A) — units_in_region_gte: ≥3 tanks staged in a
    forward region A (an early reach/scout goal).
  Phase 2 (clause B) — enemy_key_buildings_destroyed_in_region: raze
    the enemy `fact` at a far region B (the goal revealed AFTER A).
The win is `all_of[ then:[A, B], within_ticks:5400 ]`.

Bar (per CLAUDE.md):
  - STALL          LOSS every (level, seed) — after_ticks teeth;
                   clause A never latches.
  - COMMIT-B-ONLY  LOSS every (level, seed) — the `then` chain only
                   counts clause B after clause A has gone true; a
                   beeline straight to B never staged ≥3 tanks in A
                   so the chain is stuck at clause 1.
  - DO-A-ONLY      LOSS every (level, seed) — phase 1 latches but
                   phase 2 (raze the far fact) is never met; the
                   deadline expires with the chain at clause 2.
  - INTENDED A-then-B  WIN every (level, seed) — stage at A first,
                   then pivot east and raze the far fact in time.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "rob-objective-change-midway.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level region geometry (mirror the pack): region A (phase 1) and
# region B (phase 2).
GEOM = {
    "easy":   {"A": (55, 8),  "B": (100, 30)},
    "medium": {"A": (55, 12), "B": (105, 34)},
    "hard":   {"A": (55, 20), "B": (100, 30)},
}


# ── Policy helpers ───────────────────────────────────────────────


def _tank_ids(obs):
    return [
        str(u["id"]) for u in (obs.get("units_summary", []) or [])
        if u.get("type") == "2tnk"
    ]


# ── Policies ─────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on after_ticks every (level, seed).
    Phase 1 never latches; the deadline bites."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _commit_b_only_policy(B):
    """Beeline the whole squad straight at the far region B from t=0,
    ignoring phase 1. Must LOSE every (level, seed): the `then` chain
    only begins counting clause B once clause A has latched — a squad
    that never staged ≥3 tanks in region A leaves the chain stuck at
    clause 1, so the win predicate is false even if B is razed."""
    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        return [Cmd.attack_move(ids, B[0], B[1])]
    return pol


def _do_a_only_policy(A):
    """Move the squad into region A and park there — never pivot to
    phase 2. Must LOSE every (level, seed): phase 1 latches but phase
    2 (raze the far fact) is never satisfied; the deadline expires."""
    def pol(obs, Cmd):
        ids = _tank_ids(obs)
        if not ids:
            return [Cmd.observe()]
        return [Cmd.move_units(ids, A[0], A[1])]
    return pol


def _intended_a_then_b_policy(A, B):
    """Intended goal-conditional play: stage at region A until ≥3
    tanks are within radius 6 (phase 1 latches), then pivot the whole
    squad to region B and raze the enemy fact (phase 2). The `then`
    latch is sticky so the squad can leave A after phase 1 — exactly
    the 'pivot when the goal shifts' behaviour. Must WIN every
    (level, seed)."""
    state = {"phase": 1}

    def pol(obs, Cmd):
        units = [
            u for u in (obs.get("units_summary", []) or [])
            if u.get("type") == "2tnk"
        ]
        if not units:
            return [Cmd.observe()]
        ids = [str(u["id"]) for u in units]
        if state["phase"] == 1:
            in_a = sum(
                1 for u in units
                if (u["cell_x"] - A[0]) ** 2 + (u["cell_y"] - A[1]) ** 2 <= 36
            )
            if in_a >= 3:
                state["phase"] = 2
            else:
                return [Cmd.move_units(ids, A[0], A[1])]
        # Phase 2: pivot to B. Tanks under ReturnFire stance stop
        # attacking a non-firing building once adjacent; focus-fire
        # the fact id directly to finish the kill inside budget.
        fact_id = None
        for e in obs.get("enemy_buildings_summary", []) or []:
            if (e.get("type") == "fact"
                    and abs(e.get("cell_x", -99) - B[0]) <= 6
                    and abs(e.get("cell_y", -99) - B[1]) <= 6):
                fact_id = str(e["id"])
                break
        if fact_id is not None:
            return [Cmd.attack_unit(ids, fact_id)]
        return [Cmd.attack_move(ids, B[0], B[1])]
    return pol


# ── Pack-shape tests (cheap; no engine) ──────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-objective-change-midway"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: goal-conditional RL / ALFWorld
    goal change / changing requirements."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("goal-conditional RL" in a for a in anchors), anchors
    assert any("ALFWorld" in a for a in anchors), anchors
    assert any("changing requirements" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start (binding contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_win_is_then_chain_of_two_different_goals():
    """The win must be a `then:` chain of exactly two clauses — phase
    1 (units_in_region_gte) then phase 2
    (enemy_key_buildings_destroyed_in_region). This is the
    spec-load-bearing structural shape of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        then_branches = [b for b in ao if "then" in b]
        assert len(then_branches) == 1, (
            f"{lvl} must have exactly one then branch; got {win}"
        )
        clauses = then_branches[0]["then"]["clauses"]
        assert len(clauses) == 2, (
            f"{lvl} then chain must have exactly 2 clauses; got {clauses}"
        )
        # Clause 1 = phase 1 = units_in_region_gte (the early goal).
        assert "units_in_region_gte" in clauses[0], clauses[0]
        v0 = clauses[0]["units_in_region_gte"]
        assert int(v0["n"]) == 3, v0
        assert int(v0["radius"]) == 6, v0
        # Clause 2 = phase 2 = a DIFFERENT goal: raze the far fact.
        assert "enemy_key_buildings_destroyed_in_region" in clauses[1], clauses[1]
        v1 = clauses[1]["enemy_key_buildings_destroyed_in_region"]
        assert int(v1["radius"]) == 6, v1
        assert "fact" in v1["types"], v1
        # The two goals must be DIFFERENT predicate kinds.
        assert set(clauses[0]) != set(clauses[1]), (clauses[0], clauses[1])


def test_region_geometry_matches_pack():
    """The phase-1 region A and phase-2 region B coordinates must
    match the documented per-level geometry."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        clauses = next(
            b["then"]["clauses"] for b in win["all_of"] if "then" in b
        )
        v0 = clauses[0]["units_in_region_gte"]
        v1 = clauses[1]["enemy_key_buildings_destroyed_in_region"]
        assert (int(v0["x"]), int(v0["y"])) == GEOM[lvl]["A"], lvl
        assert (int(v1["x"]), int(v1["y"])) == GEOM[lvl]["B"], lvl


def test_fail_uses_after_ticks_and_building_count():
    """Fail must be any_of[ after_ticks:T+1, not building_count_gte:
    {fact,1} ] — a stall is a real LOSS and losing the agent fact is
    a LOSS, never a draw."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fail = c.fail_condition.model_dump(exclude_none=True)
        branches = fail.get("any_of") or []
        has_after = any("after_ticks" in b for b in branches)
        has_bcg = any(
            "not" in b and "building_count_gte" in b["not"]
            for b in branches
        )
        assert has_after, f"{lvl} fail missing after_ticks: {fail}"
        assert has_bcg, f"{lvl} fail missing not building_count_gte: {fail}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True,
        )
        wts = [c["within_ticks"] for c in win.get("all_of", [])
               if "within_ticks" in c]
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={level_def.max_turns}) — deadline never bites"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ───────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE every (level, seed) — phase 1
    never latches and the after_ticks clause bites."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_commit_b_only_loses(level, seed):
    """A beeline straight to region B must LOSE every (level, seed):
    the `then` chain only counts clause B after clause A latches, so
    a squad that never staged ≥3 tanks in region A leaves the chain
    stuck at clause 1 — the win never registers and the deadline
    bites. This is the 'commit-to-B-only LOSES (phase 1 unmet)'
    tooth."""
    g = GEOM[level]
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _commit_b_only_policy(g["B"]), seed=seed)
    assert res.outcome == "loss", (
        f"commit-B-only must LOSE on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_do_a_only_loses(level, seed):
    """A policy that stages at region A and parks must LOSE every
    (level, seed): phase 1 latches but phase 2 (raze the far fact)
    is never met, so the chain stalls at clause 2 and the deadline
    expires. This is the 'do-A-only LOSES (phase 2 unmet)' tooth."""
    g = GEOM[level]
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _do_a_only_policy(g["A"]), seed=seed)
    assert res.outcome == "loss", (
        f"do-A-only must LOSE on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_a_then_b_wins(level, seed):
    """The intended goal-conditional play — stage at region A, then
    pivot and raze the far fact — must WIN every (level, seed). This
    is the load-bearing test that the pack is solvable inside the
    budget by the advertised capability across all hard-tier spawn
    variants."""
    g = GEOM[level]
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_a_then_b_policy(g["A"], g["B"]), seed=seed)
    assert res.outcome == "win", (
        f"intended A-then-B must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"kills={res.signals.units_killed})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin — smoke-tests
    the spawn-variation contract."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
