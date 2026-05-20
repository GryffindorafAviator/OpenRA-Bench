"""proc-checklist-no-deviation — strict ordered visit-checklist.

ACTION focus: the procedure is GIVEN as an ordered named CHECKLIST of
waypoints; the model must visit them in the EXACT prescribed order
under a restricted action API (`move_units, observe, stop`). Order is
enforced by the latched, stateful `waypoint_sequence` predicate (n:1):
out-of-order, skip-one, beeline-to-final, and idle policies can never
satisfy it.

Anchors: IFBench step-order compliance, PlanBench strict ordering,
aviation pre-flight checklist, SOP no-skip / no-reorder. See
SCENARIO_REVIEW_CHECKLIST.md (A solvency / B stability / C capability).

The no-defect / no-cheat bar:

- Every level's win MUST use `waypoint_sequence` (n:1) — not parallel
  region clauses — so order is genuinely enforced.
- Every level's `within_ticks` and `after_ticks` MUST be reachable
  inside `max_turns` (engine ~90 ticks per decision turn) — otherwise
  a staller draws instead of losing.
- `after_ticks` MUST be exactly `within_ticks + 1` so a non-completing
  run is a real LOSS (never a draw).
- The intended in-order policy WINS on every seed 1..4 and every level.
- The three adversarial policies — stall, reorder, skip-one — LOSE
  on every seed 1..4 and every level (no path-coincidence cheat).
- Pack `tools` are exactly the {move_units, observe, stop} allow-list.
- The agent starts with 3 jeeps and there is NO enemy on the map
  (the only reachable fail is the timeout — `_NO_ENEMY` classification).
- Hard tier defines ≥2 distinct `spawn_point` groups (seed-driven
  starts) so a single memorised opening can't generalise.

This file mirrors the test pattern used by `test_coordination_
ordered_rendezvous.py` and `test_strict_spec.py::test_strict_sequence_
enforces_strict_order_no_cheat`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# openra_bench.scenarios eagerly imports the Rust adapter at module
# load (schema.py:15), so collection fails without the wheel. Skip the
# whole module if the env is missing — matches test_building_planning.
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "proc-checklist-no-deviation.yaml"

# Per-level expected checklist (must mirror the YAML clauses).
_POINTS = {
    "easy":   [(40, 4), (75, 36), (118, 20)],
    "medium": [(40, 4), (75, 36), (95, 4), (118, 36)],
    "hard":   [(40, 4), (75, 36), (95, 4), (118, 36), (118, 20)],
}

SEEDS = (1, 2, 3, 4)


# ---- structural / no-cheat invariants (pure-Python, no engine) ----


def _win_clauses(c):
    return dict(c.win_condition.__pydantic_extra__ or {})["all_of"]


def _fail_clauses(c):
    return dict(c.fail_condition.__pydantic_extra__ or {})["any_of"]


def _seq_value(c):
    for cl in _win_clauses(c):
        if "waypoint_sequence" in cl:
            return cl["waypoint_sequence"]
    return None


def test_meta_capability_and_anchors():
    pack = load_pack(PACK)
    assert pack.meta.capability == "action"
    assert pack.meta.id == "proc-checklist-no-deviation"
    # The four required real-world / benchmark anchors.
    anchors = set(pack.meta.benchmark_anchor)
    assert "IFBench step-order compliance" in anchors
    assert "PlanBench strict ordering" in anchors
    assert "aviation pre-flight checklist" in anchors
    assert "SOP no-skip / no-reorder" in anchors


def test_tool_allowlist_is_strict():
    pack = load_pack(PACK)
    # The action API the model has access to — EXACTLY these three.
    assert set(pack.base.get("tools", [])) == {"move_units", "observe", "stop"}


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_level_uses_waypoint_sequence_n1_with_expected_count(level):
    c = compile_level(load_pack(PACK), level)
    win = _win_clauses(c)
    seq = _seq_value(c)
    assert seq is not None, f"{level}: must use waypoint_sequence for order"
    # n=1: the spec says 'n:1, points list'.
    assert int(seq.get("n", 1)) == 1, f"{level}: must be n:1"
    expected_points = _POINTS[level]
    assert len(seq["points"]) == len(expected_points), (
        f"{level}: expected {len(expected_points)} waypoints, "
        f"got {len(seq['points'])}"
    )
    for got, (ex, ey) in zip(seq["points"], expected_points):
        assert (int(got["x"]), int(got["y"])) == (ex, ey)
    # Critically NOT using simultaneous regions (which would let a
    # beeline-to-final WIN — the action-multiunit-coordination shape
    # is wrong for an ORDERED checklist).
    assert not any("units_in_region_gte" in cl for cl in win), (
        f"{level}: should use waypoint_sequence, not units_in_region_gte"
    )


def test_waypoint_count_scales_with_difficulty():
    pack = load_pack(PACK)
    counts = [len(_seq_value(compile_level(pack, lvl))["points"])
              for lvl in ("easy", "medium", "hard")]
    assert counts == [3, 4, 5], (
        f"waypoint count must ladder 3 -> 4 -> 5 (one per tier), got {counts}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_deadline_and_timeout_are_reachable_and_bind(level):
    c = compile_level(load_pack(PACK), level)
    win = _win_clauses(c)
    fail = _fail_clauses(c)
    within = next(cl["within_ticks"] for cl in win if "within_ticks" in cl)
    after = next(cl["after_ticks"] for cl in fail if "after_ticks" in cl)
    max_tick = 93 + 90 * (c.max_turns - 1)
    # Deadline and timeout both reachable inside the turn budget.
    assert within <= max_tick, (level, within, max_tick)
    assert after <= max_tick, (level, after, max_tick)
    # Timeout strictly one tick past the deadline so a non-completing
    # run is a real LOSS (never a draw).
    assert after == within + 1, (level, after, within)


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_fail_condition_has_no_force_loss_combat_clause(level):
    # _NO_ENEMY pack: the only reachable fail is the timeout. The
    # `not own_units_gte:1` belt-and-suspenders clause is allowed
    # (the agent starts with 3 jeeps so it can't mis-fire on turn 1)
    # but the combat-driven `units_lost_lte` / `units_killed_gte`
    # clauses must NOT appear.
    c = compile_level(load_pack(PACK), level)
    fail = _fail_clauses(c)
    keys = {k for cl in fail for k in cl}
    assert "units_lost_lte" not in keys and "units_killed_gte" not in keys
    # Combat-fail absent, but the timeout LOSS path is present.
    assert any("after_ticks" in cl for cl in fail)


# ---- pure-Python predicate ordering invariants -----------------------


class _FakeSignals:
    def __init__(self):
        self.game_tick = 100
        self.seq_progress: dict = {}


def _ctx(units):
    return WinContext(
        signals=_FakeSignals(),
        render_state={"units_summary": [
            {"cell_x": x, "cell_y": y} for x, y in units
        ]},
    )


def test_predicate_skip_one_never_satisfies():
    """Predicate-level: skipping the middle waypoint NEVER satisfies."""
    sig = _FakeSignals()
    spec = {"waypoint_sequence": {
        "id": "T-skip", "n": 1, "radius": 5,
        "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 90, "y": 10}],
    }}
    ctx_at = lambda xy: WinContext(  # noqa: E731
        signals=sig,
        render_state={"units_summary": [{"cell_x": xy[0], "cell_y": xy[1]}]},
    )
    # W1 — advance.
    evaluate(spec, ctx_at((10, 10)))
    # Now jump to W3 (skip W2) repeatedly — must NEVER advance past 1.
    for _ in range(8):
        assert evaluate(spec, ctx_at((90, 10))) is False
    assert sig.seq_progress["T-skip"] == 1


def test_predicate_reorder_never_satisfies():
    """Predicate-level: visiting W2 first then never visiting W1 NEVER
    satisfies (latch stays at idx 0)."""
    sig = _FakeSignals()
    spec = {"waypoint_sequence": {
        "id": "T-reord", "n": 1, "radius": 5,
        "points": [{"x": 10, "y": 10}, {"x": 50, "y": 10}, {"x": 90, "y": 10}],
    }}
    ctx_at = lambda xy: WinContext(  # noqa: E731
        signals=sig,
        render_state={"units_summary": [{"cell_x": xy[0], "cell_y": xy[1]}]},
    )
    # Visit W2 / W3 only (in any number) — latch is stuck at W1.
    for _ in range(6):
        assert evaluate(spec, ctx_at((50, 10))) is False
        assert evaluate(spec, ctx_at((90, 10))) is False
    assert sig.seq_progress["T-reord"] == 0


# ---- live-engine scripted-policy sweep (4 policies × 4 seeds × 3 levels) ----
# This is the no-defect / no-cheat bar: the intended in-order policy
# WINS on every cell; stall / reorder / skip-one all LOSE on every
# cell. No path-coincidence cheat.


def _in_radius(units, x, y, r=5):
    return any((u["cell_x"] - x) ** 2 + (u["cell_y"] - y) ** 2 <= r * r
               for u in units)


def _make_visit(points):
    """Scripted policy: send every available unit to the next-needed
    waypoint, advance state when arrived (within radius 5)."""
    state = {"idx": 0}

    def fn(rs, Command):
        units = rs.get("units_summary", []) or []
        if not units:
            return [Command.observe()]
        i = state["idx"]
        if i >= len(points):
            return [Command.observe()]
        tx, ty = points[i]
        if _in_radius(units, tx, ty):
            state["idx"] = i + 1
            i = state["idx"]
            if i >= len(points):
                return [Command.observe()]
            tx, ty = points[i]
        ids = [str(u["id"]) for u in units]
        return [Command.move_units(ids, target_x=tx, target_y=ty)]

    return fn


def _stall_policy(rs, Command):
    return [Command.observe()]


def _make_reorder(points):
    # Swap W1 and W2 — true out-of-order policy. The latch can never
    # advance past idx 0 because W1 is visited AFTER W2.
    pts = list(points)
    pts[0], pts[1] = pts[1], pts[0]
    return _make_visit(pts)


def _make_skip_middle(points):
    # Drop the middle waypoint (latch never completes past it).
    mid = len(points) // 2
    return _make_visit(points[:mid] + points[mid + 1:])


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_in_order_policy_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, f"{level}: rush-hour-arena must resolve"
    res = run_level(c, _make_visit(_POINTS[level]), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed={seed}: intended in-order play must WIN, "
        f"got {res.outcome} (turns={res.turns})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_policy_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: stall must LOSE on the timeout, "
        f"got {res.outcome} (turns={res.turns})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_reorder_policy_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _make_reorder(_POINTS[level]), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: out-of-order play (swap W1<->W2) must "
        f"LOSE — the latch is stuck — got {res.outcome} (turns={res.turns})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_skip_one_policy_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _make_skip_middle(_POINTS[level]), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed={seed}: skip-one play must LOSE — the latch "
        f"never completes — got {res.outcome} (turns={res.turns})"
    )


# ---- hard-tier spawn variation (UPGRADED contract) -------------------


def test_hard_has_two_spawn_point_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups for seed-driven "
        f"start variation; got {sorted(sp)}"
    )


def test_hard_seeds_produce_distinct_starts():
    """Different seeds must place the agent's start cluster differently
    (the whole point of the two spawn_point groups)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import _scenario_to_tmp_yaml, RustEnvPool
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in SEEDS:
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(tuple(sorted((x["cell_x"], x["cell_y"]) for x in u)))
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )
