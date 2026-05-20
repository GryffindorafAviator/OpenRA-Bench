"""proc-ordered-action-strict — strict ordered building placements.

Verifies the `then:[A,B,...]` composite enforces a literal ORDER on
building placements: only the intended (in-order) policy WINS; every
out-of-order, idle, or brute-shotgun policy LOSES, on every level and
every seed 1..4 (no-cheat-no-defect bar).

Capability: action (procedural compliance). The pack is the Group I
seed for the strict-ordered-action cell — anchor PlanBench strict
ordering, IFBench step-order compliance, PERT/CPM precedence,
aviation pre-flight checklist, database migration ordered DDL.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "proc-ordered-action-strict.yaml"

# Per-level ordered cell list (must match the YAML clauses).
_POINTS = {
    "easy":   [(30, 20), (90, 20)],
    "medium": [(30, 20), (60, 20), (90, 20)],
    "hard":   [(30, 20), (50, 20), (70, 20), (90, 20)],
}

SEEDS = (1, 2, 3, 4)


# ---- policies (scripted, no model) ------------------------------------


def _has_pbox_at(pbox_xys, target, r2: int = 25) -> bool:
    tx, ty = target
    return any((x - tx) ** 2 + (y - ty) ** 2 <= r2 for (x, y) in pbox_xys)


def _make_intended(points):
    """Strict in-order: build pbox, place at next-needed point only."""

    def fn(rs, Command):
        bt = [
            (b["type"], b["cell_x"], b["cell_y"])
            for b in rs.get("own_buildings", [])
        ]
        pbox = [(x, y) for (t, x, y) in bt if t == "pbox"]
        for tgt in points:
            if not _has_pbox_at(pbox, tgt):
                prod = rs.get("production", []) or []
                cmds = []
                if "pbox" not in prod:
                    cmds.append(Command.build("pbox"))
                cmds.append(Command.place_building("pbox", tgt[0], tgt[1]))
                return cmds
        return [Command.observe()]

    return fn


def _make_wrong_order(points):
    """Same as intended, but in REVERSED order — places the last point
    FIRST, which must immediately trip the out-of-order fail clause."""
    return _make_intended(list(reversed(points)))


def _stall(rs, Command):
    """Idle. Must lose on the timeout."""
    return [Command.observe()]


def _make_brute_all_at_once_wrong_finish(points):
    """Brute shotgun: queue `pbox` every turn AND fire `place_building`
    at multiple cells each turn — but the LAST cell in the spam list is
    a deliberately WRONG (out-of-region) cell. The engine produces one
    pbox per build cycle and (with `pbox` first in the spam) places at
    the earliest valid spam target; the wrong-cell suffix wastes the
    final unit. The procedure never completes ⇒ LOSS (on the order
    fail clause or the deadline; hard adds the building-total ceiling)."""
    bad_cell = (120, 5)
    spam = list(points[:-1]) + [bad_cell]

    def fn(rs, Command):
        prod = rs.get("production", []) or []
        cmds = []
        if "pbox" not in prod:
            cmds.append(Command.build("pbox"))
        for tx, ty in spam:
            cmds.append(Command.place_building("pbox", tx, ty))
        return cmds

    return fn


# ---- structural tests --------------------------------------------------


def test_pack_compiles_with_three_levels_and_then_composite():
    pack = load_pack(PACK)
    assert pack.meta.id == "proc-ordered-action-strict"
    assert pack.meta.capability == "action"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    assert pack.meta.benchmark_anchor, "must declare benchmark_anchor"
    # The whole pack point: each level's win uses `then:[...]`.
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        win = dict(c.win_condition.__pydantic_extra__ or {})
        all_of = win.get("all_of") or []
        assert any("then" in cl for cl in all_of), (
            f"{lvl}: win_condition must use `then:[...]`, got {all_of}"
        )
        # Number of clauses scales with difficulty.
        then_clause = next(cl["then"] for cl in all_of if "then" in cl)
        assert len(then_clause["clauses"]) == len(_POINTS[lvl])


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, not a draw: the after_ticks fail
    deadline must be reachable inside max_turns (tick ≈ 93+90·(N-1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    clauses = fc.get("any_of", [fc])
    afts = [cl["after_ticks"] for cl in clauses if "after_ticks" in cl]
    assert afts, f"{level}: fail must include an after_ticks deadline"
    reachable = 93 + 90 * (c.max_turns - 1)
    assert min(afts) <= reachable, (
        f"{level}: after_ticks {afts} unreachable in {c.max_turns} turns "
        f"(max tick {reachable}) — draw degeneracy"
    )


# ---- scripted-policy parametric tests ---------------------------------


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_ordered_policy_wins(level, seed):
    """Only this policy honours the strict order — and so it WINS on
    every level and every hard seed (1..4)."""
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported
    res = run_level(c, _make_intended(_POINTS[level]), seed=seed)
    assert res.outcome == "win", (
        f"{level} seed{seed} intended: should WIN, got {res.outcome}; "
        f"buildings={res.signals.own_buildings}"
    )
    # The intended chain debited the budget (every pbox costs 600).
    assert res.signals.cash < c.starting_cash


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_wrong_order_policy_loses(level, seed):
    """Reverse-order placement trips the out-of-order fail clause the
    instant the first (wrong) building lands ⇒ unrecoverable LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _make_wrong_order(_POINTS[level]), seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed} wrong-order: should LOSE on out-of-order, "
        f"got {res.outcome}; buildings={res.signals.own_buildings}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_policy_loses(level, seed):
    """Idle policy loses on the timeout — confirms the deadline bites."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed} stall: should LOSE on timeout, got "
        f"{res.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", SEEDS)
def test_brute_all_at_once_wrong_finish_loses(level, seed):
    """Brute shotgun (queue + place at every cell every turn) with a
    deliberately WRONG final placement never satisfies the ordered
    chain ⇒ LOSS (via the order fail clause when an earlier-region
    placement lags, OR the building-total ceiling on hard, OR the
    deadline)."""
    c = compile_level(load_pack(PACK), level)
    fn = _make_brute_all_at_once_wrong_finish(_POINTS[level])
    res = run_level(c, fn, seed=seed)
    assert res.outcome == "loss", (
        f"{level} seed{seed} brute-wrong-finish: should LOSE, got "
        f"{res.outcome}; buildings={res.signals.own_buildings}"
    )


# ---- determinism -------------------------------------------------------


def test_intended_run_is_deterministic():
    c = compile_level(load_pack(PACK), "medium")
    fn = _make_intended(_POINTS["medium"])
    a = run_level(c, fn, seed=3)
    b = run_level(c, fn, seed=3)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must yield identical outcome / cash / turns"
