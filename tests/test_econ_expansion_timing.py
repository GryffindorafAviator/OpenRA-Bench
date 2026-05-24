"""No-cheat + solvency proof for `econ-expansion-timing`.

The capability: capacity-expansion TIMING under a hard deadline. The
agent inherits a running economy (fact + proc + powr + weap + 1 harv)
and reserve cash for EXACTLY one more harvester. The economy bar M is
set so a single harvester cannot reach it before the deadline T; a
second harvester roughly doubles income but costs 1100 up front and
warms up over ~450 ticks, so its payback period only fits the budget
if the buy is committed EARLY — before a break-even tick.

For every level (and every hard seed 1-4) the no-cheat bar holds:

  - STALL (only `observe`)              → LOSS  (idles at starting_cash)
  - ONE-HARV-ONLY (harvest, no expand)  → LOSS  (income too slow)
  - ARMY-DRAIN (harvest + spam `e1`)    → LOSS  (e1 prereq-blocked;
                                          == one-harv-only, still short)
  - BUILD-TOO-LATE (2nd harv at t~1800) → LOSS  (payback window misses)
  - INTENDED (2nd harv bought EARLY)    → WIN   on every level / seed

Plus tick/turn alignment, fail-condition reachability, hard spawn
contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK_PATH = PACKS_DIR / "econ-expansion-timing.yaml"

# All near-patch mine cells across levels — the scripted harvest
# policy tries each (the engine picks the reachable one per spawn).
ALL_NEAR_MINES = [
    (22, 18),
    (22, 22),  # easy + medium pre-placed
    (22, 10),
    (22, 14),  # hard NORTH spawn
    (22, 28),
    (22, 32),  # hard SOUTH spawn
]


# ───────────────────────── scripted policies ──────────────────────────────


def stall(rs, Command):
    """Do-nothing — must lose: starting_cash (1100) < every bar."""
    return [Command.observe()]


def one_harv_only(rs, Command):
    """Run the pre-placed harvester only — the BASELINE. The engine's
    `auto_route_idle_harvesters` hook installs a Harvest activity on
    any owned idle harvester whose owner has a `proc`; the pack ships
    a pre-placed harv + proc so the single harv auto-cycles without
    any explicit `harvest` order. Income tops out near 4000 cr over
    30 turns — below every tier's bar."""
    return [Command.observe()]


def army_drain(rs, Command):
    """Spam `e1` — the "spend the reserve on a non-revenue line"
    decoy. e1 has no Infantry-queue host (no tent), so the build is
    prereq-blocked; the play degenerates to one-harv-only and still
    misses the bar."""
    return [Command.build("e1")]


def _expand_factory(build_after_tick: int):
    """Reinvestment factory: rely on the engine's auto-harvest (the
    pre-placed harv auto-cycles the mines at (22,18)/(22,22) the
    moment the pre-placed proc exists) AND buy ONE extra harvester
    once `game_tick >= build_after_tick`. Closure-local state so each
    test invocation starts fresh. The 2nd harv auto-routes the same
    way once it spawns (the Vehicle-queue is hosted by the pre-placed
    `weap`)."""

    def make():
        bought = [False]

        def p(rs, Command):
            if (
                not bought[0]
                and rs.get("game_tick", 0) >= build_after_tick
                and rs.get("cash", 0) >= 1100
            ):
                bought[0] = True
                return [Command.build("harv")]
            return [Command.observe()]

        return p

    return make


# ───────────────────────── solvency: intended WINS ────────────────────────


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_intended_early_expand_wins(level):
    """Buying the 2nd harvester EARLY (turn 1) clears the bar before
    the deadline — the intended capability."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _expand_factory(0)(), seed=1)
    ev = r.signals.cash + r.signals.resources
    assert r.outcome == "win", (
        f"{level} intended early-expand should WIN, got {r.outcome} "
        f"(EV={ev}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_intended_early_expand_wins_every_seed(seed):
    """Hard bar (10200 EV by tick 4500) on each of two spawn groups
    clears only with an EARLY 2nd-harv buy. Must WIN on every seed."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, _expand_factory(0)(), seed=seed)
    ev = r.signals.cash + r.signals.resources
    assert r.outcome == "win", (
        f"hard intended early-expand seed={seed} should WIN, got "
        f"{r.outcome} (EV={ev}, tick={r.signals.game_tick})"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """Stall idles at starting_cash (1100) < every bar. The
    `after_ticks` deadline must bite as a real LOSS (not a draw)."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, stall, seed=1)
    assert r.outcome == "loss", (
        f"{level} stall must be a real timeout LOSS, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_one_harv_only_loses_on_every_level(level):
    """A single harvester's post-warmup income (~1.1 cash/tick) tops
    out below every bar before the deadline — running one harvester
    and never expanding LOSES on every level."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, one_harv_only, seed=1)
    assert r.outcome == "loss", (
        f"{level} one-harv-only must LOSE (income too slow), got "
        f"{r.outcome} (EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_army_drain_loses_on_every_level(level):
    """Spending the reserve on `e1` instead of a harvester does not
    grow income (and here the build is prereq-blocked anyway) — the
    bar is missed on every level."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, army_drain, seed=1)
    assert r.outcome == "loss", (
        f"{level} army-drain must LOSE (reserve not on revenue), got "
        f"{r.outcome} (EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_build_too_late_loses_on_every_level(level):
    """Buying the 2nd harvester LATE (tick ~1800, turn ~20) is the
    core discrimination: the doubled income does not run long enough
    for the capex to amortise before the deadline — LOSS on every
    level. This is what makes the pack a capex-TIMING test, not a
    capex-existence test."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _expand_factory(1800)(), seed=1)
    assert r.outcome == "loss", (
        f"{level} build-too-late must LOSE (payback window missed), "
        f"got {r.outcome} (EV={r.signals.cash + r.signals.resources}, "
        f"tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_one_harv_only_loses_every_seed(seed):
    """One-harv-only LOSES on every hard seed (both spawn groups have
    identical income economics by design)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, one_harv_only, seed=seed)
    assert r.outcome == "loss", (
        f"hard one-harv-only seed={seed} must LOSE, got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


# ───────────────────────── tick/turn alignment ────────────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_deadline_reachable_inside_max_turns(level):
    """`within_ticks` (and the matching `after_ticks` fail) must be
    reachable inside max_turns — otherwise stall draws instead of
    losing. Engine advances ~90 ticks per turn (tick ≈ 93+90·(t-1))."""
    c = compile_level(load_pack(PACK_PATH), level)
    max_tick = 93 + 90 * (c.max_turns - 1)
    win_extra = c.win_condition.__pydantic_extra__ or {}
    fail_extra = (
        c.fail_condition.__pydantic_extra__ if c.fail_condition else {}
    ) or {}

    def _find(node, key):
        if not isinstance(node, dict):
            return None
        for k, v in node.items():
            if k == key:
                return int(v)
            if k in ("all_of", "any_of"):
                for c2 in v:
                    r = _find(c2, key)
                    if r is not None:
                        return r
        return None

    within = _find(win_extra, "within_ticks")
    assert within is not None, f"{level} should have within_ticks"
    assert within <= max_tick, (
        f"{level} within_ticks={within} > max reachable tick {max_tick} "
        f"(max_turns={c.max_turns}); deadline never bites"
    )
    after = _find(fail_extra, "after_ticks")
    assert after is not None, f"{level} should have an after_ticks fail"
    assert after <= max_tick, (
        f"{level} after_ticks={after} > max reachable tick {max_tick}"
    )


# ───────────────────────── hard spawn contract ────────────────────────────


def test_hard_has_two_spawn_point_groups():
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups, got {sorted(sp)}"
    )


def test_hard_seed_round_robin_produces_distinct_starts():
    """Smoke that the engine actually round-robins the spawn groups by
    seed (the whole point of declaring multiple spawn_point groups)."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK_PATH), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )
