"""No-cheat + solvency proof for `econ-target-cash-amount-by-deadline`.

The capability: reach an exact SPENDABLE CASH bar (`cash_gte: N`, not
EV) AND a 2-harvester floor by a hard tick deadline. The model must
reason whether the pre-placed pipeline will clear the bar on time and
reinvest into harvesting capacity (the 2nd harv) — without draining
cash into army units that don't grow income.

The 2-harv structural clause is the anti-stall floor (family-2 §14):
the pre-placed harv auto-harvests near the pre-placed proc, so a
cash-only bar would be trivially stallable on every tier. The 2-harv
clause forces an explicit harv build (cost $1400 from the war factory)
on every tier.

For every level (and every hard seed 1-4) the no-cheat bar holds:

  - STALL (only `observe`)              → LOSS  (2-harv clause unmet)
  - ARMY-DRAIN (harvest + spam `e1`)    → LOSS  (cash drained + no
                                                 2nd harv)
  - BASELINE (1 harv, no reinvest)      → LOSS  on every tier
                                                 (2-harv clause unmet)
  - INTENDED (1 harv + build harv)      → WIN   on every level / seed

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

PACK_PATH = PACKS_DIR / "econ-target-cash-amount-by-deadline.yaml"

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
    """Do-nothing — must lose: starting_cash (600) < bar everywhere."""
    return [Command.observe()]


# Engine quirk (CLAUDE.md / startup-from-scratch / recover-from-zero-cash):
# re-issuing `Command.harvest(...)` every turn cancels the auto-cycle and
# the harv oscillates near the proc, dropping income to near-zero. The
# correct idiom is a ONE-SHOT harvest order per harv id (the engine's
# auto-cycle then carries the harv through the deposit→mine loop on its
# own). All scripted policies below use a `_sent` set to gate the order.


def _nearest_mine_for(harv_y: int) -> tuple[int, int]:
    """Pick the mine cell on the same latitude as the harv. Hard tier
    has two spawn latitudes (NORTH y=10..14, SOUTH y=28..32), each
    with its own near-patch pair. Easy/medium have patches at y=18/22.
    Returning the closest mine row keeps the harvest order targeted
    at a real ore cell so the auto-cycle engages."""
    candidates = [(22, 10), (22, 14), (22, 18), (22, 22), (22, 28), (22, 32)]
    return min(candidates, key=lambda m: abs(m[1] - harv_y))


def _harv_only_factory():
    """Pre-placed harv only, no reinvestment (the BASELINE). One-shot
    harvest per harv id (targeted at the nearest mine row) so the
    auto-cycle is allowed to run."""

    def make():
        sent = set()

        def p(rs, Command):
            harvs = [
                u for u in rs.get("units_summary", []) or []
                if u.get("type") == "harv"
            ]
            cmds = []
            for u in harvs:
                uid = str(u["id"])
                if uid in sent:
                    continue
                mx, my = _nearest_mine_for(int(u.get("cell_y", 18)))
                cmds.append(Command.harvest([uid], mx, my))
                sent.add(uid)
            return cmds if cmds else [Command.observe()]

        return p

    return make


# Back-compat alias for tests that still want a module-level callable.
def _harv_only(rs, Command):
    if not hasattr(_harv_only, "_policy"):
        _harv_only._policy = _harv_only_factory()()
    return _harv_only._policy(rs, Command)


def _army_drain_factory():
    """Harvest with the pre-placed harv (one-shot order) AND spam `e1`
    every turn that cash >= 100. e1 is ~100cr; the unit-cost drain
    treads cash near starting_cash and the 2-harv clause stays unmet —
    LOSS on every tier."""

    def make():
        sent = set()

        def p(rs, Command):
            cmds = []
            harvs = [
                u for u in rs.get("units_summary", []) or []
                if u.get("type") == "harv"
            ]
            for u in harvs:
                uid = str(u["id"])
                if uid in sent:
                    continue
                mx, my = _nearest_mine_for(int(u.get("cell_y", 18)))
                cmds.append(Command.harvest([uid], mx, my))
                sent.add(uid)
            if rs.get("cash", 0) >= 100:
                cmds.append(Command.build("e1"))
            return cmds if cmds else [Command.observe()]

        return p

    return make


def _army_drain(rs, Command):
    if not hasattr(_army_drain, "_policy"):
        _army_drain._policy = _army_drain_factory()()
    return _army_drain._policy(rs, Command)


def _intended_factory(extra_harv: int):
    """Reinvestment factory: one-shot harvest order per harv id
    (targeted at the nearest-row mine) AND queue `extra_harv`
    additional harvesters (cost 1400 each, prereq fact+powr+weap)."""

    def make():
        queued = [0]
        sent = set()

        def p(rs, Command):
            cmds = []
            harvs = [
                u for u in rs.get("units_summary", []) or []
                if u.get("type") == "harv"
            ]
            for u in harvs:
                uid = str(u["id"])
                if uid in sent:
                    continue
                mx, my = _nearest_mine_for(int(u.get("cell_y", 18)))
                cmds.append(Command.harvest([uid], mx, my))
                sent.add(uid)
            if queued[0] < extra_harv and rs.get("cash", 0) >= 1400:
                cmds.append(Command.build("harv"))
                queued[0] += 1
            return cmds if cmds else [Command.observe()]

        return p

    return make


# ───────────────────────── solvency: intended WINS ────────────────────────


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_easy_intended_wins(seed):
    """Easy bar ($1500 cash AND ≥2 harvesters by tick 1800). The 2-harv
    clause is the anti-stall floor (family-2 §14) — without it the
    auto-harvested pre-placed harv would clear the cash bar by stall.
    Build 1 extra harv ⇒ the 2-harv clause is met and the doubled
    income clears the $1500 bar inside 1800 ticks."""
    c = compile_level(load_pack(PACK_PATH), "easy")
    r = run_level(c, _intended_factory(1)(), seed=seed)
    assert r.outcome == "win", (
        f"easy intended (+1 harv) seed={seed} should WIN, got {r.outcome} "
        f"(cash={r.signals.cash}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_intended_wins(seed):
    """Medium bar (5500 by tick 4500) needs reinvestment. Run the
    INTENDED policy (build 1 extra harv); it must WIN."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    r = run_level(c, _intended_factory(1)(), seed=seed)
    assert r.outcome == "win", (
        f"medium intended (+1 harv) seed={seed} should WIN, got "
        f"{r.outcome} (cash={r.signals.cash}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_intended_wins_every_seed(seed):
    """Hard bar ($6500 cash AND ≥2 harvs by tick 4000) on each of two
    spawn groups. Build 1 extra harv: the doubled income covers the
    $1400 capex AND clears the cash bar by tick ~3963 (verified on
    seeds 1-4). Queueing >1 extra harv wastes capex relative to the
    deadline and falls short — the optimal reinvest is exactly +1."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, _intended_factory(1)(), seed=seed)
    assert r.outcome == "win", (
        f"hard intended (+1 harv) seed={seed} should WIN, got "
        f"{r.outcome} (cash={r.signals.cash}, tick={r.signals.game_tick})"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """Stall idles at starting_cash (600) < bar (1500/5500/6500). The
    `after_ticks` deadline must bite as a real LOSS (not a draw)."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, stall, seed=1)
    assert r.outcome == "loss", (
        f"{level} stall must be a real timeout LOSS, got {r.outcome} "
        f"(cash={r.signals.cash})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_army_drain_loses_on_every_level(level):
    """Spamming `e1` alongside harvest exactly absorbs the harv
    income (~100cr/turn each). Cash hovers near starting_cash and
    never clears the bar — LOSS on every level."""
    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _army_drain, seed=1)
    assert r.outcome == "loss", (
        f"{level} army-drain must LOSE (cash drained by units), got "
        f"{r.outcome} (cash={r.signals.cash})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_medium_baseline_no_reinvest_loses(seed):
    """Baseline (1 pre-placed harv, no reinvest) LOSES on every seed —
    the 2-harv structural clause is unmet regardless of how much cash
    the auto-cycle accumulates. This is what makes the medium tier a
    capex-decision test."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    r = run_level(c, _harv_only, seed=seed)
    assert r.outcome == "loss", (
        f"medium baseline seed={seed} must LOSE (2-harv clause unmet), "
        f"got {r.outcome} (cash={r.signals.cash}, "
        f"tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_hard_baseline_no_reinvest_loses(seed):
    """Baseline (1 pre-placed harv, no reinvest) LOSES on every seed —
    the 2-harv structural clause is unmet regardless of how much cash
    the auto-cycle accumulates inside the 4000-tick deadline."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    r = run_level(c, _harv_only, seed=seed)
    assert r.outcome == "loss", (
        f"hard baseline seed={seed} must LOSE (2-harv clause unmet), "
        f"got {r.outcome} (cash={r.signals.cash}, "
        f"tick={r.signals.game_tick})"
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
