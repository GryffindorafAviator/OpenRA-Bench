"""No-cheat + solvency proof for `build-power-down-defensive`
(Mid-phase Group I REASONING: reversible load-shedding via the
`power_down` toggle under an active overdraw).

The pack tests the OPERATIONS-RUNBOOK discipline: when the grid is
overdrawn (provided < drained), the agent must SHED non-essential
loads by toggling them off (`power_down`), NOT by selling them. A
model that:

  * STALLS (only `observe`) → grid stays overdrawn → win clauses fail
    → reachable timeout LOSS;
  * SELLS the same drainers (instead of powering down) → the
    `building_count_gte` clauses on those structures all FAIL → LOSS;
  * POWER-DOWNS THE LONE POWR (the destructive load-shed) → provided
    collapses to 0 → fails the `power_provided_gte:100` floor → LOSS;
  * INTENDED powers down a sufficient subset of non-essential
    drainers (tent+barr on easy; weap+dome+tent on medium/hard) →
    surplus restored, all structures intact → WIN.

This is the bench-side proof that the engine-side `PowerDown`
implementation (this commit) and the `power_provided_gte` predicate
together close the load-shedding capability gap.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-power-down-defensive.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Stall: never toggles power → grid stays overdrawn → LOSS on the
    deadline."""
    return [Command.observe()]


def _sell_drainers(rs, Command):
    """Destructive shed: SELL the drainers instead of `power_down`-ing
    them. The drainers ARE no longer drawing, but the
    `building_count_gte:{type:tent, n:1}` (etc.) win clauses now FAIL
    because the buildings are gone — a sold building cannot be
    reactivated, which is the whole reason `power_down` exists."""
    raw = rs.get("_raw", {}) or {}
    bldgs = raw.get("own_buildings", []) or []
    surplus = int(rs.get("power_provided", 0)) - int(rs.get("power_drained", 0))
    if surplus >= 0:
        return [Command.observe()]
    targets = ("tent", "barr", "weap", "dome")
    to_sell = [str(b["id"]) for b in bldgs if b["type"] in targets]
    if to_sell:
        return [Command.sell(to_sell)]
    return [Command.observe()]


def _power_down_powr(rs, Command):
    """Wrong-pick load-shed: power_down the LONE Power Plant. This
    "fixes" the surplus by collapsing supply, but the
    `power_provided_gte:100` floor immediately fails (provided →
    0). One-shot — we don't re-toggle (would oscillate)."""
    raw = rs.get("_raw", {}) or {}
    bldgs = raw.get("own_buildings", []) or []
    if int(rs.get("power_provided", 0)) > 0:
        to_shed = [str(b["id"]) for b in bldgs if b["type"] == "powr"]
        if to_shed:
            return [Command.power_down(to_shed)]
    return [Command.observe()]


def _intended_shed(rs, Command):
    """Intended load-shed: power_down a sufficient combination of
    NON-ESSENTIAL drainers to restore non-negative surplus, WITHOUT
    touching the powr and WITHOUT selling.

    Easy (overdraw −40): tent + barr sheds 40 exactly.
    Medium / Hard (overdraw −80): weap + dome + tent sheds 90 ≥ 80.
    We greedily shed weap+dome+tent on all levels — easy still wins
    (over-sheds, surplus = +30, still satisfies surplus_gte:0 and all
    building_count_gte clauses)."""
    raw = rs.get("_raw", {}) or {}
    bldgs = raw.get("own_buildings", []) or []
    surplus = int(rs.get("power_provided", 0)) - int(rs.get("power_drained", 0))
    if surplus >= 0:
        return [Command.observe()]
    # Greedy combo that covers both easy (−40) and medium/hard (−80).
    targets = ("weap", "dome", "tent")
    to_shed = [str(b["id"]) for b in bldgs if b["type"] in targets]
    if to_shed:
        return [Command.power_down(to_shed)]
    return [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-power-down-defensive"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # The load-shed test needs `power_down` (the intended primitive)
    # plus `sell` (so the destructive-baseline policy can be tested
    # and shown to lose). No build / no move / no attack — the
    # decision is exclusively which buildings to toggle off.
    for t in ("observe", "power_down", "sell", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"
    for forbidden in ("build", "place_building", "move_units", "attack_unit"):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in the load-shed palette"
        )


def test_benchmark_anchor_lists_load_shedding_and_runbook():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "load-shed" in blob or "runbook" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    win_all_of = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in win_all_of if "within_ticks" in x)
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt + 1})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_includes_power_floor_and_surplus(level):
    """Structural: every level's win must include BOTH `power_surplus_gte`
    (the load-shed teeth) AND `power_provided_gte` (the anti-cheat
    floor against killing supply to satisfy the surplus)."""
    c = compile_level(load_pack(PACK), level)
    all_of = c.win_condition.model_dump().get("all_of", [])
    keys = [list(x.keys())[0] for x in all_of]
    assert "power_surplus_gte" in keys, f"{level}: missing power_surplus_gte"
    assert "power_provided_gte" in keys, f"{level}: missing power_provided_gte"


def test_hard_has_two_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN ───────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_shed_wins(level, seed):
    """Intended load-shed (power_down a sufficient subset of non-
    essential drainers) WINS every level × every seed."""
    c, r = _run(level, _intended_shed, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended shed should WIN, got {r.outcome} "
        f"(p_prov={r.signals.power_provided}, p_drain={r.signals.power_drained}, "
        f"tick={r.signals.game_tick})"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — grid stays overdrawn
    (surplus < 0) so the surplus_gte:0 clause never satisfies →
    reachable timeout LOSS."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(p_prov={r.signals.power_provided}, p_drain={r.signals.power_drained}, "
        f"tick={r.signals.game_tick})"
    )
    # Surplus stayed negative throughout (we never powered down).
    assert (
        int(r.signals.power_provided) - int(r.signals.power_drained) < 0
    ), "stall should leave the grid overdrawn"


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_destructive_sell_loses(level, seed):
    """Selling the drainers instead of powering them down fixes the
    surplus but FAILS the `building_count_gte` win clauses (the
    structures are gone). LOSS on the deadline."""
    c, r = _run(level, _sell_drainers, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} sell must LOSE; got {r.outcome} "
        f"(p_prov={r.signals.power_provided}, p_drain={r.signals.power_drained})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_wrong_pick_power_down_powr_loses(level, seed):
    """Powering down the LONE powr zeros provided power: the
    surplus_gte:0 clause is satisfied (0 ≥ 0) but the
    `power_provided_gte:100` floor FAILS. LOSS on the deadline. This
    is the anti-cheat that the `power_provided_gte` predicate was
    introduced for — without that floor, killing supply would trivially
    "fix" the surplus."""
    c, r = _run(level, _power_down_powr, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} power_down(powr) must LOSE; got {r.outcome} "
        f"(p_prov={r.signals.power_provided}, p_drain={r.signals.power_drained})"
    )
    # Provided collapsed to 0 (the lone powr is shed).
    assert int(r.signals.power_provided) == 0, (
        f"powering down the lone powr should zero provided; "
        f"got {r.signals.power_provided}"
    )


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Hard tier contract: seeds 1-4 round-robin between the two
    declared spawn_point groups (NORTH y≈12 / SOUTH y≈28). Inert
    spawn-witness e1 per group surfaces the variation via
    units_summary so the shared hard-tier check passes (a building-
    only base produces empty units and breaks the contract)."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
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
                starts.add(tuple(sorted((x["cell_x"], x["cell_y"]) for x in u)))
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, _intended_shed, seed=2)
    b = run_level(c, _intended_shed, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
