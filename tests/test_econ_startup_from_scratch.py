"""No-cheat + solvency proof for `econ-startup-from-scratch` (Group F
opening greenfield ramp from zero).

The pack tests the cold-start opening: from a single MCV plus a small
cash budget — but NO buildings, NO harvester, NO income — the agent
must stand up the first economy pipeline (deploy MCV → fact, build
powr (prereq for proc), build proc (auto-spawns one free harv)) and
then COMMIT to harvesting so revenue actually accumulates past the
bar.

For every level + every hard seed (1-4):
  * the INTENDED full chain (deploy → powr → proc → move+harvest)
    WINS;
  * STALL (only `observe`) LOSES — no proc, no harv → both structural
    win clauses fail, clock bites as a real timeout LOSS;
  * DEPLOY-ONLY (deploys but never builds anything) LOSES — fact
    exists but no proc / harv → structural win clauses fail;
  * BRUTE-BUILD-ARMY (deploy → powr → tent → spam e1, never builds
    proc) LOSES — no proc, no income;
  * DEPLOY-BUILD-NO-HARVEST (full chain but no harvest order) LOSES —
    after the chain spend, cash is $200, well below the bar; the
    spawned harv sits idle (engine quirk: a bare `Command.harvest`
    on a freshly-spawned harv doesn't auto-path; `move_units` first
    is needed).

The 4 lazy plays + 1 intended × 3 levels × 4 seeds gives the full
no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "econ-startup-from-scratch.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _patch_for_fact(fy: int) -> tuple[int, int]:
    """easy/medium have one patch at (26,20); hard has spawn-local
    patches at (26,14) [north] and (26,26) [south]. The MCV's deploy
    fact lands near (fx, fy) ≈ (mcv_x-1, mcv_y-1) — pick the patch
    whose row is closest."""
    if abs(fy - 19) <= 2:
        return (26, 20)
    if fy < 19:
        return (26, 14)
    return (26, 26)


def _stall(rs, Command):
    """Idle: never deploys, never builds. No proc + no harv ever →
    structural win clauses fail → clock bites as LOSS."""
    return [Command.observe()]


def _deploy_only(rs, Command):
    """Deploys the MCV (fact appears) but never builds anything →
    no proc, no harv → win clauses fail → LOSS."""
    units = rs.get("units_summary", []) or []
    mcv = next(
        (u for u in units if str(u.get("type", "")).lower() == "mcv"), None
    )
    if mcv is not None:
        return [Command.deploy([str(mcv["id"])])]
    return [Command.observe()]


def _brute_build_army_factory():
    """Deploy → powr → tent → spam e1. NEVER builds proc → no harv,
    no income → cash drains → LOSS on every tier."""

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        mcv = next(
            (u for u in units if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        if mcv is not None:
            return [Command.deploy([str(mcv["id"])])]
        fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        if "powr" not in own_types:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 3, fy + 1))
            return cmds
        if "tent" not in own_types:
            cmds = []
            if "tent" not in prod:
                cmds.append(Command.build("tent"))
            cmds.append(Command.place_building("tent", fx - 2, fy + 3))
            return cmds
        if rs.get("cash", 0) >= 100:
            return [Command.build("e1")]
        return [Command.observe()]

    return policy


def _deploy_build_no_harvest_factory():
    """Full chain (deploy → powr → proc) but NO harvest order. The
    spawned-from-proc harv sits idle at the proc (a bare harvest
    order from a freshly-spawned harv doesn't auto-path — engine
    quirk; the intended chain uses `move_units` first). After the
    chain spend ($1700), cash is $200, well below the bar → LOSS."""

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        mcv = next(
            (u for u in units if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        if mcv is not None:
            return [Command.deploy([str(mcv["id"])])]
        fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        if "powr" not in own_types:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 3, fy + 1))
            return cmds
        if "proc" not in own_types:
            cmds = []
            if "proc" not in prod:
                cmds.append(Command.build("proc"))
            cmds.append(Command.place_building("proc", fx + 3, fy + 3))
            return cmds
        return [Command.observe()]

    return policy


def _intended_factory():
    """Intended capability policy: deploy MCV → build powr (prereq) →
    build proc (auto-spawns 1 free harv) → move_units the harv onto
    the local patch → harvest. The auto-cycle then sustains income
    until the bar is cleared. Wins every tier × every seed."""

    state = {"harv_moved": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        harvs_info = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        mcv = next(
            (u for u in units if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        if mcv is not None:
            return [Command.deploy([str(mcv["id"])])]
        fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        if "powr" not in own_types:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 3, fy + 1))
            return cmds
        if "proc" not in own_types:
            cmds = []
            if "proc" not in prod:
                cmds.append(Command.build("proc"))
            cmds.append(Command.place_building("proc", fx + 3, fy + 3))
            return cmds
        px, py = _patch_for_fact(fy)
        cmds = []
        for uid, cx, cy in harvs_info:
            uid_s = str(uid)
            if uid_s not in state["harv_moved"]:
                cmds.append(
                    Command.move_units([uid_s], target_x=px, target_y=py)
                )
                if abs(cx - px) <= 3 and abs(cy - py) <= 3:
                    state["harv_moved"].add(uid_s)
            else:
                cmds.append(Command.harvest([uid_s], px, py))
        return cmds if cmds else [Command.observe()]

    return policy


# ───────────────────────── helpers ────────────────────────────────────────


def _ev(res):
    return res.signals.cash + res.signals.resources


def _run(level, policy_or_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    pol = policy_or_factory() if callable(policy_or_factory) and policy_or_factory.__name__.endswith("factory") else policy_or_factory
    return c, run_level(c, pol, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-startup-from-scratch"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Spec-required tools (Wave-4 spec).
    for t in ("observe", "deploy", "build", "place_building", "harvest", "move_units", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"


def test_benchmark_anchor_lists_sc2le_and_mvp_pivot():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "sc2le" in blob, anchors
    assert "minerl" in blob or "mineral" in blob, anchors
    assert "greenfield" in blob, anchors
    assert "mvp" in blob or "startup" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns (tick ≈ 93 +
    90·(max_turns − 1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    # within_ticks + 1 == after_ticks (non-finisher LOSES on the very
    # next tick after the win window closes).
    within_clauses = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x)
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt+1})"
    )


def test_hard_has_two_spawn_groups_for_mcv():
    """Hard tier contract: ≥2 distinct agent spawn_point groups (NORTH
    base / SOUTH base) so seed round-robin varies the MCV start cell
    and a memorised opening can't generalise."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN bar ───────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_full_chain_wins(level, seed):
    """Intended deploy → powr → proc → move+harvest WINS every level
    × every seed; ev surpasses the bar via real harvest income (the
    chain spend leaves only $200, so cash alone cannot satisfy the
    bar)."""
    c, r = _run(level, _intended_factory, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended chain should WIN, got "
        f"{r.outcome}; ev={_ev(r)}, cash={r.signals.cash}, "
        f"types={r.signals.own_building_types}, tick={r.signals.game_tick}"
    )
    # Structural sanity: proc materialised AND at least one harv
    # was spawned (the auto-spawn from proc, surfaced as the
    # `harvesters` count in signals).
    types = set(r.signals.own_building_types)
    assert "proc" in types, types
    assert r.signals.harvesters >= 1, (
        f"expected at least 1 harv after proc completes, "
        f"harvesters={r.signals.harvesters}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall (only `observe`) must LOSE every level × every seed —
    no proc + no harv → structural win clauses fail → reachable
    timeout LOSS (not a draw)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, tick={r.signals.game_tick})"
    )
    # Never created proc / harv (no deploy at all).
    assert "proc" not in r.signals.own_building_types
    assert "fact" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_deploy_only_loses(level, seed):
    """Deploy-only must LOSE — fact exists, but no proc / no harv →
    structural win clauses fail → LOSS."""
    c, r = _run(level, _deploy_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} deploy-only must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, types={r.signals.own_building_types})"
    )
    # Fact landed but proc never built.
    assert "proc" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_brute_build_army_loses(level, seed):
    """Deploy → powr → tent → spam e1 (NEVER builds proc) must LOSE —
    no proc, no harv, no income; cash drains to ~0; bar unmet."""
    c, r = _run(level, _brute_build_army_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} brute-build-army must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, types={r.signals.own_building_types})"
    )
    assert "proc" not in r.signals.own_building_types
    # tent was queued (intended to enable e1) — confirms the policy
    # actually committed to the wrong build path.
    assert "tent" in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_deploy_build_no_harvest_loses(level, seed):
    """Full chain (deploy → powr → proc) but NO harvest order must
    LOSE — the spawned harv sits idle; cash after chain is $200; bar
    cannot be cleared. This is the discriminating no-cheat case for
    'built the right things but didn't commit to harvesting'."""
    c, r = _run(level, _deploy_build_no_harvest_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} deploy-build-no-harvest must LOSE; got "
        f"{r.outcome} (ev={_ev(r)}, "
        f"types={r.signals.own_building_types})"
    )
    # The chain DID complete (proc exists, harv spawned) — the
    # discriminator is that no harvest commit means EV stays at the
    # $200 cash leftover, far below any tier's bar.
    assert "proc" in r.signals.own_building_types, (
        f"chain expected to complete; types={r.signals.own_building_types}"
    )
    assert _ev(r) < 800, f"no-harvest EV must stay below easy bar (800), got {_ev(r)}"


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared
    spawn_point groups (NORTH / SOUTH) so a memorised opening can't
    generalise."""
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


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, _intended_factory(), seed=2)
    b = run_level(c, _intended_factory(), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b)), (
        f"determinism: {(a.outcome, a.turns, _ev(a))} vs "
        f"{(b.outcome, b.turns, _ev(b))}"
    )
