"""No-cheat + solvency proof for `build-rally-point-management` (Wave-8
ACTION: production logistics — set the production building's rally
point to a FORWARD staging area so freshly-built units arrive on the
front line in time to engage instead of piling up at the base default).

The pack tests the ACTION of calling `set_rally_point(tent, forward)`
ONCE at the opening so every subsequent freshly-built unit auto-walks
to the forward staging area (62,20). The minimal tool palette
(observe / build / set_rally_point / stop) means the rally is the
SOLE forward-projection mechanism — no `move_units` / `attack_move`,
so the agent cannot hand-route each batch.

For every level + every hard seed (1-4):
  * the INTENDED (rally-forward + build) policy WINS — units arrive
    in the forward disc by ~tick 903, kill the enemy `barr` at
    (62,20) by ~tick 1533–1800, both clauses satisfied inside the
    `within_ticks` window;
  * STALL (only `observe`) LOSES — never builds anything, never
    rallies → deadline bites as a real timeout LOSS;
  * NO-RALLY (only `build('e1')`, never sets rally) LOSES — units
    idle at the tent exit (23–26, 21–25), never enter the forward
    region, never kill anything → deadline bites;
  * NEAR-BASE-RALLY (sets rally to (28,22), still near the base)
    LOSES — units cluster just past the tent, never travel ~38
    cells to (62,20), never kill the forward barr → deadline bites.

The 3 lazy plays + 1 intended × 3 levels × 4 seeds covers the
no-defect / no-cheat bar from CLAUDE.md. Validation is scripted (no
model / network needed).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-rally-point-management.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Forward staging area (centreline) — fixed across all tiers and both
# hard spawn groups; the agent must compute it from the prose, not
# infer it from a moving tent position.
FORWARD_X, FORWARD_Y = 62, 20


# ───────────────────────── helpers ────────────────────────────────────────


def _tent_id(rs) -> str | None:
    """Read the tent's actor id out of the raw obs (the public
    `own_buildings` render only carries type/cell_x/cell_y; the id we
    need to address `set_rally_point` lives in `_raw.own_buildings`)."""
    raw = rs.get("_raw") or {}
    for b in (raw.get("own_buildings") or []):
        if b.get("type") == "tent":
            return str(b.get("id"))
    return None


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Observe-only: never builds anything → both `units_in_region_gte`
    and `units_killed_gte` clauses fail → deadline LOSS."""
    return [Command.observe()]


def _build_no_rally(rs, Command):
    """Build infantry forever, NEVER call set_rally_point. Default
    rally = tent's exit cell, so produced e1s idle at (23–26, 21–25)
    — none ever enter the (62,20) r=5 disc, no `barr` ever engaged,
    `units_killed` stays 0 → deadline LOSS."""
    cmds = []
    prod = rs.get("production") or []
    # Cap the build queue so the policy is deterministic.
    if prod.count("e1") < 3:
        cmds.append(Command.build("e1"))
    return cmds or [Command.observe()]


class _RallyState:
    """Per-episode one-shot latch — set_rally_point only needs to fire
    ONCE; subsequent units inherit the rally."""

    rallied: bool = False


def _intended_rally_forward(rs, Command):
    """Intended ACTION: call set_rally_point on the tent with target
    (62,20) on the FIRST observation, then keep queuing e1. Units
    auto-walk to the forward zone; cluster shreds the enemy `barr`.
    WINS every tier × every seed inside the deadline."""
    tid = _tent_id(rs)
    if tid is None:
        return [Command.observe()]
    cmds = []
    if not _RallyState.rallied:
        cmds.append(
            Command.set_rally_point([tid], target_x=FORWARD_X, target_y=FORWARD_Y)
        )
        _RallyState.rallied = True
    prod = rs.get("production") or []
    if prod.count("e1") < 3:
        cmds.append(Command.build("e1"))
    return cmds or [Command.observe()]


class _NearState:
    rallied: bool = False


def _near_base_rally(rs, Command):
    """Sets the rally to a NEAR-BASE cell (28,22) — units leave the
    tent but cluster ~4 cells east, ~34 cells short of the forward
    zone. Never enter (62,20) r=5, never kill the forward `barr` →
    deadline LOSS. Tests that the discriminator is NOT 'any rally
    call' but specifically the FORWARD rally call."""
    tid = _tent_id(rs)
    if tid is None:
        return [Command.observe()]
    cmds = []
    if not _NearState.rallied:
        cmds.append(Command.set_rally_point([tid], target_x=28, target_y=22))
        _NearState.rallied = True
    prod = rs.get("production") or []
    if prod.count("e1") < 3:
        cmds.append(Command.build("e1"))
    return cmds or [Command.observe()]


def _reset_state():
    _RallyState.rallied = False
    _NearState.rallied = False


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_minimal_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-rally-point-management"
    assert pack.meta.capability == "action"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Minimal palette: rally point is the SOLE forward-projection
    # mechanism. NO move_units / attack_move so the agent cannot
    # route produced units by hand.
    for t in ("observe", "build", "set_rally_point", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"
    for forbidden in ("move_units", "attack_move", "attack_unit", "harvest"):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in the rally-only palette"
        )


def test_benchmark_anchor_lists_sc2_rally_and_logistics():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "sc2" in blob and "rally" in blob, anchors
    assert "production logistics" in blob, anchors
    assert "warehouse" in blob and "sla" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: `after_ticks` in
    fail_condition must be reachable within `max_turns` (engine tick
    ≈ 93 + 90·(max_turns − 1) per CLAUDE.md)."""
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
    wt = next(
        int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x
    )
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt+1})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_combines_units_in_region_and_kill(level):
    """Structural: win must include `units_in_region_gte` at the
    forward zone (62,20) AND `units_killed_gte` — testing the
    rally-arrival + engagement combo, not either one alone."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    region = next(
        (x["units_in_region_gte"] for x in all_of if "units_in_region_gte" in x),
        None,
    )
    assert region is not None, f"{level}: win must include units_in_region_gte"
    assert int(region["x"]) == FORWARD_X
    assert int(region["y"]) == FORWARD_Y
    assert int(region["n"]) >= 3, f"{level}: region n must be ≥3 (got {region})"
    killed = next(
        (int(x["units_killed_gte"]) for x in all_of if "units_killed_gte" in x),
        None,
    )
    assert killed is not None and killed >= 1, (
        f"{level}: win must include units_killed_gte ≥1 (got {killed})"
    )


def test_hard_has_two_spawn_groups_for_base():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups (NORTH y≈14 / SOUTH y≈26) so
    seed round-robin varies the pre-placed fact/powr/tent and a
    memorised "rally from y=22" opening cannot generalise."""
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
def test_intended_rally_forward_wins(level, seed):
    """Intended set_rally_point(tent, 62, 20) + queue e1 WINS every
    level × every seed inside the within_ticks window. Units arrive
    in the forward (62,20) r=5 disc by ~tick 903, kill the enemy
    `barr` by ~tick 1533–1800, both clauses satisfied."""
    _reset_state()
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _intended_rally_forward, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended rally-forward should WIN, "
        f"got {r.outcome}; tick={r.signals.game_tick} "
        f"killed={r.signals.units_killed}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds, never
    rallies → both win clauses fail → reachable timeout LOSS (not a
    draw — the persistent enemy `fact` at (140,20) keeps engine
    auto-done gated)."""
    _reset_state()
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick} killed={r.signals.units_killed})"
    )
    assert r.signals.units_killed == 0


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_no_rally_loses(level, seed):
    """Build infantry but NEVER set the rally — produced units idle at
    the tent's exit cells (23–26, 21–25), never enter (62,20) r=5,
    never engage the forward `barr`. Both win clauses fail → LOSS.
    This is the headline discriminator: production without forward
    rally is exactly the warehouse-SLA failure mode the pack tests."""
    _reset_state()
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _build_no_rally, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} build-no-rally must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick} killed={r.signals.units_killed})"
    )
    assert r.signals.units_killed == 0, (
        f"{level} seed{seed} no-rally policy killed {r.signals.units_killed} "
        "— units must NOT reach the forward barr without an explicit rally"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_near_base_rally_also_loses(level, seed):
    """Sets the rally to a NEAR-BASE cell (28,22) — units leave the
    tent but cluster ~4 cells east, ~34 cells short of the forward
    zone (62,20). Never enter the disc, never kill the forward
    `barr` → LOSS. The discriminator is FORWARD rally, not 'any
    rally call'."""
    _reset_state()
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _near_base_rally, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} near-base-rally must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick} killed={r.signals.units_killed})"
    )
    assert r.signals.units_killed == 0


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared spawn_point
    groups (NORTH (26,14) / SOUTH (26,26) inert spawn-witness e1
    cells) so a memorised opening cannot generalise. The witness e1
    per group is the only thing in `units_summary` at reset (the
    pre-placed buildings are not units), so it carries the spawn
    variation."""
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
    _reset_state()
    a = run_level(c, _intended_rally_forward, seed=2)
    _reset_state()
    b = run_level(c, _intended_rally_forward, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
