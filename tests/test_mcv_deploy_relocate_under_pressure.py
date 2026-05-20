"""mcv-deploy-relocate-under-pressure scenario family, full loop on Rust.

Pack tests business-continuity relocation under attack: the agent's
original western Construction Yard (fact) is being charged by a
committed rusher band coming down the central lane, and the agent's
fresh MCV is parked dead-centre in that same lane. The correct play
is to MOVE the MCV out of the lane (to a safe north or south
shoulder) and then DEPLOY it — converting the MCV into a new fact
at a safe relocation site. The original may or may not survive;
the win bar only requires at least one fact alive AT a safe
relocation region, so the agent keeps operating even after the
original falls.

The win predicate makes the relocation load-bearing:

* `building_in_region:{type:fact, …safe-region…, count:1}` ⇒ the
  ONLY way to satisfy this without dying-in-place is to deploy the
  fresh MCV at a safe shoulder. Deploying in-place puts the new
  fact in the rusher lane and it falls too. Defending the original
  fails the medium/hard rusher tempo (original razes before the
  episode ends).
* `own_units_gte:1` ⇒ a pre-placed garrison rifle at each safe
  shoulder satisfies the SLA after deploy consumes the MCV (deploy
  removes the MCV unit; without the garrison the agent would have
  0 units post-deploy and fail the SLA).
* `units_lost_lte:2` (hard) ⇒ the hard attrition cap — losing
  more than 2 own units (the MCV is consumed by deploy and does
  NOT count as a loss) is a fail clause, so over-attrition cannot
  win.
* `within_ticks:5400` paired with `after_ticks:5401` ⇒ a non-
  finisher is a real reachable timeout LOSS in interrupt mode
  (60 turns × ≤90 ticks/step), never a draw. (In practice, every
  failure mode triggers the `not building_count_gte:fact:1` clause
  much earlier when the rusher razes the last fact.)

These tests prove with deterministic scripted policies (no model,
no network) that:

* the intended `move-out-of-lane, then deploy` policy WINS every
  level + every hard seed (1..4) regardless of N vs S shoulder
  choice;
* stall (MCV never moves, MCV never deploys) → original is razed
  → 0 facts → LOSS every level + seed;
* deploy-in-place (deploy without moving) → new fact in rusher
  lane → both facts razed → LOSS every level + seed;
* defend-only (mover the garrison toward the original base, no
  deploy) → still no relocated fact → original is razed → LOSS;
* the hard tier defines ≥2 spawn_point groups so the MCV's start
  column (and the matching safe shoulders) varies by seed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mcv-deploy-relocate-under-pressure.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ─────────────────────────────────────────────────


def _mcv(rs):
    units = rs.get("units_summary") or []
    return next(
        (u for u in units if str(u.get("type", "")).lower() == "mcv"),
        None,
    )


def stall(rs, C):
    """Observe-only — MCV never moves, never deploys. Original fact
    is razed → 0 facts → loss."""
    return [C.observe()]


def make_deploy_in_place():
    """Deploy the MCV WITHOUT moving it out of the rusher lane.
    The new fact lands at (60,20)-ish (or (40,20)-ish on hard seed
    1/3) — still in the lane — and is razed by the rusher as it
    passes through. Plus the original also razes. Both facts gone
    → loss."""

    def policy(rs, C):
        mcv = _mcv(rs)
        if mcv is None:
            return [C.observe()]
        return [C.deploy([str(mcv["id"])])]

    return policy


def make_intended_north():
    """Intended relocate-north policy: move the MCV up to y≤8 (out
    of the y=20 rusher lane), then deploy at the north shoulder.
    The new fact lands inside the safe NE region — satisfies
    building_in_region:{type:fact, …north shoulder…, count:1}. The
    pre-placed garrison rifle at the same shoulder satisfies
    own_units_gte:1 after deploy."""

    def policy(rs, C):
        mcv = _mcv(rs)
        if mcv is None:
            return [C.observe()]
        mid = str(mcv["id"])
        cx, cy = mcv["cell_x"], mcv["cell_y"]
        if cy > 9:
            return [C.move_units([mid], target_x=cx, target_y=6)]
        return [C.deploy([mid])]

    return policy


def make_intended_south():
    """Symmetric to intended_north: relocate-south to y≥31, deploy
    at the south shoulder. Tests that EITHER shoulder is a valid
    win (both safe regions are in the any_of)."""

    def policy(rs, C):
        mcv = _mcv(rs)
        if mcv is None:
            return [C.observe()]
        mid = str(mcv["id"])
        cx, cy = mcv["cell_x"], mcv["cell_y"]
        if cy < 31:
            return [C.move_units([mid], target_x=cx, target_y=34)]
        return [C.deploy([mid])]

    return policy


def defend_only(rs, C):
    """Defend-only: march the garrison rifles toward the original
    base to 'defend' it, never touch the MCV. The medium/hard
    rusher tempo razes the original before defenders can blunt it
    AND no relocated fact ever stands up. On hard the marched
    garrison also bleeds attrition past the cap. LOSS."""
    units = rs.get("units_summary") or []
    cmds = []
    for u in units:
        if str(u.get("type", "")).lower() in ("e1", "e3"):
            cmds.append(
                C.move_units(
                    [str(u["id"])], target_x=14, target_y=20
                )
            )
    if not cmds:
        cmds.append(C.observe())
    return cmds


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_rusher_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "mcv-deploy-relocate-under-pressure"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors.
    anchors = pack.meta.benchmark_anchor
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("ScienceWorld" in a for a in anchors), anchors
    assert any("MicroRTS" in a for a in anchors), anchors
    assert any(
        "business continuity" in a.lower() or "disaster recovery" in a.lower()
        for a in anchors
    ), anchors
    # Real-world / robotics meta lines required by spec.
    assert "relocat" in pack.meta.real_world_meaning.lower()
    assert "backup base" in pack.meta.robotics_analogue.lower()
    # Rusher bot must be wired through to the engine for every level.
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        enemy = c.scenario.enemy
        bot = (
            getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        )
        assert (str(bot).lower() == "rusher"), (lvl, bot)
        # Pack-wide 0-cash constraint: relocate, don't build around it.
        assert c.starting_cash == 0


def test_pack_includes_deploy_tool_for_every_level():
    """The deploy order is the load-bearing primitive — without it
    the agent cannot convert the fresh MCV into a fact."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tools = set(getattr(c.scenario, "tools", None) or [])
        assert "deploy" in tools, (lvl, tools)
        assert "move_units" in tools, (lvl, tools)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail must be
    strictly below the tick reachable at max_turns
    (≤90 ticks/step in interrupt mode)."""
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


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point
    groups so the MCV's start column (and the matching safe
    shoulders) varies by seed."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point
        for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_hard_attrition_cap_present_in_win_and_fail():
    """The hard `units_lost_lte:2` cap must be present in win AND
    fail so over-attrition cannot win and also explicitly loses."""
    c = compile_level(load_pack(PACK), "hard")
    win = c.win_condition.model_dump(exclude_none=True)
    fail = c.fail_condition.model_dump(exclude_none=True)
    win_cap = next(
        (
            clause["units_lost_lte"]
            for clause in win.get("all_of", [])
            if "units_lost_lte" in clause
        ),
        None,
    )
    assert win_cap == 2, win
    has_fail_cap = any(
        (clause.get("not") or {}).get("units_lost_lte") == 2
        for clause in fail.get("any_of", []) or []
    )
    assert has_fail_cap, fail


# ── solvency: intended WINS every level + every seed ──────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("intended_north", make_intended_north),
        ("intended_south", make_intended_south),
    ],
)
def test_intended_relocate_wins_every_level_and_seed(
    level, policy_name, policy_factory
):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "win", (
            f"{level} seed{seed} {policy_name}: relocate-then-deploy "
            f"must WIN; got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── no-cheat: every lazy / single-axis policy LOSES (not draws) ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize(
    "policy_name,policy_factory",
    [
        ("stall", lambda: stall),
        ("deploy_in_place", make_deploy_in_place),
        ("defend_only", lambda: defend_only),
    ],
)
def test_lazy_and_single_axis_policies_lose_every_level_and_seed(
    level, policy_name, policy_factory
):
    """Stall (MCV never deploys, original razed), deploy-in-place
    (fresh fact in rusher lane, both razed), defend-only (no
    relocated fact, original razed) must ALL LOSE on every level
    + every seed — no draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, policy_factory(), seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} {policy_name}: must LOSE (real fail, "
            f"not a draw); got {r.outcome} (tick={r.signals.game_tick}, "
            f"lost={r.signals.units_lost}, "
            f"buildings={r.signals.own_buildings})"
        )


# ── determinism ───────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, make_intended_north(), seed=3)
    b = run_level(c, make_intended_north(), seed=3)
    assert (a.outcome, a.turns, a.signals.units_lost) == (
        b.outcome,
        b.turns,
        b.signals.units_lost,
    ), "same seed must be deterministic"
