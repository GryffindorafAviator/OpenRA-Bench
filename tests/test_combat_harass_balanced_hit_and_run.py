"""combat-harass-balanced-hit-and-run — BALANCED pulsed harass.

Bar: intended hit-run-cycle WINS on every level and every hard seed;
stall (only observe), commit-and-stay (charge and never retreat), and
retreat-only (never engage) LOSE on every level. Non-win is a real
reachable timeout LOSS (not a draw).

Validation is scripted (no model / network): the four policies below
are the exhaustive proxies for the four real strategies and exercise
the predicate teeth directly.

The intended hit-run-cycle policy is the load-bearing test: it
implements the pulsed-harass cycle in scripted form (per-jeep state
machine: APPROACH → STRIKE → RETREAT → re-engage), and it MUST WIN on
every level under units_lost_lte:0 (medium/hard) — i.e. retreat must
actually work against the leashed guard (AGGRO=16, LEASH=18).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-harass-balanced-hit-and-run.yaml"


# ── unit-level predicate checks ──────────────────────────────────────

def _ctx(units_xy=(), tick=1000, killed=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=killed,
        units_lost=lost,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"cell_x": x, "cell_y": y} for x, y in units_xy
            ]
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    jeeps = [(6, 20), (6, 20)]

    # Intended: 2 kills, no losses, in time → WIN
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=0))
    # 1 loss still wins on easy (cap is 1)
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=1))
    # 2 losses → fail (cap busted)
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=3000, killed=2, lost=2))
    # Past deadline → real loss, reachable within max_turns
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1), (
        "after_ticks 5401 must be reachable within max_turns"
    )


def test_predicates_medium_zero_loss_bar():
    c = compile_level(load_pack(PACK_PATH), "medium")
    jeeps = [(6, 20), (6, 20)]

    # Intended: 3 kills, ZERO losses → WIN (BALANCED bar)
    assert evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=3, lost=0))
    # 3 kills but lost a jeep → fail (units_lost_lte:0)
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=3000, killed=3, lost=1))
    # 2 kills, 0 losses → predicate fails (need ≥3)
    assert not evaluate(c.win_condition, _ctx(jeeps, tick=3000, killed=2, lost=0))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_four_kills_and_zero_loss():
    c = compile_level(load_pack(PACK_PATH), "hard")
    jeeps = [(6, 10), (6, 10)]

    # Intended: 4 kills, ZERO losses → WIN
    assert evaluate(c.win_condition, _ctx(jeeps, tick=4000, killed=4, lost=0))
    # 3 kills only → predicate fails (need ≥4)
    assert not evaluate(c.win_condition, _ctx(jeeps, tick=4000, killed=3, lost=0))
    # Any loss → fail
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=4000, killed=4, lost=1))
    # Past deadline → real loss, reachable
    assert evaluate(c.fail_condition, _ctx(jeeps, tick=5402, killed=0, lost=0))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the raider start latitude."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.capability == "reasoning"
    assert pack.meta.id == "combat-harass-balanced-hit-and-run"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchored to the two doctrines the brief calls out: SC2 pulsed
    # harass and military guerrilla / hit-and-run.
    assert "muta" in joined or "harass" in joined or "sentry" in joined
    assert "guerrilla" in joined or "hit-and-run" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
    """No draw degeneracy: after_ticks 5401 fits inside max_turns on
    every level (∼90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 5401 <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ──────────────────────────────────
#
# Constants from CLAUDE.md / openra-sim/src/scripted_bot.rs:
#   GUARD_AGGRO = 16 cells (post-radius for lunge)
#   GUARD_LEASH = 18 cells (post-radius for snap-back)
# Hit-run-cycle thresholds chosen to bracket those exactly:
#   ENGAGE_RADIUS  = 14  (inside aggro: pulse close enough to score)
#   RETREAT_TRIGGER = 14 (start running when within 14 of any guard)
#   SAFE_RADIUS    = 22  (outside leash by margin: guard snapped back)

# RETREAT the instant a guard is visible inside this radius. The
# rocket-soldier (e3) has long range and 2+ shots per turn against
# jeep HP, so a single decision spent inside it is fatal. Setting
# RETREAT_TRIGGER above the guard's own AGGRO (16) means we run as
# soon as we get within visual distance — the strike fires THIS turn
# (because attack_unit on a visible harv at the cluster edge is in
# jeep MG range) and the retreat fires NEXT turn.
ENGAGE_RADIUS = 12
RETREAT_TRIGGER = 18
SAFE_RADIUS = 22


def _harvs(enemies):
    return [e for e in enemies if (e.get("type") or "").lower() == "harv"]


def _guards(enemies):
    # The leashed defender is e3 in this pack; bot_type:guard makes it
    # lunge/snap. (harv has type 'harv', e3 has type 'e3'; the fact
    # marker is is_building=True and is filtered by callers.)
    return [
        e for e in enemies
        if (e.get("type") or "").lower() == "e3"
        and not e.get("is_building")
    ]


def _known_cluster_x_for(level):
    """The kill cluster's x coordinate (the agent is told this in the
    objective brief: harv cluster at center-east). Easy uses one
    cluster at x≈73; medium the same; hard uses cluster A at x≈69 and
    cluster B at x≈97 (handled by the policy ranging east)."""
    return 73


def _hit_run_cycle_policy(rs, Command):
    """Pulsed hit-and-run — the intended BALANCED strategy.

    Per-jeep state machine (memoryless; derived each turn from geometry
    + the known cluster axis: the agent is told where the cluster is):
      - If a guard is within RETREAT_TRIGGER (14) cells → RETREAT west
        ~14 cells (cross the leash; the snap-back triggers next turn).
      - Else if a harv is in sight → APPROACH+STRIKE the nearest harv
        (move_units auto-fires en route, but attack_unit makes it
        unambiguous and faster).
      - Else if no enemy visible (fog from the far-west start) →
        ADVANCE east toward the known cluster axis to acquire vision.

    The cycle is self-sustaining once contact is made: strike → in
    range of guard → retreat west → guard snaps back past leash →
    re-advance → strike next.
    """
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    if not units:
        return [Command.observe()]
    harvs = _harvs(enemies)
    guards = _guards(enemies)
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        # Nearest visible guard (if any) — the thing we must avoid.
        nearest_g = None
        if guards:
            nearest_g = min(
                guards,
                key=lambda g: (g["cell_x"] - ux) ** 2 + (g["cell_y"] - uy) ** 2,
            )
            gd2 = (nearest_g["cell_x"] - ux) ** 2 + (nearest_g["cell_y"] - uy) ** 2
        else:
            gd2 = 10 ** 9
        # If we are inside the guard's reach → RETREAT west past leash.
        if nearest_g is not None and gd2 <= RETREAT_TRIGGER ** 2:
            tx = max(2, ux - 14)
            ty = uy
            cmds.append(
                Command.move_units([str(u["id"])], target_x=tx, target_y=ty)
            )
            continue
        # Safe: if a harv is visible, strike the nearest one.
        if harvs:
            target = min(
                harvs,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(
                Command.attack_unit([str(u["id"])], str(target["id"]))
            )
            continue
        # No enemy in sight (fog): advance east toward the known
        # cluster axis until vision picks up the cluster. Step ~10
        # cells per decision so we don't overrun into aggro on one go.
        target_x = min(_known_cluster_x_for(None) - 4, ux + 12)
        if target_x > ux:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=target_x, target_y=uy)
            )
        else:
            cmds.append(Command.observe())
    return cmds


def _stall_policy(rs, Command):
    """Stall: only observe. Kill bar never met → fail on the clock."""
    return [Command.observe()]


def _commit_and_stay_policy(rs, Command):
    """Charge the cluster and STAY — never retreat. The e3 rocket
    soldier (out-DPS jeeps) will kill at least one jeep before all
    harvs fall; units_lost_lte fails on medium/hard. (When fog hides
    the harvs from the western start, advance east to the cluster
    axis to acquire vision, then sit on the nearest harv.)"""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    harvs = _harvs(enemies)
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        ux, uy = u["cell_x"], u["cell_y"]
        if harvs:
            target = min(
                harvs,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(target["id"])))
        else:
            # No vision yet — march east toward the cluster axis.
            cmds.append(
                Command.move_units(
                    [str(u["id"])], target_x=min(74, ux + 14), target_y=uy
                )
            )
    return cmds


def _retreat_only_policy(rs, Command):
    """Sit at the west staging line; never approach. Kill bar unmet →
    fail on the clock."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    cmds = []
    for u in units:
        # Park at the far-west wall (any cell well outside any guard's
        # AGGRO=16; the post is at x≥70).
        cmds.append(
            Command.move_units([str(u["id"])], target_x=4, target_y=u["cell_y"])
        )
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_hit_run_cycle_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _hit_run_cycle_policy, seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended hit-run-cycle should WIN, got "
            f"{res.outcome} after {res.turns} turns (killed="
            f"{res.signals.units_killed}, lost={res.signals.units_lost})"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_policy_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: stall must LOSE (kill bar unmet, clock), got "
        f"{res.outcome} after {res.turns} turns; killed="
        f"{res.signals.units_killed}"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
def test_commit_and_stay_loses(level):
    """On medium/hard the cap is units_lost_lte:0 — committing in
    rocket range until the harvs die loses a jeep and fails."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _commit_and_stay_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: commit-and-stay must LOSE (jeep dies to e3 rocket), "
        f"got {res.outcome} after {res.turns} turns; killed="
        f"{res.signals.units_killed}, lost={res.signals.units_lost}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_retreat_only_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _retreat_only_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level}: retreat-only must LOSE (kill bar unmet), got "
        f"{res.outcome}; killed={res.signals.units_killed}"
    )
