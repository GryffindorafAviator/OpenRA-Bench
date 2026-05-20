"""def-retreat-and-rebuild — strategic withdrawal + rebuild at depth.

Wave-8 REASONING pack. The agent operates a FORWARD base that is
being overrun by a heavy attack from the east; the forward base
CANNOT be held inside the tick budget. A parked MCV + powr sits at
a deeper SAFE zone in the west. The intended play: retreat (let
the forward base fall), DEPLOY the safe-zone MCV (creates a fact
inside the safe radius), then BUILD a refinery (`proc`) at the safe
zone so production stands up at depth.

The bar (validated scripted at seed 1, plus seeds 1..4 on hard):
  • STALL (observe only) — LOSS. Rusher band razes the forward
    base; the MCV is never deployed; no safe-region fact ever
    appears; the `not building_count_gte:{type:fact,n:1}` fail
    clause fires once the forward fact dies, OR the deadline
    expires with the safe-region win clauses unsatisfied.
  • HOLD-FORWARD (defenders engage; MCV never touched) — LOSS.
    Same forward-fact razing + no safe-region rebuild.
  • DEPLOY-ONLY (deploy the MCV but never build proc) — LOSS.
    Safe-region fact appears post-deploy but the safe-region
    proc clause is never satisfied → deadline expires.
  • INTENDED RETREAT-AND-REBUILD — WIN. Deploy MCV early →
    fact at safe radius; build('proc') + place_building inside
    safe radius → both win clauses fire ~tick 1000-1100.

Validation is scripted (no model / network) — uses
`openra_bench.eval_core.run_level`.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "def-retreat-and-rebuild.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(own_buildings=(), tick=1000, n_units=1):
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    units = [
        {"id": str(i), "type": "e1", "cell_x": 4, "cell_y": 20, "owner": "agent"}
        for i in range(n_units)
    ]
    return WinContext(signals=sig, render_state={"units_summary": units})


def test_predicates_easy_win_requires_both_fact_and_proc_in_safe_region():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # WINNING state: safe-region fact (deploy lands at ~(9,19)) + safe-
    # region proc (placed at (12,20)), unit alive, in time.
    safe_fact = ("fact", 9, 19)
    safe_proc = ("proc", 12, 20)
    assert evaluate(c.win_condition, _ctx([safe_fact, safe_proc], tick=2000))
    assert not evaluate(c.fail_condition, _ctx([safe_fact, safe_proc], tick=2000))

    # FAIL clauses on losing states.
    # Forward base still alive but no safe rebuild → win unsatisfied.
    fwd_fact = ("fact", 40, 20)
    fwd_proc = ("proc", 44, 20)
    assert not evaluate(c.win_condition, _ctx([fwd_fact, fwd_proc], tick=2000))
    # No buildings at all → fact-loss fail.
    assert evaluate(c.fail_condition, _ctx([], tick=2000))
    # Past deadline ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([safe_fact, safe_proc], tick=4502))
    # Deploy only (fact at safe, no proc) → win unsatisfied.
    assert not evaluate(c.win_condition, _ctx([safe_fact], tick=2000))
    # Proc-only at safe (no safe fact) → win unsatisfied.
    assert not evaluate(c.win_condition, _ctx([fwd_fact, safe_proc], tick=2000))
    # The fact alone (forward-base survivor) but no safe rebuild → loss.
    assert not evaluate(c.win_condition, _ctx([fwd_fact], tick=2000))
    # Deadline reachable inside max_turns (∼90 ticks/turn).
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_medium_radius_8_boundary():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Inside the r=8 safe region around (10,20).
    safe_fact = ("fact", 9, 19)
    inside_proc = ("proc", 14, 22)  # dx=4,dy=2 → 20 ≤ 64 ✓
    assert evaluate(c.win_condition, _ctx([safe_fact, inside_proc], tick=2000))
    # Outside the r=8 safe region — the proc is too far east.
    outside_proc = ("proc", 20, 20)  # dx=10 → 100 > 64
    assert not evaluate(c.win_condition, _ctx([safe_fact, outside_proc], tick=2000))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_either_safe_zone_wins():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # NORTH safe-region rebuild satisfies win.
    n_fact = ("fact", 9, 13)
    n_proc = ("proc", 12, 14)
    assert evaluate(c.win_condition, _ctx([n_fact, n_proc], tick=2000))
    # SOUTH safe-region rebuild satisfies win.
    s_fact = ("fact", 9, 25)
    s_proc = ("proc", 12, 26)
    assert evaluate(c.win_condition, _ctx([s_fact, s_proc], tick=2000))
    # Far-east mid-y placement misses BOTH safe regions (radius 8
    # around (10,14) and (10,26); a build at (40,20) is dx=30 from
    # BOTH centres → well outside).
    far_fact = ("fact", 40, 20)
    far_proc = ("proc", 44, 20)
    assert not evaluate(c.win_condition, _ctx([far_fact, far_proc], tick=2000))
    # Mixed latitude (north fact + south proc) — neither region
    # satisfies both clauses simultaneously → win unsatisfied.
    assert not evaluate(c.win_condition, _ctx([n_fact, s_proc], tick=2000))
    assert evaluate(c.fail_condition, _ctx([n_fact, n_proc], tick=4502))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the forward base + safe zone
    latitude (NORTH y=14 vs SOUTH y=26)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "def-retreat-and-rebuild"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    # Anchors must reference the strategic-withdrawal / CICERO retreat /
    # PlanBench replanning doctrine that motivates the pack.
    assert "withdrawal" in joined or "cicero" in joined or "retreat" in joined
    assert "planbench" in joined or "replanning" in joined or "continuity" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        # Tools required for the capability: observe + deploy + build +
        # place_building (the rebuild primitive) plus move/attack for
        # the hold-forward attempt to be expressible (and lose).
        tools = set(c.scenario.tools or [])
        for t in ("observe", "deploy", "build", "place_building",
                  "move_units", "attack_unit"):
            assert t in tools, f"{lvl}: missing tool {t}"


def test_timeout_loss_reachable_every_level():
    """No draw degeneracy: the after_ticks fail must fit inside
    max_turns on every level."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert 4501 <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ──────────────────────────────────


def _find_mcv(rs):
    for u in rs.get("units_summary", []) or []:
        if str(u.get("type", "")).lower() == "mcv":
            return u
    return None


def _stall_policy(rs, Command):
    return [Command.observe()]


def _hold_forward_policy(rs, Command):
    """Engage the rushers with the 2 forward defenders; never touch
    the safe MCV. The rush is too heavy → forward base razed AND no
    safe-zone rebuild → LOSS."""
    units = rs.get("units_summary", []) or []
    # Forward defenders are the e1s NEAR the forward base (x ≥ 40).
    defenders = [
        u for u in units
        if str(u.get("type", "")).lower() == "e1"
        and int(u.get("cell_x", 0)) >= 40
    ]
    enemies = [
        e for e in (rs.get("enemy_summary", []) or [])
        if not e.get("is_building") and int(e.get("cell_x", 0)) >= 50
    ]
    cmds = []
    for u in defenders:
        if enemies:
            ux, uy = u["cell_x"], u["cell_y"]
            target = min(
                enemies,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(target["id"])))
    return cmds or [Command.observe()]


def _deploy_only_policy(rs, Command):
    """Deploy the safe-zone MCV (satisfies the safe-region FACT
    clause) but never build the safe-region proc → win unsatisfied
    on the proc clause → after_ticks LOSS."""
    mcv = _find_mcv(rs)
    if mcv is not None:
        return [Command.deploy([str(mcv["id"])])]
    return [Command.observe()]


def _intended_retreat_and_rebuild_policy():
    """Deploy the safe-zone MCV (creates a fact inside the safe
    radius), then build('proc') + place_building('proc') inside the
    safe radius. The forward base is allowed to fall — the safe
    rebuild carries the win clauses by the deadline."""
    state = {"deployed": False, "safe_y": None}

    def policy(rs, Command):
        bldgs = rs.get("own_buildings", []) or []
        if not state["deployed"]:
            mcv = _find_mcv(rs)
            if mcv is not None:
                state["safe_y"] = int(mcv["cell_y"])
                state["deployed"] = True
                return [Command.deploy([str(mcv["id"])])]
        sy = state["safe_y"] or 20
        n_proc_in_safe = sum(
            1 for b in bldgs
            if b.get("type") == "proc"
            and (int(b.get("cell_x", 0)) - 10) ** 2
            + (int(b.get("cell_y", 0)) - sy) ** 2
            <= 64
        )
        prod = rs.get("production", []) or []
        n_in_q = sum(
            1 for p in prod
            if isinstance(p, dict) and p.get("item") == "proc"
        )
        cmds = []
        if n_proc_in_safe < 1:
            if n_in_q == 0:
                cmds.append(Command.build("proc"))
            # Place at (12, safe_y) — adjacent to the post-deploy
            # fact at ~(9, safe_y-1) AND inside the radius-8 safe
            # region around (10, safe_y).
            cmds.append(Command.place_building("proc", 12, sy))
        return cmds or [Command.observe()]

    return policy


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_retreat_and_rebuild_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_retreat_and_rebuild_policy(), seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended retreat-and-rebuild should "
            f"WIN, got {res.outcome} after {res.turns} turns "
            f"(tick={res.signals.game_tick}); "
            f"buildings={res.signals.own_buildings}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _stall_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: stall must LOSE, got {res.outcome} "
            f"after {res.turns} turns; "
            f"buildings={sorted(res.signals.own_building_types)}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_hold_forward_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _hold_forward_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: hold-forward must LOSE (forward base "
            f"razed regardless AND no safe-zone rebuild), got "
            f"{res.outcome} after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_deploy_only_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _deploy_only_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: deploy-only (no proc rebuild) must "
            f"LOSE, got {res.outcome} after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}"
        )
