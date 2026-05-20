"""mfb-base-1-defend-base-2-build — secure-expand under harass.

The bar: the intended SPLIT policy (defenders engage the patrol;
fresh MCV drives east and deploys inside the target region) WINS on
every level and every hard seed. Three failure modes all LOSE on
every level + seed:
  • STALL  — only `observe()` issued; defenders are HoldFire so the
              patrol razes base #1's `fact` unopposed → fail by
              `not has_building:fact`.
  • MCV-ONLY-EAST — push the MCV east, never engage the patrol;
              same razed-fact outcome.
  • ALL-DEFEND   — engage the patrol, never deploy the MCV; the
              second `fact` never appears → clock kills the run
              (`after_ticks` fail).

Validation is scripted (no model / network) — uses
`openra_bench.eval_core.run_level`.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "mfb-base-1-defend-base-2-build.yaml"


# ── unit-level predicate checks ──────────────────────────────────────


def _ctx(own_buildings=(), tick=1000, n_units=3):
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
        {"id": str(i), "type": "e1", "cell_x": 13, "cell_y": 30 + i, "owner": "agent"}
        for i in range(n_units)
    ]
    return WinContext(signals=sig, render_state={"units_summary": units})


def test_predicates_easy_win_requires_both_fact_in_east_and_base_1_holds():
    c = compile_level(load_pack(PACK_PATH), "easy")
    base1 = ("fact", 15, 30)
    east = ("fact", 130, 30)  # at east target center
    # WIN: 2 facts, second one inside target region, units alive, in time.
    assert evaluate(c.win_condition, _ctx([base1, east], tick=4000))
    # FAIL clauses don't fire on the winning state.
    assert not evaluate(c.fail_condition, _ctx([base1, east], tick=4000))

    # FAIL: only 1 fact (never deployed) → win not satisfied.
    assert not evaluate(c.win_condition, _ctx([base1], tick=4000))
    # FAIL: 2nd fact at center (out of region) → win not satisfied.
    middle = ("fact", 80, 30)
    assert not evaluate(c.win_condition, _ctx([base1, middle], tick=4000))
    # If base #1 is lost AND only the east fact remains: only 1 fact
    # total → building_count_gte:2 fails. The win predicate also
    # requires ≥2 facts, so a "deploy east but lose base #1" play
    # is NOT a win either. Good — the agent must keep base #1's fact
    # alive AND land a 2nd fact in the east region.
    assert not evaluate(c.win_condition, _ctx([east], tick=4000))
    # Past deadline ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([base1, east], tick=6302))
    # Lost the lone fact ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([], tick=4000))
    # Lost every unit ⇒ fail.
    assert evaluate(c.fail_condition, _ctx([base1], tick=4000, n_units=0))
    # Deadline reachable inside max_turns (∼90 ticks/turn).
    assert 6301 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_medium_radius_8_boundary():
    c = compile_level(load_pack(PACK_PATH), "medium")
    base1 = ("fact", 15, 30)
    inside = ("fact", 132, 28)  # dist² = 4+4 = 8 ≤ 64 ✓
    assert evaluate(c.win_condition, _ctx([base1, inside], tick=3000))
    outside = ("fact", 139, 30)  # dist 9 > 8
    assert not evaluate(c.win_condition, _ctx([base1, outside], tick=3000))
    assert 4501 <= 93 + 90 * (c.max_turns - 1)


def test_predicates_hard_either_corner_wins():
    c = compile_level(load_pack(PACK_PATH), "hard")
    base1 = ("fact", 15, 30)
    ne = ("fact", 130, 15)
    se = ("fact", 130, 45)
    assert evaluate(c.win_condition, _ctx([base1, ne], tick=3000))
    assert evaluate(c.win_condition, _ctx([base1, se], tick=3000))
    # FAIL: 2nd fact at mid (neither NE nor SE region).
    mid = ("fact", 130, 30)  # dist 15 from each candidate center > 8
    assert not evaluate(c.win_condition, _ctx([base1, mid], tick=3000))
    # FAIL: deploy NEXT TO base #1 (still outside both east regions).
    near_base = ("fact", 22, 30)
    assert not evaluate(c.win_condition, _ctx([base1, near_base], tick=3000))
    assert evaluate(c.fail_condition, _ctx([base1, ne], tick=5402))
    assert 5401 <= 93 + 90 * (c.max_turns - 1)


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the fresh MCV's staging latitude
    (NE vs SE) and the chosen target region flips accordingly."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_pack_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mfb-base-1-defend-base-2-build"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    joined = " ".join(anchors).lower()
    assert "sc2" in joined or "secure-expand" in joined
    assert "microrts" in joined or "bcp" in joined or "site-b" in joined
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None
        # tools must include deploy + move_units + attack_unit + observe.
        tools = set(c.scenario.tools or [])
        for t in ("observe", "deploy", "move_units", "attack_unit",
                  "attack_move", "stop"):
            assert t in tools, f"{lvl}: missing tool {t}"


def test_timeout_loss_reachable_every_level():
    """No draw degeneracy: the level's after_ticks fail must fit
    inside max_turns on every level."""
    pack = load_pack(PACK_PATH)
    bars = {"easy": 6301, "medium": 4501, "hard": 5401}
    for lvl, bar in bars.items():
        c = compile_level(pack, lvl)
        assert bar <= 93 + 90 * (c.max_turns - 1), lvl


# ── engine-driven scripted policies ──────────────────────────────────


def _find_mcv(rs):
    for u in rs.get("units_summary", []) or []:
        if str(u.get("type", "")).lower() == "mcv":
            return u
    return None


def _agent_riflemen(rs):
    return [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == "e1"
    ]


def _west_enemies(rs):
    """Enemy non-buildings on the western half (the home-side patrol)."""
    out = []
    for e in (rs.get("enemy_summary", []) or []):
        if e.get("is_building"):
            continue
        if e.get("cell_x", 999) < 80:
            out.append(e)
    return out


def _nearest_east_target(mcv, level):
    if level != "hard":
        return (130, 30)
    if mcv is None:
        return (130, 15)
    dn = (mcv["cell_x"] - 130) ** 2 + (mcv["cell_y"] - 15) ** 2
    ds = (mcv["cell_x"] - 130) ** 2 + (mcv["cell_y"] - 45) ** 2
    return (130, 15) if dn <= ds else (130, 45)


def _intended_split_policy_for(level):
    """Two-track policy: defenders engage the western probe; MCV
    drives east and deploys inside the target region."""

    def _policy(rs, Command):
        cmds = []
        # Track A: defenders engage nearest western enemy.
        enemies = _west_enemies(rs)
        riflemen = _agent_riflemen(rs)
        if enemies and riflemen:
            for u in riflemen:
                ux, uy = u["cell_x"], u["cell_y"]
                target = min(
                    enemies,
                    key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
                )
                cmds.append(Command.attack_unit([str(u["id"])], str(target["id"])))
        # Track B: drive the MCV east and deploy in region.
        mcv = _find_mcv(rs)
        if mcv is not None:
            tx, ty = _nearest_east_target(mcv, level)
            dx = mcv["cell_x"] - tx
            dy = mcv["cell_y"] - ty
            if dx * dx + dy * dy <= 36:
                cmds.append(Command.deploy([str(mcv["id"])]))
            else:
                cmds.append(
                    Command.move_units([str(mcv["id"])], target_x=tx, target_y=ty)
                )
        if not cmds:
            return [Command.observe()]
        return cmds

    return _policy


def _stall_policy(rs, Command):
    return [Command.observe()]


def _mcv_only_east_policy_for(level):
    """Only push the MCV east; never engage the patrol. The HoldFire
    defenders sit; the patrol razes base #1's fact."""

    def _policy(rs, Command):
        mcv = _find_mcv(rs)
        if mcv is None:
            return [Command.observe()]
        tx, ty = _nearest_east_target(mcv, level)
        dx = mcv["cell_x"] - tx
        dy = mcv["cell_y"] - ty
        if dx * dx + dy * dy <= 36:
            return [Command.deploy([str(mcv["id"])])]
        return [Command.move_units([str(mcv["id"])], target_x=tx, target_y=ty)]

    return _policy


def _all_defend_policy(rs, Command):
    """Defenders engage; MCV never moves / never deploys. The second
    fact never appears → clock LOSS."""
    enemies = _west_enemies(rs)
    riflemen = _agent_riflemen(rs)
    cmds = []
    if enemies and riflemen:
        for u in riflemen:
            ux, uy = u["cell_x"], u["cell_y"]
            target = min(
                enemies,
                key=lambda e: (e["cell_x"] - ux) ** 2 + (e["cell_y"] - uy) ** 2,
            )
            cmds.append(Command.attack_unit([str(u["id"])], str(target["id"])))
    if not cmds:
        return [Command.observe()]
    return cmds


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_intended_split_wins(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _intended_split_policy_for(level), seed=s)
        assert res.outcome == "win", (
            f"{level} seed={s}: intended split should WIN, got "
            f"{res.outcome} after {res.turns} turns; "
            f"buildings={res.signals.own_buildings}, "
            f"lost={res.signals.units_lost}"
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
def test_mcv_only_east_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _mcv_only_east_policy_for(level), seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: MCV-only-east must LOSE (base #1 "
            f"razed under unopposed patrol), got {res.outcome} after "
            f"{res.turns} turns; buildings={res.signals.own_buildings}"
        )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_all_defend_loses(level):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    seeds = (1, 2, 3, 4) if level == "hard" else (1,)
    for s in seeds:
        res = run_level(c, _all_defend_policy, seed=s)
        assert res.outcome == "loss", (
            f"{level} seed={s}: all-defend must LOSE (2nd fact never "
            f"appears, clock kills the run), got {res.outcome} after "
            f"{res.turns} turns; buildings={res.signals.own_buildings}"
        )
