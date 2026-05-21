"""No-cheat + solvency proof for `build-repair-priority-under-fire`
(Wave-10 REASONING: criticality-weighted repair triage).

Three of the agent's structures are under simultaneous attrition from
separate grenadier bands. The damage picture is deliberately misleading
and the agent's only job is the TRIAGE DECISION — which building(s) to
put the `repair` organ on:

  * proc (refinery)  — HIGH value, on a LETHAL trajectory: dies inside
    ~3 turns unrepaired; required by the win predicate.
  * pbox (pillbox)   — LOW value DECOY: pre-placed at health 30 so it
    LOOKS the most damaged, but it is Heavy-armoured (a light grenadier
    band barely scratches it), is not in the win predicate, and
    survives on its own. Repairing it is wasted effort.
  * weap (war factory) — MEDIUM: on easy NO band sits on the war
    factory directly, so a bridged proc keeps its band pinned and the
    war factory rides out untouched; on medium / hard a light band ALSO
    chews the war factory so it too needs repair.

The pack carries NO agent combat units and NO kill bar — the engine
movement fixes made pre-placed defenders resolve a clustered fire-fight
in one ~90-tick window with the buildings never scratched, collapsing
the old damage race. The triage is now a pure `repair`-target decision
against an unopposed band. The win requires the buildings to SURVIVE a
danger window (`after_ticks` floor + `within_ticks` ceiling), not
merely be alive on turn 1.

No-cheat bar (proven below for every level x seed 1-4):

  * STALL (only `observe`) -> proc never bridged -> refinery dies ~turn
    4 -> `not building_count_gte:{proc}` fail fires -> LOSS.
  * REPAIR-PBOX-FIRST (toggle repair on the most-damaged-LOOKING
    building, the pbox) -> pbox saved (it survives on its own anyway);
    the proc gets no repair and dies -> LOSS.
  * REPAIR-PROC-ONLY -> WIN on easy (no band sits on the war factory,
    so a bridged proc keeps its band pinned) but LOSS on medium / hard
    (the light war-factory band kills the unrepaired weap ->
    `not building_count_gte:{weap}` fail).
  * INTENDED triage (repair proc — easy; proc + weap — medium / hard)
    -> autorepair out-paces the grenade chip, the buildings ride the
    bands out, the survival window opens -> WIN.

Repair-everything also wins (it repairs the proc + weap) — that is
within the bar: the failure mode this pack discriminates is "repair the
worst-LOOKING building", not "be thorough".
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-repair-priority-under-fire.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _repair_types(rs, Command, types):
    """Toggle `repair` on every own building whose type is in `types`
    and is below full HP."""
    cmds = []
    for b in (rs.get("_raw", {}) or {}).get("own_buildings", []) or []:
        if b.get("type") in types and float(b.get("hp_pct", 1.0)) < 0.999:
            cmds.append(Command.repair([str(b["id"])]))
    return cmds or [Command.observe()]


def _stall(rs, Command):
    return [Command.observe()]


def _repair_pbox(rs, Command):
    """Wrong triage: repair the most-damaged-LOOKING building (pbox)."""
    return _repair_types(rs, Command, {"pbox"})


def _repair_proc(rs, Command):
    return _repair_types(rs, Command, {"proc"})


def _repair_proc_weap(rs, Command):
    """Intended triage on medium / hard: the two critical buildings."""
    return _repair_types(rs, Command, {"proc", "weap"})


def _repair_all(rs, Command):
    return _repair_types(rs, Command, {"proc", "weap", "pbox", "fix", "fact"})


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_repair_tool():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-repair-priority-under-fire"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # The triage is exclusively a `repair`-target decision.
    for t in ("observe", "repair"):
        assert t in tools, f"missing tool {t} in {tools}"
    for forbidden in ("build", "place_building", "move_units", "attack_unit"):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in the repair-triage palette"
        )


def test_benchmark_anchor_lists_triage():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "triage" in blob or "maintenance" in blob, anchors


def test_pbox_decoy_is_pre_damaged_every_level():
    """The pbox decoy is pre-placed already damaged (health: 30) so it
    LOOKS the most-damaged building — the bait for the wrong triage."""
    pack = load_pack(PACK)
    for level in LEVELS:
        c = compile_level(pack, level)
        pboxes = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "pbox"
        ]
        assert pboxes, f"{level}: no pbox decoy"
        for a in pboxes:
            assert a.health == 30, (
                f"{level}: pbox decoy must be pre-damaged (health 30), "
                f"got {a.health}"
            )


def test_no_agent_combat_units():
    """The pack carries NO agent combat units — the triage is a pure
    `repair`-target decision against an unopposed band."""
    pack = load_pack(PACK)
    buildings = {"proc", "weap", "pbox", "fix", "fact"}
    for level in LEVELS:
        c = compile_level(pack, level)
        agent_types = {
            a.type for a in c.scenario.actors if a.owner == "agent"
        }
        assert agent_types <= buildings, (
            f"{level}: agent must own only buildings, got {agent_types}"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: the `after_ticks`
    fail must be reachable inside `max_turns` and equal within_ticks+1."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fail = c.fail_condition.model_dump()["any_of"]
    after_ticks = int(fail[0]["after_ticks"])
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
def test_win_has_survival_window(level):
    """The win must require the buildings to SURVIVE a danger window —
    an `after_ticks` floor below the `within_ticks` ceiling — so the win
    cannot latch on turn 1 before any damage lands."""
    c = compile_level(load_pack(PACK), level)
    all_of = c.win_condition.model_dump().get("all_of", [])
    floor = next(int(x["after_ticks"]) for x in all_of if "after_ticks" in x)
    ceiling = next(
        int(x["within_ticks"]) for x in all_of if "within_ticks" in x
    )
    assert 0 < floor < ceiling, (
        f"{level}: win needs a survival window 0 < after_ticks "
        f"({floor}) < within_ticks ({ceiling})"
    )
    reachable = 93 + 90 * (c.max_turns - 1)
    assert floor < reachable, (
        f"{level}: win after_ticks {floor} unreachable within "
        f"{c.max_turns} turns (max tick {reachable})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_fail_trips_on_either_required_building(level):
    """Losing the proc OR the weap must be an immediate clean LOSS — no
    draw can leak through a building-death + auto-`done` race."""
    c = compile_level(load_pack(PACK), level)
    fail = c.fail_condition.model_dump()["any_of"]
    not_clauses = [x.get("not", {}) for x in fail if "not" in x]
    types = {
        c2.get("building_count_gte", {}).get("type")
        for c2 in not_clauses
    }
    assert "proc" in types, f"{level}: fail must trip on proc loss"
    assert "weap" in types, f"{level}: fail must trip on weap loss"


@pytest.mark.parametrize("level", LEVELS)
def test_win_requires_proc_and_weap(level):
    c = compile_level(load_pack(PACK), level)
    all_of = c.win_condition.model_dump().get("all_of", [])
    bc = {
        x["building_count_gte"]["type"]
        for x in all_of
        if "building_count_gte" in x
    }
    assert {"proc", "weap"} <= bc, f"{level}: win must require proc and weap"


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
def test_intended_triage_wins(level, seed):
    """Intended triage — repair proc (easy) / proc + weap (medium,
    hard) — WINS every level × every seed."""
    c, r = _run(level, _repair_proc_weap, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended triage should WIN, got {r.outcome} "
        f"(tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_repair_everything_wins(level, seed):
    """Repairing every damaged building also wins — it DOES repair the
    proc + weap. Thoroughness is not the failure mode under test."""
    c, r = _run(level, _repair_all, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: repair-all should WIN, got {r.outcome}"
    )


# ───────────────────────── no-cheat: lazy / wrong plays LOSE ───────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE — the proc band is never bridged; the refinery
    dies ~turn 4 and the `not building_count_gte:{proc}` fail fires."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_repair_pbox_first_loses(level, seed):
    """Repairing the most-damaged-LOOKING building (the low-value pbox
    decoy) must LOSE — the proc gets no repair and dies."""
    c, r = _run(level, _repair_pbox, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} repair-pbox-first must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_repair_proc_only_wins_easy(seed):
    """On easy no band sits on the war factory, so a bridged proc keeps
    its band pinned — repairing only the proc is sufficient → WIN."""
    c, r = _run("easy", _repair_proc, seed=seed)
    assert r.outcome == "win", (
        f"easy seed{seed} repair-proc-only should WIN, got {r.outcome}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_repair_proc_only_loses_on_medium_and_hard(level, seed):
    """On medium / hard a light band ALSO chews the war factory —
    repairing only the proc is no longer enough → LOSS (the deeper
    2-of-3 triage axis)."""
    c, r = _run(level, _repair_proc, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} repair-proc-only must LOSE; got {r.outcome} "
        f"(tick={r.signals.game_tick})"
    )
