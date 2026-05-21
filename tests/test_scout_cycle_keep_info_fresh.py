"""scout-cycle-keep-info-fresh — Wave-9 information-freshness pack.

Capability: PERCEPTION (observation is not a one-shot — the agent
must CYCLE its scout back into a previously-observed region after
mid-episode reinforcements arrive there).

This pack is the first to use the Wave-9 engine feature
`scheduled_events:` (oramap.rs::ScheduledEventKind::SpawnActors,
fired by env.rs::fire_scheduled_events on the world-tick the event's
`tick` is reached). Without mid-episode actor spawn no scenario
could test the information-FRESHNESS perception loop.

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall          → LOSS every level/seed (clock; nothing seen)
  * scout-once     → LOSS every level/seed: the agent scouts the
                     swarm at (60,20) once, commits the tanks, then
                     pulls the jeeps back; the tick-1500
                     reinforcement spawns DEEP at (78,20) — a cell
                     no field unit still covers — so the third
                     `then:` clause (enemies_discovered_gte:N_total)
                     never latches.
  * intended cycle → WIN every level/seed: one jeep is cycled out
                     to (76,20) so it has vision of (78,20) when
                     the reinforcement spawns; the tanks kill the
                     swarm for the K bar.

The kill bar is load-bearing because the agent's 3 tanks are
stance:0 (a scenario-declared stance:3 ground unit auto-hunts the
whole map on the live engine — that would let a stall policy win
for free).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACK_PATH = (
    Path(__file__).parent.parent
    / "openra_bench"
    / "scenarios"
    / "packs"
    / "scout-cycle-keep-info-fresh.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-cycle-keep-info-fresh"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_every_level_declares_a_scheduled_event():
    """The Wave-9 feature is the whole point of this pack — every
    level must carry a `scheduled_events:` SpawnActors entry."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        sched = getattr(c, "scheduled_events", None) or []
        assert sched, f"{lvl} has no scheduled_events block"
        kinds = {e.get("type") for e in sched}
        assert "spawn_actors" in kinds, (
            f"{lvl} scheduled_events missing a spawn_actors event: {sched}"
        )
        # The reinforcement fires mid-episode (well after t=0).
        assert all(e["tick"] >= 1000 for e in sched), sched


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout
    or fact-loss."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside max_turns.
    Engine advances ~90 ticks/turn → reachable max = 93 + 90·(N-1).
    The fail `after_ticks` must ALSO be reachable so a non-win run
    is a real LOSS (not a silent DRAW)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        fail = compile_level(pack, lvl).fail_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list[int] = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable}"
            )
        ats: list[int] = []
        _collect(fail, "after_ticks", ats)
        assert ats, f"{lvl} fail has no after_ticks leaf"
        # The fail deadline must be reachable so timeouts LOSE.
        assert max(ats) <= reachable, (
            f"{lvl} fail.after_ticks={max(ats)} > reachable={reachable} "
            f"— deadline never bites ⇒ draw"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


# ── Scripted policies ─────────────────────────────────────────────


def _stall(_rs, Command):
    return [Command.observe()]


def _scout_once(rs, Command):
    """One-shot scout: jeeps push to the swarm at (58,20) until
    tick 800, then retreat to base and stay. Tanks attack the e3
    swarm. The deep-fog reinforcement at (78,20) is never re-
    scouted → the third `then:` clause never latches."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    tick = rs.get("game_tick", 0)
    jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
    tanks = [u for u in units if str(u.get("type", "")).lower() == "2tnk"]
    cmds = []
    for u in jeeps:
        if tick < 800:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=58, target_y=20)
            )
        else:
            cmds.append(
                Command.move_units([str(u["id"])], target_x=12, target_y=20)
            )
    e3 = [e for e in enemies if str(e.get("type", "")).lower() == "e3"]
    if e3:
        for u in tanks:
            cmds.append(
                Command.attack_unit([str(u["id"])], str(e3[0]["id"]))
            )
    return cmds or [Command.observe()]


def _scout_cycle(rs, Command):
    """Intended policy: jeep 0 is cycled OUT to (76,20) deep fog so
    it has vision of the (78,20) reinforcement when it spawns at
    tick 1500; jeep 1 holds near the swarm at (58,20). The 3 tanks
    are committed to kill the e3 swarm for the K bar."""
    units = rs.get("units_summary", []) or []
    enemies = rs.get("enemy_summary", []) or []
    jeeps = sorted(
        [u for u in units if str(u.get("type", "")).lower() == "jeep"],
        key=lambda u: u["id"],
    )
    tanks = [u for u in units if str(u.get("type", "")).lower() == "2tnk"]
    cmds = []
    if jeeps:
        cmds.append(
            Command.move_units([str(jeeps[0]["id"])], target_x=76, target_y=20)
        )
    if len(jeeps) > 1:
        cmds.append(
            Command.move_units([str(jeeps[1]["id"])], target_x=58, target_y=20)
        )
    e3 = [e for e in enemies if str(e.get("type", "")).lower() == "e3"]
    if e3:
        for u in tanks:
            cmds.append(
                Command.attack_unit([str(u["id"])], str(e3[0]["id"]))
            )
    return cmds or [Command.observe()]


# ── Solvency: intended scout-cycle WINS every (level, seed) ───────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_cycle_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _scout_cycle, seed=seed)
    assert res.outcome == "win", (
        f"intended scout-cycle must WIN on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick} "
        f"seen={len(res.signals.enemies_seen_ids)} "
        f"killed={res.signals.units_killed}"
    )


# ── Stability: every non-win pattern is a real reachable LOSS ─────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_scout_once_loses(level, seed):
    """The headline discrimination: a one-shot scout meets the FIRST
    detection bar but, because it never re-cycles to the deep-fog
    (78,20) reinforcement cell, can never satisfy the post-reinforce
    clause — a real reachable LOSS, never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _scout_once, seed=seed)
    assert res.outcome == "loss", (
        f"scout-once must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} "
        f"then={res.signals.then_progress}"
    )
