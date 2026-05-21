"""scout-track-enemy-movement — Wave-11 continuous target-tracking pack.

Capability: PERCEPTION (tracking a moving target is not a one-shot
observation — the agent must keep a scout MOVING WITH an enemy army
as it marches across the map, re-observing it at each successive leg,
so the agent always knows where the army is and can intercept it).

The army's multi-leg march is scripted with the Wave-9 engine
feature `scheduled_events:` — a `spawn_actors` at leg N+1 followed
60 ticks later by a `destroy_actors` of leg N. The spawn-BEFORE-
destroy ordering is load-bearing: a `destroy_actors` frees the
removed actors' ids and a `spawn_actors` that fired AFTER the free
would RECYCLE them, collapsing `enemies_discovered` (which counts
UNIQUE ids). Spawning the new leg's band before the old band's ids
are freed guarantees the relocated army carries genuinely fresh
ids, so re-acquiring it grows the discovered-id set.

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall          → LOSS every level/seed: nothing ever moves; the
                     army marches unobserved; the first detection
                     bar never latches; the `then:` chain never
                     starts; the deadline bites.
  * scout-once     → LOSS every level/seed: the agent scouts the
                     army at its STARTING leg once, latching the
                     first detection clause, then sits still. When
                     the army relocates, the parked scout has no
                     vision of the new leg → the discovered-id set
                     stops growing → the `then:` chain stalls on a
                     later clause → the deadline bites.
  * track-no-kill  → LOSS every level/seed: a scout shadows the army
                     perfectly (the full detection chain latches)
                     but the tanks never commit → units_killed_gte
                     fails. Tracking without acting on the track is
                     not a win.
  * intended track → WIN every level/seed: one jeep shadows the
                     army leg-by-leg (always re-acquiring it after
                     each relocation); the tanks are vectored to the
                     army's final leg and destroy ≥K of it.

The kill bar is load-bearing because the agent's tanks are stance:0
(a scenario-declared stance:3 ground unit auto-hunts the whole map
on the live engine — that would let a stall policy win for free).
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
    / "scout-track-enemy-movement.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# The army's march legs (x, y) and the relocation ticks per tier —
# mirrors the scheduled_events spawn cells in the pack YAML.
LEGS = {
    "easy": [(38, 10), (70, 20), (96, 30)],
    "medium": [(38, 10), (70, 12), (70, 28), (96, 30)],
    "hard": [(38, 10), (70, 12), (70, 28), (96, 30)],
}
TICKS = {
    "easy": [1400, 2700],
    "medium": [1100, 2000, 2900],
    "hard": [1100, 2000, 2900],
}


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "scout-track-enemy-movement"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_benchmark_anchor_declares_target_tracking():
    pack = load_pack(PACK_PATH)
    anchor = " ".join(pack.meta.benchmark_anchor).lower()
    assert "military target tracking" in anchor
    assert "intrusion detection" in anchor
    assert "continuous monitoring" in anchor


def test_every_level_scripts_a_multileg_march():
    """The army must MOVE: every level carries scheduled_events with
    paired spawn_actors / destroy_actors, and the spawn events must
    fire BEFORE their matching destroy (fresh-id ordering)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        sched = getattr(c, "scheduled_events", None) or []
        assert sched, f"{lvl} has no scheduled_events block"
        spawns = [e for e in sched if e.get("type") == "spawn_actors"]
        destroys = [e for e in sched if e.get("type") == "destroy_actors"]
        # At least two relocations → a genuine multi-leg march.
        assert len(spawns) >= 2, f"{lvl} needs ≥2 relocations: {sched}"
        assert len(destroys) >= 2, f"{lvl} needs ≥2 destroys: {sched}"
        # Every destroy must fire AFTER the earliest spawn (spawn-
        # before-destroy keeps the relocated band's ids fresh).
        first_spawn = min(e["tick"] for e in spawns)
        for d in destroys:
            assert d["tick"] > first_spawn, (
                f"{lvl}: destroy at {d['tick']} not after first spawn "
                f"{first_spawn} — id-recycle would collapse "
                f"enemies_discovered"
            )


def test_win_chain_is_an_ordered_then_with_one_bar_per_leg():
    """The win predicate must be a `then:` chain whose detection
    bars strictly increase — so each later bar needs another full
    band's worth of UNIQUE ids (a one-shot scout cannot complete
    it)."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)

        def _find_then(node):
            if isinstance(node, dict):
                if "then" in node:
                    return node["then"]
                for v in node.values():
                    r = _find_then(v)
                    if r:
                        return r
            elif isinstance(node, list):
                for v in node:
                    r = _find_then(v)
                    if r:
                        return r
            return None

        then = _find_then(win)
        assert then, f"{lvl} win_condition has no `then:` chain"
        bars = [
            cl["enemies_discovered_gte"]
            for cl in then["clauses"]
            if "enemies_discovered_gte" in cl
        ]
        assert len(bars) >= 3, (
            f"{lvl} needs ≥3 detection bars (one per leg), got {bars}"
        )
        assert bars == sorted(bars) and len(set(bars)) == len(bars), (
            f"{lvl} detection bars must strictly increase, got {bars}"
        )


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


def _jeeps(rs):
    return [
        u for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "jeep"
    ]


def _tanks(rs):
    return [
        u for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]


def _enemy_units(rs):
    return [
        e for e in (rs.get("enemy_summary") or [])
        if not e.get("is_building")
    ]


def _stall(_rs, Command):
    return [Command.observe()]


def _make_scout_once(legs):
    """One-shot scout: jeeps + tanks push to the army's STARTING leg
    and then sit. When the army relocates the parked units have no
    vision of the new leg → the discovered-id set stops growing."""
    ax, ay = legs[0]

    def pol(rs, Command):
        cmds = []
        for u in _jeeps(rs):
            cmds.append(Command.move_units([str(u["id"])], ax, ay))
        for u in _tanks(rs):
            cmds.append(Command.attack_move([str(u["id"])], ax, ay))
        return cmds or [Command.observe()]

    return pol


def _make_track_no_kill(legs, ticks):
    """Perfect tracking, no engagement: a jeep shadows the army leg
    by leg (the full detection chain latches) but the tanks never
    commit → the kill bar fails."""
    final = legs[-1]

    def pol(rs, Command):
        cmds = []
        t = int(rs.get("game_tick", 0))
        leg = 0
        for i, tk in enumerate(ticks):
            if t >= tk - 120:
                leg = i + 1
        leg = min(leg, len(legs) - 1)
        cur = legs[leg]
        js = _jeeps(rs)
        if js:
            cmds.append(Command.move_units([str(js[0]["id"])], cur[0], cur[1]))
        if len(js) > 1:
            cmds.append(
                Command.move_units([str(js[1]["id"])], final[0], final[1])
            )
        return cmds or [Command.observe()]

    return pol


def _make_tracker(legs, ticks):
    """Intended policy: one jeep shadows the army leg-by-leg (always
    re-acquiring it after each relocation); the other jeep pre-
    stages toward the final leg; the tanks stage near the final leg
    and attack_unit a visible enemy once the army has relocated
    there."""
    final = legs[-1]
    last_tick = ticks[-1]
    stage = (final[0] - 6, final[1] - 2)

    def pol(rs, Command):
        cmds = []
        t = int(rs.get("game_tick", 0))
        leg = 0
        for i, tk in enumerate(ticks):
            if t >= tk - 120:
                leg = i + 1
        leg = min(leg, len(legs) - 1)
        cur = legs[leg]
        js = _jeeps(rs)
        if js:
            cmds.append(Command.move_units([str(js[0]["id"])], cur[0], cur[1]))
        if len(js) > 1:
            cmds.append(
                Command.move_units([str(js[1]["id"])], final[0], final[1])
            )
        ep = _enemy_units(rs)
        if t >= last_tick:
            tgt = [
                e for e in ep
                if abs(int(e.get("cell_x", -99)) - final[0]) <= 6
                and abs(int(e.get("cell_y", -99)) - final[1]) <= 6
            ]
            if tgt:
                eid = str(tgt[0]["id"])
                for u in _tanks(rs):
                    cmds.append(Command.attack_unit([str(u["id"])], eid))
            else:
                for u in _tanks(rs):
                    cmds.append(
                        Command.attack_move([str(u["id"])], stage[0], stage[1])
                    )
        else:
            for u in _tanks(rs):
                cmds.append(
                    Command.attack_move([str(u["id"])], stage[0], stage[1])
                )
        return cmds or [Command.observe()]

    return pol


# ── Solvency: intended continuous-tracking WINS every (level, seed) ─


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_tracker_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _make_tracker(LEGS[level], TICKS[level]), seed=seed)
    assert res.outcome == "win", (
        f"intended continuous-tracking must WIN on {level} s={seed}; "
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
    detection bar but, because it never follows the army to its
    later legs, can never satisfy the later detection clauses — a
    real reachable LOSS, never a draw."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _make_scout_once(LEGS[level]), seed=seed)
    assert res.outcome == "loss", (
        f"scout-once must LOSE on {level} s={seed}; got {res.outcome} "
        f"tick={res.signals.game_tick} then={res.signals.then_progress}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_track_without_intercept_loses(level, seed):
    """Tracking without acting on the track is not a win: a policy
    that shadows the army perfectly (full detection chain latches)
    but never commits the tanks fails the kill bar."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(
        c, _make_track_no_kill(LEGS[level], TICKS[level]), seed=seed
    )
    assert res.outcome == "loss", (
        f"track-without-intercept must LOSE on {level} s={seed}; "
        f"got {res.outcome} killed={res.signals.units_killed}"
    )
