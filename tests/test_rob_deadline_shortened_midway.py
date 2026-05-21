"""rob-deadline-shortened-midway — Wave-11 schedule-compression pack.

Capability: REASONING (the project-management drill known as
crashing the schedule — a plan paced against one deadline must be
compressed when the deadline is pulled FORWARD mid-execution).

This pack uses the Wave-9 engine feature `scheduled_events:` event
kind `shorten_deadline` (oramap.rs::ScheduledEventKind::
ShortenDeadline, fired by env.rs::fire_scheduled_events). A
`shorten_deadline` event at tick 1000 clamps the episode's
`max_ticks` DOWN to the SHORTENED budget; the win predicate's
`within_ticks` is exactly that shortened value. Without a
mid-episode deadline clamp no scenario could test "the budget you
planned against just shrank — re-plan and accelerate NOW".

Scripted policies cover the bar-defining outcomes per CLAUDE.md
"no defect, no cheat":

  * stall          → LOSS every level/seed: the stance:0 tanks
                     never move, the enemy fact is never razed,
                     `within_ticks` never satisfied, the clamped
                     deadline bites.
  * attack-move    → LOSS every level/seed: a pure beeline walks
    -only beeline    the tanks to the objective cell but
                     `attack_move` does not raze an enemy BUILDING
                     by itself — the tanks sit beside an intact
                     fact.
  * leisurely      → LOSS every level/seed: the agent dawdles
                     early (as if pacing for the original generous
                     budget); by the time the tick-1000
                     `shorten_deadline` event fires, a force that
                     left late cannot raze the fact before the
                     COMPRESSED `within_ticks`.
  * intended       → WIN every level/seed: commit all four tanks
    schedule         at full speed from turn 1 and switch to
    compression      `attack_unit` on the enemy fact id the moment
                     it surfaces — the HQ falls before the
                     compressed deadline.

The strike tanks are stance:0 (a scenario-declared stance:3 ground
unit auto-hunts the whole map on the live engine — that would let
a stall policy win for free; stance:0 makes the pacing/commit
decision the load-bearing verb).
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
    / "rob-deadline-shortened-midway.yaml"
)

LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; no engine) ───────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "rob-deadline-shortened-midway"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_every_level_declares_a_shorten_deadline_event():
    """The Wave-9 `shorten_deadline` event is the whole point of
    this pack — every level must carry one, firing mid-episode and
    clamping `max_ticks` to exactly the win `within_ticks` value."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        sched = getattr(c, "scheduled_events", None) or []
        assert sched, f"{lvl} has no scheduled_events block"
        shorten = [e for e in sched if e.get("type") == "shorten_deadline"]
        assert shorten, (
            f"{lvl} scheduled_events missing a shorten_deadline event: "
            f"{sched}"
        )
        for e in shorten:
            # Fires mid-episode (well after t=0) — a deadline that
            # moves DURING execution, not a fixed t=0 budget.
            assert e["tick"] >= 500, e
            assert "new_max_ticks" in e, e


def test_shorten_clamp_matches_win_within_ticks():
    """The compressed budget the event clamps to must equal the
    win predicate's `within_ticks` — the deadline the agent must
    actually beat is exactly the value the event installs."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        sched = getattr(c, "scheduled_events", None) or []
        clamp = next(
            e["new_max_ticks"]
            for e in sched
            if e.get("type") == "shorten_deadline"
        )
        win = c.win_condition.model_dump(exclude_none=True)

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
        assert wts and all(wt == clamp for wt in wts), (
            f"{lvl}: within_ticks {wts} must equal shorten clamp {clamp}"
        )


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on the
    compressed deadline or on losing the own HQ."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_fail_after_ticks_is_one_past_within_ticks():
    """The fail `after_ticks` must be exactly win `within_ticks`+1
    so (a) the win/fail windows never overlap and (b) a non-win run
    — whose episode `done`s past the clamped deadline — reliably
    crosses the fail clause ⇒ a real LOSS, never a silent DRAW."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        fail = c.fail_condition.model_dump(exclude_none=True)

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
        ats: list[int] = []
        _collect(win, "within_ticks", wts)
        _collect(fail, "after_ticks", ats)
        assert wts and ats, f"{lvl} missing within/after ticks leaves"
        assert max(ats) == max(wts) + 1, (
            f"{lvl}: fail after_ticks {max(ats)} must be "
            f"within_ticks {max(wts)} + 1"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define >=2 agent spawn_point groups (UPGRADED
    contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs >=2 spawn groups, got {sp}"


# ── Scripted policies ─────────────────────────────────────────────


def _enemy_fact_id(rs):
    """The surfaced engine id of the enemy construction yard, or
    None until a tank's vision reveals it."""
    for b in rs.get("enemy_buildings_summary", []) or []:
        if str(b.get("type", "")).lower() == "fact":
            return str(b["id"])
    return None


def _stall(_rs, Command):
    return [Command.observe()]


def _attack_move_only(rs, Command):
    """Pure beeline: attack_move the tanks at the objective cell but
    NEVER switch to attack_unit. The tanks reach the enemy fact but
    `attack_move` does not raze a building — the fact stays intact."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    return [
        Command.attack_move(
            [str(u["id"]) for u in units], target_x=110, target_y=20
        )
    ]


def _immediate_commit(rs, Command):
    """Intended schedule-compression policy: commit all four tanks
    at full speed from turn 1 (attack_move toward the objective),
    and the instant the enemy fact surfaces switch to attack_unit
    on its id — the HQ falls before the compressed deadline."""
    units = rs.get("units_summary", []) or []
    if not units:
        return [Command.observe()]
    ids = [str(u["id"]) for u in units]
    fid = _enemy_fact_id(rs)
    if fid is not None:
        return [Command.attack_unit(ids, fid)]
    return [Command.attack_move(ids, target_x=110, target_y=20)]


def _leisurely(delay_turns):
    """Paced-for-the-original-deadline policy: dawdle (observe) for
    `delay_turns` decision turns as if the generous nominal budget
    still held, THEN commit. Each wasted turn pushes the objective
    ~90 ticks later; once the tick-1000 shorten event has fired the
    compressed budget no longer covers a late start."""
    state = {"t": 0}

    def policy(rs, Command):
        state["t"] += 1
        if state["t"] <= delay_turns:
            return [Command.observe()]
        return _immediate_commit(rs, Command)

    return policy


# ── Solvency: intended immediate commit WINS every (level, seed) ──


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_immediate_commit_wins(level, seed):
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _immediate_commit, seed=seed)
    assert res.outcome == "win", (
        f"intended immediate commit must WIN on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick}"
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
def test_attack_move_only_beeline_loses(level, seed):
    """A pure beeline reaches the objective cell but `attack_move`
    does not raze the enemy building — the fact stays intact and
    the run is a real reachable LOSS."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _attack_move_only, seed=seed)
    assert res.outcome == "loss", (
        f"attack-move-only beeline must LOSE on {level} s={seed}; "
        f"got {res.outcome} tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_leisurely_pace_loses(level, seed):
    """The headline discrimination: a force paced leisurely for the
    original generous budget cannot raze the enemy HQ before the
    COMPRESSED deadline the tick-1000 `shorten_deadline` event
    installs — a real reachable LOSS, never a draw. A 4-turn dawdle
    (~360 ticks of slack the original budget would have absorbed)
    loses on every level/seed."""
    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _leisurely(4), seed=seed)
    assert res.outcome == "loss", (
        f"leisurely pace must LOSE on {level} s={seed}; got "
        f"{res.outcome} tick={res.signals.game_tick}"
    )
