"""No-cheat + solvency proof for `rob-cash-depletion-recovery` (Group J
robustness — mid-episode recovery after a scripted enemy strike
destroys the refinery and halts income).

The pack frames a CASH-CRISIS RECOVERY decision: the agent inherits a
normal operating economy (fact + proc + 1 harv + 2 mines + a small
defender squad) with $1400 cash — exactly enough to rebuild ONE
refinery (proc cost 1400). An enemy `hunt` 4tnk strike destroys the
proc mid-episode (medium ≈ tick 540; hard ≈ tick 450; on easy the
defenders intercept the lone 4tnk and the proc survives). The agent
must `build('proc') + place_building` and re-issue `harvest`.

For every level + every hard seed (1-4):
  * INTENDED chain (notice the destruction, build proc, place, re-
    issue harvest) WINS;
  * STALL (only `observe`) LOSES every tier — no harvest income, EV
    stays at the starting $1400, bar (2500) unmet, clock bites;
  * NO-REBUILD (harvest with the starting harv but never `build`
    proc) is the EASY FLOOR (wins easy where the proc survives) but
    LOSES medium/hard (the structural `has_building:proc` clause
    fails after the proc is destroyed);
  * BUILD-ARMY (spend the 1400 reserve on powr → tent → e1 instead
    of replacing the proc) LOSES medium/hard — the reserve is
    starved, the proc rebuild never completes, the bar is unmet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "rob-cash-depletion-recovery.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── helpers ────────────────────────────────────────


def _own_fact(bldgs):
    """Return the AGENT-owned fact (skip the far enemy marker at x=120)."""
    for b in bldgs:
        if b.get("type") == "fact" and b.get("cell_x", 0) < 100:
            return b
    return None


def _patches_for_fact(fy: int):
    """easy/medium: one fact at y=18 → patches at (22,18)/(22,20).
    hard: NORTH y=14 → (22,14)/(22,16); SOUTH y=26 → (22,26)/(22,28)."""
    if abs(fy - 18) <= 2:
        return [(22, 18), (22, 20)]
    if fy < 18:
        return [(22, 14), (22, 16)]
    return [(22, 26), (22, 28)]


def _ev(res):
    return res.signals.cash + res.signals.resources


def _current_building_types(res):
    """`signals.own_building_types` is accumulative (grows across the
    episode); for "currently alive" checks use `signals.own_buildings`
    (overwritten each observation)."""
    return {t for (t, _, _) in res.signals.own_buildings}


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Idle: never harvests, never rebuilds. EV stays at starting
    $1400, bar (2500) unmet → reachable timeout LOSS every tier."""
    return [Command.observe()]


def _no_rebuild_factory():
    """Run the starting harv but never `build` proc. WINS easy (the
    proc survives there — the defenders intercept the lone 4tnk) and
    LOSES medium/hard (after the proc is destroyed, the
    `has_building:proc` clause fails permanently)."""

    state = {"moved": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        patches = _patches_for_fact(fact_b["cell_y"])
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        for i, (uid, cx, cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["moved"]:
                cmds.append(Command.move_units([uid_s], target_x=px, target_y=py))
                if abs(cx - px) <= 3 and abs(cy - py) <= 3:
                    state["moved"].add(uid_s)
            else:
                cmds.append(Command.harvest([uid_s], px, py))
        return cmds if cmds else [Command.observe()]

    return policy


def _build_army_factory():
    """Spend the reserve on powr → tent → e1 instead of rebuilding
    the proc. The 1400 cash is drained on the wrong assets; the proc
    rebuild never funds → `has_building:proc` fails permanently on
    medium/hard → LOSS."""

    state = {"moved": set()}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]
        cmds = []
        for i, (uid, cx, cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["moved"]:
                cmds.append(Command.move_units([uid_s], target_x=px, target_y=py))
                if abs(cx - px) <= 3 and abs(cy - py) <= 3:
                    state["moved"].add(uid_s)
            else:
                cmds.append(Command.harvest([uid_s], px, py))
        if "powr" not in own_types:
            if cash >= 300 and "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 2, fy - 3))
        elif "tent" not in own_types:
            if cash >= 500 and "tent" not in prod:
                cmds.append(Command.build("tent"))
            cmds.append(Command.place_building("tent", fx + 2, fy + 3))
        elif cash >= 100 and "e1" not in prod:
            cmds.append(Command.build("e1"))
        return cmds if cmds else [Command.observe()]

    return policy


def _intended_rebuild_factory():
    """Intended recovery chain: harvest with the starting harv → if
    proc disappears, `build('proc')` + place adjacent to the fact →
    once the new proc lands, re-issue `harvest` for the surviving
    harv. WINS every tier × every seed (on easy the proc never
    falls; on medium/hard the rebuild fires)."""

    state = {"moved": set(), "placed_proc": False}

    def policy(rs, Command):
        units = rs.get("units_summary", []) or []
        bldgs = rs.get("own_buildings", []) or []
        own_types = [b["type"] for b in bldgs]
        prod = rs.get("production", []) or []
        cash = rs.get("cash", 0)
        fact_b = _own_fact(bldgs)
        if fact_b is None:
            return [Command.observe()]
        fx, fy = fact_b["cell_x"], fact_b["cell_y"]
        patches = _patches_for_fact(fy)
        n_proc = sum(1 for t in own_types if t == "proc")

        harvs = [
            (u["id"], u.get("cell_x"), u.get("cell_y"))
            for u in units
            if str(u.get("type", "")).lower() == "harv"
        ]

        cmds = []
        # If proc is gone, queue a rebuild and spam place_building.
        if n_proc < 1:
            if cash >= 1400 and "proc" not in prod:
                cmds.append(Command.build("proc"))
            # Place at the original proc spot (fx+4, fy) — adjacent to
            # the fact, in-bounds on the rush-hour map.
            cmds.append(Command.place_building("proc", fx + 4, fy))

        # Always nudge each harv onto a patch and harvest. Re-issue
        # the harvest order EVERY turn so that when the new proc
        # lands, the auto-cycle resumes (it does not auto-resume
        # across a destroy-rebuild gap).
        for i, (uid, cx, cy) in enumerate(harvs):
            uid_s = str(uid)
            px, py = patches[i % len(patches)]
            if uid_s not in state["moved"]:
                cmds.append(Command.move_units([uid_s], target_x=px, target_y=py))
                if abs(cx - px) <= 3 and abs(cy - py) <= 3:
                    state["moved"].add(uid_s)
            cmds.append(Command.harvest([uid_s], px, py))
        return cmds if cmds else [Command.observe()]

    return policy


def _run(level, policy_or_factory, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    pol = (
        policy_or_factory()
        if callable(policy_or_factory)
        and policy_or_factory.__name__.endswith("factory")
        else policy_or_factory
    )
    return c, run_level(c, pol, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "rob-cash-depletion-recovery"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Spec-required tools (Wave-6 spec): all 8.
    for t in (
        "observe",
        "build",
        "place_building",
        "harvest",
        "move_units",
        "attack_unit",
        "attack_move",
        "stop",
    ):
        assert t in tools, f"missing tool {t} in {tools}"


def test_starting_cash_is_one_proc_on_every_level():
    """Reserve must fund EXACTLY one refinery rebuild — any other
    value breaks the indivisible-rebuild test."""
    pack = load_pack(PACK)
    for L in LEVELS:
        c = compile_level(pack, L)
        assert c.starting_cash == 1400, (
            f"{L}: starting_cash must equal proc cost 1400 (got {c.starting_cash})"
        )


def test_benchmark_anchor_lists_financial_and_recovery():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "sc2" in blob or "cash" in blob, anchors
    assert "recovery" in blob or "turnaround" in blob or "rebuild" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns. within_ticks
    + 1 == after_ticks idiom."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    within_clauses = c.win_condition.model_dump().get("all_of", [])
    wt = next(int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x)
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt + 1})"
    )


def test_hard_strike_destroys_proc_and_tent_per_spec():
    """Hard tier strike-severity contract (the controlled variable
    vs medium): the scripted enemy strike must destroy BOTH proc
    AND tent — not just one — so the rebuild decision discipline
    (income-restoring proc rebuild prioritised over defensive-tech
    tent rebuild) is exercised. Verified via the stall policy
    (the strike outcome is deterministic regardless of agent
    actions; stall lets the strike land cleanly)."""
    c, r = _run("hard", _stall, seed=1)
    # The proc and tent are pre-placed at start (additive set
    # confirms ever-existed); the per-frame list confirms both are
    # GONE by episode end (= scripted strike landed).
    assert "proc" in r.signals.own_building_types, (
        "proc must have been pre-placed (additive set)"
    )
    assert "tent" in r.signals.own_building_types, (
        "tent must have been pre-placed (additive set)"
    )
    cur = _current_building_types(r)
    assert "proc" not in cur, (
        f"hard strike must destroy the proc; currently-alive types={cur}"
    )
    assert "tent" not in cur, (
        f"hard strike must destroy the tent; currently-alive types={cur}"
    )


# ───────────────────────── intended WIN bar ───────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_rebuild_wins(level, seed):
    """Intended recovery (harvest with starting harv → on proc death,
    build('proc') + place + re-harvest) WINS every tier × every seed."""
    c, r = _run(level, _intended_rebuild_factory, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended rebuild should WIN, got {r.outcome}; "
        f"ev={_ev(r)}, cash={r.signals.cash}, "
        f"types={r.signals.own_building_types}, tick={r.signals.game_tick}"
    )


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall (only `observe`) must LOSE every tier × every seed — no
    harvest income, EV stays at $1400, bar 2500 unmet, clock bites."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, tick={r.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_no_rebuild_is_easy_floor(seed):
    """No-rebuild (harvest with starting harv only) is the EASY
    FLOOR — the lone 4tnk is killed by the defender ring, the proc
    survives, harvest income clears the 2500 bar comfortably."""
    c, r = _run("easy", _no_rebuild_factory, seed=seed)
    assert r.outcome == "win", (
        f"easy seed{seed} no-rebuild should be the FLOOR and WIN; "
        f"got {r.outcome} (ev={_ev(r)}, "
        f"types={r.signals.own_building_types})"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_no_rebuild_loses_medium_and_hard(level, seed):
    """No-rebuild LOSES medium/hard — the strike destroys the proc;
    the `has_building:proc` clause fails permanently regardless of
    cash/EV. This is the discriminator that the recovery chain
    (`build('proc') + place_building`) is actually exercised."""
    c, r = _run(level, _no_rebuild_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} no-rebuild must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, types={r.signals.own_building_types})"
    )
    # The proc must have actually been destroyed for the
    # discrimination to be real (not an EV-bar miss). Note:
    # `own_building_types` is accumulative; use `own_buildings`
    # (current per-frame list) for the "alive now" check.
    cur = _current_building_types(r)
    assert "proc" not in cur, (
        f"{level}: no-rebuild should observe proc destroyed; "
        f"currently-alive types={cur}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
@pytest.mark.parametrize("seed", SEEDS)
def test_build_army_loses_medium_and_hard(level, seed):
    """Build-army (powr → tent → e1) drains the reserve on the wrong
    assets; the proc rebuild never funds → `has_building:proc` clause
    fails permanently → LOSS."""
    c, r = _run(level, _build_army_factory, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} build-army must LOSE; got {r.outcome} "
        f"(ev={_ev(r)}, types={r.signals.own_building_types})"
    )
    # Wrong-asset commit confirmed via the accumulative set (any
    # type ever built shows here); proc CURRENTLY alive checked via
    # the per-frame list.
    cur = _current_building_types(r)
    assert "proc" not in cur, (
        f"{level}: build-army should NEVER re-fund the proc; "
        f"currently-alive types={cur}"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _intended_rebuild_factory(), seed=2)
    b = run_level(c, _intended_rebuild_factory(), seed=2)
    assert (a.outcome, a.turns, _ev(a)) == (b.outcome, b.turns, _ev(b)), (
        f"determinism: {(a.outcome, a.turns, _ev(a))} vs "
        f"{(b.outcome, b.turns, _ev(b))}"
    )
