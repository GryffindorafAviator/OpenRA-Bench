"""Full contributor loop for the strategy-trilemma scenario on Rust:

    pack YAML -> compile (per-level starting_cash) -> temp scenario ->
    Rust engine (fact+powr+proc+tent + hunt harasser + cash) ->
    build/place_building/move_units -> own/enemy observation ->
    declarative win/fail -> score.

The capability is **commit one indivisible budget to one path**. The
trilemma is enforced by `starting_cash` alone (harvest income is 0 on a
runnable pack today), tuned so the medium 2400 cr funds EXACTLY one of:

    EXPAND  = build+place a 2nd refinery (proc, 1400) in the far region
              centred at (60,20).
    TECH    = build+place a war factory (weap, 2000).
    ARMY    = train ≥12 e1 (12 × 100 = 1200) AND defeat the hunt
              harasser (≥1 kill credit).

Pairwise sums: 1400+2000=3400; 1400+1200=2600; 2000+1200=3200 — all >
2400, so a hedged spend funds NONE of the three arms in time. The
intended commit-one-path policy WINS exactly one arm; the lazy/split/
brute policies LOSE via the tick-aligned deadline.

Cash tuning per tier (the iteration the brief warned would be largest):
    easy   = 3000 (slack: ≥1.25× any single arm; budget still cannot
                   fund any pair).
    medium = 2400 (binding: any pair exceeds the budget).
    hard   = 2200 (lean: TECH eats 91% of it; ARMY 55%; EXPAND 64% —
                   with attrition cap units_lost_lte: 3).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scoring import score_episode

PACK = PACKS_DIR / "strategy-trilemma.yaml"


# ── scripted policies ──────────────────────────────────────────────────


def stall(rs, Command):
    """Do nothing — the hunt harasser whittles non-fact structures; the
    high-HP fact survives but the within_ticks deadline fires → loss."""
    return [Command.observe()]


def make_split_cash():
    """Hedge across all three arms: queue proc + e1 + weap concurrently,
    place proc at (60,20) and weap at (18,18) once each is ready. With
    the medium budget (2400) the weap (2000) cannot complete before the
    proc (1400) drains cash; e1 production stalls; no arm completes."""
    s = {"t": 0}

    def fn(rs, Command):
        s["t"] += 1
        t = s["t"]
        if t == 1: return [Command.build("proc")]
        if t == 2: return [Command.build("e1")]
        if t == 3: return [Command.build("weap")]
        if t == 4: return [Command.place_building("proc", 60, 20)]
        if t == 5: return [Command.place_building("weap", 18, 18)]
        return [Command.build("e1")]

    return fn


def brute_army_no_tent(rs, Command):
    """Spam build('e1'). With tent removed (see the per-test override)
    the tech tree blocks infantry production entirely — the policy
    issues build orders for ~40 turns, produces zero units, and times
    out without ever satisfying ARMY (or any arm)."""
    return [Command.build("e1")]


def make_commit_expand():
    """EXPAND: queue a 2nd proc and place it at (60,20)."""
    s = {"t": 0}

    def fn(rs, Command):
        s["t"] += 1
        own_b = rs.get("own_buildings") or []
        has_far_proc = any(
            (b.get("type") if isinstance(b, dict) else b[0]) == "proc"
            and (b.get("cell_x") if isinstance(b, dict) else b[1]) >= 54
            for b in own_b
        )
        if has_far_proc:
            return [Command.observe()]
        if s["t"] == 1:
            return [Command.build("proc")]
        return [Command.place_building("proc", 60, 20)]

    return fn


def make_commit_tech():
    """TECH: queue and place a war factory next to the base."""
    s = {"t": 0}

    def fn(rs, Command):
        s["t"] += 1
        own_b = rs.get("own_buildings") or []
        has_weap = any(
            (b.get("type") if isinstance(b, dict) else b[0]) == "weap"
            for b in own_b
        )
        if has_weap:
            return [Command.observe()]
        if s["t"] == 1:
            return [Command.build("weap")]
        return [Command.place_building("weap", 18, 18)]

    return fn


def make_commit_army():
    """ARMY: spam e1 and scatter spawned units to nearby holding cells
    (so they don't block production and so the default Defend stance
    engages the inbound hunt harasser, giving the required kill)."""
    s = {"moved": set()}

    def fn(rs, Command):
        units = rs.get("units_summary") or []
        cmds = [Command.build("e1")]
        holds = [(16, 21), (17, 22), (18, 23), (15, 24), (16, 25)]
        for u in units:
            if u.get("type") == "e1" and u["id"] not in s["moved"]:
                tx, ty = holds[len(s["moved"]) % len(holds)]
                cmds.append(Command.move_units([u["id"]], tx, ty))
                s["moved"].add(u["id"])
        return cmds

    return fn


LEVELS = ("easy", "medium", "hard")
HARD_SEEDS = (1, 2, 3, 4)


# ── tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_commit_expand_wins_every_level(level):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_commit_expand(), seed=1)
    assert res.outcome == "win", (
        f"{level}: commit-EXPAND must WIN; got {res.outcome} "
        f"(tick={res.signals.game_tick} cash={res.signals.cash})"
    )
    sc = score_episode(c, res)
    assert sc.outcome == "win"


@pytest.mark.parametrize("level", LEVELS)
def test_intended_commit_tech_wins_every_level(level):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_commit_tech(), seed=1)
    assert res.outcome == "win", (
        f"{level}: commit-TECH must WIN; got {res.outcome} "
        f"(tick={res.signals.game_tick} cash={res.signals.cash})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_intended_commit_army_wins_every_level(level):
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, make_commit_army(), seed=1)
    assert res.outcome == "win", (
        f"{level}: commit-ARMY must WIN; got {res.outcome} "
        f"(tick={res.signals.game_tick} kills={res.signals.units_killed})"
    )


@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_intended_commit_expand_wins_on_every_hard_seed(seed):
    pack = load_pack(PACK)
    c = compile_level(pack, "hard")
    res = run_level(c, make_commit_expand(), seed=seed)
    assert res.outcome == "win", (
        f"hard seed={seed}: commit-EXPAND must WIN; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_intended_commit_tech_wins_on_every_hard_seed(seed):
    pack = load_pack(PACK)
    c = compile_level(pack, "hard")
    res = run_level(c, make_commit_tech(), seed=seed)
    assert res.outcome == "win", (
        f"hard seed={seed}: commit-TECH must WIN; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", HARD_SEEDS)
def test_intended_commit_army_wins_on_every_hard_seed(seed):
    pack = load_pack(PACK)
    c = compile_level(pack, "hard")
    res = run_level(c, make_commit_army(), seed=seed)
    assert res.outcome == "win", (
        f"hard seed={seed}: commit-ARMY must WIN; got {res.outcome}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
def test_split_cash_loses_on_binding_tiers(level):
    """Hedged spend funds none of the three arms before the deadline."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    for seed in (1, 2, 3, 4):
        res = run_level(c, make_split_cash(), seed=seed)
        assert res.outcome == "loss", (
            f"{level} seed={seed}: split-cash must LOSE (no clause met); "
            f"got {res.outcome} (tick={res.signals.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_via_timeout_every_level(level):
    """The do-nothing policy: hunt erodes the base but fact survives;
    the loss arrives via the tick-aligned deadline (a real reachable
    LOSS, not a draw)."""
    pack = load_pack(PACK)
    c = compile_level(pack, level)
    res = run_level(c, stall, seed=1)
    assert res.outcome == "loss", (
        f"{level}: stall must LOSE (timeout); got {res.outcome}"
    )


def test_brute_army_without_tent_loses_to_tech_tree():
    """Without the barracks (tent), infantry production has no
    prerequisite — every `build('e1')` order is silently dropped by
    the engine and no units ever appear, so the ARMY arm cannot be
    satisfied. The episode times out as a real LOSS.

    Tent must be stripped from BOTH the pack's `base.actors` AND the
    matching level `overrides.actors` (when the level declares one) —
    a level override REPLACES base actors, so leaving tent in the
    override leaks the barracks back in and the gate never fires.
    """
    pack = load_pack(PACK)
    modified = pack.model_copy(deep=True)

    def _strip_tent(actors):
        return [
            a for a in actors
            if not (isinstance(a, dict) and a.get("type") == "tent")
        ]

    # Strip tent from base actors.
    modified.base["actors"] = _strip_tent(modified.base["actors"])
    # AND from every level override that declares its own actors block
    # (otherwise the override replaces base.actors and tent leaks back
    # in — the prereq gate would never fire).
    for level_def in (modified.levels or {}).values():
        overrides = getattr(level_def, "overrides", None) or {}
        if "actors" in overrides:
            overrides["actors"] = _strip_tent(overrides["actors"])
    c = modified.compile("medium")
    res = run_level(c, brute_army_no_tent, seed=1)
    assert res.outcome == "loss", (
        f"brute army WITHOUT tent must LOSE (tech tree blocks "
        f"production); got {res.outcome}"
    )


def test_within_ticks_is_reachable_per_tier():
    """The deadline must be reachable inside max_turns
    (tick ≤ 93 + 90·(max_turns − 1)) — otherwise a staller draws
    instead of losing (the recurring tick/turn-alignment defect)."""
    pack = load_pack(PACK)
    expected_within = {"easy": 5400, "medium": 4500, "hard": 3600}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert expected_within[lvl] <= max_tick, (
            f"{lvl}: within_ticks={expected_within[lvl]} > reachable "
            f"max_tick={max_tick} (would draw instead of losing)"
        )


def test_hard_has_at_least_two_spawn_groups():
    """Hard-tier curation contract (also enforced in
    tests/test_hard_tier.py once the pack is added to UPGRADED)."""
    pack = load_pack(PACK)
    c = compile_level(pack, "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )


def test_per_level_cash_constraint_binds():
    """The per-level starting_cash carries the binding."""
    pack = load_pack(PACK)
    assert compile_level(pack, "easy").starting_cash == 3000
    assert compile_level(pack, "medium").starting_cash == 2400
    assert compile_level(pack, "hard").starting_cash == 2200


def test_trilemma_run_is_deterministic():
    pack = load_pack(PACK)
    c = compile_level(pack, "medium")
    a = run_level(c, make_commit_tech(), seed=7)
    b = run_level(c, make_commit_tech(), seed=7)
    assert (a.outcome, a.turns, a.signals.cash) == (
        b.outcome,
        b.turns,
        b.signals.cash,
    ), "same seed must yield identical outcome"
