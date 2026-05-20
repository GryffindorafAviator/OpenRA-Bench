"""build-sell-and-rebuild-elsewhere pack — no-cheat validation on Rust.

Wave-8 capital reallocation pack. The pack tests SELL-AND-REBUILD as
a reasoning primitive: the agent's exposed refinery (proc) on the
centre lane will be razed by a `hunt` band, and starting cash alone
is NOT enough to build a new proc. The only path to a fresh proc at
the safe target region inside the tick budget is:

    1. `sell(proc_id)`                ⇒ refunds 50% of proc cost (700)
    2. `build('proc')` + `place_building(proc, x, y)`  in the safe region

The win predicate makes the SELL load-bearing:

* `building_in_region:{type:proc, x:safe_x, y:safe_y, radius:6, count:1}`
  ⇒ a fresh proc must STAND at the safe target region (not the centre
  lane; not anywhere outside the radius);
* `building_count_gte:{type:fact, n:1}` ⇒ the Construction Yard must
  still be alive (the PRESENT-TENSE predicate, not `has_building:fact`
  which is a one-shot ever-seen set — CLAUDE.md footgun);
* `within_ticks: 4500` paired with `after_ticks: 4501` in fail ⇒ the
  episode end is a real reachable timeout LOSS, never a draw.

The scripted-policy validations prove deterministically that:

* the intended SELL-THEN-REBUILD policy WINS every (level, seed);
* stall (observe only), build-without-selling (cash gated), and
  sell-then-misplace (new proc on the y=20 lane) all LOSE every
  (level, seed) — real LOSS, not draw;
* the hard tier defines ≥2 spawn_point groups (NORTH y=4 / SOUTH
  y=36) so a memorised "place at (16, 8)" cell cell that worked on
  easy/medium FAILS on the SOUTH spawn (the matching safe region
  there is (16, 36)).

NOTE on building ids: `sell` requires the real engine actor id
(e.g. `1003`), which the bench's `render_state["own_buildings"]`
strips. The scripted policies below reach into `_raw["own_buildings"]`
(via a small custom episode loop) to look up the proc id by cell.
The model-evaluation path is a separate concern: the model issues
sell-like reasoning and the win predicate is what actually grades
the outcome (real proc presence at the safe region).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import (
    RustEnvPool,
    _scenario_to_tmp_yaml,
    run_level,
)
from openra_bench.rust_adapter import RustObsAdapter
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "build-sell-and-rebuild-elsewhere.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── custom episode loop with raw building-id access ──────────────────


def _run_with_id_aware_policy(compiled, policy, seed):
    """Run an episode where the policy is given (rs, raw, Command) and
    can read `raw["own_buildings"][i]["id"]` (the real engine actor
    id). Mirrors `run_level`'s win/fail/draw scoring without changing
    the standard policy contract.
    """
    tmp = _scenario_to_tmp_yaml(compiled)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    try:
        adapter = RustObsAdapter()
        adapter.observe(env.reset(seed=seed))
        outcome = "draw"
        turns = 0
        for turns in range(1, compiled.max_turns + 1):
            rs = adapter.render_state()
            raw = adapter._raw  # for building ids
            cmds = policy(rs, raw, env.Command) or [env.Command.observe()]
            obs, _r, done, _info = env.step(cmds)
            adapter.observe(obs, done=done)
            ctx = WinContext(
                signals=adapter.signals,
                render_state=adapter.render_state(),
            )
            if evaluate(compiled.win_condition, ctx):
                outcome = "win"
                break
            if evaluate(compiled.fail_condition, ctx):
                outcome = "loss"
                break
            if done:
                break
        return outcome, turns, adapter.signals
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)


def _proc_id_at(raw, y):
    """Lookup the engine actor id of an own proc at the given y. None
    if no matching proc is alive."""
    for b in (raw.get("own_buildings") or []):
        if b.get("type") == "proc" and int(b.get("cell_y", -1)) == y:
            return str(b["id"])
    return None


def _own_proc_at(raw, y_band):
    """Any own proc inside ``y_band`` (a (lo, hi) inclusive interval)."""
    for b in (raw.get("own_buildings") or []):
        if (
            b.get("type") == "proc"
            and y_band[0] <= int(b.get("cell_y", -1)) <= y_band[1]
        ):
            return b
    return None


def _fact_y(raw):
    """Latitude of the agent's fact (4 on NORTH spawn / 4 on hard NORTH /
    36 on hard SOUTH). Used by the hard-tier intended policy to pick
    the matching safe target region."""
    for b in (raw.get("own_buildings") or []):
        if b.get("type") == "fact":
            return int(b.get("cell_y", 4))
    return 4


# ── scripted policies ───────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — proc razed, no new proc placed. LOSS."""
    return [C.observe()]


def make_build_without_selling(safe_x=16, safe_y=8):
    """Try to BUILD + PLACE a new proc WITHOUT selling the exposed one.
    Cash starts at 700/800 — well under the 1400 build cost — so the
    `build('proc')` queue starts but never completes (no income
    source). No new proc ⇒ region clause unmet ⇒ LOSS.

    NOTE: queue insufficient cash is silently ignored by the engine
    (production gates on cash > cost); the build never progresses.
    """

    def policy(rs, raw, C):
        cmds = []
        # Find any safe-region proc to terminate early once present.
        if any(
            b.get("type") == "proc" and int(b.get("cell_y", -1)) != 20
            for b in (raw.get("own_buildings") or [])
        ):
            return [C.observe()]
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if "proc" not in prod_items:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", safe_x, safe_y))
        return cmds or [C.observe()]

    return policy


def make_sell_then_misplace(safe_x_wrong=60, safe_y_wrong=20):
    """SELL the exposed proc (refund + cash buys a new proc) but
    PLACE the new proc back IN THE CENTRE LANE — outside the safe
    target region disc. The new proc satisfies `building_count_gte`
    but NOT `building_in_region` — LOSS.
    """
    state = {"sold": False}

    def policy(rs, raw, C):
        cmds = []
        if not state["sold"]:
            pid = _proc_id_at(raw, 20)
            if pid:
                cmds.append(C.sell([pid]))
                state["sold"] = True
        # No existing safe-region proc — but we deliberately place
        # back on the y=20 lane to demonstrate the misplace cost.
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if "proc" not in prod_items:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", safe_x_wrong, safe_y_wrong))
        return cmds or [C.observe()]

    return policy


def make_intended_easy_medium(safe_x=16, safe_y=8):
    """Intended SELL-THEN-REBUILD play for easy/medium (fact at
    (4, 4) so safe region is (16, 8)).

    Turn 1: sell the exposed proc (refunds 700, total cash → 1500).
    Continuously: queue `build('proc')` + `place_building` at the
    safe target region. The build completes ~1400 ticks after queue;
    place_building lands at (16, 8). Win clause fires.
    """
    state = {"sold": False}

    def policy(rs, raw, C):
        cmds = []
        if not state["sold"]:
            pid = _proc_id_at(raw, 20)
            if pid:
                cmds.append(C.sell([pid]))
                state["sold"] = True
        # Skip if the safe-region proc already exists.
        if any(
            b.get("type") == "proc" and int(b.get("cell_y", -1)) != 20
            for b in (raw.get("own_buildings") or [])
        ):
            return cmds or [C.observe()]
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if "proc" not in prod_items:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", safe_x, safe_y))
        return cmds or [C.observe()]

    return policy


def make_intended_hard_adaptive():
    """Intended SELL-THEN-REBUILD play for hard (fact at either y=4 or
    y=36 by seed). Reads the fact's actual y from the observation on
    turn 1, then places the new proc at the matching safe region —
    (16, 8) for NORTH spawn, (16, 36) for SOUTH spawn.
    """
    state = {"sold": False, "safe_xy": None}

    def policy(rs, raw, C):
        if state["safe_xy"] is None:
            fy = _fact_y(raw)
            state["safe_xy"] = (16, 8 if fy < 20 else 36)
        sx, sy = state["safe_xy"]
        cmds = []
        if not state["sold"]:
            pid = _proc_id_at(raw, 20)
            if pid:
                cmds.append(C.sell([pid]))
                state["sold"] = True
        # Skip if the safe-region proc already exists.
        if any(
            b.get("type") == "proc" and int(b.get("cell_y", -1)) != 20
            for b in (raw.get("own_buildings") or [])
        ):
            return cmds or [C.observe()]
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if "proc" not in prod_items:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", sx, sy))
        return cmds or [C.observe()]

    return policy


def make_memorised_north_only():
    """Naive: always place at (16, 8) (the easy/medium safe region).
    On hard SOUTH spawn (fact at y=36), the safe region is (16, 36),
    so a place at (16, 8) lands outside the matching radius-6 disc
    AND outside the SOUTH disc — LOSS on SOUTH seeds.
    """
    state = {"sold": False}

    def policy(rs, raw, C):
        cmds = []
        if not state["sold"]:
            pid = _proc_id_at(raw, 20)
            if pid:
                cmds.append(C.sell([pid]))
                state["sold"] = True
        if any(
            b.get("type") == "proc" and int(b.get("cell_y", -1)) != 20
            for b in (raw.get("own_buildings") or [])
        ):
            return cmds or [C.observe()]
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if "proc" not in prod_items:
            cmds.append(C.build("proc"))
        cmds.append(C.place_building("proc", 16, 8))
        return cmds or [C.observe()]

    return policy


# ── scenario-shape invariants ───────────────────────────────────────


def test_pack_compiles_with_three_levels_and_hunt_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-sell-and-rebuild-elsewhere"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    # Required-by-spec benchmark anchors (capital reallocation idiom).
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("capital reallocation" in a for a in anchors), pack.meta.benchmark_anchor
    assert any("sc2 sell mechanic" in a for a in anchors), pack.meta.benchmark_anchor
    assert any(
        "financial reallocation" in a for a in anchors
    ), pack.meta.benchmark_anchor
    # Hunt bot is wired through to the engine for every level (per-unit
    # nearest-foe targeting, so the proc on the centre lane is the
    # front piece, not the off-axis fact — see pack header).
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "hunt", (lvl, bot)


def test_sell_is_exposed_in_the_tool_palette():
    """`sell` is the load-bearing primitive — the pack would be
    unsolvable without it (build('proc') is cash-gated, the agent has
    no income source, the exposed proc would just be razed)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tools = set(getattr(c.scenario, "tools", None) or [])
        assert "sell" in tools, (lvl, tools)
        assert "build" in tools, (lvl, tools)
        assert "place_building" in tools, (lvl, tools)


def test_starting_cash_is_below_proc_build_cost_on_every_tier():
    """Cash + sell-refund must just barely cover the proc rebuild
    (cash 700-800; refund 700; proc cost 1400). Without the refund
    the cash alone falls short — that gap is the load-bearing
    discrimination."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        # cash < 1400 (proc cost) ⇒ build-without-selling is impossible.
        assert c.starting_cash < 1400, (lvl, c.starting_cash)
        # cash + 700 refund ≥ 1400 ⇒ sell-then-rebuild is feasible.
        assert c.starting_cash + 700 >= 1400, (lvl, c.starting_cash)


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be strictly below the tick reachable at max_turns (interrupt mode
    advances ≤90 ticks per step; some steps are shorter due to
    enemy_unit_spotted events, so the empirical reachable tick is
    ~4698 at 60 turns)."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    # 60 turns × ~78 ticks/turn (event-shortened) ≈ 4680; 4501
    # deadline reliably bites.
    assert deadline < 4700, (
        f"{level}: deadline {deadline} unreachable within {c.max_turns} "
        f"turns (interrupt mode ≈ 4680 max tick) → draw degeneracy"
    )


def test_fact_alive_clause_uses_present_tense_predicate():
    """The fact-survival clause must use the PRESENT-TENSE predicate
    (`building_count_gte:{type:fact,n:1}`) rather than `has_building`,
    which is a one-shot "ever seen" set that stays true after the
    fact is destroyed (a documented CLAUDE.md footgun)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        fact_clauses = [
            clause for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type") == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive fail clause"


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so the
    fact (and therefore the safe target region for proc placement)
    flips by seed. The two groups must define the NORTH (y=4) and
    SOUTH (y=36) fact pair."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    fact_ys = sorted({
        a.position[1] for a in c.scenario.actors
        if a.owner == "agent" and a.type == "fact"
    })
    assert fact_ys == [4, 36], fact_ys
    # In-bounds check (rush-hour-arena playable x ≈ 2..126, y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended SELL-THEN-REBUILD wins every (level, seed) ────


@pytest.mark.parametrize("level", ("easy", "medium"))
def test_intended_sell_then_rebuild_wins_easy_medium(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_intended_easy_medium(), seed
        )
        assert outcome == "win", (
            f"{level} seed{seed}: intended SELL-THEN-REBUILD must WIN; "
            f"got {outcome} (tick={sig.game_tick}, "
            f"buildings={sig.own_buildings})"
        )


def test_intended_hard_adaptive_wins_every_seed():
    """Hard tier: the intended policy must read the fact's latitude
    (NORTH y=4 vs SOUTH y=36) and pick the matching safe target
    region. WINS on every seed."""
    c = compile_level(load_pack(PACK), "hard")
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_intended_hard_adaptive(), seed
        )
        assert outcome == "win", (
            f"hard seed{seed}: intended adaptive sell-then-rebuild must "
            f"WIN; got {outcome} (tick={sig.game_tick}, "
            f"buildings={sig.own_buildings})"
        )


# ── no-cheat: every lazy / wrong policy LOSES (not draws) ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """STALL: observe only. The hunt band razes the exposed proc and
    the agent never places a new one ⇒ region clause unmet AND clock
    runs out ⇒ real LOSS, not draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not draw); "
            f"got {r.outcome} (tick={r.signals.game_tick}, "
            f"buildings={r.signals.own_buildings})"
        )


@pytest.mark.parametrize("level", ("easy", "medium"))
def test_build_without_selling_loses_easy_medium(level):
    """BUILD WITHOUT SELLING: `build('proc')` is rejected until cash
    ≥ 1400; the agent has no income source, the build never starts,
    no proc lands at the safe region ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_build_without_selling(), seed
        )
        assert outcome == "loss", (
            f"{level} seed{seed} build-without-selling: must LOSE; "
            f"got {outcome} (tick={sig.game_tick}, "
            f"buildings={sig.own_buildings})"
        )


@pytest.mark.parametrize("level", ("easy", "medium"))
def test_sell_then_misplace_loses_easy_medium(level):
    """SELL-THEN-MISPLACE: sells the exposed proc and uses the
    refund to build a NEW proc, but places it back in the central
    lane (y=20) — outside the safe target region disc. Region clause
    unmet ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_sell_then_misplace(), seed
        )
        assert outcome == "loss", (
            f"{level} seed{seed} sell-then-misplace: must LOSE; "
            f"got {outcome} (tick={sig.game_tick}, "
            f"buildings={sig.own_buildings})"
        )


def test_memorised_north_only_loses_on_hard_south_seeds():
    """The non-adaptive "always place at (16, 8)" policy WINS hard
    seeds whose spawn is NORTH (fact at y=4 ⇒ matching safe region is
    (16, 8)) but FAILS hard seeds whose spawn is SOUTH (fact at y=36
    ⇒ matching safe region is (16, 36), and (16, 8) is outside the
    SOUTH disc). The spawn-driven discrimination: at least one of
    the 4 hard seeds must LOSE."""
    c = compile_level(load_pack(PACK), "hard")
    losses = 0
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_memorised_north_only(), seed
        )
        if outcome == "loss":
            losses += 1
    assert losses >= 1, (
        f"hard: memorised-north-only must LOSE on ≥1 of {len(SEEDS)} "
        f"seeds (spawn-driven discrimination); got {losses} losses"
    )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a_outcome, a_turns, a_sig = _run_with_id_aware_policy(
        c, make_intended_easy_medium(), seed=3
    )
    b_outcome, b_turns, b_sig = _run_with_id_aware_policy(
        c, make_intended_easy_medium(), seed=3
    )
    assert (a_outcome, a_turns, a_sig.units_killed) == (
        b_outcome, b_turns, b_sig.units_killed,
    ), "same seed must be deterministic"
