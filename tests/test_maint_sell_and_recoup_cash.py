"""maint-sell-and-recoup-cash pack — no-cheat validation on Rust.

Wave-10 capital reallocation pack. The pack tests DIVEST-THEN-BUY as a
reasoning primitive: the agent owns an obsolete static-defence cluster
(extra pillboxes / a Tesla coil / a radar dome) and the mission has
shifted to offence — it must field a War Factory (`weap`) and a batch
of 3 medium tanks (`2tnk`). Starting cash alone does NOT cover that
purchase and there is NO income source. The only path inside the tick
budget is:

    1. `sell([building_id, ...])`        ⇒ refunds 50% of build cost
    2. `build('weap')` + `place_building`
    3. `build('2tnk')` ×3

The win predicate makes the SELL load-bearing:

* `building_count_gte:{type:weap, n:1}`   ⇒ the war factory was built
  from recouped capital;
* `unit_type_count_gte:{type:2tnk, n:3}`  ⇒ the 3-tank batch is fielded;
* `building_count_gte:{type:fact, n:1}`   ⇒ the Construction Yard is
  still alive (the PRESENT-TENSE predicate, not the one-shot
  `has_building` set);
* `within_ticks: 7800` paired with `after_ticks: 7801` in fail ⇒ the
  episode end is a real reachable timeout LOSS, never a draw.

The scripted-policy validations prove deterministically that:

* the intended SELL-THEN-BUY policy WINS every (level, seed);
* stall (observe only), build-without-selling (cash gated), and
  sell-only (divest but never buy) all LOSE every (level, seed) —
  real reachable timeout LOSS, not draw;
* the hard tier defines ≥2 spawn_point groups (NORTH y=4 / SOUTH
  y=34) so the agent base latitude varies by seed.

There is no scripted adversary: the pack is a pure capital-
reallocation budget puzzle (the budget gate is the load-bearing
discrimination; the clock is the anti-stall teeth).

NOTE on building ids: `sell` requires the real engine actor id
(e.g. `1008`), which the bench's render_state strips. The scripted
policies below reach into `_raw["own_buildings"]` (via a small custom
episode loop) to look up the sellable building ids by type.
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

PACK = PACKS_DIR / "maint-sell-and-recoup-cash.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# The obsolete-defence buildings the agent must sell, per tier (their
# types — the ids are looked up live).
_SELLABLE = {
    "easy": ("pbox", "pbox", "tsla", "dome"),
    "medium": ("pbox", "pbox", "pbox", "tsla", "hbox"),
    "hard": ("pbox", "pbox", "pbox", "tsla", "dome"),
}


# ── custom episode loop with raw building-id access ──────────────────


def _run_with_id_aware_policy(compiled, policy, seed):
    """Run an episode where the policy is given (rs, raw, Command) and
    can read `raw["own_buildings"][i]["id"]` (the real engine actor
    id). Mirrors `run_level`'s win/fail/draw scoring.
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
            raw = adapter._raw
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


def _sellable_ids(raw, types):
    """Engine actor ids of own buildings whose type is in ``types``
    (a multiset — each entry consumes one matching building)."""
    pool = list(types)
    ids = []
    for b in (raw.get("own_buildings") or []):
        t = b.get("type")
        if t in pool:
            pool.remove(t)
            ids.append(str(b["id"]))
    return ids


def _count_units(rs, t):
    """Count own units of type ``t`` from render_state's
    `units_summary` (own units do NOT surface in `_raw`)."""
    return sum(
        1 for u in (rs.get("units_summary") or []) if u.get("type") == t
    )


def _has_building(raw, t):
    return any(
        b.get("type") == t for b in (raw.get("own_buildings") or [])
    )


def _fact_y(raw):
    """Latitude of the agent's fact (4 NORTH / 34 hard SOUTH)."""
    for b in (raw.get("own_buildings") or []):
        if b.get("type") == "fact":
            return int(b.get("cell_y", 4))
    return 4


# ── scripted policies ───────────────────────────────────────────────


def stall(rs, C):
    """Observe-only — never sells, never builds. LOSS."""
    return [C.observe()]


def make_build_without_selling():
    """Try to BUILD the war factory + tanks WITHOUT selling anything.
    Cash starts at 2800-3500 — below the ~4550 weap+tank cost — and
    there is no income source, so the war factory build never starts
    (or, if weap is afforded, the 3-tank batch is not). LOSS.
    """

    def policy(rs, raw, C):
        cmds = []
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        if not _has_building(raw, "weap"):
            if "weap" not in prod_items:
                cmds.append(C.build("weap"))
            cmds.append(C.place_building("weap", 24, 6))
        else:
            if "2tnk" not in prod_items and _count_units(rs, "2tnk") < 3:
                cmds.append(C.build("2tnk"))
        return cmds or [C.observe()]

    return policy


def make_sell_only(level):
    """SELL the obsolete defences (recoups the capital) but NEVER
    build the war factory or the tanks. Win clauses unmet ⇒ LOSS.
    """
    state = {"sold": False}

    def policy(rs, raw, C):
        if not state["sold"]:
            ids = _sellable_ids(raw, _SELLABLE[level])
            if ids:
                state["sold"] = True
                return [C.sell(ids)]
        return [C.observe()]

    return policy


def make_intended(level, weap_x=24, weap_y=6):
    """Intended DIVEST-THEN-BUY play. Turn 1: sell every obsolete
    defence building (recoup the capital). Then: build + place the war
    factory; once it stands, build 3 medium tanks one at a time.
    The hard tier reads the fact's latitude so the war factory lands
    on the active spawn's latitude.
    """
    state = {"sold": False, "wxy": None}

    def policy(rs, raw, C):
        if state["wxy"] is None:
            fy = _fact_y(raw)
            # Place the war factory near the core base of the active
            # latitude (NW for NORTH, SW for SOUTH).
            state["wxy"] = (weap_x, 6 if fy < 20 else 32)
        wx, wy = state["wxy"]
        cmds = []
        # PHASE 1: divest the obsolete defences (one batched sell).
        if not state["sold"]:
            ids = _sellable_ids(raw, _SELLABLE[level])
            if ids:
                cmds.append(C.sell(ids))
                state["sold"] = True
                return cmds
        prod_items = [
            (p.get("item") if isinstance(p, dict) else p)
            for p in (rs.get("production") or [])
        ]
        # PHASE 2: build the war factory.
        if not _has_building(raw, "weap"):
            if "weap" not in prod_items:
                cmds.append(C.build("weap"))
            cmds.append(C.place_building("weap", wx, wy))
            return cmds or [C.observe()]
        # PHASE 3: produce 3 medium tanks (one at a time).
        if _count_units(rs, "2tnk") < 3:
            if "2tnk" not in prod_items:
                cmds.append(C.build("2tnk"))
        return cmds or [C.observe()]

    return policy


# ── scenario-shape invariants ───────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "maint-sell-and-recoup-cash"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    anchors = [a.lower() for a in pack.meta.benchmark_anchor]
    assert any("capital reallocation" in a for a in anchors), anchors
    assert any("asset divestment" in a for a in anchors), anchors
    assert any("sc2 sell mechanic" in a for a in anchors), anchors
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported


def test_sell_and_build_are_in_the_tool_palette():
    """`sell` is the load-bearing divestment verb; `build` /
    `place_building` are the critical-purchase primitive."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        tools = set(getattr(c.scenario, "tools", None) or [])
        assert "sell" in tools, (lvl, tools)
        assert "build" in tools, (lvl, tools)
        assert "place_building" in tools, (lvl, tools)


def test_starting_cash_is_below_the_critical_purchase_cost():
    """Cash alone must NOT cover the weap (2000) + 3×2tnk (~2550)
    purchase — that gap is the load-bearing discrimination. Cash plus
    the obsolete-defence refunds must clear it."""
    pack = load_pack(PACK)
    # 50%-of-cost refunds probed live on the engine.
    refund = {"pbox": 300, "tsla": 600, "dome": 750, "hbox": 375}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        # weap 2000 + 3 tanks ≈ 2550 ⇒ ~4550 critical-purchase cost.
        assert c.starting_cash < 4550, (lvl, c.starting_cash)
        total_refund = sum(refund[t] for t in _SELLABLE[lvl])
        assert c.starting_cash + total_refund >= 4550, (
            lvl, c.starting_cash, total_refund
        )


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS: the `after_ticks` fail clause must
    be reachable within max_turns. Interrupt mode advances ≤90 ticks
    per step; max_turns 90 ⇒ reachable tick well above 7801."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    deadline = None
    for clause in fc.get("any_of", []) or []:
        if "after_ticks" in clause:
            deadline = int(clause["after_ticks"])
    assert deadline is not None, f"{level}: no after_ticks fail clause"
    assert deadline <= 93 + 90 * (c.max_turns - 1), (
        f"{level}: deadline {deadline} unreachable within {c.max_turns} "
        f"turns → draw degeneracy"
    )


def test_fact_alive_clause_uses_present_tense_predicate():
    """The fact-survival fail clause must use the PRESENT-TENSE
    `building_count_gte:{type:fact}` (not the one-shot `has_building`,
    which stays true after the fact is destroyed)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        fc = c.fail_condition.model_dump(exclude_none=True)
        fact_clauses = [
            clause for clause in fc.get("any_of", []) or []
            if isinstance(clause, dict)
            and isinstance(clause.get("not"), dict)
            and "building_count_gte" in (clause["not"] or {})
            and (clause["not"]["building_count_gte"] or {}).get("type")
            == "fact"
        ]
        assert fact_clauses, f"{lvl}: missing present-tense fact-alive clause"


def test_win_requires_weap_and_three_tanks():
    """The win predicate must require BOTH the war factory and the
    3-tank batch (the critical purchase) — neither alone wins."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        wc = c.win_condition.model_dump(exclude_none=True)
        clauses = wc.get("all_of", []) or []
        has_weap = any(
            (cl.get("building_count_gte") or {}).get("type") == "weap"
            for cl in clauses
        )
        has_tanks = any(
            (cl.get("unit_type_count_gte") or {}).get("type") == "2tnk"
            and (cl.get("unit_type_count_gte") or {}).get("n") == 3
            for cl in clauses
        )
        assert has_weap, f"{lvl}: win must require a war factory"
        assert has_tanks, f"{lvl}: win must require 3 medium tanks"


def test_hard_has_two_spawn_point_groups_and_fact_flips():
    """Hard-tier contract: ≥2 distinct agent spawn_point groups so the
    base latitude (NORTH y=4 / SOUTH y=34) flips by seed."""
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
    assert fact_ys == [4, 34], fact_ys
    # In-bounds (rush-hour-arena playable x ≈ 2..126, y ≈ 2..38).
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


# ── solvency: intended DIVEST-THEN-BUY wins every (level, seed) ──────


@pytest.mark.parametrize("level", LEVELS)
def test_intended_sell_then_buy_wins_every_level_and_seed(level):
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_intended(level), seed
        )
        assert outcome == "win", (
            f"{level} seed{seed}: intended DIVEST-THEN-BUY must WIN; "
            f"got {outcome} (tick={sig.game_tick}, "
            f"buildings={sig.own_buildings})"
        )


# ── no-cheat: every lazy / wrong policy LOSES (not draws) ────────────


@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses_every_level_and_seed(level):
    """STALL: observe only. Never sells, never builds ⇒ win clauses
    unmet AND the clock runs out ⇒ real LOSS, not draw."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        r = run_level(c, stall, seed=seed)
        assert r.outcome == "loss", (
            f"{level} seed{seed} stall: must LOSE (real fail, not "
            f"draw); got {r.outcome} (tick={r.signals.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_build_without_selling_loses_every_level_and_seed(level):
    """BUILD WITHOUT SELLING: cash (2800-3500) is below the ~4550
    weap+tank cost and there is no income ⇒ the purchase can never be
    funded ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_build_without_selling(), seed
        )
        assert outcome == "loss", (
            f"{level} seed{seed} build-without-selling: must LOSE; "
            f"got {outcome} (tick={sig.game_tick})"
        )


@pytest.mark.parametrize("level", LEVELS)
def test_sell_only_loses_every_level_and_seed(level):
    """SELL-ONLY: divests the obsolete defences but never buys the war
    factory or the tanks ⇒ win clauses unmet ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    for seed in SEEDS:
        outcome, turns, sig = _run_with_id_aware_policy(
            c, make_sell_only(level), seed
        )
        assert outcome == "loss", (
            f"{level} seed{seed} sell-only: must LOSE; "
            f"got {outcome} (tick={sig.game_tick})"
        )


# ── determinism ──────────────────────────────────────────────────────


def test_intended_run_is_deterministic_on_easy():
    c = compile_level(load_pack(PACK), "easy")
    a_outcome, a_turns, a_sig = _run_with_id_aware_policy(
        c, make_intended("easy"), seed=3
    )
    b_outcome, b_turns, b_sig = _run_with_id_aware_policy(
        c, make_intended("easy"), seed=3
    )
    assert (a_outcome, a_turns) == (b_outcome, b_turns), (
        "same seed must be deterministic"
    )
