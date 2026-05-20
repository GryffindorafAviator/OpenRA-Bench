"""econ-buy-vs-build-decision — REASONING capability validation.

CAPEX-vs-OPEX capital allocation under rush pressure. Starting cash
($3400) funds EITHER:

  (A) BUY 4× medium tanks (2tnk @ $850) NOW from the pre-placed war
      factory — OPEX, immediate combat capability;
  (B) BUILD a SECOND war factory (weap @ $2000) — CAPEX, doubles
      long-run tank throughput but leaves only $1400 ≈ 1 tank.

A `rusher` band is staged at the lane mouth (x=70). The right call is
(A): queue 4 tanks immediately from the existing weap; they auto-fire
on the incoming rush before the fact dies. Spending on a second
factory leaves the base undefended → fact razed → LOSS.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed —
    the rusher band over-runs the undefended fact AND/OR the
    `after_ticks` deadline bites.
  * build-weap-first (queue + place weap, drain $2000, then buy
    1 tank from residual cash) LOSES every tier / every hard seed —
    cash drain + queue lockout ⇒ the first tank fields too late to
    blunt the rush ⇒ fact razed.
  * intended buy-now (queue 2tnk × 4 from the pre-placed weap, let
    them auto-fire from the base position) WINS every tier / every
    hard seed: the kill bar is met AND the fact survives AND the
    rush is over before the `within_ticks` deadline.
  * hard tier defines ≥2 agent spawn_point groups (NORTH base
    y=14 / SOUTH base y=26) round-robined by seed — the threat
    band at each latitude always places (enemy actors don't honour
    spawn_point — CLAUDE.md) but the `rusher` bot targets the
    agent centroid, so the ACTIVE threat axis flips per seed and a
    memorised "rush from y=14" opening cannot generalise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK = PACKS_DIR / "econ-buy-vs-build-decision.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scripted policies ───────────────────────────────────────────────


def _stall(rs, C):
    """Observe-only — no spend, no production. Rusher reaches the
    fact AND/OR the after_ticks deadline bites → LOSS."""
    return [C.observe()]


def _buy_now(rs, C):
    """The intended capability: spam queue 2tnk from the pre-placed
    war factory. Tanks auto-fire on the rusher band as they emerge
    from the production cell — no need to micro-move them out. 4
    tanks at rng4.75 dps22 each (88 dps total) clear the rush
    inside the kill bar before the fact dies."""
    return [C.build("2tnk")]


def _build_weap_first(rs, C):
    """The wrong call: spend $2000 + a placement turn on a SECOND
    war factory instead of pumping units from the existing one.
    Residual cash ($1400) only buys ~1 tank, and the new weap
    fields its first unit well after the rusher has reached the
    fact ⇒ fact razed ⇒ LOSS."""
    own_b = rs.get("own_buildings") or []
    weap_count = sum(1 for b in own_b if b.get("type") == "weap")
    prod = rs.get("production") or []
    prod_items = [p.get("item") for p in prod if isinstance(p, dict)]
    # Find the current fact latitude (varies on hard by seed)
    fy = 20
    for b in own_b:
        if b.get("type") == "fact":
            fy = int(b["cell_y"])
            break
    cmds = []
    if weap_count < 2:
        if "weap" not in prod_items:
            cmds.append(C.build("weap"))
        cmds.append(C.place_building("weap", 22, fy))
    else:
        cmds.append(C.build("2tnk"))
    if not cmds:
        cmds.append(C.observe())
    return cmds


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_active():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-buy-vs-build-decision"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "planbench" in anchors and "resource-allocation" in anchors, anchors
    assert "capex" in anchors and "opex" in anchors, anchors
    assert "financial allocation" in anchors, anchors


def test_pack_uses_rusher_bot_on_every_level():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported, f"{lvl}: rush-hour-arena terrain required"
        enemy = c.scenario.enemy
        bot = getattr(enemy, "bot_type", None) or getattr(enemy, "bot", None)
        assert str(bot).lower() == "rusher", f"{lvl}: enemy bot must be 'rusher'; got {bot}"


def test_tools_include_buy_and_build_surface():
    """Pack must expose [build, place_building, move_units,
    attack_unit, attack_move, stop] — the buy-vs-build decision
    interaction surface."""
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("build", "place_building", "move_units",
                     "attack_unit", "attack_move", "stop"):
        assert required in tools, f"missing tool: {required!r}"


def test_starting_cash_equals_four_tanks_or_one_weap():
    """Cash budget must be the EITHER/OR pivot: exactly 4× 2tnk
    cost ($3400) — also enough for 1× weap ($2000) with $1400
    residual = ~1 tank. Either decision is a single coherent
    capital allocation."""
    pack = load_pack(PACK)
    cash = getattr(pack, "starting_cash", None)
    assert cash == 3400, f"starting_cash must be 3400 (4× $850); got {cash}"


def test_preplaced_base_has_fact_proc_powr_weap_fix():
    """Pre-placed base on every level: fact (loss-critical), proc
    (weap prereq), powr (queue power), weap (war factory), fix
    (service depot — 2tnk allied tech gate). The buy-now option
    must be ACTIONABLE on turn 1 with no prior tech step."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = {
            a.type for a in c.scenario.actors if a.owner == "agent"
        }
        for needed in ("fact", "proc", "powr", "weap", "fix"):
            assert needed in agent_types, (
                f"{lvl}: pre-placed base missing {needed!r}; got {sorted(agent_types)}"
            )


def test_every_level_has_reachable_timeout_fail():
    """`after_ticks` fail must bite WITHIN max_turns (so stall is a
    real reachable LOSS, not a draw). within_ticks + 1 == after_ticks
    so a non-finisher is a LOSS, not a draw at the boundary."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        wt = next(
            int(c["within_ticks"])
            for c in L.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in L.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt < ceiling, f"{lvl}: within_ticks {wt} >= ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            "(boundary non-finisher must LOSE, not draw — fail one "
            "tick past win)"
        )


def test_every_level_has_a_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} needs a fail_condition"


def test_hard_has_two_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, (
        f"hard must define ≥2 agent spawn_point groups; got {sorted(sp)}"
    )
    assert sp == {0, 1}, f"expected exactly {{0, 1}}; got {sorted(sp)}"


def test_in_bounds_actors_on_every_level():
    """rush-hour-arena playable bounds ≈ x:2..126, y:2..38."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, kills=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=0,
        cash=0,
        resources=0,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_capability():
    """Win requires (kills bar AND fact alive AND in time); fail
    fires on timeout OR fact destroyed."""
    c = compile_level(load_pack(PACK), "easy")
    base_b = [("fact", 10, 20), ("weap", 18, 20)]

    # Intended: kills 3, fact alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(tick=1000, kills=3, own_buildings=base_b))
    # 2 kills (under bar) → not win
    assert not evaluate(
        c.win_condition, _ctx(tick=1000, kills=2, own_buildings=base_b)
    )
    # 3 kills but past within_ticks → not win
    assert not evaluate(
        c.win_condition, _ctx(tick=1499 + 1, kills=3, own_buildings=base_b)
    )
    # 3 kills but fact destroyed → not win
    assert not evaluate(
        c.win_condition, _ctx(tick=1000, kills=3, own_buildings=base_b[1:])
    )
    # Fact destroyed → fail
    assert evaluate(
        c.fail_condition, _ctx(tick=1000, kills=3, own_buildings=base_b[1:])
    )
    # Past after_ticks deadline → fail
    assert evaluate(
        c.fail_condition, _ctx(tick=1600, kills=0, own_buildings=base_b)
    )
    # Within deadline, fact alive → not fail
    assert not evaluate(
        c.fail_condition, _ctx(tick=1000, kills=0, own_buildings=base_b)
    )


# ── engine-driven: every lazy / wrong policy LOSES, intended WINS ───


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    """Observe-only ⇒ no kills + rush reaches fact ⇒ real LOSS,
    not a draw. The `after_ticks` clause is reachable inside
    max_turns; the `building_count_gte:fact` clause fires when
    the rush razes the fact."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE (no defenders, "
        f"fact razed or deadline bites); got {r.outcome} "
        f"tick={r.signals.game_tick} kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_weap_first_loses_every_tier_and_seed(level, seed):
    """Build second war factory first ⇒ $2000 sink + placement
    turn + queue lockout ⇒ first tank fields too late ⇒ rush
    reaches the fact ⇒ LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _build_weap_first, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: build-weap-first must LOSE (cash drain "
        f"+ queue lockout ⇒ fact razed); got {r.outcome} "
        f"tick={r.signals.game_tick} kills={r.signals.units_killed}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_buy_now_wins_every_tier_and_seed(level, seed):
    """The intended capability — queue 2tnk × 4 from the pre-placed
    war factory and let them auto-fire on the rush. Wins every
    tier and every hard seed, well inside the deadline."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _buy_now, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: buy-now (queue 2tnk × 4) must WIN; "
        f"got {r.outcome} tick={r.signals.game_tick} "
        f"kills={r.signals.units_killed}"
    )


# ── determinism ─────────────────────────────────────────────────────


def test_buy_now_run_is_deterministic_per_seed():
    """Same seed, same policy → identical outcome / kills / turns."""
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _buy_now, seed=2)
    b = run_level(c, _buy_now, seed=2)
    assert (a.outcome, a.turns, a.signals.units_killed) == (
        b.outcome, b.turns, b.signals.units_killed
    )
