"""econ-tech-vs-expand-decision — REASONING capability validation.

Capital allocation: one indivisible budget ($2000) funds EITHER

  (A) TECH   — build a SECOND war factory (weap, $2000): a
      non-revenue capability investment;
  (B) EXPAND — buy a SECOND ore harvester (harv, $1100): roughly
      doubles gathering throughput.

The regime is fixed (no near-term combat demand — the only enemy is
a token far-off passive marker) and the win is an ECONOMY target.
EXPAND clears the bar; sinking the budget into the non-revenue TECH
building leaves the economy short.

Bar (CLAUDE.md "no defect, no cheat, no draw"):

  * stall (observe-only) LOSES every tier / every hard seed — the
    economy idles at the starting reserve, far below the EV bar →
    the `after_ticks` deadline bites.
  * build-tech (build + place the war factory) LOSES every tier /
    seed — the $2000 sink craters the economy → the EV bar is
    never reached → real reachable timeout LOSS.
  * intended EXPAND (build('harv') early) WINS every tier / every
    hard seed — the doubled income compounds past the EV bar.
  * hard tier defines ≥2 agent spawn_point groups (NORTH / SOUTH
    base) round-robined by seed, each with a symmetric near-patch
    pair so a memorised opening cannot generalise.
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

PACK = PACKS_DIR / "econ-tech-vs-expand-decision.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Engine auto-harvest handles income: a harv adjacent to a proc with
# an ore patch nearby will autonomously mine without explicit harvest
# orders. Issuing explicit `harvest` commands at the same cell as a
# `mine` neutral building blocks pathing (the harv collides). Our
# policies therefore omit explicit harvest commands and let the
# engine's auto-harvest hook drive throughput.


def _stall(rs, C):
    """Observe-only — no income action. The economy idles at the
    starting reserve, far below the EV bar → LOSS."""
    return [C.observe()]


def _tech(rs, C):
    """Build the SECOND war factory — the non-revenue TECH option.
    The $2000 sink craters the economy → the EV bar is never
    reached → LOSS."""
    own = rs.get("own_buildings") or []
    nweap = sum(1 for b in own if b.get("type") == "weap")
    prod = [
        x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
    ]
    fy = 20
    for b in own:
        if b.get("type") == "fact":
            fy = b["cell_y"]
            break
    cmds = []
    if nweap < 2:
        if "weap" not in prod:
            cmds.append(C.build("weap"))
        cmds.append(C.place_building("weap", 24, fy))
    return cmds if cmds else [C.observe()]


def _expand(rs, C):
    """The intended capability — buy the SECOND harvester EARLY so
    the doubled gathering income compounds past the EV bar."""
    nharv = sum(
        1 for u in (rs.get("units_summary") or []) if u.get("type") == "harv"
    )
    prod = [
        x.get("item") for x in (rs.get("production") or []) if isinstance(x, dict)
    ]
    if nharv < 2 and "harv" not in prod:
        return [C.build("harv")]
    return [C.observe()]


# ── structural tests ────────────────────────────────────────────────


def test_pack_loads_and_meta_reasoning():
    pack = load_pack(PACK)
    assert pack.meta.id == "econ-tech-vs-expand-decision"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.real_world_meaning
    assert pack.meta.robotics_analogue
    anchors = " ".join(pack.meta.benchmark_anchor).lower()
    assert "sc2 tech-vs-expand" in anchors, anchors
    assert "planbench" in anchors and "resource-allocation" in anchors, anchors
    assert "capex allocation" in anchors, anchors


def test_starting_cash_is_the_either_or_pivot():
    """Budget $2000 — exactly one war factory (TECH) and enough for
    one harvester (EXPAND, $1100). A single indivisible allocation."""
    pack = load_pack(PACK)
    assert getattr(pack, "starting_cash", None) == 2000


def test_tools_include_build_and_harvest_surface():
    pack = load_pack(PACK)
    tools = set(pack.base.get("tools", []) if isinstance(pack.base, dict) else [])
    for required in ("build", "place_building", "harvest"):
        assert required in tools, f"missing tool: {required!r}"


def test_preplaced_pipeline_on_every_level():
    """Every level pre-places fact + proc + powr + weap + harv so the
    harvester buy (EXPAND) is actionable on turn 1 with no tech
    step."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_types = [a.type for a in c.scenario.actors if a.owner == "agent"]
        for needed in ("fact", "proc", "powr", "weap", "harv"):
            assert needed in agent_types, (
                f"{lvl}: pipeline missing {needed!r}; got {sorted(set(agent_types))}"
            )
        # The pre-placed harvester proves EXPAND means a SECOND harv.
        assert agent_types.count("harv") >= 1


def test_every_level_has_reachable_timeout_fail():
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
        assert wt + 1 == ft, f"{lvl}: within/after mismatch {wt}/{ft}"


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
    assert sp == {0, 1}, f"hard must define spawn_point groups {{0,1}}; got {sorted(sp)}"


def test_in_bounds_actors_on_every_level():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        for a in c.scenario.actors:
            x, y = a.position
            assert 2 <= x <= 126 and 2 <= y <= 38, (
                f"{lvl}: actor {a.type} at ({x},{y}) out of bounds"
            )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, tick=0, ev=0, has_fact=True, has_proc=True):
    import types

    own = []
    if has_fact:
        own.append(("fact", 8, 18))
    if has_proc:
        own.append(("proc", 12, 18))
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=ev,
        resources=0,
        own_buildings=own,
        own_building_types={t for (t, _, _) in own},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(signals=sig, render_state={"units_summary": []})


def test_predicates_enforce_economy_target():
    c = compile_level(load_pack(PACK), "easy")
    # EV at bar, fact + proc alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(tick=2000, ev=7000))
    # EV under bar → not win
    assert not evaluate(c.win_condition, _ctx(tick=2000, ev=6999))
    # EV at bar but past deadline → not win
    assert not evaluate(c.win_condition, _ctx(tick=2702, ev=8000))
    # Past after_ticks → fail
    assert evaluate(c.fail_condition, _ctx(tick=2701, ev=0))
    # fact destroyed → fail
    assert evaluate(c.fail_condition, _ctx(tick=1000, ev=6000, has_fact=False))
    # healthy in-window economy → not fail
    assert not evaluate(c.fail_condition, _ctx(tick=1000, ev=3000))


# ── engine-driven: every lazy/wrong policy LOSES, intended WINS ──────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses_every_tier_and_seed(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: stall must LOSE; got {r.outcome} "
        f"EV={r.signals.cash + r.signals.resources} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_build_tech_loses_every_tier_and_seed(level, seed):
    """Sinking the budget into the second war factory craters the
    economy → the EV bar is never reached → LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _tech, seed=seed)
    assert r.outcome == "loss", (
        f"{level}/seed{seed}: build-tech must LOSE; got {r.outcome} "
        f"EV={r.signals.cash + r.signals.resources} tick={r.signals.game_tick}"
    )


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_expand_wins_every_tier_and_seed(level, seed):
    """The intended capability — buy the second harvester early — the
    doubled income compounds past the EV bar. WINS every tier/seed."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _expand, seed=seed)
    assert r.outcome == "win", (
        f"{level}/seed{seed}: EXPAND must WIN; got {r.outcome} "
        f"EV={r.signals.cash + r.signals.resources} tick={r.signals.game_tick}"
    )


def test_expand_run_is_deterministic_per_seed():
    c = compile_level(load_pack(PACK), "medium")
    a = run_level(c, _expand, seed=2)
    b = run_level(c, _expand, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns)
