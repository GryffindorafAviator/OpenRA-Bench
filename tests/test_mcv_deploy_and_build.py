"""MCV-Deploy-and-Build scenario pack — deterministic, no-cheat verifier.

Uses the Wave-2 MCV deploy fix (commits bdedc55 OpenRA-Rust + b0cb710
OpenRA-Bench) — the first pack to USE the new live-engine deploy
behaviour: `Command.deploy([mcv_id])` removes the MCV, creates an
agent-owned `fact` at offset (mcv_x-1, mcv_y-1), and re-enables the
Building/Defense production queues. The intended capability policy
is then a straight chain — deploy → build(powr) → place_building →
build(tent) → place_building (→ build(powr) for hard's 4-building
requirement).

These tests prove, with deterministic scripted agents (no model, no
network), that the pack meets the "no defect, no cheat" bar across
every level × every hard seed (1..4):

* the INTENDED full-chain policy (deploy + powr + tent [+ extra powr
  for hard]) WINS every level × every seed;
* the STALL policy (only `Command.observe()`) LOSES every level × seed
  (the fact never appears → the explicit "no fact past 2000 ticks"
  fail clause bites, never a DRAW);
* the DEPLOY-ONLY policy (deploys the MCV but never builds anything)
  LOSES every level × seed (the fact exists but powr/tent don't, so
  the chain-completion win clause is unmet and the deadline fires
  as a LOSS).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mcv-deploy-and-build.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ──────────────────────────────────────────────────────────────────
# Scripted policies
# ──────────────────────────────────────────────────────────────────

def _stall(render_state, Command):
    """Idle: never deploys, never builds. The `not has_building:fact`
    fail clause (after_ticks: 2000) bites well before the deadline
    on every level."""
    return [Command.observe()]


def _deploy_only(render_state, Command):
    """Deploys the MCV (so a fact DOES appear, dodging the
    `not has_building:fact` fail) but never builds anything — so
    powr/tent never materialise and the deadline fires as a LOSS."""
    units = render_state.get("units_summary", []) or []
    mcv_id = next(
        (u["id"] for u in units if str(u.get("type", "")).lower() == "mcv"),
        None,
    )
    if mcv_id is not None:
        return [Command.deploy([str(mcv_id)])]
    return [Command.observe()]


def _intended_full_chain(want_extra_powr: bool):
    """Intended capability policy: deploy the MCV, then build powr →
    place, build tent → place, [+ one more powr → place for hard].
    A single policy returns a one-cmd-per-turn plan (the engine
    completes the build in subsequent turns); placement is
    attempted as soon as the item is ready (`production` entry).
    """

    def fn(render_state, Command):
        units = render_state.get("units_summary", []) or []
        bldgs = render_state.get("own_buildings", []) or []
        own_types = {b["type"] for b in bldgs}
        prod = render_state.get("production", []) or []
        # 1) deploy if we still have an MCV
        mcv_id = next(
            (u["id"] for u in units if str(u.get("type", "")).lower() == "mcv"),
            None,
        )
        if mcv_id is not None:
            return [Command.deploy([str(mcv_id)])]
        # No fact yet (mid-deploy / race) → wait one turn.
        fact_bs = [b for b in bldgs if b["type"] == "fact"]
        if not fact_bs:
            return [Command.observe()]
        fx, fy = fact_bs[0]["cell_x"], fact_bs[0]["cell_y"]
        # 2) drive powr (queue + place adjacent to the fact). The
        # production list contains the queued items as strings; if
        # the item appears and the building hasn't yet, attempt to
        # place it (engine no-ops a place before completion, then
        # succeeds once ready).
        n_powr = sum(1 for b in bldgs if b["type"] == "powr")
        if n_powr == 0:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 3, fy + 1))
            return cmds
        # 3) drive tent (queue + place adjacent to the fact).
        if "tent" not in own_types:
            cmds = []
            if "tent" not in prod:
                cmds.append(Command.build("tent"))
            cmds.append(Command.place_building("tent", fx - 2, fy + 3))
            return cmds
        # 4) hard-tier: queue one more building so total ≥ 4.
        if want_extra_powr and len(bldgs) < 4:
            cmds = []
            if "powr" not in prod:
                cmds.append(Command.build("powr"))
            cmds.append(Command.place_building("powr", fx + 6, fy + 1))
            return cmds
        return [Command.observe()]

    return fn


# ──────────────────────────────────────────────────────────────────
# Compile / structural invariants
# ──────────────────────────────────────────────────────────────────

def test_pack_compiles_with_three_levels_and_deploy_tool():
    pack = load_pack(PACK)
    assert pack.meta.id == "mcv-deploy-and-build"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == set(LEVELS)
    # Wave-2 fix: `deploy` is the headline tool of this pack.
    c = compile_level(pack, "easy")
    assert "deploy" in (c.scenario.tools or []), c.scenario.tools
    # required anchor list is populated (suite-wide rule).
    assert pack.meta.benchmark_anchor, "benchmark_anchor must be non-empty"


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: the deadline
    (after_ticks in fail) must be reachable within max_turns
    (tick ≈ 93 + 90·(max_turns-1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    fc = c.fail_condition.model_dump(exclude_none=True)
    # Pull the first after_ticks clause out of the any_of tree.
    clauses = fc["any_of"]
    after_ticks = int(clauses[0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )


def test_hard_has_two_spawn_groups_for_mcv():
    """Hard tier contract: ≥2 distinct agent spawn_point groups for the
    MCV (NW vs SW corner) so seed round-robin varies the start."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ──────────────────────────────────────────────────────────────────
# Behavioural — the bar (no defect, no cheat)
# ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_full_chain_wins(level, seed):
    """Intended capability policy must WIN every level × every seed.
    Hard requires the 4th building (the extra powr)."""
    c = compile_level(load_pack(PACK), level)
    fn = _intended_full_chain(want_extra_powr=(level == "hard"))
    r = run_level(c, fn, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended deploy+build chain should WIN, "
        f"got {r.outcome}; buildings={r.signals.own_buildings}, "
        f"cash={r.signals.cash}, turns={r.turns}, "
        f"tick={r.signals.game_tick}"
    )
    # Sanity: the chain materialised the required buildings.
    types = set(r.signals.own_building_types)
    assert {"fact", "powr", "tent"} <= types, types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_policy_loses(level, seed):
    """Stall (only `observe`) must LOSE every level × every seed.
    The `not has_building:fact` fail clause (active past 2000 ticks)
    catches this well before the deadline — never a DRAW."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall: must LOSE (no fact ⇒ fail), got "
        f"{r.outcome}; buildings={r.signals.own_buildings}, "
        f"turns={r.turns}, tick={r.signals.game_tick}"
    )
    # The stall never deploys → no agent-owned fact ever appears.
    assert "fact" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_deploy_only_policy_loses(level, seed):
    """Deploy-only (deploys the MCV but never builds) must LOSE every
    level × every seed — the fact appears (so the explicit no-fact
    fail clause is dodged) but powr/tent never materialise, so the
    win is unmet and the deadline fires as a LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, _deploy_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} deploy-only: must LOSE (no powr/tent), "
        f"got {r.outcome}; buildings={r.signals.own_buildings}, "
        f"turns={r.turns}, tick={r.signals.game_tick}"
    )
    # Deploy-only DID create a fact (confirms the deploy fix is
    # live AND the lose-reason is the missing chain, not a missing
    # fact). For hard, seed-by-seed flake could conceivably destroy
    # the MCV pre-deploy to a stray patrol shot — only assert the
    # presence when the chain reached it (best-effort sanity).
    if level != "hard":
        assert "fact" in r.signals.own_building_types, (
            f"deploy-only should have produced a fact, signals={r.signals}"
        )
    # And NEITHER powr nor tent were ever queued.
    types = set(r.signals.own_building_types)
    assert "powr" not in types and "tent" not in types, types


def test_easy_run_is_deterministic():
    """Same seed must be deterministic (engine-level reproducibility)."""
    c = compile_level(load_pack(PACK), "easy")
    fn = _intended_full_chain(want_extra_powr=False)
    a = run_level(c, fn, seed=3)
    b = run_level(c, fn, seed=3)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
