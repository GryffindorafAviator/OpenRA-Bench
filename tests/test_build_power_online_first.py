"""No-cheat + solvency proof for `build-power-online-first` (Wave-7
Group I REASONING: opening-phase build-order — power-grid bring-up
sequencing).

The pack tests the SOP-compliance opening: from a pre-placed
Construction Yard and a tight cash budget, the agent must build the
Power Plant (`powr`) FIRST and only THEN the Refinery (`proc`). The
order is enforced both by the `then:[A,B]` composite (Wave-2 stateful
happened-before latch) AND by the engine itself (powr is a hard
prerequisite of proc/tent/barr; weap needs proc).

For every level + every hard seed (1-4):
  * the INTENDED order (build powr → place powr → build proc → place
    proc) WINS;
  * STALL (only `observe`) LOSES — never builds anything → win
    clauses fail, deadline bites as a real timeout LOSS;
  * PROC-FIRST (always tries `build('proc')` first, never powr) LOSES
    — the engine silently rejects StartProduction for proc without
    powr; no production building ever appears; deadline bites;
  * ARMY-SPAM (only `build('e1')`, never builds any structure) LOSES
    — no barracks, no powr, no proc; deadline bites.

The 3 lazy plays + 1 intended × 3 levels × 4 seeds gives the full
no-defect / no-cheat coverage demanded by CLAUDE.md.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-power-online-first.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ───────────────────────── scripted policies ──────────────────────────────


def _stall(rs, Command):
    """Idle: never builds anything → win clauses fail → LOSS on the
    deadline."""
    return [Command.observe()]


def _proc_first(rs, Command):
    """Always tries `build('proc')` first; NEVER tries powr. The engine
    silently rejects StartProduction for proc without powr (CLAUDE.md
    engine fact #2; world.rs:768 'ORDER: StartProduction BLOCKED —
    missing prerequisites for proc'). No proc ever appears → clause B
    of the `then` latch never satisfies → LOSS on the deadline. This
    is the headline wrong-order discriminator."""

    def _fact(rs):
        return next(
            (b for b in (rs.get("own_buildings", []) or []) if b["type"] == "fact"),
            None,
        )

    fact_b = _fact(rs)
    if fact_b is None:
        return [Command.observe()]
    fx, fy = fact_b["cell_x"], fact_b["cell_y"]
    prod = rs.get("production", []) or []
    own_types = {
        b["type"] for b in (rs.get("own_buildings", []) or [])
    }
    cmds = []
    if "proc" not in prod and "proc" not in own_types:
        cmds.append(Command.build("proc"))
    cmds.append(Command.place_building("proc", fx + 3, fy + 3))
    return cmds


def _army_spam(rs, Command):
    """Brute army-spam: only `build('e1')`. e1 needs a tent (barracks),
    which needs powr — neither exists, so every `build('e1')` is
    silently blocked. No powr / no proc / no barracks ever appear →
    LOSS on the deadline."""
    bldgs = rs.get("own_buildings", []) or []
    fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
    if fact_b is None:
        return [Command.observe()]
    return [Command.build("e1")]


def _intended(rs, Command):
    """Intended SOP: build powr FIRST (cost 300), place it adjacent to
    the fact; then build proc (cost 1400, now unblocked), place it
    adjacent to the fact. Wins every tier × every seed before the
    deadline."""
    bldgs = rs.get("own_buildings", []) or []
    own_types = {b["type"] for b in bldgs}
    prod = rs.get("production", []) or []
    fact_b = next((b for b in bldgs if b["type"] == "fact"), None)
    if fact_b is None:
        return [Command.observe()]
    fx, fy = fact_b["cell_x"], fact_b["cell_y"]
    if "powr" not in own_types:
        cmds = []
        if "powr" not in prod:
            cmds.append(Command.build("powr"))
        cmds.append(Command.place_building("powr", fx + 3, fy + 1))
        return cmds
    if "proc" not in own_types:
        cmds = []
        if "proc" not in prod:
            cmds.append(Command.build("proc"))
        cmds.append(Command.place_building("proc", fx + 3, fy + 3))
        return cmds
    return [Command.observe()]


# ───────────────────────── helpers ────────────────────────────────────────


def _run(level, policy, seed=1):
    c = compile_level(load_pack(PACK), level)
    assert c.map_supported, "rush-hour-arena must compile"
    return c, run_level(c, policy, seed=seed)


# ───────────────────────── structural ─────────────────────────────────────


def test_pack_loads_with_three_levels_and_required_tools():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-power-online-first"
    assert pack.meta.capability == "reasoning"
    assert pack.meta.status == "active"
    assert set(pack.levels) == set(LEVELS)
    c = compile_level(pack, "easy")
    tools = set(c.scenario.tools or [])
    # Wave-7 spec: pure build-order test — no deploy / harvest /
    # move_units / attack_unit. Just observe + build + place + stop.
    for t in ("observe", "build", "place_building", "stop"):
        assert t in tools, f"missing tool {t} in {tools}"
    for forbidden in ("move_units", "attack_unit", "deploy", "harvest"):
        assert forbidden not in tools, (
            f"tool {forbidden!r} must NOT be in the SOP-only palette"
        )


def test_benchmark_anchor_lists_planbench_and_grid_bringup():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor
    assert anchors, "benchmark_anchor must be non-empty"
    blob = " | ".join(anchors).lower()
    assert "planbench" in blob, anchors
    assert "sop" in blob, anchors
    assert "electrical" in blob and "grid" in blob, anchors


@pytest.mark.parametrize("level", LEVELS)
def test_every_level_has_a_reachable_timeout_fail(level):
    """Non-win must be a real LOSS, never a DRAW: after_ticks in
    fail_condition must be reachable within max_turns (tick ≈ 93 +
    90·(max_turns − 1))."""
    c = compile_level(load_pack(PACK), level)
    assert c.fail_condition is not None
    after_ticks = int(c.fail_condition.model_dump()["any_of"][0]["after_ticks"])
    reachable = 93 + 90 * (c.max_turns - 1)
    assert after_ticks <= reachable, (
        f"{level}: fail after_ticks {after_ticks} unreachable within "
        f"{c.max_turns} turns (max tick {reachable}) → draw degeneracy"
    )
    # within_ticks + 1 == after_ticks (non-finisher LOSES on the very
    # next tick after the win window closes).
    within_clauses = c.win_condition.model_dump().get("all_of", [])
    wt = next(
        int(x["within_ticks"]) for x in within_clauses if "within_ticks" in x
    )
    assert after_ticks == wt + 1, (
        f"{level}: after_ticks {after_ticks} must equal within_ticks+1 ({wt+1})"
    )


@pytest.mark.parametrize("level", LEVELS)
def test_win_uses_then_composite_powr_before_proc(level):
    """Structural: the win clause uses a `then` composite with two
    clauses in the exact order [powr, proc] — this is the engine-side
    SOP-compliance enforcement (CLAUDE.md ordering idiom)."""
    c = compile_level(load_pack(PACK), level)
    win = c.win_condition.model_dump()
    all_of = win.get("all_of", [])
    then_node = next(
        (x["then"] for x in all_of if "then" in x), None
    )
    assert then_node is not None, f"{level}: win must include a `then` composite"
    clauses = then_node.get("clauses", [])
    assert len(clauses) == 2, f"{level}: then must have 2 clauses (powr, proc)"
    # First clause: has_building powr; second: building_count_gte proc>=1.
    assert clauses[0].get("has_building") == "powr", clauses[0]
    bc = clauses[1].get("building_count_gte") or {}
    assert bc.get("type") == "proc" and int(bc.get("n", 0)) >= 1, clauses[1]


def test_hard_has_two_spawn_groups_for_fact():
    """Hard tier contract (CLAUDE.md + tests/test_hard_tier.py): ≥2
    distinct agent spawn_point groups (NORTH y≈12 / SOUTH y≈28) so
    seed round-robin varies the pre-placed fact and a memorised
    "(20,20)" opening can't generalise."""
    c = compile_level(load_pack(PACK), "hard")
    sps = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sps) >= 2, f"hard needs ≥2 spawn groups, got {sps}"


# ───────────────────────── intended WIN ───────────────────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_intended_powr_first_then_proc_wins(level, seed):
    """Intended build powr → place powr → build proc → place proc WINS
    every level × every seed before the deadline."""
    c, r = _run(level, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed{seed}: intended powr-first chain should WIN, "
        f"got {r.outcome}; types={r.signals.own_building_types}, "
        f"cash={r.signals.cash}, tick={r.signals.game_tick}"
    )
    types = set(r.signals.own_building_types)
    assert "powr" in types, types
    assert "proc" in types, types


# ───────────────────────── no-cheat: lazy plays LOSE ──────────────────────


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_stall_loses(level, seed):
    """Stall must LOSE every level × every seed — never builds anything
    → both then-clauses fail → reachable timeout LOSS (not a draw)."""
    c, r = _run(level, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} stall must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )
    # Never built powr or proc.
    assert "powr" not in r.signals.own_building_types
    assert "proc" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_proc_first_wrong_order_loses(level, seed):
    """The headline wrong-order discriminator: always tries `proc`
    first, never powr. Engine silently rejects proc-without-powr
    (world.rs:768 'StartProduction BLOCKED — missing prerequisites
    for proc'); no production building ever appears → both then-
    clauses fail → LOSS on the deadline."""
    c, r = _run(level, _proc_first, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} proc-first must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types}, tick={r.signals.game_tick})"
    )
    assert "powr" not in r.signals.own_building_types
    assert "proc" not in r.signals.own_building_types


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("seed", SEEDS)
def test_army_spam_brute_loses(level, seed):
    """Brute army-spam (only `build('e1')`, never any structure) must
    LOSE — e1 needs tent which needs powr; no chain ever exists; no
    proc; LOSS on the deadline."""
    c, r = _run(level, _army_spam, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed{seed} army-spam must LOSE; got {r.outcome} "
        f"(types={r.signals.own_building_types})"
    )
    assert "powr" not in r.signals.own_building_types
    assert "proc" not in r.signals.own_building_types


# ───────────────────────── hard spawn round-robin ─────────────────────────


def test_hard_seed_round_robin_produces_distinct_starts():
    """Seeds 1-4 must round-robin between the two declared spawn_point
    groups (NORTH (20,12) / SOUTH (20,28)) so a memorised opening
    cannot generalise. The inert spawn-witness `e1` per group surfaces
    the variation via units_summary (the building-only pack would
    otherwise be invisible to the hard-tier shared test)."""
    from pathlib import Path

    from openra_bench.eval_core import RustEnvPool, _scenario_to_tmp_yaml
    from openra_bench.rust_adapter import RustObsAdapter

    c = compile_level(load_pack(PACK), "hard")
    tmp = _scenario_to_tmp_yaml(c)
    pool = RustEnvPool(size=1, scenario_path=tmp)
    env = pool.acquire()
    starts = set()
    try:
        for seed in (1, 2, 3, 4):
            ad = RustObsAdapter()
            ad.observe(env.reset(seed=seed))
            u = ad.render_state().get("units_summary", []) or []
            if u:
                starts.add(
                    tuple(sorted((x["cell_x"], x["cell_y"]) for x in u))
                )
    finally:
        pool.release(env)
        pool.shutdown()
        Path(tmp).unlink(missing_ok=True)
    assert len(starts) >= 2, (
        f"hard seeds 1-4 produced identical starts {starts}; "
        "spawn_point round-robin not taking effect"
    )


# ───────────────────────── determinism ────────────────────────────────────


def test_outcomes_are_deterministic_per_seed():
    """Same seed, same pack, same policy → identical outcome."""
    c = compile_level(load_pack(PACK), "easy")
    a = run_level(c, _intended, seed=2)
    b = run_level(c, _intended, seed=2)
    assert (a.outcome, a.turns) == (b.outcome, b.turns), (
        f"determinism: {(a.outcome, a.turns)} vs {(b.outcome, b.turns)}"
    )
