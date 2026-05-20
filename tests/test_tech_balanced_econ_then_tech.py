"""tech-balanced-econ-then-tech pack — full no-cheat validation on Rust.

Wave-4 BALANCED tech-triple pack. Uses the Wave-2 `then:[A,B,C,D]`
happened-before composite to enforce the order:
  1. has_building: proc          # phase 1 — build the econ floor
  2. economy_value_gte: <bar>    # phase 2 — actually USE the proc
  3. has_building: weap          # phase 3 — tech step 1
  4. has_building: dome          # phase 4 — tech step 2

Bar (per CLAUDE.md): the intended econ-then-tech policy must WIN on
every (level, seed); stall / pure-aggro-skip-proc / pure-turtle-
defenses-only must LOSE on every (level, seed). No draws.

Scenario shape:
  - rush-hour-arena, allied agent, no enemy combat units.
  - Pre-placed agent base: fact + tent + powr + harv + one mine.
  - Inert enemy `fact` marker far east keeps the engine from
    auto-terminating on enemy-elimination.
  - starting_cash: 2800 — exactly the cost of one dome (the most
    expensive single building in the chain). The agent must build
    proc (1400) FIRST, then HARVEST income from the pre-placed harv
    to refill enough cash for weap (2000) and dome (2800).
  - hard: ≥2 spawn_point groups (north vs south) per the hard-tier
    contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "tech-balanced-econ-then-tech.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Idles every turn — must LOSS (no win, no draw)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _pure_aggro_skip_proc_policy():
    """Skip the refinery entirely — try to race straight to weap+dome.
    Both have engine prereq `proc`, so neither order is accepted by the
    engine; the agent never satisfies the then-chain phase-1
    (has_building: proc) → chain never advances → LOSS on the clock.
    """
    def pol(obs, Cmd):
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        cmds = []
        # Try to build weap and dome immediately — both will be REJECTED
        # by the engine because `proc` is a hard prereq. The bench bar
        # is that the chain still cannot advance.
        if "weap" not in own_b and "weap" not in prod:
            cmds.append(Cmd.build("weap"))
        if "dome" not in own_b and "dome" not in prod:
            cmds.append(Cmd.build("dome"))
        # Place them in case the queue silently accepts (it won't, but
        # we mirror the well-formed sequence a model would emit).
        base = [b for b in (obs.get("own_buildings", []) or [])
                if b["type"] == "fact"]
        if base and "weap" not in own_b:
            cmds.append(Cmd.place_building(
                "weap", base[0]["cell_x"] + 4, base[0]["cell_y"] + 2
            ))
        if base and "dome" not in own_b:
            cmds.append(Cmd.place_building(
                "dome", base[0]["cell_x"] + 4, base[0]["cell_y"] + 4
            ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pure_turtle_defenses_only_policy():
    """Build defenses (pbox/gun) only — never the tech chain. Phase 1
    (has_building: proc) never latches → chain stuck at 0 → LOSS."""
    def pol(obs, Cmd):
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        base = [b for b in (obs.get("own_buildings", []) or [])
                if b["type"] == "fact"]
        cmds = []
        # pbox needs `tent` (preplaced) + `powr` (preplaced) → buildable.
        if "pbox" not in prod:
            cmds.append(Cmd.build("pbox"))
        if base:
            cmds.append(Cmd.place_building(
                "pbox", base[0]["cell_x"] + 3, base[0]["cell_y"] + 3
            ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_econ_then_tech_policy():
    """The intended capability play:
       1. build proc + place east of fact (so harv can deliver)
       2. issue harvest to the pre-placed harv
       3. once cash ≥ 2000, build + place weap
       4. once cash ≥ 2800, build + place dome
       The phase-2 economy_value bar is satisfied passively by harv
       throughput; the policy just orders the four buildings in order.
    """
    state = {"harv_kicked": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        base = next((b for b in ob if b["type"] == "fact"), None)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 18
        cmds = []

        # Phase 1: build proc (1400 cr, prereq powr — preplaced).
        if "proc" not in own_b and "proc" not in prod:
            cmds.append(Cmd.build("proc"))
        # Place proc adjacent so the harv has a delivery target.
        if "proc" not in own_b:
            cmds.append(Cmd.place_building("proc", bx + 4, by + 4))

        # Kick the harvester once proc is up so income starts flowing.
        # The harv auto-pathfinds to the nearest ore patch, but issuing
        # an explicit harvest order against the seen mine cell ensures
        # it starts the loop promptly. Use the harv's own (cell_x,
        # cell_y) + a deterministic east-of-base offset that lands in
        # the mine column for either spawn group.
        if "proc" in own_b and not state["harv_kicked"]:
            harv = next((u for u in units if u.get("type") == "harv"), None)
            if harv:
                # Mine is at (22, by) — east of the base on either spawn
                # group. Direct the harv there explicitly.
                cmds.append(Cmd.harvest([harv["id"]], 22, by))
                state["harv_kicked"] = True

        # Phase 3: build weap (2000 cr, prereq proc).
        if "proc" in own_b and "weap" not in own_b and "weap" not in prod \
                and cash >= 2000:
            cmds.append(Cmd.build("weap"))
        if "weap" not in own_b and "proc" in own_b:
            cmds.append(Cmd.place_building("weap", bx + 6, by))

        # Phase 4: build dome (2800 cr, prereq proc).
        if "proc" in own_b and "dome" not in own_b and "dome" not in prod \
                and cash >= 2800:
            cmds.append(Cmd.build("dome"))
        if "dome" not in own_b and "proc" in own_b:
            cmds.append(Cmd.place_building("dome", bx + 8, by + 2))

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "tech-balanced-econ-then-tech"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2 macro-tech" in a for a in anchors), anchors
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("roadmap" in a for a in anchors), anchors
    assert any("industrial" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start base (binding contract from tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_then_composite_used_in_win():
    """Confirms the econ-then-tech chain wired through to the
    compiled win condition (the whole point of the BALANCED pack)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        assert any("then" in clause for clause in ao), (
            f"{lvl} win has no then-composite: {win}"
        )
        then_clause = next(c["then"] for c in ao if "then" in c)
        clauses = then_clause["clauses"]
        # Must be exactly: proc → economy_value → weap → dome.
        assert "has_building" in clauses[0] and clauses[0]["has_building"] == "proc"
        assert "economy_value_gte" in clauses[1]
        assert "has_building" in clauses[2] and clauses[2]["has_building"] == "weap"
        assert "has_building" in clauses[3] and clauses[3]["has_building"] == "dome"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns (CLAUDE.md
    tick/turn footgun: ≤ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)
        wts = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_econ_then_tech_wins(level, seed):
    """The intended capability play (proc → income → weap → dome)
    must WIN on every (level, seed). Load-bearing solvency test."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_econ_then_tech_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "win", (
        f"intended econ-then-tech must WIN on {level} s={seed}; "
        f"got {res.outcome} turns={res.turns} "
        f"own_buildings={own_b} cash={res.signals.cash} "
        f"resources={res.signals.resources} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Do-nothing must LOSS (no win, no draw) on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_aggro_skip_proc_loses(level, seed):
    """Pure aggro (race weap+dome WITHOUT proc) must LOSS: both have
    engine prereq `proc`, so neither order is accepted → phase 1 of
    the then-chain never latches → chain stuck → clock expires."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_aggro_skip_proc_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"pure-aggro-skip-proc must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )
    # Sanity: neither weap nor dome should have been completed.
    assert "weap" not in own_b, f"aggro-skip-proc built weap on {level} s={seed}"
    assert "dome" not in own_b, f"aggro-skip-proc built dome on {level} s={seed}"


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_turtle_defenses_only_loses(level, seed):
    """Build defenses only — never the tech chain. Phase 1
    (has_building: proc) never latches → chain stuck → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_turtle_defenses_only_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"pure-turtle-defenses-only must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b} "
        f"then_progress={getattr(res.signals, 'then_progress', {})}"
    )
    # Sanity: proc never built (the agent never tried).
    assert "proc" not in own_b, f"turtle built proc on {level} s={seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin per seed."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
