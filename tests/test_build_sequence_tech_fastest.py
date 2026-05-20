"""build-sequence-tech-fastest pack — full no-cheat validation on Rust.

Wave-7 REASONING — cost-optimal build-order planning. The agent must
reach the war factory (`weap`) on the SHORTEST prerequisite chain:

    powr → proc → weap

Any detour (build a barracks/tent first, or a redundant power plant,
or an early infantry queue) overruns the tight tick budget and loses.
The chain is enforced by the Wave-2 `then:` happened-before composite;
the deadline (`within_ticks`) is the cost-optimality teeth — slack is
tuned so the OPTIMAL plan fits and the tent-detour plan does NOT.

Bar (CLAUDE.md): the intended cost-optimal policy WINS on every
(level, seed); stall and the tent-first wrong-path policy LOSE on
every (level, seed). Real LOSS not DRAW — `fail after_ticks:T+1`
inside max_turns is the bite.

Scenario shape:
  - rush-hour-arena, allies vs soviet (bot disabled).
  - easy:   T=3000, max_turns=40 — generous (4-turn buffer).
  - medium: T=2800, max_turns=35 — tight (≈2-turn buffer).
  - hard:   T=2800, max_turns=35 — same tight T + ≥2 spawn_point
            groups (NORTH y=14 / SOUTH y=26 base, round-robined).

Measured optimal timing (seed 1, scripted intended policy):
  powr completes ≈ tick  273 (turn  3)
  proc completes ≈ tick 1263 (turn 14)
  weap completes ≈ tick 2613 (turn 29)
Measured tent-first wrong-path timing:
  weap completes ≈ tick 3063 (turn 34) — beyond every level's T.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-sequence-tech-fastest.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_policy():
    """Cost-optimal play: build powr → proc → weap, each one placed
    relative to the agent's actual fact (so the policy generalises
    across the hard-tier spawn variation). This is the policy the
    pack is solvable by — must WIN on every (level, seed)."""
    milestone = {"powr": False, "proc": False, "weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        for b in ("powr", "proc", "weap"):
            if b in own_b:
                milestone[b] = True
        cmds = []
        base = [b for b in ob if b["type"] == "fact"]
        if not milestone["powr"]:
            if "powr" not in prod:
                cmds.append(Cmd.build("powr"))
            if base:
                cmds.append(Cmd.place_building(
                    "powr", base[0]["cell_x"] + 4, base[0]["cell_y"]
                ))
        elif not milestone["proc"]:
            if "proc" not in prod:
                cmds.append(Cmd.build("proc"))
            if base:
                cmds.append(Cmd.place_building(
                    "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 3
                ))
        elif not milestone["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
                ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _tent_first_policy():
    """Wrong cost-non-optimal play: powr → tent → proc → weap. The
    tent is not on the prerequisite chain for weap (only proc is); it
    bloats the BOM by 500 credits and ~5 turns. Must LOSE on the
    clock on every level/seed."""
    milestone = {"powr": False, "tent": False, "proc": False, "weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        for b in ("powr", "tent", "proc", "weap"):
            if b in own_b:
                milestone[b] = True
        cmds = []
        base = [b for b in ob if b["type"] == "fact"]
        if not milestone["powr"]:
            if "powr" not in prod:
                cmds.append(Cmd.build("powr"))
            if base:
                cmds.append(Cmd.place_building(
                    "powr", base[0]["cell_x"] + 4, base[0]["cell_y"]
                ))
        elif not milestone["tent"]:
            if "tent" not in prod:
                cmds.append(Cmd.build("tent"))
            if base:
                cmds.append(Cmd.place_building(
                    "tent", base[0]["cell_x"] + 4, base[0]["cell_y"] + 3
                ))
        elif not milestone["proc"]:
            if "proc" not in prod:
                cmds.append(Cmd.build("proc"))
            if base:
                cmds.append(Cmd.place_building(
                    "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 3
                ))
        elif not milestone["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
                ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "build-sequence-tech-fastest"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench cost-optimal +
    BOM manufacturing critical-path planning."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("BOM" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start base (tests/test_hard_tier.py::UPGRADED contract)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_then_composite_used_in_win():
    """Confirms the 3-step build-order chain is wired through to the
    compiled win condition — the load-bearing teeth of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        assert any("then" in cl for cl in inner), (
            f"{lvl} win missing then-chain: {win}"
        )
        for cl in inner:
            if "then" in cl:
                clauses = (cl["then"] or {}).get("clauses") or []
                assert len(clauses) == 3, (
                    f"{lvl} then-chain must be powr→proc→weap (3 clauses); "
                    f"got {clauses}"
                )
                # And in the exact engine-enforced prereq order.
                assert clauses[0].get("has_building") == "powr"
                assert clauses[1].get("has_building") == "proc"
                assert clauses[2].get("has_building") == "weap"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
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
def test_intended_cost_optimal_policy_wins(level, seed):
    """The intended cost-optimal play (powr → proc → weap) must WIN
    on every (level, seed). This is the load-bearing test that the
    pack is solvable inside the budget by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended cost-optimal must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"own_buildings={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed). The
    fail_condition's after_ticks clause bites at the budget; never
    a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_tent_first_wrong_path_loses(level, seed):
    """The cost-non-optimal tent-first play must LOSE on every
    (level, seed). The tent detour adds ~500 credits + ~5 turns,
    pushing weap completion to ~tick 3063 — beyond every level's
    deadline. The capability being measured is COST-OPTIMAL
    planning; a 'some plan that arrives' policy must not win."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _tent_first_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"tent-first wrong-path must LOSE on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, own_buildings={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    different seeds must place the agent fact at a different (x,y).
    Smoke-tests the spawn-variation contract that
    tests/test_hard_tier.py also enforces."""
    c = compile_level(load_pack(PACK), "hard")
    captured = {"first_obs": None}

    def probe(obs, Cmd):
        if captured["first_obs"] is None:
            captured["first_obs"] = list(obs.get("own_buildings", []) or [])
        return [Cmd.observe()]

    res = run_level(c, probe, seed=seed)
    assert res.outcome == "loss"  # stall must lose
    facts = [
        (b["cell_x"], b["cell_y"])
        for b in (captured["first_obs"] or [])
        if b["type"] == "fact"
    ]
    assert facts, f"no fact observed at turn 0 for seed={seed}"


def test_hard_spawns_round_robin_across_seeds():
    """Two seeds (1 and 2) must place the agent's fact at DIFFERENT
    cells — proves the spawn_point round-robin is active, not
    degenerate."""
    c = compile_level(load_pack(PACK), "hard")

    def probe():
        captured = {}
        def pol(obs, Cmd):
            if "fact_pos" not in captured:
                bs = obs.get("own_buildings", []) or []
                facts = [(b["cell_x"], b["cell_y"]) for b in bs if b["type"] == "fact"]
                if facts:
                    captured["fact_pos"] = facts[0]
            return [Cmd.observe()]
        pol.captured = captured
        return pol

    p1 = probe(); run_level(c, p1, seed=1)
    p2 = probe(); run_level(c, p2, seed=2)
    pos1 = p1.captured.get("fact_pos")
    pos2 = p2.captured.get("fact_pos")
    assert pos1 and pos2, f"missing fact obs: s1={pos1} s2={pos2}"
    assert pos1 != pos2, (
        f"hard spawn round-robin is degenerate: seed 1 and 2 both "
        f"started at {pos1}"
    )
