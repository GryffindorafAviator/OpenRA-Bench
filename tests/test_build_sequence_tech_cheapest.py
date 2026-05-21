"""build-sequence-tech-cheapest pack — full no-cheat validation on Rust.

Wave-11 REASONING — cost-MINIMAL build-order planning. Sibling of
build-sequence-tech-fastest (the time-optimal axis); here the binding
constraint is MONEY. The agent must reach the war factory (`weap`) on
the ONLY affordable prerequisite chain:

    powr → proc → weap

There is NO ore on the map and NO harvester income — the starting cash
is the entire, non-replenishing budget, tuned to exactly the cost of
the minimal path (powr $300 + proc $1400 + weap $2000 = $3700). Any
detour (build a barracks/tent or a pillbox first) bloats the bill of
materials, exhausts the fixed budget, and weap can then never be
funded — the `then:` chain never completes and the episode times out.
The clock budget is GENEROUS: a policy loses by being WASTEFUL, not
by being slow.

Bar (CLAUDE.md): the intended cost-minimal policy WINS on every
(level, seed); stall and the wasteful-spend policies LOSE on every
(level, seed). Real LOSS not DRAW — `fail after_ticks:T+1` inside
max_turns is the bite.

Scenario shape:
  - rush-hour-arena, allies vs soviet (bot disabled).
  - easy:   budget $3750, T=3200, max_turns=40 — 50-credit slack.
  - medium: budget $3720, T=3200, max_turns=40 — 20-credit slack.
  - hard:   budget $3720, T=3200, max_turns=40 — same tight budget
            + ≥2 spawn_point groups (NORTH y=14 / SOUTH y=26 base,
            round-robined by seed).

Measured (seed 1, scripted policies):
  intended  powr→proc→weap completes ≈ tick 2613 (well under T=3200)
  tent-first wasteful: cash hits $0 ≈ tick 2703, weap stuck in queue
    forever (no income ⇒ no recovery) ⇒ after_ticks LOSS at T+1.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "build-sequence-tech-cheapest.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_policy():
    """Cost-minimal play: build powr → proc → weap, each placed
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


def _wasteful_policy(extra: str):
    """Cost-non-minimal play: powr → <extra> → proc → weap, where
    <extra> ('tent' $500 or 'pbox' $600) is NOT on weap's prerequisite
    chain. The detour bloats the bill of materials past the fixed
    budget, so weap can never be funded — cash hits $0 mid-queue and,
    with no ore/income, never recovers. Must LOSE on every
    (level, seed). The capability measured is COST-MINIMAL planning;
    a 'some plan that arrives' policy must not win."""
    milestone = {"powr": False, extra: False, "proc": False, "weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        for b in ("powr", extra, "proc", "weap"):
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
        elif not milestone[extra]:
            if extra not in prod:
                cmds.append(Cmd.build(extra))
            if base:
                cmds.append(Cmd.place_building(
                    extra, base[0]["cell_x"] + 4, base[0]["cell_y"] + 3
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
    assert pack.meta.id == "build-sequence-tech-cheapest"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench cost-optimal +
    BOM cost minimization + budget-constrained planning."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("BOM" in a for a in anchors), anchors
    assert any("budget" in a for a in anchors), anchors


def test_budget_is_near_minimal_path_cost():
    """The whole pack hinges on starting_cash being tuned to the
    minimal-path cost (powr 300 + proc 1400 + weap 2000 = 3700) with
    near-zero slack — enough to fund the minimal chain, never enough
    to also afford a non-load-bearing structure."""
    pack = load_pack(PACK)
    minimal = 3700
    for lvl in LEVELS:
        cash = pack.levels[lvl].starting_cash
        assert minimal <= cash <= minimal + 100, (
            f"{lvl} starting_cash={cash} not near-minimal (3700 + ≤100 "
            f"slack); a wasteful detour must overrun the budget"
        )


def test_no_ore_patches_placed():
    """The budget must be the entire, non-replenishing money supply:
    no `mine` actors ⇒ no harvester income ⇒ a wasteful spend can
    never be recovered no matter how generous the clock is."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        mines = [a for a in c.scenario.actors if a.type == "mine"]
        assert not mines, f"{lvl} has ore patches {mines} — income would "\
            "let a wasteful policy recover; budget must be fixed"


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
def test_intended_cost_minimal_policy_wins(level, seed):
    """The intended cost-minimal play (powr → proc → weap) must WIN
    on every (level, seed). This is the load-bearing test that the
    pack is solvable inside the budget by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended cost-minimal must WIN on {level} s={seed}; "
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
@pytest.mark.parametrize("extra", ("tent", "pbox"))
def test_wasteful_spend_loses(level, seed, extra):
    """The cost-non-minimal wasteful play (powr → <extra> → proc →
    weap) must LOSE on every (level, seed). The <extra> detour
    ('tent' $500 / 'pbox' $600) bloats the bill of materials past the
    fixed budget; weap can never be funded (cash hits $0 mid-queue,
    no income ⇒ no recovery) and the `then:` chain never completes.
    The capability measured is COST-MINIMAL planning."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _wasteful_policy(extra), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"wasteful {extra}-first must LOSE on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, own_buildings={res.signals.own_building_types})"
    )
    # weap must NOT have been built — the budget could not fund it.
    assert "weap" not in (res.signals.own_building_types or []), (
        f"wasteful {extra}-first built weap on {level} s={seed} — the "
        f"budget trap leaked (own_buildings={res.signals.own_building_types})"
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
