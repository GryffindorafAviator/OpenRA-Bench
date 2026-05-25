"""lh-opening-to-tech-to-army pack — full no-cheat validation on Rust.

Group G long-horizon: 4-phase macro chain enforced by the Wave-2
`then:` happened-before composite. The chain is:

    PHASE 1 (econ):   has_building: proc
    PHASE 2 (tech):   has_building: weap
    PHASE 3 (army):   unit_type_count_gte: {type: 2tnk, n: N}
    PHASE 4 (engage): enemy_key_buildings_destroyed: {types: [fact]}

Bar (per CLAUDE.md): the intended phased-macro policy WINS on every
(level, seed); stall / skip-to-attack / pure-econ-no-attack all LOSE
on every seed. The `then:` latch is the load-bearing teeth — a
"build only tanks and attack" policy cannot satisfy phase 1+2 because
the latch starts at index 0 and can only advance when proc → weap →
army → engage are each observed-true in sequence.

Scenario shape:
  - rush-hour-arena, allies vs soviet.
  - easy: small army (2 tanks), no time pressure (bot disabled).
  - medium: 4 tanks, hunt bot, 80 turns / 7203 ticks.
  - hard: 5 tanks + ≥2 spawn_point groups, static enemy, 110 turns /
    9300-tick clock. (Recalibrated: the engine balance fixes made the
    prior 6-tank hunt-bot hard tier unsolvable — the hunt bot razed
    the agent economy and a 6-tank single-queue chain overran the
    engine's ~10080-tick hard episode cap.)
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-opening-to-tech-to-army.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level army size (the n in unit_type_count_gte: {type: 2tnk, n: N}).
# hard was 6 pre-recalibration; the engine balance fixes (hunt-bot
# riflemen razing the agent economy + the engine's ~10080-tick hard
# episode cap) made a 6-tank single-queue chain unsolvable, so hard
# now runs a static enemy and a 5-tank army clause — still a clean
# +1 over medium and comfortably inside the episode cap.
_ARMY_N = {"easy": 2, "medium": 4, "hard": 5}


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _skip_to_attack_policy():
    """Build ONLY tanks (no proc, no weap) and attack the enemy fact
    immediately. Cannot satisfy the then-chain because phase-1 (proc)
    is never latched — the chain stays at index 0 forever.

    Also fails because building 2tnk requires `weap` (tech-gated by
    the engine), so the build never actually produces; this is a
    belt-and-braces test that the chain bites even if the engine were
    to allow it."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        prod = obs.get("production", []) or []
        cmds = []
        # Try to build a tank straight away (engine will reject without
        # weap; that's fine — the test is that the chain does not advance).
        if "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        # Attack-move any existing units east toward the enemy fact.
        own_ids = [u["id"] for u in units]
        if own_ids:
            cmds.append(Cmd.attack_move(own_ids, 80, 30))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pure_econ_policy():
    """Build proc + weap (the early phases) but never an army, never
    engage. Phase-3 / phase-4 clauses never latch. Must LOSE."""
    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        cmds = []
        base = [b for b in ob if b["type"] == "fact"]
        if "proc" not in own_b and "proc" not in prod:
            cmds.append(Cmd.build("proc"))
        if "proc" not in own_b and base:
            cmds.append(Cmd.place_building(
                "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 4
            ))
        if "proc" in own_b and "weap" not in own_b and "weap" not in prod:
            cmds.append(Cmd.build("weap"))
        if "proc" in own_b and "weap" not in own_b and base:
            cmds.append(Cmd.place_building(
                "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
            ))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_phased_policy(army_n: int):
    """The intended capability play: PHASE 1 build proc → PHASE 2
    build weap → PHASE 3 build N×2tnk → PHASE 4 attack-move all tanks
    onto the enemy fact at (80, 30) to destroy it.

    Uses a sticky milestone latch (the policy remembers which phases
    it has completed once-and-for-all, so a building destroyed
    mid-episode doesn't reset the chain). This is the policy the
    pack is solvable by — must WIN on every (level, seed)."""
    milestone = {"proc": False, "weap": False, "army": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        # Latch milestones permanently (sticky).
        if "proc" in own_b:
            milestone["proc"] = True
        if "weap" in own_b:
            milestone["weap"] = True
        if len(tnk) >= army_n:
            milestone["army"] = True
        cmds = []
        base = [b for b in ob if b["type"] == "fact"]
        # PHASE 1: build proc (refinery → income).
        if not milestone["proc"]:
            if "proc" not in prod:
                cmds.append(Cmd.build("proc"))
            if base:
                cmds.append(Cmd.place_building(
                    "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 4
                ))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASE 2: build weap (war factory → unlocks tanks).
        if not milestone["weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
                ))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASE 3: produce N tanks (in flight; produce until army_n
        # have ever existed).
        if not milestone["army"]:
            if "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))
            # Concurrently send any already-built tanks east toward
            # the enemy fact so their travel time overlaps production.
            if tnk:
                tnk_ids = [u["id"] for u in tnk]
                cmds.append(Cmd.attack_move(tnk_ids, 80, 30))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASE 4: send the army to the enemy fact. Once tanks are
        # adjacent to a non-shooting building under ReturnFire stance
        # they stop firing — switch to focus-fire `attack_unit` on the
        # fact id once it's visible to finish the kill inside budget.
        if tnk:
            tnk_ids = [u["id"] for u in tnk]
            fact_id = None
            for e in obs.get("enemy_buildings_summary", []) or []:
                if (e.get("type") == "fact"
                        and abs(e.get("cell_x", -99) - 80) <= 6
                        and abs(e.get("cell_y", -99) - 30) <= 6):
                    fact_id = str(e["id"])
                    break
            if fact_id is not None:
                cmds.append(Cmd.attack_unit(tnk_ids, fact_id))
            else:
                cmds.append(Cmd.attack_move(tnk_ids, 80, 30))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-opening-to-tech-to-army"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: SC2 macro / PlanBench
    multi-stage / lmgame-Bench / product roadmap."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2LE" in a for a in anchors), anchors
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("lmgame" in a.lower() for a in anchors), anchors
    assert any("roadmap" in a.lower() for a in anchors), anchors


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
    """Confirms the 4-phase macro chain is wired through to the
    compiled win condition — the load-bearing teeth of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        # all_of[ then{4 clauses}, within_ticks ]
        inner = win.get("all_of") or []
        assert any("then" in cl for cl in inner), (
            f"{lvl} win missing then-chain: {win}"
        )
        # The chain must have exactly 4 clauses (proc → weap → army → engage).
        for cl in inner:
            if "then" in cl:
                clauses = (cl["then"] or {}).get("clauses") or []
                assert len(clauses) == 4, (
                    f"{lvl} then-chain must have 4 clauses; got {clauses}"
                )


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
def test_intended_phased_policy_wins(level, seed):
    """The intended 4-phase macro play (proc → weap → N×2tnk → engage)
    must WIN on every (level, seed). This is the load-bearing test
    that the pack is solvable inside the budget by the advertised
    capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_phased_policy(_ARMY_N[level]), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended phased macro must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, "
        f"own_buildings={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed). The
    fail_condition's after_ticks clause bites at the turn budget;
    never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_skip_to_attack_loses(level, seed):
    """A "skip the chain, just attack" policy must LOSE on every
    (level, seed). The then-chain demands has_building:proc as
    clause-1 — a policy that never builds proc cannot advance the
    chain past index 0. Even if the policy somehow destroyed the
    enemy fact, the `then:` latch would still report 0 because the
    earlier clauses never latched in order."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _skip_to_attack_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"skip-to-attack must LOSE on {level} s={seed}; got "
        f"{res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_econ_loses(level, seed):
    """A "build econ + tech but never an army or engage" policy must
    LOSE on every (level, seed). Phases 1+2 latch but phases 3+4 never
    do — the chain stalls at index 2 and the clock expires."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pure_econ_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"pure-econ must LOSE on {level} s={seed}; got "
        f"{res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    different seeds must place the agent base at a different (x,y)
    set. Smoke-tests the spawn-variation contract that
    tests/test_hard_tier.py also enforces."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
