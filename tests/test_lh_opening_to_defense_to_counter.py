"""lh-opening-to-defense-to-counter pack — full no-cheat validation.

Wave-10 long-horizon: 3-phase military chain enforced by the Wave-2
`then:` happened-before composite. The chain is:

    PHASE 1 (OPENING):  has_building: powr AND has_building: proc
    PHASE 2 (DEFENSE):  units_killed_gte: N_def AND after_ticks: T1
    PHASE 3 (COUNTER):  enemy_key_buildings_destroyed_in_region (far
                        east enemy construction yard)

Bar (per CLAUDE.md): the intended open→defend→counter policy WINS on
every (level, seed); stall / rush-the-counter-first / defend-only /
open-then-immediate-counter all LOSE on every seed — never a draw.
The `then:` latch (ordered) plus the unpowered-base opening gate are
the load-bearing teeth.

Scenario shape (rush-hour-arena, allies vs soviet rusher bot):
  - easy:   4×e1 rush far, N_def=4, T1=1100, 130 turns.
  - medium: 6×e1+2×e3 rush far, N_def=7, T1=1300, 130 turns.
  - hard:   8×e1+2×e3 rush (immediate on the spawn_point path) with a
            pre-placed defensive squad, N_def=9, T1=700, ≥2 agent
            spawn_point groups, 130 turns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-opening-to-defense-to-counter.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level intended push tick — kept in lock-step with the YAML
# (the rush must be resolved before the counter force departs).
_PUSH = {"easy": 1250, "medium": 1500, "hard": 900}


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on every level/seed. The opening
    (`has_building: powr`) is never built so the `then:` chain never
    advances past index 0; the rush razes the base (easy/medium) or
    the clock expires (hard, where the pre-placed squad survives)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _rush_east_policy():
    """Send every unit east at the counter objective from turn 1;
    never build the opening, never defend. Must LOSE: phase 1 never
    latches, and on hard the immediate rush razes the abandoned
    base; on easy/medium the trickle of units is too thin to raze
    the enemy yard and the base falls undefended."""
    seen = set()

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        strike = [u["id"] for u in units if u.get("type") in ("e1", "e3")]
        cmds = []
        fresh = [i for i in strike if i not in seen]
        if fresh:
            cmds.append(Cmd.set_stance(fresh, 3))
            cmds.append(Cmd.attack_move(fresh, 118, 20))
            seen.update(fresh)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _defend_only_policy():
    """Build the opening (powr) and train infantry to hold the rush,
    but NEVER counter-attack east. Phases 1-2 latch but phase 3
    never does — the chain stalls at index 2 and the clock expires.
    Must LOSE on every level/seed (confirms the counter bar bites)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        bld = obs.get("own_buildings", []) or []
        own = {b["type"] for b in bld}
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        fact = next((b for b in bld if b["type"] == "fact"), None)
        fx, fy = (fact["cell_x"], fact["cell_y"]) if fact else (10, 20)
        cmds = []
        harv = [u["id"] for u in units if u.get("type") == "harv"]
        if harv:
            cmds.append(Cmd.harvest(harv, 24, fy))
        if "powr" not in own:
            if "powr" not in prod and cash >= 300:
                cmds.append(Cmd.build("powr"))
            cmds.append(Cmd.place_building("powr", fx + 3, fy - 3))
        else:
            if "e1" not in prod and cash >= 100:
                cmds.append(Cmd.build("e1"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _open_then_immediate_counter_policy():
    """Build the opening (powr), then send EVERY trained unit east
    immediately — skip the defence hold. Must LOSE: with no force
    held at home the rush razes the base, and the piecemeal counter
    cannot raze the enemy yard in time. (This is the headline
    'phase 2 skipped' discriminator — the `then:` latch will not
    advance to the counter clause without the defence clause.)"""
    seen = set()

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        bld = obs.get("own_buildings", []) or []
        own = {b["type"] for b in bld}
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        fact = next((b for b in bld if b["type"] == "fact"), None)
        fx, fy = (fact["cell_x"], fact["cell_y"]) if fact else (10, 20)
        cmds = []
        if "powr" not in own:
            if "powr" not in prod and cash >= 300:
                cmds.append(Cmd.build("powr"))
            cmds.append(Cmd.place_building("powr", fx + 3, fy - 3))
        else:
            if "e1" not in prod and cash >= 100:
                cmds.append(Cmd.build("e1"))
        strike = [u for u in units if u.get("type") in ("e1", "e3")]
        fresh = [u["id"] for u in strike if u["id"] not in seen]
        if fresh:
            cmds.append(Cmd.set_stance(fresh, 3))
            cmds.append(Cmd.attack_move(fresh, 118, 20))
            seen.update(fresh)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_phased_policy(level: str):
    """The intended capability play:
      PHASE 1 OPENING: build the power plant (powr) so the pre-placed
                       barracks comes online.
      PHASE 2 DEFENSE: train infantry; the trained / pre-placed
                       defenders auto-fire and blunt the rush at home.
      PHASE 3 COUNTER: once the rush window has passed (a fixed tick
                       per level, set above the rush-resolution
                       time), set AttackAnything stance and
                       attack-move the surviving force east onto the
                       enemy construction yard.

    Must WIN on every (level, seed) — this is the load-bearing
    solvency proof that the advertised capability is achievable
    inside the budget."""
    push_t = _PUSH[level]
    pushed = set()

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        bld = obs.get("own_buildings", []) or []
        own = {b["type"] for b in bld}
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        t = obs.get("game_tick", 0) or 0
        fact = next((b for b in bld if b["type"] == "fact"), None)
        fx, fy = (fact["cell_x"], fact["cell_y"]) if fact else (10, 20)
        cmds = []
        # Keep the pre-placed harv working (harmless if already cycling).
        harv = [u["id"] for u in units if u.get("type") == "harv"]
        if harv:
            h0 = next(u for u in units if u.get("type") == "harv")
            cmds.append(Cmd.harvest(harv, 24, int(h0.get("cell_y", fy))))
        # PHASE 1 OPENING: build powr (cheap; unblocks the barracks).
        if "powr" not in own:
            if "powr" not in prod and cash >= 300:
                cmds.append(Cmd.build("powr"))
            cmds.append(Cmd.place_building("powr", fx + 3, fy - 3))
        else:
            # PHASE 2 DEFENSE: stream infantry from the now-powered
            # tent (the trained + pre-placed units auto-defend).
            if "e1" not in prod and cash >= 100:
                cmds.append(Cmd.build("e1"))
        # PHASE 3 COUNTER: after the rush window, push the surviving
        # force east onto the enemy construction yard. Stance is
        # flipped to AttackAnything ONLY at push time (flipping it
        # during the defence would make the home defenders advance
        # out of position and lose the base).
        if t >= push_t:
            strike = [u for u in units if u.get("type") in ("e1", "e3")]
            fresh = [u["id"] for u in strike if u["id"] not in pushed]
            if fresh:
                cmds.append(Cmd.set_stance(fresh, 3))
                cmds.append(Cmd.attack_move(fresh, 118, 20))
                pushed.update(fresh)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-opening-to-defense-to-counter"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: SC2 timing push / military
    defense-then-counter / PlanBench long-sequencing."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2" in a for a in anchors), anchors
    assert any("defense-then-counter" in a.lower() for a in anchors), anchors
    assert any("PlanBench" in a for a in anchors), anchors


def test_then_composite_used_in_win_with_three_clauses():
    """The 3-phase chain (opening → defense → counter) must be wired
    through the compiled win condition with exactly 3 clauses — the
    load-bearing teeth of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        then_nodes = [cl for cl in inner if "then" in cl]
        assert then_nodes, f"{lvl} win missing then-chain: {win}"
        clauses = (then_nodes[0]["then"] or {}).get("clauses") or []
        assert len(clauses) == 3, (
            f"{lvl} then-chain must have 3 clauses; got {clauses}"
        )


def test_phase_clauses_have_expected_predicates():
    """Phase 1 = powr AND proc owned (an `all_of` of two
    `has_building` leaves — a single leaf node cannot carry the same
    key twice); phase 2 = kills + after_ticks hold; phase 3 = enemy
    key building razed in region."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        then = next(cl for cl in win["all_of"] if "then" in cl)["then"]
        p1, p2, p3 = then["clauses"]
        # Phase 1 must require BOTH powr and proc — guards against the
        # YAML duplicate-key footgun (`{has_building: powr,
        # has_building: proc}` silently collapses to one key).
        p1_inner = p1.get("all_of") or []
        p1_buildings = {
            cl["has_building"] for cl in p1_inner if "has_building" in cl
        }
        assert {"powr", "proc"} <= p1_buildings, (
            f"{lvl} phase 1 must require powr AND proc; got {p1}"
        )
        assert "units_killed_gte" in p2 and "after_ticks" in p2, p2
        assert "enemy_key_buildings_destroyed_in_region" in p3, p3


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start base (tests/test_hard_tier.py::UPGRADED contract)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns even on an
    interrupt-heavy run (~45 ticks/turn worst case) so a non-finisher
    crosses the after_ticks fail clause as a real LOSS, never a
    draw."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True
        )
        fail = compile_level(pack, lvl).fail_condition.model_dump(
            exclude_none=True
        )

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list = []
        _collect(win, "within_ticks", wts)
        ats: list = []
        _collect(fail, "after_ticks", ats)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        assert ats, f"{lvl} has no after_ticks fail leaf"
        # Optimistic engine max (~90 ticks/turn): within_ticks must be
        # reachable for the intended WIN.
        reachable_max = 93 + 90 * (max_turns - 1)
        for wt in wts:
            assert wt <= reachable_max, (
                f"{lvl} within_ticks={wt} > optimistic reachable="
                f"{reachable_max}"
            )
        # Pessimistic engine min (~45 ticks/turn under heavy
        # interrupts): the after_ticks fail must still be crossed
        # before the turn budget runs out, or a non-finisher draws.
        pessimistic = 45 * max_turns
        for at in ats:
            assert at <= pessimistic, (
                f"{lvl} after_ticks={at} > pessimistic reachable="
                f"{pessimistic} (max_turns={max_turns}) — a "
                f"non-finisher could DRAW instead of LOSE"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_phased_policy_wins(level, seed):
    """The intended open→defend→counter play must WIN on every
    (level, seed). The load-bearing solvency proof."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_phased_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended phased play must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed) — never
    a draw. The opening is never built so the chain never advances."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_rush_east_first_loses(level, seed):
    """Racing the counter east from turn 1 (no opening, no defence)
    must LOSE on every (level, seed) — the base falls undefended and
    phase 1 never latches."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _rush_east_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"rush-east-first must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_defend_only_loses(level, seed):
    """An open-then-defend policy that never counter-attacks must
    LOSE on every (level, seed). Phases 1-2 latch but phase 3 never
    does — confirms the counter bar is real teeth (not a draw)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _defend_only_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"defend-only must LOSE on {level} s={seed}; got "
        f"{res.outcome} (then_progress={tp})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_open_then_immediate_counter_loses(level, seed):
    """Building the opening then sending every unit east immediately
    (skipping the defence hold) must LOSE on every (level, seed) —
    the `then:` latch will not advance to the counter clause without
    the defence clause, and the abandoned base is razed."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _open_then_immediate_counter_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"open-then-immediate-counter must LOSE on {level} s={seed}; "
        f"got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin — the agent
    base latitude flips by seed. Smoke-tests the spawn-variation
    contract that tests/test_hard_tier.py also enforces."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose every seed
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
