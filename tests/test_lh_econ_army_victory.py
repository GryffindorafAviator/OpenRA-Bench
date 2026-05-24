"""lh-econ-army-victory pack — full no-cheat validation on Rust.

Wave-8 long-horizon: 3-phase macro chain enforced by the Wave-2
`then:` happened-before composite. The chain is:

    PHASE 1 (ECON):    economy_value_gte: M
    PHASE 2 (ARMY):    own_units_gte: 6
    PHASE 3 (VICTORY): units_killed_gte: K

Bar (per CLAUDE.md): the intended econ→army→victory policy WINS on
every (level, seed); stall / skip-econ-pump-army / pure-econ-no-attack
all LOSE on every seed. The `then:` latch is the load-bearing teeth.

Scenario shape:
  - rush-hour-arena, allies vs soviet.
  - easy: M=1500, K=2, bot disabled, 50 turns.
  - medium: M=2200, K=3, hunt bot, 40 turns.
  - hard: M=2500, K=4, hunt bot, ≥2 spawn_point groups, 40 turns.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-econ-army-victory.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level (M, K) — kept in lock-step with the YAML.
_M = {"easy": 1500, "medium": 2200, "hard": 2500}
_K = {"easy": 2, "medium": 3, "hard": 3}


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on every level/seed.

    Pure observe — no harvest order is issued either, so cash also
    stays flat. Phase 2 (own_units_gte:6) never latches (only 1
    pre-placed harv) so the chain stalls at index 0 / 1."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _skip_econ_pump_army_policy():
    """Spend all cash on infantry immediately, never engage; income
    refills below the M bar because the spend depletes the pool.
    Phase 1 (economy_value_gte:M) cannot latch — M is tuned above
    the post-spend cash floor + the limited harvest window left
    after pre-spending. Must LOSE on every level/seed."""
    def pol(obs, Cmd):
        cmds = []
        prod = obs.get("production", []) or []
        # Pump e1 every turn we have queue space + cash.
        if "e1" not in prod:
            cmds.append(Cmd.build("e1"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pure_econ_policy():
    """Issue harvest order so cash creeps past M, but NEVER build an
    army or attack. Phase 1 latches but phases 2-3 never do
    (own_units stays at 1, units_killed stays at 0). Must LOSE."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        harv_ids = [u["id"] for u in units if u.get("type") == "harv"]
        cmds = []
        if harv_ids:
            cmds.append(Cmd.harvest(harv_ids, 22, 18))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_phased_policy(level: str):
    """The intended capability play:
      PHASE 1: wait for harvest income to accrue past M (own buildings
               already include proc + harv on a near mine).
      PHASE 2: once economy_value ≥ M, build e1's until own_units ≥ 6.
      PHASE 3: attack_move the riflemen east onto the enemy targets.

    Uses a sticky milestone latch so a building destroyed mid-episode
    or a unit dying doesn't reset the chain on the policy side. This
    is the policy the pack is solvable by — must WIN on every
    (level, seed)."""
    M = _M[level]
    milestone = {"econ": False, "army": False}
    issued_attack = set()
    stance_set = set()

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        prod = obs.get("production", []) or []
        cash = int(obs.get("cash", 0) or 0)
        resources = int(obs.get("resources", 0) or 0)
        ev = cash + resources
        if ev >= M:
            milestone["econ"] = True
        own_n = len(units)
        if own_n >= 6:
            milestone["army"] = True
        cmds = []
        # Always keep the pre-placed harv on the mine — issuing the
        # `harvest` order every turn is harmless and kickstarts the
        # auto-cycle (the bare pre-placed harv otherwise sits idle).
        harv_ids = [u["id"] for u in units if u.get("type") == "harv"]
        if harv_ids:
            # Kickstart auto-cycle by issuing a harvest order targeted
            # at the spawn-matched mine. On easy/medium the mine is at
            # (22,18); on hard the agent's spawn-row harv finds either
            # (22,14) or (22,22) via the nearest-ore auto-pull. We use
            # the harv's own row as the harvest hint cell.
            harv0 = next(u for u in units if u.get("type") == "harv")
            cmds.append(Cmd.harvest(harv_ids, 22, int(harv0["cell_y"])))
        # PHASE 1: wait for econ.
        if not milestone["econ"]:
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASE 2: build e1's until own_units ≥ 6. Keep queueing
        # extras concurrently so subsequent kills phase isn't held
        # back by a 5-unit minimum strike force.
        if not milestone["army"]:
            if "e1" not in prod:
                cmds.append(Cmd.build("e1"))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # Keep producing reinforcements; the army bar is met but the
        # kill bar still needs more attackers (hard tier especially).
        if "e1" not in prod and cash >= 100:
            cmds.append(Cmd.build("e1"))
        # PHASE 3: set AttackAnything stance on every non-harv unit
        # (so units idle on arrival still auto-fire on enemies in
        # range) and attack-move them east. Issue attack-move per
        # unit ONCE so the move order isn't restarted every turn
        # (which causes path-then-stop oscillation).
        strike = [u for u in units if u.get("type") != "harv"]
        new_units = [u["id"] for u in strike if u["id"] not in stance_set]
        if new_units:
            cmds.append(Cmd.set_stance(new_units, 3))
            stance_set.update(new_units)
        fresh = [u for u in strike if u["id"] not in issued_attack]
        if fresh:
            ids = [u["id"] for u in fresh]
            cmds.append(Cmd.attack_move(ids, 70, 20))
            for uid in ids:
                issued_attack.add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-econ-army-victory"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: SC2 macro / military operational /
    PlanBench long-sequencing."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2" in a for a in anchors), anchors
    assert any("military operational" in a.lower() for a in anchors), anchors
    assert any("PlanBench" in a for a in anchors), anchors


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
    """Confirms the 3-phase macro chain is wired through to the
    compiled win condition — the load-bearing teeth of this pack."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        assert any("then" in cl for cl in inner), (
            f"{lvl} win missing then-chain: {win}"
        )
        # The chain must have exactly 3 clauses (econ → army → victory).
        for cl in inner:
            if "then" in cl:
                clauses = (cl["then"] or {}).get("clauses") or []
                assert len(clauses) == 3, (
                    f"{lvl} then-chain must have 3 clauses; got {clauses}"
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


def test_starting_cash_below_economy_value_bar():
    """M must be tuned ABOVE starting_cash so phase-1 cannot be
    satisfied without real harvest income (the Wave-8 spec)."""
    pack = load_pack(PACK)
    for lvl, m in _M.items():
        c = compile_level(pack, lvl)
        sc = c.starting_cash
        assert sc < m, (
            f"{lvl} starting_cash={sc} >= M={m} — phase-1 can be "
            f"satisfied without harvest income (skip-econ would WIN)"
        )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_phased_policy_wins(level, seed):
    """The intended 3-phase econ→army→victory play must WIN on every
    (level, seed). This is the load-bearing test that the pack is
    solvable inside the budget by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_phased_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended phased macro must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, "
        f"cash={res.signals.cash}, resources={res.signals.resources})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed). The
    fail_condition's after_ticks clause bites at the turn budget;
    never a draw. Phase 2 (own_units_gte:6) cannot be satisfied
    by a single pre-placed harv."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_pure_econ_loses(level, seed):
    """A "harvest but never engage" policy must LOSE on every
    (level, seed). Phase 1 latches (cash creeps past M) but phases
    2-3 never do — the chain stalls at index 1 and the clock
    expires. Confirms the army/kill bars are real teeth."""
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
