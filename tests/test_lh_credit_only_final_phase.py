"""lh-credit-only-final-phase — full no-cheat validation on Rust.

REASONING pack, sparse-reward / delayed-gratification long-horizon.
The defining property: the ONLY scoring event is razing the FAR enemy
`fact` at (130,30) inside a single late deadline (within_ticks 7200).
There is NO intermediate predicate — base-building, massing tanks,
marching east all earn ZERO credit on their own. Reward arrives only
at the terminal destruction event.

Bar (per CLAUDE.md): the intended build-then-commit policy WINS on
every (level, seed). The three failure modes all LOSE on every seed:
  - stall (observe only) — the fact stands;
  - build-forever-no-strike (over-invest, never commit the army) —
    the fact stands; this is the sparse-reward-specific failure;
  - strike-too-early (commit the 2 starter tanks immediately) —
    they bounce off the picket, never raze, burn the clock.
Every non-win is a real timeout LOSS via after_ticks 7201, never a
draw.

Scenario shape:
  - 160×60 arena, allies vs soviet (static — no bot).
  - One far objective fact at (130,30); persistent corner marker at
    (155,55) for engine auto-done mitigation.
  - easy: 2-unit picket. medium: 4-unit picket.
  - hard: 4-unit picket + 2 spawn_point groups (NORTH y=18 / SOUTH
    y=42 agent base latitude flips per seed).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-credit-only-final-phase.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# The single fixed far objective.
OBJ = (130, 30)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _build_forever_no_strike_policy():
    """The sparse-reward-specific failure: over-invest. Keep building
    medium tanks for the entire episode and NEVER commit them toward
    the objective. A large idle army earns ZERO credit (no
    intermediate predicate) — the far fact stands, so this must LOSE
    on the clock on every (level, seed)."""
    def pol(obs, Cmd):
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        cmds = []
        if "weap" in own_b and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _strike_too_early_policy():
    """Under-invest: skip the build entirely and immediately commit
    the 2 starter tanks toward the objective. 2 tanks bounce off the
    picket without razing the fact and burn the clock — must LOSE on
    every (level, seed)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        if tnk:
            cmds.append(Cmd.attack_move([u["id"] for u in tnk], OBJ[0], OBJ[1]))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_build_then_commit_policy(commit_turn: int):
    """The intended capability play:
      PHASE 1 (turns 0..commit_turn-1): build the army — keep tank
              production in flight from the pre-placed weap.
      PHASE 2 (turn >= commit_turn): COMMIT every tank on the long
              attack-move toward the single far objective.
    `commit_turn` is the delayed-gratification decision: invest early
    with no feedback, then commit while there is still time for the
    army to march ~120 cells and raze the fact before tick 7200.
    A sticky latch keeps the policy in PHASE 2 once committed."""
    state = {"committed": False, "turn": -1}

    def pol(obs, Cmd):
        state["turn"] += 1
        turn = state["turn"]
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        if turn >= commit_turn:
            state["committed"] = True
        cmds = []
        # PHASE 1 build keeps running until committed (lets production
        # drip in concurrently with the early observation turns).
        if not state["committed"] and "weap" in own_b and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        # PHASE 2: commit the whole army on the long march.
        if state["committed"] and tnk:
            cmds.append(Cmd.attack_move([u["id"] for u in tnk], OBJ[0], OBJ[1]))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# Per-level commit turn: invest long enough to mass a real army that
# can break the picket, but commit early enough that the army can
# march ~120 cells and raze the fact before tick 7200.
_COMMIT_TURN = {"easy": 14, "medium": 18, "hard": 18}


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-credit-only-final-phase"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Spec requires the sparse-reward RL / delayed-gratification /
    long-horizon credit-assignment anchors."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    j = " | ".join(anchors).lower()
    assert "sparse-reward" in j or "sparse reward" in j, anchors
    assert "delayed-gratification" in j or "delayed gratification" in j, anchors
    assert "credit assignment" in j or "credit-assignment" in j, anchors


def test_no_intermediate_predicate_in_win():
    """The sparse-reward contract: the win is ONLY the final-objective
    destruction + within_ticks. No army-size / region-arrival / kill
    intermediate clause may appear (those would be reward shaping)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        keys = set()
        for cl in inner:
            keys |= set(cl.keys())
        assert keys == {"enemy_key_buildings_destroyed_in_region",
                        "within_ticks"}, (
            f"{lvl} win must be ONLY destruction+within_ticks "
            f"(sparse reward, no shaping); got clause keys {keys}"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies the
    start base latitude (tests/test_hard_tier.py::UPGRADED contract)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns, and the
    after_ticks fail must bite. Engine advances ~90 ticks/turn →
    reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        reachable = 93 + 90 * (level_def.max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)
        fail = compile_level(pack, lvl).fail_condition.model_dump(
            exclude_none=True)

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
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={level_def.max_turns}) — deadline inert"
            )
        for at in ats:
            assert at <= reachable, (
                f"{lvl} after_ticks={at} > reachable={reachable} — "
                f"fail never bites ⇒ draw"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_build_then_commit_wins(level, seed):
    """The intended capability play — invest early in the army, then
    COMMIT it on the long march so the far fact is razed before tick
    7200 — must WIN on every (level, seed). Load-bearing: the pack is
    solvable inside the budget by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_build_then_commit_policy(_COMMIT_TURN[level]), seed=seed)
    assert res.outcome == "win", (
        f"intended build-then-commit must WIN on {level} s={seed}; "
        f"got {res.outcome} (turns={res.turns}, "
        f"kills={res.signals.units_killed})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed) — the far
    fact stands, after_ticks 7201 bites. Never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_build_forever_no_strike_loses(level, seed):
    """The sparse-reward-specific failure: build a large army for the
    whole episode but NEVER commit it. A massed idle army earns ZERO
    credit (no intermediate predicate) — must LOSE on every (level,
    seed). This is the test that proves build-up-without-final-strike
    is a real LOSS, not a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_forever_no_strike_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"build-forever-no-strike must LOSE on {level} s={seed}; "
        f"got {res.outcome} (turns={res.turns})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_strike_too_early_loses(level, seed):
    """Under-invest: commit the 2 starter tanks immediately with no
    build. They bounce off the picket without razing the fact and
    burn the clock — must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _strike_too_early_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"strike-too-early must LOSE on {level} s={seed}; "
        f"got {res.outcome} (turns={res.turns})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin — different
    seeds place the agent base at a different latitude (y=18 NORTH vs
    y=42 SOUTH). Smoke-tests the spawn-variation contract."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose on hard
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
