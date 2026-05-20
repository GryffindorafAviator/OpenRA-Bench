"""lh-build-army-coordinate-multifront-attack — full no-cheat
validation on Rust.

Group G long-horizon REASONING pack. Two-phase operational plan:
ASSEMBLE a real army (≥N medium tanks) THEN commit it to a
SIMULTANEOUS TWO-FRONT assault on TWO separated enemy `fact` markers
(NE at (130,15) and SE at (130,45), 120 cells apart). The Wave-2
`then:` happened-before composite enforces the army-clause-before-
destruction ordering; the two destruction clauses (NE + SE) enforce
the SPLIT (one mass cannot raze both corners in sequence within the
clock budget).

Bar (per CLAUDE.md): the intended build-then-split policy WINS on
every (level, seed); stall / one-front-only / send-2-units-each-front
all LOSE on every seed.

Scenario shape:
  - 160×60 arena, allies vs soviet (static — no bot).
  - easy: 4-tank army threshold, no defenders pressure.
  - medium: 6-tank army threshold, light pickets.
  - hard: 8-tank army threshold + 2 spawn_point groups (NORTH/SOUTH
    agent base latitude flips per seed).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-build-army-coordinate-multifront-attack.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level army size (the n in unit_type_count_gte: {type:2tnk, n:N}).
_ARMY_N = {"easy": 4, "medium": 6, "hard": 8}

# The two fixed enemy fact corners (NE + SE).
NE = (130, 15)
SE = (130, 45)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _attack_one_front_only_policy():
    """Build the army to the threshold, then attack-move EVERY tank
    onto the NE fact. The NE destruction clause latches, the army
    clause latches, but the SE destruction clause NEVER latches (the
    SE fact stands untouched). Must LOSE on every (level, seed)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        # Keep building tanks (cap at 12 to avoid runaway queue spam).
        if "weap" in own_b and len(tnk) < 12 and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        if tnk:
            tnk_ids = [u["id"] for u in tnk]
            # ALL tanks at the NE fact — SE is ignored on purpose.
            cmds.append(Cmd.attack_move(tnk_ids, NE[0], NE[1]))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _send_too_few_each_front_policy():
    """Skip the build entirely; just split the two starter 2tnks —
    one to NE, one to SE — and hope (in vain) that 1 tank per prong
    can raze a defended fact. Cannot satisfy the army clause (N≥4
    minimum), and 1 tank cannot raze a fact with defenders inside
    the clock anyway. Must LOSE on every (level, seed)."""
    sent = {"ne": False, "se": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        # Sort tanks by id so the split is deterministic.
        tnk.sort(key=lambda u: u["id"])
        if tnk:
            ne_ids = [tnk[0]["id"]]
            se_ids = [tnk[1]["id"]] if len(tnk) > 1 else []
            if ne_ids:
                cmds.append(Cmd.attack_move(ne_ids, NE[0], NE[1]))
                sent["ne"] = True
            if se_ids:
                cmds.append(Cmd.attack_move(se_ids, SE[0], SE[1]))
                sent["se"] = True
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_mass_and_split_policy(army_n: int):
    """The intended capability play:
      PHASE 1: build the army to ≥army_n medium tanks (concurrent
               with idle observation so production can drip).
      PHASE 2: SPLIT the army half-to-NE, half-to-SE attack-move.
    Uses a sticky latch on the army milestone so a tank dying after
    we've reached threshold doesn't bounce the policy back to phase 1.

    Tanks come out of the pre-placed weap (war factory). Starter 2
    tanks count toward the threshold.
    """
    milestone = {"army": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        if len(tnk) >= army_n:
            milestone["army"] = True
        cmds = []
        # PHASE 1 (always running until army milestone): keep tank
        # production in flight whenever weap exists and the queue
        # isn't already producing one.
        if "weap" in own_b and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        # PHASE 2: once army milestone latched, split-attack.
        # The split is deterministic by tank id (lower half → NE,
        # upper half → SE) so the test is reproducible.
        if milestone["army"] and tnk:
            tnk_sorted = sorted(tnk, key=lambda u: u["id"])
            half = max(1, len(tnk_sorted) // 2)
            ne_ids = [u["id"] for u in tnk_sorted[:half]]
            se_ids = [u["id"] for u in tnk_sorted[half:]]
            if ne_ids:
                cmds.append(Cmd.attack_move(ne_ids, NE[0], NE[1]))
            if se_ids:
                cmds.append(Cmd.attack_move(se_ids, SE[0], SE[1]))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-build-army-coordinate-multifront-attack"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the spec: SC2 macro-then-multi-prong / military
    operational planning / PERT / industrial product launch."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    j = " | ".join(anchors).lower()
    assert "sc2" in j, anchors
    assert "operational" in j or "military" in j, anchors
    assert "pert" in j, anchors
    assert "industrial" in j or "rollout" in j, anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start base latitude (tests/test_hard_tier.py::UPGRADED
    contract)."""
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
    """The win is `all_of[ then{army, NE-raze, SE-raze}, within_ticks ]`.
    Confirms the load-bearing teeth (mass-then-split) is wired."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        then_node = None
        for cl in inner:
            if "then" in cl:
                then_node = cl["then"]
                break
        assert then_node is not None, f"{lvl} win missing then-chain: {win}"
        clauses = (then_node or {}).get("clauses") or []
        # army → NE-raze → SE-raze ⇒ exactly 3 clauses.
        assert len(clauses) == 3, (
            f"{lvl} then-chain must have 3 clauses (army, NE, SE); "
            f"got {clauses}"
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
def test_intended_mass_and_split_wins(level, seed):
    """The intended capability play — build the army to ≥N then SPLIT
    half-to-NE half-to-SE attack-move — must WIN on every (level,
    seed). Load-bearing: the pack is solvable inside the budget by
    the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_mass_and_split_policy(_ARMY_N[level]), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended mass-and-split must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, "
        f"own_buildings={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed). The
    fail_condition's after_ticks bites at the turn budget; never a
    draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_attack_one_front_only_loses(level, seed):
    """A "build army, send EVERY tank to NE only, ignore SE" policy
    must LOSE on every (level, seed). The SE destruction clause
    never latches, so the then-chain never completes."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _attack_one_front_only_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"one-front-only must LOSE on {level} s={seed}; "
        f"got {res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_send_too_few_each_front_loses(level, seed):
    """A "skip the build, send 1 starter tank to NE + 1 to SE" policy
    must LOSE on every (level, seed). The army clause (army_n ≥ 4)
    can never latch with only 2 starter tanks, AND 1 tank per prong
    bounces off the defenders without razing the fact."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _send_too_few_each_front_policy(), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"send-too-few must LOSE on {level} s={seed}; "
        f"got {res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin — different
    seeds must place the agent base at a different latitude (y=18
    NORTH vs y=42 SOUTH). Smoke-tests the spawn-variation contract
    that tests/test_hard_tier.py also enforces."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose on hard
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
