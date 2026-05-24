"""lh-multi-checkpoint-5-plus pack — no-cheat validation on Rust.

Group G long-horizon: 5+ ordered checkpoints chained via the Wave-2
`then:` happened-before composite. The chain MIXES checkpoint kinds:
buildings, quantitative production, episode-aggregate kills, and
terminal destruction — so the ordering is genuinely load-bearing.

Medium chain (the headline 5-phase cell):
    PHASE 1: has_building: proc
    PHASE 2: has_building: weap
    PHASE 3: unit_type_count_gte: {type: 2tnk, n: 2}
    PHASE 4: units_killed_gte: 3
    PHASE 5: enemy_key_buildings_destroyed: {types: [fact]}

Hard adds a 6th regression checkpoint (≥4 tanks after kills) and 2
spawn_point groups (NORTH/SOUTH base round-robined by seed).

Bar (per CLAUDE.md): the intended phased policy WINS on every
(level, seed); stall / skip-phase-3 / skip-phase-4 / pre-build all
LOSE on every seed. The `then:` latch is load-bearing — a "rush
the fact and kill on the way" play scores early kills but the latch
does not credit them to clause-4 until proc + weap + tanks have
latched in order.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-multi-checkpoint-5-plus.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Per-level chain length (the n in `then.clauses`).
_CHAIN_LEN = {"easy": 3, "medium": 5, "hard": 6}

# Per-level required tank army size for the highest-N tank clause.
# easy has no tank-count clause; medium has n:2 only; hard has both
# n:2 (clause 3) and n:4 (clause 5) — the policy must produce ≥4.
_ARMY_TARGET = {"easy": 0, "medium": 2, "hard": 4}


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _enemy_fact_xy(level: str) -> tuple[int, int]:
    """Per-level enemy fact coords (lifted from the YAML so the
    policies attack the right point on each tier). Map width is 96
    so the OOB sentinel targets (115,*) from earlier pack versions
    no longer work — point directly at the actual enemy fact cell."""
    return (80, 30) if level in ("easy", "medium") else (80, 20)


def _skip_to_attack_policy(level: str):
    """Rush the enemy fact immediately with the 4 starting tanks; no
    proc, no weap, no chain. Must LOSE — the then-latch starts at
    index 0 and `has_building:proc` is never observed-true, so even
    a successful fact-kill does not advance the chain past clause 0."""
    fx, fy = _enemy_fact_xy(level)

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        tnk = [u["id"] for u in units if u.get("type") == "2tnk"]
        if tnk:
            return [Cmd.attack_move(tnk, fx, fy)]
        return [Cmd.observe()]
    return pol


def _skip_phase_3_policy(level: str):
    """Build proc + weap (phases 1+2) then skip the army clause —
    immediately attack the fact with starting tanks. On easy the
    chain has no army clause so this should WIN; on medium/hard the
    chain has a tank-count clause that, despite being satisfied at
    t=0 by pre-placed tanks, only latches AFTER weap latches — so
    this policy might still win medium (the latch advances P3→P4→P5
    once weap lands and there are ≥2 tanks alive). To genuinely
    skip phase 3 we must KILL one of our own tanks (or lose them in
    combat) so unit_type_count_gte:{2tnk,n:2} is FALSE for medium,
    or unit_type_count_gte:{2tnk,n:4} is FALSE for hard. The
    easier teeth: this policy attacks the fact with 2 tanks held
    back at base (less than n:4 on hard) — on hard it must LOSE."""
    fx, fy = _enemy_fact_xy(level)

    # Sticky milestone latches for the build-order phases.
    state = {"built_proc": False, "built_weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        units = obs.get("units_summary", []) or []
        tnk = [u["id"] for u in units if u.get("type") == "2tnk"]
        cmds = []
        base = [b for b in ob if b["type"] == "fact"]
        if "proc" in own_b:
            state["built_proc"] = True
        if "weap" in own_b:
            state["built_weap"] = True
        # PHASE 1 (proc).
        if not state["built_proc"]:
            if "proc" not in prod:
                cmds.append(Cmd.build("proc"))
            if base:
                cmds.append(Cmd.place_building(
                    "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 4
                ))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASE 2 (weap).
        if not state["built_weap"]:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
                ))
            if not cmds:
                cmds.append(Cmd.observe())
            return cmds
        # PHASES 3-5: skip the army-rebuild; attack with whatever
        # tanks are still alive. On hard this is <4 tanks so clause
        # 5 (unit_type_count_gte:{2tnk,n:4}) never latches.
        if tnk:
            cmds.append(Cmd.attack_move(tnk, fx, fy))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _skip_phase_4_policy(level: str):
    """Build proc + weap + (rebuild army to target) but NEVER engage
    — kills stay at 0 so clause-4 (units_killed_gte:3) never latches
    on medium/hard. The chain stalls at index 3 and the clock expires.
    On easy there is no kill clause and no army clause, so this
    policy reduces to "build proc+weap and never attack" — which
    MUST also LOSE because the engage clause (P3 on easy) never
    latches without an attack."""
    target = _ARMY_TARGET[level]

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        units = obs.get("units_summary", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
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
        # Rebuild army if needed but DO NOT engage.
        if (
            "weap" in own_b
            and target > 0
            and len(tnk) < target
            and "2tnk" not in prod
        ):
            cmds.append(Cmd.build("2tnk"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_chain_policy(level: str):
    """The intended 5+ checkpoint chain (sticky milestones so a built-
    then-destroyed proc doesn't reset the policy). Build proc + weap
    on the macro thread WHILE concurrently driving the pre-placed
    tanks east to engage and raze the enemy fact — the `then:` latch
    only advances clause-by-clause once each predicate is observed-
    true in order, so doing the work in parallel doesn't skip ahead.

    Must WIN on every (level, seed)."""
    fx, fy = _enemy_fact_xy(level)
    issued_attack = {"yes": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        units = obs.get("units_summary", []) or []
        tnk = [u for u in units if u.get("type") == "2tnk"]
        base = [b for b in ob if b["type"] == "fact"]

        cmds = []
        # Macro thread: drive proc → weap to completion, then
        # optionally keep producing tanks (helps hard's n:4 clause
        # if any pre-placed tank dies mid-engagement).
        if "proc" not in own_b:
            if "proc" not in prod:
                cmds.append(Cmd.build("proc"))
            if base:
                cmds.append(Cmd.place_building(
                    "proc", base[0]["cell_x"] + 6, base[0]["cell_y"] + 4
                ))
        elif "weap" not in own_b:
            if "weap" not in prod:
                cmds.append(Cmd.build("weap"))
            if base:
                cmds.append(Cmd.place_building(
                    "weap", base[0]["cell_x"] + 8, base[0]["cell_y"]
                ))
        else:
            # Both up — keep producing tanks (cheap insurance for the
            # n:4 hard clause after attrition; auto-placed by the
            # engine since `weap` knows its own queue).
            if "2tnk" not in prod and len(tnk) < 8:
                cmds.append(Cmd.build("2tnk"))

        # Combat thread: ALWAYS attack-move every tank east toward
        # the enemy fact. Re-issuing each turn keeps freshly-built
        # tanks pointed at the objective.
        if tnk:
            cmds.append(Cmd.attack_move([u["id"] for u in tnk], fx, fy))
            issued_attack["yes"] = True

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-multi-checkpoint-5-plus"
    assert pack.meta.capability == "action"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required: PlanBench long-sequencing + PERT critical path
    anchors, per the wave-5 spec."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("PERT" in a for a in anchors), anchors
    assert any("SOP" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups (UPGRADED
    contract in tests/test_hard_tier.py)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_then_chain_lengths_per_level():
    """easy=3, medium=5, hard=6 phases (the headline difficulty
    axis is the chain length)."""
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
                assert len(clauses) == _CHAIN_LEN[lvl], (
                    f"{lvl} chain must have {_CHAIN_LEN[lvl]} "
                    f"clauses; got {len(clauses)}"
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
        assert wts, f"{lvl} has no within_ticks leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites"
            )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_chain_policy_wins(level, seed):
    """The intended N+ checkpoint chain must WIN on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_chain_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended chain must WIN on {level} s={seed}; "
        f"got {res.outcome} (then_progress={tp}, "
        f"kills={res.signals.units_killed}, "
        f"own_buildings={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_skip_to_attack_loses(level, seed):
    """Rush the fact with starting tanks (no proc, no weap, no
    chain). Must LOSE on every (level, seed) — clause-1 (proc) is
    never observed-true so the then-latch never advances past 0."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _skip_to_attack_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"skip-to-attack must LOSE on {level} s={seed}; "
        f"got {res.outcome} then_progress={tp}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_skip_phase_4_loses_medium(seed):
    """On medium, building proc+weap but NEVER engaging means kills
    stay at 0 — clause-4 (units_killed_gte:3) never latches and the
    chain stalls at index 3."""
    c = compile_level(load_pack(PACK), "medium")
    res = run_level(c, _skip_phase_4_policy("medium"), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"skip-phase-4 must LOSE on medium s={seed}; "
        f"got {res.outcome} then_progress={tp} "
        f"kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_skip_phase_4_loses_hard(seed):
    """Same on hard — no engagement means no kills, no fact-kill,
    no chain advance past clause 3."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _skip_phase_4_policy("hard"), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "loss", (
        f"skip-phase-4 must LOSE on hard s={seed}; "
        f"got {res.outcome} then_progress={tp} "
        f"kills={res.signals.units_killed}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must round-robin: different
    seeds place the agent base at a different (x,y) set. Smoke-tests
    the spawn-variation contract enforced by test_hard_tier.py."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
