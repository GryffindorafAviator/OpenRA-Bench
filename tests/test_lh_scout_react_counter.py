"""lh-scout-react-counter pack — full no-cheat validation on Rust.

Group-G long-horizon reasoning pack: observation-driven 3-phase
chain. The agent has a small fixed force (1× jeep scout + 3× 2tnk)
at the west; it must (1) SCOUT — drive the jeep east and register
≥2 enemy buildings (the construction yard AND a paired barracks),
(2) REACT — engage with the tanks to destroy ≥3 enemy units, then
(3) COUNTER — raze the enemy construction yard. Wave-2 `then:`
happened-before composite enforces the order; `within_ticks: 6300`
is the deadline; hard adds `units_lost_lte: 3` as an attrition
tooth and a third enemy posture pocket at MID-y.

Bar (per CLAUDE.md): the intended scout-react-counter policy WINS
on every (level, seed); every stall / skip-scout-attack-blind /
scout-only-no-attack policy LOSES on every level + seed. The pack
sits alongside `lh-tech-pivot-attack` (which tests build-the-
counter as the middle phase) and `mid-tech-switch-on-scout` (the
2-phase scout-then-counter sibling): this one isolates the
observation-action-counter loop on a FIXED FORCE (cash 1500 is too
lean to fund a new production line — the capability is the
reactive 3-phase chain, not the build-out).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-scout-react-counter.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    """Do nothing — must LOSE on after_ticks every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _skip_scout_attack_blind_policy():
    """Rush the tanks straight at the nearest plausible enemy fact
    coordinate without driving the scout jeep east first. The
    discovery latch (≥2 buildings) is the load-bearing first
    clause: a tank rush that smashes the fact MAY register a
    single building (the fact under attack) but the paired
    barracks (tent) sits OFF the assault axis (y-offset +4),
    outside the tank's sight while it shells the fact — so
    buildings_discovered stays at 1 and the then-chain stalls at
    clause 1. Even if a stray sight tick caught the tent, the
    chain still requires ≥3 unit kills BEFORE raze (the tank
    rush kills the cluster en route only after the fact-attack
    has begun, and the then-chain credit order matters)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = obs.get("own_buildings", []) or []
        # Pick target y by agent fact latitude (NORTH spawn → y=5,
        # SOUTH spawn → y=38, easy single-spawn → y=18).
        agent_fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if agent_fact is None:
            return [Cmd.observe()]
        ay = agent_fact["cell_y"]
        if ay < 14:
            target_y = 5
        elif ay > 24:
            target_y = 38
        else:
            target_y = 18
        tnk_ids = [u["id"] for u in units if u.get("type") == "2tnk"]
        if not tnk_ids:
            return [Cmd.observe()]
        # Attack-move directly at the enemy fact coordinate. No
        # scout move issued.
        return [Cmd.attack_move(tnk_ids, 122, target_y)]
    return pol


def _scout_only_no_attack_policy():
    """Drive the scout jeep east to register the enemy buildings
    (satisfies clause 1) but never engage with the tanks — the
    then-chain stalls at clause 2 (units_killed_gte:3), and the
    clock eventually expires. Asserts that scouting alone is not
    sufficient — the REACT phase is load-bearing."""
    state = {"went_outpost": False, "ticks_since_outpost": 0,
             "went_corner": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = obs.get("own_buildings", []) or []
        jeep = next((u for u in units if u.get("type") == "jeep"), None)
        agent_fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        if jeep is None or agent_fact is None:
            return [Cmd.observe()]
        ay = agent_fact["cell_y"]
        # Drive the jeep through the off-axis outpost (registers
        # the tent), then on to the near fact's y-band (registers
        # the fact). Both buildings get discovered, but the tanks
        # never engage so units_killed stays 0 and the chain
        # stalls at clause 2.
        if ay < 14:
            fact_y, outpost = 5, (60, 20)
        elif ay > 24:
            fact_y, outpost = 38, (60, 20)
        else:
            fact_y, outpost = 18, (60, 36)
        if not state["went_outpost"]:
            state["went_outpost"] = True
            return [Cmd.move_units([jeep["id"]], *outpost)]
        state["ticks_since_outpost"] += 1
        if state["ticks_since_outpost"] >= 8 and not state["went_corner"]:
            state["went_corner"] = True
            return [Cmd.move_units([jeep["id"]], 120, fact_y)]
        return [Cmd.observe()]
    return pol


def _intended_scout_react_counter_policy(easy_mode: bool):
    """The intended capability play. Drives the chain end-to-end:
      1. Jeep moves to the forward outpost (mid-x off-axis tent),
         then continues to the near corner — registers BOTH the
         outpost tent AND the corner fact (≥2 buildings
         discovered) ⇒ then[0] latches.
      2. Tanks attack-move at the near fact — the defending
         stance:0 cluster sits at (118, fact_y) right on the
         assault axis, so the tanks engage and kill ≥3 ⇒ then[1]
         latches.
      3. Tanks continue past the cluster and shell the fact ⇒
         then[2] (and the top-level enemy_key_buildings_destroyed)
         both latch ⇒ WIN inside 6300 ticks.
    """
    state = {"outpost_dispatched": False, "scout_dispatched": False,
             "attack_dispatched": False, "scout_ticks": 0}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = obs.get("own_buildings", []) or []
        agent_fact = next(
            (b for b in own_b if b.get("type") == "fact"), None,
        )
        jeep = next((u for u in units if u.get("type") == "jeep"), None)
        tnk_ids = [u["id"] for u in units if u.get("type") == "2tnk"]
        cmds = []

        # Pick near corner by agent latitude, and the outpost
        # coordinate that matches the level's layout.
        if easy_mode:
            fact_y = 18
            outpost = (60, 36)        # easy outpost far south
        elif agent_fact and agent_fact["cell_y"] < 18:
            fact_y = 5
            outpost = (60, 20)        # medium/hard shared mid outpost
        else:
            fact_y = 38
            outpost = (60, 20)

        # 1a. SCOUT — first dispatch jeep to the off-axis outpost.
        if jeep is not None and not state["outpost_dispatched"]:
            cmds.append(Cmd.move_units([jeep["id"]], *outpost))
            state["outpost_dispatched"] = True
        # 1b. Then redirect jeep east to the near fact's y-band
        #     (give it a few turns to traverse to the outpost
        #     and surface the tent in enemy_buildings).
        elif (jeep is not None
              and state["outpost_dispatched"]
              and not state["scout_dispatched"]):
            state["scout_ticks"] += 1
            if state["scout_ticks"] >= 8:
                cmds.append(Cmd.move_units([jeep["id"]], 120, fact_y))
                state["scout_dispatched"] = True

        # 2 + 3. REACT + COUNTER — dispatch the tanks early; one
        #    attack_move covers both phases (they engage the
        #    stance:0 cluster en route, then continue to raze
        #    the fact). The then-chain's first clause may not
        #    yet be latched when the tanks start moving — that's
        #    fine, the greedy advance picks up clauses in order
        #    once each becomes true, and the rush takes long
        #    enough that the jeep's scout completes first.
        if tnk_ids and not state["attack_dispatched"]:
            cmds.append(Cmd.attack_move(tnk_ids, 122, fact_y))
            state["attack_dispatched"] = True

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-scout-react-counter"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: SC2 reactive macro / CICERO
    info-loop / PlanBench replanning / threat-intel anchors."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2" in a for a in anchors), anchors
    assert any("CICERO" in a for a in anchors), anchors
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("threat-intel" in a for a in anchors), anchors


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies
    the start base (binding contract from tests/test_hard_tier.py)."""
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
    """The 3-phase scout-react-counter chain must be wired through
    to the compiled win condition (the whole point of the pack)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        ao = win.get("all_of") or []
        then_branches = [b for b in ao if "then" in b]
        assert len(then_branches) >= 1, f"{lvl} missing then-chain: {win}"
        clauses = then_branches[0]["then"]["clauses"]
        assert len(clauses) == 3, (
            f"{lvl} then-chain must be 3 clauses (scout, react, "
            f"counter), got {len(clauses)}"
        )
        # 1. scout latch — ≥2 buildings discovered (NOT ≥1; the
        #    spec makes scouting load-bearing by requiring a
        #    second building, defeating the rush-the-fact play).
        assert clauses[0].get("buildings_discovered_gte") == 2, clauses[0]
        # 2. react — kill at least 3 enemy units.
        assert clauses[1].get("units_killed_gte") == 3, clauses[1]
        # 3. counter — raze the enemy fact.
        assert "enemy_key_buildings_destroyed" in clauses[2], clauses[2]


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine
    advances ~90 ticks/turn → reachable max = 93 + 90·(N-1)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(
            exclude_none=True,
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

        wts = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


def test_tools_list_matches_spec():
    """Tools per spec: [observe, build, place_building, move_units,
    attack_unit, attack_move, stop]."""
    pack = load_pack(PACK)
    expected = {"observe", "build", "place_building", "move_units",
                "attack_unit", "attack_move", "stop"}
    tools = pack.base.get("tools") if isinstance(pack.base, dict) \
        else pack.base.tools
    assert set(tools) == expected, tools


def test_starting_cash_is_lean():
    """Spec calls for starting_cash: 1500 across all levels — the
    capability is the reactive 3-phase chain on the FIXED force,
    not a build-out funded from a generous war chest."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        assert pack.levels[lvl].starting_cash == 1500, lvl


def test_hard_has_attrition_cap():
    """Hard adds units_lost_lte:3 as the +1 controlled axis (one
    new variable per tier per the curation contract)."""
    c = compile_level(load_pack(PACK), "hard")
    win = c.win_condition.model_dump(exclude_none=True)

    def _has(node, key):
        if isinstance(node, dict):
            if key in node:
                return True
            return any(_has(v, key) for v in node.values())
        if isinstance(node, list):
            return any(_has(v, key) for v in node)
        return False

    assert _has(win, "units_lost_lte"), (
        "hard must include units_lost_lte tooth"
    )


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_react_counter_wins(level, seed):
    """The intended scout → react → counter capability play must
    WIN on every (level, seed). This is the load-bearing test
    that the pack is solvable inside the budget by the advertised
    capability across all hard-tier spawn variants."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_scout_react_counter_policy(easy_mode=(level == "easy")),
        seed=seed,
    )
    assert res.outcome == "win", (
        f"intended scout-react-counter must WIN on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"lost={res.signals.units_lost}, "
        f"then_progress={getattr(res.signals, 'then_progress', {})})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE (no win, no draw) on every
    (level, seed). The fail_condition's after_ticks clause bites
    at the turn budget; never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_skip_scout_attack_blind_loses(level, seed):
    """A blind tank-rush at the fact (no scout first) must LOSE
    on every (level, seed). The then-chain requires ≥2 buildings
    discovered FIRST; a tank rush only registers the fact under
    attack (1 building) and the chain stalls at clause 1 — the
    fact may eventually fall but the chain credit was never
    earned in order, so within_ticks expires without a WIN. This
    is the load-bearing 'scout is mandatory' tooth."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _skip_scout_attack_blind_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"skip-scout-attack-blind must LOSE on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"then_progress={getattr(res.signals, 'then_progress', {})})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_scout_only_no_attack_loses(level, seed):
    """A scout-only policy (jeep registers the buildings but the
    tanks never engage) must LOSE on every (level, seed). The
    then-chain stalls at clause 2 (units_killed_gte:3) — the
    REACT phase is load-bearing. The clock eventually expires."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _scout_only_no_attack_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"scout-only-no-attack must LOSE on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"then_progress={getattr(res.signals, 'then_progress', {})})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    different seeds must place the agent base at a different (x,y)
    set. Smoke-tests the spawn-variation contract."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss"  # stall must lose
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
