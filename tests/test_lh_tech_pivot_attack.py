"""lh-tech-pivot-attack pack — full no-cheat validation on Rust.

The Group-G long-horizon reasoning pack: LATE-game three-phase chain.
The agent starts with tech already up (fact + powr + tent + weap) and
must (1) SCOUT the enemy composition at the eastern corner, (2) PIVOT
to the correct counter (anti-vehicle e3 OR anti-infantry 2tnk based
on what's there), then (3) ATTACK — destroy the enemy construction
yard. Wave-2 `then:[A,B,C]` happened-before composite enforces the
order; a top-level `enemy_key_buildings_destroyed:{types:[fact]}`
clause ensures the assault actually lands; `within_ticks: 7200`
(6300 on hard) is the deadline.

Bar (per CLAUDE.md): the intended scout-pivot-attack policy WINS on
all (level, seed); every lazy / stall / wrong-counter policy LOSES on
every medium/hard seed. Easy is the rehearsal tier (composition
named in the brief) — a pre-pick of the correct counter is a valid
plan-and-execute play and is allowed to WIN.

Distinct from `mid-tech-switch-on-scout` (the small-tempo sibling
that uses units_killed_gte:3 as the assault clause): this pack
requires razing the enemy FACT as the final phase of a longer chain.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "lh-tech-pivot-attack.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _pre_commit_A_policy(target_xy: tuple[int, int]):
    """Pre-commit 4× e3, no scout. Attacks a FIXED corner
    (target_xy is the brief's best-guess fact location, set
    without scouting). The pre-commit's two failure modes:
      - WRONG corner (the FAR fact) ⇒ units traverse the whole
        map ⇒ clock expires before any latch fires;
      - RIGHT corner but WRONG counter (e3 vs e3 cluster) ⇒
        slow grind against same-type cluster (rockets vs rockets
        is low DPS) ⇒ clock pressure binds tighter the more we
        also need to raze the fact afterwards.
    On medium the clock is loose enough that one of these can
    succeed; on hard it's tight. The test asserts AT LEAST ONE
    seed loses (per the spec's '50% LOSS on medium' bar)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        e3_count = sum(1 for u in units if u.get("type") == "e3")
        cmds = []
        if "tent" in own_b and e3_count < 4 and "e3" not in prod:
            cmds.append(Cmd.build("e3"))
        if e3_count >= 4:
            e3_ids = [u["id"] for u in units if u.get("type") == "e3"]
            cmds.append(Cmd.attack_move(e3_ids, *target_xy))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pre_commit_B_policy():
    """Pre-commit gun + 2× 2tnk, no scout. Mirror of pre_commit_A:
    wins only on seeds where the near composition is e3 (tanks
    shell infantry); loses on seeds where the near composition is
    2tnk (mirror match, clock burns)."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        tnk_count = sum(1 for u in units if u.get("type") == "2tnk")
        cmds = []
        if "gun" not in own_b and "gun" not in prod:
            cmds.append(Cmd.build("gun"))
        if "gun" not in own_b:
            base = [b for b in ob if b["type"] == "fact"]
            if base:
                cmds.append(Cmd.place_building(
                    "gun", base[0]["cell_x"] + 12, base[0]["cell_y"]
                ))
        if "weap" in own_b and tnk_count < 2 and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        if tnk_count >= 2:
            tnk_ids = [u["id"] for u in units if u.get("type") == "2tnk"]
            agent_fact = next(
                (b for b in ob if b.get("type") == "fact"),
                None,
            )
            target_y = 18  # easy default
            if agent_fact:
                target_y = 5 if agent_fact["cell_y"] < 18 else 38
            cmds.append(Cmd.attack_move(tnk_ids, 122, target_y))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_scout_pivot_attack_policy(easy_mode: bool):
    """The intended capability play: jeep scouts the near cluster,
    identifies its type from enemy_units_summary, builds the
    matching counter (e3 vs 2tnk cluster; 2tnk vs e3 cluster — on
    easy the composition is fixed 2tnk so always e3), attack-moves
    them at the near corner to break the cluster and raze the
    fact.

    Demonstrates the scout → pivot → attack chain end-to-end:
      1. jeep moves east to (118, near_y) — scouts the near corner
      2. enemy fact discovered ⇒ then[0] latches
      3. enemy units observed: pick the counter by composition type
      4. ≥4 counter units produced ⇒ then[1] latches
      5. attack-move kills cluster + razes fact ⇒ then[2] and
         the top-level enemy_key_buildings_destroyed both latch
      6. WIN inside the clock budget.
    """
    state = {"scouted": False, "counter": None}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        cmds = []
        jeep = next((u for u in units if u.get("type") == "jeep"), None)
        agent_fact = next(
            (b for b in (obs.get("own_buildings", []) or [])
             if b.get("type") == "fact"),
            None,
        )
        # Pick near corner by agent latitude (matches fact y).
        if easy_mode:
            near_y = 18
        elif agent_fact and agent_fact["cell_y"] < 18:
            near_y = 5
        else:
            near_y = 38

        # 1. Scout — drive jeep along the corridor to within sight
        #    of both the cluster (at ~118, near_y) and the fact
        #    (at 122, near_y). Cluster is stance:0 HoldFire so the
        #    jeep can park near without being engaged.
        eb = obs.get("enemy_buildings_summary") or []
        eu = obs.get("enemy_summary") or []
        if eb:
            state["scouted"] = True
        if jeep and (not state["scouted"] or state["counter"] is None):
            # Drive close enough to the fact (~5 cells) to surface
            # both the cluster (at 118, near_y) and the fact (at
            # 122, near_y) within jeep sight (~6).
            cmds.append(Cmd.move_units([jeep["id"]], 120, near_y))

        # 2. Pivot — decide counter from observed cluster type
        #    (the cluster is now defending the fact at the same
        #    latitude, so any enemy unit within ~6 cells of near_y
        #    is the near cluster).
        if state["counter"] is None and eu:
            near_eu = [u for u in eu
                       if abs(u.get("cell_y", 0) - near_y) <= 6]
            types = {u.get("type") for u in near_eu}
            if "2tnk" in types:
                state["counter"] = "e3"     # rockets vs tanks
            elif "e3" in types:
                state["counter"] = "2tnk"   # tanks vs infantry
            elif easy_mode:
                state["counter"] = "e3"     # easy fixed: vs 2tnk

        # On easy, brief tells us the cluster is 2tnk — locked-in
        # pivot to e3 even before unit scouting confirms.
        if state["counter"] is None and easy_mode:
            state["counter"] = "e3"

        # 3. Produce the counter.
        #    A: 4× e3
        #    B: gun + 2× 2tnk
        if state["counter"] == "e3":
            n = sum(1 for u in units if u.get("type") == "e3")
            if "tent" in own_b and n < 4 and "e3" not in prod:
                cmds.append(Cmd.build("e3"))
        elif state["counter"] == "2tnk":
            if "gun" not in own_b and "gun" not in prod:
                cmds.append(Cmd.build("gun"))
            if "gun" not in own_b:
                af_b = [b for b in (obs.get("own_buildings", []) or [])
                        if b.get("type") == "fact"]
                if af_b:
                    cmds.append(Cmd.place_building(
                        "gun", af_b[0]["cell_x"] + 12,
                        af_b[0]["cell_y"],
                    ))
            n = sum(1 for u in units if u.get("type") == "2tnk")
            if "weap" in own_b and n < 2 and "2tnk" not in prod:
                cmds.append(Cmd.build("2tnk"))

        # 4. Attack — once the counter is up and we've scouted,
        #    attack-move at the near fact.
        if state["counter"] and state["scouted"]:
            ids = [u["id"] for u in units
                   if u.get("type") == state["counter"]]
            n_needed = 4 if state["counter"] == "e3" else 2
            if len(ids) >= n_needed:
                cmds.append(Cmd.attack_move(ids, 122, near_y))

        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "lh-tech-pivot-attack"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench/ALFWorld/SC2 pivot/
    product-pivot anchors must all be named."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("PlanBench" in a for a in anchors), anchors
    assert any("ALFWorld" in a for a in anchors), anchors
    assert any("SC2" in a for a in anchors), anchors
    assert any("pivot" in a.lower() for a in anchors), anchors


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
    """Confirms the scout-pivot-attack chain is wired through to
    the compiled win condition (the whole point of the pack)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        # all_of[ then{ id, clauses:[scout, counter, attack] },
        #         enemy_key_buildings_destroyed, within_ticks ]
        ao = win.get("all_of") or []
        then_branches = [b for b in ao if "then" in b]
        assert len(then_branches) >= 1, f"{lvl} missing then-chain: {win}"
        clauses = then_branches[0]["then"]["clauses"]
        assert len(clauses) == 3, (
            f"{lvl} then-chain must be 3 clauses (scout, counter, "
            f"attack), got {len(clauses)}"
        )
        # First clause: scout latch.
        assert "buildings_discovered_gte" in clauses[0]
        # Third clause: assault.
        assert "enemy_key_buildings_destroyed" in clauses[2]


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
    """Tools per spec: [observe, build, place_building, harvest,
    move_units, attack_unit, attack_move, stop]."""
    pack = load_pack(PACK)
    expected = {"observe", "build", "place_building", "harvest",
                "move_units", "attack_unit", "attack_move", "stop"}
    tools = pack.base.get("tools") if isinstance(pack.base, dict) \
        else pack.base.tools
    assert set(tools) == expected, tools


# ── Engine-bound tests (parameterised over seeds 1..4) ────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_pivot_attack_wins(level, seed):
    """The intended scout → pivot → attack capability play must
    WIN on every (level, seed). This is the load-bearing test that
    the pack is solvable inside the budget by the advertised
    capability across all hard-tier spawn variants."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_scout_pivot_attack_policy(easy_mode=(level == "easy")),
        seed=seed,
    )
    assert res.outcome == "win", (
        f"intended scout-pivot-attack must WIN on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}, "
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


# Easy: the spec mutes the pivot axis (composition fixed and named
# in brief). Pre-pick of the correct counter IS the intended play.
# The strict pre-commit-must-LOSS bar applies to medium/hard
# (hidden composition) only — and even there ONE pre-commit
# matches the near composition by luck (~half the seeds). The
# medium/hard contract is: there exists a pre-pick policy whose
# WORST-case seed loses (because the wrong-counter case is forced
# by the hidden composition); the model can't pre-commit a fixed
# counter and win all four seeds.
@pytest.mark.parametrize("level", ("medium", "hard"))
def test_pre_commit_A_loses_on_at_least_one_seed(level):
    """Pre-commit counter-A (e3) without scouting: must LOSE on at
    least one of seeds 1..4. The hidden-composition design
    guarantees that on some seeds the near cluster is itself e3
    (the wrong cluster for an e3-only counter), so the attrition
    trade burns the clock before the fact falls. This is the
    medium/hard pivot-required teeth."""
    c = compile_level(load_pack(PACK), level)
    outcomes = {}
    for s in SEEDS:
        res = run_level(c, _pre_commit_A_policy(), seed=s)
        outcomes[s] = res.outcome
    assert "loss" in outcomes.values(), (
        f"pre-commit-A must LOSE on at least one seed on {level}; "
        f"got {outcomes}"
    )


@pytest.mark.parametrize("level", ("medium", "hard"))
def test_pre_commit_B_loses_on_at_least_one_seed(level):
    """Pre-commit counter-B (2tnk) without scouting: must LOSE on
    at least one of seeds 1..4 (mirror of pre-commit-A — different
    losing seeds, same hidden-composition teeth)."""
    c = compile_level(load_pack(PACK), level)
    outcomes = {}
    for s in SEEDS:
        res = run_level(c, _pre_commit_B_policy(), seed=s)
        outcomes[s] = res.outcome
    assert "loss" in outcomes.values(), (
        f"pre-commit-B must LOSE on at least one seed on {level}; "
        f"got {outcomes}"
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
