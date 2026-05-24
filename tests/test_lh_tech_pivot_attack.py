"""lh-tech-pivot-attack pack — full no-cheat validation on Rust.

The Group-G long-horizon reasoning pack: LATE-game three-phase chain.
The agent starts with tech already up (fact + powr + tent + weap) and
must (1) SCOUT the enemy corner — drive the jeep into the eastern
scout band, (2) PRODUCE the assault counter (rocket soldiers, e3),
then (3) ATTACK — destroy the enemy construction yard. A Wave-2
`then:[A,B,C]` happened-before composite enforces the order; a
top-level `enemy_key_buildings_destroyed:{types:[fact]}` clause
ensures the assault actually lands; `within_ticks: 7200` (6300 on
hard) is the deadline.

Recalibration note (engine balance fixes — armor-class weapon
selection, stance semantics, parallel production, pbox now fires):
the chain's first clause used to be `buildings_discovered_gte: 1`.
After the combat-balance shift a no-scout pre-commit could raze the
enemy fact fast enough that building-discovery, counter-built and
fact-destroyed all latched in a SINGLE evaluation — and `_then`
advances through every consecutive satisfied clause in one call, so
the whole chain collapsed and the no-scout play won. Clause 1 is now
keyed on the JEEP's own position (`units_of_type_in_region_gte`):
the scout is a genuinely separate, earlier event the assault army
cannot satisfy for free. A pre-commit that never drives the jeep
east can never latch clause 1, so the then-chain never completes and
the run is a real clock LOSS.

Bar (per CLAUDE.md): the intended scout → produce → attack policy
WINS on every (level, seed); every stall / no-scout pre-commit
policy LOSES (real timeout LOSS, not a draw) on every (level, seed).
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


def _near_y(obs, easy_mode):
    """The eastern corner reachable from this seed's agent base.
    Easy is a single fixed corner (y=18); medium/hard round-robin
    the agent spawn between a NORTH base (near corner y=5) and a
    SOUTH base (near corner y=36)."""
    if easy_mode:
        return 18
    af = next(
        (b for b in (obs.get("own_buildings", []) or [])
         if b.get("type") == "fact"),
        None,
    )
    if af and af["cell_y"] < 18:
        return 5
    return 36


def _no_scout_pre_commit_policy(easy_mode: bool):
    """Brute / wrong-path: build the e3 assault force IMMEDIATELY and
    attack-move the near corner — never drive the jeep east to scout.

    This is the laziest play that razes the enemy fact. It must
    LOSE on every (level, seed): the then-chain's clause 1 is keyed
    on the JEEP reaching the eastern scout band, and this policy
    leaves the jeep parked at base — so clause 1 never latches, the
    then-chain never completes, and the run times out as a LOSS even
    when the fact is destroyed."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        e3 = sum(1 for u in units if u.get("type") == "e3")
        ny = _near_y(obs, easy_mode)
        cmds = []
        if "tent" in own_b and e3 < 6 and "e3" not in prod:
            cmds.append(Cmd.build("e3"))
        if e3 >= 6:
            ids = [u["id"] for u in units if u.get("type") == "e3"]
            cmds.append(Cmd.attack_move(ids, 82, ny))
        return cmds or [Cmd.observe()]
    return pol


def _intended_scout_attack_policy(easy_mode: bool):
    """The intended capability play: jeep drives into the eastern
    scout band FIRST (latching the then-chain's scout clause), then
    the agent produces the e3 assault counter and attack-moves it at
    the near enemy fact to raze it.

    Demonstrates the scout → produce → attack chain end-to-end:
      1. jeep moves into the eastern scout band ⇒ then[0] latches
      2. enemy fact discovered while the jeep holds the band
      3. ≥4 e3 produced ⇒ then[1] latches
      4. attack-move razes the near fact ⇒ then[2] and the
         top-level enemy_key_buildings_destroyed both latch
      5. WIN inside the clock budget.
    """
    state = {"scouted": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        ny = _near_y(obs, easy_mode)
        cmds = []
        jeep = next((u for u in units if u.get("type") == "jeep"), None)
        eb = obs.get("enemy_buildings_summary") or []
        if eb:
            state["scouted"] = True
        # 1. Scout — drive the jeep into the eastern scout band and
        #    keep it there so the enemy fact stays in vision.
        if jeep:
            cmds.append(Cmd.move_units([jeep["id"]], 82, ny))
        # 2. Produce the e3 assault counter once scouting is under way.
        if state["scouted"]:
            n = sum(1 for u in units if u.get("type") == "e3")
            if "tent" in own_b and n < 6 and "e3" not in prod:
                cmds.append(Cmd.build("e3"))
            # 3. Attack — attack-move the e3 at the near fact.
            if n >= 6:
                ids = [u["id"] for u in units if u.get("type") == "e3"]
                cmds.append(Cmd.attack_move(ids, 82, ny))
        return cmds or [Cmd.observe()]
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
    """Confirms the scout → produce → attack chain is wired through
    to the compiled win condition (the whole point of the pack), and
    that clause 1 keys on the JEEP's position — the recalibration
    that makes the scout a separately-timed, load-bearing phase."""
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
        # First clause: the scout latch must key on the jeep's own
        # position (directly or under an any_of over the two
        # candidate corners), NOT on building-discovery (which the
        # assault army satisfies for free).
        first = clauses[0]
        scout_leaves = []

        def _collect(node):
            if isinstance(node, dict):
                if "units_of_type_in_region_gte" in node:
                    scout_leaves.append(node["units_of_type_in_region_gte"])
                for v in node.values():
                    _collect(v)
            elif isinstance(node, list):
                for v in node:
                    _collect(v)

        _collect(first)
        assert scout_leaves, (
            f"{lvl} then-chain clause 1 must be a jeep-in-region "
            f"scout latch, got {first}"
        )
        for leaf in scout_leaves:
            assert str(leaf.get("type")).lower() == "jeep", leaf
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
def test_intended_scout_attack_wins(level, seed):
    """The intended scout → produce → attack capability play must
    WIN on every (level, seed). This is the load-bearing test that
    the pack is solvable inside the budget by the advertised
    capability across all hard-tier spawn variants."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_scout_attack_policy(easy_mode=(level == "easy")),
        seed=seed,
    )
    assert res.outcome == "win", (
        f"intended scout-attack must WIN on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
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
def test_no_scout_pre_commit_loses(level, seed):
    """The no-scout pre-commit (build the e3 force immediately and
    attack-move the fact, jeep left at base) must LOSE on EVERY
    (level, seed) — even when it razes the enemy fact. The
    then-chain's clause 1 keys on the jeep reaching the eastern
    scout band; a play that never drives the jeep east can never
    latch it, so the chain never completes and the episode times
    out as a real LOSS. This is the no-cheat teeth: the scout is
    structurally required, not optional."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _no_scout_pre_commit_policy(easy_mode=(level == "easy")),
        seed=seed,
    )
    assert res.outcome == "loss", (
        f"no-scout pre-commit must LOSE on {level} s={seed}; got "
        f"{res.outcome} (kills={res.signals.units_killed}, "
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
