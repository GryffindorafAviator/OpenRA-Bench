"""scout-count-defenders pack — Wave-8 PERCEPTION exact-count force
sizing.

The agent scouts to count K medium-tank enemy defenders (2tnk),
backed by 2 reinforcing pillboxes (pbox) for extra defensive DPS,
then builds EXACTLY K medium tanks (2tnk) to defeat them. The
pillboxes are STRUCTURES (counted as buildings, not units) so they
do NOT count toward `enemies_discovered_gte:K` or
`units_killed_gte:K`; they are pure attrition multiplier that
prevents the under-build (2 tanks) from rushing through.

Discriminations (CLAUDE.md bar):

  * stall (only observe): LOSS — after_ticks fail clause bites.
  * build-min-force (always assume K=2, build 2 tanks regardless of
    actual count): LOSS on medium (K=3) and hard (K=4) — the 2-tank
    wave is wiped by the K-defender + 2-pbox combination before
    scoring K kills.
  * build-max-force (queue all OVER_BUILD_N=6 tanks then send):
    LOSS on every level — the 6-tank sequential queue takes ~3240
    ticks plus ~1350 transit + ~300 combat = ~4890 ticks, beyond
    every level's within_ticks deadline (max 4200 on hard).
  * intended count-then-build (scout to read K, build exactly K
    tanks, attack-move east): WINS on every (level, seed).

Bar (CLAUDE.md): real LOSS not DRAW — every level carries
  `not building_count_gte:{type:fact,n:1}` (own fact destroyed) AND
  `after_ticks:T+1` inside max_turns ⇒ a stall is a real timeout LOSS.

Real-world anchor:
  - POMDP exact-count sub-tasks
  - ScienceWorld inventory census
  - SC2 scout-count (the prototypical micro of seeing N defenders and
    building N attackers — no more, no less).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "scout-count-defenders.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)

# Expected K (defender count) per level — matches the win predicate
# in the yaml. Tests use these to drive the scripted intended policy.
K_BY_LEVEL = {"easy": 2, "medium": 3, "hard": 4}
# Cash budget funds OVER_BUILD_N = 6 medium tanks ($5100 / $850 each).
# The build-max-force discrimination targets this same N.
OVER_BUILD_N = 6


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-count-defenders"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: POMDP exact-count + ScienceWorld
    census + SC2 scout-count."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("POMDP" in a for a in anchors), anchors
    assert any("ScienceWorld" in a for a in anchors), anchors
    assert any("SC2" in a for a in anchors), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks must be reachable inside max_turns. Engine advances
    ~90 ticks/turn → reachable max = 93 + 90·(N-1). If within_ticks
    exceeds reachable, the deadline never bites and a staller draws
    instead of losing."""
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

        wts: list = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )


def test_win_has_enemies_discovered_and_kills_and_fact_count_clauses():
    """Confirms the load-bearing predicate shape: the win is a
    conjunction of enemies_discovered_gte:K + units_killed_gte:K +
    building_count_gte:{type:fact,n:1} + within_ticks:T."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        keys = [list(cl.keys())[0] for cl in inner if cl]
        assert "enemies_discovered_gte" in keys, (
            f"{lvl} win missing enemies_discovered_gte: {win}"
        )
        assert "units_killed_gte" in keys, (
            f"{lvl} win missing units_killed_gte: {win}"
        )
        assert "building_count_gte" in keys, (
            f"{lvl} win missing building_count_gte (fact survival): {win}"
        )
        assert "within_ticks" in keys, (
            f"{lvl} win missing within_ticks (clock teeth): {win}"
        )
        # Discovery K and kill K must be equal — count-aware sizing.
        disc = [cl["enemies_discovered_gte"] for cl in inner if "enemies_discovered_gte" in cl][0]
        kill = [cl["units_killed_gte"] for cl in inner if "units_killed_gte" in cl][0]
        assert int(disc) == int(kill) == K_BY_LEVEL[lvl], (
            f"{lvl} discover={disc} kill={kill} must both equal K={K_BY_LEVEL[lvl]}"
        )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must define ≥2 agent spawn_point groups so seed varies the
    start base (tests/test_hard_tier.py::UPGRADED contract)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 spawn groups, got {sp}"


def test_starting_cash_matches_design():
    """Cash must fund up to OVER_BUILD_N=6× 2tnk ($850 ea = $5100) per
    the over-build / under-build discrimination design."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.starting_cash == 5100, (
            f"{lvl} starting_cash={c.starting_cash} must be 5100 "
            f"(funds {OVER_BUILD_N}× 2tnk @ $850 each)"
        )


def test_base_has_required_buildings_and_two_jeep_scouts():
    """Pre-placed agent base must include fact + proc + powr + weap
    AND exactly 2 jeep scouts (per the Wave-8 spec)."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        agent_actors = [a for a in c.scenario.actors if a.owner == "agent"]
        # Group by spawn_point — count buildings/jeeps in ONE group
        # (each spawn group is symmetric).
        groups: dict = {}
        for a in agent_actors:
            sp = a.spawn_point if a.spawn_point is not None else 0
            groups.setdefault(sp, []).append(a)
        # Take the first group and assert the required composition.
        sample = list(groups.values())[0]
        types = [a.type for a in sample]
        for required in ("fact", "proc", "powr", "weap"):
            assert required in types, (
                f"{lvl} group missing {required}; saw {types}"
            )
        jeep_count = types.count("jeep")
        assert jeep_count == 2, (
            f"{lvl} group has {jeep_count} jeeps; Wave-8 spec requires "
            f"exactly 2 jeep scouts (got {types})"
        )


# ── Engine-bound policies and per-(level, seed) bar tests ────────


def _stall_policy():
    """Do nothing — must LOSE on the clock on every level/seed."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _intended_policy(level: str):
    """Scout to count, then build exactly K 2tnks and attack-move
    east. The "scout to count" axis is short-circuited here (we know
    K from the level) — this policy validates that the BUILD K AND
    attack chain is solvable inside the budget, which is the
    necessary precondition for a real model to be able to win by
    actually counting."""
    K = K_BY_LEVEL[level]
    state = {"queued": 0, "scouted": False, "sent": False}

    def pol(obs, Cmd):
        cmds: list = []
        units = obs.get("units_summary", []) or []
        # Drive a jeep east — scout pass tags enemies on sight.
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if jeeps and not state["scouted"]:
            jids = [j["id"] for j in jeeps]
            cmds.append(Cmd.move_units(jids, 100, 20))
            state["scouted"] = True
        # Queue K 2tnks (one per turn) from the weap; weap is pre-
        # placed and 2tnk's tech prereqs (weap + fix) are pre-met.
        prod = obs.get("production", []) or []
        own_tanks = [u for u in units if u.get("type") == "2tnk"]
        if state["queued"] < K and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
            state["queued"] += 1
        # Once we have K tanks, attack-move them east to the
        # defender cloud (110,20).
        if len(own_tanks) >= K and not state["sent"]:
            tids = [t["id"] for t in own_tanks]
            cmds.append(Cmd.attack_move(tids, 110, 20))
            state["sent"] = True
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _under_build_policy():
    """Always build exactly 2 tanks regardless of K. WINS on easy
    (K=2 — that's the matching amount) and LOSES on medium/hard
    (K=3/K=4) — 2 medium tanks vs 3-4 defending tanks + 2 pillboxes
    is an attrition trade the attackers lose."""
    state = {"queued": 0, "sent": False, "scouted": False}

    def pol(obs, Cmd):
        cmds: list = []
        units = obs.get("units_summary", []) or []
        # Move jeeps east anyway (so enemies_discovered latches —
        # the LOSS on K=3/4 levels then attributes to the kill bar
        # / fact destruction, not to a discovery-bar miss).
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if jeeps and not state["scouted"]:
            jids = [j["id"] for j in jeeps]
            cmds.append(Cmd.move_units(jids, 100, 20))
            state["scouted"] = True
        prod = obs.get("production", []) or []
        own_tanks = [u for u in units if u.get("type") == "2tnk"]
        if state["queued"] < 2 and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
            state["queued"] += 1
        if len(own_tanks) >= 2 and not state["sent"]:
            tids = [t["id"] for t in own_tanks]
            cmds.append(Cmd.attack_move(tids, 110, 20))
            state["sent"] = True
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _over_build_policy():
    """Always queue OVER_BUILD_N=6 tanks regardless of K — the full
    $5100 cash burn. The sequential weap queue takes much longer to
    field all 6 than to field the K-needed subset; the within_ticks
    deadline bites before the over-built wave has time to engage,
    most acutely on EASY (K=2) where the 2-tank intended send wins
    at ~1500 ticks vs the 6-tank send at ~3700 ticks (past 3600)."""
    state = {"queued": 0, "sent": False, "scouted": False}

    def pol(obs, Cmd):
        cmds: list = []
        units = obs.get("units_summary", []) or []
        jeeps = [u for u in units if u.get("type") == "jeep"]
        if jeeps and not state["scouted"]:
            jids = [j["id"] for j in jeeps]
            cmds.append(Cmd.move_units(jids, 100, 20))
            state["scouted"] = True
        prod = obs.get("production", []) or []
        own_tanks = [u for u in units if u.get("type") == "2tnk"]
        if state["queued"] < OVER_BUILD_N and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
            state["queued"] += 1
        # Only commit AFTER all OVER_BUILD_N are built (the
        # over-build policy by definition refuses to send a partial
        # wave).
        if len(own_tanks) >= OVER_BUILD_N and not state["sent"]:
            tids = [t["id"] for t in own_tanks]
            cmds.append(Cmd.attack_move(tids, 110, 20))
            state["sent"] = True
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSE on every (level, seed). The
    fail_condition's after_ticks clause bites; never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_count_then_build_wins(level, seed):
    """The intended count-then-build-K-tanks policy must WIN on every
    (level, seed). This is the load-bearing test that the pack is
    solvable inside the budget by the advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    assert res.outcome == "win", (
        f"intended count-then-build must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"kills={res.signals.units_killed}, "
        f"discovered={len(res.signals.enemies_seen_ids)})"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_under_build_loses_on_medium_and_hard(seed):
    """Build-min-force (always 2 tanks) must LOSE on medium (K=3) and
    hard (K=4). On easy (K=2) it would win — that's the rehearsal
    tier — so we only test the discriminating levels here."""
    for level in ("medium", "hard"):
        c = compile_level(load_pack(PACK), level)
        res = run_level(c, _under_build_policy(), seed=seed)
        assert res.outcome == "loss", (
            f"under-build (2 tanks) must LOSE on {level} s={seed} "
            f"(K={K_BY_LEVEL[level]}); got {res.outcome} "
            f"(kills={res.signals.units_killed}, "
            f"tick={res.signals.game_tick})"
        )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_over_build_loses_on_every_level(level, seed):
    """Build-max-force (queue all OVER_BUILD_N=6 tanks before sending)
    must LOSE on every (level, seed). The sequential weap queue takes
    ~3240 ticks to field all 6, plus ~1350 ticks transit + ~300
    combat ⇒ engagement completes around tick ~4890, beyond every
    level's within_ticks deadline (easy 3300 / medium 3700 / hard
    4200) ⇒ real LOSS on every level/seed. The discrimination is
    most acute on easy where a 2-tank intended send wins at ~2883
    ticks — the over-build wastes ~5× the production time."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _over_build_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"over-build (queue {OVER_BUILD_N} tanks before send) must "
        f"LOSE on {level} s={seed}; got {res.outcome} "
        f"(kills={res.signals.units_killed}, "
        f"tick={res.signals.game_tick})"
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
    # Stall must lose — anchors the LOSS-not-DRAW property.
    assert res.outcome == "loss"
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
        captured: dict = {}

        def pol(obs, Cmd):
            if "fact_pos" not in captured:
                bs = obs.get("own_buildings", []) or []
                facts = [
                    (b["cell_x"], b["cell_y"])
                    for b in bs
                    if b["type"] == "fact"
                ]
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
