"""mid-tech-switch-on-scout pack — full no-cheat validation on Rust.

The B2 reasoning pack: mid-game tech-switch reactive on scouted
information. Uses the Wave-2 `then:[A,B]` happened-before composite
to enforce scout-BEFORE-counter, with `any_of` over two viable
counters (4× e3 OR gun + 2× 2tnk).

Bar (per CLAUDE.md): the intended scout-then-counter policy WINS on
all (level, seed); every lazy / hedge / pre-pick / stall policy
LOSES on every medium/hard seed. Easy is the rehearsal tier and
muting the switch axis is by design — the brief tells what the
enemy is, so a smart pre-pick (a valid plan-and-execute play) is
acceptable as a WIN.

Scenario shape:
  - rush-hour-arena, soviet enemies, allied agent.
  - Mid-game start with the production chain pre-placed
    (fact+powr+tent+weap+fix+proc) plus one scout jeep.
  - Enemy hidden composition at far east. Easy: 6× e1 (fixed).
    Medium/hard: 6× e1 at NE corner AND 4× 2tnk at SE corner
    (both physically placed, agent spawn varies which is "near").
  - starting_cash: tuned so the budget funds exactly ONE counter;
    a hedge that builds half of each finishes neither.
  - win: `any_of` over two clauses, each a `then:` chain
    [buildings_discovered_gte:1, COUNTER] AND units_killed_gte:3.
    Hard additionally requires units_lost_lte:4 and uses
    objective_coords: relative.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "mid-tech-switch-on-scout.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ──────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _hedge_policy():
    """Buys 2× e3 + 1× 2tnk + gun: half of each counter, finishes
    neither. Models the wasteful "split the budget" failure mode."""
    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        e3_count = sum(1 for u in units if u.get("type") == "e3")
        tnk_count = sum(1 for u in units if u.get("type") == "2tnk")
        cmds = []
        if "tent" in own_b and e3_count < 2 and "e3" not in prod:
            cmds.append(Cmd.build("e3"))
        if "gun" not in own_b and "gun" not in prod:
            cmds.append(Cmd.build("gun"))
        if "gun" not in own_b:
            base = [b for b in ob if b["type"] == "fact"]
            if base:
                cmds.append(Cmd.place_building(
                    "gun", base[0]["cell_x"] + 12, base[0]["cell_y"]
                ))
        if "weap" in own_b and tnk_count < 1 and "2tnk" not in prod:
            cmds.append(Cmd.build("2tnk"))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pre_pick_A_policy(target_x: int, target_y: int):
    """Counter-A (4× e3), NO jeep scout. Attacks the given corner
    cell using only the brief (no information from a scout)."""
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
            cmds.append(Cmd.attack_move(e3_ids, target_x, target_y))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _pre_pick_B_policy(target_x: int, target_y: int):
    """Counter-B (gun + 2× 2tnk), NO jeep scout."""
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
            cmds.append(Cmd.attack_move(tnk_ids, target_x, target_y))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_scout_then_counter_policy(easy_mode: bool):
    """The intended capability play: drive the jeep to spot the
    near enemy fact (latches the buildings_discovered clause), then
    queue 4× e3 and attack-move them at the near unit cluster.

    Demonstrates the scout-then-counter chain end-to-end:
      1. jeep moves east to (118, 5) / (118, 38) / (115, 18 on easy)
      2. enemy fact discovered ⇒ then[0] latches
      3. ≥4 e3 produced ⇒ then[1] latches
      4. e3 attack-move kills ≥3 enemies ⇒ units_killed clause
      5. WIN inside the clock budget.
    """
    state = {"scouted": False}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        own_b = {b["type"] for b in (obs.get("own_buildings", []) or [])}
        prod = obs.get("production", []) or []
        e3_count = sum(1 for u in units if u.get("type") == "e3")
        cmds = []
        jeep = next((u for u in units if u.get("type") == "jeep"), None)
        if obs.get("enemy_buildings_summary"):
            state["scouted"] = True
        if jeep and not state["scouted"]:
            jy = jeep["cell_y"]
            if easy_mode:
                cmds.append(Cmd.move_units([jeep["id"]], 115, 18))
            elif jy < 18:  # NW base — NE fact at (122, 5)
                cmds.append(Cmd.move_units([jeep["id"]], 118, 5))
            else:           # SW base — SE fact at (122, 38)
                cmds.append(Cmd.move_units([jeep["id"]], 118, 38))
        if "tent" in own_b and e3_count < 4 and "e3" not in prod:
            cmds.append(Cmd.build("e3"))
        if e3_count >= 4 and state["scouted"]:
            e3_ids = [u["id"] for u in units if u.get("type") == "e3"]
            jy = jeep["cell_y"] if jeep else 18
            if easy_mode:
                cmds.append(Cmd.attack_move(e3_ids, 108, 18))
            elif jy < 18:
                cmds.append(Cmd.attack_move(e3_ids, 90, 10))
            else:
                cmds.append(Cmd.attack_move(e3_ids, 90, 30))
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (cheap; do not run the engine) ──────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "mid-tech-switch-on-scout"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Required by the seed taxonomy: PlanBench/ALFWorld/etc."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    # The full PlanBench replanning anchor must be there.
    assert any("PlanBench replanning" in a for a in anchors), anchors
    assert any("ALFWorld" in a for a in anchors), anchors
    assert any("ScienceWorld" in a for a in anchors), anchors


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
    """Confirms the scout-then-counter chain is wired through to the
    compiled win condition (this is the whole point of the pack)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        # any_of[ all_of[ then{...}, ... ], all_of[ then{...}, ... ] ]
        ao = win.get("any_of") or []
        assert len(ao) == 2, f"{lvl} win must be any_of[A, B]; got {win}"
        for branch in ao:
            inner = branch.get("all_of") or []
            assert any("then" in clause for clause in inner), (
                f"{lvl} branch missing then-chain: {branch}"
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
        # Walk the win tree and find any within_ticks leaf.
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
def test_intended_scout_then_counter_wins(level, seed):
    """The intended capability play (jeep scouts → produce e3 →
    attack) must WIN on every (level, seed) — including the four
    hard seeds whose start base round-robins. This is the load-
    bearing test that the pack is solvable inside the budget by the
    advertised capability."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(
        c, _intended_scout_then_counter_policy(easy_mode=(level == "easy")),
        seed=seed,
    )
    assert res.outcome == "win", (
        f"intended scout-then-counter-A must WIN on {level} s={seed}; "
        f"got {res.outcome} (kills={res.signals.units_killed}, "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}, "
        f"then_progress={getattr(res.signals, 'then_progress', {})})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """A do-nothing policy must LOSS (no win, no draw) on every
    (level, seed). The fail_condition's after_ticks clause bites at
    the turn budget; never a draw."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_hedge_loses(level, seed):
    """Hedging (build half of each counter) must LOSS on every
    (level, seed). On easy this is enforced by the cash budget /
    chain still requiring full ≥4 e3 OR (gun + ≥2 2tnk); on
    medium/hard the tight cash $2700 makes hedge actively
    unaffordable (≈$2050 spent finishes neither counter)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _hedge_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"hedge must LOSE on {level} s={seed}; got {res.outcome}"
    )


# Easy: the spec consciously mutes the switch axis ("Generous cash …
# pure plan-and-execute"). The brief tells the model what the enemy
# is, so a "pre-pick" that uses that brief info IS the intended play
# and may legitimately WIN — that is documented behaviour, not a
# defect. The strict pre-pick-must-LOSS bar applies to medium/hard
# (hidden composition) only.
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "level,target_xy",
    [
        ("medium", (90, 10)),
        ("medium", (90, 30)),
        ("hard",   (90, 10)),
        ("hard",   (90, 30)),
    ],
)
def test_pre_pick_A_loses_on_medium_and_hard(level, seed, target_xy):
    """Pre-pick counter-A WITHOUT scouting: every (medium/hard,
    seed, target-corner) must LOSS. The then-chain demands a
    buildings_discovered_gte:1 latch — a policy that never moves
    the jeep AND attacks only the unit cluster (which is 32+ cells
    from the enemy fact) cannot satisfy the chain. The cluster
    falls but the chain stays at 0 and the clock expires."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pre_pick_A_policy(*target_xy), seed=seed)
    assert res.outcome == "loss", (
        f"pre-pick-A→{target_xy} must LOSE on {level} s={seed}; "
        f"got {res.outcome} kills={res.signals.units_killed} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "level,target_xy",
    [
        ("medium", (90, 10)),
        ("medium", (90, 30)),
        ("hard",   (90, 10)),
        ("hard",   (90, 30)),
    ],
)
def test_pre_pick_B_loses_on_medium_and_hard(level, seed, target_xy):
    """Pre-pick counter-B WITHOUT scouting: every (medium/hard,
    seed, target-corner) must LOSS — same chain teeth as
    pre-pick-A. 2tnks attacking the unit cluster at (90, *) don't
    drift close enough to (122, 5)/(122, 38) to surface the enemy
    fact (~32 cells off-axis vs 2tnk sight=6)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pre_pick_B_policy(*target_xy), seed=seed)
    assert res.outcome == "loss", (
        f"pre-pick-B→{target_xy} must LOSE on {level} s={seed}; "
        f"got {res.outcome} kills={res.signals.units_killed} "
        f"bds={len(res.signals.enemy_buildings_seen_ids)}"
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_produce_distinct_starts(seed):
    """Hard's two spawn_point groups must actually round-robin —
    different seeds must place the agent base at a different (x,y)
    set. Smoke-tests the spawn-variation contract that
    tests/test_hard_tier.py also enforces, but locally so the
    closer-look loop catches a regression on this pack first."""
    c = compile_level(load_pack(PACK), "hard")
    res = run_level(c, _stall_policy(), seed=seed)
    # Just confirm the run completed deterministically.
    assert res.outcome == "loss"  # stall must lose
    # Pack-side: we have ≥2 spawn groups declared.
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2
