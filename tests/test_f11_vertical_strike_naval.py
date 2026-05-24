"""F11 — `f11-vertical-strike-naval` (combined-arms ground + naval).

The model must build BOTH a War Factory AND a Shipyard, field BOTH
medium tanks AND at least one destroyer, then raze enemy buildings
split across two reachability domains: a COASTAL building (in
destroyer shore-strike range from the water channel) AND a deep
INLAND building (far past any naval reach — only ground reaches).

Bar (CLAUDE.md "no defect, no cheat"):
- stall (observe-only)        LOSES every (tier, seed).
- brute  (rush harv east)     LOSES every (tier, seed).
- all-ground (weap + tanks)   LOSES every (tier, seed) — clause
                              P2 (`has_building: syrd`) never
                              latches; `then:` chain stalls.
- all-navy  (syrd + dds)      LOSES every (tier, seed) — clause
                              P1 (`has_building: weap`) never
                              latches; on med/hard the overland
                              tank rush also breaches the base.
- intended  (combined arms)   WINS every (tier, seed).

Plus structural pins:
- `then:` chain is wired through into the compiled win condition
  (weap → syrd → force → kills).
- `within_ticks` ≤ reachable max (`93 + 90·(max_turns - 1)`); the
  deadline bites for a real LOSS, not DRAW.
- `termination.enemy_units_killed: false` (F11 §79 mandatory).
- Hard tier has ≥2 agent spawn_point groups (NORTH y=12 / SOUTH
  y=28) so seed varies the start corner.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "f11-vertical-strike-naval.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — must LOSE on the clock on every (tier, seed)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_rush_policy():
    """Drive starter harv east toward enemy base. No production —
    fails P1/P2 and the deadline."""
    state = {"issued": set()}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        cmds = []
        for u in units:
            uid = u["id"]
            if uid in state["issued"]:
                continue
            cmds.append(Cmd.attack_move([uid], 95, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_ground_policy(level):
    """Build weap + 2tnk only; never syrd/dd. Win clause P2
    (`has_building: syrd`) never latches → `then:` stalls →
    deadline LOSS."""
    tank_target = {"easy": 4, "medium": 5, "hard": 6}[level]
    state = {"tank_orders": 0, "attack_issued": set()}
    milestone = {"weap": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "weap" in own_b:
            milestone["weap"] = True
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        if milestone["weap"] and len(tanks) < tank_target:
            if "2tnk" not in prod_items and len(tanks) >= state["tank_orders"]:
                cmds.append(Cmd.build("2tnk"))
                state["tank_orders"] += 1
        if milestone["weap"] and len(tanks) >= 2:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    # Drive east toward inland enemy.
                    cmds.append(Cmd.attack_move([uid], 95, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_navy_policy(level):
    """Build syrd + dd only; never weap/2tnk. Win clause P1
    (`has_building: weap`) never latches. Additionally the dd can
    only range coastal targets, so the kill clause cannot satisfy
    'inland' contribution. Deadline LOSS."""
    dd_target = {"easy": 2, "medium": 2, "hard": 3}[level]
    state = {"dd_orders": 0, "attack_issued": set()}
    milestone = {"syrd": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        dds = [u for u in units if u.get("type") == "dd"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "syrd" in own_b:
            milestone["syrd"] = True
        if not milestone["syrd"]:
            if "syrd" not in prod_items:
                cmds.append(Cmd.build("syrd"))
            # Place syrd at (20, by) — east edge x=22 touches water
            # rect starting at x=22. (Syrd footprint 3x3; placing at
            # (20, by) means east edge is x=22.)
            cmds.append(Cmd.place_building("syrd", 20, by))
        if milestone["syrd"] and len(dds) < dd_target:
            if "dd" not in prod_items and len(dds) >= state["dd_orders"]:
                cmds.append(Cmd.build("dd"))
                state["dd_orders"] += 1
        if milestone["syrd"] and len(dds) >= 1:
            for u in dds:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    # Drive dd north along water channel to coastal target.
                    cmds.append(Cmd.attack_move([uid], 24, 8))
                    state["attack_issued"].add(uid)
                    # NOTE: all-navy must lose. The dd can range the
                    # coastal Barracks but cannot reach the inland
                    # Construction Yard — the kill quota ≥N (with at
                    # least one inland required structurally by the
                    # then chain's P4) cannot be met by navy alone.
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Combined-arms build-then-strike:
      1. Build weap (Building queue).
      2. Build syrd (Building queue, serial after weap) — placed at
         (20, by) so east edge touches water at x=22.
      3. Train tanks (Vehicle queue, parallel) — N per tier.
      4. Train destroyers (Ship queue, parallel) — M per tier.
      5. When win-bar count met: attack-move tanks east toward inland
         enemy; attack-move dd north along water to coastal target.
    Wins every (tier, seed) on this pack.
    """
    # Order a buffer above the win bar so attrition (raid losses)
    # doesn't starve the quota.
    tank_target = {"easy": 3, "medium": 5, "hard": 8}[level]
    dd_target = {"easy": 1, "medium": 2, "hard": 3}[level]
    state = {"tank_orders": 0, "dd_orders": 0,
             "tank_attacks": set(), "dd_attacks": set()}
    milestone = {"weap": False, "syrd": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        dds = [u for u in units if u.get("type") == "dd"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "weap" in own_b:
            milestone["weap"] = True
        if "syrd" in own_b:
            milestone["syrd"] = True
        # Build weap first (it's the larger of the two; getting tanks
        # rolling sooner helps absorb the wave).
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        # Then syrd, placed adjacent to water (water starts at x=22;
        # 3x3 syrd at (20, by) has east edge x=22 touching water).
        if milestone["weap"] and not milestone["syrd"]:
            if "syrd" not in prod_items:
                cmds.append(Cmd.build("syrd"))
            cmds.append(Cmd.place_building("syrd", 20, by))
        # Launch when the WIN-BAR count is met (independent of build
        # buffer above target). Keep re-ordering if units die.
        win_tanks = {"easy": 2, "medium": 3, "hard": 3}[level]
        win_dds = {"easy": 1, "medium": 1, "hard": 2}[level]
        # Total tanks "alive or in queue" estimate — keep ordering
        # until that hits tank_target so attrition is recovered.
        tanks_in_play = len(tanks) + (1 if "2tnk" in prod_items else 0)
        dds_in_play = len(dds) + (1 if "dd" in prod_items else 0)
        if milestone["weap"] and tanks_in_play < tank_target:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
                state["tank_orders"] += 1
        if milestone["syrd"] and dds_in_play < dd_target:
            if "dd" not in prod_items:
                cmds.append(Cmd.build("dd"))
                state["dd_orders"] += 1
        # Dispatch destroyers individually upon spawn — head north
        # to the coastal target's water cell (24,8). dds don't suffer
        # ground-wave attrition so eager dispatch is safe.
        for u in dds:
            uid = u["id"]
            if uid not in state["dd_attacks"]:
                cmds.append(Cmd.attack_move([uid], 24, 8))
                state["dd_attacks"].add(uid)
        # Dispatch tanks individually on spawn — over-build covers
        # attrition so a 3-tank simultaneous cohort exists at SOME
        # tick (the then-chain P3 clause latches on that snapshot,
        # then advances to P4 regardless of subsequent attrition).
        # The kill phase accumulates from streaming arrivals.
        for u in tanks:
            uid = u["id"]
            if uid not in state["tank_attacks"]:
                cmds.append(Cmd.attack_move([uid], 95, 20))
                state["tank_attacks"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (no engine) ────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "f11-vertical-strike-naval"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Family-11 anchor metadata required for taxonomy slotting."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_then_chain_wired_into_win_condition():
    """The full-game `then:` chain (per F11 §68) must appear in every
    level's win predicate — load-bearing teeth that forbids 'raze
    first, then claim production' inversions."""
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
                # P1 weap, P2 syrd, P3 force composition, P4 kills.
                assert len(clauses) == 4, (
                    f"{lvl} then-chain must be 4 phases (weap → syrd → "
                    f"force → kills); got {clauses}"
                )
                assert clauses[0].get("has_building") == "weap"
                assert clauses[1].get("has_building") == "syrd"
                assert "all_of" in clauses[2]
                assert "enemy_buildings_destroyed_gte" in clauses[3]


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND ≤ 9999
    (engine cap). Without this the deadline never bites → DRAW."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        max_turns = L.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        ENGINE_CAP = 9999

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        win = L.win_condition.model_dump(exclude_none=True)
        fail = L.fail_condition.model_dump(exclude_none=True)
        wts, fts = [], []
        _collect(win, "within_ticks", wts)
        _collect(fail, "after_ticks", fts)
        assert wts, f"{lvl} has no within_ticks (no clock teeth)"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks {wt} > reachable {reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
            assert wt <= ENGINE_CAP, (
                f"{lvl} within_ticks {wt} > engine cap {ENGINE_CAP}"
            )
        for ft in fts:
            assert ft <= reachable + 1, (
                f"{lvl} after_ticks {ft} > reachable {reachable}+1"
            )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must declare ≥2 agent spawn_point groups so seed
    round-robins the start base (NORTH y=12 / SOUTH y=28)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 agent spawn groups, got {sp}"


def test_hard_spawns_round_robin_across_seeds():
    """Seed 1 and seed 2 must place the agent fact at different cells
    — proves the spawn_point round-robin is active."""
    c = compile_level(load_pack(PACK), "hard")

    def probe():
        captured = {}

        def pol(obs, Cmd):
            if "fact_pos" not in captured:
                bs = obs.get("own_buildings", []) or []
                facts = [(b["cell_x"], b["cell_y"])
                         for b in bs if b["type"] == "fact"
                         and b["cell_x"] < 50]
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
        f"hard spawn round-robin degenerate: seeds 1 and 2 both at {pos1}"
    )


def test_termination_flags_set_for_no_draw():
    """F11 §79: `enemy_units_killed: false` is the canonical fix for
    the 'all enemy combat units killed but buildings still standing'
    DRAW race."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("enemy_units_killed") is False, (
        f"pack needs enemy_units_killed: false; got {term}"
    )


def test_water_rect_declared():
    """Naval packs must declare water_rect so a syrd can be built
    adjacent to water."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        assert c.water_rect, f"{lvl} missing water_rect"
        assert len(c.water_rect) == 4, c.water_rect


# ── Engine-bound tests (parameterised over seeds 1..4) ──────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Observe-only must LOSE on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_brute_rush_loses(level, seed):
    """Driving the starter harv east — no production, fails P1/P2."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_rush_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute-rush must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_ground_loses(level, seed):
    """All-ground (weap + tanks only) — clause P2 (`has_building:
    syrd`) never latches → `then:` chain stalls → deadline LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_ground_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-ground must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_navy_loses(level, seed):
    """All-navy (syrd + dds only) — clause P1 (`has_building: weap`)
    never latches; dd cannot range the deep-inland enemy. LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_navy_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-navy must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_combined_arms_wins(level, seed):
    """Intended combined-arms play (weap + syrd + tanks + dd + strike)
    must WIN on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended combined-arms must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"own_b={res.signals.own_building_types}, "
        f"kills={res.signals.units_killed})"
    )
