"""F11 — `f11-full-combined-arms` (ground + air + naval, all three).

The most ambitious F11 pack: the model must build THREE production
buildings (weap + hpad + syrd), field at least one (or N, on hard)
unit of EACH arm (medium tank + helicopter + destroyer), then raze
enemy buildings split across three reachability domains: coastal
(dd shore-strike), midfield (heli's domain — terrain-ignoring), and
deep inland (ground or air).

Bar (CLAUDE.md "no defect, no cheat"):
- stall (observe-only)         LOSES every (tier, seed).
- brute  (rush harv east)      LOSES every (tier, seed).
- all-ground (weap + tanks)    LOSES — clauses P2 (hpad) AND P3
                                (syrd) never latch.
- all-air   (hpad + helis)     LOSES — clauses P1 (weap) AND P3
                                (syrd) never latch; on med/hard,
                                AA also shreds heli-only force.
- all-navy  (syrd + dds)       LOSES — clauses P1 (weap) AND P2
                                (hpad) never latch; navy can't
                                reach midfield/inland.
- intended  (tri-arm)          WINS every (tier, seed).

Plus structural pins:
- `then:` chain weap → hpad → syrd → force → kills.
- `within_ticks` ≤ reachable max; deadline bites for real LOSS.
- `termination.enemy_units_killed: false` (F11 §79 mandatory).
- Hard tier has ≥2 agent spawn_point groups.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "f11-full-combined-arms.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_rush_policy():
    state = {"issued": set()}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        cmds = []
        for u in units:
            uid = u["id"]
            if uid in state["issued"]:
                continue
            cmds.append(Cmd.attack_move([uid], 115, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_ground_policy(level):
    """Build weap + tanks only. Clauses P2 (hpad) AND P3 (syrd) never
    latch → `then:` stalls → deadline LOSS."""
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
                    cmds.append(Cmd.attack_move([uid], 115, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_air_policy(level):
    """Build hpad + helis only. P1 (weap) AND P3 (syrd) never latch."""
    heli_target = {"easy": 3, "medium": 4, "hard": 4}[level]
    state = {"heli_orders": 0, "attack_issued": set()}
    milestone = {"hpad": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        helis = [u for u in units if u.get("type") == "heli"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "hpad" in own_b:
            milestone["hpad"] = True
        if not milestone["hpad"]:
            if "hpad" not in prod_items:
                cmds.append(Cmd.build("hpad"))
            cmds.append(Cmd.place_building("hpad", bx + 4, by + 6))
        if milestone["hpad"] and len(helis) < heli_target:
            if "heli" not in prod_items and len(helis) >= state["heli_orders"]:
                cmds.append(Cmd.build("heli"))
                state["heli_orders"] += 1
        if milestone["hpad"] and len(helis) >= 1:
            for u in helis:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    cmds.append(Cmd.attack_move([uid], 70, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_navy_policy(level):
    """Build syrd + dds only. P1 (weap) AND P2 (hpad) never latch."""
    dd_target = {"easy": 2, "medium": 3, "hard": 3}[level]
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
            cmds.append(Cmd.place_building("syrd", 20, by))
        if milestone["syrd"] and len(dds) < dd_target:
            if "dd" not in prod_items and len(dds) >= state["dd_orders"]:
                cmds.append(Cmd.build("dd"))
                state["dd_orders"] += 1
        if milestone["syrd"] and len(dds) >= 1:
            for u in dds:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    cmds.append(Cmd.attack_move([uid], 24, 8))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Tri-arm build-then-strike:
      1. Build weap (Building queue).
      2. Build hpad (Building queue, serial after weap).
      3. Build syrd (Building queue, serial after hpad) — placed at
         (20, by) so east edge touches water at x=22.
      4. Train tanks (Vehicle), helis (Aircraft), dds (Ship) in parallel.
      5. Dispatch each unit eagerly on spawn:
         - tanks → east toward inland (115, 20)
         - helis → midfield (70, 20)
         - dds   → coastal (24, 8) — northernmost water cell
    Wins every (tier, seed) on this pack.
    """
    # Over-build above the win bar to absorb attrition.
    tank_target = {"easy": 3, "medium": 4, "hard": 5}[level]
    heli_target = {"easy": 2, "medium": 3, "hard": 4}[level]
    dd_target = {"easy": 1, "medium": 2, "hard": 3}[level]
    state = {
        "tank_orders": 0, "heli_orders": 0, "dd_orders": 0,
        "tank_attacks": set(), "heli_attacks": set(), "dd_attacks": set(),
    }
    milestone = {"weap": False, "hpad": False, "syrd": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        helis = [u for u in units if u.get("type") == "heli"]
        dds = [u for u in units if u.get("type") == "dd"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "weap" in own_b: milestone["weap"] = True
        if "hpad" in own_b: milestone["hpad"] = True
        if "syrd" in own_b: milestone["syrd"] = True
        # Build order: weap → hpad → syrd (serial on Building queue).
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        elif not milestone["hpad"]:
            if "hpad" not in prod_items:
                cmds.append(Cmd.build("hpad"))
            cmds.append(Cmd.place_building("hpad", bx + 4, by + 6))
        elif not milestone["syrd"]:
            if "syrd" not in prod_items:
                cmds.append(Cmd.build("syrd"))
            # syrd footprint 3x3 placed at (20, by) — east edge x=22
            # touches water rect (which starts at x=22).
            cmds.append(Cmd.place_building("syrd", 20, by))
        # Parallel unit production queues: Vehicle / Aircraft / Ship.
        tanks_in_play = len(tanks) + (1 if "2tnk" in prod_items else 0)
        helis_in_play = len(helis) + (1 if "heli" in prod_items else 0)
        dds_in_play = len(dds) + (1 if "dd" in prod_items else 0)
        if milestone["weap"] and tanks_in_play < tank_target:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
                state["tank_orders"] += 1
        if milestone["hpad"] and helis_in_play < heli_target:
            if "heli" not in prod_items:
                cmds.append(Cmd.build("heli"))
                state["heli_orders"] += 1
        if milestone["syrd"] and dds_in_play < dd_target:
            if "dd" not in prod_items:
                cmds.append(Cmd.build("dd"))
                state["dd_orders"] += 1
        # Dispatch each unit eagerly on spawn. The combined-arms
        # plan (each arm to a different domain):
        # - tanks: inland (115,20). Pass through midfield en route;
        #   attack_move autofires on hostiles encountered.
        # - helis: midfield (70,20) on easy (no AA), inland (115,20)
        #   on med/hard (midfield AA shreds heli approach).
        # - dds: bidirectional coastal split — (24,8) and (24,33).
        heli_target_x, heli_target_y = (70, 20) if level == "easy" else (115, 20)
        for u in tanks:
            uid = u["id"]
            if uid not in state["tank_attacks"]:
                cmds.append(Cmd.attack_move([uid], 115, 20))
                state["tank_attacks"].add(uid)
        for u in helis:
            uid = u["id"]
            if uid not in state["heli_attacks"]:
                cmds.append(Cmd.attack_move(
                    [uid], heli_target_x, heli_target_y,
                ))
                state["heli_attacks"].add(uid)
        # Dispatch dds in ALTERNATING directions (north / south
        # coastal targets) so both coastal buildings get razed.
        for i, u in enumerate(dds):
            uid = u["id"]
            if uid not in state["dd_attacks"]:
                # Alternate by spawn-order index.
                target = (24, 8) if (len(state["dd_attacks"]) % 2 == 0) else (24, 33)
                cmds.append(Cmd.attack_move([uid], *target))
                state["dd_attacks"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (no engine) ────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "f11-full-combined-arms"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_then_chain_wired_into_win_condition():
    """5-clause then-chain: weap → hpad → syrd → force → kills."""
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
                assert len(clauses) == 5, (
                    f"{lvl} then-chain must be 5 phases "
                    f"(weap → hpad → syrd → force → kills); got {clauses}"
                )
                assert clauses[0].get("has_building") == "weap"
                assert clauses[1].get("has_building") == "hpad"
                assert clauses[2].get("has_building") == "syrd"
                assert "all_of" in clauses[3]
                assert "enemy_buildings_destroyed_gte" in clauses[4]


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND <
    `base.termination.max_ticks` (engine commit 493898e removed the
    historical 10000-tick hard cap; the scenario-declared
    `termination.max_ticks` is now the authoritative ceiling)."""
    pack = load_pack(PACK)
    engine_cap = (pack.base.get("termination") or {}).get("max_ticks")
    assert engine_cap is not None, "base.termination.max_ticks missing"
    for lvl in LEVELS:
        L = pack.levels[lvl]
        max_turns = L.max_turns
        reachable = 93 + 90 * (max_turns - 1)

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
        assert wts, f"{lvl} has no within_ticks"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks {wt} > reachable {reachable}"
            )
            assert wt < engine_cap, (
                f"{lvl} within_ticks {wt} >= engine cap {engine_cap} "
                f"(base.termination.max_ticks)"
            )
        for ft in fts:
            assert ft <= reachable + 1


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 agent spawn groups, got {sp}"


def test_hard_spawns_round_robin_across_seeds():
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
        f"hard spawn round-robin degenerate: seeds 1 and 2 at {pos1}"
    )


def test_termination_flags_set_for_no_draw():
    """F11 §79."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("enemy_units_killed") is False, (
        f"pack needs enemy_units_killed: false; got {term}"
    )


def test_water_rect_declared():
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        assert c.water_rect, f"{lvl} missing water_rect"
        assert len(c.water_rect) == 4


# ── Engine-bound tests ──────────────────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_brute_rush_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_rush_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute-rush must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_ground_loses(level, seed):
    """All-ground — clauses P2 (hpad) AND P3 (syrd) never latch."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_ground_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-ground must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_air_loses(level, seed):
    """All-air — clauses P1 (weap) AND P3 (syrd) never latch."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_air_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-air must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_navy_loses(level, seed):
    """All-navy — clauses P1 (weap) AND P2 (hpad) never latch."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_navy_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-navy must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_tri_arm_wins(level, seed):
    """The intended tri-arm play must WIN every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended tri-arm must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"own_b={res.signals.own_building_types}, "
        f"kills={res.signals.units_killed})"
    )
