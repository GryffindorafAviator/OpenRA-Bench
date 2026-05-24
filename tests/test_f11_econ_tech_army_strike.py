"""F11 `f11-econ-tech-army-strike` — full econ→tech→army→strike arc.

UNLIKE the POC (vertical-strike-ground-air) the tech-chain itself is
the test: ONLY `fact + powr + proc + harv` are pre-placed (the econ
seed). The agent must BUILD `weap` AND `fix` to satisfy P1/P2; the
predicate's `then:` chain forbids "raze first, then claim production".

Bar (CLAUDE.md "no defect, no cheat"):
- stall (observe-only)          LOSES every (tier, seed).
- brute  (rush starter harv east) LOSES every (tier, seed).
- over-econ (build extra proc, no army) LOSES every (tier, seed) —
                                clause P3 (≥N 2tnk) never latches.
- under-tech (build weap + tanks, never fix) LOSES every (tier, seed)
                                — clause P2 (`has_building: fix`)
                                never latches → `then:` stalls.
- intended (full econ→tech→army→strike) WINS every (tier, seed).

Plus structural pins:
- `then:` chain wired through into the compiled win condition.
- `within_ticks` ≤ reachable max (`93 + 90·(max_turns − 1)`) AND
  ≤ 9999 (engine max_ticks cap); deadline bites for a real LOSS.
- Hard tier has ≥2 agent spawn_point groups (NORTH y=12 / SOUTH
  y=28); seed 1 vs seed 2 round-robin distinct.
- `termination.enemy_units_killed: false` — §79 anti-DRAW.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "f11-econ-tech-army-strike.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — must LOSE on the clock on every (tier, seed)."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_rush_policy():
    """Drive starter harv(s) east toward enemy base. No production —
    fails P1/P2/P3 and the deadline."""
    state = {"issued": set()}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        cmds = []
        for u in units:
            uid = u["id"]
            if uid in state["issued"]:
                continue
            cmds.append(Cmd.attack_move([uid], 100, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _over_econ_policy(level):
    """Build 2 extra procs, never weap. Clause P1 (`has_building:
    weap`) never latches → `then:` stalls forever → deadline LOSS."""
    state = {"procs_placed": 0}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        own_proc = [b for b in ob if b["type"] == "proc"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        # Build up to 3 procs total (the existing one + 2 more).
        target = 3
        if len(own_proc) < target:
            if "proc" not in prod_items and len(own_proc) >= state["procs_placed"] + 1:
                # already-placed catch-up; rate-limit via observe
                pass
            if "proc" not in prod_items:
                cmds.append(Cmd.build("proc"))
            offsets = [(8, -4), (8, 4)]
            i = max(0, len(own_proc) - 1)  # 0->first new, 1->second new
            if i < len(offsets):
                dx, dy = offsets[i]
                cmds.append(Cmd.place_building("proc", bx + dx, by + dy))
                state["procs_placed"] = i + 1
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _under_tech_policy(level):
    """Build weap + tanks only; never fix. Clause P2 (`has_building:
    fix`) never latches → `then:` chain stalls at P2 → deadline LOSS."""
    tank_target = {"easy": 6, "medium": 7, "hard": 8}[level]
    enemy_x = {"easy": 82, "medium": 98, "hard": 98}[level]
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
        # Note: 2tnk's actual prereqs are `fix + vehicles.allies +
        # techlevel.medium`; without `fix` the 2tnk order is
        # silently dropped by the engine. So this under-tech policy
        # also fails to produce tanks at all — but that's still a
        # LOSS regardless (P2 never latches; P3 also never latches).
        # Keeping the policy intentionally naive — it represents
        # "model thinks weap alone unlocks tanks".
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        elif len(tanks) < tank_target:
            if "2tnk" not in prod_items and len(tanks) >= state["tank_orders"]:
                cmds.append(Cmd.build("2tnk"))
                state["tank_orders"] += 1
        if milestone["weap"] and len(tanks) >= 4:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Full econ→tech→army→strike:
      1. Build weap (Building queue).
      2. Build fix (Building queue, serial after weap).
      3. Train tanks (Vehicle queue, parallel) — N per tier.
      4. Once N tanks ready, attack-move them east.
    Wins every (tier, seed)."""
    # Over-produce to absorb attrition: build until we exceed the
    # bar by 1-2 so a single wave-loss doesn't trip the "alive at
    # terminal frame" count. (The bar is `unit_type_count_gte`,
    # which counts ALIVE units.)
    tank_target = {"easy": 4, "medium": 6, "hard": 7}[level]
    strike_threshold = {"easy": 4, "medium": 5, "hard": 6}[level]
    enemy_x = {"easy": 82, "medium": 98, "hard": 98}[level]
    state = {"attack_issued": set(), "rallied": set()}
    milestone = {"weap": False, "fix": False}

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
        if "fix" in own_b:
            milestone["fix"] = True
        # Serial Building-queue: weap → fix. Vehicle-queue 2tnk
        # is GATED on `fix` COMPLETE (2tnk prereq is `fix +
        # vehicles.allies + techlevel.medium` per ra/rules/vehicles.yaml).
        # We sequence weap → fix → tanks via the elif chain. The
        # Vehicle queue is single-stream off the single weap, so
        # one `build('2tnk')` per turn while the queue is empty
        # keeps it fed; subsequent orders are queued.
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        elif not milestone["fix"]:
            if "fix" not in prod_items:
                cmds.append(Cmd.build("fix"))
            cmds.append(Cmd.place_building("fix", bx + 6, by + 6))
        elif len(tanks) < tank_target:
            # Issue a build order whenever Vehicle queue idle.
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
            # Clear spawn cells WITHOUT pushing tanks east into the
            # hunt-wave: rally NEAR the base (bx+12, by) so the
            # weap's exit cells stay clear but the formation isn't
            # split / picked off in transit.
            rally_x = bx + 12
            for u in tanks:
                uid = u["id"]
                if uid not in state["rallied"]:
                    cmds.append(Cmd.attack_move([uid], rally_x, by))
                    state["rallied"].add(uid)
        if len(tanks) >= strike_threshold:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (no engine) ────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "f11-econ-tech-army-strike"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_then_chain_wired_into_win_condition():
    """`then:` chain P1=weap, P2=fix, P3=≥N 2tnk, P4=≥K enemy razed."""
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
                assert len(clauses) == 4, (
                    f"{lvl} then-chain must be 4 phases "
                    f"(weap → fix → tanks → kills); got {clauses}"
                )
                assert clauses[0].get("has_building") == "weap"
                assert clauses[1].get("has_building") == "fix"
                # unit_type_count_gte may be a dict with type/n.
                assert "unit_type_count_gte" in clauses[2]
                assert "enemy_buildings_destroyed_gte" in clauses[3]


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND <
    `base.termination.max_ticks` (engine commit 493898e removed the
    historical 10000-tick hard cap; the scenario-declared
    `termination.max_ticks` is now the authoritative ceiling).
    Without this the deadline never bites → DRAW."""
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
                f"{lvl} within_ticks {wt} > reachable {reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
            assert wt < engine_cap, (
                f"{lvl} within_ticks {wt} >= engine cap {engine_cap} "
                f"(base.termination.max_ticks)"
            )
        for ft in fts:
            assert ft <= reachable + 1, (
                f"{lvl} after_ticks {ft} > reachable {reachable}+1"
            )


def test_termination_flag_set_for_no_draw():
    """§79: win clause needs `enemy_buildings_destroyed_gte`, so engine
    must NOT auto-`done` on enemy-unit-wipe (DRAW race)."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("enemy_units_killed") is False, (
        f"pack needs enemy_units_killed: false; got {term}"
    )


def test_hard_tier_has_seed_driven_spawn_groups():
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 agent spawn groups, got {sp}"


def test_hard_spawns_round_robin_across_seeds():
    """Seed 1 and seed 2 must place the agent fact at distinct cells."""
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
        f"hard spawn round-robin degenerate: s1={pos1} s2={pos2}"
    )


# ── Engine-bound tests (parameterised over seeds 1..4) ──────────────


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
        f"brute must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_over_econ_loses(level, seed):
    """Build extra procs, never weap. P1 never latches → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _over_econ_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"over-econ must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_under_tech_loses(level, seed):
    """Build weap + tanks but NEVER fix. P2 never latches → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _under_tech_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"under-tech must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_full_arc_wins(level, seed):
    """Full econ→tech→army→strike must WIN on every (tier, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended full-arc must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"own_b={res.signals.own_building_types}, "
        f"kills={res.signals.units_killed})"
    )
