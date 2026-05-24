"""F11 proof-of-concept pack — `f11-vertical-strike-ground-air`.

The first Family-11 (full-game) pack: the combined-arms test where a
build-then-strike chain must field BOTH ground AND air arms. The
`then:` chain forbids "raze first, then claim production" — the
predicate gates on building-build → unit-build → enemy-raze in that
order.

Bar (CLAUDE.md "no defect, no cheat"):
- stall (observe-only)        LOSES every (tier, seed).
- brute  (rush harv east)     LOSES every (tier, seed).
- all-ground (weap + tanks)   LOSES every (tier, seed) — clause
                              P2 (`has_building: hpad`) never
                              latches; `then:` chain stalls.
- all-air   (hpad + helis)    LOSES every (tier, seed) — clause
                              P1 (`has_building: weap`) never
                              latches; on hard the AA also shreds
                              the heli-only strike.
- intended  (combined arms)   WINS every (tier, seed).

Plus structural pins:
- `then:` chain is wired through into the compiled win condition.
- `within_ticks` ≤ reachable max (`93 + 90·(max_turns − 1)`); the
  deadline bites for a real LOSS, not DRAW.
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

PACK = PACKS_DIR / "f11-vertical-strike-ground-air.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — must LOSE on the clock (or units_killed) on
    every (tier, seed)."""
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
            cmds.append(Cmd.attack_move([uid], 100, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_ground_policy(level):
    """Build weap + 2tnk only; never hpad/heli. Win clause P2
    (`has_building: hpad`) never latches → `then:` stalls →
    deadline LOSS."""
    tank_target = {"easy": 4, "medium": 5, "hard": 6}[level]
    enemy_x = {"easy": 85, "medium": 100, "hard": 100}[level]
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
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _all_air_policy(level):
    """Build hpad + heli only; never weap/2tnk. Win clause P1
    (`has_building: weap`) never latches; on hard the AA also
    damages a heli-only force. Deadline LOSS."""
    heli_target = {"easy": 2, "medium": 2, "hard": 3}[level]
    enemy_x = {"easy": 85, "medium": 100, "hard": 100}[level]
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
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attack_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Combined-arms build-then-strike:
      1. Build weap (Building queue).
      2. Build hpad (Building queue, serial after weap).
      3. Train tanks (Vehicle queue, parallel) — N per tier.
      4. Train helis (Aircraft queue, parallel) — M per tier.
      5. When the win-bar count is met, attack-move both arms east.
    Wins every (tier, seed) on this pack.
    """
    tank_target = {"easy": 2, "medium": 3, "hard": 3}[level]
    heli_target = {"easy": 1, "medium": 1, "hard": 1}[level]
    enemy_x = {"easy": 85, "medium": 100, "hard": 100}[level]
    state = {"tank_orders": 0, "heli_orders": 0, "attack_issued": set()}
    milestone = {"weap": False, "hpad": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        helis = [u for u in units if u.get("type") == "heli"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (8, 20)
        cmds = []
        if "weap" in own_b:
            milestone["weap"] = True
        if "hpad" in own_b:
            milestone["hpad"] = True
        if not milestone["weap"]:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 10, by))
        if milestone["weap"] and not milestone["hpad"]:
            if "hpad" not in prod_items:
                cmds.append(Cmd.build("hpad"))
            # Place hpad south-east of base — the (bx+4, by+6) cell
            # leaves room around the hpad footprint so the spawned
            # heli has a free cell to materialize on. A tighter
            # placement (e.g. bx+14, by+3) collides with subsequent
            # heli spawns and the heli silently fails to materialise.
            cmds.append(Cmd.place_building("hpad", bx + 4, by + 6))
        if milestone["weap"] and len(tanks) < tank_target:
            if "2tnk" not in prod_items and len(tanks) >= state["tank_orders"]:
                cmds.append(Cmd.build("2tnk"))
                state["tank_orders"] += 1
        if milestone["hpad"] and len(helis) < heli_target:
            if "heli" not in prod_items and len(helis) >= state["heli_orders"]:
                cmds.append(Cmd.build("heli"))
                state["heli_orders"] += 1
        if len(tanks) >= tank_target and len(helis) >= heli_target:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attack_issued"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attack_issued"].add(uid)
            for u in helis:
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
    assert pack.meta.id == "f11-vertical-strike-ground-air"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    """Family-11 anchor metadata required for taxonomy slotting."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_then_chain_wired_into_win_condition():
    """The full-game `then:` chain (per F11 §68) must appear in every
    level's win predicate — it's the load-bearing teeth that forbids
    'raze first, then claim production' inversions."""
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
                # P1 weap, P2 hpad, P3 force composition, P4 kills.
                assert len(clauses) == 4, (
                    f"{lvl} then-chain must be 4 phases (weap → hpad → "
                    f"force → kills); got {clauses}"
                )
                assert clauses[0].get("has_building") == "weap"
                assert clauses[1].get("has_building") == "hpad"
                assert "all_of" in clauses[2]
                assert "enemy_buildings_destroyed_gte" in clauses[3]


def test_every_level_has_fail_condition():
    """No silent draws — every level must be able to emit a LOSS."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND ≤
    `base.termination.max_ticks` (engine commit 493898e removed the
    historical 10000-tick hard cap; the scenario-declared
    `termination.max_ticks` is now the authoritative ceiling).
    Without this the deadline can never bite → DRAW degeneracy
    (CLAUDE.md criterion #2)."""
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
        assert wts, f"{lvl} has no within_ticks (no clock teeth)"
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


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must declare ≥2 agent spawn_point groups so seed
    round-robins the start base (NORTH y=12 / SOUTH y=28)."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 agent spawn groups, got {sp}"


def test_hard_spawns_round_robin_across_seeds():
    """Seed 1 and seed 2 must place the agent fact at different
    cells — proves the spawn_point round-robin is active, not
    degenerate."""
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
        f"hard spawn round-robin is degenerate: seed 1 and 2 both "
        f"started at {pos1}"
    )


def test_termination_flags_set_for_no_draw():
    """`enemy_units_killed: false` is the canonical fix for the
    'all enemy combat units killed but buildings still standing'
    DRAW race; the win predicate requires building destruction,
    so the engine must stay alive past unit-wipe."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("enemy_units_killed") is False, (
        f"hard pack needs enemy_units_killed: false; got {term}"
    )


# ── Engine-bound tests (parameterised over seeds 1..4) ──────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Observe-only must LOSE on every (level, seed). Either the
    deadline bites or the enemy wave wipes the harvs (then `not
    own_units_gte: 1` fires)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _stall_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_brute_rush_loses(level, seed):
    """Driving the starter harv east toward the enemy base — no
    production buildings, fails P1 + P2. Must LOSE on every
    (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_rush_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute-rush must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_all_ground_loses(level, seed):
    """All-ground (only weap + tanks) — clause P2 (`has_building:
    hpad`) never latches → `then:` chain stalls forever → deadline
    LOSS on every (level, seed)."""
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
    """All-air (only hpad + helis) — clause P1 (`has_building:
    weap`) never latches → `then:` stalls. On hard, AA also
    damages the heli-only strike. LOSS on every (level, seed)."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _all_air_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"all-air must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_combined_arms_wins(level, seed):
    """The intended combined-arms play (weap + hpad + tanks +
    heli + strike) must WIN on every (level, seed). This is the
    load-bearing test that the pack is solvable inside the budget
    by the advertised capability."""
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
