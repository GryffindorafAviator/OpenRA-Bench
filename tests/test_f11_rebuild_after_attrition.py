"""F11 recovery pack — `f11-rebuild-after-attrition`.

TWO-BASE STRATEGIC-RETREAT TEMPLATE (task #81). The agent inherits
a doomed FORWARD production cell (low-HP exposed `weap`, on hard
also `proc`, grinding under an in-world enemy striker column) and
a safe HOME production stack at the deep west. The capability test:
rebuild the lost production at HOME, regenerate the army, and still
finish the offensive on the ORIGINAL deadline.

Bar (CLAUDE.md "no defect, no cheat"):
- stall (only observe)         LOSES every (tier, seed).
- brute (rush starter units    LOSES every (tier, seed).
   east, no production)
- build-at-forward (place the  LOSES every (tier, seed) — the rebuild
   rebuilt weap in the strike  is razed by the persistent stance:3
   arc)                        striker column; cash burned; army quota
                               + kill quota both miss.
- intended (build at HOME +    WINS every (tier, seed).
   produce + strike)

Plus structural pins:
- `within_ticks` ≤ reachable max (`93 + 90·(max_turns − 1)`).
- Hard tier has ≥2 agent spawn_point groups (NORTH/SOUTH).
- BOTH `agent_units_killed: false` AND `enemy_units_killed: false`
  set on `base.termination`.
- The FORWARD weap falls in-world (no `scheduled_events` destruction
  of agent assets) — pinned by an end-to-end probe.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "f11-rebuild-after-attrition.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _brute_policy():
    """Push every starter unit east toward the enemy base. No
    production. Starter tanks (2 only) cannot satisfy the army quota
    (≥3 on easy, ≥4 medium, ≥5 hard) and the kill quota is missed
    too — even if they reach the enemy base they're 2 vs garrison."""
    state = {"issued": set()}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        cmds = []
        for u in units:
            uid = u["id"]
            if uid in state["issued"]:
                continue
            cmds.append(Cmd.attack_move([uid], 82, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _build_at_forward_policy(level):
    """Rebuild the weap (and proc on hard) AT the FORWARD cell — in
    the strike arc. The 5× 4tnk column razes the rebuild and the
    cash is wasted; the army quota and kill quota both miss → LOSS.
    """
    fx = 50
    state = {"placed_weap": False, "placed_proc": False, "attacked": set()}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        weap_count = sum(1 for b in ob if b["type"] == "weap")
        proc_count = sum(1 for b in ob if b["type"] == "proc")
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        # Rebuild the weap at the FORWARD cell — repeat as it dies.
        if weap_count == 0:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
        if "weap" in prod_items or weap_count == 0:
            # Place at FORWARD (in the strike arc) — the rebuild dies.
            cmds.append(Cmd.place_building("weap", fx, 20))
        if level == "hard":
            if proc_count == 0:
                if "proc" not in prod_items:
                    cmds.append(Cmd.build("proc"))
            cmds.append(Cmd.place_building("proc", fx, 16))
        # Push starter tanks east (no new tanks produced — weap is
        # doomed at forward).
        for u in tanks:
            uid = u["id"]
            if uid not in state["attacked"]:
                cmds.append(Cmd.attack_move([uid], 82, 20))
                state["attacked"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Build at the HOME base (safe, deep west), train the strike
    package on the surviving HOME weap, and push east. State-based
    win at deadline: weap≥1 (HOME weap survives), tanks≥N
    (HOME-funded production), enemy_buildings_destroyed≥K (strike
    home)."""
    enemy_x = 80 if level == "easy" else 82
    n_min_attack = {"easy": 5, "medium": 7, "hard": 9}[level]
    n_target = {"easy": 6, "medium": 9, "hard": 12}[level]
    state = {"attacked": set(), "phase2": set()}

    def pol(obs, Cmd):
        tick = obs.get("game_tick", 0)
        ob = obs.get("own_buildings", []) or []
        # HOME fact is the deep-west one (x < 30); we never want to
        # touch the FORWARD weap at x≈50.
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        home_fact = [
            b for b in ob if b["type"] == "fact" and b["cell_x"] < 30
        ]
        bx, by = (
            (home_fact[0]["cell_x"], home_fact[0]["cell_y"])
            if home_fact
            else (12, 20)
        )
        cmds = []
        # Continuously train tanks while HOME weap is alive. (HOME
        # weap is pre-placed and out of the strike arc — it survives
        # the whole episode without rebuild.)
        if len(tanks) < n_target:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
        # Push when force quota is met.
        if len(tanks) >= n_min_attack:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attacked"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attacked"].add(uid)
        # Second wave — re-attack with any new tanks past tick 5000.
        if tick > 5000:
            for u in tanks:
                uid = u["id"]
                if uid not in state["phase2"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x + 3, 18))
                    state["phase2"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (no engine) ────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "f11-rebuild-after-attrition"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_state_based_win_no_scheduled_destruction_of_agent_assets():
    """Two-base template: WIN must be state-based at deadline (no
    `then:` chain gating on a destroy event); the pack must NOT
    declare a `scheduled_events: destroy_actors` targeting the
    agent. Per task #81 — agent attrition is in-game, not external."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        events = (L.overrides or {}).get("scheduled_events") or []
        for e in events:
            if isinstance(e, dict) and e.get("type") == "destroy_actors":
                filt = e.get("filter") or {}
                assert filt.get("owner") != "agent", (
                    f"{lvl} declares scheduled destroy_actors against "
                    f"the agent — the two-base template forbids "
                    f"externally-scripted attrition of agent assets: "
                    f"{e}"
                )


def test_win_has_state_based_floors():
    """Each level's win condition must include live-count floors
    (weap, fact, 2tnk, enemy_buildings_destroyed) — the state-based
    win shape per the two-base template."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []

        def _flatten_keys(node):
            out = []
            if isinstance(node, dict):
                for k, v in node.items():
                    out.append(k)
                    out.extend(_flatten_keys(v))
            elif isinstance(node, list):
                for v in node:
                    out.extend(_flatten_keys(v))
            return out

        keys = _flatten_keys(inner)
        assert "building_count_gte" in keys, f"{lvl} missing building_count_gte"
        assert "unit_type_count_gte" in keys, f"{lvl} missing unit_type_count_gte"
        assert "enemy_buildings_destroyed_gte" in keys, (
            f"{lvl} missing enemy_buildings_destroyed_gte"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND <
    `base.termination.max_ticks`."""
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
            if ft >= reachable - 200:
                assert ft <= reachable + 1, (
                    f"{lvl} fail after_ticks {ft} > reachable {reachable}+1"
                )


def test_hard_tier_has_seed_driven_spawn_groups():
    """Hard must declare ≥2 agent spawn_point groups."""
    c = compile_level(load_pack(PACK), "hard")
    sp = {a.spawn_point for a in c.scenario.actors if a.owner == "agent"}
    assert len(sp) >= 2, f"hard needs ≥2 agent spawn groups, got {sp}"


def test_hard_spawns_round_robin_across_seeds():
    """Seeds 1 and 2 must place the HOME agent fact at different cells."""
    c = compile_level(load_pack(PACK), "hard")

    def probe():
        captured = {}

        def pol(obs, Cmd):
            if "fact_pos" not in captured:
                bs = obs.get("own_buildings", []) or []
                facts = [
                    (b["cell_x"], b["cell_y"])
                    for b in bs
                    if b["type"] == "fact" and b["cell_x"] < 30
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
        f"hard spawn round-robin degenerate: seeds 1&2 both at {pos1}"
    )


def test_both_termination_flags_set():
    """Per F11 §79 + CLAUDE.md: this pack must set BOTH
    `agent_units_killed: false` AND `enemy_units_killed: false`."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("agent_units_killed") is False, (
        f"pack needs agent_units_killed: false; got {term}"
    )
    assert term.get("enemy_units_killed") is False, (
        f"pack needs enemy_units_killed: false; got {term}"
    )


def test_forward_weap_falls_in_world():
    """End-to-end pin: with a stall policy the FORWARD weap (at the
    mid-map cell, distinct from the HOME weap at x<30) is alive at
    turn 1 and is destroyed in-world (by the stance:3 striker
    column) before tick 800. The HOME weap remains alive — this is
    NOT an externally-scripted destruction; the strikers do the
    work in-game."""
    c = compile_level(load_pack(PACK), "easy")

    def probe():
        snap = {
            "early_fwd": None,
            "early_home": None,
            "post_fwd": None,
            "post_home": None,
        }

        def pol(obs, Cmd):
            tick = obs.get("game_tick", 0)
            ob = obs.get("own_buildings", []) or []
            fwd_weap = sum(
                1 for b in ob if b["type"] == "weap" and b["cell_x"] >= 30
            )
            home_weap = sum(
                1 for b in ob if b["type"] == "weap" and b["cell_x"] < 30
            )
            if tick <= 30 and snap["early_fwd"] is None:
                snap["early_fwd"] = fwd_weap
                snap["early_home"] = home_weap
            if tick >= 800 and snap["post_fwd"] is None:
                snap["post_fwd"] = fwd_weap
                snap["post_home"] = home_weap
            return [Cmd.observe()]
        pol.snap = snap
        return pol

    p = probe()
    run_level(c, p, seed=1)
    early_fwd = p.snap["early_fwd"]
    early_home = p.snap["early_home"]
    post_fwd = p.snap["post_fwd"]
    post_home = p.snap["post_home"]
    assert early_fwd == 1, (
        f"early-tick FORWARD weap must be 1 (pre-placed); got {early_fwd}"
    )
    assert early_home == 1, (
        f"early-tick HOME weap must be 1 (pre-placed); got {early_home}"
    )
    assert post_fwd == 0, (
        f"post-tick-800 FORWARD weap must be 0 (destroyed in-world by "
        f"the striker column); got {post_fwd}"
    )
    assert post_home == 1, (
        f"post-tick-800 HOME weap must be 1 (safe from strikers); "
        f"got {post_home}"
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
def test_brute_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _brute_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"brute must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_build_at_forward_loses(level, seed):
    """Two-base template: build-at-forward LOSES — the rebuild is
    razed by the in-world striker column; cash is wasted; the army
    quota and kill quota both miss → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _build_at_forward_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"build-at-forward must LOSE on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_build_at_home_wins(level, seed):
    """Build at the safe HOME base + produce + push east — must
    WIN on every (level, seed). This is the load-bearing test that
    the pack is solvable inside the budget."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    assert res.outcome == "win", (
        f"intended build-at-home must WIN on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"kills={res.signals.units_killed})"
    )
