"""F11 reactive-idiom pack — `f11-rebuild-after-attrition`.

Scripted attrition destroys part of the agent's pre-placed base
mid-episode. The agent must REBUILD the destroyed production
building AND continue producing AND finish the offensive under the
ORIGINAL deadline.

Bar (CLAUDE.md "no defect, no cheat"):
- stall (only observe)         LOSES every (tier, seed).
- brute (rush starter units    LOSES every (tier, seed).
   east)
- pre-attrition-only (build    LOSES every (tier, seed) — the
   to N tanks pre-tick-800,    `building_count_gte:weap:1` clause
   then push, no rebuild)      AFTER attrition is not satisfied;
                               LIVE-count of weap is 0 after the
                               destroy_actors event.
- intended (rebuild + produce  WINS every (tier, seed).
   + strike)

Plus structural pins:
- `then:` chain wired through into the compiled win condition.
- `within_ticks` ≤ reachable max (`93 + 90·(max_turns − 1)`).
- Hard tier has ≥2 agent spawn_point groups (NORTH/SOUTH).
- BOTH `agent_units_killed: false` AND `enemy_units_killed: false`
  set per §79 + F11 §74.
- Scheduled `destroy_actors` event fires at tick 800 (the
  load-bearing attrition trigger).
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
    production, no rebuild — fails both the rebuild and the kill
    clauses."""
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


def _pre_attrition_only_policy():
    """Build to N tanks BEFORE the attrition fires at tick 800, then
    push. After the attrition wipes the `weap` (and on hard the
    `proc`), the model never rebuilds. The post-attrition
    `building_count_gte:weap:1` clause is never satisfied →
    `then:` chain stalls → deadline LOSS."""
    state = {"attacked": set()}

    def pol(obs, Cmd):
        tick = obs.get("game_tick", 0)
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        cmds = []
        # Only build BEFORE the attrition.
        if tick < 800 and len(tanks) < 5:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
        # Push east always.
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
    """Detect the attrition (weap_count drops to 0 after tick 800;
    on hard also proc_count drops to 0 after tick 2000), rebuild
    the destroyed building(s) via `build` + `place_building`, then
    continuously train tanks and attack-move east. Two attack
    phases (initial sweep + re-engage) to ensure helis/tanks hit
    enemy buildings rather than stalling at the picket cells."""
    enemy_x = 80 if level == "easy" else 82
    state = {
        "weap_destroyed": False,
        "proc_destroyed": False,
        "attacked": set(),
        "phase2": set(),
    }

    def pol(obs, Cmd):
        tick = obs.get("game_tick", 0)
        ob = obs.get("own_buildings", []) or []
        own_b_types = [b["type"] for b in ob]
        weap_count = own_b_types.count("weap")
        proc_count = own_b_types.count("proc")
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (12, 20)
        cmds = []
        if tick > 850 and weap_count == 0:
            state["weap_destroyed"] = True
        if tick > 2050 and proc_count == 0:
            state["proc_destroyed"] = True
        if state["weap_destroyed"] and weap_count == 0:
            if "weap" not in prod_items:
                cmds.append(Cmd.build("weap"))
            cmds.append(Cmd.place_building("weap", bx + 8, by + 2))
        if state["proc_destroyed"] and proc_count == 0:
            if "proc" not in prod_items:
                cmds.append(Cmd.build("proc"))
            cmds.append(Cmd.place_building("proc", bx + 4, by - 4))
        if weap_count >= 1 and len(tanks) < 9:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
        n_min_attack = {"easy": 3, "medium": 4, "hard": 5}[level]
        if len(tanks) >= n_min_attack:
            for u in tanks:
                uid = u["id"]
                if uid not in state["attacked"]:
                    cmds.append(Cmd.attack_move([uid], enemy_x, 20))
                    state["attacked"].add(uid)
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


def test_then_chain_wired_into_win_condition():
    """The full-chain `then:` predicate per F11 §68 + §74 — must
    appear in every level's win condition, and clauses must include
    a post-attrition gate (`after_ticks`) plus a live-count rebuild
    check (`building_count_gte:weap:1`)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        inner = win.get("all_of") or []
        assert any("then" in cl for cl in inner), (
            f"{lvl} win missing then-chain: {win}"
        )
        # Find the then-chain clauses and verify the rebuild gate.
        for cl in inner:
            if "then" in cl:
                clauses = (cl["then"] or {}).get("clauses") or []
                assert any(
                    "building_count_gte" in c
                    and (c["building_count_gte"] or {}).get("type") == "weap"
                    for c in clauses
                ), f"{lvl} then-chain missing weap rebuild clause"
                assert any(
                    "after_ticks" in c for c in clauses
                ), f"{lvl} then-chain missing post-attrition gate"


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
    """`within_ticks` ≤ reachable max tick `93 + 90·(N-1)` AND
    ≤ 9999 (engine `max_ticks` cap)."""
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
        assert wts, f"{lvl} has no within_ticks"
        for wt in wts:
            assert wt <= min(reachable, ENGINE_CAP), (
                f"{lvl} within_ticks {wt} > min(reachable {reachable}, "
                f"engine cap {ENGINE_CAP})"
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
    """Seeds 1 and 2 must place the agent fact at different cells."""
    c = compile_level(load_pack(PACK), "hard")

    def probe():
        captured = {}

        def pol(obs, Cmd):
            if "fact_pos" not in captured:
                bs = obs.get("own_buildings", []) or []
                facts = [
                    (b["cell_x"], b["cell_y"])
                    for b in bs
                    if b["type"] == "fact" and b["cell_x"] < 50
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


def test_both_termination_flags_set_for_attrition_pack():
    """Per F11 §74 + CLAUDE.md: rebuild-after-attrition MUST set
    BOTH `agent_units_killed: false` AND `enemy_units_killed: false`.
    The former keeps the run alive past mid-episode unit loss; the
    latter prevents DRAW when the strike kills enemy combat units
    before razing buildings."""
    pack = load_pack(PACK)
    term = pack.base.get("termination") or {}
    assert term.get("agent_units_killed") is False, (
        f"pack needs agent_units_killed: false; got {term}"
    )
    assert term.get("enemy_units_killed") is False, (
        f"pack needs enemy_units_killed: false; got {term}"
    )


def test_attrition_event_scheduled_at_tick_800():
    """Per F11 §74: the canonical attrition event fires at tick
    ~800. Every level must schedule a `destroy_actors` event at
    that tick."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        L = pack.levels[lvl]
        events = (L.overrides or {}).get("scheduled_events") or []
        destroy_ticks = [
            e.get("tick")
            for e in events
            if isinstance(e, dict) and e.get("type") == "destroy_actors"
        ]
        assert 800 in destroy_ticks, (
            f"{lvl} must schedule destroy_actors at tick 800; "
            f"got destroy_ticks={destroy_ticks}"
        )


def test_hard_has_second_attrition_event_at_tick_2000():
    """Hard tier additionally schedules a second `destroy_actors`
    at tick 2000 (per F11 §74 hard-tier: TWO destroy events at
    different ticks — weap @ 800, proc @ 2000)."""
    pack = load_pack(PACK)
    L = pack.levels["hard"]
    events = (L.overrides or {}).get("scheduled_events") or []
    destroy_ticks = sorted(
        {
            e.get("tick")
            for e in events
            if isinstance(e, dict) and e.get("type") == "destroy_actors"
        }
    )
    assert destroy_ticks == [800, 2000], (
        f"hard needs destroy_actors at ticks [800, 2000]; got "
        f"{destroy_ticks}"
    )


def test_attrition_event_actually_destroys_weap_on_engine():
    """End-to-end pin: with a stall policy the pre-placed weap is
    LIVE at turn 1 (well before tick 800) and is GONE by turn ≥10
    (well after tick 800). This proves the scheduled
    `destroy_actors` filter targets the right region and the
    engine's destroy_actors event handler is wired through."""
    c = compile_level(load_pack(PACK), "easy")

    def probe():
        snap = {"early_weap": None, "post_weap": None}

        def pol(obs, Cmd):
            tick = obs.get("game_tick", 0)
            ob = obs.get("own_buildings", []) or []
            weap_count = sum(1 for b in ob if b["type"] == "weap")
            if tick <= 100 and snap["early_weap"] is None:
                snap["early_weap"] = weap_count
            if tick >= 900 and snap["post_weap"] is None:
                snap["post_weap"] = weap_count
            return [Cmd.observe()]
        pol.snap = snap
        return pol

    p = probe()
    run_level(c, p, seed=1)
    early = p.snap["early_weap"]
    post = p.snap["post_weap"]
    assert early == 1, (
        f"early-tick weap count must be 1 (pre-placed); got {early}"
    )
    assert post == 0, (
        f"post-tick-800 weap count must be 0 (destroyed by attrition "
        f"event); got {post}"
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
def test_pre_attrition_only_loses(level, seed):
    """Per F11 §74: a model that builds to N tanks pre-attrition
    and never rebuilds the destroyed weap LOSES — the
    `building_count_gte:weap:1` post-attrition clause is not
    satisfied so the `then:` chain never advances."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _pre_attrition_only_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"pre-attrition-only must LOSE on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_rebuild_wins(level, seed):
    """Detect the attrition + rebuild + produce + strike — must
    WIN on every (level, seed). This is the load-bearing test
    that the pack is solvable inside the budget."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended rebuild must WIN on {level} s={seed}; got "
        f"{res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"kills={res.signals.units_killed})"
    )
