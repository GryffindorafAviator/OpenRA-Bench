"""F11 `f11-defense-then-counter` — offensive-from-t=0 idiom.

UNLIKE the other F11 packs (static-then-scheduled enemy pressure),
the enemy is hunting from tick 0. The model must defend the opening
rush (3-5× 2tnk closing on the base) with its pre-placed starter
tanks while simultaneously building production capacity for the
counter-strike. Pre-placed: `fact + powr + proc + tent + weap +
2-3× starter 2tnk + harv`; the model BUILDS `fix` (the 2tnk prereq)
so the tank production line is live.

Bar (CLAUDE.md "no defect, no cheat"):
- stall (observe-only) LOSES every (tier, seed) — opening rush
  kills harv (own_units_gte fail) AND/OR fact (P1 fail).
- panic-rush (charge starter tanks east at t=0, no defense, no
  production) LOSES every (tier, seed) — opening rush kills the
  un-defended base while the starter tanks are out of position.
- over-defense (build-only, never strike east) LOSES every
  (tier, seed) — P4 enemy_buildings_destroyed_gte never latches
  → deadline LOSS.
- intended (defend-then-counter) WINS every (tier, seed).

Plus structural pins:
- `then:` chain wired through with 4 phases (survive → econ-intact
  → counter-force → strike).
- within_ticks ≤ reachable max (93+90·(N-1)) AND ≤ 9999.
- termination.enemy_units_killed: false (§79).
- hard tier has ≥2 spawn_point groups; seed 1 vs 2 round-robin
  distinct.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "f11-defense-then-counter.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Policies ─────────────────────────────────────────────────────────


def _stall_policy():
    """Observe-only — opening rush kills harv / fact → LOSS."""
    def pol(obs, Cmd):
        return [Cmd.observe()]
    return pol


def _panic_rush_policy():
    """Charge starter tanks east at t=0; no production, no defense.
    The opening enemy 2tnk wave reaches the base from the other
    direction (relative-pass) and kills harv/fact. LOSS."""
    state = {"issued": set()}

    def pol(obs, Cmd):
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        cmds = []
        for u in tanks:
            uid = u["id"]
            if uid in state["issued"]:
                continue
            cmds.append(Cmd.attack_move([uid], 90, 20))
            state["issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _over_defense_policy(level):
    """Build fix + train extra 2tnk but NEVER send east. P4
    (enemy_buildings_destroyed_gte) never latches → deadline LOSS."""
    tank_target = {"easy": 6, "medium": 7, "hard": 8}[level]
    state = {"defenders_engaged": set()}
    milestone = {"fix": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (12, 20)
        cmds = []
        if "fix" in own_b:
            milestone["fix"] = True
        if not milestone["fix"]:
            if "fix" not in prod_items:
                cmds.append(Cmd.build("fix"))
            cmds.append(Cmd.place_building("fix", bx + 8, by))
        elif len(tanks) < tank_target:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
        # Defend in-place: enemies in range engaged by attack_unit.
        enemies = obs.get("enemies_in_range_short", []) or []
        for u in tanks:
            uid = u["id"]
            if uid in state["defenders_engaged"]:
                continue
            if enemies:
                cmds.append(Cmd.attack_unit([uid], enemies[0].get("id")
                                            if isinstance(enemies[0], dict)
                                            else enemies[0]))
                state["defenders_engaged"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


def _intended_policy(level):
    """Defense-then-counter:
      1. ACTIVELY DEFEND: every turn, command every alive tank to
         attack_move TOWARD the nearest visible enemy. Stance:2 auto-
         fire alone is too passive — the rush closes in 3v2-then-
         3v1 chains before defenders bring all guns to bear. An
         explicit attack_move concentrates fire on the rush.
      2. Build fix (Building queue) — 2tnk prereq.
      3. Train extra 2tnk until we have an over-margin (target +1
         to absorb attrition).
      4. Once we hit the strike threshold, attack_move SURVIVORS
         east to the enemy base. (Some die en route; the survivors
         that arrive at (82,20) raze ≥1 enemy production building
         to satisfy P4.)
    Wins every (tier, seed)."""
    tank_target = {"easy": 5, "medium": 7, "hard": 7}[level]
    strike_threshold = {"easy": 4, "medium": 5, "hard": 5}[level]
    # Strike opens once `after_ticks` survive-window has passed so
    # P1 latches BEFORE the survivors depart.
    strike_opens_after = {"easy": 1000, "medium": 2400, "hard": 2300}[level]
    state = {"strike_issued": set()}
    milestone = {"fix": False}

    def pol(obs, Cmd):
        ob = obs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = obs.get("production", []) or []
        prod_items = set(prod) if all(isinstance(p, str) for p in prod) else set()
        units = obs.get("units_summary", []) or []
        tanks = [u for u in units if u.get("type") == "2tnk"]
        base = [b for b in ob if b["type"] == "fact"]
        bx, by = (base[0]["cell_x"], base[0]["cell_y"]) if base else (12, 20)
        game_tick = obs.get("game_tick", 0)
        cmds = []
        if "fix" in own_b:
            milestone["fix"] = True
        # Serial: fix first (prereq for 2tnk), then over-produce
        # tanks. Pre-placed weap is the production seat.
        if not milestone["fix"]:
            if "fix" not in prod_items:
                cmds.append(Cmd.build("fix"))
            cmds.append(Cmd.place_building("fix", bx + 8, by))
        elif len(tanks) < tank_target:
            if "2tnk" not in prod_items:
                cmds.append(Cmd.build("2tnk"))
        # PHASE A — DEFEND (game_tick < strike_opens_after):
        # focus-fire on the incoming rush. Find the closest
        # enemy unit/building and attack_move every tank toward it.
        if game_tick < strike_opens_after:
            enemies = obs.get("enemy_positions", []) or []
            # Pick the closest enemy by Chebyshev distance to base.
            def _dist(e):
                ex = e.get("cell_x", 999)
                ey = e.get("cell_y", 999)
                return max(abs(ex - bx), abs(ey - by))
            closest = min(enemies, key=_dist) if enemies else None
            if closest:
                tx, ty = closest["cell_x"], closest["cell_y"]
                # Only re-issue attack_move when target changes — but
                # cheap to issue every turn since attack_move is
                # idempotent for in-progress orders.
                tank_ids = [u["id"] for u in tanks]
                if tank_ids:
                    cmds.append(Cmd.attack_move(tank_ids, tx, ty))
        else:
            # PHASE B — STRIKE: send unsent tanks east; keep any
            # already-engaged tanks engaged.
            if len(tanks) >= strike_threshold:
                for u in tanks:
                    uid = u["id"]
                    if uid not in state["strike_issued"]:
                        cmds.append(Cmd.attack_move([uid], 82, 20))
                        state["strike_issued"].add(uid)
        if not cmds:
            cmds.append(Cmd.observe())
        return cmds
    return pol


# ── Pack-shape tests (no engine) ────────────────────────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK)
    assert pack.meta.id == "f11-defense-then-counter"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_set():
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert anchors, "benchmark_anchor required"


def test_then_chain_wired_into_win_condition():
    """`then:` chain P1=survive (fact alive after_ticks T), P2=harv
    alive, P3=≥N 2tnk, P4=≥K enemy razed."""
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
                    f"{lvl} then-chain must be 4 phases; got {clauses}"
                )
                # P1 must be a composite: survive after_ticks + fact.
                p1 = clauses[0]
                p1_inner = p1.get("all_of") or []
                assert any("after_ticks" in c for c in p1_inner), (
                    f"{lvl} P1 must gate on after_ticks (rush window); "
                    f"got {p1}"
                )
                # P2 harv alive.
                p2 = clauses[1].get("unit_type_count_gte") or {}
                assert p2.get("type") == "harv"
                # P3 2tnk count.
                p3 = clauses[2].get("unit_type_count_gte") or {}
                assert p3.get("type") == "2tnk"
                # P4 strike.
                assert "enemy_buildings_destroyed_gte" in clauses[3]


def test_every_level_has_fail_condition():
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_timeout_loss_is_reachable_on_every_level():
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
            assert wt <= reachable, (
                f"{lvl} within_ticks {wt} > reachable {reachable}"
            )
            assert wt <= ENGINE_CAP, f"{lvl} within_ticks {wt} > {ENGINE_CAP}"
        for ft in fts:
            assert ft <= reachable + 1, (
                f"{lvl} after_ticks {ft} > reachable {reachable}+1"
            )


def test_termination_flag_set_for_no_draw():
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
def test_panic_rush_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _panic_rush_policy(), seed=seed)
    assert res.outcome == "loss", (
        f"panic-rush must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_over_defense_loses(level, seed):
    """Build-only, never strike east. P4 never latches → LOSS."""
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _over_defense_policy(level), seed=seed)
    assert res.outcome == "loss", (
        f"over-defense must LOSE on {level} s={seed}; got {res.outcome} "
        f"(tick={res.signals.game_tick}, "
        f"own_b={res.signals.own_building_types})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_defense_then_counter_wins(level, seed):
    c = compile_level(load_pack(PACK), level)
    res = run_level(c, _intended_policy(level), seed=seed)
    tp = getattr(res.signals, "then_progress", {}) or {}
    assert res.outcome == "win", (
        f"intended defense-then-counter must WIN on {level} s={seed}; "
        f"got {res.outcome} (tick={res.signals.game_tick}, "
        f"then_progress={tp}, "
        f"own_b={res.signals.own_building_types}, "
        f"kills={res.signals.units_killed})"
    )
