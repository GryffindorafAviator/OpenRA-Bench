"""scout-detect-incoming-army — perception pack: early-warning scout
to localise an incoming army's approach axis + targeted intercept.

The pack tests TWO chained skills:

  1) PERCEPTION (the scout step): push jeeps forward to localise
     which lane the hunt squad is arriving on (the agent's
     base-only vision cannot read the eastern frontier in time).
  2) ACTION (the intercept step): re-position the medium-tank
     defenders forward on the localised lane so the tank cannon
     engages the e1 stack at range BEFORE the squad closes to
     short range and overruns the base.

Win predicate (load-bearing across all four axes):
  * `enemies_discovered_gte:4`  — the SPOT bar (the SC2 early-warn
    scout / NORAD / IDS anchor — the agent must SEE ≥4 of the
    incoming force).
  * `units_killed_gte:3`         — the INTERCEPT bar (the targeted
    response must actually kill at least part of the army).
  * `units_lost_lte:2`           — the PRESERVATION bar (out of 5
    starting units; a sloppy intercept that bleeds 3+ units busts
    the attrition cap even if the kill bar is met).
  * `units_in_region_gte` at the
    NORTH-flank (45,10) intercept point (easy) OR `any_of` over
    the NORTH (45,10) AND SOUTH (45,30) intercept points (medium
    / hard) — the FORWARD-INTERCEPT clause. A base-only auto-
    defence that satisfies the kill bar still FAILS this clause
    because every defender stays west of x≈20, well outside the
    forward intercept radius.
  * `within_ticks:2500` paired with `after_ticks:2501` ⇒ a non-
    finisher is a real reachable timeout LOSS (max_turns 50 →
    reachable tick 93+90·49=4503 ≥ 2501 even in interrupt mode),
    never a draw degeneracy.

Scripted policies cover the four bar-defining outcomes:
  * stall              → LOSS (tanks overrun at base, units_lost>2
                         OR after_ticks fires before the kill bar
                         and position clause are met)
  * no-scout blind     → LOSS (units march east at y=20 mid-lane,
                         position clause unmet on either flank)
  * scout-no-position  → LOSS (scouts forward but tanks stay at
                         base, e3 rockets bust the attrition cap)
  * intended           → WIN (scouts localise the lane, tanks
                         attack_unit on the squad's e1s at cannon
                         range, kill ≥3 and keep ≥3 alive)
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "scout-detect-incoming-army.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── scenario-shape invariants ─────────────────────────────────────────


def test_pack_compiles_with_three_levels_and_hunt_bot():
    pack = load_pack(PACK)
    assert pack.meta.id == "scout-detect-incoming-army"
    assert pack.meta.capability == "perception"
    assert set(pack.levels) == {"easy", "medium", "hard"}
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.map_supported
        bot = getattr(c.scenario.enemy, "bot_type", None) or getattr(
            c.scenario.enemy, "bot", None
        )
        assert str(bot).lower() == "hunt", (lvl, bot)


def test_benchmark_anchor_set():
    """Wave-6 seed-taxonomy contract: anchors must call out the
    SC2 early-warn scout / NORAD / IDS / military recon framings."""
    pack = load_pack(PACK)
    anchors = pack.meta.benchmark_anchor or []
    assert any("SC2 early-warn scout" in a for a in anchors), anchors
    assert any("NORAD" in a for a in anchors), anchors
    assert any("intrusion detection" in a.lower() for a in anchors), anchors
    assert any(
        "reconnaissance" in a.lower() or "recon" in a.lower()
        for a in anchors
    ), anchors


def test_every_level_has_fail_condition():
    """No silent draws — every level emits a real LOSS on timeout
    or attrition or force-wipe."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks / after_ticks must be reachable inside max_turns
    (engine ≤90 ticks/turn → reachable max = 93 + 90·(N-1)).
    Otherwise the deadline never bites ⇒ DRAW degeneracy."""
    pack = load_pack(PACK)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        max_turns = pack.levels[lvl].max_turns
        reachable = 93 + 90 * (max_turns - 1)
        # Collect every within_ticks/after_ticks leaf.
        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)

        wts: list[int] = []
        win = c.win_condition.model_dump(exclude_none=True)
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks leaf"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable} "
                f"(max_turns={max_turns}) — deadline never bites ⇒ draw"
            )
        # The after_ticks fail clause must ALSO be reachable.
        fts: list[int] = []
        fc = c.fail_condition.model_dump(exclude_none=True)
        _collect(fc, "after_ticks", fts)
        assert fts, f"{lvl} has no after_ticks fail leaf"
        for ft in fts:
            assert ft <= reachable, (
                f"{lvl} after_ticks={ft} > reachable={reachable} "
                f"(max_turns={max_turns}) — fail never bites ⇒ draw"
            )


def test_forward_position_clause_present_in_win():
    """The load-bearing perception teeth: every level's win must
    require ≥1 agent unit in a FORWARD intercept region (not a
    base-only zone). Without this, stall / no-scout policies that
    happen to meet the kill bar via base auto-defence could win."""
    pack = load_pack(PACK)

    def _walk(node, out):
        if isinstance(node, dict):
            if "units_in_region_gte" in node:
                out.append(node["units_in_region_gte"])
            for v in node.values():
                _walk(v, out)
        elif isinstance(node, list):
            for v in node:
                _walk(v, out)

    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        win = c.win_condition.model_dump(exclude_none=True)
        found: list = []
        _walk(win, found)
        assert found, f"{lvl}: missing units_in_region_gte forward clause"
        # Every region must be FORWARD of the base (x≥30): a clause
        # at the spawn cells would be satisfied by stall.
        for region in found:
            assert region["x"] >= 30, (
                f"{lvl}: forward region must have x>=30, got {region}"
            )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier contract: ≥2 distinct seed-driven spawn_point groups
    so the base latitude varies by seed."""
    c = compile_level(load_pack(PACK), "hard")
    groups = {
        a.spawn_point for a in c.scenario.actors
        if a.owner == "agent" and a.spawn_point is not None
    }
    assert groups == {0, 1}, groups
    # In-bounds check (rush-hour-arena playable y ≈ 2..38, x ≈ 2..126):
    for a in c.scenario.actors:
        x, y = a.position
        assert 2 <= x <= 126 and 2 <= y <= 38, (a.type, a.position)


def test_actor_composition_2_jeeps_3_tanks():
    """Per-spec actor manifest: 2 jeep scouts + 3 medium tanks PER
    spawn group at the WEST base. Easy/medium have one implicit
    group (=2 jeeps + 3 tanks total); hard has TWO declared spawn
    groups (=4 jeeps + 6 tanks total — the spawn_point filter
    selects ONE group at reset, leaving exactly 2 jeeps + 3 tanks
    in play)."""
    for lvl in LEVELS:
        c = compile_level(load_pack(PACK), lvl)
        jeeps = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "jeep"
        ]
        tanks = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        groups = {
            a.spawn_point for a in c.scenario.actors
            if a.owner == "agent" and a.spawn_point is not None
        }
        n_groups = max(1, len(groups))
        assert len(jeeps) == 2 * n_groups, (lvl, jeeps)
        assert len(tanks) == 3 * n_groups, (lvl, tanks)


# ── scripted policies ─────────────────────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary") or []


def stall(rs, C):
    """Observe-only — every unit sits at the base spawn. The hunt
    squad closes and the e3 rockets in the squad shred the tanks
    at close range; units_lost > 2 → LOSS even if the squad is
    eventually wiped, AND the forward-position clause is never
    satisfied."""
    return [C.observe()]


def no_scout_blind_east(rs, C):
    """No scouting: attack_move all units east at mid-latitude
    (y=20). The mid-lane line satisfies NEITHER the NORTH-forward
    (y=10) nor the SOUTH-forward (y=30) position clause, so even
    when the units accidentally kill enough enemies the win clause
    is unmet → after_ticks fires → LOSS."""
    units = _own_units(rs)
    if not units:
        return [C.observe()]
    ids = [str(u["id"]) for u in units]
    return [C.attack_move(ids, target_x=80, target_y=20)]


def scout_but_no_position(rs, C):
    """Scouts push out (one NORTH, one SOUTH) — detection bar met
    — but tanks STAY at the base. The hunt squad closes on the
    tanks; e3 rockets bust the attrition cap → LOSS even though
    detection succeeded."""
    units = _own_units(rs)
    if not units:
        return [C.observe()]
    jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
    cmds = []
    if len(jeeps) >= 1:
        cmds.append(C.move_units([str(jeeps[0]["id"])], target_x=70, target_y=10))
    if len(jeeps) >= 2:
        cmds.append(C.move_units([str(jeeps[1]["id"])], target_x=70, target_y=30))
    if not cmds:
        cmds.append(C.observe())
    return cmds


def make_intended():
    """Scout-and-react: jeeps push forward on BOTH candidate lanes
    on turn 1; the moment any enemy unit is spotted, lock in the
    commit lane (the one with the closer enemy) and have tanks
    attack_unit on the nearest visible hostile on that lane (or
    any hostile if the commit lane has none visible yet). The
    attack_unit verb keeps tanks engaged rather than parking at
    a static cell."""
    state = {"turn": 0, "commit_y": None}

    def policy(rs, C):
        state["turn"] += 1
        units = _own_units(rs)
        if not units:
            return [C.observe()]
        defs = [u for u in units if str(u.get("type", "")).lower() == "2tnk"]
        jeeps = [u for u in units if str(u.get("type", "")).lower() == "jeep"]
        es = rs.get("enemy_summary") or []
        rusher_units = [
            e for e in es
            if str(e.get("type", "")).lower() in ("e1", "e3")
        ]
        # Localise threat axis from the first visible enemy(s).
        if rusher_units and state["commit_y"] is None:
            n = [e for e in rusher_units if int(e.get("cell_y", 20)) < 20]
            s = [e for e in rusher_units if int(e.get("cell_y", 20)) >= 20]
            if n and s:
                nx = min(int(e.get("cell_x", 100)) for e in n)
                sx = min(int(e.get("cell_x", 100)) for e in s)
                state["commit_y"] = 10 if nx <= sx else 30
            elif n:
                state["commit_y"] = 10
            elif s:
                state["commit_y"] = 30
        cmds = []
        # Turn 1: scout push (one NORTH, one SOUTH).
        if jeeps and state["turn"] == 1:
            cmds.append(C.move_units([str(jeeps[0]["id"])], target_x=40, target_y=10))
            if len(jeeps) > 1:
                cmds.append(C.move_units([str(jeeps[1]["id"])], target_x=40, target_y=30))
        # Tanks: attack_unit on nearest visible hostile if any (the
        # attack_unit verb keeps tanks engaged), else attack_move
        # forward toward the commit lane (or NORTH by default).
        if defs:
            def_ids = [str(d["id"]) for d in defs]
            if rusher_units:
                # Filter to the committed lane if known.
                if state["commit_y"] is not None:
                    lane = [
                        e for e in rusher_units
                        if abs(int(e.get("cell_y", 20)) - state["commit_y"]) < 10
                    ]
                    if not lane:
                        lane = rusher_units
                else:
                    lane = rusher_units
                tx = sum(int(d["cell_x"]) for d in defs) // len(defs)
                ty = sum(int(d["cell_y"]) for d in defs) // len(defs)
                tgt = min(
                    lane,
                    key=lambda e: (int(e.get("cell_x", 100)) - tx) ** 2
                    + (int(e.get("cell_y", 20)) - ty) ** 2,
                )
                cmds.append(C.attack_unit(def_ids, str(tgt.get("id"))))
            else:
                ty = state["commit_y"] if state["commit_y"] is not None else 10
                cmds.append(C.attack_move(def_ids, target_x=40, target_y=ty))
        if not cmds:
            cmds.append(C.observe())
        return cmds

    return policy


# ── solvency: intended WINS every level + every hard seed ─────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_scout_and_intercept_wins(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, make_intended(), seed=seed)
    assert r.outcome == "win", (
        f"intended scout-and-intercept must WIN on {level} s={seed}; "
        f"got {r.outcome} (tick={r.signals.game_tick}, "
        f"kills={r.signals.units_killed}, "
        f"lost={r.signals.units_lost}, "
        f"seen={len(r.signals.enemies_seen_ids)})"
    )


# ── no-cheat: every lazy / blind / partial policy LOSES (not draws) ──


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, stall, seed=seed)
    assert r.outcome == "loss", (
        f"stall must LOSE on {level} s={seed}; got {r.outcome} "
        f"(tick={r.signals.game_tick}, kills={r.signals.units_killed}, "
        f"lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_no_scout_blind_east_loses(level, seed):
    """Blind mid-lane attack-move never satisfies the forward
    position clause on N or S — even when it accidentally wipes
    the army it still LOSES."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, no_scout_blind_east, seed=seed)
    assert r.outcome == "loss", (
        f"no-scout blind must LOSE on {level} s={seed}; got {r.outcome} "
        f"(tick={r.signals.game_tick}, kills={r.signals.units_killed}, "
        f"lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_scout_but_no_position_loses(level, seed):
    """Scouts forward but tanks STAY at the base — the squad
    closes and the e3 rockets bust the attrition cap → LOSS."""
    c = compile_level(load_pack(PACK), level)
    r = run_level(c, scout_but_no_position, seed=seed)
    assert r.outcome == "loss", (
        f"scout-no-position must LOSE on {level} s={seed}; got {r.outcome} "
        f"(tick={r.signals.game_tick}, kills={r.signals.units_killed}, "
        f"lost={r.signals.units_lost})"
    )


# ── Spawn-variation contract (hard) ───────────────────────────────────


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_seeds_run_to_terminal(seed):
    """Hard's two spawn_point groups must actually round-robin
    cleanly — stall on each seed completes deterministically (the
    cross-seed start-cell distinctness contract is enforced by
    tests/test_hard_tier.py::test_curated_hard_still_compiles_and_runs)."""
    c = compile_level(load_pack(PACK), "hard")
    r = run_level(c, stall, seed=seed)
    assert r.outcome == "loss"
