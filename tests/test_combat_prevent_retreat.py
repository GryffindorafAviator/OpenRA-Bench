"""combat-prevent-retreat — cut-off encirclement before engaging.

The bar: 4× medium tanks (2tnk) start at the west edge; a mixed e1+e3
infantry cluster holds the centre (60, 16..23). The win predicate
requires:

  • `units_killed_gte: K`              — the full cluster wiped
    (K=5 easy / 7 medium / 8 hard)
  • `units_in_region_gte:{85,20,r=8,n=1}` — ≥1 own tank in the east
    cut-off region at evaluation time
  • `own_units_gte: 3`                  — at least 3 tanks survive
  • `within_ticks: 4500`                — stallers can't draw

This combination forces the Cannae idiom: ONE flank tank routes
off-axis (via y=5..10 from a NORTH spawn or y=30..35 from a SOUTH
spawn, out of e3 Dragon range) to take the eastern anvil; the
remaining THREE engage the centre from the west. The cluster's
anti-armour e3s punish a head-on column charge (`brute_east` loses
≥2 tanks → survival cap busts); a centre-only charge leaves no tank
east (in-region clause fails); a solo east tank can't pull the kill
bar in budget.

Discrimination (verified per-level × per-seed 1–4):

  • stall (observe-only)              — LOSS on every level/seed
                                        (no kills, deadline fires).
  • brute_east (attack_move all to 85,20)
                                       — LOSS: column bled by e3
                                         Dragons; survival cap busts.
  • brute_centre (attack_move all to 60,20)
                                       — LOSS: cluster killed but
                                         no tank east → in-region
                                         clause fails → after_ticks.
  • east_only (move 1 east, hold rest)— LOSS: kill bar never met
                                         (1 tank vs cluster too slow).
  • intended (cut-off then engage)    — WIN every level/seed.

Validation is scripted (no model / network)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-prevent-retreat.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-prevent-retreat"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list ≥3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("encirclement", "cannae", "sc2"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(*, units=(), tick=1000, kills=0, lost=0):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=kills,
        units_lost=lost,
        cash=0,
        resources=0,
        power_provided=0,
        power_drained=0,
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        explored_percent=0.0,
        then_progress={},
        seq_progress={},
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _tanks_at(n, x=85, y=20):
    return [
        {"cell_x": x, "cell_y": y, "type": "2tnk", "id": str(2000 + i)}
        for i in range(n)
    ]


def test_predicates_per_level():
    """Region n=1 + kill bar + survival cap + deadline all load-bearing."""
    pack = load_pack(PACK_PATH)
    # (kill_bar, surv_min, within, after_fail)
    expectations = {
        "easy":   (5, 3, 4500, 4501),
        "medium": (7, 3, 4500, 4501),
        "hard":   (8, 3, 4500, 4501),
    }
    for lvl, (kb, surv, w, af) in expectations.items():
        c = compile_level(pack, lvl)
        # Intended win: ≥1 tank in east region + kill bar + ≥surv alive.
        ctx_win = _ctx(units=_tanks_at(surv), tick=w - 100, kills=kb, lost=4 - surv)
        assert evaluate(c.win_condition, ctx_win), f"{lvl}: intended should WIN"
        # No tank in east region → in-region clause unmet.
        ctx_no_east = _ctx(
            units=[{"cell_x": 60, "cell_y": 20, "type": "2tnk", "id": str(3000 + i)}
                   for i in range(surv)],
            tick=w - 100, kills=kb, lost=4 - surv,
        )
        assert not evaluate(c.win_condition, ctx_no_east), (
            f"{lvl}: 0 tanks in east region must not satisfy win"
        )
        # Kill bar unmet → not a win.
        ctx_low_kills = _ctx(units=_tanks_at(surv), tick=w - 100, kills=kb - 1, lost=0)
        assert not evaluate(c.win_condition, ctx_low_kills), (
            f"{lvl}: kills < {kb} must not satisfy win"
        )
        # Survival cap busted → not a win + fails.
        ctx_lost = _ctx(units=_tanks_at(surv - 1), tick=w - 100, kills=kb, lost=4 - surv + 1)
        assert not evaluate(c.win_condition, ctx_lost)
        assert evaluate(c.fail_condition, ctx_lost), (
            f"{lvl}: survival<{surv} must FAIL"
        )
        # Timeout busted → fails.
        ctx_late = _ctx(units=_tanks_at(surv), tick=af + 1, kills=kb, lost=0)
        assert evaluate(c.fail_condition, ctx_late), (
            f"{lvl}: tick>after must FAIL"
        )


def test_timeout_reachable_inside_max_turns():
    """No DRAW degeneracy: after_ticks fail trigger must be reachable
    within max_turns (engine advances ~90 ticks per turn ⇒ max tick
    ≈ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 4501 <= max_tick, (
            f"{lvl}: after_ticks 4501 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )


def test_tanks_are_2tnk_and_west_edge():
    """Agent strike force is 4× 2tnk at the west edge (x=6) on every
    level — the encirclement geometry is load-bearing."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        agent = [a for a in c.scenario.actors if a.owner == "agent"]
        for a in agent:
            assert a.type == "2tnk", f"{lvl}: agent must be 2tnk; got {a.type}"
            assert a.position[0] == 6, (
                f"{lvl}: agent must start at x=6 (west edge); got {a.position}"
            )
        # Easy/medium: 4 tanks. Hard: 4 per spawn_point (8 declared
        # actors total, but only one spawn_point group places per seed).
        if lvl == "hard":
            assert len(agent) == 8, f"hard: declares 4 tanks × 2 spawn_points = 8"
        else:
            assert len(agent) == 4, f"{lvl}: 4 tanks expected; got {len(agent)}"


def test_enemy_cluster_e1_plus_e3_mix():
    """The centre cluster mixes e1 (filler) + e3 (anti-armour). The
    e3 component is load-bearing — without it a brute attack-move
    column charge wouldn't bleed below the survival cap."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        cluster = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type in ("e1", "e3")
        ]
        assert cluster, f"{lvl}: no central cluster declared"
        e3 = [a for a in cluster if a.type == "e3"]
        assert len(e3) >= 2, (
            f"{lvl}: cluster needs ≥2 e3 to bleed a frontal column charge; "
            f"got {len(e3)}"
        )
        # No cluster member at a known silently-no-place cell.
        for a in cluster:
            assert tuple(a.position) != (50, 20), (
                f"{lvl}: cluster member at (50,20) — CLAUDE.md no-place footgun"
            )
            assert a.position[0] == 60, (
                f"{lvl}: cluster member off-column; got {a.position}"
            )


def test_persistent_far_enemy_marker_present():
    """Engine auto-`done` mitigation: a far unarmed enemy `fact` keeps
    the episode alive past full cluster elimination so the in-region
    predicate evaluates on the terminal frame (no DRAW collapse)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        far_facts = [
            a for a in c.scenario.actors
            if a.owner == "enemy" and a.type == "fact"
            and a.position[0] >= 100
        ]
        assert far_facts, f"{lvl}: needs a far persistent enemy fact sentinel"


def test_enemy_is_static_no_bot_type():
    """Enemies are stationary defenders (no `bot_type`) — a hunt bot
    would chase the lone cutoff tank as it flanks, breaking the
    off-axis safe-route assumption AND letting a staller win when
    enemies walk into the tanks."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else getattr(pack.base, "enemy", {})
    assert (enemy or {}).get("bot_type") is None, (
        "enemy must not declare a bot_type — see file header docstring"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins which staging corridor (north y=12 or south y=28)
    the tank column starts in. The natural flank route flips per seed
    so no memorised opening generalises."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


# ── engine-driven scripted policies ─────────────────────────────────


def _tank_ids(rs):
    return [
        str(u["id"])
        for u in (rs.get("units_summary") or [])
        if str(u.get("type", "")).lower() == "2tnk"
    ]


def _stall(rs, Command):
    """Pure observe — no kills, no in-region, deadline LOSS."""
    return [Command.observe()]


def _brute_east(rs, Command):
    """All 4 tanks attack-move EAST through the cluster toward (85,20).
    The column passes within e3 Dragon range and bleeds below the
    survival cap → LOSS on every level."""
    ids = _tank_ids(rs)
    return [Command.attack_move(ids, 85, 20)] if ids else [Command.observe()]


def _brute_centre(rs, Command):
    """All 4 tanks attack-move the centre cluster at (60,20). May clear
    the cluster (kills bar met) but NO TANK IS IN THE EAST REGION at
    evaluation time → in-region clause fails → after_ticks LOSS."""
    ids = _tank_ids(rs)
    return [Command.attack_move(ids, 60, 20)] if ids else [Command.observe()]


def _east_only(rs, Command):
    """Send ONLY one tank east, hold the rest. The solo east tank
    can't pull the kill bar in budget; held tanks don't engage →
    kill bar fails → LOSS."""
    ids = _tank_ids(rs)
    if not ids:
        return [Command.observe()]
    return [
        Command.move_units([ids[0]], target_x=85, target_y=20),
        Command.stop(ids[1:]),
    ]


def _make_intended():
    """Cannae intended play. Pick the off-axis safe latitude based on
    the actual spawn (NORTH spawn → off-y=5; SOUTH spawn → off-y=35),
    pick the cutoff tank closest to that latitude, route it via three
    waypoints out of e3 Dragon range to (85,20). The remaining three
    tanks attack-move the centre cluster from the west."""
    state = {}

    def policy(rs, Command):
        units = [
            u for u in (rs.get("units_summary") or [])
            if str(u.get("type", "")).lower() == "2tnk"
        ]
        if not units:
            return [Command.observe()]
        if "off_y" not in state:
            avg = sum(int(u["cell_y"]) for u in units) / len(units)
            state["off_y"] = 5 if avg < 20 else 35
            us = sorted(units, key=lambda u: abs(int(u["cell_y"]) - state["off_y"]))
            state["cutoff"] = str(us[0]["id"])
            state["phase"] = 0
        cmds = []
        cutoff = state["cutoff"]
        cu = next((u for u in units if str(u["id"]) == cutoff), None)
        others = [str(u["id"]) for u in units if str(u["id"]) != cutoff]
        off_y = state["off_y"]
        if cu is not None:
            cx, cy = int(cu["cell_x"]), int(cu["cell_y"])
            wps = [(20, off_y), (85, off_y), (85, 20)]
            tx, ty = wps[min(state["phase"], len(wps) - 1)]
            cmds.append(Command.move_units([cutoff], target_x=tx, target_y=ty))
            if (
                abs(cx - tx) <= 2
                and abs(cy - ty) <= 2
                and state["phase"] < len(wps) - 1
            ):
                state["phase"] += 1
        if others:
            cmds.append(Command.attack_move(others, 60, 20))
        return cmds or [Command.observe()]

    return policy


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no engagement → kill bar + region clause unmet); got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_east_loses(level, seed):
    """Attack-move all tanks east through the cluster. The e3 Dragon
    fire bleeds the column below the survival cap → `own_units_gte:3`
    fails → LOSS on every level/seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_east, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute_east must LOSE (column bled by "
        f"e3 Dragons); got {r.outcome} "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_centre_loses(level, seed):
    """Attack-move all tanks to the centre cluster (60,20). Even when
    the cluster is wiped (kill bar met), NO TANK IS IN THE EAST CUT-
    OFF REGION → in-region clause fails → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_centre, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute_centre must LOSE (no tank in "
        f"east region); got {r.outcome} "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_east_only_loses(level, seed):
    """Send only one tank east; hold the rest. The solo east tank
    can't pull the kill bar in budget → LOSS on every level/seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _east_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: east_only must LOSE (kill bar unmet); "
        f"got {r.outcome} (kills={r.signals.units_killed}, "
        f"lost={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_cutoff_then_engage_wins(level, seed):
    """Cannae intended play: ONE tank flanks via the off-axis lane
    (y=5 from NORTH spawn, y=35 from SOUTH spawn — both out of e3
    Dragon range) to the eastern anvil at (85,20); the remaining
    THREE attack-move the centre cluster. Kill bar AND in-region AND
    survival all satisfied → WIN on every level/seed."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _make_intended(), seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended cut-off-then-engage should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, lost={r.signals.units_lost})"
    )
