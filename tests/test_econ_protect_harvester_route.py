"""econ-protect-harvester-route — ACTION en-route intercept validation.

The pack stages a long-distance harvester commute (proc at the west
base; mine at the far east on the same y-band) with a worker-priority
`raider` heavy tank driving at the harv from the far end. Two 2tnk
defenders pre-placed at the base on stance:2 (Defend) DO NOT have
line of sight to the mid-route kill zone — the agent must MOVE
them east along the harv's commute axis to intercept the raider on
the road. Standing the defenders idle at base lets the raider kill
the commuting harv → `not unit_type_count_gte:{harv,1}` fires → LOSS.

Recalibrated 2026-05 after the engine movement fixes ((A) attack_unit
on out-of-sight targets paths at real Mobile speed, (B) moving units
fire & take fire en route): under the old engine a 1tnk raider only
chipped the undefended commuting harv to ~34% HP and then chased it
home into the base defenders, so the "stand-at-base" turtle inversion
won for free. The raider is now a 2tnk that genuinely kills the
undefended harv on the route while the intended intercept still wins.

Real-world anchor: military convoy protection on the road / SC2
harass defense / supply-line doctrine.

Bar (CLAUDE.md "no defect, no cheat"):
   - stall LOSES every tier / every hard seed (no harvest order → no
     income AND defenders idle → raider kills harv anyway → LOSS).
   - defenders-stand-at-base LOSES every tier / every hard seed (harv
     is harvesting, but the base-bound defenders never see the raider
     and the harv dies on the route → LOSS).
   - intended intercept-on-route WINS every tier / seed (defenders
     attack_move east along the harv's lane and kill the raider
     before contact; harv completes round trips; bar cleared).
   - hard tier defines ≥2 agent spawn_point groups (NORTH y=14 /
     SOUTH y=26 supply routes) so a memorised opening cannot
     generalise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "econ-protect-harvester-route.yaml"


# ── declarative / schema invariants (no engine needed) ─────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "econ-protect-harvester-route"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) >= 3, (
        f"benchmark_anchor must list ≥3 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    # The three spec-mandated anchors.
    for needle in ("sc2 harass defense",
                   "military convoy protection",
                   "supply-line doctrine"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def test_uses_raider_bot():
    """The pack must declare the Wave-2 `raider` bot — worker-priority
    targeting binds the threat to the harv (no auto-retarget to the
    defenders) so the harv stays at risk until the defenders close
    range and engage."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "raider", f"expected raider bot, got {bot!r}"


def test_defenders_are_stance_2_defend():
    """stance:2 (Defend) for the defenders is REQUIRED by the design.
    stance:1 (ReturnFire) means defenders never engage because the
    raider attacks the harv only and never fires on the tanks — so
    stance:1 collapses the discrimination. stance:3 (AttackAnything)
    is wrong the OTHER way: idle defenders auto-HUNT the whole map and
    destroy the persistent far enemy fact (collapsing the run to DRAW
    on auto-done) and a "stand at base" stall then wins for free.
    stance:2 auto-fires only on an enemy in weapon range and never
    advances; the agent's load-bearing decision is to drive the
    defenders east onto the route (attack_move / attack_unit override
    stance)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        defenders = [
            a for a in c.scenario.actors
            if a.owner == "agent" and a.type == "2tnk"
        ]
        assert defenders, f"{lvl}: pack has no 2tnk defenders"
        for d in defenders:
            stance = getattr(d, "stance", None)
            # stance may be parsed onto the actor as `stance` attr;
            # fall back to raw dict-ish lookup on the actor's extras.
            if stance is None:
                # Pydantic-ish actor: extras may live under
                # `model_extra` (pydantic v2). Fail explicitly if we
                # cannot find it — the pack MUST declare stance:2.
                raw = getattr(d, "model_extra", None) or {}
                stance = raw.get("stance")
            assert stance == 2, (
                f"{lvl}: defender at {d.position} must be stance:2 "
                f"(Defend); got stance={stance!r}"
            )


def test_two_defenders_per_spawn_group():
    """The pack stages exactly 2 defender 2tnks per spawn group. Each
    spawn must carry its own defender pair (CLAUDE.md: any agent
    actor declaring spawn_point filters out agent actors WITHOUT one,
    so defenders are duplicated across BOTH hard groups)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        defs = [a for a in c.scenario.actors
                if a.owner == "agent" and a.type == "2tnk"]
        assert len(defs) == 2, f"{lvl}: expected 2 defenders, got {len(defs)}"
    c = compile_level(pack, "hard")
    per_group: dict[int, int] = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "2tnk":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group[sp] = per_group.get(sp, 0) + 1
    assert all(n == 2 for n in per_group.values()), (
        f"hard: each spawn group needs 2 defenders, got {per_group}"
    )


def test_hard_has_two_spawn_point_groups():
    """Hard curation: ≥2 distinct agent spawn_point groups (NORTH
    route y=14 / SOUTH route y=26) round-robined by seed. Engine-
    roundtrip asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_hard_spawn_groups_have_distinct_latitudes():
    """The hard tier's spawn variation must move the harv's commute
    axis — i.e. each spawn's harv (and its defender pair) sits at a
    distinct y-band. A memorised "send defenders east on y=20"
    opening (the easy/medium latitude) does NOT match either hard
    spawn (NORTH y=14 / SOUTH y=26)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    per_group_y: dict[int, set[int]] = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "harv":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group_y.setdefault(sp, set()).add(a.position[1])
    # Expect at least two distinct group y-bands.
    assert len(per_group_y) >= 2, (
        f"hard: expected ≥2 spawn groups with harvs, got {per_group_y}"
    )
    distinct = {next(iter(v)) for v in per_group_y.values() if len(v) == 1}
    assert len(distinct) >= 2, (
        f"hard: each spawn group's harv must sit at a distinct y, "
        f"got per-group y-set {per_group_y}"
    )


def test_each_level_has_exactly_one_harv_per_spawn_group():
    """One commuting harvester per spawn — the win clause asks for
    `unit_type_count_gte:{harv,1}` and the death of THAT single harv
    is the load-bearing failure mode."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium"):
        c = compile_level(pack, lvl)
        harvs = [a for a in c.scenario.actors
                 if a.owner == "agent" and a.type == "harv"]
        assert len(harvs) == 1, f"{lvl}: expected 1 harv, got {len(harvs)}"
    c = compile_level(pack, "hard")
    per_group: dict[int, int] = {}
    for a in c.scenario.actors:
        if a.owner != "agent" or a.type != "harv":
            continue
        sp = a.spawn_point if a.spawn_point is not None else 0
        per_group[sp] = per_group.get(sp, 0) + 1
    assert all(n == 1 for n in per_group.values()), (
        f"hard: each spawn group needs exactly 1 harv, got {per_group}"
    )


def test_kill_zone_is_beyond_base_sight():
    """The mine is FAR from the base so the raider intercepts the
    harv MID-ROUTE, well beyond defender base sight (~4-5 cells).
    A defender at x=15 cannot see the kill zone at x≈40..50; this
    is what forces "drive defenders east" as the load-bearing
    action rather than "ring defenders at the patch"."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        # Find each defender and each mine; require min distance
        # between any defender and the nearest mine to exceed 30
        # cells (well past any tank sight envelope).
        defenders = [a for a in c.scenario.actors
                     if a.owner == "agent" and a.type == "2tnk"]
        mines = [a for a in c.scenario.actors
                 if a.owner == "neutral" and a.type == "mine"]
        assert defenders and mines, (
            f"{lvl}: need at least one defender and one mine"
        )
        for d in defenders:
            # Same-spawn defender ↔ same-spawn mine distance: the
            # commute on the active y-band.
            dy = d.position[1]
            same_lat_mines = [m for m in mines if m.position[1] == dy]
            if not same_lat_mines:
                continue
            min_dx = min(abs(m.position[0] - d.position[0])
                         for m in same_lat_mines)
            assert min_dx >= 30, (
                f"{lvl}: defender at {d.position} too close to a "
                f"same-latitude mine (Δx={min_dx}); the kill zone "
                f"must be beyond base sight to force en-route intercept"
            )


def test_persistent_far_enemy_fact_present():
    """Engine auto-done guard: a persistent unarmed enemy `fact` at
    the far east ((120,20)) prevents the engine from terminating the
    run the moment the raider 2tnk(s) die (CLAUDE.md: enemy-elim
    auto-done collapses runs to DRAW before the economy bar
    evaluates)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        facts = [a for a in c.scenario.actors
                 if a.owner == "enemy" and a.type == "fact"]
        assert facts, f"{lvl}: needs at least one persistent enemy fact"
        # The far fact must be deep east so it can't be incidentally
        # killed on the route (raider 2tnk is staged at x=90; the
        # marker fact is at x=120 well east of that).
        far = max(facts, key=lambda a: a.position[0])
        assert far.position[0] >= 110, (
            f"{lvl}: persistent enemy fact must be far east "
            f"(>=x=110); got {far.position}"
        )


def test_all_tiers_have_reachable_deadlines():
    """Tick-alignment idiom: within_ticks ≤ ceiling AND after_ticks ≤
    ceiling AND within_ticks + 1 == after_ticks (a non-finisher
    LOSES the same tick the deadline elapses, no draw)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        ceiling = 93 + 90 * (c.max_turns - 1)
        wt = next(
            int(c["within_ticks"])
            for c in c.win_condition.model_dump()["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in c.fail_condition.model_dump()["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"
        assert wt + 1 == ft, (
            f"{lvl}: within_ticks {wt} / after_ticks {ft} mismatch "
            f"(non-finisher must LOSE the same tick the deadline lapses)"
        )


# ── predicate-level (no engine) ─────────────────────────────────────


def _ctx(*, units=(), tick=1000, cash=0, resources=0, own_buildings=()):
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        cash=cash,
        resources=resources,
        own_buildings=list(own_buildings),
        own_building_types={str(t).lower() for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_enforce_capability():
    """Win requires (EV bar AND harv alive AND fact AND proc) in
    time; fail fires on timeout OR harv dead OR fact destroyed."""
    c = compile_level(load_pack(PACK_PATH), "medium")
    one_harv = [{"cell_x": 30, "cell_y": 20, "type": "harv", "id": "9001"}]
    base_buildings = [("fact", 8, 20), ("proc", 12, 20)]

    # Intended: bar met, harv alive, base intact, in time → WIN.
    assert evaluate(
        c.win_condition,
        _ctx(units=one_harv, tick=2000, cash=1500,
             own_buildings=base_buildings),
    )
    # Bar short → not a WIN.
    assert not evaluate(
        c.win_condition,
        _ctx(units=one_harv, tick=2000, cash=1499,
             own_buildings=base_buildings),
    )
    # Harv dead → not a WIN AND a real FAIL.
    assert not evaluate(
        c.win_condition,
        _ctx(units=[], tick=2000, cash=4000,
             own_buildings=base_buildings),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, cash=4000,
             own_buildings=base_buildings),
    )
    # Timeout (tick past after_ticks) with bar unmet → FAIL.
    assert evaluate(
        c.fail_condition,
        _ctx(units=one_harv, tick=5402, cash=0,
             own_buildings=base_buildings),
    )
    # fact destroyed → FAIL.
    assert evaluate(
        c.fail_condition,
        _ctx(units=one_harv, tick=2000, cash=4000,
             own_buildings=[("proc", 12, 20)]),
    )


# ── engine-driven scripted policies ────────────────────────────────


def _own_units(rs):
    return rs.get("units_summary", []) or []


def _harv(rs):
    for u in _own_units(rs):
        if str(u.get("type", "")).lower() == "harv":
            return u
    return None


def _defenders(rs):
    return [u for u in _own_units(rs)
            if str(u.get("type", "")).lower() == "2tnk"]


def _stall(rs, Command):
    """Pure observe — defenders never move and the harv never gets a
    harvest order. The raider beelines on the idle harv and kills
    it → `not unit_type_count_gte:{harv,1}` fires → LOSS."""
    return [Command.observe()]


def _defenders_stand_at_base(rs, Command):
    """The lazy "turtle" inversion the scenario is engineered
    against: harvest IS re-issued to the harv (so income would
    accumulate if the harv survived), but the defenders are
    explicitly told to stop / hold at base. The defender sight
    envelope does not reach the kill zone (x≈40..50) and the harv
    dies on the route → LOSS."""
    cmds = []
    h = _harv(rs)
    if h is not None:
        # Issue harvest to the active mine on the harv's own y-band.
        my = int(h.get("cell_y", 20))
        cmds.append(Command.harvest([str(h["id"])], 60, my))
    for d in _defenders(rs):
        cmds.append(Command.stop([str(d["id"])]))
    return cmds or [Command.observe()]


def _raider(rs):
    """The inbound enemy 2tnk raider, once it enters the agent's
    vision (surfaced in `enemy_summary`)."""
    for e in (rs.get("enemy_summary") or []):
        if str(e.get("type", "")).lower() == "2tnk":
            return e
    return None


def _intended_intercept_on_route(rs, Command):
    """The intended capability:

      1. Re-issue harvest each turn to the (single) harv so it keeps
         commuting between the proc and the far mine on its OWN
         y-band.
      2. Drive the defender pair EAST along the harv's lane to
         INTERCEPT the raider: attack_unit it once it is in vision,
         else attack_move toward its last-seen position. The
         defenders are stance:2 (Defend) — they will not advance on
         their own, so the explicit attack order is load-bearing.
         The defender pair out-damages the light-tank raider and
         destroys it before it reaches the harv; income accumulates.

    The (active-spawn) harv's y-band determines the harvest target;
    the raider's tracked position drives the intercept — works for
    easy/medium (y=20), hard NORTH (y=14), hard SOUTH (y=26).
    """
    cmds = []
    h = _harv(rs)
    defenders = _defenders(rs)
    if h is None and not defenders:
        return [Command.observe()]
    raider = _raider(rs)
    if h is not None:
        my = int(h.get("cell_y", 20))
        cmds.append(Command.harvest([str(h["id"])], 60, my))
        for d in defenders:
            if raider is not None:
                cmds.append(Command.attack_unit([str(d["id"])],
                                                str(raider["id"])))
            else:
                # Raider not yet in vision — push east along the
                # harv's lane to close the intercept distance.
                cmds.append(Command.attack_move([str(d["id"])], 55, my))
    else:
        # Lost the harv before issuing the first order — best-effort
        # is to still push the defenders east on their own row.
        for d in defenders:
            dy = int(d.get("cell_y", 20))
            cmds.append(Command.attack_move([str(d["id"])], 55, dy))
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_intercept_on_route_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_intercept_on_route, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended intercept-on-route should "
        f"WIN, got {r.outcome} after {r.turns} turns "
        f"(losses={r.signals.units_lost} ev="
        f"{r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    """No harvest order → no income; defenders idle → raider kills
    the harv on the route. Either the EV bar lapses on after_ticks
    OR the harv dies first — both routes are real LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real LOSS "
        f"(no income AND harv unprotected), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_defenders_stand_at_base_loses(level, seed):
    """The harv harvests but the defenders are held at base —
    they never have sight of the mid-route kill zone (x≈40..50),
    the raider kills the harv on the route → LOSS via
    `not unit_type_count_gte:{harv,1}`. This is the
    "defenders-stand-at-base" inversion the scenario is engineered
    against."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _defenders_stand_at_base, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: defenders-stand-at-base must LOSE "
        f"(harv dies on the route while base-bound defenders are "
        f"out of sight), got {r.outcome} "
        f"(losses={r.signals.units_lost})"
    )
