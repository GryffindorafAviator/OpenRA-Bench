"""rob-multiple-simultaneous-pressures — compound-incident triage.

Wave-6 REASONING pack: three live pressures must be handled at once —
DEFENCE (hunt charge on the fact), ECONOMY (raider tanks on the harv
patch), TECH (`weap` by deadline). Single-axis policies cannot win;
the intended multi-handle keeps all three loops running in parallel.

Bar (per CLAUDE.md): the intended multi-handle policy must WIN on
every (level, seed); stall / focus-defence-skip-tech / focus-tech-
lose-econ must LOSE on every (level, seed). No draws.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "rob-multiple-simultaneous-pressures.yaml"
LEVELS = ("easy", "medium", "hard")
SEEDS = (1, 2, 3, 4)


# ── Pack-shape tests (cheap; do not run the engine) ─────────────────


def test_pack_compiles_with_three_levels():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "rob-multiple-simultaneous-pressures"
    assert pack.meta.capability == "reasoning"
    assert set(pack.levels) == {"easy", "medium", "hard"}


def test_meta_benchmark_anchor_includes_required_anchors():
    pack = load_pack(PACK_PATH)
    anchors = pack.meta.benchmark_anchor or []
    joined = " ".join(anchors).lower()
    for needle in (
        "sc2 multi-front",
        "ops triage",
        "dr multi-system",
        "military",
    ):
        assert needle in joined, f"missing anchor keyword: {needle!r}"


def test_uses_hunt_bot():
    """A `hunt` bot drives both pressure axes — the raiders at the
    patch hunt the harvesters; the flanking infantry hunt the base
    buildings. The pack must declare `hunt`."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "hunt", f"expected hunt bot, got {bot!r}"


def test_starting_cash_is_2000_exact():
    """starting_cash must equal weap cost (2000) so cash slack cannot
    be used to bypass either the tech or the income-from-harvest loop."""
    pack = load_pack(PACK_PATH)
    assert pack.starting_cash == 2000, (
        f"expected starting_cash 2000 (weap cost), got {pack.starting_cash}"
    )


def test_every_level_has_fail_condition():
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        c = compile_level(pack, lvl)
        assert c.fail_condition is not None, f"{lvl} missing fail_condition"


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the base latitude."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    sp = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(sp) >= 2, f"hard needs ≥2 spawn_point groups, got {sp}"


def test_tick_budget_aligned_with_max_turns():
    """within_ticks and after_ticks must be reachable inside max_turns
    (~90 ticks/turn): otherwise the deadline never bites and a stall
    draws instead of losing."""
    pack = load_pack(PACK_PATH)
    for lvl in LEVELS:
        level_def = pack.levels[lvl]
        max_turns = level_def.max_turns
        reachable = 93 + 90 * (max_turns - 1)
        win = compile_level(pack, lvl).win_condition.model_dump(exclude_none=True)

        def _collect(node, key, out):
            if isinstance(node, dict):
                if key in node:
                    out.append(node[key])
                for v in node.values():
                    _collect(v, key, out)
            elif isinstance(node, list):
                for v in node:
                    _collect(v, key, out)
        wts: list = []
        _collect(win, "within_ticks", wts)
        assert wts, f"{lvl} has no within_ticks clock teeth"
        for wt in wts:
            assert wt <= reachable, (
                f"{lvl} within_ticks={wt} > reachable={reachable}"
            )
        fail = compile_level(pack, lvl).fail_condition.model_dump(exclude_none=True)
        ats: list = []
        _collect(fail, "after_ticks", ats)
        assert ats, f"{lvl} has no after_ticks fail teeth"
        for at in ats:
            assert at <= reachable, (
                f"{lvl} after_ticks={at} > reachable={reachable}"
            )


# ── Predicate-level checks (synthesised WinContext, no engine) ──────


def _ctx(*, own_buildings=(), units=(), tick=1000, cash=2000, resources=0):
    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
        own_buildings=list(own_buildings),
        own_building_types={t for (t, _, _) in own_buildings},
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        cash=cash,
        resources=resources,
        then_progress={},
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    full = [
        ("fact", 10, 18), ("proc", 8, 18), ("powr", 8, 20),
        ("weap", 14, 18),
    ]
    harvs = [
        {"cell_x": 14, "cell_y": 16, "type": "harv"},
        {"cell_x": 14, "cell_y": 22, "type": "harv"},
    ]
    # WIN: weap + proc + ≥1 harv + fact + EV ≥ 1000, in time.
    assert evaluate(
        c.win_condition,
        _ctx(own_buildings=full, units=harvs, tick=3000, cash=1200),
    )
    # No weap → not a win.
    no_weap = [b for b in full if b[0] != "weap"]
    assert not evaluate(
        c.win_condition,
        _ctx(own_buildings=no_weap, units=harvs, tick=3000, cash=1200),
    )
    # EV below 1000 → not a win (focus-tech-lose-econ failure mode).
    assert not evaluate(
        c.win_condition,
        _ctx(own_buildings=full, units=harvs, tick=3000, cash=0),
    )
    # No harv → fail.
    assert evaluate(
        c.fail_condition,
        _ctx(own_buildings=full, units=[], tick=3000, cash=1200),
    )
    # No fact (base razed) → fail.
    no_fact = [b for b in full if b[0] != "fact"]
    assert evaluate(
        c.fail_condition,
        _ctx(own_buildings=no_fact, units=harvs, tick=3000, cash=1200),
    )
    # Past deadline → fail.
    assert evaluate(
        c.fail_condition,
        _ctx(own_buildings=full, units=harvs, tick=5402, cash=1200),
    )


def test_predicates_hard_tighter_clock():
    """Hard's within_ticks is 4500 (tighter than easy/medium's 5400)."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    full = [
        ("fact", 10, 14), ("proc", 8, 14), ("powr", 8, 16),
        ("weap", 14, 14),
    ]
    harvs = [{"cell_x": 14, "cell_y": 12, "type": "harv"}]
    assert evaluate(
        c.win_condition,
        _ctx(own_buildings=full, units=harvs, tick=4000, cash=1200),
    )
    assert evaluate(
        c.fail_condition,
        _ctx(own_buildings=full, units=harvs, tick=4501, cash=1200),
    )


# ── engine-driven scripted policies ─────────────────────────────────


def _find_units(rs, type_name):
    return [
        u for u in (rs.get("units_summary", []) or [])
        if str(u.get("type", "")).lower() == type_name
    ]


def _base_xy(rs):
    facts = [
        b for b in (rs.get("own_buildings", []) or [])
        if str(b.get("type", "")).lower() == "fact"
    ]
    if not facts:
        return None
    return min(facts, key=lambda b: b["cell_x"])


def _stall(rs, Command):
    """Idle every turn — no harvest, no build, no defence orders.
    Defenders auto-engage but harvs never harvest (no income) and
    weap is never built ⇒ has_building:weap never latches ⇒ timeout
    LOSS via after_ticks."""
    return [Command.observe()]


def _focus_defence_skip_tech(rs, Command):
    """Hold every defender at the lane mouth; KICK harvs on the patch
    (income flows); but NEVER queue weap. Weap is required by the win
    predicate, so this LOSES on the clock even though the base may
    survive."""
    cmds = []
    base = _base_xy(rs)
    bx = base["cell_x"] if base else 10
    by = base["cell_y"] if base else 18
    harvs = _find_units(rs, "harv")
    patch_xs = [24, 24, 24, 24]
    patch_ys = [by - 2, by + 2, by - 4, by + 4]
    for i, h in enumerate(harvs):
        mx = patch_xs[i % len(patch_xs)]
        my = patch_ys[i % len(patch_ys)]
        cmds.append(Command.harvest([str(h["id"])], mx, my))
    for u in (rs.get("units_summary", []) or []):
        if str(u.get("type", "")).lower() == "1tnk":
            cmds.append(
                Command.move_units([str(u["id"])], target_x=bx + 4, target_y=by)
            )
    if not cmds:
        cmds.append(Command.observe())
    return cmds


def _focus_tech_lose_econ_policy():
    """Queue weap from turn 1 (using the starting cash) BUT pull every
    defender AND harvester to the base — the patch is abandoned, so
    raiders eat the harvs (unit_type_count_gte:{harv,1} ⇒ fail). Even
    if weap finishes, the harv-count fail clause kills the run."""
    state = {"weap_queued": False, "weap_attempts": 0}

    def pol(rs, Command):
        ob = rs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = rs.get("production", []) or []
        cash = int(rs.get("cash", 0) or 0)
        base = _base_xy(rs)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 18
        cmds = []
        if (
            "weap" not in own_b
            and "weap" not in prod
            and cash >= 2000
        ):
            cmds.append(Command.build("weap"))
            state["weap_queued"] = True
        if "weap" not in own_b:
            i = state["weap_attempts"]
            cmds.append(
                Command.place_building("weap", bx + 6 + (i % 4), by - 2 + (i // 4))
            )
            state["weap_attempts"] += 1
        for u in (rs.get("units_summary", []) or []):
            cmds.append(
                Command.move_units([str(u["id"])], target_x=bx, target_y=by)
            )
        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return pol


def _intended_multi_handle_policy():
    """The intended REASONING capability play: handle all three
    pressures in parallel from turn 1.

       PRESSURE A (economy/raid): keep both harvesters in `harvest`
         mode every turn — they cycle between proc and the local
         mine; income accrues to top up the tech buy.
       PRESSURE B (defence): ACTIVELY command the tank ring. Each
         turn the squad attack-moves onto the nearest visible enemy
         (raider tank or flanking hunter) so it concentrates fire
         and kills the raiders before a harvester falls. Passive
         Defend stance is NOT enough — the two raiders out-trade an
         un-commanded ring and a harv dies (the harv-count fail
         clause then bites).
       PRESSURE C (tech): queue `weap` on turn 1 with the starting
         cash, place it immediately in a safe slot west of the base.
    """
    import math

    state = {"weap_queued": False, "weap_attempts": 0}

    def pol(rs, Command):
        ob = rs.get("own_buildings", []) or []
        own_b = {b["type"] for b in ob}
        prod = rs.get("production", []) or []
        cash = int(rs.get("cash", 0) or 0)
        base = _base_xy(rs)
        bx = base["cell_x"] if base else 10
        by = base["cell_y"] if base else 18
        cmds = []

        # PRESSURE C — tech.
        if (
            "weap" not in own_b
            and "weap" not in prod
            and cash >= 2000
        ):
            cmds.append(Command.build("weap"))
            state["weap_queued"] = True
        if "weap" not in own_b:
            i = state["weap_attempts"]
            cmds.append(
                Command.place_building(
                    "weap", bx + 4 + (i % 4), by + (i // 4)
                )
            )
            state["weap_attempts"] += 1

        # PRESSURE A — keep both harvesters working the patch.
        harvs = _find_units(rs, "harv")
        patch_xs = [24, 24, 24, 24]
        patch_ys = [by - 2, by + 2, by - 4, by + 4]
        for i, h in enumerate(harvs):
            mx = patch_xs[i % len(patch_xs)]
            my = patch_ys[i % len(patch_ys)]
            cmds.append(Command.harvest([str(h["id"])], mx, my))

        # PRESSURE B — actively command the tank ring onto the
        # nearest visible enemy (raider or hunter).
        raw = rs.get("_raw", {}) or {}
        ep = raw.get("enemy_positions") or []
        tnks = _find_units(rs, "1tnk")
        if ep and tnks:
            pts = [
                (int(e.get("cell_x", 0)), int(e.get("cell_y", 0)))
                for e in ep
                if isinstance(e, dict)
            ]
            if pts:
                tx, ty = min(
                    pts, key=lambda p: math.hypot(p[0] - bx, p[1] - by)
                )
                cmds.append(
                    Command.attack_move(
                        [str(t["id"]) for t in tnks], tx, ty
                    )
                )

        if not cmds:
            cmds.append(Command.observe())
        return cmds

    return pol


# ── Engine-bound tests (parameterised over seeds 1..4) ───────────────


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_intended_multi_handle_wins(level, seed):
    """Intended multi-handle must WIN on every (level, seed)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _intended_multi_handle_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "win", (
        f"intended multi-handle must WIN on {level} s={seed}; got "
        f"{res.outcome} turns={res.turns} own_buildings={own_b} "
        f"units_lost={res.signals.units_lost} cash={res.signals.cash}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_stall_loses(level, seed):
    """Stall must be a real timeout LOSS on every (level, seed)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _stall, seed=seed)
    assert res.outcome == "loss", (
        f"stall must LOSS on {level} s={seed}; got {res.outcome} "
        f"own_buildings={res.signals.own_building_types}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_focus_defence_skip_tech_loses(level, seed):
    """Focus on defence and never queue weap ⇒ has_building:weap
    never latches ⇒ timeout LOSS on every (level, seed)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _focus_defence_skip_tech, seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"focus-defence-skip-tech must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("level", LEVELS)
def test_focus_tech_lose_econ_loses(level, seed):
    """Focus on tech (queue weap, pull everything home) ⇒ harvs
    abandoned ⇒ fail by harv-count (or by clock). Must LOSE."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, _focus_tech_lose_econ_policy(), seed=seed)
    own_b = res.signals.own_building_types
    assert res.outcome == "loss", (
        f"focus-tech-lose-econ must LOSE on {level} s={seed}; got "
        f"{res.outcome} own_buildings={own_b} "
        f"units_lost={res.signals.units_lost}"
    )
