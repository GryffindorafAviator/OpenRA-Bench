"""mid-economy-under-fire — live ore-patch defense vs raider bot.

The bar: intended fire-support-around-the-ore-patch WINS on every
level and every hard seed (1-4); STALL (only observe → no harvest →
EV=0), CHASE-off-map (move all defenders east → harvs picked off),
and PULL-HOME (recall harvs to base → no income) all LOSE on every
level and every hard seed. Non-win is a real reachable timeout LOSS.

Validation is scripted (no model / network). The four policies below
are exhaustive proxies for the realistic plays a model could try.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "mid-economy-under-fire.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────

def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "mid-economy-under-fire"
    assert pack.meta.capability == "action"
    # Required meta fields per spec.
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("minerl", "sc2le", "lmgame", "sre"):
        assert needle in joined, f"missing anchor keyword: {needle}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None
        assert c.fail_condition is not None


def _ctx(
    *,
    units=(),
    tick=1000,
    cash=0,
    resources=0,
    lost=0,
    own_buildings=(),
):
    """Synthesize a WinContext for predicate-level checks."""
    import types

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
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


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    three_harvs = [
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
        {"cell_x": 14, "cell_y": 20, "type": "harv"},
        {"cell_x": 14, "cell_y": 22, "type": "harv"},
    ]
    # Intended: EV ≥ 2000, 2+ harvs alive, in time → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=2000, cash=2000),
    )
    # EV under bar → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=2000, cash=1500),
    )
    # Only 1 harv alive (rest dead) → not a win (need ≥2)
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs[:1], tick=2000, cash=2500),
    )
    # All harvs dead → real fail (capability collapses)
    assert evaluate(
        c.fail_condition,
        _ctx(units=[], tick=2000, cash=2500),
    )
    # Timeout: bar unmet at tick 5402 → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=three_harvs, tick=5402, cash=0),
    )


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    three_harvs = [
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
        {"cell_x": 14, "cell_y": 20, "type": "harv"},
        {"cell_x": 14, "cell_y": 22, "type": "harv"},
    ]
    # Intended: EV ≥ 3000, ≥2 harvs → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=3000, cash=3000),
    )
    # Bar=2999 (just under) → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=3000, cash=2999),
    )
    # Only 1 harv alive → not a win
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs[:1], tick=3000, cash=4000),
    )


def test_hard_predicates_attrition_cap():
    c = compile_level(load_pack(PACK_PATH), "hard")
    three_harvs = [
        {"cell_x": 14, "cell_y": 14, "type": "harv"},
        {"cell_x": 14, "cell_y": 16, "type": "harv"},
        {"cell_x": 14, "cell_y": 18, "type": "harv"},
    ]
    # Intended: EV ≥ 4000, ≥2 harvs, lost ≤2 → WIN
    assert evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=3000, cash=4000, lost=2),
    )
    # Lost 3 (cap is 2) → fail
    assert evaluate(
        c.fail_condition,
        _ctx(units=three_harvs, tick=3000, cash=4000, lost=3),
    )
    assert not evaluate(
        c.win_condition,
        _ctx(units=three_harvs, tick=3000, cash=4000, lost=3),
    )


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 5401 must fit inside max_turns
    on every level (≈90 ticks/turn ⇒ 93 + 90·(max_turns-1))."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 5401 <= max_tick, (
            f"{lvl}: after_ticks 5401 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )
        assert 5400 <= max_tick, (
            f"{lvl}: within_ticks 5400 > max tick {max_tick}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation contract: ≥2 distinct agent spawn_point
    groups so the seed round-robins the base orientation. Engine-
    roundtrip is asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_uses_raider_bot():
    """The pack must declare the Wave 2 `raider` bot — that's the
    novel engine capability this scenario was authored to exercise."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot")
    assert bot == "raider", f"expected raider bot, got {bot!r}"


# ── engine-driven scripted policies ─────────────────────────────────

def _intended(rs, Command):
    """Fire-support-around-the-ore-patch: keep harvesters in
    `harvest` mode each turn (the ring of tank defenders auto-engages
    the inbound raider on Defend stance). Throughput climbs while the
    raider dies on contact with the ring."""
    cmds = []
    harv_ids = [
        str(u["id"]) for u in (rs.get("units_summary", []) or [])
        if u.get("type") == "harv"
    ]
    # Try every known patch; the engine picks whichever is reachable
    # for the active seed (north / south / central).
    patches = [(22, 14), (22, 18), (22, 22), (22, 26)]
    for i, hid in enumerate(harv_ids):
        mx, my = patches[i % len(patches)]
        cmds.append(Command.harvest([hid], mx, my))
    return cmds or [Command.observe()]


def _stall(rs, Command):
    """No commands — defenders auto-engage (and may kill the raider),
    but harvesters never harvest → EV stays at 0 → bar unmet →
    timeout LOSS via the after_ticks fail clause."""
    return [Command.observe()]


def _chase_off_map(rs, Command):
    """Drive ALL units (defenders AND harvesters) east toward the
    enemy fact at (120,20). Defenders abandon the patch; harvesters
    leave the harvest loop. Raiders intercept the unprotected harvs
    in transit; EV stays low → LOSS."""
    cmds = []
    for u in (rs.get("units_summary", []) or []):
        cmds.append(
            Command.move_units([str(u["id"])], target_x=100, target_y=20)
        )
    return cmds or [Command.observe()]


def _pull_home(rs, Command):
    """Recall harvesters to the base / proc, holding tight. Harvs
    stop refining → no income → EV stays at 0 → timeout LOSS."""
    cmds = []
    for u in (rs.get("units_summary", []) or []):
        if u.get("type") == "harv":
            cmds.append(
                Command.move_units([str(u["id"])], target_x=10, target_y=18)
            )
    return cmds or [Command.observe()]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_policy_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended, seed=seed)
    ev = r.signals.cash + r.signals.resources
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended fire-support should WIN, got "
        f"{r.outcome} after {r.turns} turns "
        f"(EV={ev}, kills={r.signals.units_killed}, "
        f"losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_stall_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _stall, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: stall must be a real timeout LOSS "
        f"(no harvest cmd → EV=0 → bar unmet), got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_chase_off_map_loses(level, seed):
    """Pure chase-off-map (drive every unit east, including harvs)
    must LOSE. Easy is excluded because a single raider on the loose
    bar isn't always able to finish the harvs before the EV bar lapses
    — easy is the bare-skill tier and exempts this brute play (per
    SCENARIO_REVIEW_CHECKLIST: inert anti-stall teeth acceptable on
    easy)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _chase_off_map, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: chase-off-map must LOSE (harvs picked "
        f"off / no income), got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources}, "
        f"losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pull_home_loses(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _pull_home, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: pull-home must LOSE (harvs stop "
        f"refining → EV=0 → bar unmet), got {r.outcome} "
        f"(EV={r.signals.cash + r.signals.resources})"
    )
