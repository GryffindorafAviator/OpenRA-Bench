"""combat-focus-fire-priority — focus-fire the high-DPS target FIRST.

The bar: intended focus-e3-first WINS on every level and every hard
seed (1-4); STALL, BRUTE attack_move, and SPREAD-attack-e1-first all
LOSE on every level and every hard seed. Every enemy squad is a STATIC
stance:2 (Defend) shield-wall-plus-rocket-rear formation, so the
focus-fire priority is a real decision rather than a melee. Non-win is
a real LOSS: stall via the `after_ticks` timeout fail clause, brute /
spread via the `units_lost_lte` attrition cap.

Validation is scripted (no model / network).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
PACK_PATH = PACKS / "combat-focus-fire-priority.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-focus-fire-priority"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "microrts", "strike-package", "rps-counter"):
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
        own_buildings=[],
        own_building_types=set(),
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={"units_summary": list(units)},
    )


def _alive(n):
    return [
        {"cell_x": 10, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)}
        for i in range(n)
    ]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: kills 7 (whole squad), ≤1 tank lost, in time → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(4), tick=2000, kills=7, lost=1))
    # Kill bar unmet (only 6 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(3), tick=2000, kills=6, lost=0))
    # Attrition cap busted (2 lost > 1) → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(2), tick=2000, kills=3, lost=2))
    # All tanks dead → fail (own_units_gte:1 trips via fail clause)
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=3, lost=4))
    # Timeout with bar unmet → fail (after_ticks 2701)
    assert evaluate(c.fail_condition, _ctx(units=_alive(4), tick=2702, kills=6, lost=0))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: kills 11, 0 lost → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(4), tick=2000, kills=11, lost=0))
    # Bar unmet (only 10 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(4), tick=2000, kills=10, lost=0))
    # Any tank lost (cap is 0) → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=2000, kills=11, lost=1))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=11, lost=4))
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(4), tick=2702, kills=10, lost=0))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: kills 18, 0 lost → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(4), tick=2000, kills=18, lost=0))
    # Bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(4), tick=2000, kills=17, lost=0))
    # Any tank lost (cap is 0) → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(3), tick=2000, kills=18, lost=1))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2000, kills=18, lost=4))
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(4), tick=2702, kills=17, lost=0))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 2701 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 2701 <= max_tick, (
            f"{lvl}: after_ticks 2701 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )
        assert 2700 <= max_tick, (
            f"{lvl}: within_ticks 2700 > max reachable tick {max_tick}"
        )


def test_hard_has_two_spawn_point_groups():
    """Hard-tier curation: ≥2 distinct agent spawn_point groups so the
    seed round-robins the west-edge corridor (north / south). Engine-
    roundtrip is asserted by tests/test_hard_tier.py."""
    c = compile_level(load_pack(PACK_PATH), "hard")
    groups = {
        (a.spawn_point if a.spawn_point is not None else 0)
        for a in c.scenario.actors
        if a.owner == "agent"
    }
    assert len(groups) >= 2, f"hard needs ≥2 spawn_point groups, got {groups}"


def test_enemy_squad_has_rocket_soldier_and_rifle_infantry():
    """The priority threat (e3 = anti-tank rocket soldier) must be
    present on every level alongside rifle infantry (e1) escorts;
    that contrast is what makes target prioritization the
    load-bearing capability of the pack."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        n_e3 = sum(1 for t in types if t == "e3")
        n_e1 = sum(1 for t in types if t == "e1")
        assert n_e3 >= 1, f"{lvl}: must include ≥1 e3 (priority threat); got {types}"
        assert n_e1 >= 2, f"{lvl}: must include ≥2 e1 (escort); got {types}"
        # Persistent far enemy marker (engine auto-done mitigation).
        assert "fact" in types, f"{lvl}: needs a persistent enemy fact"


# ── engine-driven scripted policies ─────────────────────────────────


def _own_ids(rs):
    return [str(u["id"]) for u in (rs.get("units_summary", []) or [])]


def _enemies_of_type(rs, want_types):
    out = []
    for e in (rs.get("enemy_summary") or []):
        t = (e.get("type") or e.get("actor_type") or "").lower()
        if t in want_types:
            out.append(e)
    return out


def _stall(rs, Command):
    """Pure observe — the enemy squad is static (stance:2 Defend, never
    advances), the tanks never engage → kill bar unmet → after_ticks
    LOSS."""
    return [Command.observe()]


def _brute_attack_move(rs, Command):
    """Brute: every tank attack_moves toward the enemy cluster centre.
    The engine auto-targets the NEAREST visible hostile, which spreads
    fire across the front-line e1s. The rocket-soldier e3 stays alive
    long enough to bust the attrition cap → LOSS on medium/hard."""
    cmds = []
    for uid in _own_ids(rs):
        cmds.append(Command.attack_move([uid], 71, 20))
    return cmds or [Command.observe()]


def _spread_attack_e1_first(rs, Command):
    """Explicitly target an e1 first (priority inversion). Each turn
    re-target the closest visible e1; only fall through to the e3 if
    no e1s remain. The e3 keeps firing for the entire e1 mop-up and
    kills tanks → busts attrition cap → LOSS."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    e1s = _enemies_of_type(rs, {"e1"})
    if e1s:
        rxs = [u["cell_x"] for u in rs.get("units_summary", [])]
        rys = [u["cell_y"] for u in rs.get("units_summary", [])]
        cx = sum(rxs) / len(rxs)
        cy = sum(rys) / len(rys)
        e1s.sort(key=lambda e: (e["cell_x"] - cx) ** 2 + (e["cell_y"] - cy) ** 2)
        tid = e1s[0].get("id")
        if tid is not None:
            return [Command.attack_unit(own, str(tid))]
    e3s = _enemies_of_type(rs, {"e3"})
    if e3s:
        tid = e3s[0].get("id")
        if tid is not None:
            return [Command.attack_unit(own, str(tid))]
    # Nothing in sight — advance to contact.
    return [Command.attack_move([uid], 71, 20) for uid in own]


def _intended_focus_e3(rs, Command):
    """Focus-fire the visible rocket soldier (e3) FIRST with ALL tanks
    (4-vs-1 concentrated fire kills it in 1-2 decision ticks); once
    all e3s are down, mop up the rifle infantry. This is the focus-
    fire doctrine — kill the high-DPS threat before it can score
    tank kills."""
    own = _own_ids(rs)
    if not own:
        return [Command.observe()]
    e3s = _enemies_of_type(rs, {"e3"})
    if e3s:
        rxs = [u["cell_x"] for u in rs.get("units_summary", [])]
        rys = [u["cell_y"] for u in rs.get("units_summary", [])]
        cx = sum(rxs) / len(rxs)
        cy = sum(rys) / len(rys)
        e3s.sort(key=lambda e: (e["cell_x"] - cx) ** 2 + (e["cell_y"] - cy) ** 2)
        tid = e3s[0].get("id")
        if tid is not None:
            return [Command.attack_unit(own, str(tid))]
    e1s = _enemies_of_type(rs, {"e1"})
    if e1s:
        tid = e1s[0].get("id")
        if tid is not None:
            return [Command.attack_unit(own, str(tid))]
    # No targets in sight — advance east toward the cluster centre.
    return [Command.attack_move([uid], 71, 20) for uid in own]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_focus_e3_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended_focus_e3, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended focus-e3-first should WIN, "
        f"got {r.outcome} after {r.turns} turns "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
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
        f"(no engagement → kill bar unmet), got {r.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_brute_attack_move_loses(level, seed):
    """Brute attack_move (auto-target the nearest hostile) must LOSE
    on every level — the engine spreads fire across the e1 shield
    line and the static rocket rear-rank picks off enough tanks to
    bust the attrition cap before the kill bar is met."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _brute_attack_move, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: brute attack_move must LOSE (e3 picks "
        f"off tanks via spread fire), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_spread_attack_e1_first_loses(level, seed):
    """Spread / priority-inverted: attack the rifle infantry first
    (the close, soft targets). The static rocket soldiers keep firing
    through the entire e1 mop-up and kill enough tanks to bust the
    attrition cap before the kill bar is met → LOSS on every level."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _spread_attack_e1_first, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: e1-first must LOSE (e3 alive too long "
        f"→ tank kills), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )
