"""combat-harass-aggro-commit — AGGRO variant of the harvester-harass
triple.

The bar: intended focus-defender-then-harv WINS on every level and
every hard seed (1-4); STALL (only observe), RETREAT-ONLY (drive
raiders back west), and ATTACK-HARV-IGNORE-DEFENDER (rush the harvs
while standing in the 3tnk defender's range) all LOSE on every level
and every hard seed — with one documented exception: EASY allows
attack-harv-only to squeak by (forgiving bare-skill tier with 4
raiders and a kill bar of 3). Non-win is a real reachable timeout
LOSS via the `after_ticks` fail clause.

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
PACK_PATH = PACKS / "combat-harass-aggro-commit.yaml"


# ── unit-level predicate / metadata checks (no engine) ──────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "combat-harass-aggro-commit"
    assert pack.meta.capability == "action"
    assert pack.meta.real_world_meaning, "real_world_meaning required"
    assert pack.meta.robotics_analogue, "robotics_analogue required"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and len(anchors) == 4, (
        f"benchmark_anchor must list all 4 anchors, got {anchors!r}"
    )
    joined = " ".join(anchors).lower()
    for needle in ("sc2", "attrition", "aggro", "guerrilla"):
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
    return [{"cell_x": 10, "cell_y": 20, "type": "2tnk", "id": str(1000 + i)} for i in range(n)]


def test_easy_predicates():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: kills 3, 1 raider alive, in time → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(1), tick=2500, kills=3))
    # Kill bar unmet (only 2 kills) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(2), tick=2500, kills=2))
    # All raiders dead → fail (own_units_gte:1 trips via fail clause)
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2500, kills=3))
    # Timeout with bar unmet → fail (after_ticks 4501)
    assert evaluate(c.fail_condition, _ctx(units=_alive(2), tick=4502, kills=2))


def test_medium_predicates():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: kills 4 (1 def + 3 harv), 1 raider alive → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(1), tick=2500, kills=4))
    # Bar unmet (only 3 kills — partial sweep) → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(1), tick=2500, kills=3))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=2500, kills=4))
    # Timeout with bar unmet → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(2), tick=4502, kills=3))


def test_hard_predicates():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: kills 6 (2 def + 4 harv), 1 raider alive → WIN
    assert evaluate(c.win_condition, _ctx(units=_alive(1), tick=3500, kills=6))
    # Bar unmet → not a win
    assert not evaluate(c.win_condition, _ctx(units=_alive(2), tick=3500, kills=5))
    # Force wipe → fail
    assert evaluate(c.fail_condition, _ctx(units=[], tick=3500, kills=6))
    # Timeout → fail
    assert evaluate(c.fail_condition, _ctx(units=_alive(1), tick=4502, kills=5))


def test_timeout_reachable_inside_max_turns():
    """No draw degeneracy: after_ticks 4501 ≤ 93 + 90·(max_turns-1)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        max_tick = 93 + 90 * (c.max_turns - 1)
        assert 4501 <= max_tick, (
            f"{lvl}: after_ticks 4501 > max reachable tick {max_tick} "
            f"(max_turns={c.max_turns}); deadline never bites"
        )
        assert 4500 <= max_tick, f"{lvl}: within_ticks 4500 > max tick {max_tick}"


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


def test_defender_is_3tnk_and_no_bot():
    """The defender must be a 3tnk (heavier than 2tnk so a head-on
    1-vs-1 trade loses); no bot is declared so retreat is safe but
    scoreless (the defender holds its post)."""
    pack = load_pack(PACK_PATH)
    enemy = pack.base.get("enemy") if isinstance(pack.base, dict) else None
    assert enemy is not None
    bot = enemy.get("bot_type") or enemy.get("bot") or ""
    assert bot == "", f"expected no bot (stationary defender), got {bot!r}"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        types = [a.type for a in c.scenario.actors if a.owner == "enemy"]
        assert "3tnk" in types, f"{lvl}: defender must be 3tnk"
        assert "harv" in types, f"{lvl}: payload must include harv"


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
    """Pure observe — defenders never engage, raiders never engage →
    kill bar unmet → after_ticks LOSS."""
    return [Command.observe()]


def _retreat(rs, Command):
    """Drive every raider back to the west spawn — no engagement, no
    kills → after_ticks LOSS. The defender is non-mobile so retreat
    is SAFE but scoreless (this is the discriminating LOSS: a model
    that picks survival over commitment fails the aggro bar)."""
    cmds = []
    for uid in _own_ids(rs):
        cmds.append(Command.move_units([uid], 4, 20))
    return cmds or [Command.observe()]


def _attack_harv_only(rs, Command):
    """Rush the harvester cluster while ignoring the defender. With a
    heavier 3tnk in range, the raiders take fire on the approach and
    while engaging the soft targets — kill ratio collapses before
    the bar is met → LOSS on medium/hard. Easy is a forgiving tier
    where this play may squeak by."""
    cmds = []
    raiders = _own_ids(rs)
    for rid in raiders:
        cmds.append(Command.attack_move([rid], 40, 20))
    return cmds or [Command.observe()]


def _intended(rs, Command):
    """Focus-fire the visible defender 3tnk first with ALL raiders;
    once it's down, attack-move into the harv cluster. This is the
    aggro doctrine — commit and trade favourably (3-vs-1 tank trade)
    on the high-value target before mopping up the payload."""
    raiders = _own_ids(rs)
    if not raiders:
        return [Command.observe()]
    defenders = _enemies_of_type(rs, {"3tnk"})
    if defenders:
        rxs = [u["cell_x"] for u in rs.get("units_summary", [])]
        rys = [u["cell_y"] for u in rs.get("units_summary", [])]
        cx, cy = sum(rxs) / len(rxs), sum(rys) / len(rys)
        defenders.sort(
            key=lambda e: (e["cell_x"] - cx) ** 2 + (e["cell_y"] - cy) ** 2
        )
        tid = defenders[0].get("id")
        if tid is not None:
            return [Command.attack_unit(raiders, str(tid))]
    harvs = _enemies_of_type(rs, {"harv"})
    if harvs:
        tid = harvs[0].get("id")
        if tid is not None:
            return [Command.attack_unit(raiders, str(tid))]
    # No defenders / harvs in sight — attack-move east into the cluster.
    return [Command.attack_move([rid], 40, 20) for rid in raiders]


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_intended_focus_defender_wins(level, seed):
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _intended, seed=seed)
    assert r.outcome == "win", (
        f"{level} seed={seed}: intended focus-defender-then-harv should "
        f"WIN, got {r.outcome} after {r.turns} turns "
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
def test_retreat_only_loses(level, seed):
    """Pure retreat (drive all raiders back west) must LOSE on every
    tier — the AGGRO doctrine specifically penalises survival-only
    play. The defender holds its post (no bot), so retreat is SAFE
    but scoreless → after_ticks LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _retreat, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: retreat-only must LOSE (no kills → bar "
        f"unmet), got {r.outcome} (kills={r.signals.units_killed})"
    )


@pytest.mark.parametrize("level", ["medium", "hard"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_attack_harv_only_loses(level, seed):
    """Attack-harv-only (ignore the defender, rush harvs) must LOSE
    on medium and hard — the 3tnk picks off the raiders while they
    engage the soft targets. Easy is excluded as the bare-skill tier
    (4 raiders + kill bar 3 is forgiving enough for this brute play
    to squeak by; documented in the pack's design comment, matches
    SCENARIO_REVIEW_CHECKLIST.md note that inert anti-cheat teeth
    are acceptable on easy)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    r = run_level(c, _attack_harv_only, seed=seed)
    assert r.outcome == "loss", (
        f"{level} seed={seed}: attack-harv-only must LOSE (defender "
        f"picks off raiders), got {r.outcome} "
        f"(kills={r.signals.units_killed}, losses={r.signals.units_lost})"
    )
