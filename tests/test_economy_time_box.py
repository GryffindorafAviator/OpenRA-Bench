"""Full contributor-loop validation for economy-time-box (defect fix).

Originally quarantined as redundant with `economy-force-buildup` AND
suffering from two structural defects per the family-2 audit:
  1. `within_ticks: 22000/16000/12000` exceeded the engine ceiling at
     max_turns 60/70/80 (only ~5403/6303/7203 ticks reachable) so the
     deadline NEVER bit — stall → DRAW, not LOSS.
  2. NO `fail_condition` on any tier → non-win silently DRAW.

Fix:
- `within_ticks` re-tuned to 5000/5500/6500 (well inside engine ceiling)
- `fail_condition` adds `after_ticks: within_ticks+1` (one-tick-past
  deadline backstop) PLUS `not has_building:fact` (force razed → LOSS).
- Un-quarantined with a re-tightened bar that discriminates from
  `economy-force-buildup` (which tests POWR + e3 QUALITY axis): this
  pack tests SPEND BREADTH — multiple building categories AND a fielded
  force on a tighter custom 48x40 arena.

Bar:
- stall LOSES every tier (deadline now bites — no DRAW degeneracy).
- units-only spam LOSES every tier (building_total stays at 3 < 4).
- buildings-only spend LOSES every tier (own_units stays at 0 < N).
- the intended balanced spend (1+ extra building category + N units)
  WINS every tier.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK_PATH = PACKS_DIR / "economy-time-box.yaml"


# ─── unit-level predicate checks (synthesized WinContext) ────────────


def _ctx(*, tick=1000, units=0, buildings=0, lost=0, has_fact=True):
    """Synthesize a WinContext for predicate-level checks.

    units → reflected in units_summary length AND own_units_count
    buildings → building_total_gte counts entries in `own_buildings`
    has_fact → ensures `has_building:fact` keeps firing
    """
    import types

    own_building_types = {"fact", "powr", "tent"} if has_fact else {"powr", "tent"}
    own_buildings = [{"type": t} for t in own_building_types] + [
        {"type": "powr"} for _ in range(max(0, buildings - len(own_building_types)))
    ]

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        own_buildings=own_buildings,
        own_building_types=own_building_types,
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"id": i, "cell_x": 12 + i, "cell_y": 18, "type": "e1"}
                for i in range(units)
            ],
            "own_buildings": own_buildings,
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 4 units + 4 buildings, in time → WIN
    assert evaluate(c.win_condition, _ctx(tick=3000, units=4, buildings=4))
    # 3 units only → fail (need ≥4)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=3, buildings=4))
    # 3 buildings only → fail (need ≥4 total)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=4, buildings=3))
    # Past deadline → fail-clause fires
    assert evaluate(c.fail_condition, _ctx(tick=5002, units=4, buildings=4))
    # Fact razed → fail-clause fires regardless of cash/units
    assert evaluate(c.fail_condition, _ctx(tick=3000, units=4, buildings=4, has_fact=False))


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 5 units + 4 buildings → WIN
    assert evaluate(c.win_condition, _ctx(tick=3000, units=5, buildings=4))
    # 4 units only → fail (need ≥5)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=4, buildings=4))
    # 3 buildings only → fail
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=5, buildings=3))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=5502, units=5, buildings=4))


def test_predicates_hard():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 6 units + 5 buildings + 0 losses → WIN
    assert evaluate(c.win_condition, _ctx(tick=3000, units=6, buildings=5, lost=0))
    # 1 loss → fail (units_lost_lte: 0)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=6, buildings=5, lost=1))
    # 5 units only → fail (need ≥6)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=5, buildings=5, lost=0))
    # 4 buildings only → fail (need ≥5)
    assert not evaluate(c.win_condition, _ctx(tick=3000, units=6, buildings=4, lost=0))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=6502, units=6, buildings=5, lost=0))
    # Any unit lost → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=3000, units=6, buildings=5, lost=1))


# ─── structural / metadata contract ───────────────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.status == "active", "defect fix must un-quarantine the pack"
    assert pack.meta.id == "economy-time-box"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None, (
            f"{lvl}: must have BOTH win and fail conditions (defect fix)"
        )


def test_timeout_loss_is_reachable_on_every_level():
    """Defect fix bar: `within_ticks` AND the fail-clause `after_ticks`
    must sit AT-OR-BELOW the engine ceiling 93 + 90*(max_turns - 1) on
    every level so the deadline actually bites (no DRAW degeneracy)."""
    pack = load_pack(PACK_PATH)
    for lvl in ("easy", "medium", "hard"):
        L = pack.levels[lvl]
        ceiling = 93 + 90 * (L.max_turns - 1)
        win_d = L.win_condition.model_dump()
        fail_d = L.fail_condition.model_dump()
        wt = next(
            int(c["within_ticks"])
            for c in win_d["all_of"]
            if "within_ticks" in c
        )
        ft = next(
            int(c["after_ticks"])
            for c in fail_d["any_of"]
            if "after_ticks" in c
        )
        assert wt <= ceiling, (
            f"{lvl}: within_ticks {wt} > engine ceiling {ceiling} — "
            "deadline never bites (DEFECT)"
        )
        assert ft <= ceiling, (
            f"{lvl}: after_ticks {ft} > engine ceiling {ceiling}"
        )
        # one-tick-past-deadline idiom (non-finisher LOSES)
        assert ft == wt + 1, (
            f"{lvl}: after_ticks {ft} should be exactly within_ticks+1 ({wt + 1})"
        )


# ─── engine-driven scripted policies ──────────────────────────────────


def stall_policy(rs, Command):
    """Stall: only observe. Defect fix gate — with the deadline
    re-aligned and fail_condition present, stall MUST be a real LOSS."""
    return [Command.observe()]


def units_only_factory(unit_type="e1"):
    """Spend the whole budget on infantry only. Building total stays
    at 3 (the pre-placed fact/powr/tent), under the bar (≥4)."""
    def f(rs, Command):
        if rs.get("cash", 0) >= 100:
            return [Command.build(unit_type)]
        return [Command.observe()]
    return f


def buildings_only_factory():
    """Spend the budget on buildings only. own_units stays at 0,
    under any tier's bar."""
    state = {"queued": [], "placed_at": [(20, 18), (20, 22), (24, 18), (24, 22)]}

    def f(rs, Command):
        cash = rs.get("cash", 0)
        cmds = []
        # Queue serial: powr (300), silo (150 — needs proc, skip), proc (1400)
        # In order: 2nd powr, then proc.
        if "powr" not in state["queued"] and cash >= 300:
            cmds.append(Command.build("powr"))
            state["queued"].append("powr")
        elif "proc" not in state["queued"] and cash >= 1400:
            cmds.append(Command.build("proc"))
            state["queued"].append("proc")
        return cmds or [Command.observe()]
    return f


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """Defect fix gate: with `within_ticks` re-aligned and the
    `fail_condition` `after_ticks` backstop present, stall MUST be a
    real timeout LOSS — never a DRAW."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, stall_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level} stall must LOSE (defect fix): got {res.outcome} "
        f"tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_units_only_spam_loses(level):
    """Units-only spam (no new buildings) LOSES on `building_total_gte`."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, units_only_factory("e1"), seed=1)
    assert res.outcome == "loss", (
        f"{level} units-only spam must LOSE (no new buildings); got "
        f"{res.outcome}"
    )


@pytest.mark.parametrize("level", ["easy", "medium"])
def test_buildings_only_loses(level):
    """Buildings-only spend (no units trained) LOSES on `own_units_gte`.
    (Hard is omitted because the buildings_only path may also fail the
    `units_lost_lte:0` clause harmlessly — the bar still bites either
    way; this test isolates the unit-count failure mode on easy/med.)"""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, buildings_only_factory(), seed=1)
    assert res.outcome == "loss", (
        f"{level} buildings-only must LOSE (no units trained); got "
        f"{res.outcome}"
    )
