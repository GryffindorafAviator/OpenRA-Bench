"""Full contributor-loop validation for economy-force-buildup.

The pack tests fixed-budget POWER + e3 QUALITY allocation: the budget
buys exactly one coherent build — a structural investment (2nd powr,
300cr) PLUS a force of capable units (anti-vehicle infantry e3, 300cr
each — not the cheapest filler e1 100cr). Spamming e1 clears no bar;
over-spending misses the deadline.

Bar (per CLAUDE.md "no defect, no cheat"):
- stall LOSES every tier (no spend → 0 buildings/units past 3).
- e1-spam (cheap rifle only) LOSES every tier (e3-quality bar unmet).
- intended balanced spend (2nd powr + N e3) WINS every tier.
- hard tier adds units_lost_lte:0 — any loss LOSES.
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK_PATH = PACKS_DIR / "economy-force-buildup.yaml"


# ─── unit-level predicate checks ──────────────────────────────────────


def _ctx(*, tick=1000, e3_count=0, powr_count=1, lost=0):
    """Synthesize a WinContext for predicate-level checks. `own_buildings`
    is a list of (type, cell_x, cell_y) tuples per win_conditions.py."""
    import types

    own_building_types = {"fact", "powr", "tent"}
    own_buildings = [
        ("fact", 10, 18),
        ("tent", 10, 22),
    ] + [("powr", 14 + 2 * i, 18) for i in range(powr_count)]

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=lost,
        own_buildings=own_buildings,
        own_building_types=own_building_types,
        enemies_seen_ids=set(),
        enemy_buildings_seen_ids=set(),
        power_provided=200,
        power_drained=100,
    )
    return WinContext(
        signals=sig,
        render_state={
            "units_summary": [
                {"id": i, "cell_x": 12 + i, "cell_y": 18, "type": "e3"}
                for i in range(e3_count)
            ],
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 4 e3 + 2 powr → WIN
    assert evaluate(c.win_condition, _ctx(tick=2000, e3_count=4, powr_count=2))
    # 3 e3 only → fail (need ≥4)
    assert not evaluate(c.win_condition, _ctx(tick=2000, e3_count=3, powr_count=2))
    # Only 1 powr → fail (need 2)
    assert not evaluate(c.win_condition, _ctx(tick=2000, e3_count=4, powr_count=1))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=2704, e3_count=4, powr_count=2))


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    assert evaluate(c.win_condition, _ctx(tick=2000, e3_count=5, powr_count=2))
    # 4 e3 → fail (need ≥5)
    assert not evaluate(c.win_condition, _ctx(tick=2000, e3_count=4, powr_count=2))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=2524, e3_count=5, powr_count=2))


def test_predicates_hard():
    c = compile_level(load_pack(PACK_PATH), "hard")
    assert evaluate(c.win_condition, _ctx(tick=2000, e3_count=6, powr_count=2, lost=0))
    # 1 loss → fail (units_lost_lte: 0)
    assert not evaluate(c.win_condition, _ctx(tick=2000, e3_count=6, powr_count=2, lost=1))
    # 5 e3 → fail (need ≥6)
    assert not evaluate(c.win_condition, _ctx(tick=2000, e3_count=5, powr_count=2, lost=0))
    # Past deadline OR any loss → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=2164, e3_count=6, powr_count=2, lost=0))
    assert evaluate(c.fail_condition, _ctx(tick=1000, e3_count=6, powr_count=2, lost=1))


# ─── structural ───────────────────────────────────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "economy-force-buildup"
    assert pack.meta.capability == "reasoning"
    anchors = pack.meta.benchmark_anchor
    assert isinstance(anchors, list) and anchors, "benchmark_anchor required"
    for lvl in ("easy", "medium", "hard"):
        c = compile_level(pack, lvl)
        assert c.map_supported
        assert c.win_condition is not None and c.fail_condition is not None


def test_timeout_loss_is_reachable_on_every_level():
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
        # Easy/medium fail is a bare after_ticks dict; hard is any_of.
        if "any_of" in fail_d:
            ft = next(
                int(c["after_ticks"])
                for c in fail_d["any_of"]
                if "after_ticks" in c
            )
        else:
            ft = int(fail_d["after_ticks"])
        assert wt <= ceiling, f"{lvl}: within_ticks {wt} > ceiling {ceiling}"
        assert ft <= ceiling, f"{lvl}: after_ticks {ft} > ceiling {ceiling}"


# ─── engine-driven scripted policies ──────────────────────────────────


def stall_policy(rs, Command):
    """Pure idle — must LOSE every tier (no e3 produced, no 2nd powr)."""
    return [Command.observe()]


def e1_spam_factory():
    """Spam the cheap rifle (e1, 100cr). Even if cash is fully converted
    to e1, the win predicate requires E3 quality (`unit_type_count_gte:
    {type:e3,n:N}`) and a 2ND POWR — neither bar clears. LOSS."""
    def f(rs, Command):
        if rs.get("cash", 0) >= 100:
            return [Command.build("e1")]
        return [Command.observe()]
    return f


def intended_balanced_factory(n_e3):
    """Intended: queue a 2nd powr immediately, place it, then train N e3.
    Defense and infantry are SEPARATE production queues per CLAUDE.md
    so build('powr') and build('e3') can be queued in parallel."""
    state = {"powr_queued": False, "powr_placed": False, "e3_queued": 0}

    def f(rs, Command):
        cmds = []
        # Phase 1: queue and place 2nd powr (top priority — power
        # supports any further build).
        if not state["powr_queued"] and rs.get("cash", 0) >= 300:
            cmds.append(Command.build("powr"))
            state["powr_queued"] = True
        # Place any ready building (the engine returns ready buildings
        # via `place_building` once the build completes).
        if not state["powr_placed"]:
            ready = rs.get("ready_to_place", []) or []
            for r in ready:
                if r.get("type") == "powr":
                    cmds.append(
                        Command.place_building("powr", target_x=14, target_y=22)
                    )
                    state["powr_placed"] = True
                    break
        # Phase 2: queue e3 (infantry queue is independent of building
        # queue — fires in parallel).
        if state["e3_queued"] < n_e3 and rs.get("cash", 0) >= 300:
            cmds.append(Command.build("e3"))
            state["e3_queued"] += 1
        return cmds or [Command.observe()]
    return f


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """Stall must LOSE — no 2nd powr, no e3 → predicate unsatisfied;
    after_ticks deadline bites (reachable inside max_turns)."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, stall_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level} stall must LOSE; got {res.outcome} tick={res.signals.game_tick}"
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_e1_spam_loses_on_every_level(level):
    """Cheap-rifle spam must LOSE: even with cash fully converted,
    the e3 quality bar (unit_type_count_gte) and 2nd powr bar are unmet."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, e1_spam_factory(), seed=1)
    assert res.outcome == "loss", (
        f"{level} e1-spam must LOSE (e3 quality bar unmet); got "
        f"{res.outcome}"
    )
