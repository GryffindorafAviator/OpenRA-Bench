"""Full contributor-loop validation for economy-investment.

The pack tests indivisible capital allocation: the budget buys exactly
one coherent path — either WIDE (2nd proc + 2nd powr) or DEEP (whole
budget on infantry). A hedged spend funds neither and loses.

Per-tier path:
- easy: WIDE only — $1700 = $1400 proc + $300 powr (no units required).
- medium: WIDE + 3 utilising units — $2000.
- hard: DEEP — keep single proc, train 22 infantry. $2200.

Bar (per CLAUDE.md "no defect, no cheat"):
- stall LOSES every tier (after_ticks deadline reachable; non-finisher).
- "split-budget" hedged play LOSES on every tier (neither path completes).
- intended single-path WINS its tier.
- WIDE path LOSES hard tier (can't reach 22 units with budget consumed
  by structures).
"""

from __future__ import annotations

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level
from openra_bench.scenarios.win_conditions import WinContext, evaluate

PACK_PATH = PACKS_DIR / "economy-investment.yaml"


# ─── unit-level predicate checks ──────────────────────────────────────


def _ctx(*, tick=1000, units=0, proc_count=1, powr_count=1):
    """Synthesize a WinContext. `own_buildings` is a list of
    (type, cell_x, cell_y) tuples per win_conditions.py."""
    import types

    own_building_types = {"fact", "powr", "tent", "proc"}
    own_buildings = (
        [("fact", 10, 18), ("tent", 14, 22)]
        + [("proc", 10 + 2 * i, 22) for i in range(proc_count)]
        + [("powr", 14 + 2 * i, 18) for i in range(powr_count)]
    )

    sig = types.SimpleNamespace(
        game_tick=tick,
        units_killed=0,
        units_lost=0,
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
                {"id": i, "cell_x": 12 + (i % 8), "cell_y": 22, "type": "e1"}
                for i in range(units)
            ],
        },
    )


def test_predicates_easy():
    c = compile_level(load_pack(PACK_PATH), "easy")
    # Intended: 2 proc + 2 powr → WIN (units not required)
    assert evaluate(c.win_condition, _ctx(tick=1500, units=0, proc_count=2, powr_count=2))
    # 1 proc → fail
    assert not evaluate(c.win_condition, _ctx(tick=1500, proc_count=1, powr_count=2))
    # 1 powr → fail
    assert not evaluate(c.win_condition, _ctx(tick=1500, proc_count=2, powr_count=1))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=1894, proc_count=2, powr_count=2))


def test_predicates_medium():
    c = compile_level(load_pack(PACK_PATH), "medium")
    # Intended: 2 proc + 2 powr + 3 units → WIN
    assert evaluate(c.win_condition, _ctx(tick=2000, units=3, proc_count=2, powr_count=2))
    # 2 units → fail (need 3)
    assert not evaluate(c.win_condition, _ctx(tick=2000, units=2, proc_count=2, powr_count=2))
    # 1 proc → fail
    assert not evaluate(c.win_condition, _ctx(tick=2000, units=3, proc_count=1, powr_count=2))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=2074, units=3, proc_count=2, powr_count=2))


def test_predicates_hard():
    c = compile_level(load_pack(PACK_PATH), "hard")
    # Intended: 22 units + 1 proc → WIN
    assert evaluate(c.win_condition, _ctx(tick=2500, units=22, proc_count=1, powr_count=1))
    # 21 units → fail
    assert not evaluate(c.win_condition, _ctx(tick=2500, units=21, proc_count=1, powr_count=1))
    # Past deadline → LOSS
    assert evaluate(c.fail_condition, _ctx(tick=2614, units=22, proc_count=1, powr_count=1))


# ─── structural ───────────────────────────────────────────────────────


def test_pack_compiles_and_meta_fields_populated():
    pack = load_pack(PACK_PATH)
    assert pack.meta.id == "economy-investment"
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
        # fail is a bare {after_ticks: N} dict on this pack
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
    """No spend → only the pre-placed proc/powr count → LOSS every tier."""
    return [Command.observe()]


def wide_factory():
    """WIDE intended: queue powr, place; queue proc, place. Wins easy."""
    state = {"powr_queued": False, "powr_placed": False,
             "proc_queued": False, "proc_placed": False}

    def f(rs, Command):
        cash = rs.get("cash", 0)
        cmds = []
        if not state["powr_queued"] and cash >= 300:
            cmds.append(Command.build("powr"))
            state["powr_queued"] = True
        if not state["proc_queued"] and cash >= 1400:
            cmds.append(Command.build("proc"))
            state["proc_queued"] = True
        # Place any ready building (engine returns via ready_to_place).
        ready = rs.get("ready_to_place", []) or []
        for r in ready:
            if r.get("type") == "powr" and not state["powr_placed"]:
                cmds.append(Command.place_building("powr", target_x=18, target_y=18))
                state["powr_placed"] = True
            if r.get("type") == "proc" and not state["proc_placed"]:
                cmds.append(Command.place_building("proc", target_x=18, target_y=22))
                state["proc_placed"] = True
        return cmds or [Command.observe()]
    return f


def deep_factory():
    """DEEP intended: train infantry as long as cash allows. For hard
    (22 units), move them clear of the barracks once produced to avoid
    production stalls."""
    state = {"moves_issued": set()}

    def f(rs, Command):
        cmds = []
        if rs.get("cash", 0) >= 100:
            cmds.append(Command.build("e1"))
        # Scatter newly produced infantry clear of (14,22) tent exit.
        for u in rs.get("units_summary", []) or []:
            if u.get("type") == "e1":
                uid = str(u["id"])
                if uid in state["moves_issued"]:
                    continue
                ux, uy = u["cell_x"], u["cell_y"]
                if abs(ux - 14) <= 3 and abs(uy - 22) <= 3:
                    # Disperse east in a vertical fan.
                    target_y = (int(uid) % 11) + 12  # 12..22
                    cmds.append(
                        Command.move_units([uid], target_x=22 + (int(uid) % 4), target_y=target_y)
                    )
                    state["moves_issued"].add(uid)
        return cmds or [Command.observe()]
    return f


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_stall_loses_on_every_level(level):
    """Stall must LOSE — pre-placed proc/powr count is 1 each, below
    the easy/medium 2-each bar; hard requires 22 units, also unmet."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), level)
    res = run_level(c, stall_policy, seed=1)
    assert res.outcome == "loss", (
        f"{level} stall must LOSE; got {res.outcome} tick={res.signals.game_tick}"
    )


def test_wide_decoy_loses_hard():
    """WIDE (2nd proc + 2nd powr) consumes $1700 leaving only $500 for
    the 22-unit hard bar — at most ~5 e1 → LOSS."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "hard")
    res = run_level(c, wide_factory(), seed=1)
    assert res.outcome == "loss", (
        f"hard WIDE decoy must LOSE (own_units bar unmet); got "
        f"{res.outcome}"
    )


def test_deep_decoy_loses_easy():
    """DEEP (units only) doesn't add the 2nd proc/powr → fails easy's
    structural bar regardless of how many units are trained."""
    pytest.importorskip("openra_train")
    from openra_bench.eval_core import run_level

    c = compile_level(load_pack(PACK_PATH), "easy")
    res = run_level(c, deep_factory(), seed=1)
    assert res.outcome == "loss", (
        f"easy DEEP decoy must LOSE (proc/powr bar unmet); got "
        f"{res.outcome}"
    )
