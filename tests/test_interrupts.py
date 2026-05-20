"""Step 4 (bench): interrupt-driven loop. When a scenario enables
interrupt signals, run_level advances with step_until_event so the
agent is re-prompted the moment an event fires; the interrupt is
recorded in the trace and playback. Back-compat: no signals → fixed
step, interrupt always None."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.eval_core import run_level
from openra_bench.playback import Playback
from openra_bench.scenarios.schema import ScenarioPack

_META = {
    "id": "interrupt-selftest",
    "title": "Interrupt Self Test",
    "capability": "reasoning",
    "real_world_meaning": "event-driven re-planning when capacity arrives",
    "robotics_analogue": "re-plan on subsystem-online interrupt",
    "author": "ci",
}
_BASE = {
    "agent": {"faction": "allies"},
    "enemy": {"faction": "soviet"},
    "tools": ["build", "move_units", "stop"],
    "actors": [
        {"type": "barr", "owner": "agent", "position": [10, 20]},
        {"type": "powr", "owner": "agent", "position": [14, 20]},
        {"type": "jeep", "owner": "agent", "position": [8, 18]},
        {"type": "e1", "owner": "enemy", "position": [114, 34], "stance": 2},
    ],
    "termination": {"max_ticks": 40000},
}


def _pack(interrupts: dict | None):
    base = dict(_BASE)
    if interrupts is not None:
        base = {**base, "interrupts": interrupts}
    lvl = {
        "description": "build a unit; run long enough to finish it",
        "win_condition": {"explored_pct_gte": 999},  # never → runs to max_turns
        "max_turns": 30,
    }
    return ScenarioPack(
        meta=_META, base_map="rush-hour-arena", starting_cash=4000,
        base=base, levels={"easy": lvl, "medium": lvl, "hard": lvl},
    )


def _builder(rs, Command):
    return [Command.build("e1")]


def test_production_complete_interrupt_recorded(tmp_path):
    c = _pack({"production_complete": True}).compile("easy", map_supported=True)
    pb = Playback(tmp_path, "interrupt:easy", 1)
    res = run_level(c, _builder, seed=1, playback=pb)

    fired = [t for t in res.trace if t.get("interrupt")]
    assert any(
        "production_complete" in (t["interrupt"] or "") for t in fired
    ), f"expected a production_complete interrupt; got {[t.get('interrupt') for t in res.trace]}"

    lines = [
        json.loads(x)
        for x in (pb.dir / "turns.jsonl").read_text().splitlines()
        if x
    ]
    assert any(
        ln.get("interrupt") and "production_complete" in ln["interrupt"]
        for ln in lines
    ), "interrupt must be persisted in playback turns.jsonl"


def test_no_interrupts_is_fixed_step_backward_compatible():
    c = _pack(None).compile("easy", map_supported=True)
    res = run_level(c, _builder, seed=1)
    assert all(t["interrupt"] is None for t in res.trace), (
        "without enabled signals the loop must use fixed step (no interrupts)"
    )
    assert res.outcome in {"win", "draw", "loss"}
