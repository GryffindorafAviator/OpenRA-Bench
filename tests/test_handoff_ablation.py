"""The handoff ablation — hand a model a partially-played game.

A `prefix` controller plays the first K turns, then the model inherits
the live state. A GOOD prefix (winning trajectory) tests
capitalize-on-advantage; a BAD prefix (`stall`) tests recovery — and
the `passivity` stat (observe/stop-only turns) quantifies the
freeze-and-panic failure mode the recovery cell is built to expose.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.handoff import (HandoffController, TrajectoryController,
                                  _load_trajectory, run_handoff, stall_policy)
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import compile_level

PACKS = Path(__file__).parent.parent / "openra_bench" / "scenarios" / "packs"
_PACK = "perception-count-the-threat.yaml"


def _compiled(level: str = "easy"):
    return compile_level(load_pack(PACKS / _PACK), level)


# ── Trajectory loading / replay ──────────────────────────────────────

def test_load_trajectory_list_passthrough():
    traj = [[{"name": "observe", "arguments": {}}]]
    assert _load_trajectory(traj) is traj


def test_load_trajectory_from_messages_json(tmp_path):
    msgs = [
        {"role": "system", "content": "x"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function", "function": {
                "name": "move_units",
                "arguments": {"unit_ids": [1], "target_x": 5, "target_y": 5},
            }}]},
        {"role": "tool", "tool_call_id": "c0", "content": "ok"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": "observe", "arguments": {}}}]},
    ]
    p = tmp_path / "messages.json"
    p.write_text(json.dumps(msgs))
    turns = _load_trajectory(p)
    assert len(turns) == 2
    assert turns[0][0]["name"] == "move_units"
    assert turns[1][0]["name"] == "observe"


def test_trajectory_controller_replays_then_falls_back():
    import openra_train

    C = openra_train.Command
    tc = TrajectoryController([
        [{"name": "observe", "arguments": {}}],
        [{"name": "stop", "arguments": {"unit_ids": [1]}}],
    ])
    tc.reset(None)
    assert "Observe" in repr(tc.act({}, C)[0])
    assert "Stop" in repr(tc.act({}, C)[0])
    # past the recording's end → observe
    assert "Observe" in repr(tc.act({}, C)[0])


# ── Handoff switch + passivity ───────────────────────────────────────

def test_handoff_switches_prefix_to_main_at_k():
    pcalls, mcalls = [], []

    def prefix(rs, C):
        pcalls.append(1)
        return [C.observe()]

    def main(rs, C):
        mcalls.append(1)
        return [C.observe()]

    res = run_handoff(_compiled("easy"), main=main, prefix=prefix, k=3, seed=1)
    assert len(pcalls) == 3, "prefix must play exactly k turns"
    assert len(mcalls) == res.turns - 3, "main plays the remainder"
    assert res.handoff_stats["k"] == 3
    assert res.handoff_stats["main_turns"] == len(mcalls)


def test_passivity_is_one_when_main_freezes():
    """A main that only ever observes scores passivity 1.0 — the
    freeze-and-panic signal; an active policy scores low."""
    from openra_bench.eval_core import scripted_explore_agent

    frozen = run_handoff(
        _compiled("medium"), main=stall_policy, prefix=stall_policy,
        k=2, seed=1,
    )
    assert frozen.handoff_stats["passivity"] == 1.0

    active = run_handoff(
        _compiled("medium"), main=scripted_explore_agent,
        prefix=stall_policy, k=2, seed=1,
    )
    assert active.handoff_stats["passivity"] < 0.5


def test_k0_handoff_is_a_full_episode():
    from openra_bench.eval_core import scripted_explore_agent

    res = run_handoff(
        _compiled("easy"), main=scripted_explore_agent,
        prefix=stall_policy, k=0, seed=1,
    )
    assert res.handoff_stats["main_turns"] == res.turns


# ── Sweep wiring ─────────────────────────────────────────────────────

def test_handoff_sweep_expands_base_and_bad_cells():
    from openra_bench.run_eval import evaluate

    out = evaluate(
        [PACKS / _PACK], levels=["easy"], seeds=[1],
        handoff_sweep=True, dry_run=True,
    )
    assert set(out["cells"]) == {
        "perception-count-the-threat:easy:handoff-base",
        "perception-count-the-threat:easy:handoff-bad",
    }


def test_find_win_trajectory_matches_a_banked_win(tmp_path):
    from openra_bench.run_eval import _find_win_trajectory

    d = tmp_path / "run" / "p__seed1"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps(
        {"pack_id": "p", "level": "easy", "seed": 1, "outcome": "win"}))
    (d / "messages.json").write_text("[]")
    assert _find_win_trajectory(
        tmp_path, "p:easy:handoff-good", 1
    ) == str(d / "messages.json")
    # a different seed / a loss is not matched
    assert _find_win_trajectory(tmp_path, "p:easy:handoff-good", 2) is None
