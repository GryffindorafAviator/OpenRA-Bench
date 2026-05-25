"""Battle viewer data layer: index, cascade, turn stepping, compare.

Builds a synthetic playback tree with two runs/models on the *same*
scenario+seed (the comparison case) and asserts the run→model→scenario
cascade, per-turn assembly, clamping, and the compare-pairing rule
(B locked to A's scenario+seed, A excluded).
"""

from __future__ import annotations

import pytest
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")
from openra_bench.battle_viewer import (
    compare_candidates,
    episode_view,
    find,
    models,
    runs,
    scan,
    scenarios,
)
from openra_bench.playback import Playback


class _Sig:
    game_tick = 100
    cash = 0
    resources = 0
    explored_percent = 0.0
    units_killed = 0
    units_lost = 0
    enemies_seen_ids: list = []


def _make(root, run_id, model, scenario, seed, n_turns, outcome):
    pb = Playback(root / f"{run_id}__{model}", scenario, seed)
    pb.run_id, pb.model = run_id, model
    for t in range(1, n_turns + 1):
        pb.record_turn(
            t, {"minimap": f"map{t}", "units_summary": [], "enemy_summary": []},
            [f"Command::Move({t})"], _Sig(), None,
            goal={"leaves": [{"name": "units_killed_gte", "current": t,
                              "target": n_turns, "ratio": t / n_turns,
                              "satisfied": t == n_turns}],
                  "reward_vector": {"military": t / n_turns},
                  # `objective_blocking_ratio` is the new key
                  # (worst-leaf ratio); `objective_progress` is the
                  # deprecated alias kept one release for back-compat.
                  "objective_blocking_ratio": t / n_turns,
                  "objective_progress": t / n_turns,
                  "won": t == n_turns},
        )
    pb.write_messages([
        {"role": "system", "content": "s"},
        *sum(([{"role": "user", "content": f"briefing turn {t}"},
               {"role": "assistant", "content": f"act {t}",
                "reasoning": f"think {t}", "tool_calls": []}]
              for t in range(1, n_turns + 1)), []),
    ])
    pb.finalize({"scenario": scenario, "seed": seed, "outcome": outcome,
                 "run_id": run_id, "model": model})
    (pb.dir / "score.json").write_text(
        '{"composite": 0.5, "objective_progress": 1.0}'
    )
    return pb.dir


def test_cascade_and_compare(tmp_path):
    sc = "perception-frontier-reading:easy:public"
    _make(tmp_path, "run-A", "modelX", sc, 7, 3, "win")
    _make(tmp_path, "run-A", "modelY", sc, 7, 4, "loss")
    _make(tmp_path, "run-B", "modelX", sc, 7, 2, "draw")
    _make(tmp_path, "run-A", "modelX", "other:easy:public", 1, 2, "loss")

    idx = scan(tmp_path)
    assert set(runs(idx)) == {"run-A", "run-B"}
    assert set(models(idx, "run-A")) == {"modelX", "modelY"}
    # modelX in run-A has two scenarios (sc@7 and other@1)
    assert f"{sc}@7" in scenarios(idx, "run-A", "modelX")
    assert "other:easy:public@1" in scenarios(idx, "run-A", "modelX")

    a = find(idx, "run-A", "modelX", f"{sc}@7")
    assert a is not None and a.outcome == "win"

    # compare candidates: same scenario+seed, A excluded → modelY/run-A
    # and modelX/run-B (NOT the other-scenario episode, NOT A itself)
    cands = {(e.run_id, e.model) for e in compare_candidates(idx, a)}
    assert cands == {("run-A", "modelY"), ("run-B", "modelX")}


def test_episode_view_steps_and_clamps(tmp_path):
    d = _make(tmp_path, "r", "m", "s:easy:public", 0, 3, "win")
    v0 = episode_view(d, 0)
    assert v0["n_turns"] == 3 and v0["turn"] == 1
    assert v0["briefing"] == "briefing turn 1"
    assert v0["reasoning"] == "think 1"
    # Per-leaf table is the public surface; the scalar
    # `objective_progress` is no longer in the view dict (it
    # averaged unrelated clauses and produced misleading near-win
    # readings — see goal_tracker). The per-leaf detail is
    # `leaves_final`, with explicit current/target/satisfied.
    assert "objective_progress" not in v0
    assert v0["leaves_final"]
    lf = v0["leaves_final"][0]
    assert lf["name"] == "units_killed_gte"
    assert lf["current"] == 1 and lf["target"] == 3
    assert lf["satisfied"] is False
    # clamp past the end → last turn, not an error
    vend = episode_view(d, 99)
    assert vend["turn_idx"] == 2 and vend["turn"] == 3
    assert vend["won"] is True
    assert vend["leaves_final"][0]["satisfied"] is True
    # clamp below 0
    assert episode_view(d, -5)["turn_idx"] == 0
