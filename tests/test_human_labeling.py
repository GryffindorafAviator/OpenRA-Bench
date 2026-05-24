"""Phase 2 — the human-labeling machine.

`openra_bench/human_labeling.py` lets a human play the exact scenarios
the LLM plays. This file pins:

* the pixel⇄cell transforms (`minimap_click_to_cell` /
  `cell_to_minimap_rect`) — round-trip, edge clamping, bad-input guards;
* click-to-selection / click-to-target resolution against render_state;
* `HumanAction.to_tool_call` — every gesture normalises to the SAME
  tool-call shape the LLM emits (and malformed gestures are dropped);
* `HumanController` — satisfies the Phase 1 Controller contract, records
  a playback-compatible transcript, and runs end-to-end through
  `run_level` exactly like a model agent.
"""

from __future__ import annotations

import pytest

from openra_bench.controller import EpisodeContext, is_controller
from openra_bench.human_labeling import (
    HumanAction,
    HumanController,
    ScriptedInputSource,
    cell_to_minimap_rect,
    enemy_at_cell,
    minimap_click_to_cell,
    own_buildings_at_cell,
    own_units_at_cell,
)


# ── Pixel ⇄ cell transforms ─────────────────────────────────────────


def test_click_to_cell_basic():
    # 128×40 map rendered at CELL=6 → 768×240 px. A click at the centre
    # of cell (10, 5) is pixel (63, 33).
    assert minimap_click_to_cell(63, 33, 768, 240, 128, 40) == (10, 5)
    # Top-left corner.
    assert minimap_click_to_cell(0, 0, 768, 240, 128, 40) == (0, 0)


def test_click_to_cell_clamps_out_of_bounds():
    # Past the right/bottom edge → clamped to the last cell.
    assert minimap_click_to_cell(9999, 9999, 768, 240, 128, 40) == (127, 39)
    # Negative → clamped to 0.
    assert minimap_click_to_cell(-50, -50, 768, 240, 128, 40) == (0, 0)


def test_click_to_cell_rejects_bad_dimensions():
    for bad in [
        (10, 10, 0, 240, 128, 40),
        (10, 10, 768, 240, 0, 40),
        (10, 10, 768, -1, 128, 40),
    ]:
        with pytest.raises(ValueError):
            minimap_click_to_cell(*bad)


def test_cell_to_rect_round_trips():
    # Clicking anywhere inside a cell's rect recovers that cell.
    for cx, cy in [(0, 0), (10, 5), (127, 39), (64, 20)]:
        left, top, w, h = cell_to_minimap_rect(cx, cy, 768, 240, 128, 40)
        assert w >= 1 and h >= 1
        mid_x, mid_y = left + w // 2, top + h // 2
        assert minimap_click_to_cell(mid_x, mid_y, 768, 240, 128, 40) == (
            cx,
            cy,
        )


def test_transforms_are_renderer_agnostic():
    # Same cell, two different rendered sizes — both resolve correctly.
    assert minimap_click_to_cell(400, 150, 800, 300, 128, 40) == (64, 20)
    assert minimap_click_to_cell(200, 75, 400, 150, 128, 40) == (64, 20)


# ── Click → selection / target resolution ───────────────────────────


_RS = {
    "units_summary": [
        {"id": 7, "cell_x": 20, "cell_y": 10},
        {"id": 8, "cell_x": 21, "cell_y": 10},
        {"id": 9, "cell_x": 80, "cell_y": 30},
    ],
    "own_buildings": [
        {"id": 100, "cell_x": 18, "cell_y": 12},
    ],
    "enemy_summary": [
        {"id": 50, "cell_x": 60, "cell_y": 20},
        {"id": 51, "cell_x": 62, "cell_y": 21},
    ],
}


def test_own_units_at_cell_selects_by_proximity():
    sel = own_units_at_cell(_RS, 20, 10, radius=1)
    assert set(sel) == {"7", "8"}  # both within 1 cell
    assert own_units_at_cell(_RS, 80, 30, radius=0) == ["9"]
    assert own_units_at_cell(_RS, 5, 5, radius=1) == []  # empty region


def test_own_buildings_at_cell():
    assert own_buildings_at_cell(_RS, 18, 12, radius=1) == ["100"]
    assert own_buildings_at_cell(_RS, 20, 10, radius=1) == []


def test_enemy_at_cell_picks_nearest():
    # Click on (60,20) → enemy 50 sits there exactly.
    assert enemy_at_cell(_RS, 60, 20, radius=1) == "50"
    # Click nearer 51.
    assert enemy_at_cell(_RS, 62, 21, radius=1) == "51"
    # No enemy near the agent base.
    assert enemy_at_cell(_RS, 20, 10, radius=1) is None


# ── HumanAction → tool call ─────────────────────────────────────────


def test_action_move_normalises_to_move_units():
    a = HumanAction(mode="move", units=["7", "8"], target=(40, 20))
    assert a.to_tool_call() == {
        "name": "move_units",
        "arguments": {
            "unit_ids": ["7", "8"],
            "target_x": 40,
            "target_y": 20,
        },
    }


def test_action_attack_with_target_id_is_attack_unit():
    a = HumanAction(mode="attack", units=["7"], target_id="50")
    tc = a.to_tool_call()
    assert tc["name"] == "attack_unit"
    assert tc["arguments"] == {"unit_ids": ["7"], "target_id": "50"}


def test_action_attack_without_enemy_falls_back_to_attack_move():
    # Click on empty ground in attack mode → close on the cell.
    a = HumanAction(mode="attack", units=["7"], target=(55, 25))
    tc = a.to_tool_call()
    assert tc["name"] == "attack_move"
    assert tc["arguments"]["target_x"] == 55


def test_action_build_and_place():
    assert HumanAction(mode="build", unit_type="pbox").to_tool_call() == {
        "name": "build",
        "arguments": {"item": "pbox"},
    }
    place = HumanAction(
        mode="place_building", unit_type="pbox", target=(30, 15)
    ).to_tool_call()
    assert place["name"] == "place_building"
    assert place["arguments"] == {
        "item": "pbox",
        "target_x": 30,
        "target_y": 15,
    }


def test_action_unit_only_and_stance():
    assert HumanAction(mode="stop", units=["7"]).to_tool_call() == {
        "name": "stop",
        "arguments": {"unit_ids": ["7"]},
    }
    st = HumanAction(
        mode="set_stance", units=["7", "8"], stance=3
    ).to_tool_call()
    assert st == {
        "name": "set_stance",
        "arguments": {"unit_ids": ["7", "8"], "stance": 3},
    }


def test_action_observe_and_surrender():
    assert HumanAction(mode="observe").to_tool_call() == {
        "name": "observe",
        "arguments": {},
    }
    assert HumanAction(mode="surrender").to_tool_call() == {
        "name": "surrender",
        "arguments": {},
    }


def test_malformed_actions_drop_to_none():
    # move without units / without target → not a valid command.
    assert HumanAction(mode="move", target=(1, 2)).to_tool_call() is None
    assert HumanAction(mode="move", units=["7"]).to_tool_call() is None
    # attack with neither target_id nor cell.
    assert HumanAction(mode="attack", units=["7"]).to_tool_call() is None
    # build with no item; unknown gesture.
    assert HumanAction(mode="build").to_tool_call() is None
    assert HumanAction(mode="nonsense", units=["7"]).to_tool_call() is None


def test_action_describe_is_human_readable():
    d = HumanAction(mode="move", units=["7", "8"], target=(40, 20)).describe()
    assert "move" in d and "40,20" in d


# ── ScriptedInputSource ─────────────────────────────────────────────


def test_scripted_input_source_yields_then_observes():
    src = ScriptedInputSource(
        [
            [HumanAction(mode="move", units=["7"], target=(5, 5))],
            [HumanAction(mode="stop", units=["7"])],
        ]
    )
    assert src({})[0].mode == "move"
    assert src({})[0].mode == "stop"
    # Exhausted → pass-turn observe forever.
    assert src({})[0].mode == "observe"
    assert src({})[0].mode == "observe"


# ── HumanController contract + transcript ───────────────────────────


def test_human_controller_satisfies_controller_contract():
    c = HumanController([], name="tester")
    assert is_controller(c)
    assert c.name == "tester"


def test_human_controller_reset_seeds_transcript():
    c = HumanController([])
    c.reset(
        EpisodeContext(
            pack_id="demo", level="easy", seed=2, objective="hold the line"
        )
    )
    assert c.history[0]["role"] == "system"
    assert "demo:easy" in c.history[0]["content"]
    assert "hold the line" in c.history[0]["content"]
    assert c.stats == {"turns": 0, "tool_calls": 0, "empty_replies": 0}


# ── Engine-backed: command translation + end-to-end run ─────────────

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)


def test_human_actions_translate_to_real_commands():
    """A human's gestures become real engine Commands via the SAME
    `agent._to_commands` the LLM path uses."""
    from openra_train import Command

    from openra_bench.human_labeling import human_actions_to_commands

    actions = [
        HumanAction(mode="move", units=["1"], target=(10, 10)),
        HumanAction(mode="observe"),
    ]
    cmds = human_actions_to_commands(actions, Command)
    assert len(cmds) == 2
    assert "MoveUnits" in repr(cmds[0])
    assert "Observe" in repr(cmds[1])


def _smallest_easy_pack():
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import PACKS_DIR, compile_level

    best = None
    for f in sorted(PACKS_DIR.glob("*.yaml")):
        if f.name.startswith(("_", "TEMPLATE")):
            continue
        try:
            pack = load_pack(f)
            if pack.meta.status != "active" or "easy" not in pack.levels:
                continue
            c = compile_level(pack, "easy")
        except Exception:  # noqa: BLE001
            continue
        if c.map_supported and (best is None or c.max_turns < best.max_turns):
            best = c
    return best


def test_human_controller_runs_end_to_end_through_run_level():
    """A HumanController driven by a scripted session runs through
    `run_level` exactly like a model agent — producing a scored
    EpisodeResult and a playback-compatible transcript."""
    from openra_bench.eval_core import run_level

    compiled = _smallest_easy_pack()
    assert compiled is not None

    # A trivial scripted human: observe every turn (a pass-turn run).
    ctrl = HumanController(ScriptedInputSource([]), name="human-test")
    res = run_level(compiled, ctrl, seed=1)

    assert res.outcome in ("win", "loss", "draw")
    assert ctrl.stats["turns"] >= 1
    # Transcript is playback-shaped: system seed + per-turn user/assistant.
    assert ctrl.history[0]["role"] == "system"
    roles = {m["role"] for m in ctrl.history}
    assert {"user", "assistant"} <= roles
    # Every assistant turn carries structured tool_calls (observe here).
    asst = [m for m in ctrl.history if m["role"] == "assistant"]
    assert asst and all("tool_calls" in m for m in asst)


def test_run_human_session_scores_a_named_pack():
    """`run_human_session` compiles a real pack by id and scores a
    human run through the same path as a model run."""
    from openra_bench.human_labeling import run_human_session

    compiled = _smallest_easy_pack()
    assert compiled is not None

    res = run_human_session(
        compiled.pack_id, level="easy", seed=1, input_source=[]
    )
    assert res.outcome in ("win", "loss", "draw")
    assert res.scenario == f"{compiled.pack_id}:easy"
    assert res.seed == 1


# ── InteractiveSession (GUI-driven turn stepping) ───────────────────


def test_interactive_session_steps_turn_by_turn():
    """The session inverts run_level's loop: the caller drives one turn
    per submit_turn(), and it terminates with a scored outcome — the
    backend the Gradio 'Play' tab wraps."""
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(compiled, seed=1, record=False)
    try:
        # The observation is a real render_state — what the LLM sees.
        rs = sess.render_state()
        assert isinstance(rs, dict)
        st = sess.status()
        assert st["turn"] == 0 and not st["done"]
        assert st["max_turns"] == compiled.max_turns

        # Drive observe-only turns until the session ends.
        guard = compiled.max_turns + 5
        steps = 0
        while not sess.done and steps < guard:
            out = sess.submit_turn([HumanAction(mode="observe")])
            steps += 1
            assert out["turn"] == steps
        assert sess.done
        assert sess.outcome in ("win", "loss", "draw")
        # A submit after termination is a no-op.
        frozen = sess.turn
        sess.submit_turn([HumanAction(mode="observe")])
        assert sess.turn == frozen
    finally:
        sess.close()
        sess.close()  # idempotent


def test_interactive_session_from_pack():
    """`InteractiveSession.from_pack` opens a session by pack id and
    exposes the engine Command factory for click translation."""
    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    sess = InteractiveSession.from_pack(
        compiled.pack_id, "easy", seed=2, record=False
    )
    try:
        assert sess.seed == 2
        assert sess.Command is not None
        # A move gesture translates and applies without error.
        rs = sess.render_state()
        units = rs.get("units_summary") or []
        if units:
            uid = str(units[0]["id"])
            sess.submit_turn(
                [HumanAction(mode="move", units=[uid], target=(30, 18))]
            )
            assert sess.turn == 1
    finally:
        sess.close()


def test_interactive_session_emits_standard_playback(tmp_path):
    """A recorded human session produces the SAME Playback artifact a
    model run does — turns.jsonl + per-turn minimap PNGs + messages.json
    + a manifest — so the Battle Viewer / leaderboard treat human and
    LLM runs apples-to-apples."""
    import json
    from pathlib import Path

    from openra_bench.human_labeling import InteractiveSession

    compiled = _smallest_easy_pack()
    assert compiled is not None

    sess = InteractiveSession(
        compiled, seed=1, record=True, playback_root=tmp_path,
        player="Human",
    )
    try:
        guard = compiled.max_turns + 5
        steps = 0
        while not sess.done and steps < guard:
            sess.submit_turn([HumanAction(mode="observe")])
            steps += 1
        assert sess.done
        save = sess.status()["save_path"]
        assert save, "a finished recorded session must report save_path"
        run_dir = Path(save)
        # Standard Playback layout — identical to a model run's.
        assert (run_dir / "turns.jsonl").is_file()
        assert (run_dir / "messages.json").is_file()
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["model"] == "Human"
        assert manifest["agent_type"] == "Human"
        assert manifest["pack_id"] == compiled.pack_id
        assert manifest["outcome"] in ("win", "loss", "draw")
        assert manifest["turns"] == sess.turn
        # Per-turn minimap PNG frames were written.
        assert list(run_dir.glob("minimap_turn*.png"))
        # turns.jsonl has one record per turn.
        lines = [
            ln for ln in (run_dir / "turns.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        assert len(lines) == sess.turn
    finally:
        sess.close()


def test_interactive_session_adds_play_hints_for_exact_objectives():
    from openra_bench.human_labeling import InteractiveSession

    sess = InteractiveSession.from_pack(
        "action-multiunit-coordination", "easy", seed=1, record=False
    )
    try:
        rs = sess.render_state()
        types = {u.get("type") for u in rs.get("units_summary") or []}
        assert {"1tnk", "2tnk"} <= types
        regions = rs.get("objective_regions") or []
        assert {r["x"] for r in regions} == {44}
        assert {r["y"] for r in regions} == {4, 34}
    finally:
        sess.close()
