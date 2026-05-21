"""Phase 3 — the 1v1 full-macro adversarial harness.

`openra_bench/one_v_one.py` drives two Controllers over one shared
episode via the engine's `step_1v1` two-player command channel. This
file pins:

* the engine `step_1v1` itself — both players' commands apply into the
  same frame, each side gets its OWN fog-of-war observation;
* `run_1v1` end-to-end — the match terminates and decides a winner;
* perspective correctness — the enemy controller sees the enemy's
  actors as its own units, distinct from the agent's view;
* both controllers' commands actually reach the engine.
"""

from __future__ import annotations

import pytest

from openra_bench.controller import BaseController, EpisodeContext

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)


def test_engine_exposes_step_1v1():
    """The rebuilt wheel must carry the two-player command channel."""
    from openra_train import OpenRAEnv

    assert hasattr(OpenRAEnv, "step_1v1"), (
        "engine wheel lacks step_1v1 — rebuild with maturin develop"
    )


def _combat_scenario_path() -> str:
    """Compile a combat pack (both sides have units) to a temp scenario
    YAML the Rust env can load."""
    from openra_bench.eval_core import _scenario_to_tmp_yaml
    from openra_bench.scenarios import load_pack
    from openra_bench.scenarios.loader import PACKS_DIR, compile_level

    for f in sorted(PACKS_DIR.glob("combat-*.yaml")):
        try:
            pack = load_pack(f)
            if pack.meta.status != "active" or "easy" not in pack.levels:
                continue
            compiled = compile_level(pack, "easy")
        except Exception:  # noqa: BLE001
            continue
        if compiled.map_supported:
            return _scenario_to_tmp_yaml(compiled)
    raise RuntimeError("no runnable combat pack found")


def _stall(render_state, Command):
    return [Command.observe()]


def test_run_1v1_terminates_with_a_winner():
    """Two scripted controllers play a full match; it ends and the
    harness decides a winner from the final boards."""
    from openra_bench.one_v_one import OneVOneResult, run_1v1

    path = _combat_scenario_path()
    res = run_1v1(path, _stall, _stall, seed=1, max_turns=60)

    assert isinstance(res, OneVOneResult)
    assert res.winner in ("agent", "enemy", "draw")
    assert res.reason
    assert res.turns >= 0
    assert res.ticks >= 0
    # Both sides' per-turn traces were recorded.
    assert len(res.agent_trace) == len(res.enemy_trace)


class _Recorder(BaseController):
    """Captures the first render_state it is asked to act on, and the
    EpisodeContext it was reset with."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self.first_rs: dict | None = None
        self.ctx: EpisodeContext | None = None
        self.act_calls = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self.ctx = ctx

    def act(self, observation, Command):
        if self.first_rs is None:
            self.first_rs = observation
        self.act_calls += 1
        return [Command.observe()]


def test_each_side_gets_its_own_perspective():
    """The agent and enemy controllers are driven with side-stamped
    EpisodeContexts and fed DISTINCT fog-of-war observations."""
    from openra_bench.one_v_one import run_1v1

    path = _combat_scenario_path()
    agent_rec = _Recorder("agent-side")
    enemy_rec = _Recorder("enemy-side")
    run_1v1(path, agent_rec, enemy_rec, seed=1, max_turns=20)

    # reset() stamped each side correctly.
    assert agent_rec.ctx is not None and agent_rec.ctx.side == "agent"
    assert enemy_rec.ctx is not None and enemy_rec.ctx.side == "enemy"
    # Both controllers were actually driven.
    assert agent_rec.act_calls >= 1 and enemy_rec.act_calls >= 1
    # Each saw its own board — the agent's own units are the enemy's
    # opponents and vice versa, so the two observations are not equal.
    a_units = {
        u.get("id") for u in (agent_rec.first_rs or {}).get(
            "units_summary", []
        )
    }
    e_units = {
        u.get("id") for u in (enemy_rec.first_rs or {}).get(
            "units_summary", []
        )
    }
    # At least one side must field units; the id sets must be disjoint
    # (no actor is "own" to both players).
    assert a_units or e_units, "neither side had any units"
    assert a_units.isdisjoint(e_units), (
        "agent and enemy observations share an own-unit id — "
        "perspective leak"
    )


def test_both_controllers_commands_reach_the_engine():
    """A controller that orders its units to move advances the game on
    BOTH sides — proof step_1v1 applies each side's orders."""
    from openra_bench.one_v_one import run_1v1

    path = _combat_scenario_path()

    def _wander(render_state, Command):
        # Order every own unit toward the map centre.
        cmds = []
        for u in render_state.get("units_summary", []) or []:
            uid = u.get("id")
            if uid is not None:
                cmds.append(
                    Command.move_units([str(uid)], target_x=40, target_y=20)
                )
        return cmds or [Command.observe()]

    res = run_1v1(path, _wander, _wander, seed=2, max_turns=40)
    assert res.winner in ("agent", "enemy", "draw")
    # The match advanced real ticks (the engine ran, not a no-op).
    assert res.ticks > 0
    # Commands were issued on at least one recorded turn per side.
    assert any(t["n_cmds"] >= 1 for t in res.agent_trace)
    assert any(t["n_cmds"] >= 1 for t in res.enemy_trace)


def test_human_controller_can_drive_a_1v1_side():
    """Phase 2 x Phase 3 integration — a HumanController (the
    human-labeling backend) drives one side of a 1v1 match against a
    scripted opponent, proving the human-vs-bot 1v1 path works."""
    from openra_bench.human_labeling import HumanController, ScriptedInputSource
    from openra_bench.one_v_one import run_1v1

    path = _combat_scenario_path()
    # A scripted "human": observe every turn (a pass-turn human).
    human = HumanController(ScriptedInputSource([]), name="human")
    res = run_1v1(path, human, _stall, seed=1, max_turns=40)

    assert res.winner in ("agent", "enemy", "draw")
    assert res.agent_name == "human"
    assert res.ticks >= 0
    # The HumanController was actually driven each turn.
    assert len(res.agent_trace) == len(res.enemy_trace)
