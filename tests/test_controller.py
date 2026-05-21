"""Phase 1 — the unified Controller contract.

`openra_bench/controller.py` is the keystone of the human-labeling
machine and the 1v1 adversarial harness: LLM agents, human labelers and
scripted reference policies all implement one interface,

    controller.act(observation, Command) -> [Command]

and `run_level` / `run_episode` drive any of them. This file pins:

* the coercion layer (`as_controller`) — a bare `agent_fn`, a bound
  method, and an existing Controller all resolve correctly, so the ~190
  legacy test files that pass a bare function keep working;
* the introspection surface (`history` / `stats`) the playback writer
  reads survives the coercion;
* `ModelAgent` structurally satisfies the contract;
* an end-to-end `run_level` smoke: the SAME scripted policy produces a
  byte-identical outcome whether passed as a bare function or wrapped
  in a Controller.
"""

from __future__ import annotations

import pytest

from openra_bench.controller import (
    BaseController,
    Controller,
    EpisodeContext,
    FunctionController,
    as_controller,
    introspection_source,
    is_controller,
)


# ── Coercion (no engine needed) ─────────────────────────────────────


def _bare_policy(render_state, Command):
    """A legacy-shape agent_fn: ignore the world, just observe."""
    return [("OBSERVE", id(render_state))]


def test_as_controller_wraps_a_bare_function():
    c = as_controller(_bare_policy)
    assert is_controller(c)
    assert isinstance(c, FunctionController)
    # The wrapper delegates verbatim.
    out = c.act({"k": 1}, Command=None)
    assert out[0][0] == "OBSERVE"
    # Name defaults to the function's __name__.
    assert c.name == "_bare_policy"


def test_as_controller_is_idempotent_on_a_controller():
    c1 = as_controller(_bare_policy)
    c2 = as_controller(c1)
    assert c2 is c1, "coercing a Controller must return it unchanged"


def test_as_controller_rejects_non_callable():
    with pytest.raises(TypeError):
        as_controller(42)
    with pytest.raises(TypeError):
        as_controller("not a policy")


def test_as_controller_named_override():
    c = as_controller(_bare_policy, name="custom")
    assert c.name == "custom"


def test_is_controller_discriminates():
    # A bare function is callable but NOT a Controller.
    assert not is_controller(_bare_policy)
    assert not is_controller(lambda rs, C: [])
    # A FunctionController is.
    assert is_controller(as_controller(_bare_policy))


def test_base_controller_act_is_abstract():
    b = BaseController(name="x")
    assert b.name == "x"
    assert b.history == [] and b.stats == {}
    with pytest.raises(NotImplementedError):
        b.act({}, Command=None)


def test_episode_context_defaults():
    ctx = EpisodeContext()
    assert ctx.side == "agent"
    assert ctx.seed == 0 and ctx.max_turns == 0
    assert ctx.extra == {}
    ctx2 = EpisodeContext(pack_id="p", level="hard", side="enemy", seed=3)
    assert (ctx2.pack_id, ctx2.level, ctx2.side, ctx2.seed) == (
        "p", "hard", "enemy", 3
    )


# ── Bound-method source recovery (the playback path) ────────────────


class _FakeAgent:
    """Stand-in for ModelAgent: a bound `agent_fn` plus history/stats."""

    def __init__(self):
        self.history = [{"role": "system", "content": "hi"}]
        self.stats = {"turns": 0}

    def agent_fn(self, render_state, Command):
        self.stats["turns"] += 1
        return [("ACT", self.stats["turns"])]


def test_bound_method_source_is_recovered_for_playback():
    agent = _FakeAgent()
    c = as_controller(agent.agent_fn)
    assert is_controller(c)
    # The bound instance is reachable so playback can dump history/stats.
    assert c.source is agent
    assert introspection_source(c) is agent
    c.act({}, None)
    assert agent.stats["turns"] == 1


def test_introspection_source_falls_back_to_controller():
    c = as_controller(_bare_policy)  # plain function, no __self__
    assert c.source is None
    assert introspection_source(c) is c


# ── Subclassing BaseController ──────────────────────────────────────


class _CountingController(BaseController):
    def __init__(self):
        super().__init__(name="counter")
        self.turns = 0
        self.reset_calls = 0

    def reset(self, ctx: EpisodeContext) -> None:
        self.reset_calls += 1
        self.last_ctx = ctx

    def act(self, observation, Command):
        self.turns += 1
        return [("TURN", self.turns)]


def test_subclassed_controller_satisfies_contract():
    c = _CountingController()
    assert is_controller(c)
    assert isinstance(c, Controller)  # runtime_checkable structural
    c.reset(EpisodeContext(pack_id="p", side="enemy"))
    assert c.reset_calls == 1 and c.last_ctx.side == "enemy"
    assert c.act({}, None) == [("TURN", 1)]
    assert as_controller(c) is c


# ── ModelAgent structurally conforms ────────────────────────────────


def test_model_agent_class_exposes_controller_contract():
    # Structural check on the class — constructing a ModelAgent needs a
    # provider, but the contract is method presence.
    from openra_bench.agent import ModelAgent

    for member in ("act", "reset", "agent_fn"):
        assert callable(getattr(ModelAgent, member, None)), (
            f"ModelAgent must expose {member}() for the Controller contract"
        )


# ── End-to-end: bare fn vs Controller produce identical runs ────────

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip(
    "openra_rl_training", reason="Rust env wheel not installed"
)


def _stall(render_state, Command):
    return [Command.observe()]


def _smallest_easy_pack():
    """The active pack with the fewest easy-level turns — keeps the
    end-to-end smoke fast and deterministic."""
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
        if not c.map_supported:
            continue
        if best is None or c.max_turns < best.max_turns:
            best = c
    return best


def test_run_level_identical_for_bare_fn_and_controller():
    """The SAME scripted policy must yield a byte-identical EpisodeResult
    whether passed as a bare agent_fn or wrapped in a Controller — proof
    the coercion layer is transparent."""
    from openra_bench.eval_core import run_level

    compiled = _smallest_easy_pack()
    assert compiled is not None, "no runnable active pack found"

    r_fn = run_level(compiled, _stall, seed=1)
    r_ctrl = run_level(compiled, as_controller(_stall, name="stall"), seed=1)

    assert r_fn.outcome == r_ctrl.outcome
    assert r_fn.turns == r_ctrl.turns
    assert r_fn.actions_issued == r_ctrl.actions_issued
    assert r_fn.signals.game_tick == r_ctrl.signals.game_tick


def test_run_level_drives_a_subclassed_controller():
    """A BaseController subclass — the shape HumanController and the
    scripted-bot wrapper will take — runs end-to-end and its per-episode
    `reset` hook fires with a populated EpisodeContext."""
    from openra_bench.eval_core import run_level

    compiled = _smallest_easy_pack()
    assert compiled is not None

    class _StallController(BaseController):
        def __init__(self):
            super().__init__(name="stall-ctrl")
            self.acts = 0
            self.ctx = None

        def reset(self, ctx: EpisodeContext) -> None:
            self.ctx = ctx

        def act(self, observation, Command):
            self.acts += 1
            return [Command.observe()]

    ctrl = _StallController()
    res = run_level(compiled, ctrl, seed=1)
    assert res.outcome in ("win", "loss", "draw")
    assert ctrl.acts >= 1, "act() must have been called"
    assert ctrl.ctx is not None and ctrl.ctx.pack_id == compiled.pack_id
    assert ctrl.ctx.level == "easy" and ctrl.ctx.seed == 1
