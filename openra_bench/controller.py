"""Unified policy interface for the OpenRA-Bench eval stack.

Every actor that can drive a side of a scenario — an LLM agent, a human
labeler, a scripted reference policy — implements the same contract:

    controller.act(observation, Command) -> list[Command]

This is the keystone of the human-labeling machine and the 1v1
adversarial harness: one harness, interchangeable policy backends.
`run_level` / `run_episode` drive a single Controller; a 1v1 match
drives two, one per side, each fed its own side-specific observation.

Back-compat is non-negotiable: the historical policy shape was a bare
callable ``agent_fn(render_state, Command) -> [Command]`` and ~190 test
files still pass one. `as_controller()` adapts any such callable (or a
`ModelAgent` bound method) into a Controller, so every existing scripted
policy and test keeps working unchanged — the eval loop simply coerces
its policy argument through `as_controller()` before stepping.

Design notes
------------
* `act` keeps `Command` as an explicit parameter rather than binding it
  at construction. `Command` is the pyo3 `openra_train.Command` factory
  handle, only available once an env exists; threading it per-call keeps
  Controllers constructible without an engine (cheap to unit-test) and
  is byte-identical to the legacy `agent_fn` signature.
* `reset(ctx)` is the per-episode lifecycle hook. Scripted policies
  ignore it; the model agent re-arms history; a human controller would
  reset its click queue. The 1v1 harness calls it once per side with a
  `side`-stamped `EpisodeContext`.
* `history` / `stats` are the optional introspection surface the
  playback writer reads. `BaseController` provides empty defaults so a
  caller can read them unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

# A bare legacy policy: (render_state, Command) -> [Command].
PolicyFn = Callable[[dict, Any], list]


@dataclass
class EpisodeContext:
    """What a Controller is told once, at episode start (`reset`).

    A scenario eval populates `pack_id` / `level` / `seed` / `objective`;
    a 1v1 match additionally stamps `side` so the two Controllers know
    which colour they are driving."""

    pack_id: str = ""
    level: str = ""
    seed: int = 0
    side: str = "agent"  # "agent" | "enemy" — which side this drives
    objective: str = ""
    max_turns: int = 0
    extra: dict = field(default_factory=dict)


@runtime_checkable
class Controller(Protocol):
    """A policy that observes the world and emits engine Commands.

    Structural — anything exposing `name`, `reset`, and `act` satisfies
    it; `ModelAgent` does so without importing this module."""

    name: str

    def reset(self, ctx: "EpisodeContext") -> None: ...

    def act(self, observation: dict, Command: Any) -> list: ...


def is_controller(obj: Any) -> bool:
    """True if `obj` already satisfies the Controller contract.

    Deliberately structural and stricter than `isinstance(obj,
    Controller)`: a bare function is callable but is NOT a Controller,
    so it must carry callable `act` AND `reset` attributes — which a
    plain function never does."""
    return callable(getattr(obj, "act", None)) and callable(
        getattr(obj, "reset", None)
    )


class BaseController:
    """Convenience base: a no-op `reset`, a `name`, empty introspection.

    Subclass and implement `act`. Concrete eval policies (the human
    bridge, scripted reference wrappers) derive from this so they share
    one introspection surface (`history`, `stats`)."""

    name: str = "controller"

    def __init__(self, name: str | None = None) -> None:
        if name:
            self.name = name
        self.history: list[dict] = []
        self.stats: dict[str, Any] = {}

    def reset(self, ctx: EpisodeContext) -> None:  # noqa: D401
        """Per-episode lifecycle hook. Default: no-op."""

    def act(self, observation: dict, Command: Any) -> list:
        raise NotImplementedError(
            f"{type(self).__name__} must implement act()"
        )


class FunctionController(BaseController):
    """Adapt a bare ``agent_fn(render_state, Command) -> [Command]``
    callable into a Controller — the back-compat bridge for every
    scripted reference policy and the legacy `scripted_explore_agent`.

    When the callable is a bound method (e.g. ``ModelAgent.agent_fn``),
    its ``__self__`` is captured as `source` so the eval loop can still
    reach the underlying object's `history` / `stats` for playback."""

    def __init__(
        self, fn: PolicyFn, name: str | None = None
    ) -> None:
        super().__init__(
            name or getattr(fn, "__name__", None) or "fn"
        )
        self._fn = fn
        self.source: Any = getattr(fn, "__self__", None)

    def act(self, observation: dict, Command: Any) -> list:
        return self._fn(observation, Command)


def as_controller(policy: Any, name: str | None = None) -> Controller:
    """Coerce anything policy-shaped into a Controller.

    Accepts, in priority order:
      * an object already satisfying the Controller contract — returned
        as-is (idempotent);
      * any callable — a bare `agent_fn` or a bound method — wrapped in
        a `FunctionController` (a bound method's `__self__` is kept
        reachable via `.source`).

    Raises `TypeError` for anything else."""
    if is_controller(policy):
        return policy
    if callable(policy):
        return FunctionController(policy, name)
    raise TypeError(
        f"cannot coerce {type(policy).__name__} into a Controller: "
        "expected a Controller, a ModelAgent, or an "
        "agent_fn(render_state, Command) -> [Command] callable"
    )


def introspection_source(controller: Controller) -> Any:
    """The object carrying `history` / `stats` for playback.

    For a `FunctionController` wrapping a bound method this is the bound
    instance (`.source`); otherwise it is the Controller itself."""
    src = getattr(controller, "source", None)
    return src if src is not None else controller
