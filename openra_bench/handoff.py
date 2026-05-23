"""Handoff ablation — hand a model a partially-played game.

A handoff episode is split: a `prefix` controller plays the first `k`
turns, then the model inherits the live game state and finishes it.
It is a PURE STATE handoff — the model gets no transcript of the
prefix, only the board it produced ("take over from here").

Sweeping the prefix QUALITY decomposes two capabilities:

* a **good** prefix (a winning trajectory) → can the model *capitalize
  on an advantage*? A flat-low outcome curve means it derails even a
  won position.
* a **bad** prefix (a losing trajectory, or `stall`) → can the model
  *recover from a deficit*? This is the controlled measurement of the
  freeze-and-panic failure mode: handed a losing board, does the model
  fight (retreat / redirect) or sit on `observe`/`stop`? The
  `passivity` stat on the result quantifies exactly that.

The prefix is a recorded run replayed turn-for-turn. Because engine
actor ids are seed-deterministic, a replayed trajectory MUST come from
the same `pack:level:seed` as the handoff episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .controller import BaseController, as_controller, introspection_source

# A turn is "passive" when the model issued nothing but these — the
# freeze-and-panic tell (low-commitment default instead of an active
# retreat / redirect).
_PASSIVE_TOOLS = {"observe", "stop"}


def stall_policy(render_state: dict, Command: Any) -> list:
    """The canonical losing prefix: do nothing, every turn. Synthesises
    a guaranteed-deficit handoff with no recorded trajectory needed."""
    return [Command.observe()]


def _load_trajectory(source: Any) -> list[list[dict]]:
    """Per-turn tool-call lists from a recorded run. `source` may be a
    ready list, a Playback directory, or a path to its messages.json."""
    if isinstance(source, list):
        return source
    p = Path(source)
    if p.is_dir():
        p = p / "messages.json"
    msgs = json.loads(p.read_text())
    turns: list[list[dict]] = []
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        calls: list[dict] = []
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {}
            calls.append({"name": fn.get("name"), "arguments": args or {}})
        turns.append(calls)
    return turns


class TrajectoryController(BaseController):
    """Replays a recorded run: turn N re-issues the commands the
    recorded agent issued on its turn N. Past the recording's end it
    falls back to `observe()`. Used as a deterministic handoff prefix —
    a `win`-outcome run is a good prefix, a `loss` is a bad one."""

    def __init__(self, source: Any, name: str | None = None) -> None:
        super().__init__(name or "trajectory")
        self._turns = _load_trajectory(source)
        self._i = 0

    def reset(self, ctx: Any) -> None:
        self._i = 0

    def act(self, observation: dict, Command: Any) -> list:
        from .agent import _to_commands

        if self._i < len(self._turns):
            calls = self._turns[self._i]
            self._i += 1
            return _to_commands(calls, Command) or [Command.observe()]
        return [Command.observe()]


def _is_passive(cmds: list, _cmd_name) -> bool:
    """A turn with no command other than observe/stop (or no command)."""
    if not cmds:
        return True
    return all((_cmd_name(c) or "") in _PASSIVE_TOOLS for c in cmds)


class HandoffController(BaseController):
    """`prefix` plays turns 0..k-1, then `main` inherits the live state
    and finishes the episode. Pure state handoff — `main` carries no
    transcript of the prefix.

    `handoff_stats` accumulates, over the MAIN agent's turns only:
    `main_turns`, `passive_turns` (observe/stop-only), and `passivity`
    (their ratio) — the freeze-and-panic signal. When the prefix handed
    `main` a losing position, `passivity` IS passivity-under-pressure."""

    def __init__(
        self, prefix: Any, main: Any, k: int, name: str | None = None
    ) -> None:
        super().__init__(name or f"handoff-k{int(k)}")
        self._prefix = as_controller(prefix)
        self._main = as_controller(main)
        self._k = max(0, int(k))
        self._turn = 0
        # Playback should record the MAIN agent's transcript, not this
        # wrapper's — expose it as the introspection source.
        self.source = introspection_source(self._main)
        self.handoff_stats = self._fresh_stats()

    def _fresh_stats(self) -> dict:
        return {
            "k": self._k, "main_turns": 0,
            "passive_turns": 0, "passivity": 0.0,
        }

    def reset(self, ctx: Any) -> None:
        self._turn = 0
        self._prefix.reset(ctx)
        self._main.reset(ctx)
        self.handoff_stats = self._fresh_stats()

    def act(self, observation: dict, Command: Any) -> list:
        if self._turn < self._k:
            self._turn += 1
            return self._prefix.act(observation, Command)
        cmds = self._main.act(observation, Command)
        self._turn += 1
        from .eval_core import _cmd_tool_name

        st = self.handoff_stats
        st["main_turns"] += 1
        if _is_passive(cmds, _cmd_tool_name):
            st["passive_turns"] += 1
        st["passivity"] = st["passive_turns"] / st["main_turns"]
        return cmds


def run_handoff(
    compiled: Any, main: Any, prefix: Any, k: int,
    seed: int = 0, playback: Any = None,
):
    """Run a handoff episode: `prefix` plays the first `k` turns, `main`
    finishes. Returns the `EpisodeResult` with `.handoff_stats` attached
    (k, main_turns, passive_turns, passivity)."""
    from .eval_core import run_level

    ctrl = HandoffController(prefix, main, k)
    res = run_level(compiled, ctrl, seed, playback)
    res.handoff_stats = dict(ctrl.handoff_stats)
    return res
