"""Playback persistence (pipeline step 7).

Saves, per scenario episode, everything needed to *replay what
happened*: the full model⇄env message transcript (system / user /
assistant, including the minimap image as a data-URL), a per-turn
record (tick, commands issued, signal snapshot, ASCII minimap), and a
manifest with scenario meta + outcome + score. Written under a
dedicated folder so people can inspect every run.

Layout:
  <root>/<pack:level>/seed<seed>/
      manifest.json      scenario meta, outcome, scorecard, agent stats
      turns.jsonl        one JSON object per turn
      messages.json      full chat transcript (ModelAgent only)
      minimap_turnNN.png rendered minimap per turn (when available)

Entirely optional and additive: `run_level(..., playback=None)` is the
default and changes nothing.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(_jsonable(v) for v in o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


class Playback:
    """Per-episode recorder. Create one per (scenario, seed)."""

    def __init__(self, root: str | Path, cell: str, seed: int):
        self.dir = Path(root) / cell.replace("/", "_") / f"seed{seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._turns_fh = open(self.dir / "turns.jsonl", "w")
        self._n = 0

    def record_turn(
        self,
        turn: int,
        render_state: dict,
        cmds: list,
        signals: Any,
        minimap_png_b64: str | None = None,
        interrupt: str | None = None,
        goal: dict | None = None,
    ) -> None:
        self._n += 1
        if minimap_png_b64:
            try:
                (self.dir / f"minimap_turn{turn:03d}.png").write_bytes(
                    base64.b64decode(minimap_png_b64)
                )
            except Exception:  # noqa: BLE001 — never break a run on I/O
                pass
        rec = {
            "turn": turn,
            "tick": getattr(signals, "game_tick", None),
            "interrupt": interrupt,
            "commands": [repr(c) for c in cmds],
            "ascii_minimap": render_state.get("minimap", ""),
            "signals": _jsonable(
                {
                    "cash": getattr(signals, "cash", 0),
                    "economy_value": getattr(signals, "cash", 0)
                    + getattr(signals, "resources", 0),
                    "explored_percent": round(
                        getattr(signals, "explored_percent", 0.0), 2
                    ),
                    "units_killed": getattr(signals, "units_killed", 0),
                    "units_lost": getattr(signals, "units_lost", 0),
                    "enemies_seen": len(getattr(signals, "enemies_seen_ids", [])),
                }
            ),
            "units": render_state.get("units_summary", []),
            "enemies": render_state.get("enemy_summary", []),
            # Per-turn goal tracker: win-condition leaf progress AND the
            # normalized cumulative reward vector, side by side.
            "goal": goal or {},
        }
        self._turns_fh.write(json.dumps(_jsonable(rec)) + "\n")
        self._turns_fh.flush()

    def write_messages(self, history: list[dict]) -> None:
        """Full model⇄env transcript (ModelAgent.history) — system /
        user (briefing + minimap data-URL) / assistant / tool."""
        (self.dir / "messages.json").write_text(
            json.dumps(_jsonable(history), indent=2)
        )

    def finalize(self, manifest: dict) -> None:
        (self.dir / "manifest.json").write_text(
            json.dumps(_jsonable(manifest), indent=2)
        )
        try:
            self._turns_fh.close()
        except Exception:  # noqa: BLE001
            pass
