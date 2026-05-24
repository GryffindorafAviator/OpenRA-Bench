"""Playback persistence (pipeline step 7).

Saves, per scenario episode, everything needed to *replay what
happened*: the full model⇄env message transcript (system / user /
assistant, including the minimap image as a data-URL), a per-turn
record (tick, commands issued, signal snapshot, units, enemies, and goal progress), and a
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
    """Per-episode recorder. Create one per (scenario, seed).

    When `draft=True`, all writes go to a `.draft/` sibling of the
    intended target. `final_dir` records where `promote()` will move
    the staged dir on commit. This is the human-review-before-save
    path used by the manual-play UI; existing automated callers
    (run_level, etc.) get the legacy direct-write behaviour
    untouched.
    """

    def __init__(
        self,
        root: str | Path,
        cell: str,
        seed: int,
        draft: bool = False,
    ):
        safe_cell = "".join(
            ch if ch not in '<>:"/\\|?*' and ord(ch) >= 32 else "_"
            for ch in str(cell)
        )
        self.final_dir = Path(root) / safe_cell / f"seed{seed}"
        if draft:
            # Stage to a sibling .draft/<safe_cell>/seedN so the real
            # final path stays absent on disk until promote() runs.
            self.dir = (
                Path(root) / ".draft" / safe_cell / f"seed{seed}"
            )
        else:
            self.dir = self.final_dir
        self.is_draft = bool(draft)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._turns_fh = open(self.dir / "turns.jsonl", "w")
        self._n = 0
        # Set by run_eval so manifests carry run/model identity for the
        # viewer's run → model → scenario filter.
        self.run_id: str | None = None
        self.model: str | None = None

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
                if "," in minimap_png_b64 and minimap_png_b64.lstrip().startswith("data:"):
                    minimap_png_b64 = minimap_png_b64.split(",", 1)[1]
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

    def promote(self) -> Path:
        """Move a draft Playback dir to its real final location.
        Returns the final path. Idempotent on already-promoted dirs.
        Only valid for draft Playbacks (`is_draft=True`)."""
        import shutil

        if not self.is_draft:
            return self.dir
        if self.dir == self.final_dir:
            return self.final_dir
        if not self.dir.exists():
            # Nothing staged — caller already promoted or never finalized.
            return self.final_dir
        # Ensure parent of the final dir exists; replace any stale
        # final dir (e.g. a prior crashed commit) so the new draft
        # wins.
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            shutil.rmtree(self.final_dir)
        shutil.move(str(self.dir), str(self.final_dir))
        # Clean now-empty parents of the .draft/ tree.
        self._cleanup_empty_draft_parents()
        self.dir = self.final_dir
        self.is_draft = False
        return self.final_dir

    def _cleanup_empty_draft_parents(self) -> None:
        """Walk up from `self.dir`'s parent removing empty dirs, BUT
        stop at (and never delete) the `.draft/` boundary or higher.
        Without this safeguard a sparsely-populated draft tree could
        collapse all the way up to the playback root and erase the
        caller's tmp_path."""
        try:
            parent = self.dir.parent
            while parent.exists() and not any(parent.iterdir()):
                # Stop once we'd remove the `.draft` boundary itself —
                # the playback root is not ours to delete.
                if parent.name == ".draft":
                    parent.rmdir()
                    break
                parent.rmdir()
                parent = parent.parent
        except OSError:
            pass

    def discard(self) -> None:
        """Delete a draft Playback dir without promoting it. Safe to
        call multiple times. Only valid for draft Playbacks."""
        import shutil

        if not self.is_draft:
            return
        try:
            self._turns_fh.close()
        except Exception:  # noqa: BLE001
            pass
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self._cleanup_empty_draft_parents()
