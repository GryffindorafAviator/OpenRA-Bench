"""Playback persistence (pipeline step 7).

Saves, per scenario episode, everything needed to *replay what
happened*: the full model⇄env message transcript (system / user /
assistant, including the minimap image as a data-URL), a per-turn
record (tick, commands issued, signal snapshot, units, enemies, and goal progress), and a
manifest with scenario meta + outcome + score. Written under a
dedicated folder so people can inspect every run.

Layout:
  <root>/<pack:level>/seed<seed>[_rep<R>]/
      manifest.json      scenario meta, outcome, scorecard, agent stats
      turns.jsonl        one JSON object per turn
      messages.json      full chat transcript (ModelAgent only)
      minimap_turnNN.png rendered minimap per turn (when available)

The `_rep<R>` suffix is appended ONLY when `repeat > 0` so existing
per-cell layouts (no rep suffix) stay readable by the viewer. Pass^N
stability sweeps pass `repeat=R` for every replay so per-rep
transcripts don't collide on disk.

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
        repeat: int = 0,
    ):
        safe_cell = "".join(
            ch if ch not in '<>:"/\\|?*' and ord(ch) >= 32 else "_"
            for ch in str(cell)
        )
        # Remember the per-session root (e.g.
        # `<playback>/run-<ts>__<player>`) so cleanup after discard can
        # walk up to (and remove) the run dir without straying past it
        # into the user's playback root.
        self._root_dir = Path(root)
        self.seed = int(seed)
        self.repeat = int(repeat)
        # `_rep<R>` suffix is appended ONLY when repeat > 0 so existing
        # bare-seed dirs (pre-bug-fix data and single-rep runs) stay at
        # their canonical paths. Pass^N reps each get a distinct dir so
        # rep transcripts no longer overwrite each other (PR fix).
        seed_dirname = f"seed{seed}" if self.repeat == 0 else (
            f"seed{seed}_rep{self.repeat}"
        )
        self.final_dir = self._root_dir / safe_cell / seed_dirname
        if draft:
            # Stage to a sibling .draft/<safe_cell>/seedN so the real
            # final path stays absent on disk until promote() runs.
            self.dir = (
                self._root_dir / ".draft" / safe_cell / seed_dirname
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
        # Close the per-turn JSONL handle BEFORE moving. On Windows /
        # NFS the open handle would either block the move outright or
        # leave the file unreadable at the destination; closing here
        # is a no-op on POSIX where finalize() already closed it.
        try:
            self._turns_fh.close()
        except Exception:  # noqa: BLE001
            pass
        # Ensure parent of the final dir exists; replace any stale
        # final dir (e.g. a prior crashed commit) so the new draft
        # wins.
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            # ignore_errors=True so a partial rmtree (permission /
            # lock on a sub-file) can't wedge promote() into raising —
            # the subsequent move will either succeed (rmtree cleared
            # the leaf) or fail with a clearer FileExistsError than
            # PermissionError mid-walk.
            shutil.rmtree(self.final_dir, ignore_errors=True)
        try:
            shutil.move(str(self.dir), str(self.final_dir))
        except FileExistsError:
            # Race with another writer that re-created final_dir
            # between our rmtree and our move. Best-effort: copy the
            # contents over and drop the draft. This is the only
            # branch where promote() can silently merge state, but
            # it's far better than leaving the draft and the final
            # both on disk and the caller wedged.
            self.final_dir.mkdir(parents=True, exist_ok=True)
            for entry in self.dir.iterdir():
                target = self.final_dir / entry.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target, ignore_errors=True)
                    else:
                        try:
                            target.unlink()
                        except OSError:
                            pass
                try:
                    shutil.move(str(entry), str(target))
                except OSError:
                    pass
            shutil.rmtree(self.dir, ignore_errors=True)
        # Clean now-empty parents of the .draft/ tree.
        self._cleanup_empty_draft_parents()
        self.dir = self.final_dir
        self.is_draft = False
        return self.final_dir

    def _cleanup_empty_draft_parents(self) -> None:
        """Walk up from `self.dir`'s parent removing empty dirs. The
        walk stops at (but is allowed to delete) the per-session root
        `self._root_dir` (e.g. `<playback>/run-<ts>__<player>`); it
        never strays above the session root into the shared playback
        root. Without this safeguard a sparsely-populated draft tree
        could either (a) leave an empty `run-…/` carcass behind after
        Discard — the user-visible bug — or (b) collapse all the way
        up and erase the caller's playback root."""
        try:
            parent = self.dir.parent
            while parent.exists() and not any(parent.iterdir()):
                is_session_root = (
                    self._root_dir is not None
                    and parent.resolve() == self._root_dir.resolve()
                )
                parent.rmdir()
                if is_session_root:
                    # Don't ascend past the session root.
                    break
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
