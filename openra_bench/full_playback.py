"""Audit-ready per-cell data capture (Phase 4 paper-collection format).

This is the **audit format** used by `scripts/collect_eval_data.py`. It
is additive over `openra_bench.playback.Playback`: that one stays the
inspect-by-human format (legible per-turn record, terse signals); this
one captures EVERYTHING needed to forensically replay or re-score a
run after the fact — full obs (including `_raw` and `spatial`), the
exact briefing the model saw, the system prompt, the literal HTTP
request body sent to the provider, the literal response (content +
tool_calls + finish_reason + usage), engine warnings, and a `terminal`
block on the final turn (outcome, final_obs, wall-clock, tokens).

Layout: one JSONL per (model, pack, level, seed, fog_mode) cell at

    <root>/<pack>__<level>__seed<N>__<fog_mode>.jsonl

PNG minimaps go alongside in a sibling dir of the same stem:

    <root>/<pack>__<level>__seed<N>__<fog_mode>/turn_<N>.png

The JSONL line for a turn refs the PNG by relative path (relative to
the JSONL file's parent). A `terminal:` field on the final line marks
the episode complete — `scripts/collect_eval_data.py --resume` uses
that marker to skip cells that already finished cleanly.

Why a separate writer (vs extending Playback): the legacy Playback
format is what `scripts/view_playback.py` reads and what `run_eval.py`
emits today; rewriting it would invalidate every existing playback
dir and break the viewer for ~1000 historical episodes. FullPlayback
runs ALONGSIDE Playback when both are configured; either can be
disabled independently.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(o: Any) -> Any:
    """Recursive serializer matching playback._jsonable but tolerant of
    bytes (base64'd) and tuples (preserved as lists). Bytes are wrapped
    as `{"__b64__": <base64-string>}` so a round-trip recovers them."""
    if isinstance(o, bytes):
        return {"__b64__": base64.b64encode(o).decode("ascii")}
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, set):
        return sorted(_jsonable(v) for v in o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def _safe(s: str) -> str:
    """Slug a model id / pack id for path use: keep alnum, `.`, `_`, `-`;
    everything else (especially `/`) becomes `_`."""
    out = []
    for ch in s:
        out.append(ch if (ch.isalnum() or ch in "._-") else "_")
    return "".join(out)


def cell_stem(pack_id: str, level: str, seed: int, fog_mode: str) -> str:
    return f"{_safe(pack_id)}__{_safe(level)}__seed{int(seed)}__{_safe(fog_mode)}"


class FullPlayback:
    """Per-cell audit-ready JSONL + PNG writer.

    Construct one per (pack, level, seed, fog_mode) cell. Call
    `record_turn(...)` once per model turn (mirroring how the legacy
    Playback is driven from `run_level`); call `finalize(...)` on
    episode end (it emits the `terminal:` field merged into the last
    turn line, NOT a new line — so the file is one-line-per-turn and
    the terminal marker is unambiguous).

    Concurrency: one cell == one subprocess in the collector, so no
    cross-cell locking is needed. Within a process, this class is NOT
    thread-safe (the eval loop is single-threaded per episode).
    """

    def __init__(
        self,
        root: str | Path,
        pack_id: str,
        level: str,
        seed: int,
        fog_mode: str,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stem = cell_stem(pack_id, level, seed, fog_mode)
        self.jsonl_path = self.root / f"{self.stem}.jsonl"
        self.png_dir = self.root / self.stem
        self.png_dir.mkdir(parents=True, exist_ok=True)
        # Use a sidecar tmp until finalize, then move atomically over the
        # final path so a `--resume` scan only sees complete files.
        self._tmp_path = self.root / f"{self.stem}.jsonl.partial"
        self._fh = open(self._tmp_path, "w")
        self.pack_id = pack_id
        self.level = level
        self.seed = seed
        self.fog_mode = fog_mode
        self._t0 = time.time()
        # Buffer the last turn line so finalize() can merge `terminal:`
        # into it instead of writing a fresh trailing line (one line ==
        # one turn; the terminal block is a field on the last line).
        self._last_rec: dict | None = None
        # Token accounting across the whole episode (provider-reported
        # usage from each model call); the totals land in `terminal:`.
        self._tokens_in = 0
        self._tokens_out = 0
        # First turn carries the system_prompt; subsequent turns repeat
        # it as `null` to keep the per-turn record uniform but small.
        self._sysp_written = False

    # ── per-turn ──────────────────────────────────────────────────────

    def record_turn(
        self,
        *,
        turn: int,
        tick: int | None,
        obs: dict,
        briefing: str,
        system_prompt: str,
        model_request: dict | None,
        model_response: dict | None,
        commands_issued: list,
        engine_warnings: list[str],
        signals: Any,
        minimap_png_b64: str | None = None,
        done: bool = False,
        interrupt: str | None = None,
        extra: dict | None = None,
    ) -> None:
        # Flush the previously-buffered line now that we know we're past
        # it (a new turn started, so the prior turn was NOT terminal).
        if self._last_rec is not None:
            self._fh.write(json.dumps(_jsonable(self._last_rec)) + "\n")
            self._fh.flush()
            self._last_rec = None

        # Track per-call tokens (provider returns them in response.usage)
        u = (model_response or {}).get("usage") or {}
        self._tokens_in += int(u.get("prompt_tokens", 0) or 0)
        self._tokens_out += int(u.get("completion_tokens", 0) or 0)

        png_rel: str | None = None
        if minimap_png_b64:
            try:
                png_path = self.png_dir / f"turn_{int(turn):03d}.png"
                png_path.write_bytes(base64.b64decode(minimap_png_b64))
                # Relative to the JSONL's parent dir so the file is
                # portable (you can move the run dir without breaking
                # refs).
                png_rel = f"{self.stem}/turn_{int(turn):03d}.png"
            except Exception:  # noqa: BLE001 — never break a run on I/O
                png_rel = None

        rec: dict = {
            "turn": int(turn),
            "tick": tick,
            "interrupt": interrupt,
            "obs": _jsonable(obs),
            "briefing": briefing,
            "system_prompt": system_prompt if not self._sysp_written else None,
            "model_request": _jsonable(model_request) if model_request else None,
            "model_response": _jsonable(model_response) if model_response else None,
            "commands_issued": [repr(c) for c in commands_issued],
            "engine_warnings": list(engine_warnings or []),
            "signals": _jsonable(_signal_snapshot(signals)),
            "minimap_png": png_rel,
            "done": bool(done),
        }
        if extra:
            rec["extra"] = _jsonable(extra)
        self._sysp_written = True
        # Buffer; finalize() merges `terminal:` into this line.
        self._last_rec = rec

    def finalize(
        self,
        *,
        outcome: str,
        final_obs: dict | None,
        manifest_extra: dict | None = None,
    ) -> None:
        """Stamp the buffered last line with the `terminal:` block,
        flush, and atomically move the partial file over the final
        path so `--resume` sees a fully-complete cell."""
        wall = round(time.time() - self._t0, 3)
        terminal = {
            "outcome": outcome,
            "final_obs": _jsonable(final_obs) if final_obs is not None else None,
            "wall_clock_seconds": wall,
            "total_tokens_in": int(self._tokens_in),
            "total_tokens_out": int(self._tokens_out),
        }
        if manifest_extra:
            terminal["manifest"] = _jsonable(manifest_extra)

        if self._last_rec is None:
            # Episode produced zero turns (engine crashed on reset, say).
            # Emit a synthetic terminal-only record so `--resume` can
            # still detect "this cell was attempted and completed".
            self._last_rec = {
                "turn": 0,
                "tick": None,
                "interrupt": None,
                "obs": None,
                "briefing": "",
                "system_prompt": None,
                "model_request": None,
                "model_response": None,
                "commands_issued": [],
                "engine_warnings": [],
                "signals": {},
                "minimap_png": None,
                "done": True,
            }
        self._last_rec["terminal"] = terminal
        self._fh.write(json.dumps(_jsonable(self._last_rec)) + "\n")
        self._fh.flush()
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._tmp_path.replace(self.jsonl_path)
        except Exception:  # noqa: BLE001 — last-ditch fallback
            try:
                self.jsonl_path.write_text(self._tmp_path.read_text())
            except Exception:  # noqa: BLE001
                pass

    def abort(self) -> None:
        """Close without finalizing. The .partial file stays on disk so
        a post-hoc diagnostic can inspect what was captured before the
        crash; the final .jsonl is NOT created, so `--resume` will
        correctly retry this cell on the next invocation."""
        try:
            if self._last_rec is not None:
                self._fh.write(json.dumps(_jsonable(self._last_rec)) + "\n")
            self._fh.flush()
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass


def _signal_snapshot(signals: Any) -> dict:
    """Pull every primitive scalar / list off an EpisodeSignals (or
    duck-typed shim) into a JSON-safe dict. Defensive: a missing attr
    is just absent. Mirrors the existing playback shape so downstream
    tools recognise the same field names, but adds a few signals
    (resources, harvesters, tool_violations) that the existing playback
    omits."""
    fields = (
        "game_tick",
        "cash",
        "resources",
        "resource_capacity",
        "power_provided",
        "power_drained",
        "harvesters",
        "explored_percent",
        "units_killed",
        "units_lost",
        "enemies_seen_ids",
        "enemy_buildings_seen_ids",
        "production_items",
        "tool_violations",
        "outcome",
    )
    out: dict[str, Any] = {}
    for f in fields:
        if not hasattr(signals, f):
            continue
        v = getattr(signals, f)
        if isinstance(v, (set, frozenset)):
            v = sorted(_jsonable(x) for x in v)
        out[f] = _jsonable(v)
    # Convenience: counts + computed signals downstream uses heavily.
    if "enemies_seen_ids" in out and isinstance(out["enemies_seen_ids"], list):
        out["enemies_seen_count"] = len(out["enemies_seen_ids"])
    if "enemy_buildings_seen_ids" in out and isinstance(
        out["enemy_buildings_seen_ids"], list
    ):
        out["enemy_buildings_seen_count"] = len(out["enemy_buildings_seen_ids"])
    if "cash" in out and "resources" in out:
        try:
            out["economy_value"] = int(out["cash"]) + int(out["resources"])
        except (TypeError, ValueError):
            pass
    return out


def is_complete_cell(jsonl_path: str | Path) -> bool:
    """True iff `jsonl_path` exists, is non-empty, and the LAST line
    carries a `terminal:` field. The resume scanner uses this so a
    crash mid-cell (partial .jsonl with no terminal) is correctly
    retried, while a cleanly finished cell is correctly skipped.

    Reads only the file tail (~64KB) — safe to call on thousands of
    cells in a scan."""
    p = Path(jsonl_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    # Tail read: locate the last newline boundary so we can parse just
    # the trailing line without loading the whole file. The terminal
    # line carries the FULL final_obs (which on a long episode with
    # spatial tensors can be many MB), so we walk back from EOF until
    # we cross a newline rather than guessing a fixed window.
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            # Walk back in 256KB chunks until we hit a newline OR start.
            chunk = 256 * 1024
            buf = b""
            pos = size
            while pos > 0:
                step = min(chunk, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                # The last line begins right after the LAST newline in
                # buf (excluding a trailing newline). We need TWO
                # newlines (or start-of-file + one) to be sure we have
                # the complete last line.
                stripped = buf.rstrip(b"\n")
                nl = stripped.rfind(b"\n")
                if nl != -1:
                    last_bytes = stripped[nl + 1:]
                    break
                if pos == 0:
                    last_bytes = stripped
                    break
            last = last_bytes.decode("utf-8", errors="replace").strip()
        if not last:
            return False
        rec = json.loads(last)
    except (OSError, ValueError):
        return False
    return isinstance(rec, dict) and "terminal" in rec
