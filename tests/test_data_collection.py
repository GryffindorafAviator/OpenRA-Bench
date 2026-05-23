"""Phase 4 data-collection plumbing: audit-format JSONL + the
`scripts/collect_eval_data.py` driver.

Three smoke tests that don't need a Together API key:

* `test_dry_run_plan_is_valid` — the planner expands the right cells
  and a `--dry-run` invocation produces a parseable plan sidecar.
* `test_full_playback_writes_audit_jsonl` — a live 1-cell run with a
  fake-model controller produces a valid JSONL whose last line carries
  a `terminal:` block, plus the per-turn minimap PNGs.
* `test_resume_skips_completed_cell` — `is_complete_cell` correctly
  detects a finished cell so `--resume` skips it.

All tests use the bench's scripted controllers; no network / no
provider call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")
pytest.importorskip("openra_rl_training", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.full_playback import (
    FullPlayback,
    cell_stem,
    is_complete_cell,
)
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

# Smallest deterministic pack the rest of the suite already exercises.
PACK = PACKS_DIR / "perception-frontier-reading.yaml"


# ── Planner / CLI dry-run ────────────────────────────────────────────


def test_dry_run_plan_is_valid(tmp_path: Path):
    """The collector script's --dry-run produces a sidecar plan JSON
    that lists exactly the cells implied by the inputs, without
    spawning any subprocess."""
    out = tmp_path / "runs" / "smoke"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "collect_eval_data.py"),
        "--models",
        "Qwen/Qwen3.5-9B,google/gemma-4-31B-it",
        "--packs",
        "perception-frontier-reading,action-multiunit-coordination",
        "--levels",
        "easy,medium",
        "--seeds",
        "1,2",
        "--fog-modes",
        "vision,structured",
        "--run-label",
        "smoke",
        "--output-dir",
        str(out),
        "--dry-run",
    ]
    rc = subprocess.call(cmd)
    assert rc == 0, "dry-run must exit 0"
    sidecar = out / "_dry_run_plan.json"
    assert sidecar.exists(), "dry-run sidecar plan should be written"
    data = json.loads(sidecar.read_text())
    # 2 models * 2 packs * 2 levels * 2 seeds * 2 fogs = 32 cells
    assert len(data["plan"]) == 32, (
        f"expected 32 cells in plan, got {len(data['plan'])}"
    )
    # Cost lines must be present and finite.
    assert data["cost"]["total_usd"] > 0
    assert data["cost"]["total_cells"] == 32


# ── Audit JSONL shape ────────────────────────────────────────────────


class _ScriptedAgent:
    """Stand-in ModelAgent: exposes `history` (so legacy playback works)
    AND the audit-capture attrs (`audit_capture`, `last_briefing`,
    `last_request`, `last_response`, `system_prompt`) so a FullPlayback
    line is exercised end-to-end without a network call."""

    def __init__(self):
        self.history = [{"role": "system", "content": "SYS"}]
        self.stats = {"turns": 0, "tool_calls": 0, "empty_replies": 0}
        # FullPlayback contract.
        self.audit_capture = False
        self.system_prompt = "FAKE SYSTEM PROMPT"
        self.last_briefing = ""
        self.last_request: dict | None = None
        self.last_response: dict | None = None

    def agent_fn(self, render_state, Command):
        self.stats["turns"] += 1
        self.last_briefing = (
            f"turn {self.stats['turns']}: tick={render_state.get('game_tick')}"
        )
        self.last_request = {
            "url": "https://fake/api",
            "body": {"model": "fake", "messages": []},
        }
        self.last_response = {
            "raw": {"id": f"fake-{self.stats['turns']}"},
            "text": "stalling",
            "tool_calls": [{"name": "observe", "arguments": {}}],
            "reasoning": "",
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            "finish_reason": "stop",
        }
        return [Command.observe()]


def test_full_playback_writes_audit_jsonl(tmp_path: Path):
    c = compile_level(load_pack(PACK), "easy")
    agent = _ScriptedAgent()
    fp = FullPlayback(
        tmp_path, pack_id=c.pack_id, level="easy", seed=1, fog_mode="vision"
    )
    res = run_level(c, agent.agent_fn, seed=1, playback=None, full_playback=fp)

    jsonl = tmp_path / f"{cell_stem(c.pack_id, 'easy', 1, 'vision')}.jsonl"
    assert jsonl.exists(), "audit JSONL must land at the canonical stem"
    lines = [json.loads(x) for x in jsonl.read_text().splitlines() if x]
    assert lines, "audit JSONL must have at least one line"
    # One line per model turn (no separate terminal frame — the terminal
    # block merges into the LAST turn line).
    assert len(lines) == res.turns, (
        f"expected {res.turns} lines, got {len(lines)}"
    )
    first = lines[0]
    # System prompt only on turn 1.
    assert first["system_prompt"] == "FAKE SYSTEM PROMPT"
    if len(lines) > 1:
        assert lines[1]["system_prompt"] is None
    # Full obs must be present (not truncated).
    assert isinstance(first["obs"], dict) and "minimap" in first["obs"]
    # Briefing + model request/response captured.
    assert first["briefing"].startswith("turn 1")
    assert first["model_request"]["url"] == "https://fake/api"
    assert first["model_response"]["text"] == "stalling"
    # Engine warnings carried through (may be empty list — that's fine).
    assert isinstance(first["engine_warnings"], list)
    # Commands captured as repr strings (audit format).
    assert first["commands_issued"] and "Observe" in first["commands_issued"][0]
    # Last line has the terminal block with totals.
    term = lines[-1]["terminal"]
    assert term["outcome"] in {"win", "loss", "draw"}
    assert term["wall_clock_seconds"] >= 0
    # Token totals are simply summed from the per-turn responses (5 out per
    # turn in the fake → res.turns * 5).
    assert term["total_tokens_out"] == res.turns * 5

    # is_complete_cell sees the terminal marker.
    assert is_complete_cell(jsonl) is True


# ── Resume detection ─────────────────────────────────────────────────


def test_resume_skips_completed_cell(tmp_path: Path):
    """`is_complete_cell` returns True for a JSONL whose last line has a
    `terminal:` block, False otherwise. This is what `--resume` relies
    on to skip cells without re-spawning their subprocess."""
    # A finished cell.
    done = tmp_path / "done.jsonl"
    done.write_text(
        json.dumps({"turn": 1, "tick": 93, "obs": {}}) + "\n"
        + json.dumps(
            {
                "turn": 2,
                "tick": 183,
                "obs": {},
                "terminal": {
                    "outcome": "win",
                    "wall_clock_seconds": 1.0,
                    "total_tokens_in": 0,
                    "total_tokens_out": 0,
                },
            }
        )
        + "\n"
    )
    assert is_complete_cell(done) is True

    # An interrupted cell: no terminal on the last line.
    half = tmp_path / "half.jsonl"
    half.write_text(
        json.dumps({"turn": 1, "tick": 93, "obs": {}}) + "\n"
        + json.dumps({"turn": 2, "tick": 183, "obs": {}}) + "\n"
    )
    assert is_complete_cell(half) is False

    # An empty file / missing file.
    assert is_complete_cell(tmp_path / "missing.jsonl") is False
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert is_complete_cell(empty) is False
