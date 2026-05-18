"""Pipeline step 7: per-episode playback persistence — transcript
(incl. minimap), per-turn record, manifest, score — so runs are
inspectable. Additive: default (no playback) changes nothing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("openra_train", reason="Rust env wheel not installed")

from openra_bench.eval_core import run_level
from openra_bench.playback import Playback
from openra_bench.run_eval import evaluate
from openra_bench.scenarios import load_pack
from openra_bench.scenarios.loader import PACKS_DIR, compile_level

PACK = PACKS_DIR / "perception-frontier-reading.yaml"


class _AgentWithHistory:
    """Stand-in ModelAgent: a bound method exposes the instance, whose
    .history/.stats run_level must persist."""

    def __init__(self):
        self.history = [{"role": "system", "content": "SYS"}]
        self.stats = {"turns": 0, "tool_calls": 0}

    def agent_fn(self, render_state, Command):
        self.stats["turns"] += 1
        self.history.append({"role": "user", "content": "briefing"})
        self.history.append({"role": "assistant", "content": "act"})
        return [Command.observe()]


def test_run_level_playback_writes_transcript_and_manifest(tmp_path):
    c = compile_level(load_pack(PACK), "easy")
    agent = _AgentWithHistory()
    pb = Playback(tmp_path, c.pack_id + ":easy", 1)
    res = run_level(c, agent.agent_fn, seed=1, playback=pb)

    d = pb.dir
    assert (d / "turns.jsonl").exists()
    lines = [
        json.loads(x) for x in (d / "turns.jsonl").read_text().splitlines() if x
    ]
    assert len(lines) == res.turns
    assert "ascii_minimap" in lines[0] and "commands" in lines[0]

    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["outcome"] == res.outcome
    assert manifest["turns"] == res.turns
    assert manifest["capability"] == c.meta.capability
    assert manifest["agent_stats"]["turns"] == res.turns

    msgs = json.loads((d / "messages.json").read_text())
    assert msgs[0]["role"] == "system"
    assert any(m["role"] == "assistant" for m in msgs)


def test_run_level_without_playback_is_unchanged(tmp_path):
    c = compile_level(load_pack(PACK), "easy")
    run_level(c, lambda rs, C: [C.observe()], seed=1)  # playback=None
    assert not any(tmp_path.iterdir())  # nothing written


def test_run_eval_playback_tree_with_score(tmp_path):
    out = evaluate(
        packs=[PACK],
        levels=["easy"],
        seeds=[1],
        playback_root=tmp_path,
    )
    assert out["overall"]["n"] == 1
    # <root>/<pack:level:public>/seed1/{manifest,turns,score}.json
    seed_dirs = list(tmp_path.glob("*/seed1"))
    assert len(seed_dirs) == 1
    sd = seed_dirs[0]
    for f in ("manifest.json", "turns.jsonl", "score.json"):
        assert (sd / f).exists(), f"missing {f}"
    score = json.loads((sd / "score.json").read_text())
    assert "composite" in score and "weakest_link" in score
