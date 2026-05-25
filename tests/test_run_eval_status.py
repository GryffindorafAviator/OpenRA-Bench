"""`python -m openra_bench.run_eval status --out <dir>` tests.

The status command is the "is the production sweep still alive"
read-only probe. It reads journals + score.json files; tolerates
partial/empty/corrupt journals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_journal(p: Path, header: dict | None, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if header is not None:
        lines.append(json.dumps(header))
    for r in rows:
        lines.append(json.dumps(r))
    p.write_text("\n".join(lines) + "\n")


def test_status_summary_extracts_basic_fields(tmp_path):
    from openra_bench.run_eval import _status_summary

    pb = tmp_path / "pb"
    _write_journal(
        pb / "_journal__mod.jsonl",
        header={"_meta": True, "run_id": "rA", "model": "mod",
                "code_version": "abc1234"},
        rows=[
            {"_key": "k1", "cell": "p:easy", "outcome": "win",
             "composite": 0.7, "seed": 1, "turns": 12},
            {"_key": "k2", "cell": "p:easy", "outcome": "loss",
             "composite": 0.2, "seed": 2, "turns": 30},
            {"_key": "k3", "cell": "p:hard", "outcome": "draw",
             "composite": 0.5, "seed": 1, "turns": 30},
            {"_key": "k4", "cell": "p:hard", "outcome": "error",
             "composite": 0.0, "seed": 2, "turns": 0},
        ],
    )
    snap = _status_summary(pb)
    assert snap["total_journaled"] == 4
    assert snap["outcomes"]["win"] == 1
    assert snap["outcomes"]["loss"] == 1
    assert snap["outcomes"]["draw"] == 1
    assert snap["outcomes"]["error"] == 1
    assert snap["header"]["run_id"] == "rA"
    assert snap["compose_mean"] > 0
    # Last cell points at the most recently appended row.
    assert snap["last_cell"]["cell"] == "p:hard"
    assert snap["last_cell"]["outcome"] == "error"


def test_status_handles_missing_dir(tmp_path):
    from openra_bench.run_eval import _status_summary

    snap = _status_summary(tmp_path / "nope")
    assert "error" in snap
    assert "does not exist" in snap["error"]


def test_status_handles_empty_dir(tmp_path):
    from openra_bench.run_eval import _status_summary

    pb = tmp_path / "pb"
    pb.mkdir()
    snap = _status_summary(pb)
    assert snap["total_journaled"] == 0
    assert snap["outcomes"] == {"win": 0, "loss": 0, "draw": 0, "error": 0}
    assert snap["header"] is None


def test_status_handles_torn_last_line(tmp_path):
    from openra_bench.run_eval import _status_summary

    pb = tmp_path / "pb"
    pb.mkdir()
    jp = pb / "_journal__mod.jsonl"
    jp.write_text(
        json.dumps({"_meta": True, "run_id": "rA", "model": "mod"}) + "\n"
        + json.dumps({"_key": "k1", "cell": "p:easy", "outcome": "win",
                      "composite": 0.5, "seed": 1, "turns": 10}) + "\n"
        + '{"_key": "torn"'  # no newline / invalid JSON
    )
    snap = _status_summary(pb)
    # Torn tail tolerated; only the well-formed data row counted.
    assert snap["total_journaled"] == 1
    assert snap["outcomes"]["win"] == 1


def test_status_finds_score_json_on_disk(tmp_path):
    from openra_bench.run_eval import _status_summary

    pb = tmp_path / "pb"
    _write_journal(
        pb / "_journal__mod.jsonl",
        header={"_meta": True, "run_id": "rA", "model": "mod"},
        rows=[
            {"_key": "k1", "cell": "p:easy", "outcome": "win",
             "composite": 0.7, "seed": 1, "turns": 12},
        ],
    )
    # Drop a score.json under the canonical <run_id>__<model>/ tree.
    sc_dir = pb / "rA__mod" / "p_easy_public__seed1"
    sc_dir.mkdir(parents=True)
    (sc_dir / "score.json").write_text('{"outcome": "win"}')
    snap = _status_summary(pb)
    assert snap["scores_on_disk"] == 1


def test_status_format_includes_all_required_fields(tmp_path):
    from openra_bench.run_eval import _status_summary, _format_status

    pb = tmp_path / "pb"
    _write_journal(
        pb / "_journal__mod.jsonl",
        header={"_meta": True, "run_id": "rA", "model": "mod",
                "code_version": "abc1234"},
        rows=[
            {"_key": "k1", "cell": "p:easy", "outcome": "win",
             "composite": 0.7, "seed": 1, "turns": 12},
            {"_key": "k2", "cell": "p:easy", "outcome": "loss",
             "composite": 0.2, "seed": 2, "turns": 30},
        ],
    )
    txt = _format_status(_status_summary(pb))
    for field in (
        "Run dir:", "Started:", "Cells:", "Outcomes:",
        "Avg compose:", "Last cell:",
    ):
        assert field in txt, f"missing {field!r} in:\n{txt}"


def test_status_main_exit_zero(tmp_path, capsys):
    from openra_bench.run_eval import _status_main

    pb = tmp_path / "pb"
    _write_journal(
        pb / "_journal__mod.jsonl",
        header={"_meta": True, "run_id": "rA", "model": "mod"},
        rows=[
            {"_key": "k1", "cell": "p:easy", "outcome": "win",
             "composite": 0.7, "seed": 1, "turns": 12},
        ],
    )
    rc = _status_main(["--out", str(pb)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "p:easy" in out
