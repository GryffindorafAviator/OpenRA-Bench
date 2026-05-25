"""Production-eval hardening tests (v11 sweep audit fixes).

These exercise the three new resilience knobs:
  * `--strict-resume` — verify journal ↔ on-disk `score.json`
  * `RunJournal` in-memory `_key` dedupe
  * `_meta` header validation (matching `run_id` required for resume)

All deterministic — no engine, no network, no provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openra_bench.resilience import (
    DuplicateJournalKey,
    JournalRunIdMismatch,
    RunJournal,
    episode_key,
)


# ── dedupe ─────────────────────────────────────────────────────────────────


def test_journal_rejects_duplicate_key_within_process(tmp_path):
    """Appending the same `_key` twice within a single process must
    raise `DuplicateJournalKey`. This catches the v1.0 sweep footgun
    where `adversarial-duel:easy` ended up in the journal twice."""
    j = RunJournal(tmp_path / "j.jsonl", run_id="r1", model="m")
    k = episode_key("p", "easy", "public", 1)
    j.append(k, {"cell": "p:easy", "outcome": "win"})
    with pytest.raises(DuplicateJournalKey):
        j.append(k, {"cell": "p:easy", "outcome": "win"})


def test_journal_resume_seeds_dedupe_from_existing_rows(tmp_path):
    """A re-opened journal seeds its in-memory `_key` set from the
    on-disk records, so a resume can't accidentally re-append a key
    the prior process already wrote (the v1.0 mismatch class)."""
    p = tmp_path / "j.jsonl"
    j1 = RunJournal(p, run_id="r1", model="m")
    k = episode_key("p", "easy", "public", 1)
    j1.append(k, {"cell": "p:easy", "outcome": "win"})

    # Re-open the same file: the dedupe set must contain the prior key.
    j2 = RunJournal(p, run_id="r1", model="m")
    with pytest.raises(DuplicateJournalKey):
        j2.append(k, {"cell": "p:easy", "outcome": "win"})


# ── header / run_id ────────────────────────────────────────────────────────


def test_journal_writes_meta_header_on_first_append(tmp_path):
    p = tmp_path / "j.jsonl"
    j = RunJournal(p, run_id="rA", model="mod", code_version="abc1234")
    j.append(episode_key("p", "easy", "public", 1),
             {"cell": "p:easy", "outcome": "win"})
    first = p.read_text().splitlines()[0]
    meta = json.loads(first)
    assert meta["_meta"] is True
    assert meta["run_id"] == "rA"
    assert meta["model"] == "mod"
    assert meta["code_version"] == "abc1234"
    hdr = j.header()
    assert hdr is not None and hdr["run_id"] == "rA"


def test_journal_run_id_mismatch_raises(tmp_path):
    """A journal whose header was written by run_id=A cannot be
    re-opened with run_id=B unless `ignore_run_id=True`."""
    p = tmp_path / "j.jsonl"
    j1 = RunJournal(p, run_id="rA", model="m")
    j1.append(episode_key("p", "easy", "public", 1),
              {"cell": "p:easy", "outcome": "win"})

    # Same run_id: fine.
    RunJournal(p, run_id="rA", model="m")

    # Different run_id: hard error unless ignored.
    with pytest.raises(JournalRunIdMismatch):
        RunJournal(p, run_id="rB", model="m")
    # ignore_run_id is the explicit "I am merging two runs" knob.
    RunJournal(p, run_id="rB", model="m", ignore_run_id=True)


def test_journal_done_keys_skips_meta_line(tmp_path):
    p = tmp_path / "j.jsonl"
    j = RunJournal(p, run_id="rA", model="m")
    k = episode_key("p", "easy", "public", 1)
    j.append(k, {"cell": "p:easy", "outcome": "win"})
    # Header line must not be counted as a done key.
    assert j.done_keys() == {k}
    # `records()` strips the meta line.
    assert [r["_key"] for r in j.records()] == [k]


# ── strict resume gate (unit; engine not required) ─────────────────────────


def _make_journaled_run(playback_root: Path, run_id: str, model: str,
                        cell: str, split: str, seed: int, outcome: str) -> Path:
    """Build a fake playback dir layout so `_strict_resume_gate` has
    something to glob. Returns the `score.json` path so tests can
    delete / mutate it."""
    import re as _re

    safe_model = _re.sub(r"[^A-Za-z0-9._-]+", "_", model)
    safe_cell = _re.sub(r"[^A-Za-z0-9._-]+", "_", f"{cell}:{split}")
    dir_ = playback_root / f"{run_id}__{safe_model}" / f"{safe_cell}__seed{seed}"
    dir_.mkdir(parents=True, exist_ok=True)
    sc = dir_ / "score.json"
    sc.write_text(json.dumps({
        "outcome": outcome, "composite": 0.5,
    }))
    return sc


def test_strict_resume_drops_orphan_journal_entries(tmp_path):
    from openra_bench.run_eval import _strict_resume_gate

    pb = tmp_path / "pb"
    pb.mkdir()
    run_id = "rA"
    model = "mod"
    # Two journal entries; ONLY the first has a matching score.json.
    sc1 = _make_journaled_run(
        pb, run_id, model, "p:easy", "public", 1, "win",
    )
    assert sc1.exists()
    prior = [
        {"_key": episode_key("p", "easy", "public", 1),
         "cell": "p:easy", "split": "public", "seed": 1, "outcome": "win"},
        # No score.json on disk for this one — orphan.
        {"_key": episode_key("p", "easy", "public", 2),
         "cell": "p:easy", "split": "public", "seed": 2, "outcome": "win"},
    ]
    done, kept, stale = _strict_resume_gate(
        journal=None, prior=prior, playback_root=pb,
        run_id=run_id, safe_model=model,
    )
    assert len(kept) == 1
    assert episode_key("p", "easy", "public", 1) in done
    assert episode_key("p", "easy", "public", 2) not in done
    assert len(stale) == 1
    assert "missing score.json" in stale[0]


def test_strict_resume_drops_disagreeing_outcomes(tmp_path):
    from openra_bench.run_eval import _strict_resume_gate

    pb = tmp_path / "pb"
    pb.mkdir()
    run_id = "rA"
    model = "mod"
    sc = _make_journaled_run(
        pb, run_id, model, "p:easy", "public", 1, outcome="loss",
    )
    # Journal says win; score.json says loss. Strict gate must drop.
    prior = [
        {"_key": episode_key("p", "easy", "public", 1),
         "cell": "p:easy", "split": "public", "seed": 1, "outcome": "win"},
    ]
    done, kept, stale = _strict_resume_gate(
        journal=None, prior=prior, playback_root=pb,
        run_id=run_id, safe_model=model,
    )
    assert kept == [] and done == set()
    assert len(stale) == 1
    assert "disagrees" in stale[0]
    # Disk side untouched by the gate (it just dropped the journal row).
    assert sc.exists()


def test_strict_resume_keeps_agreeing_cells(tmp_path):
    from openra_bench.run_eval import _strict_resume_gate

    pb = tmp_path / "pb"
    pb.mkdir()
    run_id = "rA"
    model = "mod"
    _make_journaled_run(pb, run_id, model, "p:easy", "public", 1, "win")
    _make_journaled_run(pb, run_id, model, "p:easy", "public", 2, "loss")
    prior = [
        {"_key": episode_key("p", "easy", "public", 1),
         "cell": "p:easy", "split": "public", "seed": 1, "outcome": "win"},
        {"_key": episode_key("p", "easy", "public", 2),
         "cell": "p:easy", "split": "public", "seed": 2, "outcome": "loss"},
    ]
    done, kept, stale = _strict_resume_gate(
        journal=None, prior=prior, playback_root=pb,
        run_id=run_id, safe_model=model,
    )
    assert len(kept) == 2
    assert len(done) == 2
    assert stale == []


def test_strict_resume_drops_corrupt_score_json(tmp_path):
    from openra_bench.run_eval import _strict_resume_gate

    pb = tmp_path / "pb"
    pb.mkdir()
    run_id = "rA"
    model = "mod"
    sc = _make_journaled_run(
        pb, run_id, model, "p:easy", "public", 1, "win",
    )
    sc.write_text("{not json")  # corrupt
    prior = [
        {"_key": episode_key("p", "easy", "public", 1),
         "cell": "p:easy", "split": "public", "seed": 1, "outcome": "win"},
    ]
    done, kept, stale = _strict_resume_gate(
        journal=None, prior=prior, playback_root=pb,
        run_id=run_id, safe_model=model,
    )
    assert kept == [] and done == set()
    assert len(stale) == 1 and "corrupt" in stale[0]


# ── adaptive concurrency ───────────────────────────────────────────────────


def test_adaptive_concurrency_halves_on_error_burst(capsys):
    """Inject errors over the first 20 cells and confirm the pool
    halves. We can't probe `cap` directly because it's a local — but
    the stderr log line is the documented signal."""
    from openra_bench.run_eval import _run_adaptive_pool

    # 25 tasks; first 20 return "error", rest return "win". The
    # rolling 20-window will hit 100% errors at completion 20 ⇒ halve.
    tasks = list(range(25))
    results: list[dict] = []

    def run_fn(t):
        return {
            "cell": f"c{t}",
            "outcome": "error" if t < 20 else "win",
        }

    def record_fn(rec):
        results.append(rec)

    _run_adaptive_pool(tasks, run_fn, record_fn, initial_concurrency=4)
    assert len(results) == 25
    err = capsys.readouterr().err
    assert "halving concurrency 4" in err


def test_adaptive_concurrency_logs_no_change_under_clean_run(capsys):
    """A clean run (all wins) must not log any halve/restore line.
    The pool starts at the initial concurrency and stays there."""
    from openra_bench.run_eval import _run_adaptive_pool

    tasks = list(range(30))
    results: list[dict] = []

    def run_fn(t):
        return {"cell": f"c{t}", "outcome": "win"}

    def record_fn(rec):
        results.append(rec)

    _run_adaptive_pool(tasks, run_fn, record_fn, initial_concurrency=2)
    assert len(results) == 30
    err = capsys.readouterr().err
    assert "halving concurrency" not in err
    assert "restoring concurrency" not in err
