"""Reconcile orphaned score.json files into the journal.

Background: commit 3ab9b417 introduced a resume-gate bug that silently raised
DuplicateJournalKey on re-queued rep=0 tasks for pass@1-done cells. Episodes
completed (score.json written) but the journal append failed.

For each cell (pack, level, seed), this script:
  1. Counts existing journal records and notes which rep slots are filled.
  2. Counts score.json files on disk across all playback dirs for that cell.
  3. If score.json count > journal count, the OLDEST score.json files
     correspond to the existing journal records (already accounted for). The
     NEWEST extras are orphans — assigned to the next-empty rep slots.

Usage:
    PYTHONPATH=. python3 tools/reconcile_orphaned_scores.py --dry-run
    PYTHONPATH=. python3 tools/reconcile_orphaned_scores.py   # write
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from openra_bench.resilience import RunJournal, episode_key


PROD_DIR = Path("data/runs/v1.1-prod")
MODELS = ["qwen3.5-9b", "qwen3.6-35b-a3b", "gemma-4-31b-it",
          "gpt-5.4", "gpt-5.4-mini"]


def _parse_cell_dir(name: str) -> tuple[str, str, str] | None:
    parts = name.rsplit("_", 2)
    if len(parts) != 3:
        return None
    pack, level, split = parts
    return pack, level, split


def _base_key_of(k: str) -> str:
    return k.split("|rep")[0] if "|rep" in k else k


def _rep_of(k: str) -> int:
    if "|rep" not in k:
        return 0
    tail = k.rsplit("|rep", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return 0


def reconcile_model(model: str, dry_run: bool, max_repeats: int = 3) -> dict:
    base = PROD_DIR / model / "scenarios"
    jp = next(base.glob("journal*.jsonl"), None)
    if jp is None:
        return {"model": model, "skipped": "no journal"}

    # Existing slots: {base_key: set_of_reps_in_journal}
    existing_reps: defaultdict[str, set[int]] = defaultdict(set)
    with open(jp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("_meta"):
                continue
            k = rec.get("_key", "")
            if k:
                existing_reps[_base_key_of(k)].add(_rep_of(k))

    # Find all score.json files, group by base_key, sort by mtime ascending.
    playback_root = base / "playback"
    if not playback_root.exists():
        return {"model": model, "skipped": "no playback dir"}

    by_cell: defaultdict[str, list[dict]] = defaultdict(list)
    for run_dir in sorted(playback_root.iterdir()):
        if not run_dir.is_dir():
            continue
        for cell_dir in sorted(run_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            parsed = _parse_cell_dir(cell_dir.name)
            if parsed is None:
                continue
            pack, level, split = parsed
            for seed_dir in sorted(cell_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                m = re.match(r"seed(\d+)", seed_dir.name)
                if not m:
                    continue
                seed = int(m.group(1))
                sp = seed_dir / "score.json"
                if not sp.exists():
                    continue
                try:
                    score = json.loads(sp.read_text())
                except Exception:
                    continue
                if score.get("outcome") is None:
                    continue
                mf = seed_dir / "manifest.json"
                manifest = {}
                if mf.exists():
                    try:
                        manifest = json.loads(mf.read_text())
                    except Exception:
                        pass
                fog = manifest.get("fog_mode") or "vision"
                bk = episode_key(pack, level, split, seed, fog)
                by_cell[bk].append({
                    "score": score, "manifest": manifest, "mtime": sp.stat().st_mtime,
                    "score_path": sp, "pack": pack, "level": level,
                    "split": split, "seed": seed, "fog": fog,
                })

    # For each cell, the OLDEST N score.json correspond to the N journal
    # entries already present. The NEWEST extras are orphans. Assign each
    # orphan to the next-empty rep slot for that cell.
    appendable: list[tuple[str, dict]] = []
    total_score_files = 0
    total_already_matched = 0
    total_orphans = 0
    over_repeats_dropped = 0
    for bk, score_list in by_cell.items():
        score_list.sort(key=lambda x: x["mtime"])
        total_score_files += len(score_list)
        n_journaled = len(existing_reps[bk])
        # Slots already accounted for
        total_already_matched += min(n_journaled, len(score_list))
        orphans = score_list[n_journaled:]
        # Find next-empty rep slot(s)
        next_reps = sorted(set(range(max_repeats)) - existing_reps[bk])
        for orphan, rep in zip(orphans, next_reps):
            total_orphans += 1
            score = orphan["score"]
            manifest = orphan["manifest"]
            rec = {
                "cell": f"{orphan['pack']}:{orphan['level']}",
                "capability": manifest.get("capability") or score.get("capability"),
                "split": orphan["split"],
                "seed": orphan["seed"],
                "repeat": rep,
                "fog_mode": orphan["fog"],
                "outcome": score.get("outcome"),
                "composite": score.get("composite") or score.get("score") or 0.0,
                "turns": (score.get("turns") or manifest.get("turns")),
                "perception": score.get("perception", 0.0),
                "reasoning": score.get("reasoning", 0.0),
                "action": score.get("action", 0.0),
                "speed": score.get("speed", 0.0),
                "objective_progress": score.get("objective_progress") or 0.0,
                "objective_blocking_ratio": score.get("objective_blocking_ratio") or 0.0,
                "passivity": score.get("passivity") or 0.0,
                "leaves_final": score.get("leaves_final") or score.get("leaves") or [],
                "reward_vector": score.get("reward_vector") or [],
                "notes": "[reconciled from orphaned score.json — bug 3ab9b417]",
                "weakest_link": score.get("weakest_link"),
                "win_turns": score.get("win_turns"),
                "handoff": score.get("handoff", "none"),
            }
            key = episode_key(orphan["pack"], orphan["level"], orphan["split"],
                              orphan["seed"], orphan["fog"], repeat=rep)
            appendable.append((key, rec))
        # Any extra orphans beyond max_repeats are dropped
        if len(orphans) > len(next_reps):
            over_repeats_dropped += len(orphans) - len(next_reps)

    print(f"\n--- {model} ---")
    print(f"  journal entries before: {sum(len(s) for s in existing_reps.values())}")
    print(f"  total score.json on disk: {total_score_files}")
    print(f"  already-journaled matches: {total_already_matched}")
    print(f"  orphans to append: {total_orphans}")
    print(f"  dropped (would exceed max_repeats={max_repeats}): {over_repeats_dropped}")

    if not dry_run and appendable:
        rj = RunJournal(jp, ignore_run_id=True)
        appended = 0
        failures = 0
        for key, rec in appendable:
            try:
                rj.append(key, rec)
                appended += 1
            except Exception as e:  # noqa: BLE001
                failures += 1
                if failures <= 3:
                    print(f"  ! append failed for {key}: {e}")
        print(f"  ✓ appended {appended}/{len(appendable)} (failures={failures})")

    return {
        "model": model,
        "score_files": total_score_files,
        "orphans_to_append": total_orphans,
        "dropped": over_repeats_dropped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-repeats", type=int, default=3)
    a = ap.parse_args()
    results = [reconcile_model(m, a.dry_run, a.max_repeats) for m in MODELS]
    print("\n=== summary ===")
    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
