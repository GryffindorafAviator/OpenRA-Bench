"""Audit: how many pass^N reps had their per-turn transcripts overwritten?

PRE-PR-FIX BUG: `openra_bench.run_eval._run_one` gated the Playback
(and FullPlayback) writers on `rep == 0`, so reps 1..N-1 of every
(pack, level, seed) cell silently OVERWROTE the rep-0 playback dir.
The journal kept the outcome / composite / signals for every rep,
but the per-turn `messages.json` / `turns.jsonl` / minimap PNGs for
each non-first rep were lost forever.

This script scans a run directory (or a parent of run dirs) and
reports:

  • Cells with >1 journal row but only ONE on-disk playback dir
    (the pure-overwritten case — every rep but the last is gone).
  • Cells with >1 journal row AND >1 on-disk playback dir (the
    post-fix layout — rep dirs separated, nothing lost).
  • The total count of rep-transcripts permanently lost.

Usage:
    python3 tools/audit_overwritten_reps.py <run-dir-or-parent>

Output: a short text summary + a CSV at
`audits/overwritten_reps_<run-name>.csv` (one row per affected cell)
when there is anything to report.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Same sanitizer as `run_eval._score_path_candidates` — keep alnum and
# `.`/`_`/`-`, anything else becomes `_`.
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _scan_run(run_dir: Path) -> dict:
    """Return a dict summarising one `<run_id>__<safe_model>/` dir.

    Layout (post-fix): `<safe_cell>/seedN[_repR]/`.  Layout (pre-fix):
    `<safe_cell>/seedN/` (rep collisions silent)."""
    if not run_dir.is_dir():
        return {}
    # Journal layouts seen in the wild:
    #   <run_dir>.jsonl                 (sibling)
    #   <parent>/_journal__<model>.jsonl   (production layout)
    #   <run_dir>/_journal*.jsonl       (nested fallback)
    journal: Path | None = run_dir.parent / f"{run_dir.name}.jsonl"
    if not journal.exists():
        sibs = list(run_dir.parent.glob("_journal__*.jsonl"))
        journal = sibs[0] if sibs else None
    if journal is None or not journal.exists():
        nested = [
            p for p in run_dir.rglob("*.jsonl")
            if "messages" not in p.name
        ]
        journal = nested[0] if nested else None

    # Count rep rows per (cell, split, seed) from the journal.
    rep_rows: dict[tuple, set[int]] = defaultdict(set)
    if journal is not None and journal.exists():
        for line in journal.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(r, dict):
                continue
            if r.get("_meta") is not None:
                continue
            cell = r.get("cell")
            split = r.get("split", "public")
            seed = r.get("seed")
            rep = int(r.get("repeat", 0) or 0)
            if cell is None or seed is None:
                continue
            rep_rows[(str(cell), str(split), int(seed))].add(rep)

    # Count on-disk dirs per (cell, split, seed).
    disk_dirs: dict[tuple, list[str]] = defaultdict(list)
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        safe_cell = sub.name  # e.g. "pack_easy_public"
        for seed_dir in sub.iterdir():
            if not seed_dir.is_dir():
                continue
            m = re.match(r"^seed(\d+)(?:_rep(\d+))?$", seed_dir.name)
            if not m:
                continue
            seed = int(m.group(1))
            # Match safe_cell back to (cell, split). The sanitizer
            # makes this lossy in theory, but in practice the cell
            # always has the form `<pack>_<level>_<split>`. We use the
            # safe_cell as the key and re-link to journal rows by the
            # same sanitizer.
            disk_dirs[(safe_cell, seed)].append(seed_dir.name)

    return {
        "run_dir": run_dir,
        "journal": journal,
        "rep_rows": rep_rows,
        "disk_dirs": disk_dirs,
    }


def _safe_cell(cell: str, split: str) -> str:
    return _SAFE_RE.sub("_", f"{cell}:{split}")


def _audit(run_dir: Path) -> tuple[int, int, list[dict]]:
    """Return (total_journaled_reps, total_overwritten_lost, rows)."""
    info = _scan_run(run_dir)
    if not info:
        return 0, 0, []
    rep_rows = info["rep_rows"]
    disk_dirs = info["disk_dirs"]
    total_reps = 0
    total_lost = 0
    rows: list[dict] = []
    for (cell, split, seed), reps in rep_rows.items():
        n_reps = len(reps)
        total_reps += n_reps
        if n_reps <= 1:
            continue
        sc = _safe_cell(cell, split)
        on_disk = disk_dirs.get((sc, seed), [])
        n_disk = len(on_disk)
        lost = max(0, n_reps - n_disk)
        if lost > 0:
            total_lost += lost
            rows.append({
                "cell": cell,
                "split": split,
                "seed": seed,
                "journal_reps": n_reps,
                "on_disk_dirs": n_disk,
                "lost_transcripts": lost,
                "disk_dirs_seen": ",".join(sorted(on_disk)),
            })
    return total_reps, total_lost, rows


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write(
            "usage: audit_overwritten_reps.py <run-dir | parent-of-run-dirs>\n"
        )
        return 2
    root = Path(argv[0]).resolve()
    if not root.exists():
        sys.stderr.write(f"path does not exist: {root}\n")
        return 2

    # Accept either a single `<run_id>__<safe_model>/` or a parent
    # holding many of them. Heuristic: a run dir has child dirs whose
    # leaf name starts with a sanitized cell (contains `_`).
    targets: list[Path] = []
    if any((root / p).is_dir() for p in root.iterdir() if p.name.startswith("seed")):
        # `root` itself looks unusual; just try it.
        targets.append(root)
    else:
        # Try children first; if none look like run dirs, fall back.
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            # A run dir has subdirs containing seed* leaves.
            has_seed_grandchild = any(
                any(re.match(r"^seed\d+", g.name) for g in gc.iterdir())
                for gc in child.iterdir() if gc.is_dir()
            )
            if has_seed_grandchild:
                targets.append(child)
        if not targets:
            targets.append(root)

    grand_reps = 0
    grand_lost = 0
    all_rows: list[dict] = []
    for t in targets:
        n_reps, n_lost, rows = _audit(t)
        grand_reps += n_reps
        grand_lost += n_lost
        print(
            f"{t}:  journaled_reps={n_reps:>6d}  "
            f"overwritten_lost={n_lost:>6d}  affected_cells={len(rows)}"
        )
        all_rows.extend({**r, "run_dir": str(t)} for r in rows)

    print(
        f"\nTOTAL: journaled_reps={grand_reps}, "
        f"overwritten_rep_transcripts_lost={grand_lost}"
    )

    if all_rows:
        out = Path("audits") / (
            f"overwritten_reps_{root.name or 'root'}.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = list(all_rows[0].keys())
        with out.open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in all_rows:
                fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
        print(f"detail: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
