"""v1.1-prod status dashboard — per-rep success counts EXCLUDING errors.

The journal counts include errored cells (outcome=error), which inflates
the displayed progress. This tool reads each model's journal and reports
*successful* completions per repeat, with the error count alongside.

Usage:
    python3 tools/v1_1_status.py [run_root]

Default run_root = data/runs/v1.1-prod
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

MODELS = [
    "qwen3.5-9b",
    "qwen3.6-35b-a3b",
    "gemma-4-31b-it",
    "gpt-5.4",
    "gpt-5.4-mini",
]
CELLS_PER_REP = 648


def _journal_paths(run_dir: Path) -> list[Path]:
    return [
        Path(p)
        for p in glob.glob(str(run_dir / "journal__*.jsonl"))
        if not p.endswith(".bak")
    ]


def _scan(jpaths: list[Path]) -> tuple[Counter, Counter]:
    """Return (success_per_rep, error_per_rep)."""
    success = Counter()
    errs = Counter()
    for p in jpaths:
        try:
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(r, dict) or r.get("_meta"):
                        continue
                    rep = r.get("repeat", 0) or 0
                    if r.get("outcome") == "error":
                        errs[rep] += 1
                    else:
                        success[rep] += 1
        except FileNotFoundError:
            continue
    return success, errs


def _log_age(run_dir: Path) -> str:
    log = run_dir / "run_eval.log"
    if not log.exists():
        return "no log"
    age = int(time.time() - log.stat().st_mtime)
    return f"{age}s ago"


def _live_spt(run_dir: Path, window_s: int = 120) -> tuple[int, float, float, float, float]:
    """Read fresh progress.json files in this run's playback tree and return
    (n_active_cells, mean_spt, min_spt, max_spt, mean_turn).
    Returns (0, 0, 0, 0, 0) when no live cells in the window."""
    pb = run_dir / "playback"
    if not pb.exists():
        return (0, 0.0, 0.0, 0.0, 0.0)
    now = time.time()
    spts: list[float] = []
    turns: list[int] = []
    for p in pb.rglob("progress.json"):
        try:
            if now - p.stat().st_mtime > window_s:
                continue
            r = json.loads(p.read_text())
            spt = r.get("sec_per_turn")
            if isinstance(spt, (int, float)) and spt > 0:
                spts.append(float(spt))
                turns.append(int(r.get("turn", 0)))
        except Exception:
            continue
    if not spts:
        return (0, 0.0, 0.0, 0.0, 0.0)
    return (
        len(spts),
        sum(spts) / len(spts),
        min(spts),
        max(spts),
        sum(turns) / len(turns),
    )


def main(run_root: str) -> None:
    root = Path(run_root)
    print(f"# v1.1 status @ {time.strftime('%H:%M:%S')}  ({root})")
    print()
    print(f"{'model':<20} {'p@1':<24} {'p^2':<24} {'p^3':<24} {'log':<10}")
    print("-" * 105)
    for m in MODELS:
        rd = root / m / "scenarios"
        if not rd.exists():
            print(f"{m:<20} (no run dir)")
            continue
        succ, errs = _scan(_journal_paths(rd))

        def fmt(rep: int) -> str:
            ok = succ.get(rep, 0)
            e = errs.get(rep, 0)
            tail = f" err={e}" if e else ""
            marker = " ✓" if ok >= CELLS_PER_REP else ""
            return f"{ok}/{CELLS_PER_REP}{marker}{tail}"

        print(f"{m:<20} {fmt(0):<24} {fmt(1):<24} {fmt(2):<24} {_log_age(rd):<10}")
    print()
    print("[ pass^3 live s/turn (fresh progress.json, last 2 min) ]")
    print(f"  {'model':<20} {'n_active':>8} {'s/turn':>10} {'min':>7} {'max':>7} {'avg_turn':>10}")
    for m in MODELS:
        rd = root / m / "scenarios"
        if not rd.exists():
            continue
        n, mean_spt, mn, mx, mean_turn = _live_spt(rd)
        if n == 0:
            print(f"  {m:<20} {'-':>8} {'(no live progress.json)':>30}")
        else:
            print(
                f"  {m:<20} {n:>8d} {mean_spt:>9.1f}s {mn:>6.1f}s {mx:>6.1f}s {mean_turn:>10.0f}"
            )
    print()
    onev = root.parent / "v1.1-prod-1v1"
    if onev.exists():
        n_eps = sum(1 for _ in onev.rglob("score.json"))
        print(f"[1v1]  {onev}: {n_eps} episode score.json files")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/runs/v1.1-prod"
    main(root)
