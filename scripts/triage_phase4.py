#!/usr/bin/env python3
"""Phase 4 collection triage: scan completed JSONLs and produce a
per-(model × pack × level) outcome table + per-loss command histogram
so failures can be classified into engine / scenario / model.

Usage:
    python3 scripts/triage_phase4.py [<root>] ...

If no roots, defaults to data/runs/paper-v1*.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _command_kind(cmd) -> str:
    if isinstance(cmd, dict):
        return cmd.get("kind") or "?"
    if isinstance(cmd, str) and "::" in cmd:
        return cmd.split("::", 1)[1].split(" ", 1)[0]
    return "?"


def triage_cell(jsonl: Path) -> dict | None:
    """Return a dict per cell: model, pack, level, seed, outcome, n_turns,
    cmd_kinds (Counter). None if the JSONL has no terminal line."""
    try:
        lines = jsonl.read_text().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    term = last.get("terminal") or {}
    if "outcome" not in term:
        return None
    # JSONL stem: <pack>__<level>__seed<N>__<fog>
    parts = jsonl.stem.split("__")
    pack, level, seed_part, fog = parts[0], parts[1], parts[2], parts[3]
    # Parent dir: <ts>__<model_safe>
    model = jsonl.parent.name.split("__", 1)[1] if "__" in jsonl.parent.name else jsonl.parent.name
    cmd_kinds: Counter = Counter()
    for ln in lines:
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        for c in d.get("commands_issued") or []:
            cmd_kinds[_command_kind(c)] += 1
    return {
        "model": model,
        "pack": pack,
        "level": level,
        "seed": int(seed_part.replace("seed", "")),
        "outcome": term["outcome"],
        "n_turns": int(term.get("turns") or len(lines)),
        "cmd_kinds": cmd_kinds,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", default=[],
                    help="Run dirs (default: data/runs/paper-v1*)")
    args = ap.parse_args(argv[1:])
    repo = Path(__file__).resolve().parent.parent
    roots = [Path(r) for r in args.roots] if args.roots else \
        sorted((repo / "data" / "runs").glob("paper-v1*"))
    records = []
    for root in roots:
        if not root.is_dir():
            continue
        for jsonl in root.rglob("*.jsonl"):
            r = triage_cell(jsonl)
            if r is not None:
                records.append(r)
    if not records:
        print("(no completed cells found)")
        return 1
    # Outcome matrix: model × (pack, level) → Counter of outcomes
    matrix: dict = defaultdict(lambda: defaultdict(Counter))
    losses = []
    for r in records:
        matrix[r["model"]][(r["pack"], r["level"])][r["outcome"]] += 1
        if r["outcome"] != "win":
            losses.append(r)

    print(f"=== Phase 4 triage on {len(records)} completed cells ===\n")
    print(f"Models in scope: {sorted(matrix.keys())}\n")
    for model, packs in sorted(matrix.items()):
        print(f"--- {model} ---")
        print(f"{'pack':30s} {'level':6s} {'W':>2s} {'L':>2s} {'D':>2s}")
        for (pack, level), counts in sorted(packs.items()):
            print(f"  {pack[:28]:28s} {level:6s} "
                  f"{counts.get('win',0):>2d} {counts.get('loss',0):>2d} {counts.get('draw',0):>2d}")
        print()

    print(f"=== {len(losses)} loss/draw cells with command histogram ===")
    print(f"{'model':30s} {'pack':30s} {'lvl':6s} {'sd':>2s} {'out':5s} {'turns':>5s}  top cmds")
    for r in losses:
        top = " ".join(f"{k}:{n}" for k, n in r["cmd_kinds"].most_common(3))
        print(f"  {r['model'][:30]:30s} {r['pack'][:28]:28s} {r['level']:6s} "
              f"{r['seed']:>2d} {r['outcome']:5s} {r['n_turns']:>5d}  {top}")

    # Classification per loss
    print(f"\n=== Failure classification ===")
    f1 = sum(1 for r in losses
             if r["outcome"] == "loss"
             and (set(r["cmd_kinds"].keys()) <= {"MoveUnits", "Observe", "Stop"}))
    f2 = sum(1 for r in losses
             if r["outcome"] == "loss"
             and "FireSuperweapon" in r["cmd_kinds"]
             and r["pack"] == "spec-nuke-strike")
    other = len([r for r in losses if r["outcome"] == "loss"]) - f1 - f2
    print(f"  F1 (passive/walk-only, no special verb): {f1} losses")
    print(f"  F2 (superweapon mis-aim):                {f2} losses")
    print(f"  Other (verb invoked, still lost):        {other} losses")
    print(f"  Draws (need replay):                     "
          f"{sum(1 for r in losses if r['outcome']=='draw')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
