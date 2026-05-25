#!/usr/bin/env python3
"""Audit the no-cheat bar across every active scenario pack.

For each (pack, level, seed), run a small set of degenerate scripted
policies (stall + brute) and record outcome + key signals. Emit a CSV
the rebalance work can fan out against.

Stall policy = `observe()` only. Should LOSE on every (pack, level,
seed) per the no-cheat bar. A stall WIN means the pack's bar is below
what the engine produces unattended — almost always an auto-harvest
baseline issue (the bug we already fixed in economy-harvest-timebox
easy).

Brute policy = `attack_move` every owned unit toward the centroid of
enemy positions (or, if none visible, toward the objective region
center if the pack defines one). A meaningful subset of packs should
also see brute LOSE.

The point is to surface bar-breaks the existing pytest tests don't
assert. Many packs have no scripted "stall must lose" test — those
are the silent defects we can't see in CI today.

Usage:
    python3 tools/validate_pack_bar.py [--out audits/pack_bar_status.csv] \\
                                       [--packs <glob>] [--seeds N] \\
                                       [--levels easy,medium,hard]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from openra_bench.eval_core import run_level  # noqa: E402
from openra_bench.scenarios import load_pack  # noqa: E402
from openra_bench.scenarios.loader import PACKS_DIR, compile_level  # noqa: E402


def _objective_region(compiled) -> tuple[int, int] | None:
    """Try to pull an (x, y) target from the pack's win_condition."""
    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("units_in_region_gte", "building_in_region",
                        "all_units_in_region"):
                    if isinstance(v, dict):
                        x, y = v.get("x"), v.get("y")
                        if x is not None and y is not None:
                            return (int(x), int(y))
                r = walk(v)
                if r:
                    return r
        elif isinstance(node, list):
            for x in node:
                r = walk(x)
                if r:
                    return r
        return None
    return walk(compiled.win_condition)


def _enemy_centroid(rs) -> tuple[int, int] | None:
    ep = rs.get("enemy_positions") or []
    if not ep:
        ep = rs.get("enemy_summary") or []
    xs, ys = [], []
    for e in ep:
        if isinstance(e, dict) and "cell_x" in e and "cell_y" in e:
            xs.append(int(e["cell_x"]))
            ys.append(int(e["cell_y"]))
    if not xs:
        return None
    return (sum(xs) // len(xs), sum(ys) // len(ys))


def stall_policy(rs, Command):
    return [Command.observe()]


def make_brute_policy(objective: tuple[int, int] | None):
    def pol(rs, Command):
        units = rs.get("units_summary") or []
        ids = [str(u["id"]) for u in units if "id" in u]
        if not ids:
            return [Command.observe()]
        target = _enemy_centroid(rs) or objective
        if target is None:
            return [Command.observe()]
        return [Command.attack_move(ids, int(target[0]), int(target[1]))]
    return pol


def _summarize(res) -> dict[str, Any]:
    s = res.signals
    return {
        "outcome": res.outcome,
        "turns": res.turns,
        "game_tick": s.game_tick,
        "cash": s.cash,
        "resources": s.resources,
        "economy_value": s.cash + s.resources,
        "units_killed": s.units_killed,
        "units_lost": s.units_lost,
        "explored_pct": round(s.explored_percent, 2),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="audits/pack_bar_status.csv",
        help="output CSV path (default audits/pack_bar_status.csv)",
    )
    ap.add_argument(
        "--packs", default="*.yaml",
        help="glob to filter packs (default *.yaml)",
    )
    ap.add_argument(
        "--seeds", default="1,2,3,4",
        help="comma-separated seeds (default 1,2,3,4)",
    )
    ap.add_argument(
        "--levels", default="easy,medium,hard",
        help="comma-separated levels (default easy,medium,hard)",
    )
    ap.add_argument(
        "--brute", action="store_true",
        help="also run the brute policy (slower; default stall only)",
    )
    args = ap.parse_args(argv[1:])

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    pack_files = sorted(PACKS_DIR.glob(args.packs))
    # Skip non-pack templates / disabled packs.
    pack_files = [
        f for f in pack_files
        if not f.name.startswith(("_", "TEMPLATE", "."))
    ]

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pack", "level", "seed", "policy", "outcome", "turns",
        "game_tick", "cash", "resources", "economy_value",
        "units_killed", "units_lost", "explored_pct", "error",
    ]
    rows: list[dict] = []
    t0 = time.monotonic()
    n_packs = len(pack_files)
    for i, fp in enumerate(pack_files, 1):
        pack_id = fp.stem
        try:
            pack = load_pack(fp)
            if getattr(pack.meta, "status", "active") != "active":
                continue
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {pack_id}: load failed ({e})", file=sys.stderr)
            continue
        sys.stderr.write(f"[{i}/{n_packs}] {pack_id}\n")
        for level in levels:
            try:
                c = compile_level(pack, level)
            except Exception:
                continue
            policies = [("stall", stall_policy)]
            if args.brute:
                obj = _objective_region(c)
                policies.append(("brute", make_brute_policy(obj)))
            for pol_name, pol in policies:
                for seed in seeds:
                    row = {
                        "pack": pack_id, "level": level, "seed": seed,
                        "policy": pol_name, "outcome": "", "turns": 0,
                        "game_tick": 0, "cash": 0, "resources": 0,
                        "economy_value": 0, "units_killed": 0,
                        "units_lost": 0, "explored_pct": 0, "error": "",
                    }
                    try:
                        res = run_level(c, pol, seed=seed)
                        row.update(_summarize(res))
                    except Exception as e:  # noqa: BLE001
                        row["error"] = repr(e)[:200]
                        row["outcome"] = "error"
                    rows.append(row)
    elapsed = time.monotonic() - t0

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Summary at end.
    total = len(rows)
    stall_wins = [r for r in rows if r["policy"] == "stall" and r["outcome"] == "win"]
    stall_errs = [r for r in rows if r["policy"] == "stall" and r["outcome"] == "error"]
    print(json.dumps({
        "rows": total,
        "elapsed_sec": round(elapsed, 1),
        "stall_wins": len(stall_wins),
        "stall_wins_packs": sorted({r["pack"] for r in stall_wins}),
        "stall_errors": len(stall_errs),
        "out": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
