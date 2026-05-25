#!/usr/bin/env python3
"""Post-process pack_bar_status.csv into a per-pack diagnosis.

For each pack: stall-loses count, stall-wins count (BAR BREAK), draws,
errors. Cluster by family. Emit a markdown report + a flat list of
packs needing rebalance.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path


_FAMILY_PREFIXES = [
    ("family1-combat-micro", ("combat-", "action-", "harass-")),
    ("family2-economy", ("econ-", "economy-")),
    ("family3-defense", ("def-", "defense-", "build-defensive-",
                        "build-def-")),
    ("family4-scout-perception", ("scout-", "perception-", "navigation-")),
    ("family5-long-horizon", ("lh-", "longhorizon-")),
    ("family6-build-tech-power", ("build-", "tech-", "power-", "building-")),
    ("family7-procedure-robustness", ("proc-", "strict-", "maint-", "rob-")),
    ("family8-multi-front-coord", ("mfb-", "mcv-", "coord-", "coordination-")),
    ("family9-tempo-strategy", ("tp-", "tempo-", "strategy-", "adv-",
                                "artofwar-", "adversarial-", "mid-")),
    ("family10-special-misc", ("spec-", "custom-", "rush-hour")),
    ("family11-full-game", ("f11-",)),
]


def family_of(pack_id: str) -> str:
    for fam, prefixes in _FAMILY_PREFIXES:
        if any(pack_id.startswith(p) for p in prefixes):
            return fam
    return "uncategorized"


def main():
    if len(sys.argv) < 2:
        print("usage: analyze_pack_bar.py <pack_bar_status.csv>", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(sys.argv[1])))
    # group by pack
    by_pack: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_pack[r["pack"]].append(r)

    # only consider stall policy for the BAR analysis
    stall_breakers: list[tuple[str, list[dict]]] = []
    stall_clean: list[str] = []
    error_packs: list[str] = []
    for pack, rs in by_pack.items():
        stalls = [r for r in rs if r["policy"] == "stall"]
        if not stalls:
            continue
        if any(r["outcome"] == "error" for r in stalls):
            error_packs.append(pack)
            continue
        wins = [r for r in stalls if r["outcome"] == "win"]
        if wins:
            stall_breakers.append((pack, wins))
        else:
            stall_clean.append(pack)

    # group by family
    by_fam: dict[str, dict[str, list]] = defaultdict(
        lambda: {"clean": [], "broken": [], "error": []},
    )
    for pack, wins in stall_breakers:
        by_fam[family_of(pack)]["broken"].append((pack, wins))
    for pack in stall_clean:
        by_fam[family_of(pack)]["clean"].append(pack)
    for pack in error_packs:
        by_fam[family_of(pack)]["error"].append(pack)

    print("=" * 70)
    print("STALL-WINS BAR AUDIT — packs where observe-only WINS")
    print("(violates no-cheat bar; intended capability isn't being measured)")
    print("=" * 70)
    print()
    total_broken = len(stall_breakers)
    total_clean = len(stall_clean)
    total_error = len(error_packs)
    grand = total_broken + total_clean + total_error
    print(f"  {total_broken}/{grand} packs ({100*total_broken/grand:.1f}%) "
          f"have a STALL=WIN cell — bar is broken")
    print(f"  {total_clean}/{grand} packs ({100*total_clean/grand:.1f}%) "
          f"are clean (stall LOSES every level × seed)")
    print(f"  {total_error}/{grand} packs ({100*total_error/grand:.1f}%) "
          f"errored on at least one cell")
    print()

    for fam in sorted(by_fam):
        slot = by_fam[fam]
        if not slot["broken"] and not slot["error"]:
            continue
        print(f"### {fam}  "
              f"({len(slot['broken'])} broken, {len(slot['error'])} error, "
              f"{len(slot['clean'])} clean)")
        for pack, wins in sorted(slot["broken"]):
            cells = ", ".join(f"{w['level']}/s{w['seed']}" for w in wins)
            ev = wins[0]["economy_value"]
            print(f"  ❌ {pack:50s} stall WINS on: {cells}  (ev={ev})")
        for pack in sorted(slot["error"]):
            print(f"  ⚠  {pack:50s} (errored — investigate)")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
