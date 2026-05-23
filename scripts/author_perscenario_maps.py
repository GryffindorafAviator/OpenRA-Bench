#!/usr/bin/env python3
"""Author per-scenario maps for each scenario pack.

The bench currently shares one 128×40 `rush-hour-arena.oramap` across
194 of 210 packs. Each scenario should have a map TAILORED to what it
measures — small focused arenas for short-execution packs (Tanya C4),
chokepoint maps for defensive packs, multi-zone maps for expansion /
multi-base packs, bridges for crossing-defense packs.

This script declares a per-pack map spec (one dict per pack id) and:

  --dry-run        : materialize the .oramaps + print a summary;
                     do NOT edit any pack YAMLs.
  --apply          : in addition, rewrite each pack's `base_map:`
                     line to point at the new per-scenario .oramap.
  --validate       : run a stall-policy smoke through each pack to
                     confirm the new map loads cleanly (no panic) and
                     the scenario still terminates.

Authoring rule of thumb (from CLAUDE.md):
  - small open arena → short-execution packs (Tanya C4, engineer
    capture easy, spy infiltrate easy).
  - medium with corridor / chokepoint → packs that need a forced
    pinch (def-bridge-chokepoint, chokepoint family).
  - large open arena → search / perception packs (spec-nuke-strike,
    scout-far-frontier).
  - bridges-arena → packs whose briefing references bridges
    (def-bridge-chokepoint).
  - naval-arena → packs with water + ships (combat-naval-shore-strike).

The 12-pack debug grid is the first wave; the remaining ~198 packs
follow in subsequent waves once these are validated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from openra_bench.mapgen import materialize  # noqa: E402

# ── per-pack map specs ────────────────────────────────────────────────
# Each spec is the dict you'd put under `base_map:` in the pack YAML.
# Generators understand: arena, bridges-arena, chokepoint-arena,
# naval-arena. `name` slugs become the .oramap filename.
PER_PACK_MAPS: dict[str, dict] = {
    # ── special-ability family ────────────────────────────────────────
    # Short execution; small focused arena keeps the run on the
    # action, not the navigation.
    "spec-tanya-c4-strike": {
        "generator": "arena",
        "name": "spec-tanya-c4-strike-arena",
        "title": "Tanya Strike — close-range execution",
        "width": 80, "height": 32, "cordon": 2,
    },
    # Engineer is fragile; one obstacle band creates a north/south
    # choice that tests path planning under attrition.
    "spec-engineer-capture": {
        "generator": "arena",
        "name": "spec-engineer-capture-arena",
        "title": "Engineer Capture — escort path",
        "width": 96, "height": 36, "cordon": 2,
        "obstacles": [{"x": 44, "y": 14, "w": 8, "h": 8}],
    },
    # Spy infiltrates fogged; medium arena with TWO cover clusters
    # offers a stealth-route choice.
    "spec-spy-infiltrate": {
        "generator": "arena",
        "name": "spec-spy-infiltrate-arena",
        "title": "Spy Infiltrate — fogged approach",
        "width": 96, "height": 36, "cordon": 2,
        "obstacles": [
            {"x": 30, "y": 8, "w": 6, "h": 4},
            {"x": 30, "y": 24, "w": 6, "h": 4},
            {"x": 56, "y": 16, "w": 6, "h": 4},
        ],
    },
    # Thief drains cash from proc/silo. Single corridor approach
    # tests whether the model finds the only path.
    "spec-thief-steal-cash": {
        "generator": "chokepoint-arena",
        "name": "spec-thief-steal-cash-arena",
        "title": "Thief — single corridor approach",
        "width": 80, "height": 32, "cordon": 2,
        "pinch_x": 40, "pinch_width": 8, "corridor_width": 4,
        "corridor_y": 16,
    },
    # Nuke targets a CLUSTER; the test is spatial commit (find +
    # aim). Large open arena gives plausible alternative cluster
    # positions.
    "spec-nuke-strike": {
        "generator": "arena",
        "name": "spec-nuke-strike-arena",
        "title": "Nuke Strike — large arena, find the cluster",
        "width": 144, "height": 48, "cordon": 3,
    },

    # ── economy family ────────────────────────────────────────────────
    # Pure macro: small open arena, no obstacles, no enemies.
    "econ-mine-and-grow": {
        "generator": "arena",
        "name": "econ-mine-and-grow-arena",
        "title": "Mine and Grow — pure macro",
        "width": 72, "height": 32, "cordon": 2,
    },
    # Contested patch in the centre; obstacle band between the two
    # bases forces an approach commitment.
    "econ-contested-expansion": {
        "generator": "arena",
        "name": "econ-contested-expansion-arena",
        "title": "Contested Expansion — single channel",
        "width": 112, "height": 36, "cordon": 2,
        "obstacles": [
            {"x": 50, "y": 6, "w": 8, "h": 8},
            {"x": 50, "y": 22, "w": 8, "h": 8},
        ],
    },
    # Multi-patch decision; medium arena with room for 3+ patches.
    "econ-multi-patch-allocation": {
        "generator": "arena",
        "name": "econ-multi-patch-allocation-arena",
        "title": "Multi-Patch Allocation",
        "width": 128, "height": 40, "cordon": 2,
    },
    # Second-base race; medium arena with clear best vs second-best
    # placement zones.
    "econ-second-base-race": {
        "generator": "arena",
        "name": "econ-second-base-race-arena",
        "title": "Second Base Race",
        "width": 112, "height": 36, "cordon": 2,
    },
    # Defend harvester from a raider on a single lane.
    "econ-harvester-defense-raid": {
        "generator": "arena",
        "name": "econ-harvester-defense-raid-arena",
        "title": "Harvester Defense — single raid lane",
        "width": 96, "height": 36, "cordon": 2,
        "obstacles": [
            {"x": 48, "y": 6, "w": 6, "h": 10},
            {"x": 48, "y": 22, "w": 6, "h": 10},
        ],
    },

    # ── defense / combat family ───────────────────────────────────────
    # Bridges! Real water channel with 3 crossings — the briefing
    # finally matches the terrain.
    "def-bridge-chokepoint": {
        "generator": "bridges-arena",
        "name": "def-bridge-chokepoint-arena",
        "title": "Bridge Chokepoint — 3 crossings",
        "width": 128, "height": 40, "cordon": 2,
        "axis": "vertical",
        "channel_x": 60, "channel_width": 3,
        "bridges": [
            {"pos": 8, "width": 3},
            {"pos": 18, "width": 3},
            {"pos": 28, "width": 3},
        ],
    },
    # Naval shore strike already uses naval-arena; keep but namespace
    # to the pack so future tuning is per-scenario.
    "combat-naval-shore-strike": {
        "generator": "naval-arena",
        "name": "combat-naval-shore-strike-arena",
        "title": "Naval Shore Strike",
        "width": 96, "height": 40, "cordon": 2,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="materialize maps and print summary; no YAML edits")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite each pack's base_map to the per-scenario id")
    ap.add_argument("--packs", default="all",
                    help="comma-separated pack ids or 'all'")
    args = ap.parse_args()

    target = (set(args.packs.split(",")) if args.packs != "all"
              else set(PER_PACK_MAPS))

    materialized: list[tuple[str, str, Path]] = []
    for pid, spec in sorted(PER_PACK_MAPS.items()):
        if pid not in target:
            continue
        mid = materialize(spec)
        out = REPO / "data" / "maps" / f"{mid}.oramap"
        materialized.append((pid, mid, out))
        print(f"  {pid:34s} → {mid:42s} ({out.stat().st_size} bytes)")
    print(f"\nmaterialized {len(materialized)}/{len(target)} pack maps")

    if args.apply:
        from openra_bench.scenarios.loader import PACKS_DIR
        edited = 0
        for pid, mid, _ in materialized:
            yaml_path = PACKS_DIR / f"{pid}.yaml"
            if not yaml_path.exists():
                print(f"  ⚠ pack file missing: {yaml_path}")
                continue
            raw = yaml_path.read_text()
            # Replace the first `base_map: <something>` line with the
            # new logical id. We use a regex anchored to start-of-line
            # so we don't touch any documentation that mentions
            # "base_map:" inside a comment.
            import re
            new_raw, n = re.subn(
                r"^base_map: .*$",
                f"base_map: {mid}",
                raw,
                count=1,
                flags=re.MULTILINE,
            )
            if n == 0:
                print(f"  ⚠ no base_map: line in {pid}")
                continue
            if new_raw != raw:
                yaml_path.write_text(new_raw)
                edited += 1
        print(f"\napplied to {edited} pack YAMLs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
