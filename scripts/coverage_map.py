"""Coverage map — where the 200 packs sit on the RTS phase ×
decision-divergence matrix from the original scenario plan.

Two views:

* Family histogram (data-driven, no judgement) — counts per pack-name
  prefix. The bench was authored in waves by family; this shows the
  raw distribution.
* Phase × Decision-class heatmap (heuristic) — maps each family to a
  best-fit (phase, decision) cell using a curated dictionary, then
  tallies. Empty / thin cells = coverage gaps.

Output: a printable summary + a JSON dump for the paper's §3 figure.

Run from the repo root:
  python scripts/coverage_map.py [--out coverage.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openra_bench.scenarios import load_pack  # noqa: E402
from openra_bench.scenarios.loader import PACKS_DIR  # noqa: E402

# ── canonical matrix from the original scenario plan ────────────────
PHASES = ("opening", "early-mid", "mid", "mid-late", "late", "cross-cutting")
DECISIONS = (
    "build-order / commit",
    "scout / perception",
    "live econ / mid mgmt",
    "expansion / multi-base",
    "defense positioning",
    "combat micro",
    "concede / isolate / decoy",
    "tempo / time-pressure",
    "procedural compliance",
    "coordination",
    "long-horizon multi-phase",
    "adversarial / counter-read",
    "robustness / recovery",
    "navigation / spatial",
)

# Family prefix -> (phase, decision). Stays a heuristic — pack names
# are the bench's primary structure (built in waves by family).
FAMILY_MAP: dict[str, tuple[str, str]] = {
    "perception":   ("early-mid", "scout / perception"),
    "scout":        ("early-mid", "scout / perception"),
    "navigation":   ("opening",   "navigation / spatial"),
    "build":        ("opening",   "build-order / commit"),
    "tech":         ("mid",       "build-order / commit"),
    "econ":         ("mid",       "live econ / mid mgmt"),
    "economy":      ("mid",       "live econ / mid mgmt"),
    "mid":          ("mid",       "live econ / mid mgmt"),
    "maint":        ("mid",       "live econ / mid mgmt"),
    "power":        ("mid",       "live econ / mid mgmt"),
    "mcv":          ("mid",       "expansion / multi-base"),
    "mfb":          ("mid",       "expansion / multi-base"),
    "expansion":    ("mid",       "expansion / multi-base"),
    "def":          ("mid",       "defense positioning"),
    "defense":      ("mid",       "defense positioning"),
    "combat":       ("mid",       "combat micro"),
    "artofwar":     ("mid-late",  "concede / isolate / decoy"),
    "strategy":     ("mid-late",  "concede / isolate / decoy"),
    "tempo":        ("cross-cutting", "tempo / time-pressure"),
    "tp":           ("cross-cutting", "tempo / time-pressure"),
    "action":       ("cross-cutting", "procedural compliance"),
    "strict":       ("cross-cutting", "procedural compliance"),
    "proc":         ("cross-cutting", "procedural compliance"),
    "coord":        ("cross-cutting", "coordination"),
    "coordination": ("cross-cutting", "coordination"),
    "lh":           ("late",       "long-horizon multi-phase"),
    "longhorizon":  ("late",       "long-horizon multi-phase"),
    "adv":          ("cross-cutting", "adversarial / counter-read"),
    "adversarial":  ("cross-cutting", "adversarial / counter-read"),
    "rob":          ("cross-cutting", "robustness / recovery"),
    "spec":         ("late",       "build-order / commit"),
    "transport":    ("mid-late",  "expansion / multi-base"),
    "rush":         ("opening",   "build-order / commit"),
    "custom":       ("opening",   "navigation / spatial"),
}


def _classify(pack_stem: str) -> tuple[str, str]:
    fam = pack_stem.split("-")[0]
    return FAMILY_MAP.get(fam, ("?", "?"))


def main(argv: list[str]) -> int:
    out_path = None
    if "--out" in argv:
        out_path = argv[argv.index("--out") + 1]

    packs = []
    for p in sorted(PACKS_DIR.glob("*.yaml")):
        if p.name.startswith(("_", "TEMPLATE")):
            continue
        try:
            d = load_pack(p)
        except Exception:  # noqa: BLE001
            continue
        if getattr(d.meta, "status", "active") == "quarantine":
            continue
        packs.append((p.stem, d))

    # ── view 1: family histogram ────────────────────────────────────
    fam_count = Counter(p.split("-")[0] for p, _ in packs)
    print("=" * 72)
    print(f"FAMILY HISTOGRAM — {len(packs)} active packs across "
          f"{len(fam_count)} families")
    print("=" * 72)
    for fam, n in fam_count.most_common():
        mark = "" if fam in FAMILY_MAP else "   <- UNMAPPED"
        print(f"  {fam:<16} {n:>4d}  {mark}")

    # ── view 2: meta.capability cross-tab ───────────────────────────
    cap_count = Counter((getattr(d.meta, "capability", "?")) for _, d in packs)
    print()
    print("=" * 72)
    print("CAPABILITY TAG DISTRIBUTION")
    print("=" * 72)
    for cap, n in cap_count.most_common():
        print(f"  {cap:<16} {n:>4d}")
    if cap_count.get("adversarial", 0) < 5:
        print("  WARNING: adversarial = "
              f"{cap_count.get('adversarial', 0)} — full end-to-end "
              "macro lives in the 1v1 battleground, but the pack tag "
              "is severely under-represented.")

    # ── view 3: phase × decision heatmap (heuristic) ────────────────
    grid: dict[tuple[str, str], list[str]] = defaultdict(list)
    unmapped = []
    for stem, _ in packs:
        ph, dc = _classify(stem)
        if ph == "?":
            unmapped.append(stem)
        else:
            grid[(ph, dc)].append(stem)

    print()
    print("=" * 72)
    print("PHASE × DECISION HEATMAP (counts; '.' = empty cell)")
    print("=" * 72)
    print(f"  {'decision \\ phase':<32}  " + "  ".join(f"{p:>10s}" for p in PHASES))
    for dec in DECISIONS:
        row = []
        for ph in PHASES:
            n = len(grid.get((ph, dec), []))
            row.append(f"{n:>10d}" if n else f"{'.':>10}")
        print(f"  {dec:<32}  " + "  ".join(row))

    empty_cells = [
        (ph, dc) for ph in PHASES for dc in DECISIONS
        if (ph, dc) not in grid
    ]
    print(f"\n  empty cells: {len(empty_cells)} / {len(PHASES) * len(DECISIONS)}")
    if unmapped:
        print(f"  unmapped families ({len(unmapped)} packs): "
              f"{sorted({p.split('-')[0] for p in unmapped})}")

    if out_path:
        payload = {
            "total_packs": len(packs),
            "family_histogram": dict(fam_count.most_common()),
            "capability_distribution": dict(cap_count.most_common()),
            "phase_decision_grid": {
                f"{ph}|{dc}": grid.get((ph, dc), [])
                for ph in PHASES for dc in DECISIONS
            },
            "empty_cells": [f"{ph}|{dc}" for ph, dc in empty_cells],
            "unmapped": unmapped,
        }
        Path(out_path).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
