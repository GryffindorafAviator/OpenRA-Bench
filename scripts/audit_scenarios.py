"""Re-audit every active pack against the no-cheat bar.

The bench's central authoring rule is "every lazy / brute / stall
policy must LOSE on every level"; a pack where the `stall` (observe-
only) policy WINS at any level is defective ("laziest-play-wins").
This script re-runs the static bar across all 200 packs after the
fact — engine fixes since authoring can have drifted things, so the
audit catches benchmark rot.

Optionally adds an EMPIRICAL layer: from one or more
`run_eval --out` report JSONs, flags packs where every model wins
(too easy / a trivial idiom dominates) or every model loses
(unsolvable / a load-bearing predicate is mis-tuned).

Run from the repo root:
  python scripts/audit_scenarios.py [report.json ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Project-relative imports.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openra_bench.eval_core import run_level  # noqa: E402
from openra_bench.scenarios import load_pack  # noqa: E402
from openra_bench.scenarios.loader import PACKS_DIR, compile_level  # noqa: E402

LEVELS = ("easy", "medium", "hard")
SEEDS = (1,)

# Packs where "do nothing" is the INTENDED winning policy by design —
# the audit's stall-wins-is-a-defect rule doesn't apply. Each pack
# here is justified in its own test file's docstring.
STALL_WINS_BY_DESIGN: frozenset[str] = frozenset({
    # Positional-discipline pack: pre-positioned `stance:2` flanker
    # crossfire — the player's job is NOT to move the flankers, so
    # `observe` IS the intended policy. See tests/test_def_with_ambush.py.
    "def-with-ambush",
})


def stall(_render_state, Command):
    return [Command.observe()]


def _active_packs() -> list[Path]:
    out: list[Path] = []
    for p in sorted(PACKS_DIR.glob("*.yaml")):
        if p.name.startswith(("_", "TEMPLATE")):
            continue
        try:
            d = load_pack(p)
        except Exception:  # noqa: BLE001
            continue
        if getattr(d.meta, "status", "active") == "quarantine":
            continue
        out.append(p)
    return out


def _one(pack_path: Path, level: str, seed: int) -> dict:
    """Run stall on one cell; report whether it (defectively) won."""
    try:
        c = compile_level(load_pack(pack_path), level)
        if not c.map_supported:
            return {"pack": pack_path.stem, "level": level, "seed": seed,
                    "outcome": "skip", "reason": "map_not_loadable"}
        res = run_level(c, stall, seed=seed)
        return {"pack": pack_path.stem, "level": level, "seed": seed,
                "outcome": res.outcome, "turns": res.turns}
    except Exception as e:  # noqa: BLE001
        return {"pack": pack_path.stem, "level": level, "seed": seed,
                "outcome": "error", "reason": f"{type(e).__name__}: {e}"[:120]}


def static_audit(concurrency: int = 8) -> list[dict]:
    """Run stall on every active pack × level. Returns per-cell records."""
    packs = _active_packs()
    tasks = [(p, lv, sd) for p in packs for lv in LEVELS for sd in SEEDS]
    print(f"[static] {len(packs)} packs × {len(LEVELS)} levels × "
          f"{len(SEEDS)} seeds = {len(tasks)} episodes", file=sys.stderr)
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            out.append(rec)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} done", file=sys.stderr)
    return out


def report_static(records: list[dict]) -> None:
    """Print the static-audit summary + the defect list."""
    by_pack: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_pack[r["pack"]].append(r)

    stall_wins: list[tuple[str, str]] = []     # (pack, level)
    errors: list[tuple[str, str, str]] = []    # (pack, level, reason)
    skips: list[tuple[str, str, str]] = []
    clean_packs = 0
    exempt_noted: list[str] = []
    for pack, rs in by_pack.items():
        outs = {r["level"]: r for r in rs}
        if pack in STALL_WINS_BY_DESIGN:
            exempt_noted.append(pack)
            continue
        if all(r["outcome"] == "loss" for r in rs):
            clean_packs += 1
        for lv, r in outs.items():
            if r["outcome"] == "win":
                stall_wins.append((pack, lv))
            elif r["outcome"] == "draw":
                stall_wins.append((pack, f"{lv}/DRAW"))
            elif r["outcome"] == "error":
                errors.append((pack, lv, r.get("reason", "")))
            elif r["outcome"] == "skip":
                skips.append((pack, lv, r.get("reason", "")))

    print("=" * 72)
    print(f"STATIC AUDIT — stall policy on {len(by_pack)} packs × {len(LEVELS)} levels")
    print("=" * 72)
    print(f"  clean (all 3 levels LOSS — intended)   : {clean_packs}")
    print(f"  defects (stall wins or draws)          : {len(stall_wins)}")
    print(f"  errors                                  : {len(errors)}")
    print(f"  skipped (map not loadable)             : {len(skips)}")
    print(f"  exempt (stall-wins-by-design)          : {len(exempt_noted)}"
          f"  {exempt_noted if exempt_noted else ''}")
    if stall_wins:
        print("\n--- DEFECTS — laziest play wins (re-author or retune) ---")
        for pack, lv in sorted(stall_wins):
            print(f"  {pack}  [{lv}]")
    if errors:
        print("\n--- ERRORS ---")
        for pack, lv, why in sorted(errors)[:20]:
            print(f"  {pack}  [{lv}]  {why}")
        if len(errors) > 20:
            print(f"  ...{len(errors) - 20} more")


def empirical_audit(report_paths: list[str]) -> None:
    """From run_eval JSON reports, flag packs where every model wins
    (too easy) or every model loses (unsolvable). Needs ≥2 models'
    reports to mean anything."""
    print()
    print("=" * 72)
    print(f"EMPIRICAL AUDIT — {len(report_paths)} model report(s)")
    print("=" * 72)
    # cell → list of (model, outcome)
    per_cell: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in report_paths:
        try:
            rep = json.loads(Path(path).read_text())
        except Exception as e:  # noqa: BLE001
            print(f"  could not load {path}: {e}", file=sys.stderr)
            continue
        model = rep.get("model") or Path(path).stem
        for e in rep.get("episodes", []):
            if e.get("outcome") in {"win", "loss", "draw"}:
                per_cell[e["cell"]].append((model, e["outcome"]))

    n_models = len(report_paths)
    too_easy, unsolvable = [], []
    for cell, rs in per_cell.items():
        if len({m for m, _ in rs}) < n_models:
            continue  # not all models covered this cell
        outs = [o for _, o in rs]
        if all(o == "win" for o in outs):
            too_easy.append(cell)
        elif all(o == "loss" for o in outs):
            unsolvable.append(cell)

    print(f"  cells covered by ALL {n_models} models : {sum(1 for rs in per_cell.values() if len({m for m,_ in rs}) >= n_models)}")
    print(f"  too easy (every model wins)            : {len(too_easy)}")
    print(f"  unsolvable (every model loses)         : {len(unsolvable)}")
    if too_easy:
        print("\n--- TOO EASY (every model wins) ---")
        for c in sorted(too_easy):
            print(f"  {c}")
    if unsolvable:
        print("\n--- UNSOLVABLE (every model loses) ---")
        for c in sorted(unsolvable):
            print(f"  {c}")


def main(argv: list[str]) -> int:
    records = static_audit()
    report_static(records)
    if argv[1:]:
        empirical_audit(argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
