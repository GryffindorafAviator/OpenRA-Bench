#!/usr/bin/env python3
"""Phase 4 paper-collection driver: spawn one subprocess per
(model, pack, level, seed, fog_mode) cell so a full Together-roster
eval produces complete, untruncated, audit-ready JSONL + PNG data.

Each cell runs as an isolated `python -m openra_bench.run_eval`
invocation pinned to a SINGLE pack / level / seed / fog-mode, with the
audit recorder (`--full-playback <dir>`) writing one JSONL per cell:

    <output-dir>/<timestamp>__<model_safe>/<pack>__<level>__seed<N>__<fog>.jsonl
    <output-dir>/<timestamp>__<model_safe>/<pack>__<level>__seed<N>__<fog>/turn_<N>.png

Behaviours:

* `--resume`: skips a cell whose JSONL already has a `terminal:` line
  (`openra_bench.full_playback.is_complete_cell`). A partial / crashed
  cell is correctly retried.
* `--parallel-cells N`: up to N cell subprocesses run concurrently.
  One crashing cell does not abort the run.
* `--dry-run`: prints the cell list + estimated token / USD cost
  without spawning anything.
* `--cost-estimate`: same as dry-run for the cost lines only.
* Per-cell logs land in `<output-dir>/.logs/<cell-id>.log` so
  post-hoc you can diagnose any cell that exited non-zero.

Defaults are tuned for the Together AI roster the playback dirs use
(`Qwen/Qwen3.5-9B`, `Qwen/Qwen3.6-Plus`, `qwen/qwen3.6-flash`,
`google/gemma-4-31B-it`, `moonshotai/Kimi-K2.6`); override via flags.

NOT a runner of the Together API directly: this script orchestrates
the existing `openra_bench.run_eval` CLI, which already speaks the
Together-compatible (OpenAI Chat Completions) wire format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running from a checkout without `pip install -e .`.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from openra_bench.full_playback import cell_stem, is_complete_cell  # noqa: E402
from openra_bench.scenarios.loader import PACKS_DIR  # noqa: E402

# ── Together AI pricing (USD per 1M tokens) ───────────────────────────
# Snapshot as of Phase 4 prep (May 2026). Used by --cost-estimate. The
# values are conservative upper bounds — if Together's published price
# is lower today the estimate will overshoot, which is the right side
# to err on for budget approval. Update inline as prices change.
#
# Source: https://together.ai/pricing (look up each model)
_TOGETHER_PRICES: dict[str, tuple[float, float]] = {
    # model_id_lower -> (in_per_M, out_per_M)
    "qwen/qwen3.5-9b": (0.20, 0.20),
    "qwen/qwen3.6-plus": (0.50, 1.50),
    "qwen/qwen3.6-flash": (0.18, 0.18),
    "google/gemma-4-31b-it": (0.25, 0.25),
    "moonshotai/kimi-k2.6": (0.60, 2.50),
}
_DEFAULT_PRICE = (0.50, 1.50)  # safe fallback if a model isn't listed

# ── Token-per-turn priors (from playback/pilot_* historical runs) ─────
# Used by --cost-estimate. These are AVERAGES over the existing pilot
# runs (perception + handoff pilots, May 2026). Real numbers will vary
# per cell; the estimate is order-of-magnitude.
_AVG_TURNS_PER_CELL = 18  # mean across pilots; max_turns is 36 typically
_AVG_PROMPT_TOK_PER_TURN = 4500   # ~prompt+image+codex, sliding window
_AVG_COMPLETION_TOK_PER_TURN = 250  # tool call + brief reasoning

# Default Together roster (from the existing playback dirs).
_DEFAULT_MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-Plus",
    "qwen/qwen3.6-flash",
    "google/gemma-4-31B-it",
    "moonshotai/Kimi-K2.6",
]


def _safe_model(m: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", m)


def _list_all_packs() -> list[str]:
    """Every active (non-quarantine, non-template) pack id under
    `openra_bench/scenarios/packs/`. Sorted for stable ordering."""
    out: list[str] = []
    for p in sorted(PACKS_DIR.glob("*.yaml")):
        if p.name.startswith(("_", "TEMPLATE")):
            continue
        # Pack id == filename stem (matches load_pack convention).
        out.append(p.stem)
    return out


def _resolve_packs(spec: str | None) -> list[str]:
    """Resolve `--packs`:
    * `all`         → every active pack
    * `@FILE`       → one pack id per line
    * `a,b,c`       → comma-separated list
    * `path/to/dir` → every *.yaml in that dir (stem == id)
    * None/empty    → `all`
    """
    if not spec or spec.lower() == "all":
        return _list_all_packs()
    if spec.startswith("@"):
        ids = []
        for line in Path(spec[1:]).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.append(line)
        return ids
    # Comma list short-circuits the dir check (avoids `File name too long`
    # when a long comma list is passed).
    if "," in spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    # Possibly a single pack id OR a directory path. Heuristic: if it
    # contains a `/` OR exists on disk, treat as path.
    if "/" in spec or len(spec) > 200 or Path(spec).exists():
        p = Path(spec)
        if p.is_dir():
            return sorted(x.stem for x in p.glob("*.yaml"))
    return [spec.strip()]


def _price_for(model: str) -> tuple[float, float]:
    return _TOGETHER_PRICES.get(model.lower().strip(), _DEFAULT_PRICE)


def _estimate_cost(cells_per_model: dict[str, int]) -> dict:
    """Per-model token + USD estimate from `cells_per_model`. Uses the
    pilot-historical priors at module top; conservative on the high
    side. Returns a printable structured dict."""
    rows = []
    total_usd = 0.0
    total_tok_in = 0
    total_tok_out = 0
    for model, n_cells in sorted(cells_per_model.items()):
        in_per_m, out_per_m = _price_for(model)
        tok_in = n_cells * _AVG_TURNS_PER_CELL * _AVG_PROMPT_TOK_PER_TURN
        tok_out = n_cells * _AVG_TURNS_PER_CELL * _AVG_COMPLETION_TOK_PER_TURN
        usd = (tok_in / 1_000_000) * in_per_m + (tok_out / 1_000_000) * out_per_m
        rows.append(
            {
                "model": model,
                "cells": n_cells,
                "est_tokens_in": tok_in,
                "est_tokens_out": tok_out,
                "in_per_M": in_per_m,
                "out_per_M": out_per_m,
                "est_usd": round(usd, 2),
            }
        )
        total_usd += usd
        total_tok_in += tok_in
        total_tok_out += tok_out
    return {
        "per_model": rows,
        "total_cells": sum(cells_per_model.values()),
        "total_tokens_in": total_tok_in,
        "total_tokens_out": total_tok_out,
        "total_usd": round(total_usd, 2),
        "assumptions": {
            "avg_turns_per_cell": _AVG_TURNS_PER_CELL,
            "avg_prompt_tokens_per_turn": _AVG_PROMPT_TOK_PER_TURN,
            "avg_completion_tokens_per_turn": _AVG_COMPLETION_TOK_PER_TURN,
            "source": "pilot pilot_perception + pilot_handoff means, May 2026",
        },
    }


# ── Plan + run ────────────────────────────────────────────────────────


def build_plan(args) -> list[dict]:
    """Expand the args into the flat list of (model, pack, level, seed,
    fog_mode, repeat) cells. Each item carries the JSONL path it will
    write to so `--resume` can scan and skip."""
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    packs = _resolve_packs(args.packs)
    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    fogs = [s.strip() for s in args.fog_modes.split(",") if s.strip()]
    rng_reps = range(max(1, int(args.repeats)))

    out: list[dict] = []
    out_root = Path(args.output_dir)
    for model in models:
        model_dir = out_root / f"{args.timestamp}__{_safe_model(model)}"
        for pack in packs:
            for level in levels:
                for seed in seeds:
                    for fog in fogs:
                        for rep in rng_reps:
                            stem = cell_stem(pack, level, seed, fog)
                            if rep > 0:
                                stem = f"{stem}__rep{rep}"
                            jsonl = model_dir / f"{stem}.jsonl"
                            out.append(
                                {
                                    "model": model,
                                    "pack": pack,
                                    "level": level,
                                    "seed": seed,
                                    "fog_mode": fog,
                                    "repeat": rep,
                                    "model_dir": str(model_dir),
                                    "jsonl_path": str(jsonl),
                                    "cell_id": (
                                        f"{_safe_model(model)}__{stem}"
                                    ),
                                }
                            )
    return out


def filter_resume(plan: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split `plan` into (todo, done) on resume — `done` is anything
    whose JSONL exists AND ends with a `terminal:` line."""
    todo: list[dict] = []
    done: list[dict] = []
    for c in plan:
        if is_complete_cell(c["jsonl_path"]):
            done.append(c)
        else:
            todo.append(c)
    return todo, done


def _pack_path_for(pack_id: str) -> Path:
    return PACKS_DIR / f"{pack_id}.yaml"


def _run_cell(cell: dict, args, python_bin: str) -> dict:
    """Spawn one `python -m openra_bench.run_eval` for a single cell.
    Returns a result dict with rc / log_path / jsonl_path."""
    model_dir = Path(cell["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.output_dir) / ".logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{cell['cell_id']}.log"

    # The audit dir is the per-model timestamp dir; the inner JSONL stem
    # is derived deterministically by FullPlayback from pack/level/seed/
    # fog. We pre-create the parent so concurrent cells don't race on it.
    cmd = [
        python_bin,
        "-m",
        "openra_bench.run_eval",
        "--provider",
        args.provider,
        "--model",
        cell["model"],
        "--packs",
        str(_pack_path_for(cell["pack"])),
        "--levels",
        cell["level"],
        "--seeds",
        str(cell["seed"]),
        "--fog-mode",
        cell["fog_mode"],
        "--full-playback",
        str(model_dir),
        # Always emit a per-cell stats file under .logs so the report has
        # something even when --no-stats is fine to ignore.
        "--out",
        str(log_dir / f"{cell['cell_id']}.stats.json"),
        # Empty seeds list arg quirk; we pin exactly one seed.
    ]
    # Resume / repeat: a per-cell `--repeats N` would multiply within
    # the subprocess; we keep that to 1 (default) since the OUTER loop
    # owns the repeat index — each repeat is a distinct subprocess so
    # the JSONL stems differ (rep0 / repN). Future: thread the repeat
    # index into the FullPlayback stem (the planner already encodes it
    # in `cell_id`; the inner JSONL would still be `…__seedN__fog.jsonl`
    # which a rep>0 cell would collide on — so we run rep0 only via
    # this script for now and document repeats as a separate sweep).

    env = dict(os.environ)
    # The CostMeter inside run_eval uses ProviderConfig.price_in/out_per_m
    # which we don't pass here; cost capture is best-effort and the
    # FullPlayback `terminal.total_tokens_*` is the authoritative token
    # count for downstream costing.

    started = time.time()
    with open(log_path, "w") as fh:
        fh.write(f"# cmd: {shlex.join(cmd)}\n")
        fh.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fh.flush()
        try:
            rc = subprocess.call(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env
            )
        except Exception as e:  # noqa: BLE001
            fh.write(f"\n# spawn failed: {type(e).__name__}: {e}\n")
            rc = -1
    return {
        "cell_id": cell["cell_id"],
        "rc": rc,
        "elapsed_s": round(time.time() - started, 2),
        "log_path": str(log_path),
        "jsonl_path": cell["jsonl_path"],
        "complete": is_complete_cell(cell["jsonl_path"]),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="collect_eval_data.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--models",
        default=",".join(_DEFAULT_MODELS),
        help="comma-separated Together model ids "
        f"(default: {','.join(_DEFAULT_MODELS)})",
    )
    ap.add_argument(
        "--packs",
        default="all",
        help="`all` | comma-separated pack ids | @file_with_one_per_line",
    )
    ap.add_argument("--levels", default="easy,medium,hard")
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument(
        "--fog-modes",
        default="vision",
        help="comma-separated subset of "
        "structured,structured-clear,vision,vision-clear,image,image-clear",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run each cell N times (rep0 + repN suffix). >1 currently "
        "shares the JSONL stem — keep at 1 for paper-grade collection.",
    )
    ap.add_argument(
        "--run-label",
        default="paper-collection",
        help="logical name for this run (becomes a path segment under "
        "--output-dir's parent if you want; informational here)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="root for the audit dirs "
        "(default: data/runs/<run-label>)",
    )
    ap.add_argument(
        "--parallel-cells",
        type=int,
        default=1,
        help="how many cell subprocesses to run at once",
    )
    ap.add_argument(
        "--provider",
        default="together",
        help="provider name forwarded to run_eval (default: together)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip cells whose JSONL already ends with a terminal line",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned cell list + cost estimate; spawn nothing",
    )
    ap.add_argument(
        "--cost-estimate",
        action="store_true",
        help="print the cost estimate and exit (no plan listing)",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help=f"python binary used to launch run_eval (default: {sys.executable})",
    )
    ap.add_argument(
        "--timestamp",
        default=None,
        help="override the timestamp segment of the per-model dir "
        "(default: now UTC, format YYYYMMDD-HHMMSS)",
    )
    a = ap.parse_args(argv[1:])

    a.timestamp = a.timestamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if not a.output_dir:
        a.output_dir = str(REPO / "data" / "runs" / a.run_label)

    plan = build_plan(a)
    if not plan:
        print("collect_eval_data: empty plan — nothing to do", file=sys.stderr)
        return 2

    # Cell counts per model for the cost estimator.
    cells_per_model: dict[str, int] = {}
    for c in plan:
        cells_per_model[c["model"]] = cells_per_model.get(c["model"], 0) + 1
    cost = _estimate_cost(cells_per_model)

    if a.cost_estimate or a.dry_run:
        print("== cost estimate ==")
        for row in cost["per_model"]:
            print(
                f"  {row['model']:<32} cells={row['cells']:>5}  "
                f"tok_in={row['est_tokens_in']:>10,}  "
                f"tok_out={row['est_tokens_out']:>9,}  "
                f"@ ${row['in_per_M']}/${row['out_per_M']} per M  "
                f"≈ ${row['est_usd']:>8.2f}"
            )
        print(
            f"  TOTAL: cells={cost['total_cells']}  "
            f"tok_in={cost['total_tokens_in']:,}  "
            f"tok_out={cost['total_tokens_out']:,}  "
            f"≈ ${cost['total_usd']:.2f}"
        )
        print(f"  assumptions: {cost['assumptions']}")
        if a.cost_estimate and not a.dry_run:
            return 0

    todo = plan
    done: list[dict] = []
    if a.resume:
        todo, done = filter_resume(plan)
        print(
            f"resume: {len(done)}/{len(plan)} cells already complete; "
            f"{len(todo)} to run",
            file=sys.stderr,
        )

    if a.dry_run:
        print(f"\n== plan ({len(todo)} cells, {a.parallel_cells} parallel) ==")
        # Truncate listing to a sane head; full plan goes to a sidecar.
        for c in todo[:50]:
            print(
                f"  {c['model']:<30} {c['pack']:<40} {c['level']:<6} "
                f"seed={c['seed']} fog={c['fog_mode']:<18} -> "
                f"{c['jsonl_path']}"
            )
        if len(todo) > 50:
            print(f"  ... ({len(todo) - 50} more)")
        # Sidecar plan dump for the reviewer.
        sidecar = Path(a.output_dir) / "_dry_run_plan.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps(
                {"plan": todo, "cost": cost, "args": vars(a)},
                indent=2,
                default=str,
            )
        )
        print(f"\nwrote full plan: {sidecar}")
        return 0

    # ── Live run ──────────────────────────────────────────────────────
    out_root = Path(a.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".logs").mkdir(parents=True, exist_ok=True)
    # Manifest the invocation so post-hoc you can reproduce.
    (out_root / "_invocation.json").write_text(
        json.dumps(
            {
                "args": vars(a),
                "cost_estimate": cost,
                "planned": len(plan),
                "todo": len(todo),
                "resumed": len(done),
                "started": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            },
            indent=2,
            default=str,
        )
    )

    results: list[dict] = []
    fail = 0
    completed = 0
    started = time.time()
    if a.parallel_cells <= 1:
        for c in todo:
            r = _run_cell(c, a, a.python)
            results.append(r)
            completed += 1 if r["complete"] else 0
            fail += 0 if r["rc"] == 0 else 1
            print(
                f"[{len(results)}/{len(todo)}] {c['cell_id']} "
                f"rc={r['rc']} t={r['elapsed_s']}s "
                f"complete={r['complete']}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=a.parallel_cells) as ex:
            futs = {ex.submit(_run_cell, c, a, a.python): c for c in todo}
            for fu in as_completed(futs):
                r = fu.result()
                results.append(r)
                completed += 1 if r["complete"] else 0
                fail += 0 if r["rc"] == 0 else 1
                print(
                    f"[{len(results)}/{len(todo)}] {r['cell_id']} "
                    f"rc={r['rc']} t={r['elapsed_s']}s "
                    f"complete={r['complete']}",
                    flush=True,
                )

    elapsed = round(time.time() - started, 1)
    summary = {
        "ran": len(results),
        "complete": completed,
        "failed": fail,
        "resumed": len(done),
        "total_planned": len(plan),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"\ndone in {elapsed}s: {completed}/{len(todo)} complete, "
        f"{fail} failed; see {out_root}/_summary.json"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
